import argparse
import gc
import itertools
from pprint import pprint
import subprocess
import glob
import os
import shutil
from os.path import join, isdir
from os import listdir, makedirs

from lightning import Trainer
import torch
import yaml
from datasets.ml2hp import MotionLeap2Dataset
from datasets.mmhgdhgr import MultiModalHandGestureDatasetForHandGestureRecognition
from datasets.tiny_hgr import TinyHandGestureRecognitionDataset
import generate_configs
import train
from model import HandGestureRecognitionModel
from torch.utils.data import DataLoader, Subset
from utils import get_device_from_string, get_loso_runs, get_optimistic_splits, get_train_test_splits
import time
import warnings

# For CPU quantization
import torch.quantization

_possible_datasets = ["ml2hp", "mmhgdhgr", "tiny_hgr"]

def run_inference(model, dataloader, device, precision, n_samples):
    """
    Runs a manual inference loop to measure performance.
    """
    model.to(device)
    model.eval()

    total_inference_time = 0
    autocast_dtype = torch.float32
    if precision == 'bf16':
        autocast_dtype = torch.bfloat16
    
    # Warm-up run (especially important for GPUs)
    with torch.no_grad():
        for batch in dataloader:
            # Move data to the target device
            if isinstance(batch, list):
                batch = [b for b in batch]
            else:
                batch = batch
            
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                 model.step(batch, "eval")
            break

    # Measurement run
    with torch.no_grad():
        for batch in dataloader:
            # Move data to the target device
            if isinstance(batch, list):
                batch = [b for b in batch]
            else:
                batch = batch
            
            start_time = time.time()
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                model.step(batch, "eval")
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            
    avg_inference_time_ms = (total_inference_time / n_samples) * 1000
    return avg_inference_time_ms


def main():

    # sets some parameters
    torch.set_float32_matmul_precision("medium")
    num_workers = os.cpu_count() - 2

    # parses line arguments
    parser = argparse.ArgumentParser(description="Run ablation training pipeline.")
    parser.add_argument(
        "path",
        help=f"The path to the checkpoints to validate.",
    )
    line_args = parser.parse_args()

    # opens the configuration file for the experiment
    cfg_path = join(line_args.path, "cfg.yaml")
    with open(cfg_path, "r") as file:
        cfg_dict = yaml.safe_load(file)
    pprint(cfg_dict)

    # sets up the dataset(s)
    if cfg_dict["dataset"] == "ml2hp":
        dataset = MotionLeap2Dataset(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
        )
        dataset.set_mode(
            return_horizontal_images=cfg_dict["use_horizontal_image"],
            return_vertical_images=cfg_dict["use_vertical_image"],
            return_horizontal_landmarks=cfg_dict["use_horizontal_landmarks"],
            return_vertical_landmarks=cfg_dict["use_vertical_landmarks"],
        )
    elif cfg_dict["dataset"] == "mmhgdhgr":
        dataset = MultiModalHandGestureDatasetForHandGestureRecognition(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
            img_size=224,
        )
        dataset.set_mode(
            return_images=any(
                [cfg_dict["use_horizontal_image"], cfg_dict["use_vertical_image"]]
            ),
            return_landmarks=any(
                [
                    cfg_dict["use_horizontal_landmarks"],
                    cfg_dict["use_vertical_landmarks"],
                ]
            ),
        )
    elif cfg_dict["dataset"] == "tiny_hgr":
        dataset = TinyHandGestureRecognitionDataset(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
            img_size=224,
        )
        dataset.set_mode(
            return_images=any(
                [cfg_dict["use_horizontal_image"], cfg_dict["use_vertical_image"]]
            ),
            return_landmarks=any(
                [
                    cfg_dict["use_horizontal_landmarks"],
                    cfg_dict["use_vertical_landmarks"],
                ]
            ),
        )

    # sets up the validation scheme
    if cfg_dict["dataset"] == "ml2hp" or cfg_dict["validation"] == "loso":
        runs = get_loso_runs(dataset=dataset)
    elif cfg_dict["dataset"] in {"mmhgdhgr"}:
        runs = get_train_test_splits(dataset=dataset)
    elif cfg_dict["dataset"] in {"tiny_hgr"}:
        runs = get_optimistic_splits(dataset=dataset)
    else:
        raise NotImplementedError()

    # Define test configurations
    test_configs = [
        {'device': 'cuda', 'precision': '32'},
        {'device': 'cuda', 'precision': 'bf16'},
        {'device': 'cpu', 'precision': '32'},
        {'device': 'cpu', 'precision': 'int8'},
    ]

    # loops over runs
    for run_name in listdir(line_args.path):
        if not isdir(join(line_args.path, run_name)):
            continue
        run_path = join(line_args.path, run_name)
        # loads the checkpoint from the run
        try:
            checkpoint_path = [
                join(run_path, f) for f in listdir(run_path) if f.endswith(".ckpt")
            ][0]
        except:
            print(f"no .ckpt files found in {run_path}. skipping")
            continue

        if len(runs) > 1:
            # matches with the correct run object
            for run in runs:
                if run["subject_id"] == run_name:
                    break
            assert run["subject_id"] == run_name
        else:
            run = runs[0]

        N_SAMPLES = 100
        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"][:N_SAMPLES]),  # type: ignore
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )
        
        print(f"\n===== Starting Tests for Run: {run_name} =====")

        for config in test_configs:
            device = config['device']
            precision = config['precision']

            # --- Skip unsupported configurations ---
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"--- Skipping test: CUDA not available ---")
                continue
            if precision == 'bf16' and device == 'cpu':
                # BF16 is primarily for accelerators
                continue
            if precision == 'bf16' and device == 'cuda' and not torch.cuda.is_bf16_supported():
                print(f"--- Skipping test: This GPU does not support BF16 ---")
                continue


            print(f"\n--- Testing | Device: {device.upper()} | Precision: {precision.upper()} ---")

            # --- Reload a fresh model for each test to ensure no side-effects ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = HandGestureRecognitionModel.load_from_checkpoint(checkpoint_path, map_location='cpu', strict=False)
                model.eval()

            # --- Prepare model and run inference ---
            inference_time_ms = 0
            memory_used_mb = 0

            if precision == 'int8':
                if device == 'cpu':
                    print("Preparing INT8 quantized model for CPU...")
                    # Apply dynamic quantization
                    model_quantized = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                    )
                    inference_time_ms = run_inference(model_quantized, dataloader_val, device, precision, N_SAMPLES)
                    
                    # Report model size
                    fp32_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    quantized_path = "quantized_model_temp.pth"
                    torch.save(model_quantized.state_dict(), quantized_path)
                    int8_size = os.path.getsize(quantized_path) / (1024 * 1024)
                    os.remove(quantized_path)
                    print(f"Model Size (FP32): {fp32_size:.2f} MB")
                    print(f"Model Size (INT8): {int8_size:.2f} MB")
                else:
                    # INT8 on CUDA is more complex (e.g., via TensorRT) and not implemented here
                    print("Skipping INT8 on CUDA (not implemented in this script).")
                    continue
            else:
                # This handles FP32 and BF16
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                
                inference_time_ms = run_inference(model, dataloader_val, device, precision, N_SAMPLES)
                
                if device == 'cuda':
                    memory_used_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)


            # --- Print results for the current configuration ---
            print(f"Single Inference Speed: {inference_time_ms:.3f} ms")
            if memory_used_mb > 0:
                print(f"Peak Memory Used: {memory_used_mb:.3f} MB")
            
            # Clean up memory
            del model
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()