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

_possible_datasets = ["ml2hp", "mmhgdhgr", "tiny_hgr"]

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
    
    # defines the device to use
    device = get_device_from_string(cfg_dict["device"])  # "cuda" or "cpu"

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

        # instantiates the model. it is normal to have errors since some keys are meant not to be stored in the checkpoint
        model = HandGestureRecognitionModel.load_from_checkpoint(checkpoint_path, map_location=device, strict=False)
        trainer = Trainer(
            accelerator=device,
            log_every_n_steps=10,
            precision="16-mixed",
            gradient_clip_val=1.0,
            max_epochs=cfg_dict["max_epochs"],
            accumulate_grad_batches=cfg_dict["accumulate_grad_batches"],
            enable_model_summary=True,
            enable_checkpointing=True,
        )
        

        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]),  # type: ignore
            batch_size=cfg_dict["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=True,
        )
        print(f"testing on the VALIDATION dataloader")
        trainer.test(model=model, dataloaders=dataloader_val)

        if "test_idx" in run.keys():
            dataloader_test = DataLoader(
                dataset=Subset(dataset, indices=run["test_idx"]),  # type: ignore
                batch_size=cfg_dict["batch_size"],
                shuffle=False,
                pin_memory=False,
                num_workers=num_workers,
                persistent_workers=True,
            )
            print(f"testing on the TEST dataloader")
        trainer.test(model=model, dataloaders=dataloader_test)

if __name__ == "__main__":
    main()
