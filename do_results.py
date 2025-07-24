import argparse
import gc
import itertools
import subprocess
import glob
import os
import shutil
from os.path import join, isdir
from os import listdir, makedirs

import torch
import yaml
import generate_configs
import train
from model import HandGestureRecognitionModel

_possible_datasets = ["ml2hp", "mmhgdhgr", "tiny_hgr"]
_possible_validations = ["loso", "simple"]
def main():
    
    parser = argparse.ArgumentParser(description="Run ablation training pipeline.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=_possible_datasets,
        help=f"The dataset to use. Must be one of {_possible_datasets}.",
    )
    parser.add_argument(
        "--validation",
        required=False,
        default="loso",
        choices=_possible_validations,
        help=f"The validation scheme to use. Must be one of {_possible_validations}.",
    )
    parser.add_argument(
        "--image_backbone",
        required=False,
        default="convnextv2-t",
        help="Image backbone to use.",
    )
    parser.add_argument(
        "--train_image_backbone",
        action="store_true",
        help="Whether to finetune the image backbone.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--landmarks_backbone",
        required=False,
        default="mlp",
        help="Landmarks backbone to use.",
    )
    parser.add_argument(
        "--use_horizontal_image",
        action="store_true",
        help="Whether to use horizontal images.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--use_vertical_image",
        action="store_true",
        help="Whether to use vertical images.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--use_horizontal_landmarks",
        action="store_true",
        help="Whether to use horizontal landmarks.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--use_vertical_landmarks",
        action="store_true",
        help="Whether to use vertical landmarks.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--normalize_landmarks",
        action="store_true",
        help="Whether to use normalize landmarks.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--disable_checkpointing",
        action="store_true",
        help="Whether not to save model checkpoints.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="The maximum amount of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="The learning rate of AdamW.",
    )
    line_args = parser.parse_args()

    if line_args.dataset not in {"mmhgdhgr", "tiny_hgr"} and line_args.validation == "simple":
        raise Exception(
            f"Validation scheme 'simple' is not supported for dataset {line_args.dataset}. "
            "Please use 'loso' instead."
        )
    if line_args.image_backbone.lower() == "none":
        line_args.image_backbone = None
    if line_args.landmarks_backbone.lower() == "none":
        line_args.landmarks_backbone = None
    assert (
        line_args.image_backbone is not None or line_args.landmarks_backbone is not None
    ), "You must provide at least one of --image_backbone or --landmarks_backbone."
    _possible_image_backbones = (
        HandGestureRecognitionModel._possible_image_backbones | {"all"}
    )
    _possible_landmarks_backbones = (
        HandGestureRecognitionModel._possible_landmarks_backbones | {"all"}
    )
    assert (
        line_args.image_backbone in _possible_image_backbones
    ), f"got {line_args.image_backbone}, but expected one of {_possible_image_backbones}"
    assert (
        line_args.landmarks_backbone in _possible_landmarks_backbones
    ), f"got {line_args.landmarks_backbone}, but expected one of {_possible_landmarks_backbones}"

    # generate configs with provided arguments
    cfgs_path = join("cfgs", "results")
    if isdir(cfgs_path):
        shutil.rmtree(cfgs_path)
    makedirs(cfgs_path, exist_ok=True)
    batch_size, accumulate_grad_batches = 128, 1
    if line_args.image_backbone is not None and "dinov2" in line_args.image_backbone:
        batch_size, accumulate_grad_batches = 64, 1
    elif not line_args.use_horizontal_image and not line_args.use_vertical_image:
        batch_size = 512

    # dataset-specific parameters
    cfg = generate_configs.create_dict(
        dataset=line_args.dataset,
        datasets_path="../../datasets",
        checkpoints_path=f"./checkpoints/{line_args.dataset}_results",
        validation=line_args.validation,
        image_backbone_name=(
            line_args.image_backbone
            if any([line_args.use_horizontal_image, line_args.use_vertical_image])
            else None
        ),
        landmarks_backbone_name=(
            line_args.landmarks_backbone
            if any([line_args.use_horizontal_landmarks, line_args.use_vertical_landmarks])
            else None
        ),
        use_horizontal_image=line_args.use_horizontal_image,
        use_vertical_image=line_args.use_vertical_image,
        use_horizontal_landmarks=line_args.use_horizontal_landmarks,
        use_vertical_landmarks=line_args.use_vertical_landmarks,
        normalize_landmarks=line_args.normalize_landmarks,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=line_args.max_epochs,
        lr=line_args.lr,
        train_image_backbone=line_args.train_image_backbone,
    )
    filename = f"{cfg['name']}.yaml"
    with open(join(cfgs_path, filename), "w") as file:
        yaml.dump(cfg, file)

    for cfg_name in sorted(listdir(cfgs_path)):
        cfg_path = join(cfgs_path, cfg_name)
        # opens the .yaml file
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
        # starts the training
        try:
            train.main(
                cfg=cfg_path,
                disable_checkpointing=line_args.disable_checkpointing,
                run_name=f"results_{cfg['name']}",
            )
        except Exception as e:
            print(e)
        torch.cuda.empty_cache()
        gc.collect()
        
    # config_files = sorted(glob.glob(join(cfgs_path, "*.yaml")))
    # for config_path in config_files:
    #     print(f"Running training with config: {config_path}")
    #     subprocess.run([
    #         "python", "train.py",
    #         "--cfg", config_path,
    #         "--disable_checkpointing"
    #     ], check=True)

    # print("All training jobs have been done. You can find them in ./checkpoints/ablation/")


if __name__ == "__main__":
    main()
