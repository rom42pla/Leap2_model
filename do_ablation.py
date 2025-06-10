import argparse
import subprocess
import glob
import os
from os.path import join
from os import listdir

import yaml
import generate_configs
import train
from model import BWHandGestureRecognitionModel


def main():
    parser = argparse.ArgumentParser(description="Run ablation training pipeline.")
    parser.add_argument(
        "--image_backbone", required=False, default="all", help="Image backbone to use."
    )
    parser.add_argument(
        "--landmarks_backbone",
        required=False,
        default="all",
        help="Landmarks backbone to use.",
    )
    line_args = parser.parse_args()

    if line_args.image_backbone.lower() == "none":
        line_args.image_backbone = None
    if line_args.landmarks_backbone.lower() == "none":
        line_args.landmarks_backbone = None
    assert (
        line_args.image_backbone is not None or line_args.landmarks_backbone is not None
    ), "You must provide at least one of --image_backbone or --landmarks_backbone."
    _possible_image_backbones = (
        BWHandGestureRecognitionModel._possible_image_backbones | {"all"}
    )
    _possible_landmarks_backbones = (
        BWHandGestureRecognitionModel._possible_landmarks_backbones | {"all"}
    )
    assert (
        line_args.image_backbone in _possible_image_backbones
    ), f"got {line_args.image_backbone}, but expected one of {_possible_image_backbones}"
    assert (
        line_args.landmarks_backbone in _possible_landmarks_backbones
    ), f"got {line_args.landmarks_backbone}, but expected one of {_possible_landmarks_backbones}"

    # generate configs with provided arguments
    batch_size = 128
    if line_args.image_backbone is not None and "dinov2" in line_args.image_backbone:
        batch_size = 64
    generate_configs.main(dataset="ml2hp", datasets_path="../../datasets", batch_size=batch_size)  

    # match the possible config files
    matched_configs = []
    cfgs_path = join("cfgs", "ablation")
    for cfg_name in listdir(cfgs_path):
        # opens the .yaml file
        with open(join(cfgs_path, cfg_name), "r") as file:
            cfg = yaml.safe_load(file)
        matched = True
        if (
            line_args.image_backbone != "all"
            and cfg["image_backbone_name"] != line_args.image_backbone
        ) or (
            line_args.landmarks_backbone != "all"
            and cfg["landmarks_backbone_name"] != line_args.landmarks_backbone
        ):
            matched = False
        if matched:
            matched_configs.append(cfg_name)

    for cfg_name in matched_configs:
        cfg_path = join(cfgs_path, cfg_name)
        # opens the .yaml file
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
        # starts the training
        try:
            train.main(
                cfg=cfg_path,
                disable_checkpointing=True,
                run_name=f"ablation_{cfg['name']}_{line_args.landmarks_backbone}",
                limit_subjects=5,
            )
        except Exception as e:
            print(e)
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
