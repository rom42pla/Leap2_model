import argparse
import itertools
import subprocess
import glob
import os
import shutil
from os.path import join, isdir
from os import listdir, makedirs

import yaml
import generate_configs
import train
from model import BWHandGestureRecognitionModel


def main():
    parser = argparse.ArgumentParser(description="Run ablation training pipeline.")
    parser.add_argument(
        "--image_backbone",
        required=False,
        default="convnextv2-b",
        help="Image backbone to use.",
    )
    parser.add_argument(
        "--landmarks_backbone",
        required=False,
        default="mlp",
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
    cfgs_path = join("cfgs", "results")
    if isdir(cfgs_path):
        shutil.rmtree(cfgs_path)
    makedirs(cfgs_path, exist_ok=True)
    batch_size, accumulate_grad_batches = 128, 1
    if line_args.image_backbone is not None and "dinov2" in line_args.image_backbone:
        batch_size, accumulate_grad_batches = 64, 1
    for (
        use_horizontal_image,
        use_vertical_image,
        use_horizontal_landmarks,
        use_vertical_landmarks,
    ) in itertools.product([True, False], [True, False], [True, False], [True, False]):
        num_images = sum(
            [1 if use_horizontal_image else 0, 1 if use_vertical_image else 0]
        )
        num_landmarks = sum(
            [1 if use_horizontal_landmarks else 0, 1 if use_vertical_landmarks else 0]
        )
        if (
            (num_images == 0 and num_landmarks == 0)
            or (num_images == 2 and num_landmarks == 1)
            or (num_images == 1 and num_landmarks == 2)
        ):
            # skip the case where no
            continue
        if (
            use_horizontal_image
            and not use_vertical_image
            and not use_horizontal_landmarks
            and use_vertical_landmarks
        ) or (
            not use_horizontal_image
            and use_vertical_image
            and use_horizontal_landmarks
            and not use_vertical_landmarks
        ):
            continue
        cfg = generate_configs.create_dict(
            dataset="ml2hp",
            datasets_path="../../datasets",
            checkpoints_path="./checkpoints/results",
            image_backbone_name=(
                line_args.image_backbone
                if any([use_horizontal_image, use_vertical_image])
                else None
            ),
            landmarks_backbone_name=(
                line_args.landmarks_backbone
                if any([use_horizontal_landmarks, use_vertical_landmarks])
                else None
            ),
            use_horizontal_image=use_horizontal_image,
            use_vertical_image=use_vertical_image,
            use_horizontal_landmarks=use_horizontal_landmarks,
            use_vertical_landmarks=use_vertical_landmarks,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=10,
        )
        filename = f"{cfg['name']}.yaml"
        with open(join(cfgs_path, filename), "w") as file:
            yaml.dump(cfg, file)

    for cfg_name in listdir(cfgs_path):
        cfg_path = join(cfgs_path, cfg_name)
        # opens the .yaml file
        with open(cfg_path, "r") as file:
            cfg = yaml.safe_load(file)
        # starts the training
        try:
            train.main(
                cfg=cfg_path,
                disable_checkpointing=False,
                run_name=f"ablation_{cfg['name']}_{line_args.landmarks_backbone}",
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
