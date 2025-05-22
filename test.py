import os
from os import listdir
from os.path import join
import gc
from pprint import pprint
import argparse
from datetime import datetime
import torch
from torch.utils.data import Subset, DataLoader
from lightning.pytorch import Trainer
from tqdm import tqdm
import yaml

from datasets.hand_pose_dataset import HandPoseDataset

from model import Model
from utils import (
    get_device_from_string,
    get_loso_runs,
    set_global_seed,
)

if __name__ == "__main__":
    # setup
    torch.set_float32_matmul_precision("medium")

    # arguments parsing
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--cfg", type=str, help="Path to the configuration", required=True
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        help="Path to where the checkpoints are located",
        required=True,
    )
    line_args = vars(parser.parse_args())

    # loads the configuration file
    num_workers = os.cpu_count() // 2 # type: ignore
    with open(line_args["cfg"], "r") as fp:
        cfg = yaml.safe_load(fp)
    pprint(cfg)

    # sets the random seed
    set_global_seed(seed=cfg["seed"])

    # sets up the dataset(s)
    dataset = HandPoseDataset(
        dataset_path=cfg["dataset_path"],
    )

    # sets up the validation scheme
    if cfg["validation"] not in ["loso"]:
        raise NotImplementedError(
            f"only loso scheme have been implemented, got {cfg['validation']}"
        )
    runs = get_loso_runs(dataset=dataset)

    # setup the model
    device = get_device_from_string(cfg["device"])  # "cuda" or "cpu"

    # loops over runs
    for i_run, run in enumerate(runs):
        if not run["subject_id"] in listdir(line_args["checkpoints_path"]): # type: ignore
            print(
                f"skipping subject {run['subject_id']} as there's no associated checkpoints" # type: ignore
            )
            continue

        # splits the dataset
        dataloader = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]), # type: ignore
            batch_size=cfg["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
        )

        # loads the model for the subject
        checkpoint_path = join(
            line_args["checkpoints_path"],
            run["subject_id"], # type: ignore
            [
                filename
                for filename in listdir(
                    join(line_args["checkpoints_path"], run["subject_id"]) # type: ignore
                )
                if filename.endswith(".ckpt")
            ][0],
        )

        model = Model.load_from_checkpoint(checkpoint_path, map_location=device)
        for param in model.parameters():
            param.requires_grad = False
        # model.to(device)
        model.eval()

        # do the training
        trainer = Trainer(
            logger=None,
            accelerator=device,
            precision="16-mixed",
            max_epochs=1,
            enable_model_summary=True,
            enable_checkpointing=False,
        )
        print(f"testing on subject {run['subject_id']} using model {checkpoint_path}") # type: ignore
        trainer.test(model, dataloader)

    # frees some memory
    del dataset
    gc.collect()
