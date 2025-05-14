import os
from os import listdir, makedirs
from os.path import join, splitext, basename
import gc
from pprint import pprint
import argparse
from datetime import datetime
import torch
from torch.utils.data import Subset, DataLoader
from lightning.pytorch import Trainer

# import wandb
import torchmetrics
from tqdm import tqdm
import yaml

from datasets.hand_pose_dataset import HandPoseDataset

from model import Model
from utils import (
    find_samples_of_subject,
    get_device_from_string,
    get_k_fold_runs,
    get_loso_runs,
    get_simple_runs,
    set_global_seed,
)

if __name__ == "__main__":
    # setup
    torch.set_float32_matmul_precision("medium")

    # arguments parsing
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("cfg", type=str, help="Path to the configuration")
    parser.add_argument(
        "checkpoints_path", type=str, help="Path to where the checkpoints are located"
    )
    line_args = vars(parser.parse_args())

    # loads the configuration file
    num_workers = os.cpu_count() // 2
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
    model = Model(
        num_labels=dataset.num_labels,
        num_landmarks=dataset.num_landmarks,
        img_channels=dataset.img_channels,
        img_size=dataset.img_size,
    )

    # loops over runs
    for i_run, run in enumerate(runs):
        if not run["subject_id"] in listdir(line_args["checkpoints_path"]):
            print(
                f"skipping subject {run['subject_id']} as there's no associated checkpoints"
            )
            continue

        # splits the dataset
        subset = Subset(dataset, indices=run["val_idx"])
        dataloader = DataLoader(
            dataset=subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=False,
        )

        # loads the model for the subject
        checkpoint_path = join(
            line_args["checkpoints_path"],
            run["subject_id"],
            [
                filename
                for filename in listdir(
                    join(line_args["checkpoints_path"], run["subject_id"])
                )
                if filename.endswith(".pth")
            ][0],
        )

        model.to("cpu")
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        for param in model.parameters(): param.requires_grad = False;
        model.to(device)
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
        print(f"testing on subject {run['subject_id']} using model {checkpoint_path}")
        # with torch.no_grad():
        #     for batch in tqdm(dataloader):
        #         labels = batch["label"].to(device)
        #         outs = model(**batch)
        #         print(torchmetrics.functional.f1_score(preds=outs["cls_logits"], target=labels, task="multiclass", num_classes=model.num_classes, average="micro"))
        trainer.test(model, dataloader)
        
    # wandb.finish()

    # frees some memory
    del dataset
    gc.collect()
