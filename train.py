import os
from os import makedirs
from os.path import join, splitext, basename
import gc
from pprint import pprint
import argparse
from datetime import datetime
import torch
from torch.utils.data import Subset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
import wandb
import yaml
from tqdm import tqdm

from datasets.hand_pose_dataset import HandPoseDataset

from model import Model
from utils import (
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
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--cfg", type=str, help="Path to the configuration", required=True)
    parser.add_argument("--run_name", type=str, help="The name to prepend to the run when logging", required=False)
    line_args = vars(parser.parse_args())

    # loads the configuration file
    num_workers = os.cpu_count() // 2 # type: ignore
    with open(line_args["cfg"], "r") as fp:
        cfg = yaml.safe_load(fp)
    pprint(cfg)

    # sets the random seed
    set_global_seed(seed=cfg["seed"])

    # sets the logging folder
    datetime_str: str = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name: str = f"{datetime_str}_{cfg['dataset']}_{cfg['validation']}"
    experiment_path: str = join(cfg["checkpoints_path"], experiment_name)
    makedirs(experiment_path, exist_ok=True)

    # sets up the dataset(s)
    dataset = HandPoseDataset(
        dataset_path=cfg["dataset_path"],
    )

    # sets up the validation scheme
    if cfg["validation"] in ["k_fold", "kfold"]:
        raise NotImplementedError
        runs = get_k_fold_runs(k=args["k"], dataset=dataset)
    elif cfg["validation"] == "loso":
        runs = get_loso_runs(dataset=dataset)
    elif cfg["validation"] == "simple":
        runs = get_simple_runs(dataset=dataset, train_perc=cfg["train_perc"])
    else:
        raise NotImplementedError

    # setup the model
    device = get_device_from_string(cfg["device"])  # "cuda" or "cpu"
    model = Model(
        num_labels=dataset.num_labels,
        num_landmarks=dataset.num_landmarks,
        img_channels=dataset.img_channels,
        img_size=dataset.img_size,
    )

    # saves the initial weights of the model
    initial_state_dict_path = join(".", "_initial_state_dict.pth")
    torch.save({"model_state_dict": model.state_dict()}, initial_state_dict_path)

    # metas
    date = datetime.now().strftime("%Y%m%d_%H%M")
    cfg_name = splitext(basename(line_args["cfg"]))[0]
    experiment_name = f"{date}_{cfg_name}"
    if line_args["run_name"] is not None:
        experiment_name += f"_{line_args['run_name']}"
    # saves the parameters used in the config
    with open(join(experiment_path, "cfg.yaml"), 'w') as fp:
        yaml.dump(cfg, fp, default_flow_style=False)

    # loops over runs
    for i_run, run in enumerate(runs):
        if cfg["validation"] == "loso":
            print(
                f"doing run for subject {run['subject_id']} ({((i_run+1)/len(runs)) * 100:.1f}%)" # type: ignore
            )
            run_name = run['subject_id'] # type: ignore
            
        else:
            print(
                f"doing run {i_run+1} of {len(runs)} ({((i_run+1)/len(runs)) * 100:.1f}%)"
            )
            run_name = f"run_{i_run}"
        experiment_run_path = join(experiment_path, run_name)
        makedirs(experiment_run_path, exist_ok=True)

        # splits the dataset
        dataloader_train = DataLoader(
            dataset=Subset(dataset, indices=run["train_idx"]), # type: ignore
            batch_size=cfg["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=num_workers // 2,
            persistent_workers=True,
        )
        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]), # type: ignore
            batch_size=cfg["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=True,
        )

        # initialize the model
        model.to("cpu")
        model.load_state_dict(
            torch.load(initial_state_dict_path, weights_only=True)["model_state_dict"]
        )
        model.to(device)

        wandb_logger = WandbLogger(
            project="ml2hp", name=experiment_name, log_model=False, prefix=run_name
        )

        # do the training
        checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_run_path,
            filename="{epoch:01d}-{cls_f1_loss:.4f}-{cls_f1_val:.4f}",
            monitor=f"cls_f1_val",
            mode="max",
            every_n_epochs=1,
        )
        trainer = Trainer(
            logger=wandb_logger,
            accelerator=device,
            precision="16-mixed",
            max_epochs=cfg["max_epochs"],
            enable_model_summary=True,
            enable_checkpointing=True,
            default_root_dir=experiment_run_path,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, dataloader_train, dataloader_val)
    wandb.finish()

    # frees some memory
    del dataset
    gc.collect()
