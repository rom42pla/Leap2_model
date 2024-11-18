import argparse
from os import makedirs
from os.path import join
import itertools
import yaml

# parses line args
parser = argparse.ArgumentParser(
    prog="Leap2 Model",
    description="Generate configs for the pipeline"
)
parser.add_argument("--path", default=join(".", "cfgs"), help="Where to save the configs")
parser.add_argument("--datasets_path", default=join("..", "..", "datasets"),
                    help="Where the datasets are located")
parser.add_argument("--batch_size", default=32)
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--max_epochs", default=10)
parser.add_argument("--seed", default=42)
line_args = vars(parser.parse_args())

# creates the directory
makedirs(line_args["path"], exist_ok=True)

# defines the parameters
datasets = ["ml2hp"]
validations = ["fixed", "kfold", "loso"]

# loops over each configuration
for dataset, validation in itertools.product(datasets, validations):
    content = {
        "dataset": dataset,
        "dataset_path": join(line_args["datasets_path"], dataset),
        "device": "auto",
        "seed": line_args["seed"],
        "validation": validation,
        "k": 10,
        "train_perc": 0.8,
        "checkpoints_path": "./checkpoints",
        "batch_size": line_args["batch_size"],
        "max_epochs": line_args["max_epochs"],
        "lr": line_args["lr"],
    }
    filename = f"{dataset}_{validation}.yaml"
    with open(join(line_args["path"], filename), 'w') as file:
        yaml.dump(content, file)