# Leap2_dataloader
Repository to efficiently load Leap2 data. The entire pipeline is meant to process the data using Pytorch dataloaders.

## TL;DR
`python train.py cfgs/ml2hp_fixed.yaml`

## Pre-processing

If it does not already exist, a folder `_preprocessed_dataset` inside the current directory is created.
This folder will contain the preprocessed landmarks, for faster use during training/inference.

## Pre-requisites

### Dependencies
The required dependendencies to run the code are defined in the `requirements.txt` file. 
For simplicity, use `conda` to create a virtual environment:

```bash
conda create --name ml2hp
conda activate ml2hp
```

Once in the `ml2hp` environment, install the dependencies:

```bash
conda install pip
pip install -r requirements.txt
```

### Datasets

Make sure to put the `ml2hp` dataset into `../../datasets`:

```
|-- datasets
|   `-- ml2hp
|       |-- 001
|       |-- 002
|       |-- 003
|       |-- 004
|       |-- 005
|       |-- 006
|       |-- 007
|       |-- 008
|       |-- 009
|       |-- 010
|       |-- 011
|       |-- 012
|       |-- 013
|       |-- 014
|       |-- 015
|       |-- 016
|       |-- 017
|       |-- 018
|       |-- 019
|       |-- 020
|       |-- 021
|       |-- hand_properties_horizontal_cleaned.csv
|       |-- hand_properties_vertical_cleaned.csv
|       |-- Readme.txt
|       `-- subjects_info.csv
|-- repos
|   |-- Leap2_model
|   |   |-- cfgs
|   |   |-- checkpoints
|   |   |-- datasets
|   |   |-- generate_configs.py
|   |   |-- _initial_state_dict.pth
|   |   |-- ml2hp
|   |   |-- model.py
|   |   |-- plots.py
|   |   |-- _preprocessed_landmarks
|   |   |-- __pycache__
|   |   |-- README.md
|   |   |-- requirements.txt
|   |   |-- train.py
|   |   |-- utils.py
|   |   `-- wandb
```