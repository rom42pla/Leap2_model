
import os
from pprint import pprint
import argparse
import datetime
import yaml
from tqdm import tqdm

from datasets.hand_pose_dataset import HandPoseDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # arguments parsing
    parser = argparse.ArgumentParser(description='Train a multimodal model')
    parser.add_argument("cfg_path", type=str, help='Path to the configuration')
    line_args = vars(parser.parse_args())
    
    # loads the configuration file
    num_workers = os.cpu_count() // 2
    with open(line_args["cfg_path"], 'r') as fp:
        cfg = yaml.safe_load(fp)
    pprint(cfg)
    
    train_dataset = HandPoseDataset(
        dataset_path=cfg["dataset_path"],
    )
    val_dataset = train_dataset

    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=os.cpu_count() // 2)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=os.cpu_count() // 2)

    for batch in tqdm(train_dataloader, desc="Reading entire dataset..."):
        pass

    print("Done!")