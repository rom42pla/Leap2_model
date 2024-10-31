import os
import pandas as pd
from PIL import Image

import torch

from torch.utils.data import Dataset


def collate_fn(batch):
    '''
    Function to collate the batch of data
    '''
    landmarks_horizontal = [item[0] for item in batch]
    landmarks_vertical = [item[1] for item in batch]
    horizontal_image = [item[2] for item in batch]
    vertical_image = [item[3] for item in batch]
    subject_id = [item[4] for item in batch]
    hands = [item[5] for item in batch]
    poses = [item[6] for item in batch]
    frames_id = [item[7] for item in batch]
    labels = [item[8] for item in batch]

    return landmarks_horizontal, landmarks_vertical, horizontal_image, vertical_image, subject_id, hands, poses, frames_id, labels


class HandPoseDataset(Dataset):
    def __init__(self, dfs_horizontal_viewpoints, dfs_vertical_viewpoints, img_path):
        self.dfs_horizontal_viewpoints = dfs_horizontal_viewpoints
        self.dfs_vertical_viewpoints = dfs_vertical_viewpoints
        self.img_path = img_path

    def __len__(self):
        return len(self.dfs_horizontal_viewpoints)
    
    def __getitem__(self, idx):
        '''
        Function to get the item at the idx position
        '''
        # Take the landmarks which from the column number 5 up to the last two
        landmarks_horizontal = torch.tensor(self.dfs_horizontal_viewpoints.iloc[idx, 5:-2].values.astype(float))
        landmarks_vertical = torch.tensor(self.dfs_vertical_viewpoints.iloc[idx, 5:-2].values.astype(float))
        # Take the sample from the dataframes
        sample_horizontal = self.dfs_horizontal_viewpoints.iloc[idx]
        sample_vertical = self.dfs_vertical_viewpoints.iloc[idx]

        # Take the subject_id from the dataframes
        subject_id_horizontal = sample_horizontal['subject_id']
        subject_id_vertical = sample_vertical['subject_id']

        # Check that is the same subject and then put subj_id in the form "0XX" being XX the number of the subject
        if subject_id_horizontal != subject_id_vertical:
            raise ValueError(f"Subject ID {subject_id_horizontal} and {subject_id_vertical} do not match")
        subject_id = f"{subject_id_horizontal:03}"

        # Check that the pose is the same
        if sample_horizontal['pose'] != sample_vertical['pose']:
            raise ValueError(f"Pose {sample_horizontal['pose']} and {sample_vertical['pose']} do not match")
        
        # Check that the hand is the same
        if sample_horizontal['which_hand'] != sample_vertical['which_hand']:
            raise ValueError(f"Hand {sample_horizontal['which_hand']} and {sample_vertical['which_hand']} do not match")
        
        # Check that the frame_id is the same
        if sample_horizontal['frame_id'] != sample_vertical['frame_id']:
            raise ValueError(f"Frame ID {sample_horizontal['frame_id']} and {sample_vertical['frame_id']} do not match")

        # Take the pose
        pose = sample_horizontal['pose']

        # Take the frame_id and put it in the form "0XX" being XX the number of the frame_id considering it can go up to 1000
        frame_id = f"{sample_horizontal['frame_id']:03}"

        # Take the label amd which hand is being used
        hand = sample_horizontal['which_hand']
        label = sample_horizontal['pose_index']

        # Load the images for the horizontal and vertical viewpoints that matches the sample and convert them to RGB
        horizontal_image = Image.open(os.path.join(self.img_path, subject_id_horizontal, hand, pose, 'Horizontal', 'images', f"{frame_id}_left.bmp")).convert('RGB')
        vertical_image = Image.open(os.path.join(self.img_path, subject_id_vertical, hand, pose, 'Vertical', 'images', f"{frame_id}_left.bmp")).convert('RGB')
                
        return landmarks_horizontal, landmarks_vertical, horizontal_image, vertical_image, subject_id, hand, pose, frame_id, label