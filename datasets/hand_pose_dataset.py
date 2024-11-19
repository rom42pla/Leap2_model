import os
from os.path import join, isdir
import pandas as pd
from PIL import Image

import torch

from torch.utils.data import Dataset
from torchvision import transforms as T


class HandPoseDataset(Dataset):
    def __init__(self, 
                 dataset_path, 
                 img_size=224):
        assert isdir(dataset_path)
        self.dataset_path = dataset_path
        
        self.poses_dict = {
            "Dislike": 0, "Three": 1, "Spiderman": 2, "Spok": 3, 
                           "OK": 4, "ClosedFist": 5, "Call": 6, "L": 7,
                           "One": 8, "Rock": 9, "Four": 10, "Stop": 11, 
                           "Tiger": 12, "OpenPalm": 13, "Like": 14, "C": 15, "Two": 16
            }
        
        # loads the viewpoints
        self.dfs_horizontal_viewpoints = pd.read_csv(join(self.dataset_path, 'hand_properties_horizontal_cleaned.csv'))
        self.dfs_vertical_viewpoints = pd.read_csv(
            join(self.dataset_path, 'hand_properties_vertical_cleaned.csv'))
        
        # replace the nan values with the mean of the column
        cols_with_nan_horizontal = self.dfs_horizontal_viewpoints.columns[self.dfs_horizontal_viewpoints.isna(
        ).any()]
        cols_with_nan_vertical = self.dfs_vertical_viewpoints.columns[self.dfs_vertical_viewpoints.isna(
        ).any()]
        self.dfs_horizontal_viewpoints[cols_with_nan_horizontal] = self.dfs_horizontal_viewpoints[cols_with_nan_horizontal].apply(
            lambda col: col.fillna(col.mean()), axis=0)
        self.dfs_vertical_viewpoints[cols_with_nan_vertical] = self.dfs_vertical_viewpoints[cols_with_nan_vertical].apply(
            lambda col: col.fillna(col.mean()), axis=0)
        
        self.subject_ids = sorted(self.dfs_horizontal_viewpoints["subject_id"].unique().tolist())
        self.num_labels = len(self.poses_dict)
        self.num_landmarks = self.dfs_horizontal_viewpoints.iloc[:,
                                                                 5:-2].shape[-1] + self.dfs_vertical_viewpoints.iloc[:, 5:-2].shape[-1]
        
        # preprocessing for the images
        assert isinstance(img_size, int), f"img_size must be an int, got {img_size} ({type(img_size)})"
        assert img_size >= 1, f"img_size must be >= 1, got {img_size}"
        self.img_channels = 1
        self.img_size = img_size
        self.img_shape = (1, self.img_size, self.img_size)
        self.images_transforms = T.Compose([
            T.Resize(size=self.img_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.dfs_horizontal_viewpoints)
    
    def get_indices_per_subject(self):
        indices_per_subject = {}
        for i_row, row_subject in enumerate(self.dfs_horizontal_viewpoints["subject_id"].tolist()):
            if row_subject not in indices_per_subject:
                indices_per_subject[row_subject] = []
            indices_per_subject[row_subject].append(i_row)
        return indices_per_subject
    
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
        horizontal_image = Image.open(os.path.join(self.dataset_path, f"{subject_id_horizontal:03}", hand, pose, 'Horizontal', 'images', f"{frame_id}_left.bmp")).convert('RGB')
        vertical_image = Image.open(os.path.join(self.dataset_path, f"{subject_id_vertical:03}", hand, pose, 'Vertical', 'images', f"{frame_id}_left.bmp")).convert('RGB')
        
        horizontal_image = self.images_transforms(horizontal_image)
        vertical_image = self.images_transforms(vertical_image)
        
        assert tuple(horizontal_image.shape) == self.img_shape, f"expected shape {self.img_shape}, got {horizontal_image.shape}" 
        assert tuple(
            vertical_image.shape) == self.img_shape, f"expected shape {self.img_shape}, got {vertical_image.shape}"
        # return landmarks_horizontal, landmarks_vertical, horizontal_image, vertical_image, subject_id, hand, pose, frame_id, label
        
        outs = {
            "landmarks_horizontal": landmarks_horizontal,
            "landmarks_vertical": landmarks_vertical,
            "horizontal_image": horizontal_image,
            "vertical_image": vertical_image,
            "subject_id": subject_id,
            "hand": hand,
            "pose": pose,
            "frame_id": frame_id,
            "label": label,
        }
        return outs
        