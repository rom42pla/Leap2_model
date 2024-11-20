import itertools
from multiprocessing import Pool
import os
from os.path import join, isdir
import pandas as pd
import polars as pl
from PIL import Image

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm


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
        self.dfs_landmarks_horizontal = pl.read_csv(
            join(self.dataset_path, 'hand_properties_horizontal_cleaned.csv'))
        self.dfs_landmarks_vertical = pl.read_csv(
            join(self.dataset_path, 'hand_properties_vertical_cleaned.csv'))
        assert self.dfs_landmarks_horizontal.shape == self.dfs_landmarks_vertical.shape

        self.subject_ids = sorted(
            self.dfs_landmarks_horizontal["subject_id"].unique().to_list())
        self.hands = sorted(
            self.dfs_landmarks_horizontal["which_hand"].unique().to_list())
        self.poses = sorted(
            self.dfs_landmarks_horizontal["pose"].unique().to_list())
        assert set(self.poses) == set(self.poses_dict.keys()), f"there are unknown poses"

        self.landmarks_columns_indices = list(
            range(5, self.dfs_landmarks_horizontal.shape[-1] - 2))
        self.landmarks_columns = [self.dfs_landmarks_horizontal.columns[i] for i in self.landmarks_columns_indices]

        # # replace the nan values with the mean of the column
        # for df in [self.dfs_landmarks_horizontal, self.dfs_landmarks_vertical]:
        #     df_values_only = df[:, self.landmarks_columns_indices]
        #     cols_with_nans = df_values_only[df_values_only.columns[df_values_only.is_nan(
        #     ).any()]].columns
        #     df[cols_with_nans] = df[cols_with_nans].apply(
        #         lambda col: col.fillna(col.mean()), axis=0)
        #     assert not df.isna().values.any(), f"there are nans in df_landmarks"
        
        for df in [self.dfs_landmarks_horizontal, self.dfs_landmarks_vertical]:
            # identify columns with null values
            nulls = df[:, self.landmarks_columns_indices].null_count()
            cols_with_nans = [col for col in nulls.columns if nulls[col].sum() > 0]
            for col in cols_with_nans:
                df = df.with_columns([pl.col(col).fill_null(pl.col(col).mean()).alias(col)])
            assert df.null_count().to_numpy().sum() == 0, \
                "There are still NaNs in df_landmarks"

        self.subject_ids = sorted(
            self.dfs_landmarks_horizontal["subject_id"].unique().to_list())
        self.num_labels = len(self.poses_dict)
        self.num_landmarks = self.dfs_landmarks_horizontal[:, 5:-2].shape[-1] + self.dfs_landmarks_vertical[:, 5:-2].shape[-1]
        self.hands = sorted(
            self.dfs_landmarks_horizontal["which_hand"].unique().to_list())
        
        # Loop through horizontal and vertical landmarks
        for df in [self.dfs_landmarks_horizontal, self.dfs_landmarks_vertical]:
            df = df.to_pandas(use_pyarrow_extension_array=False)
            for subject, hand, pose in tqdm(
                list(itertools.product(self.subject_ids, self.hands, self.poses)),
                desc="Normalizing data"
            ):
                mask = (df["subject_id"] == subject) & (
                    df["which_hand"] == hand) & (
                    df["pose"] == pose)
                mask_indices = mask[mask].index

                data = df.iloc[mask_indices, self.landmarks_columns_indices]   

                # standardize to zero mean and unit variance
                means, stds = data.mean(axis=0), data.std(axis=0)
                standardized_data = (data - means) / (stds + 1e-7)
                
                # normalize values between -1 and 1
                mins, maxs = standardized_data.min(axis=0), standardized_data.max(axis=0)
                normalized_data = 2 * \
                    (standardized_data - mins) / (maxs - mins) - 1
                    
                # assign the new data
                df.loc[mask, self.landmarks_columns] = normalized_data
                
                    
                
                    
                
                
        # preprocessing for the images
        assert isinstance(
            img_size, int), f"img_size must be an int, got {img_size} ({type(img_size)})"
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
        return len(self.dfs_landmarks_horizontal)

    @staticmethod
    def normalize_landmarks(args):
        """Function to normalize landmarks for a specific subject and hand."""
        subject, hand, landmarks_columns, dfs = args
        dfs_normalized = []

        for df in dfs:
            mask = (df["subject_id"] == subject) & (df["which_hand"] == hand)

            # Standardize to zero mean and unit variance
            stats = df.loc[mask, landmarks_columns].agg(["mean", "std"])
            means, stds = stats.loc["mean"], stats.loc["std"]
            assert all(len(stat) == df.loc[mask, landmarks_columns].shape[1] for stat in [means, stds]), \
                f"Statistics mismatch for subject {subject}, hand {hand}"
            df.loc[mask, landmarks_columns] = (
                df.loc[mask, landmarks_columns] - means) / stds

            # Normalize values between -1 and 1
            stats = df.loc[mask, landmarks_columns].agg(["min", "max"])
            mins, maxs = stats.loc["min"], stats.loc["max"]
            assert all(len(stats) == df.loc[mask, landmarks_columns].shape[1] for stats in [mins, maxs]), \
                f"Statistics mismatch for subject {subject}, hand {hand}"
            df.loc[mask, landmarks_columns] = (
                2 * (df.loc[mask, landmarks_columns] - mins) / (maxs - mins)) - 1

            dfs_normalized.append(df)

        return dfs_normalized

    def get_indices_per_subject(self):
        indices_per_subject = {}
        for i_row, row_subject in enumerate(self.dfs_landmarks_horizontal["subject_id"].tolist()):
            if row_subject not in indices_per_subject:
                indices_per_subject[row_subject] = []
            indices_per_subject[row_subject].append(i_row)
        return indices_per_subject

    def __getitem__(self, idx):
        '''
        Function to get the item at the idx position
        '''
        # Take the landmarks which from the column number 5 up to the last two
        landmarks_horizontal = torch.tensor(
            self.dfs_landmarks_horizontal.iloc[idx, 5:-2].values.astype(float))
        landmarks_vertical = torch.tensor(
            self.dfs_landmarks_vertical.iloc[idx, 5:-2].values.astype(float))
        # Take the sample from the dataframes
        sample_horizontal = self.dfs_landmarks_horizontal.iloc[idx]
        sample_vertical = self.dfs_landmarks_vertical.iloc[idx]

        # Take the subject_id from the dataframes
        subject_id_horizontal = sample_horizontal['subject_id']
        subject_id_vertical = sample_vertical['subject_id']

        # Check that is the same subject and then put subj_id in the form "0XX" being XX the number of the subject
        if subject_id_horizontal != subject_id_vertical:
            raise ValueError(
                f"Subject ID {subject_id_horizontal} and {subject_id_vertical} do not match")
        subject_id = f"{subject_id_horizontal:03}"

        # Check that the pose is the same
        if sample_horizontal['pose'] != sample_vertical['pose']:
            raise ValueError(
                f"Pose {sample_horizontal['pose']} and {sample_vertical['pose']} do not match")

        # Check that the hand is the same
        if sample_horizontal['which_hand'] != sample_vertical['which_hand']:
            raise ValueError(
                f"Hand {sample_horizontal['which_hand']} and {sample_vertical['which_hand']} do not match")

        # Check that the frame_id is the same
        if sample_horizontal['frame_id'] != sample_vertical['frame_id']:
            raise ValueError(
                f"Frame ID {sample_horizontal['frame_id']} and {sample_vertical['frame_id']} do not match")

        # Take the pose
        pose = sample_horizontal['pose']

        # Take the frame_id and put it in the form "0XX" being XX the number of the frame_id considering it can go up to 1000
        frame_id = f"{sample_horizontal['frame_id']:03}"

        # Take the label amd which hand is being used
        hand = sample_horizontal['which_hand']
        label = sample_horizontal['pose_index']

        # Load the images for the horizontal and vertical viewpoints that matches the sample and convert them to RGB
        horizontal_image = Image.open(os.path.join(
            self.dataset_path, f"{subject_id_horizontal:03}", hand, pose, 'Horizontal', 'images', f"{frame_id}_left.bmp")).convert('RGB')
        vertical_image = Image.open(os.path.join(
            self.dataset_path, f"{subject_id_vertical:03}", hand, pose, 'Vertical', 'images', f"{frame_id}_left.bmp")).convert('RGB')

        horizontal_image = self.images_transforms(horizontal_image)
        vertical_image = self.images_transforms(vertical_image)

        assert tuple(
            horizontal_image.shape) == self.img_shape, f"expected shape {self.img_shape}, got {horizontal_image.shape}"
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


if __name__ == "__main__":
    dataset = HandPoseDataset(dataset_path="../../datasets/ml2hp")
    for k, v in dataset[0].items():
        if isinstance(v, torch.Tensor):
            print(
                f"key {k} with shape {tuple(v.shape)} has: min {v.min()}, max {v.max()}, mean {v.mean()}, std {v.std()}")
