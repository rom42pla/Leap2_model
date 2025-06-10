from copy import deepcopy
from pprint import pprint
import itertools
import os
from os import makedirs, listdir
from os.path import join, isdir, exists
import shutil
from typing import Dict, List
import pandas as pd
import polars as pl
from PIL import Image
import json

import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
import yaml


class HandPoseDataset(Dataset):
    _possible_images_needed = ["left", "right", "both"]

    def __init__(
        self,
        dataset_path,
        preprocessed_landmarks_path="_preprocessed_landmarks",
        images_needed="left",
        img_size=512,
        normalize_landmarks=True,
    ):
        assert isdir(dataset_path)
        self.dataset_path = dataset_path
        assert images_needed in self._possible_images_needed
        self.images_needed = images_needed
        self.normalize_landmarks = normalize_landmarks

        self.poses_dict = {
            "Dislike": 0,
            "Three": 1,
            "Spiderman": 2,
            "Spok": 3,
            "OK": 4,
            "ClosedFist": 5,
            "Call": 6,
            "L": 7,
            "One": 8,
            "Rock": 9,
            "Four": 10,
            "Stop": 11,
            "Tiger": 12,
            "OpenPalm": 13,
            "Like": 14,
            "C": 15,
            "Two": 16,
        }

        self.preprocessed_dataset_path = preprocessed_landmarks_path
        if isdir(self.preprocessed_dataset_path):
            if not exists(join(self.preprocessed_dataset_path, "info.yaml")):
                raise FileNotFoundError(
                    f"info.yaml not found in {self.preprocessed_dataset_path}. Please delete the folder and run again."
                )
            with open(join(self.preprocessed_dataset_path, "info.yaml"), "r") as fp:
                info = yaml.safe_load(fp)
            if info['normalized_landmarks'] != self.normalize_landmarks:
                shutil.rmtree(self.preprocessed_dataset_path)
                print("rebuilding the preprocessed dataset because of different normalization")
        if not exists(self.preprocessed_dataset_path):
            makedirs(self.preprocessed_dataset_path)
            # parses the cleaned landmarks
            df_landmarks_raw = self.load_raw_landmarks(
                csv_horizontal_path=join(
                    self.dataset_path, "hand_properties_horizontal_cleaned.csv"
                ),
                csv_vertical_path=join(
                    self.dataset_path, "hand_properties_vertical_cleaned.csv"
                ),
            )
            # retrieves the columns indices with actual values from landmarks
            self.landmarks_columns_indices = list(
                range(5, df_landmarks_raw.shape[-1] - 2)
            )
            # fills the missing values in the landmarks
            df_landmarks_raw = self.fill_nans(
                df=df_landmarks_raw,
                landmarks_columns_indices=self.landmarks_columns_indices,
            )
            # standardize and normalize the landmarks
            if self.normalize_landmarks:
                self.df_landmarks_prep = self.preprocess_landmarks(
                    df=df_landmarks_raw,
                    landmarks_columns_indices=self.landmarks_columns_indices,
                )
            # save the preprocessed landmarks on disk
            self.save_landmarks_data_on_disk(
                self.df_landmarks_prep,
                self.dataset_path,
                self.preprocessed_dataset_path,
            )
            # saves the info about the dataset
            info = {
                "normalized_landmarks": self.normalize_landmarks,
            }
            with open(join(self.preprocessed_dataset_path, "info.yaml"), "w") as fp:
                yaml.dump(info, fp, default_flow_style=False)

        # loads the infos for each sample
        self.samples = self.parse_samples(
            dataset_path=self.dataset_path,
            preprocessed_landmarks_path=self.preprocessed_dataset_path,
            poses_dict=self.poses_dict,
            images_needed=self.images_needed,
        )
        self.subject_ids = {sample["subject_id"] for sample in self.samples}
        self.num_labels = len(self.poses_dict)
        self.num_landmarks = (
            np.load(self.samples[0]["landmarks_horizontal"]).size
            + np.load(self.samples[0]["landmarks_vertical"]).size
        )

        # preprocessing for the images
        assert isinstance(
            img_size, int
        ), f"img_size must be an int, got {img_size} ({type(img_size)})"
        assert img_size >= 1, f"img_size must be >= 1, got {img_size}"
        self.img_channels = 1
        self.img_size = img_size
        self.img_shape = (1, self.img_size, self.img_size)
        self.images_transforms = T.Compose(
            [
                T.Resize(size=self.img_size),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

        # mode-related attributes
        self.return_horizontal_landmarks = True
        self.return_vertical_landmarks = True
        self.return_horizontal_images = True
        self.return_vertical_images = True

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_raw_landmarks(csv_horizontal_path, csv_vertical_path):
        columns_to_drop = ["image_path"]
        # loads the viewpoints using polars, which is way faster than pandas
        df_landmarks_horizontal = pl.read_csv(csv_horizontal_path).drop(columns_to_drop)
        df_landmarks_vertical = pl.read_csv(csv_vertical_path).drop(columns_to_drop)
        assert (
            df_landmarks_horizontal.shape == df_landmarks_vertical.shape
        ), f"df_landmarks_horizontal.shape != df_landmarks_vertical.shape"
        # concatenate the two dataframes. the 'device' column keeps track of which landmarks come from which device
        df_landmarks = pl.concat(
            [df_landmarks_horizontal, df_landmarks_vertical], how="vertical"
        )
        assert tuple(df_landmarks.shape) == (
            df_landmarks_horizontal.shape[0] + df_landmarks_vertical.shape[0],
            df_landmarks_horizontal.shape[1],
        ), f"{df_landmarks.shape} != [{df_landmarks_horizontal.shape[0] + df_landmarks_vertical.shape[0]}, {df_landmarks_horizontal.shape[1]}]"
        # and back to pandas
        df_landmarks = df_landmarks.to_pandas(use_pyarrow_extension_array=False)
        return df_landmarks

    @staticmethod
    def load_preprocessed_landmarks(csv_path):
        # loads the viewpoints using polars, which is way faster than pandas
        df = pl.read_csv(csv_path).to_pandas(use_pyarrow_extension_array=False)
        # builds the dictionary
        samples = []
        for row in tqdm(
            pl.from_pandas(df).iter_rows(named=True),
            desc="Building the json",
            total=len(df),
        ):
            samples.append(
                {
                    "frame_id": row["frame_id"],
                    "subject_id": row["subject_id"],
                    "which_hand": row["which_hand"],
                    "pose": row["pose"],
                    "landmarks_horizontal": [
                        row[col] for col in df.columns if col.endswith("_horizontal")
                    ],
                    "landmarks_vertical": [
                        row[col] for col in df.columns if col.endswith("_vertical")
                    ],
                }
            )
        return df

    @staticmethod
    def fill_nans(df, landmarks_columns_indices):
        cols_with_nans = df[df.columns[df.isna().any()]].columns
        for col in cols_with_nans:
            mean = df.loc[:, col].mean()
            assert not np.isnan(mean), f"mean is nan for {col}"
            df.loc[:, col] = df.loc[:, col].fillna(mean)
            assert not df.loc[:, col].isnull().values.any(), f"there are nans in {col}"
        assert not df.isnull().values.any(), f"there are nans"
        return df

    @staticmethod
    def preprocess_landmarks(df, landmarks_columns_indices):
        subject_ids, hands, poses, devices = [
            df[col].unique().tolist()
            for col in ["subject_id", "which_hand", "pose", "device"]
        ]

        df_values = df.values
        for subject_id, hand, pose, device in tqdm(
            list(itertools.product(subject_ids, hands, poses, devices)),
            desc="Preprocessing data",
        ):
            # builds a mask of the rowss that match the current subject, hand, pose, and device
            mask = (
                (df["subject_id"].values == subject_id)
                & (df["which_hand"].values == hand)
                & (df["pose"].values == pose)
                & (df["device"].values == device)
            )
            mask_indices = np.where(mask)[0]
            # mask_indices = mask[mask].index

            data = df_values[np.ix_(mask_indices, landmarks_columns_indices)].astype(
                np.float32
            )
            # data = df.iloc[mask_indices, landmarks_columns_indices]

            # standardize to zero mean and unit variance
            means, stds = data.mean(axis=0), data.std(axis=0)
            standardized_data = (data - means) / (stds + 1e-7)

            # normalize values between -1 and 1
            mins, maxs = standardized_data.min(axis=0), standardized_data.max(axis=0)
            normalized_data = (
                2 * (standardized_data - mins) / ((maxs - mins) - 1 + 1e-7)
            )
            assert not np.isnan(
                normalized_data
            ).any(), f"there are nans in {subject_id}, {hand}, {pose}, {device}"

            # assign the new data
            # df.iloc[mask_indices, landmarks_columns_indices] = normalized_data
            df_values[np.ix_(mask_indices, landmarks_columns_indices)] = normalized_data

        df_prep = pd.DataFrame(df_values, columns=df.columns)

        # splits the df into horizontal and vertical ones, based on the device used
        df_horizontal = df_prep[df_prep["device"] == "Horizontal"]
        df_vertical = df_prep[df_prep["device"] == "Vertical"]
        assert (
            df_horizontal.shape == df_vertical.shape
        ), f"{df_horizontal.shape} != {df_vertical.shape}"

        # merge the two dataframes in a single one
        id_cols = ["frame_id", "subject_id", "which_hand", "pose"]
        df = pd.merge(
            left=df_horizontal,
            right=df_vertical,
            how="inner",
            on=id_cols,
            suffixes=("_horizontal", "_vertical"),
        )
        assert (
            df.shape[0] == df_horizontal.shape[0]
        ), f"{df.shape} != {df_horizontal.shape}"
        assert all(
            [col in df.columns for col in id_cols]
        ), f"{id_cols} not in {df.columns}"
        df = df.reset_index().drop(
            columns=[
                col
                for col in df.columns
                if col.startswith("device") or col.startswith("pose_index")
            ]
        )

        return df

    @staticmethod
    def save_landmarks_data_on_disk(df, dataset_path, output_path, img_size=224):
        for row in tqdm(
            pl.from_pandas(df).iter_rows(named=True),
            desc=f"Saving preprocessed landmarks to {output_path}",
            total=len(df),
        ):
            # parses all the metas
            frame_id, subject_id, hand, pose = [
                str(row[col])
                for col in ["frame_id", "subject_id", "which_hand", "pose"]
            ]
            landmarks_path = join(output_path, subject_id.zfill(3), hand, pose)
            # eventually creates the folder
            for device in ["Horizontal", "Vertical"]:
                # retrieves the landmarks from the big dataframe
                arr_landmarks = np.asarray(
                    [
                        float(row[col])
                        for col in df.columns
                        if col.endswith(f"_{device.lower()}")
                    ]
                ).astype(np.float32)
                # saves the landmarks to disk
                landmarks_per_device_path = join(landmarks_path, device, "landmarks")
                if not isdir(landmarks_per_device_path):
                    os.makedirs(landmarks_per_device_path)
                np.save(
                    join(landmarks_per_device_path, f"{frame_id.zfill(3)}.npy"),
                    arr_landmarks,
                )
                # # saves images on disk
                # dataset_images_path = join(dataset_path, str(subject_id).zfill(3), hand, pose, device, "images")
                # preprocessed_images_path = join(output_path, str(subject_id).zfill(3), hand, pose, device, "images")
                # if not isdir(preprocessed_images_path):
                #     os.makedirs(preprocessed_images_path)
                # for image_name in listdir(dataset_images_path):
                #     if not image_name.endswith(".bmp"):
                #         continue
                #     image = Image.open(join(dataset_images_path, image_name))
                #     image = image.resize((img_size, img_size))
                #     image.save(join(preprocessed_images_path, image_name[-4:]+".jpg"), optimize=True, quality=50)

    @staticmethod
    def parse_samples(
        dataset_path, preprocessed_landmarks_path, images_needed="both", poses_dict=None
    ):
        assert images_needed in {"left", "right", "both"}, f"got {images_needed}"
        samples = []
        subject_ids = [f for f in listdir(dataset_path) if isdir(join(dataset_path, f))]
        hands = listdir(join(dataset_path, subject_ids[0]))
        poses = listdir(join(dataset_path, subject_ids[0], hands[0]))
        frame_ids = [
            filename.split("_")[0].zfill(3)
            for filename in listdir(
                join(
                    dataset_path,
                    subject_ids[0],
                    hands[0],
                    poses[0],
                    "Horizontal",
                    "images",
                )
            )
        ]
        for subject_id, hand, pose, frame_id in tqdm(
            list(itertools.product(subject_ids, hands, poses, frame_ids)),
            desc=f"Parsing samples",
        ):
            if hand == "Left_Hand" and images_needed == "right":
                continue
            elif hand == "Right_Hand" and images_needed == "left":
                continue
            # parses the sample
            sample = {
                "subject_id": subject_id,
                "hand": hand,
                "pose": pose,
            }
            if poses_dict is not None:
                if pose not in poses_dict:
                    raise BaseException(
                        f"Unrecognized pose '{pose}'. Poses are: {list(poses_dict.keys())}"
                    )
                sample["label"] = poses_dict[pose]
            for device in ["Horizontal", "Vertical"]:
                # parses the landmarks
                sample[f"landmarks_{device.lower()}"] = join(
                    preprocessed_landmarks_path,
                    subject_id,
                    hand,
                    pose,
                    device,
                    "landmarks",
                    f"{frame_id}.npy",
                )
                assert exists(
                    sample[f"landmarks_{device.lower()}"]
                ), f"{sample[f'landmarks_{device.lower()}']} does not exist"
                # parses the images
                for direction in ["left", "right"]:
                    sample[f"image_{device.lower()}_{direction}"] = join(
                        dataset_path,
                        subject_id,
                        hand,
                        pose,
                        device,
                        "images",
                        f"{frame_id}_{direction}.bmp",
                    )
                    assert exists(
                        sample[f"image_{device.lower()}_{direction}"]
                    ), f"{sample[f'image_{device.lower()}_{direction}']} does not exist"
            samples.append(sample)
        return samples

    def get_indices_per_subject(self) -> Dict[str, List[int]]:
        indices_per_subject = {}
        for i_sample, sample in enumerate(self.samples):
            subject_id = sample["subject_id"]
            if subject_id not in indices_per_subject:
                indices_per_subject[subject_id] = []
            indices_per_subject[subject_id].append(i_sample)
        return indices_per_subject

    def set_mode(
        self,
        return_horizontal_images=None,
        return_vertical_images=None,
        return_horizontal_landmarks=None,
        return_vertical_landmarks=None,
    ):
        if return_horizontal_images is not None:
            assert isinstance(return_horizontal_images, bool)
            self.return_horizontal_images = return_horizontal_images
        if return_vertical_images is not None:
            assert isinstance(return_vertical_images, bool)
            self.return_vertical_images = return_vertical_images
        if return_horizontal_landmarks is not None:
            assert isinstance(return_horizontal_landmarks, bool)
            self.return_horizontal_landmarks = return_horizontal_landmarks
        if return_vertical_landmarks is not None:
            assert isinstance(return_vertical_landmarks, bool)
            self.return_vertical_landmarks = return_vertical_landmarks
        # sets new correct parameters
        self.img_channels = sum(
            [
                1 if self.return_horizontal_images else 0,
                1 if self.return_vertical_images else 0,
            ]
        )
        self.num_landmarks = sum(
            [
                (
                    np.load(self.samples[0]["landmarks_horizontal"]).size
                    if self.return_horizontal_landmarks
                    else 0
                ),
                (
                    np.load(self.samples[0]["landmarks_vertical"]).size
                    if self.return_vertical_landmarks
                    else 0
                ),
            ]
        )

    @staticmethod
    def _get_subject_ids() -> List[str]:
        return sorted([str(i).zfill(3) for i in range(1, 21+1)])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        outs = {"label": sample["label"]}
        for key in sample:
            if key.startswith("image"):
                # skips some images that are not needed
                if key.endswith("left") and self.images_needed == "right":
                    continue
                elif key.endswith("right") and self.images_needed == "left":
                    continue
                # loads the image
                image = self.images_transforms(Image.open(sample[key]))
                # adjusts some keys if there is just one image needed
                if "horizontal" in key and self.return_horizontal_images:
                    outs["image_horizontal"] = image
                elif "vertical" in key and self.return_vertical_images:
                    outs["image_vertical"] = image
            elif key.startswith("landmarks"):
                if ("horizontal" in key and self.return_horizontal_landmarks) or (
                    "vertical" in key and self.return_vertical_landmarks
                ):
                    outs[key] = torch.from_numpy(
                        np.load(sample[key], allow_pickle=True)
                    ).float()
                
        # checks if the keys are correct
        if self.return_horizontal_landmarks and "landmarks_horizontal" not in outs:
            raise KeyError(
                "landmarks_horizontal not in outs, but return_horizontal_landmarks is True"
            )
        if self.return_vertical_landmarks and "landmarks_vertical" not in outs:
            raise KeyError(
                "landmarks_vertical not in outs, but return_vertical_landmarks is True"
            )
        if self.return_horizontal_images and "image_horizontal" not in outs:
            raise KeyError(
                "image_horizontal not in outs, but return_horizontal_images is True"
            )
        if self.return_vertical_images and "image_vertical" not in outs:
            raise KeyError(
                "image_vertical not in outs, but return_vertical_images is True"
            )
        
        # returns the outputs
        return outs


if __name__ == "__main__":
    dataset = HandPoseDataset(dataset_path="../../datasets/ml2hp")
    for k, v in dataset[0].items():
        if isinstance(v, torch.Tensor):
            print(
                f"key {k} with shape {tuple(v.shape)} has: min {v.min()}, max {v.max()}, mean {v.mean()}, std {v.std()}"
            )

    # tests
    for (
        return_horizontal_images,
        return_vertical_images,
        return_horizontal_landmarks,
        return_vertical_landmarks,
    ) in tqdm(
        list(
            itertools.product(
                [True, False], [True, False], [True, False], [True, False]
            )
        ),
        desc="trying all modes combinations",
    ):
        dataset.set_mode(
            return_horizontal_images=return_horizontal_images,
            return_vertical_images=return_vertical_images,
            return_horizontal_landmarks=return_horizontal_landmarks,
            return_vertical_landmarks=return_vertical_landmarks,
        )
        batch = dataset[
            0
        ]  # keys are ['label', 'landmarks_horizontal', 'image_horizontal', 'landmarks_vertical', 'image_vertical']
        if return_horizontal_images:
            assert "image_horizontal" in batch, f"keys are {batch.keys()}"
        if return_vertical_images:
            assert "image_vertical" in batch
        if return_horizontal_landmarks:
            assert "landmarks_horizontal" in batch
        if return_vertical_landmarks:
            assert "landmarks_vertical" in batch
