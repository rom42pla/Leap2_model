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


class MultiModalHandGestureDatasetForHandGestureRecognition(Dataset):
    _possible_splits = ["train", "test", "all"]

    def __init__(
        self,
        dataset_path,
        preprocessed_landmarks_path="_mmhgdhgr_preprocessed_landmarks",
        img_size=256,
        split="train",
        normalize_landmarks=True,
    ):
        assert isdir(dataset_path)
        self.dataset_path = dataset_path
        self.normalize_landmarks = normalize_landmarks
        assert (
            split in self._possible_splits
        ), f"split must be one of {self._possible_splits}, got {split}"
        self.split = split

        self.poses_dict = {
            "C": 0,
            "down": 1,
            "fist_moved": 2,
            "five": 3,
            "four": 4,
            "hang": 5,
            "heavy": 6,
            "index": 7,
            "L": 8,
            "ok": 9,
            "palm": 10,
            "palm_m": 11,
            "palm_u": 12,
            "three": 13,
            "two": 14,
            "up": 15,
        }

        self.preprocessed_dataset_path = preprocessed_landmarks_path
        if isdir(self.preprocessed_dataset_path):
            # checks if the info.yaml file exists
            if not exists(join(self.preprocessed_dataset_path, "info.yaml")):
                raise FileNotFoundError(
                    f"info.yaml not found in {self.preprocessed_dataset_path}. Please delete the folder and run again."
                )
            # creates the info.yaml file
            with open(join(self.preprocessed_dataset_path, "info.yaml"), "r") as fp:
                info = yaml.safe_load(fp)
            # eventually delete the existing saved landmarks
            if info["normalized_landmarks"] != self.normalize_landmarks:
                shutil.rmtree(self.preprocessed_dataset_path)
                print(
                    "rebuilding the preprocessed dataset because of different normalization"
                )
        if not exists(self.preprocessed_dataset_path):
            # creates the preprocessed dataset folder
            makedirs(self.preprocessed_dataset_path)
            # parses the cleaned landmarks
            df_landmarks_raw = self.load_raw_landmarks(
                df_landmarks_train_path=join(
                    self.dataset_path,
                    "pose",
                    "train_pose",
                    "coordinates_frames_labelled.csv",
                ),
                df_landmarks_test_path=join(
                    self.dataset_path,
                    "pose",
                    "test_pose",
                    "coordinates_frames_labelled.csv",
                ),
            )
            # retrieves the columns indices with actual values from landmarks
            self.landmarks_columns_indices = list(range(1, 43))
            # fills the missing values in the landmarks
            # df_landmarks_raw = self.fill_nans(
            #     df=df_landmarks_raw,
            #     landmarks_columns_indices=self.landmarks_columns_indices,
            # )
            # standardize and normalize the landmarks
            self.df_landmarks_prep = self.preprocess_landmarks(
                df=df_landmarks_raw,
                landmarks_columns_indices=self.landmarks_columns_indices,
                normalize_landmarks=self.normalize_landmarks,
            )
            # save the preprocessed landmarks on disk
            self.save_landmarks_data_on_disk(
                df=self.df_landmarks_prep,
                landmarks_columns_indices=self.landmarks_columns_indices,
                output_path=self.preprocessed_dataset_path,
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
        )
        self.subject_ids = {sample["subject_id"] for sample in self.samples}
        self.num_labels = len(self.poses_dict)
        self.num_landmarks = np.load(
            self.samples[0]["landmarks"]
        ).size

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
                T.CenterCrop(size=self.img_size),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

        # mode-related attributes
        self.return_landmarks = True
        self.return_images = True

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_raw_landmarks(df_landmarks_train_path=None, df_landmarks_test_path=None):
        assert (
            df_landmarks_train_path is not None or df_landmarks_test_path is not None
        ), "at least one of df_landmarks_train_path or df_landmarks_test_path must be provided"
        dfs = []
        if df_landmarks_train_path is not None:
            dfs.append(
                pl.read_csv(df_landmarks_train_path).to_pandas(
                    use_pyarrow_extension_array=False
                )
            )
            dfs[-1]["split"] = "train"
        if df_landmarks_test_path is not None:
            dfs.append(
                pl.read_csv(df_landmarks_test_path).to_pandas(
                    use_pyarrow_extension_array=False
                )
            )
            dfs[-1]["split"] = "test"
        # eventually concatenates the two dataframes
        if len(dfs) == 1:
            df_landmarks = dfs[0]
        if len(dfs) >= 2:
            # concatenate the two dataframes. the 'device' column keeps track of which landmarks come from which device
            df_landmarks = pd.concat(dfs)
        df_landmarks = df_landmarks.rename(
            columns={
                "user": "subject_id",
                "symbol": "pose",
                "frame": "frame_id",
            }
        )
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
    def preprocess_landmarks(
        df, landmarks_columns_indices, normalize_landmarks: bool = True
    ):
        subject_ids, poses = [
            df[col].unique().tolist() for col in ["subject_id", "pose"]
        ]
        df_values = df.values

        for subject_id, pose in tqdm(
            list(itertools.product(subject_ids, poses)),
            desc="Preprocessing data",
        ):
            # builds a mask of the rowss that match the current subject, hand, pose, and device
            mask = (df["subject_id"].values == subject_id) & (df["pose"].values == pose)
            mask_indices = np.where(mask)[0]
            # mask_indices = mask[mask].index

            data = df_values[np.ix_(mask_indices, landmarks_columns_indices)].astype(
                np.float32
            )
            if data.size == 0:
                print(f"skipping {subject_id}, {pose} because there are no landmarks")
                continue
            if normalize_landmarks:
                # standardize to zero mean and unit variance
                means, stds = data.mean(axis=0), data.std(axis=0)
                standardized_data = (data - means) / (stds + 1e-7)

                # normalize values between -1 and 1
                mins, maxs = standardized_data.min(axis=0), standardized_data.max(
                    axis=0
                )
                normalized_data = (
                    2 * (standardized_data - mins) / ((maxs - mins) - 1 + 1e-7)
                )
                assert not np.isnan(
                    normalized_data
                ).any(), f"there are nans in {subject_id}, {pose}"

                # assign the new data
                # df.iloc[mask_indices, landmarks_columns_indices] = normalized_data
                df_values[np.ix_(mask_indices, landmarks_columns_indices)] = (
                    normalized_data
                )

        df_prep = pd.DataFrame(df_values, columns=df.columns)

        # # splits the df into horizontal and vertical ones, based on the device used
        # df_horizontal = df_prep[df_prep["device"] == "Horizontal"]
        # df_vertical = df_prep[df_prep["device"] == "Vertical"]
        # assert (
        #     df_horizontal.shape == df_vertical.shape
        # ), f"{df_horizontal.shape} != {df_vertical.shape}"

        # # merge the two dataframes in a single one
        # id_cols = ["frame_id", "subject_id", "which_hand", "pose"]
        # df = pd.merge(
        #     left=df_horizontal,
        #     right=df_vertical,
        #     how="inner",
        #     on=id_cols,
        #     suffixes=("_horizontal", "_vertical"),
        # )
        # assert (
        #     df.shape[0] == df_horizontal.shape[0]
        # ), f"{df.shape} != {df_horizontal.shape}"
        # assert all(
        #     [col in df.columns for col in id_cols]
        # ), f"{id_cols} not in {df.columns}"
        # df = df.reset_index().drop(
        #     columns=[
        #         col
        #         for col in df.columns
        #         if col.startswith("device") or col.startswith("pose_index")
        #     ]
        # )
        return df_prep

    @staticmethod
    def save_landmarks_data_on_disk(df, landmarks_columns_indices, output_path):
        landmarks_columns = [
            col
            for i_col, col in enumerate(df.columns)
            if i_col in landmarks_columns_indices
        ]
        for row in tqdm(
            pl.from_pandas(df).iter_rows(named=True),
            desc=f"Saving preprocessed landmarks to {output_path}",
            total=len(df),
        ):
            # parses all the metas
            frame_id, subject_id, pose, split = [
                str(row[col]) for col in ["frame_id", "subject_id", "pose", "split"]
            ]
            # retrieves the landmarks from the big dataframe
            arr_landmarks = np.asarray(
                [float(row[col]) for col in landmarks_columns]
            ).astype(np.float32)
            # saves the landmarks to disk
            path = join(output_path, "pose", subject_id, f"{split}_pose", pose)
            if not isdir(path):
                os.makedirs(path)
            np.save(
                join(path, f"{frame_id}.npy"),
                arr_landmarks,
            )

    @staticmethod
    def parse_samples(dataset_path, preprocessed_landmarks_path, poses_dict):
        samples = []
        subject_ids = [f for f in listdir(join(preprocessed_landmarks_path, "pose"))]
        for subject_id in subject_ids:
            for split in [
                s.split("_")[0]
                for s in listdir(join(preprocessed_landmarks_path, "pose", subject_id))
            ]:
                for pose in listdir(
                    join(
                        preprocessed_landmarks_path, "pose", subject_id, f"{split}_pose"
                    )
                ):
                    sample = {
                        "subject_id": subject_id,
                        "pose": pose,
                        "split": split,
                        "label": poses_dict[pose],
                    }
                    for frame_id in [
                        s.split(".")[0]
                        for s in listdir(
                            join(
                                preprocessed_landmarks_path,
                                "pose",
                                subject_id,
                                f"{split}_pose",
                                pose,
                            )
                        )
                        # if s.endswith("_l.npy")
                    ]:
                        sample["landmarks"] = join(
                            preprocessed_landmarks_path,
                            "pose",
                            subject_id,
                            f"{split}_pose",
                            pose,
                            f"{frame_id}.npy",
                        )
                        assert exists(
                            sample["landmarks"]
                        ), f"{sample['landmarks']} does not exist"
                        sample["image"] = join(
                            dataset_path,
                            "near-infrared",
                            subject_id,
                            f"{split}_pose",
                            pose,
                            f"{frame_id}.png",
                        )
                        assert exists(
                            sample["image"]
                        ), f"{sample['image']} does not exist"
                        samples.append(deepcopy(sample))
        # splits = [s.split("_")[0] for s in listdir(join(preprocessed_landmarks_path, "pose", subject_ids[0]))]
        # poses = listdir(join(preprocessed_landmarks_path, "pose", subject_ids[0], f"{splits[0]}_pose"))
        # frame_ids = [
        #     filename.split("_")[0].zfill(3)
        #     for filename in listdir(
        #         join(
        #             dataset_path,
        #             "near-infrared",
        #             subject_ids[0],
        #             poses[0],
        #         )
        #     )
        # ]
        # for subject_id, pose in tqdm(
        #     list(itertools.product(subject_ids, poses, frame_ids)),
        #     desc=f"Parsing samples",
        # ):
        #     if hand == "Left_Hand" and images_needed == "right":
        #         continue
        #     elif hand == "Right_Hand" and images_needed == "left":
        #         continue
        #     # parses the sample
        #     sample = {
        #         "subject_id": subject_id,
        #         "hand": hand,
        #         "pose": pose,
        #     }
        #     if poses_dict is not None:
        #         if pose not in poses_dict:
        #             raise BaseException(
        #                 f"Unrecognized pose '{pose}'. Poses are: {list(poses_dict.keys())}"
        #             )
        #         sample["label"] = poses_dict[pose]
        #     for device in ["Horizontal", "Vertical"]:
        #         # parses the landmarks
        #         sample[f"landmarks_{device.lower()}"] = join(
        #             preprocessed_landmarks_path,
        #             subject_id,
        #             hand,
        #             pose,
        #             device,
        #             "landmarks",
        #             f"{frame_id}.npy",
        #         )
        #         assert exists(
        #             sample[f"landmarks_{device.lower()}"]
        #         ), f"{sample[f'landmarks_{device.lower()}']} does not exist"
        #         # parses the images
        #         for direction in ["left", "right"]:
        #             sample[f"image_{device.lower()}_{direction}"] = join(
        #                 dataset_path,
        #                 subject_id,
        #                 hand,
        #                 pose,
        #                 device,
        #                 "images",
        #                 f"{frame_id}_{direction}.bmp",
        #             )
        #             assert exists(
        #                 sample[f"image_{device.lower()}_{direction}"]
        #             ), f"{sample[f'image_{device.lower()}_{direction}']} does not exist"
        #     samples.append(sample)
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
        return_images=None,
        return_landmarks=None,
        **kwargs
    ):
        if return_images is not None:
            assert isinstance(return_images, bool)
            self.return_images = return_images
        if return_landmarks is not None:
            assert isinstance(return_landmarks, bool)
            self.return_landmarks = return_landmarks

    @staticmethod
    def _get_subject_ids() -> List[str]:
        return sorted([f"user_{str(i).zfill(2)}" for i in range(1, 25 + 1)])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        outs = {"label": sample["label"]}
        for key in sample:
            if key == "image":
                # loads the image
                outs["image"] = self.images_transforms(Image.open(sample[key]))
            elif key == "landmarks":
                outs[key] = torch.from_numpy(
                    np.load(sample[key], allow_pickle=True)
                ).float()
        # returns the outputs
        return outs


if __name__ == "__main__":
    dataset = MultiModalHandGestureDatasetForHandGestureRecognition(
        dataset_path="../../datasets/mmhgdhgr"
    )

    # tests
    for (
        return_images,
        return_landmarks,
    ) in tqdm(
        list(
            itertools.product(
                [True, False], [True, False],
            )
        ),
        desc="trying all modes combinations",
    ):
        dataset.set_mode(
            return_images=return_images,
            return_landmarks=return_landmarks,
        )
        batch = dataset[
            0
        ]  # keys are ['label', 'landmarks', 'image']
        if return_images:
            assert "image" in batch, f"keys are {batch.keys()}"
        if return_landmarks:
            assert "landmarks" in batch, f"keys are {batch.keys()}"
