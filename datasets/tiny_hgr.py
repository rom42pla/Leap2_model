from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pprint import pprint
import itertools
import os
from os import makedirs, listdir
from os.path import join, isdir, exists, basename
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


class TinyHandGestureRecognitionDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        preprocessed_landmarks_path="_tiny_hgr_preprocessed",
        img_size=1080,
        normalize_landmarks=True,
    ):
        # assert isdir(dataset_path)
        self.dataset_path = dataset_path
        self.normalize_landmarks = normalize_landmarks

        self.poses_dict = {
            "fist": 0,
            "l": 1,
            "I": 1,  # there's an error for tomas_2
            "ok": 2,
            "palm": 3,
            "pointer": 4,
            "thumb down": 5,
            "thumb up": 6,
        }

        # preprocessing for the images
        assert isinstance(
            img_size, int
        ), f"img_size must be an int, got {img_size} ({type(img_size)})"
        assert img_size >= 1, f"img_size must be >= 1, got {img_size}"
        self.img_channels = 3
        self.img_size = img_size
        self.img_shape = (1, self.img_size, self.img_size)
        self.images_transforms = T.Compose(
            [
                T.Resize(size=self.img_size),
                T.CenterCrop(size=self.img_size),
            ]
        )

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
            # eventually delete the preprocessed dataset if the normalization or img_size is different
            if (
                info["normalized_landmarks"] != self.normalize_landmarks
                or info["img_size"] != self.img_size
            ):
                shutil.rmtree(self.preprocessed_dataset_path)
                print(
                    "rebuilding the preprocessed dataset because of different normalization"
                )
        if not exists(self.preprocessed_dataset_path):
            # creates the preprocessed dataset folder
            makedirs(self.preprocessed_dataset_path)
            # parses the cleaned landmarks
            df_landmarks_raw = self.load_raw_landmarks(
                df_landmarks_path=join(
                    self.dataset_path,
                    "coordinates_frames_labelled.csv",
                ),
            )
            # retrieves the columns indices with actual values from landmarks
            self.landmarks_columns_indices = list(range(42))
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
            # save the preprocessed images on disk
            self.save_images_on_disk(
                df=self.df_landmarks_prep,
                dataset_path=self.dataset_path,
                output_path=self.preprocessed_dataset_path,
                images_transforms=self.images_transforms,
            )
            # saves the info about the dataset
            info = {
                "img_size": self.img_size,
                "normalized_landmarks": self.normalize_landmarks,
            }
            with open(join(self.preprocessed_dataset_path, "info.yaml"), "w") as fp:
                yaml.dump(info, fp, default_flow_style=False)

        # loads the infos for each sample
        self.samples = self.parse_samples(
            dataset_path=self.dataset_path,
            preprocessed_dataset_path=self.preprocessed_dataset_path,
            poses_dict=self.poses_dict,
        )
        self.subject_ids = {sample["subject_id"] for sample in self.samples}
        self.num_labels = len(self.poses_dict)
        self.num_landmarks = np.load(self.samples[0]["landmarks"]).size

        # mode-related attributes
        self.return_landmarks = True
        self.return_images = True

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_raw_landmarks(df_landmarks_path):
        assert df_landmarks_path is not None
        df_landmarks = pd.read_csv(df_landmarks_path)
        df_landmarks = df_landmarks.rename(
            columns={
                "user": "subject_id",
                "symbol": "pose",
                "path": "frame_id",
            }
        )
        df_landmarks["frame_id"] = df_landmarks["frame_id"].apply(
            lambda x: basename(x).split("\\")[-1]
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
            # builds a mask of the rows that match the current subject, hand, pose, and device
            mask = (df["subject_id"].values == subject_id) & (df["pose"].values == pose)
            mask_indices = np.where(mask)[0]
            # mask_indices = mask[mask].index

            data = df_values[np.ix_(mask_indices, landmarks_columns_indices)].astype(
                np.float32
            )
            if data.size == 0:
                # print(f"skipping {subject_id}, {pose} because there are no landmarks")
                continue
            if normalize_landmarks:
                normalized_data = np.copy(data).reshape(-1, 21, 2)

                for i_sample in range(normalized_data.shape[0]):
                    sample = normalized_data[i_sample]
                    for i in range(sample.shape[0]):
                        if (sample[i, 0] + sample[i, 1]) != 0:
                            x_center = sample[i, 0]
                            y_center = sample[i, 1]
                            for k in range(2, sample.shape[1], 2):
                                sample[i, k] = sample[i, k] - x_center
                                sample[i, k + 1] = sample[i, k + 1] - y_center
                    normalized_data[i_sample] = sample

                normalized_data = normalized_data.reshape(-1, data.shape[1])

                # # standardize to zero mean and unit variance
                # means, stds = normalized_data.mean(axis=0), normalized_data.std(axis=0)
                # normalized_data = (normalized_data - means) / (stds + 1e-7)

                # # normalize values between -1 and 1
                # mins, maxs = normalized_data.min(axis=0), normalized_data.max(
                #     axis=0
                # )
                # normalized_data = (
                #     2 * (normalized_data - mins) / ((maxs - mins) - 1 + 1e-7)
                # )

                assert not np.isnan(
                    normalized_data
                ).any(), f"there are nans in {subject_id}, {pose}"

                # assign the new data
                # df.iloc[mask_indices, landmarks_columns_indices] = normalized_data
                df_values[np.ix_(mask_indices, landmarks_columns_indices)] = (
                    normalized_data
                )

        df_prep = pd.DataFrame(df_values, columns=df.columns)

        return df_prep

    @staticmethod
    def save_landmarks_data_on_disk(df, landmarks_columns_indices, output_path):
        landmarks_columns = [
            col
            for i_col, col in enumerate(df.columns)
            if i_col in landmarks_columns_indices
        ]
        for _, row in tqdm(
            df.iterrows(),
            desc=f"Saving preprocessed landmarks to {output_path}",
            total=len(df),
        ):
            # parses all the metas
            frame_id, subject_id, pose = [
                str(row[col]) for col in ["frame_id", "subject_id", "pose"]
            ]
            # retrieves the landmarks from the big dataframe
            arr_landmarks = np.asarray(
                [float(row[col]) for col in landmarks_columns]
            ).astype(np.float32)
            # saves the landmarks to disk
            path = join(output_path, subject_id, pose, "landmarks")
            if not isdir(path):
                os.makedirs(path)
            np.save(
                join(path, f"{frame_id}.npy"),
                arr_landmarks,
            )

    # @staticmethod
    # def save_images_on_disk(df, dataset_path, output_path, images_transforms):
    #     for _, row in tqdm(
    #         df.iterrows(),
    #         desc=f"Saving preprocessed images to {output_path}",
    #         total=len(df),
    #     ):
    #         # parses all the metas
    #         frame_id, subject_id, pose = [
    #             str(row[col]) for col in ["frame_id", "subject_id", "pose"]
    #         ]
    #         # loads the image from the big dataframe
    #         img = images_transforms(Image.open(join(dataset_path, subject_id, pose, frame_id)))
    #         # saves the landmarks to disk
    #         path = join(output_path, subject_id, pose, "images")
    #         if not isdir(path):
    #             os.makedirs(path)
    #         # saves the image
    #         img.save(
    #             join(path, f"{frame_id}.jpg"),
    #         )

    @staticmethod
    def save_images_on_disk(df, dataset_path, output_path, images_transforms):
        global process_row

        def process_row(row_data, dataset_path, output_path, images_transforms):
            index, row = row_data
            frame_id, subject_id, pose = [
                str(row[col]) for col in ["frame_id", "subject_id", "pose"]
            ]
            try:
                img_path = join(dataset_path, subject_id, pose, frame_id)
                img = images_transforms(Image.open(img_path))

                save_dir = join(output_path, subject_id, pose, "images")
                os.makedirs(save_dir, exist_ok=True)

                img.save(join(save_dir, f"{frame_id}.png"))
            except Exception as e:
                print(f"Error processing {frame_id}: {e}")

        # Use a pool of workers to process rows in parallel
        with Pool(processes=os.cpu_count()) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(
                            process_row,
                            dataset_path=dataset_path,
                            output_path=output_path,
                            images_transforms=images_transforms,
                        ),
                        df.iterrows(),
                    ),
                    total=len(df),
                    desc=f"Saving preprocessed images to {output_path}",
                )
            )

    @staticmethod
    def parse_samples(dataset_path, preprocessed_dataset_path, poses_dict):
        samples = []
        subject_ids = [
            f
            for f in listdir(preprocessed_dataset_path)
            if isdir(join(preprocessed_dataset_path, f))
        ]
        for subject_id in subject_ids:
            for pose in listdir(join(preprocessed_dataset_path, subject_id)):
                sample = {
                    "subject_id": subject_id,
                    "pose": pose,
                    "label": poses_dict[pose],
                }

                for frame_id in [
                    ".".join(s.split(".")[:-1])
                    for s in listdir(
                        join(
                            preprocessed_dataset_path,
                            subject_id,
                            pose,
                            "images",
                        )
                    )
                ]:
                    sample["landmarks"] = join(
                        preprocessed_dataset_path,
                        subject_id,
                        pose,
                        "landmarks",
                        f"{frame_id}.npy",
                    )
                    assert exists(
                        sample["landmarks"]
                    ), f"{sample['landmarks']} does not exist"
                    sample["image"] = join(
                        preprocessed_dataset_path,
                        subject_id,
                        pose,
                        "images",
                        f"{frame_id}.png",
                    )
                    assert exists(sample["image"]), f"{sample['image']} does not exist"
                    samples.append(deepcopy(sample))
        return samples

    def get_indices_per_subject(self) -> Dict[str, List[int]]:
        indices_per_subject = {}
        for i_sample, sample in enumerate(self.samples):
            subject_id = sample["subject_id"]
            if subject_id not in indices_per_subject:
                indices_per_subject[subject_id] = []
            indices_per_subject[subject_id].append(i_sample)
        return indices_per_subject

    def set_mode(self, return_images=None, return_landmarks=None, **kwargs):
        if return_images is not None:
            assert isinstance(return_images, bool)
            self.return_images = return_images
        if return_landmarks is not None:
            assert isinstance(return_landmarks, bool)
            self.return_landmarks = return_landmarks

    @staticmethod
    def _get_subject_ids() -> List[str]:
        return sorted(
            [
                "alberto",
                "federico",
                "esther",
                "cesar_2",
                "jesus",
                "lorena",
                "dani_2",
                "marta",
                "carlos_ca",
                "rafa",
                "arturo",
                "cesar",
                "mateo",
                "carlos_c",
                "carmen",
                "alfredo",
                "aishwary",
                "ana",
                "tomas",
                "cristina",
                "carlos_r",
                "lara",
                "vir",
                "dani",
                "raquel",
                "susana",
                "tomas_2",
                "javier",
                "marie",
                "richard",
                "pablo",
                "carlos_c_2",
                "fili",
                "ana_m",
                "pablo_2",
                "susana_2",
                "raul",
                "samira",
                "sergio",
                "jose_luis",
            ]
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        outs = {"label": sample["label"]}
        for key in sample:
            if key == "image":
                # loads the image
                outs["image"] = T.ToTensor()(
                    self.images_transforms(Image.open(sample[key]))
                )
            elif key == "landmarks":
                outs[key] = torch.from_numpy(
                    np.load(sample[key], allow_pickle=True)
                ).float()
        # returns the outputs
        return outs


if __name__ == "__main__":
    dataset = TinyHandGestureRecognitionDataset(
        dataset_path="../../datasets/tiny_hgr",
        normalize_landmarks=True,
        img_size=224,
    )
    # # plots the image
    # import matplotlib.pyplot as plt
    # plt.imshow(dataset[22123]["image"].permute(1, 2, 0).numpy())
    # plt.show()
    # tests
    for (
        return_images,
        return_landmarks,
    ) in tqdm(
        list(
            itertools.product(
                [True, False],
                [True, False],
            )
        ),
        desc="trying all modes combinations",
    ):
        dataset.set_mode(
            return_images=return_images,
            return_landmarks=return_landmarks,
        )
        batch = dataset[0]  # keys are ['label', 'landmarks', 'image']
        if return_images:
            assert "image" in batch, f"keys are {batch.keys()}"
        if return_landmarks:
            assert "landmarks" in batch, f"keys are {batch.keys()}"
