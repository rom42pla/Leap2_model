import os
import cv2

import pandas as pd

from src.dataloaders.hand_pose_dataset import HandPoseDataset


def read_left_hand_horizontal_csv(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the left hand from the horizontal camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - df: pd.DataFrame: dataframe with the hand properties
    '''
    file_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Horizontal', 'hand_properties.csv')
    df = pd.read_csv(file_path)
    return df


def read_left_hand_vertical_csv(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the left hand from the vertical camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - df: pd.DataFrame: dataframe with the hand properties
    '''
    file_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Vertical', 'hand_properties.csv')
    df = pd.read_csv(file_path)
    return df


def read_right_hand_horizontal_csv(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the right hand from the horizontal camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - df: pd.DataFrame: dataframe with the hand properties
    '''
    file_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Horizontal', 'hand_properties.csv')
    df = pd.read_csv(file_path)
    return df


def read_right_hand_vertical_csv(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the right hand from the vertical camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - df: pd.DataFrame: dataframe with the hand properties
    '''
    file_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Vertical', 'hand_properties.csv')
    df = pd.read_csv(file_path)
    return df


def read_pose_image_by_pose_id(root_dir:str, user_id: str, hand: str, hand_pose: str, orientation: str, viewpoint: str, pose_id: str):
    '''
    Function to read the image of a specific pose_id for a specific hand and orientation
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand: str: hand, must be one of 'Left_Hand' or 'Right_Hand'
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    - orientation: str: orientation, must be one of 'left' or 'right'
    - viewpoint: str: viewpoint, must be one of 'horizontal' or 'vertical'
    - pose_id: str: pose id of the image to read
    return:
    - img: np.array: image of the pose
    '''
    if orientation.capitalize() not in ['Horizontal', 'Vertical']:
        raise ValueError("Orientation must be one of 'horizontal' or 'vertical'")
    if viewpoint.lower() not in ['left', 'right']:
        raise ValueError("Orientation must be one of 'left' or 'right'")
    if hand not in ['Left_Hand', 'Right_Hand']:
        raise ValueError("Hand must be one of 'Left_Hand' or 'Right_Hand'")
    file_path = os.path.join(root_dir, user_id, hand, hand_pose, orientation.capitalize(), 'images', f'{pose_id}_{viewpoint.lower()}.bmp')
    img = cv2.imread(file_path)
    return img


def read_viewpoint_dataset(train_users_path: str):
    '''
    Function to read the viewpoint dataset for all the users in the train_users_path
    params:
    - train_users_path: str: path to the directory where the train users are stored
    '''
    train_users = os.listdir(train_users_path)
    dfs_vertical_views = []
    dfs_horizontal_views = []
    for train_user in train_users:
        # List the hands in the user path
        train_hands_path = os.path.join(train_users_path, train_user)
        train_hands = os.listdir(train_hands_path)
        for hand in train_hands:
            # List the poses in the hand path
            train_poses_path = os.path.join(train_hands_path, hand)
            train_poses = os.listdir(train_poses_path)
            # Add the hand properties to the list of dataframes
            for train_pose in train_poses:
                # List the viewpoints in the pose path
                train_viewpoints_path = os.path.join(train_poses_path, train_pose)
                train_viewpoints = os.listdir(train_viewpoints_path)
                for viewpoint in train_viewpoints:
                    # Read the hand properties csv file
                    hand_properties_path = os.path.join(train_viewpoints_path, viewpoint)
                    hand_properties_df = pd.read_csv(hand_properties_path)
                    # Add the hand properties to the list of dataframes depending on the viewpoint
                    if 'Horizontal' in viewpoint:
                        dfs_horizontal_views.append(hand_properties_df)
                    elif 'Vertical' in viewpoint:
                        dfs_vertical_views.append(hand_properties_df)
        dfs_horizontal_viewpoints = pd.concat(dfs_horizontal_views)
        dfs_vertical_viewpoints = pd.concat(dfs_vertical_views)
    return dfs_horizontal_viewpoints, dfs_vertical_viewpoints


# Function to create datasets
def create_datasets(dfs_horizontal_viewpoints, 
                    dfs_vertical_viewpoints, 
                    val_subject_id, 
                    train_subject_id=None,
                    images_path=None):
    '''
    Creates train and validation datasets for the given subject IDs from the provided dataframes.
    
    Args:
        dfs_horizontal_viewpoints: DataFrame containing horizontal viewpoint data.
        dfs_vertical_viewpoints: DataFrame containing vertical viewpoint data.
        val_subject_id: The subject ID used to create the validation dataset.
        train_subject_id: The subject ID used to create the training dataset (default is None).
        images_path: Path to the images (default is None).
        
    Returns:
        train_dataset: The dataset used for training (HandPoseDataset instance).
        val_dataset: The dataset used for validation (HandPoseDataset instance).
    '''
    if train_subject_id is not None:
        # More like debug mode
        dfs_horizontal_viewpoints_train = dfs_horizontal_viewpoints[dfs_horizontal_viewpoints['subject_id'] == train_subject_id]
        dfs_horizontal_viewpoints_val = dfs_horizontal_viewpoints[dfs_horizontal_viewpoints['subject_id'] == val_subject_id]
        dfs_vertical_viewpoints_train = dfs_vertical_viewpoints[dfs_vertical_viewpoints['subject_id'] == train_subject_id]
        dfs_vertical_viewpoints_val = dfs_vertical_viewpoints[dfs_vertical_viewpoints['subject_id'] == val_subject_id]
    else:
        # Perform LOSO-CV
        # Take the viewpoitns of the validation subject_id for validation and the rest for training
        dfs_horizontal_viewpoints_train = dfs_horizontal_viewpoints[dfs_horizontal_viewpoints['subject_id'] != val_subject_id]
        dfs_horizontal_viewpoints_val = dfs_horizontal_viewpoints[dfs_horizontal_viewpoints['subject_id'] == val_subject_id]
        dfs_vertical_viewpoints_train = dfs_vertical_viewpoints[dfs_vertical_viewpoints['subject_id'] != val_subject_id]
        dfs_vertical_viewpoints_val = dfs_vertical_viewpoints[dfs_vertical_viewpoints['subject_id'] == val_subject_id]

    if images_path is None:
        raise ValueError("The path to the images is required")
    
    train_dataset = HandPoseDataset(dfs_horizontal_viewpoints_train, 
                                    dfs_vertical_viewpoints_train,
                                    images_path)
    val_dataset = HandPoseDataset(dfs_horizontal_viewpoints_val, 
                                  dfs_vertical_viewpoints_val,
                                  images_path)
    
    return train_dataset, val_dataset

