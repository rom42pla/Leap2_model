import os
import json
import torch
import pandas as pd

from src.utils.utils import get_all_poses, get_hand_images, clean_columns


# Create a function to create the dataset for a specific user given the user_id, the root directory of the dataset and the data directory where the dataset will be saved
def create_user_dataset(user_id: str, root_dir: str, data_dir: str, side: str = 'left', save: bool = True):
    '''
    Function to create the dataset for a specific user given the user_id, the root directory of the dataset and the data directory where the dataset will be saved
    params:
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - root_dir: str: root directory of the dataset
    - data_dir: str: data directory where the dataset will be saved
    - side: str: side of the hand to be used, must be 'left' or 'right'
    - save: bool: flag to save the dataset to a csv file
    '''
    train_dir = os.path.join(data_dir, 'train_' + user_id)
    os.makedirs(train_dir, exist_ok=True)
    # Read the poses dictionary
    if os.path.exists(os.path.join('data', 'poses_dict.txt')):
        with open(os.path.join('data', 'poses_dict.txt'), 'r') as f:
            poses_dict = json.load(f)
    else:
        raise FileNotFoundError('The poses_dict.txt file was not found in the data directory')
    for hand in os.listdir(os.path.join(root_dir, user_id)):
        hand_dir = os.path.join(train_dir, hand)
        os.makedirs(hand_dir, exist_ok=True)
        for pose in get_all_poses(root_dir, user_id, hand):
            pose_dir = os.path.join(hand_dir, pose)
            os.makedirs(pose_dir, exist_ok=True)
            for orientation in os.listdir(os.path.join(root_dir, user_id, hand, pose)):
                orientation_dir = os.path.join(pose_dir, orientation)
                df = pd.read_csv(os.path.join(root_dir, user_id, hand, pose, orientation, 'hand_properties.csv'))
                images = get_hand_images(os.path.join(root_dir, user_id, hand, pose, orientation, 'images'), side=side)
                # Add the images path to the dataframe in a new column called image_path checking that the path exists
                df['image_path'] = [os.path.join(root_dir, user_id, hand, pose, orientation, 'images', img) for img in images if os.path.exists(os.path.join(root_dir, user_id, hand, pose, orientation, 'images', img))]
                # Add the pose index to the dataframe in a new column called pose_index using the poses_dict and the value in pose column
                df['pose_index'] = df['pose'].map(poses_dict)
                # Save the dataframe to a csv file in the pose directory adding the orientation and the side of the camera used to capture the images
                if save:
                    df.to_csv(os.path.join(pose_dir, f'hand_properties_{orientation}_{side}.csv'), index=False)


def create_viewpoint_dataset(train_users_path: str, dataframes_path: str = None, save: bool = False):
    '''
    Function to create the dataset for all the users in the train_users_path
    params:
    - train_users_path: str: path to the directory where the train users are stored
    - save: bool: flag to save the dataset to a csv file
    '''
    if dataframes_path is not None:
        dfs_horizontal_viewpoints = pd.read_csv(os.path.join(dataframes_path, 'hand_properties_horizontal_cleaned.csv'))
        dfs_vertical_viewpoints = pd.read_csv(os.path.join(dataframes_path, 'hand_properties_vertical_cleaned.csv'))
    else:
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
                        ####### THIS PART IS NOT NEEDED SINCE THE DATA HAS BEEN CLEANED IN A PREPROCESSING STAGE OTHERWISE USE IT ########
                        # Since train_007, train_018 and train_019 have been created using encoding ISO-8859-1 we need to read them using this encoding
                        # if train_user in ['train_007', 'train_019', 'train_018']:
                            # hand_properties_df = pd.read_csv(hand_properties_path)
                            # Clean the dataframe columns
                            # hand_properties_df = clean_columns(hand_properties_df, log_file_path)
                            # Ensure that all elements in the columns from the 5th to the last two are float
                            # hand_properties_df.iloc[:, 5:-2] = hand_properties_df.iloc[:, 5:-2].astype(float)
                        # else:
                        hand_properties_df = pd.read_csv(hand_properties_path)
                        # Add the hand properties to the list of dataframes depending on the viewpoint
                        if 'Horizontal' in viewpoint:
                            dfs_horizontal_views.append(hand_properties_df)
                        elif 'Vertical' in viewpoint:
                            dfs_vertical_views.append(hand_properties_df)
            dfs_horizontal_viewpoints = pd.concat(dfs_horizontal_views)
            dfs_vertical_viewpoints = pd.concat(dfs_vertical_views)
        if save:
            dfs_horizontal_viewpoints.to_csv(os.path.join(train_users_path, '..', 'hand_properties_horizontal.csv'), index=False)
            dfs_vertical_viewpoints.to_csv(os.path.join(train_users_path, '..', 'hand_properties_vertical.csv'), index=False)
    return dfs_horizontal_viewpoints, dfs_vertical_viewpoints


def save_predictions_to_csv(tensors_list, save_path, save=False):
    """
    Given a list of tensors with shape [batch_size, num_classes], computes the predicted class for each sample 
    (using argmax over the num_classes dimension) and saves the results as a CSV at the given save_path.
    
    Args:
        tensors_list (list of torch.Tensor): List of tensors with shape [batch_size, num_classes]
        save_path (str): The path where the final DataFrame should be saved.
        
    Returns:
        final_predicted_classes (numpy.ndarray): Array of predicted classes (shape: [batch size, 1])
    """
    predicted_classes = []

    # Loop over the list of tensors
    for tensor in tensors_list:
        # Compute the argmax along the num_classes dimension (dim=1)
        argmax_result = torch.argmax(tensor, dim=1)
        # Append the result to the list
        predicted_classes.append(argmax_result)

    # Concatenate all argmax results into a single vector
    final_predicted_classes = torch.cat(predicted_classes, dim=0)

    # Convert to numpy and reshape into a column vector
    final_predicted_classes = final_predicted_classes.cpu().numpy().reshape(-1, 1)

    # Create a DataFrame with the predicted class column
    df = pd.DataFrame(final_predicted_classes, columns=['PredictedClass'])
    if save:
        # Save the DataFrame to the specified path
        df.to_csv(save_path, index=False)
    
    print(f"Predictions saved to {save_path}")

    return final_predicted_classes