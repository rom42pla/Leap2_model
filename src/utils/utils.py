import os
import re
import string
import random

import pandas as pd

def get_all_poses(root_dir, user_id, hand):
    '''
    Function to get all the available poses for a specific hand
    '''
    hand_path = os.path.join(root_dir, user_id, hand)
    poses = os.listdir(hand_path)
    return poses


def get_dataset_length(root_dir, user_id, hand):
    '''
    Function to get the length of the dataset for a specific hand
    '''
    hand_path = os.path.join(root_dir, user_id, hand)
    poses = os.listdir(hand_path)
    # Select a random pose to get the length of the dataset
    random_pose = random.choice(poses)
    # Randomly select the orientation
    orientation = random.choice(['Horizontal', 'Vertical'])
    # Get the path to the pose
    pose_path = os.path.join(hand_path, random_pose, orientation, 'images')
    # Get the length of the dataset but dividing by 2 since we have the left and right camera viewpoints and check that it is divisible by 2
    if len(os.listdir(pose_path)) % 2 != 0:
        raise ValueError('The dataset is not divisible by 2 check dataset integrity for user: {}, hand: {}, pose: {} and orientation: {}'.format(user_id, hand, random_pose, orientation))
    length = len(os.listdir(pose_path)) // 2
    return length


def get_user_hand_images_paths(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the left hand from the horizontal camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - left_hand_horizontal_path: str: path to the hand_properties.csv file for the left hand from the horizontal camera viewpoint
    - rigth_hand_horizontal_path: str: path to the hand_properties.csv file for the right hand from the horizontal camera viewpoint
    - left_hand_vertical_path: str: path to the hand_properties.csv file for the left hand from the vertical camera viewpoint
    - rigth_hand_vertical_path: str: path to the hand_properties.csv file for the right hand from the vertical camera viewpoint
    '''
    left_hand_horizontal_image_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Horizontal', 'images')
    rigth_hand_horizontal_image_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Horizontal', 'images')
    left_hand_vertical_image_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Vertical', 'images')
    rigth_hand_vertical_image_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Vertical', 'images')
    return left_hand_horizontal_image_path, rigth_hand_horizontal_image_path, left_hand_vertical_image_path, rigth_hand_vertical_image_path
    

def get_user_hand_properties_paths(root_dir: str, user_id: str, hand_pose: str):
    '''
    Function to read the hand_properties.csv file for the left hand from the horizontal camera viewpoint
    params:
    - root_dir: str: root directory of the dataset
    - user_id: str: user id, must be in the format '00X' being X the id of the user
    - hand_pose: str: hand pose, must be one of the 17 available hand poses
    return:
    - left_hand_horizontal_path: str: path to the hand_properties.csv file for the left hand from the horizontal camera viewpoint
    - rigth_hand_horizontal_path: str: path to the hand_properties.csv file for the right hand from the horizontal camera viewpoint
    - left_hand_vertical_path: str: path to the hand_properties.csv file for the left hand from the vertical camera viewpoint
    - rigth_hand_vertical_path: str: path to the hand_properties.csv file for the right hand from the vertical camera viewpoint
    '''
    left_hand_horizontal_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Horizontal', 'hand_properties.csv')
    rigth_hand_horizontal_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Horizontal', 'hand_properties.csv')
    left_hand_vertical_path = os.path.join(root_dir, user_id, 'Left_Hand', hand_pose, 'Vertical', 'hand_properties.csv')
    rigth_hand_vertical_path = os.path.join(root_dir, user_id, 'Right_Hand', hand_pose, 'Vertical', 'hand_properties.csv')
    return left_hand_horizontal_path, rigth_hand_horizontal_path, left_hand_vertical_path, rigth_hand_vertical_path



# Function to list all the images in a directory and take only those ending in _left. Also check that the final list length is 1000
def get_hand_images(image_path: str, side: str = 'left'):
    '''
    Function to list all the images in a directory and take only those ending in _left. Also check that the final list length is 1000
    params:
    - image_path: str: path to the directory containing the images
    return:
    - left_hand_images: list: list of images ending in _left
    '''
    if side == 'left':
        left_hand_images = [img for img in os.listdir(image_path) if img.endswith('_left.bmp')]
        # Sort the list by their image number being in the format 'XXXX_left.bmp'
        left_hand_images = sorted(left_hand_images, key=lambda x: int(x.split('_')[0]))
        assert len(left_hand_images) == 1000
        return left_hand_images
    elif side == 'right':
        right_hand_images = [img for img in os.listdir(image_path) if img.endswith('_right.bmp')]
        # Sort the list by their image number being in the format 'XXXX_right.bmp'
        right_hand_images = sorted(right_hand_images, key=lambda x: int(x.split('_')[0]))
        assert len(right_hand_images) == 1000
        return right_hand_images
    return left_hand_images


def clean_columns(df: pd.DataFrame, log_file_path: str):
    '''
    Function to clean the columns in the DataFrame by removing non-printable characters, 
    non-digit characters, and strange characters (like Ã, Â).
    
    Params:
    - df: pd.DataFrame: DataFrame to be cleaned
    - log_file_path: str: path to the log file to save the original and cleaned values
    
    Return:
    - df: pd.DataFrame: cleaned DataFrame
    '''
    # Open a log file to record original and cleaned values if the file does not exist create it
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write('column, frame_id, which_hand, pose, original_value, cleaned_value\n')
        
    # Loop over the columns from 5th to -2 (adjust this range as needed)
    for col in df.columns[5:-2]:
        # Iterate over each value in the column
        for idx, val in df[col].items():
            # Retrieve frame_id, which_hand, and pose for logging
            frame_id = df.at[idx, 'frame_id']
            which_hand = df.at[idx, 'which_hand']
            pose = df.at[idx, 'pose']
            original_val = val  # Save the original value for logging
            
            # Clean the value and cast it to float
            final_val = clean_and_cast_to_float(val)
                
            # Log the cleaned value (if it changed)
            if original_val != final_val:
                with open(log_file_path, 'a') as f:
                    f.write(f"{col}, {frame_id}, {which_hand}, {pose}, {original_val}, {str(final_val)}\n")
                
                # Update the DataFrame with the cleaned value
                df.at[idx, col] = final_val
            else:
                # If the value is not a string, leave it unchanged
                df.at[idx, col] = val

    return df

def clean_and_cast_to_float(value):
    '''
    Function to clean a value and cast it to float. The cleaning process removes non-printable characters,
    non-digit characters, and strange characters (like Ã, Â).

    Params:
    - value: str or float: value to be cleaned and cast to float

    Return:
    - float: cleaned value cast to float
    '''
    if isinstance(value, (int, float)):
        return value
    # Remove non-printable characters using a regex, allow only digits, '.' and '-'
    printable = set(string.printable)
    clean_value = ''.join([ch for ch in str(value) if ch in printable])

    # Using regex to keep only digits, '.', and '-' at the start for negative numbers
    clean_value = re.sub(r'[^\d.-]', '', clean_value)

    # Check for proper formatting of the negative sign (at the beginning) and decimal point
    if clean_value.count('-') > 1 or (clean_value.count('-') == 1 and clean_value.index('-') != 0):
        clean_value = clean_value.replace('-', '')  # Remove all invalid '-' signs

    # Return as float
    try:
        return float(clean_value)
    except ValueError:
        raise ValueError(f"Cannot convert {value} to float after cleaning.")