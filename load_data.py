
import os
import json
import pandas as pd
import argparse
import datetime

from src.utils.readers import create_datasets
from src.utils.writers import create_viewpoint_dataset

from src.dataloaders.hand_pose_dataset import collate_fn
from torch.utils.data import DataLoader

# Set the parallelism configuration to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description='Train a multimodal model')
    parser.add_argument('--img_path', type=str, default="/autofs/thau00a/home/sestebanro/thau01/Multiview_Leap2_Hand_Pose_Dataset/", help='Path to the images')
    parser.add_argument("--subject_id_val", type=int, default=None, help="Subject id to use as validation set")
    parser.add_argument("--subject_id_train", type=int, default=None, help="Subject id to use as training set")
    parser.add_argument("--dataframes_path", type=str, default=None, help="Path to the dataframes")
    parser.add_argument("--samples_per_pose", type=int, default=None, help="Number of samples per pose")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility')
    return parser.parse_args()

args = parse_args()

init_time = datetime.datetime.now()
group_name = datetime.datetime.now().strftime("%d-%b_%H:%M:%S")

# Get the absolute path of the current directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Read the dictionary of poses if it exists
poses_dict_path = os.path.join(root_dir, "data", "poses_dict.json")
if os.path.exists(poses_dict_path):
    poses_dict = json.load(open(poses_dict_path))
    # Invert the dictionary
    poses_dict = {v: k for k, v in poses_dict.items()}

train_users_path = os.path.join(root_dir, "data", "train_datasets")
    
# Create the dataset for all the users in the train_users_path
dfs_horizontal_viewpoints, dfs_vertical_viewpoints = create_viewpoint_dataset(train_users_path, 
                                                                              dataframes_path=args.dataframes_path, 
                                                                              save=True)

# Replace the nan values with the mean of the column
cols_with_nan_hori = dfs_horizontal_viewpoints.columns[dfs_horizontal_viewpoints.isna().any()]
cols_with_nan_vert = dfs_vertical_viewpoints.columns[dfs_vertical_viewpoints.isna().any()]

# Step 2: Fill NaN values with the mean only in the columns with NaN values
dfs_horizontal_viewpoints[cols_with_nan_hori] = dfs_horizontal_viewpoints[cols_with_nan_hori].apply(lambda col: col.fillna(col.mean()), axis=0)
dfs_vertical_viewpoints[cols_with_nan_vert] = dfs_vertical_viewpoints[cols_with_nan_vert].apply(lambda col: col.fillna(col.mean()), axis=0)

### Not needed in general just to debug ###
# # Get the first 100 samples of each subject_id in the horizontal viewpoint
# dfs_horizontal_viewpoints = dfs_horizontal_viewpoints.groupby('subject_id').head(100).reset_index(drop=True)
# # Get the first 100 samples of each subject_id in the vertical viewpoint
# dfs_vertical_viewpoints = dfs_vertical_viewpoints.groupby('subject_id').head(100).reset_index(drop=True)

# Group by 'subject_id' and 'pose' for horizontal viewpoint and sample up to n rows per subject_id and pose
if args.samples_per_pose is not None:
    dfs_horizontal_viewpoints = dfs_horizontal_viewpoints.groupby('subject_id').apply(
        lambda x: x.groupby('pose').head(args.samples_per_pose)
    ).reset_index(drop=True)

    # Group by 'subject_id' and 'pose' for vertical viewpoint and sample up to samples_per_pose rows per subject_id
    dfs_vertical_viewpoints = dfs_vertical_viewpoints.groupby('subject_id').apply(
        lambda x: x.groupby('pose').head(args.samples_per_pose)
    ).reset_index(drop=True)

# Get all the poses
poses = pd.unique(dfs_horizontal_viewpoints['pose'])
# Create a string sepated by commas of the poses
poses_str = ', '.join(poses)

# Loop over the subject_ids and take the corresponding idx as the validation set to compute a LOSO-CV
# subject_ids = [7, 19] # Debugging
subject_ids = dfs_horizontal_viewpoints['subject_id'].unique()

if __name__ == '__main__':
    # Create datasets
    train_dataset, val_dataset = create_datasets(dfs_horizontal_viewpoints, 
                                                 dfs_vertical_viewpoints, 
                                                 args.subject_id_val, 
                                                 train_subject_id=args.subject_id_train,
                                                 images_path=args.img_path)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


    print("Done!")