# Leap2_dataloader
Repository to efficiently load Leap2 data. The entire pipeline is meant to process the data using Pytorch dataloaders.


## Table of contents

The structure of the repository is described in this section. There is a dictionary (poses_dict.json) that contains the translation of each pose into a specific value so there is a unified representation of the poses.

The file that defines the pipeline to properly load the data is **load_data.py**. There is also a **load_data.sh** to launch the process directly from a shell.

The rest of the files are those with specific functions to perform the data processing process. All functions and methods are docummented on the code so refer to the specific function of interest to know which arguments are required and which is the outcome provided.

The structure of files in this repository is presented below:
```bash
Leap2_dataloader
├── data
│   └── poses_dict.json
├── load_data.py
├── load_data.sh
├── notebooks
│   └── create_train_dataframes.ipynb
├── README.md
├── requirements.txt
└── src
    ├── dataloaders
    │   ├── hand_pose_dataset.py
    └── utils
        ├── readers.py
        ├── utils.py
        └── writers.py
```

## Data loading pipeline

To load the data, you have to run **load_data.py** file. Specifically, it expects a set of arguments to work:

1. **img_path**: the path (absolute if possible) where the **Multiview_Leap2_Hand_Pose_Dataset** is stored.
2. **subject_id_val**: Subject id to be used as validation. All data corresponding to that subject will go to the validation dataset and the rest will be part of the training set.
3. **subject_id_train**: Subject id to be used as training. It can be only and integer, a list does not work for now. More meant to be used when debugging to reduce computational resources.
4. **dataframes_path**: path corresponding to the dataframes that have the information of all subject_ids. They can be directly generated from the code. To do so, take a look in **notebooks/create_train_dataframes.ipynb**. Optionally, in **load_data.py** it is possible to set **save=True** as argument in the **create_viewpoint_dataset** which is set by default as false. However, we encountered some errors when directly load the data so the usage of the notebook is recommended as it already considers that circunstancies.
5. **samples_per_pose**: the amount of samples to be taken from each pose for every subject. Helpful when debugging as selection is not random, it directly takes the first samples_per_pose elements. 
6. **batch_size**: batch size to be defined in the pytorch dataloader.
7. **seed**: set to 42 by default (feel free to use 1234 or any other you may like :) )

With that you are almost ready to go. The code is pretty straightforward: 
1. First, tt loads the poses dictionary. 
2. Secondly, it either creates the dataframes with the horizontal and vertical viewpoints or loads them if they have been previously stored (which we strongly recommend to do). 
3. As described in the paper, there is one sample with some missing values which have been replaced with the mean of the column without further validation of whether that is the best value to use.
4. The datasets are created using a custom **Dataset** named **HandPoseDataset** which can be found in **"src/dataloaders/hand_pose_dataset.py"**. 
5. Finally, the dataloader is created with the specified batch size.

In case you want to use the data in a different way, just take all that you may consider important. The **__getitem__** of the **HandPoseDataset** object is illustrative of which we consider as a simple way to select one specific sample among the entire dataset without introducing too many latency in the code. 


### Debug in visual studio code

In case you are familiar VS Code as IDE for coding and debugging, you may find helpful how my **launch.json** file looks like. If you want to modify any other parameter of those expected in **load_data** argument parser definition just add them to the args definition.
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "load_data",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/load_data.py",
            "args": [
                "--img_path", "/path/to/Multiview_Leap2_Hand_Pose_Dataset",
                "--subject_id_val", "1",
                "--dataframes_path", "/path/to/Multi_View_Leap2/data",
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

### Launch as a shell script
In case you want to run it as a bash script you can use the **load_data.sh** file. There you can easily modify the arguments to those that best suit your needs. It simply launches the **load_data.py**, which has been described in [Data loading pipeline](#data-loading-pipeline) section, with the required arguments defined.   

```bash
bash load_data.sh
```

## Dependencies

The required dependendencies to run the code are defined in the **requirements.txt** file. To install them use:
```bash
pip install -r requirements.txt
```
