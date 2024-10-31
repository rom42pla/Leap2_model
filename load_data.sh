#!/bin/bash

IMG_PATH="/autofs/thau00a/home/sestebanro/thau01/Multiview_Leap2_Hand_Pose_Dataset/"
SUBJECT_ID_VAL=1
SUBJECT_ID_TRAIN=2
DATAFRAMES_PATH="/autofs/thau00a/home/sestebanro/thau03/Multi_View_Leap2/data"
SAMPLES_PER_POSE=20
BATCH_SIZE=16
SEED=42

# python load_data.py --img_path $IMG_PATH --subject_id_val $SUBJECT_ID_VAL --subject_id_train $SUBJECT_ID_TRAIN \
#  --dataframes_path $DATAFRAMES_PATH --samples_per_pose $SAMPLES_PER_POSE --seed $SEED


python load_data.py --img_path $IMG_PATH --subject_id_val $SUBJECT_ID_VAL --dataframes_path $DATAFRAMES_PATH \
    --batch_size $BATCH_SIZE --seed $SEED