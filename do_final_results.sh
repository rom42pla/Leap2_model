###############
# ML2HP dataset
###############

# python do_results.py --use_horizontal_landmarks --use_vertical_landmarks
# python do_results.py --use_horizontal_landmarks --use_vertical_landmarks --use_horizontal_image --use_vertical_image

# python do_results.py --use_horizontal_landmarks --use_vertical_landmarks --normalize_landmarks
# python do_results.py --use_horizontal_landmarks --use_vertical_landmarks --use_horizontal_image --use_vertical_image --normalize_landmarks
# python do_results.py --use_horizontal_landmarks --use_horizontal_image --normalize_landmarks
# python do_results.py --use_vertical_landmarks --use_vertical_image --normalize_landmarks

# python do_results.py --use_horizontal_image --use_vertical_image --normalize_landmarks
# python do_results.py --use_horizontal_image --normalize_landmarks
# python do_results.py --use_vertical_image --normalize_landmarks

# python do_results.py --use_horizontal_landmarks --normalize_landmarks
# python do_results.py --use_vertical_landmarks --normalize_landmarks

##################
# MMHGDHGR dataset
##################

python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --normalize_landmarks
python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks
python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_landmarks --normalize_landmarks

python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --normalize_landmarks
python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks
python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_landmarks --normalize_landmarks

# python do_results.py --dataset=mmhgdhgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks
# python do_results.py --dataset=mmhgdhgr --use_horizontal_image --normalize_landmarks
# python do_results.py --dataset=mmhgdhgr --use_horizontal_landmarks --normalize_landmarks