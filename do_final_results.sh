##################
# Tiny HGR dataset
##################

python do_results.py --validation loso --train_image_backbone --dataset=tiny_hgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks --max_epochs 5
python do_results.py --validation loso --train_image_backbone --dataset=tiny_hgr --use_horizontal_image --normalize_landmarks --max_epochs 5
python do_results.py --validation loso --train_image_backbone --dataset=tiny_hgr --use_horizontal_landmarks --normalize_landmarks --max_epochs 30

##################
# MMHGDHGR dataset
##################

python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_landmarks --landmarks_backbone none --normalize_landmarks --max_epochs 30
python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --normalize_landmarks --max_epochs 10
python do_results.py --validation simple --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks --max_epochs 10

python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_landmarks --normalize_landmarks --max_epochs 30
python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --normalize_landmarks --max_epochs 10
python do_results.py --validation loso --train_image_backbone --dataset=mmhgdhgr --use_horizontal_image --use_horizontal_landmarks --normalize_landmarks --max_epochs 10

###############
# ML2HP dataset
###############

# python do_results.py --train_image_backbone --dataset=ml2hp --use_horizontal_landmarks --use_vertical_landmarks --max_epochs 3
# python do_results.py --train_image_backbone --dataset=ml2hp --use_horizontal_landmarks --use_vertical_landmarks --use_horizontal_image --use_vertical_image --max_epochs 3

python do_results.py --validation loso --dataset=ml2hp --use_horizontal_landmarks --use_vertical_landmarks --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_horizontal_landmarks --use_vertical_landmarks --use_horizontal_image --use_vertical_image --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_horizontal_landmarks --use_horizontal_image --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_vertical_landmarks --use_vertical_image --normalize_landmarks --max_epochs 3

python do_results.py --validation loso --dataset=ml2hp --use_horizontal_image --use_vertical_image --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_horizontal_image --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_vertical_image --normalize_landmarks --max_epochs 3

python do_results.py --validation loso --dataset=ml2hp --use_horizontal_landmarks --normalize_landmarks --max_epochs 3
python do_results.py --validation loso --dataset=ml2hp --use_vertical_landmarks --normalize_landmarks --max_epochs 3

