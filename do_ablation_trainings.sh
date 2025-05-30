#!/bin/bash

# Run generate_configs.py with default parameters
python generate_configs.py

# Loop over each generated config file and run train.py
for config in cfgs/ablation/*.yaml; do
    python train.py --cfg "$config" --disable_checkpointing
done

echo "All training jobs have been done. You can find them in ./checkpoints/ablation/"