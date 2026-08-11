#!/bin/bash

# Start base image services (Jupyter/SSH) in background
/start.sh &

# Wait for services to start
sleep 2

# Run tuning
for model in configs/models/classification/*; do
  for pretraining in configs/pretraining/*; do
    echo "$(date)" "$model" "$pretraining"
    python3 -m transfer_learning.train fit \
        --config configs/base.yaml \
        --config configs/task/segmentation_1.yaml \
        --config "$model" \
        --config configs/optimization/adamw_basic.yaml \
        --config "$pretraining" \
        --config configs/augmentation/microscope.yaml "$@"
    done
  done

# Stop pod
runpodctl stop pod $RUNPOD_POD_ID
