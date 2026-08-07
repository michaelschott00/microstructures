#!/bin/bash

# Start base image services (Jupyter/SSH) in background
/start.sh &

# Wait for services to start
sleep 2

# Run tuning
for model in configs/lightning/models/classification/*; do
  for pretraining in configs/lightning/pretraining/*; do
    for sample_size in $(seq .1 .4 1); do
      echo "$(date)" "$model" "$pretraining" "$sample_size"
      python3 -m transfer_learning.train fit \
        --config configs/lightning/base.yaml \
        --config configs/lightning/task/classification_1.yaml \
        --config "$model" \
        --config configs/lightning/optimization/adamw_basic.yaml \
        --config "$pretraining" \
        --config configs/lightning/augmentation/microscope.yaml \
        --data.init_args.sample_size "$sample_size" "$@"
      done
    done
  done

# Stop pod
runpodctl stop pod $RUNPOD_POD_ID
