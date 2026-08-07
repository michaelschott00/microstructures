#!/bin/bash

for model in configs/lightning/models/classification/*; do
  for pretraining in configs/lightning/pretraining/*; do
    python3 -m transfer_learning.train fit \
      --config configs/lightning/base.yaml \
      --config configs/lightning/task/classification_1.yaml \
      --config "$model" \
      --config configs/lightning/optimization/adamw_basic.yaml \
      --config "$pretraining" \
      --config configs/lightning/augmentation/microscope.yaml "$@" \
      2>&1 | tee "$DATA_ROOT/logs/$(date +%Y%m%d_%H%M%S)-$model-$pretraining.log"
    done
  done
