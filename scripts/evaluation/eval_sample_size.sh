#!/bin/bash

source "$(dirname "$0")/_common.sh"

parse_args "$@"
setup_env

# Run tuning
for model in configs/models/classification/*; do
  for pretraining in configs/pretraining/*; do
    for sample_size in $(seq .1 .4 1); do
      echo "$(date)" "$model" "$pretraining" "$sample_size"
      python3 -m transfer_learning.train fit \
        --config configs/base.yaml \
        --config configs/task/classification_1.yaml \
        --config "$model" \
        --config configs/optimization/adamw_basic.yaml \
        --config "$pretraining" \
        --config configs/augmentation/microscope.yaml \
        --data.init_args.sample_size "$sample_size" \
        "${EXTRA_ARGS[@]}" "${ARGS[@]}"
      done
    done
  done

teardown