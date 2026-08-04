#!/bin/bash

for model in configs/models/classification/*
do
  for pretraining in configs/pretraining/*
  do
    python -m transfer_learning.train fit \
        --config configs/base.yaml \
        --config configs/task/segmentation_1.yaml \
        --config "$model" \
        --config configs/optimization/adamw_basic.yaml \
        --config "$pretraining" \
        --config configs/augmentation/microscope.yaml
    done
  done

