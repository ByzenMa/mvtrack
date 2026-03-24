#!/usr/bin/env bash

python3 tools/test.py \
  --config_file='configs/softmax_triplet.yml' \
  MODEL.NAME "('resnet50')" \
  DATASETS.NAMES "('CRTrack')" \
  DATASETS.ROOT_DIR "('..')" \
  OUTPUT_DIR "('../outputs/crtrack_reid_test')"