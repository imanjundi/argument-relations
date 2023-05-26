#!/bin/bash

args=(
  --lang english
  --model_name_or_path sentence-transformers/all-mpnet-base-v2
  --hard_negatives False
  --annotated_samples_in_test True
  --do_eval_annotated_samples True
  )
python train_triplets_kialo.py "${args[@]}" "$@"