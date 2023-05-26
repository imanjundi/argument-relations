#!/bin/bash

args=(
  --do_eval False
)
bash eval_baseline.sh "${args[@]}" "$@"
