#!/bin/bash

source scripts/env.sh

mkdir -p logs

torchrun --nproc_per_node=4 train_check_point.py \
  --strategy ddp \
  --dataset wikitext \
  --seq_len 1024 \
  --microbatch 2 \
  --grad_accum 4 \
  --steps 300 \
  --fp16 \
  --checkpoint \
  2>&1 | tee logs/ddp_checkpoint.log
