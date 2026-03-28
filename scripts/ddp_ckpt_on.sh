#!/bin/bash
set -euo pipefail

source scripts/env.sh
mkdir -p logs outputs

torchrun --nproc_per_node=4 train_check_point.py \
  --strategy ddp \
  --dataset wikitext \
  --seq_len 1024 \
  --microbatch 2 \
  --grad_accum 8 \
  --steps 100 \
  --warmup_steps 20 \
  --fp16 \
  --checkpoint \
  --log_every 10 \
  --out_dir ./outputs/day4_ddp_ckpt1_mb4_acc4 \
  2>&1 | tee ./logs/day4_ddp_ckpt1_mb4_acc4.log