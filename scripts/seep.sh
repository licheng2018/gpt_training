#!/bin/bash

source scripts/env.sh

SEQ=(512 1024)

MB=(1 2 4 8)
ACC=(16 8 4 2)

for seq in ${SEQ[@]}
do

for i in ${!MB[@]}
do

mb=${MB[$i]}
acc=${ACC[$i]}

echo "Running seq=$seq mb=$mb acc=$acc"

torchrun --nproc_per_node=4 train.py \
  --strategy ddp \
  --dataset synthetic \
  --seq_len $seq \
  --microbatch $mb \
  --grad_accum $acc \
  --steps 200 \
  --fp16 \
  --log_every 20 \
  2>&1 | tee logs/day5_seq${seq}_mb${mb}_acc${acc}.log

done

done