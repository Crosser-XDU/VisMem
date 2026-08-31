#!/usr/bin/env bash
set -euo pipefail

accelerate launch --num_processes 8 -m main.cli.train_stage1 \
  --config configs/vismem_qwen25vl7b.yaml \
  --train_jsonl /path/to/train.jsonl \
  --output_dir outputs/stage1 \
  --epochs 1
