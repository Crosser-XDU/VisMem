#!/usr/bin/env bash
set -euo pipefail

accelerate launch --num_processes 8 -m main.cli.train_stage2 \
  --config configs/vismem_qwen25vl7b.yaml \
  --train_jsonl /path/to/train.jsonl \
  --init_from outputs/stage1/epoch0 \
  --output_dir outputs/stage2 \
  --epochs 1
