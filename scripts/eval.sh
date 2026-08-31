#!/usr/bin/env bash
set -euo pipefail

python -m main.cli.eval \
  --config configs/vismem_qwen25vl7b.yaml \
  --jsonl /path/to/eval.jsonl \
  --ckpt outputs/stage2/epoch0 \
  --enable_vismem \
  --metric substr
