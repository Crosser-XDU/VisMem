#!/usr/bin/env bash
set -euo pipefail

python -m main.cli.infer \
  --config configs/vismem_qwen25vl7b.yaml \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --ckpt outputs/stage2/epoch0 \
  --image /path/to/image.jpg \
  --prompt "Describe the image." \
  --enable_vismem
