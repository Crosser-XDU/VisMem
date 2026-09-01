#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_GPUS="${VISMEM_NUM_GPUS:-8}"
RUN_TAG="${VISMEM_RUN_TAG:-smoke_8gpu}"
MODEL_PATH="${VISMEM_MODEL_NAME_OR_PATH:-/mnt/llmshared-ssd-hd/wangruitao/Models/Qwen/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${VISMEM_SMOKE_OUTPUT_ROOT:-outputs/smoke}"
MAX_NEW_TOKENS="${VISMEM_SMOKE_MAX_NEW_TOKENS:-8}"
GROUP_SIZE="${VISMEM_SMOKE_GROUP_SIZE:-2}"
MASTER_PORT="${VISMEM_MASTER_PORT:-29501}"

case "${RUN_TAG}" in
  *[!A-Za-z0-9._-]*|'')
    echo "VISMEM_RUN_TAG may contain only letters, numbers, dot, underscore, and dash." >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${OUTPUT_ROOT}" = /* ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
else
  RUN_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}/${RUN_TAG}"
fi
DATA_DIR="${RUN_ROOT}/data"
SMOKE_CONFIG="${RUN_ROOT}/smoke_config.yaml"
TRAIN_JSONL="${DATA_DIR}/smoke.jsonl"
STAGE1_DIR="${RUN_ROOT}/stage1"
STAGE2_DIR="${RUN_ROOT}/stage2"

export VISMEM_MODEL_NAME_OR_PATH="${MODEL_PATH}"
export VISMEM_LOCAL_FILES_ONLY="${VISMEM_LOCAL_FILES_ONLY:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

cd "${REPO_ROOT}"

run_checks() {
  echo "[smoke] repository=${REPO_ROOT}"
  echo "[smoke] model=${MODEL_PATH}"
  echo "[smoke] run_root=${RUN_ROOT}"
  echo "[smoke] mode=${MODE}, processes=${NUM_GPUS}"

  test -f "${MODEL_PATH}/config.json" || {
    echo "Missing local model config: ${MODEL_PATH}/config.json" >&2
    exit 1
  }
  command -v accelerate >/dev/null || {
    echo "The accelerate command is not available in the active Python environment." >&2
    exit 1
  }

  "${PYTHON_BIN}" -m unittest discover -s tests -v
  MODEL_PATH="${MODEL_PATH}" NUM_GPUS="${NUM_GPUS}" "${PYTHON_BIN}" - <<'PY'
import ast
import json
import os
from pathlib import Path

import torch
import transformers

source_paths = sorted(Path("main").rglob("*.py")) + sorted(Path("tests").rglob("*.py"))
for source_path in source_paths:
    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
print(f"parsed_python_files={len(source_paths)}")

model_path = os.environ["MODEL_PATH"]
required_gpus = int(os.environ["NUM_GPUS"])
with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    model_type = json.load(f).get("model_type")

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"model_type={model_type}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu[{index}]={props.name}, memory={props.total_memory / 2**30:.1f} GiB")

if model_type != "qwen3_vl":
    raise SystemExit(f"Expected model_type='qwen3_vl', got {model_type!r}")
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    raise SystemExit("Transformers does not provide Qwen3VLForConditionalGeneration; install transformers>=4.57.0")
if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
    raise SystemExit(f"Expected at least {required_gpus} visible GPUs")
PY
}

prepare_assets() {
  mkdir -p "${DATA_DIR}"
  SOURCE_CONFIG="${REPO_ROOT}/configs/vismem_qwen3vl8b.yaml" \
  SMOKE_CONFIG="${SMOKE_CONFIG}" \
  TRAIN_JSONL="${TRAIN_JSONL}" \
  DATA_DIR="${DATA_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os

from PIL import Image
import yaml

source_config = os.environ["SOURCE_CONFIG"]
smoke_config = os.environ["SMOKE_CONFIG"]
train_jsonl = os.environ["TRAIN_JSONL"]
data_dir = os.environ["DATA_DIR"]

with open(source_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["model"]["model_name_or_path"] = os.environ["MODEL_PATH"]
cfg["model"]["local_files_only"] = True
cfg["training"].update(
    batch_size=1,
    grad_accum=1,
    group_size=int(os.environ["GROUP_SIZE"]),
    max_new_tokens=int(os.environ["MAX_NEW_TOKENS"]),
    paper_aligned=False,
)
with open(smoke_config, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

image_path = os.path.join(data_dir, "red.png")
Image.new("RGB", (64, 64), color=(220, 20, 20)).save(image_path)
sample = {
    "id": "smoke-0",
    "image": image_path,
    "prompt": "What is the dominant color in this image? Answer with one lowercase English word only.",
    "answer": "red",
}
with open(train_jsonl, "w", encoding="utf-8") as f:
    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
print(f"[smoke] config={smoke_config}")
print(f"[smoke] data={train_jsonl}")
PY
}

launch_stage1() {
  accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "${MASTER_PORT}" \
    -m main.cli.train_stage1 \
    --config "${SMOKE_CONFIG}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${STAGE1_DIR}" \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 1
}

launch_stage2() {
  test -f "${STAGE1_DIR}/epoch0/main.pt" || {
    echo "Missing Stage I checkpoint: ${STAGE1_DIR}/epoch0/main.pt" >&2
    exit 1
  }
  accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --main_process_port "${MASTER_PORT}" \
    -m main.cli.train_stage2 \
    --config "${SMOKE_CONFIG}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --init_from "${STAGE1_DIR}/epoch0" \
    --output_dir "${STAGE2_DIR}" \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 1 \
    --group_size "${GROUP_SIZE}"
}

case "${MODE}" in
  check)
    run_checks
    prepare_assets
    ;;
  stage1)
    run_checks
    prepare_assets
    launch_stage1
    ;;
  stage2)
    run_checks
    prepare_assets
    launch_stage2
    ;;
  all)
    run_checks
    prepare_assets
    launch_stage1
    launch_stage2
    ;;
  *)
    echo "Usage: $0 {check|stage1|stage2|all}" >&2
    exit 2
    ;;
esac

echo "[smoke] completed mode=${MODE}; outputs=${RUN_ROOT}"
