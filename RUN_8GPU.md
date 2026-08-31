# Running VisMem on 8 x 80GB GPUs

This repository now uses Accelerate for data-parallel training. On an 8 GPU machine, each process owns one full Qwen2.5-VL-7B model replica. Do not use model sharding with `device_map: auto` under DDP; the training entrypoints replace it with a per-rank device map automatically when `WORLD_SIZE > 1`.

## Environment

```bash
conda create -n vismem python=3.10 -y
conda activate vismem
pip install -r requirements.txt
```

If your Transformers build cannot load `Qwen/Qwen2.5-VL-7B-Instruct`, upgrade Transformers first:

```bash
pip install -U "transformers>=4.49.0" accelerate peft
```

## Data

The JSONL loader expects one sample per line:

```json
{"id":"0","image":"/abs/path/to/image.jpg","prompt":"Question text","answer":"Reference answer"}
```

Samples without `answer` are skipped during training.

## Stage I

Compatibility config, closest to the original public code defaults:

```bash
accelerate launch --num_processes 8 -m main.cli.train_stage1 \
  --config configs/vismem_qwen25vl7b.yaml \
  --train_jsonl /path/to/train.jsonl \
  --output_dir outputs/stage1 \
  --epochs 1
```

Paper-oriented config:

```bash
accelerate launch --num_processes 8 -m main.cli.train_stage1 \
  --config configs/vismem_qwen25vl7b_paper.yaml \
  --train_jsonl /path/to/train.jsonl \
  --output_dir outputs/stage1_paper \
  --epochs 2
```

The paper-oriented config enables `training.paper_aligned: true`. Stage I samples a group of memory-enhanced trajectories, compares each reward against the no-memory trajectory, and optimizes the memory formation path with a clipped GRPO objective.

## Stage II

```bash
accelerate launch --num_processes 8 -m main.cli.train_stage2 \
  --config configs/vismem_qwen25vl7b.yaml \
  --train_jsonl /path/to/train.jsonl \
  --init_from outputs/stage1/epoch0 \
  --output_dir outputs/stage2 \
  --epochs 1
```

Paper-oriented:

```bash
accelerate launch --num_processes 8 -m main.cli.train_stage2 \
  --config configs/vismem_qwen25vl7b_paper.yaml \
  --train_jsonl /path/to/train.jsonl \
  --init_from outputs/stage1_paper/epoch1 \
  --output_dir outputs/stage2_paper \
  --epochs 2
```

Stage II in the paper-oriented config samples candidate trajectories from the invocation policy, evaluates the reverse-memory trajectory penalty, and replays the latent memory insertions while computing the policy loss.

`paper_policy_epochs` controls how many optimizer updates reuse one sampled rollout. Values above 1 make the clipped GRPO ratio active after the first update.

## Practical 80GB Notes

Start with `batch_size: 1`. For Stage II, `group_size` controls how many candidate completions are sampled per prompt on each GPU. If memory spikes, lower `max_new_tokens`, image resolution, or `group_size` first.

## Checks

After installing dependencies on the GPU machine, run:

```bash
python -m unittest discover -s tests -v
python -m compileall main tests
bash -n scripts/train_stage1.sh scripts/train_stage2.sh scripts/infer.sh scripts/eval.sh
```
