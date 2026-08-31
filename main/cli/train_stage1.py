from __future__ import annotations
import argparse
import os
import random
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from main.utils.logging import get_logger
from main.utils.misc import set_seed, to_torch_dtype, ensure_dir
from main.utils.qwen_vl import load_qwen25vl, prepare_qwen_vl_inputs
from main.utils.distributed import build_accelerator, resolve_device_map, move_to_device
from main.model.model import VisMemModel
from main.data.jsonl_dataset import JsonlVLDataset
from main.data.collate import collate_samples
from main.cli.common import load_yaml, build_vismem_config
from main.trainer.paper_grpo import sample_paper_stage1_rollout

logger = get_logger("main.train_stage1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--memory_type", choices=["short", "long", "random"], default=None)
    args = ap.parse_args()

    cfg_dict = load_yaml(args.config)
    if args.model_name_or_path is not None:
        cfg_dict["model"]["model_name_or_path"] = args.model_name_or_path
    viscfg = build_vismem_config(cfg_dict)

    train_cfg = cfg_dict.get("training", {})
    grad_accum = args.grad_accum if args.grad_accum is not None else int(train_cfg.get("grad_accum", 1))
    accelerator = build_accelerator(grad_accum)

    set_seed(int(train_cfg.get("seed", 42)) + int(getattr(accelerator, "process_index", 0)))

    model_name = cfg_dict["model"]["model_name_or_path"]
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype","bfloat16"))
    device_map = resolve_device_map(cfg_dict["model"].get("device_map","auto"), accelerator)
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=trust)
    paper_aligned = bool(train_cfg.get("paper_aligned", False))
    ref_model = None
    if paper_aligned:
        try:
            import copy
            ref_model = copy.deepcopy(base_model).eval()
            for p in ref_model.parameters():
                p.requires_grad = False
        except Exception:
            ref_model = None
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    vismem.move_aux_modules_to_device(accelerator.device)

    # Freeze base model
    for p in vismem.base_model.parameters():
        p.requires_grad = False
    for p in vismem.query_builder.parameters():
        p.requires_grad = True
    if vismem.short_former is not None:
        for p in vismem.short_former.parameters():
            p.requires_grad = True
    if vismem.long_former is not None:
        for p in vismem.long_former.parameters():
            p.requires_grad = True
    for name, p in vismem.named_parameters():
        if "lora_" in name or name in ("m_init_short", "m_init_long"):
            p.requires_grad = True

    # Trainable params
    trainable = [p for p in vismem.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found for Stage1. Check PEFT/LoRA installation and config.")
    lr = args.lr if args.lr is not None else float(cfg_dict.get("training", {}).get("lr", 2e-4))
    opt = optim.AdamW(trainable, lr=lr)

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 1))
    if paper_aligned and batch_size != 1:
        raise ValueError("paper_aligned Stage1 expects per-device batch_size=1; with 8 GPUs this gives global batch size 8.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
    vismem, opt, dl = accelerator.prepare(vismem, opt, dl)
    unwrapped = accelerator.unwrap_model(vismem)
    memory_type = args.memory_type or str(train_cfg.get("stage1_memory_type", "short"))
    if memory_type not in ("short", "long", "random"):
        raise ValueError("stage1_memory_type must be one of: short, long, random")
    group_size = int(train_cfg.get("group_size", 16 if paper_aligned else 1))
    max_new_tokens = int(train_cfg.get("max_new_tokens", 128))
    temperature = float(train_cfg.get("stage1_temperature", train_cfg.get("stage2_temperature", 0.7)))
    top_p = float(train_cfg.get("stage1_top_p", train_cfg.get("stage2_top_p", 0.9)))
    clip_ratio = float(train_cfg.get("clip_ratio", 0.2))
    kl_beta = float(train_cfg.get("stage1_kl_beta", 0.015))
    force_position = str(train_cfg.get("stage1_force_position", "random"))
    policy_epochs = int(train_cfg.get("paper_policy_epochs", 2 if paper_aligned else 1))

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Stage1 epoch {epoch}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            answers = batch["answers"]
            keep = [i for i, answer in enumerate(answers) if answer is not None]
            if not keep:
                continue
            images = [batch["images"][i] for i in keep]
            prompts = [batch["prompts"][i] for i in keep]
            answers = [answers[i] for i in keep]

            if paper_aligned:
                rollout = sample_paper_stage1_rollout(
                    unwrapped,
                    images=images,
                    prompts=prompts,
                    answers=answers,
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    memory_type=memory_type,
                    force_position=force_position,
                )
                for _ in range(policy_epochs):
                    with accelerator.accumulate(vismem):
                        loss, metrics = vismem(
                            stage="stage1_paper",
                            rollout=rollout,
                            clip_ratio=clip_ratio,
                            kl_beta=kl_beta,
                            ref_model=ref_model,
                        )
                        opt.zero_grad()
                        accelerator.backward(loss)
                        opt.step()
                mem_type = memory_type
            else:
                with accelerator.accumulate(vismem):
                    inputs = prepare_qwen_vl_inputs(processor, prompts=prompts, images=images)
                    inputs = move_to_device(inputs, accelerator.device)
                    mem_type = random.choice(["short", "long"]) if memory_type == "random" else memory_type
                    loss_mem, loss_base = vismem(stage="stage1", inputs=inputs, target_text=answers, mem_type=mem_type)
                    loss = loss_mem - loss_base.detach()
                    opt.zero_grad()
                    accelerator.backward(loss)
                    opt.step()

            if accelerator.is_local_main_process:
                if paper_aligned:
                    pbar.set_postfix({"mem": mem_type, "reward": metrics["reward"], "loss": float(loss.detach().cpu())})
                else:
                    pbar.set_postfix({"mem": mem_type, "loss_mem": float(loss_mem.detach().cpu()), "loss_base": float(loss_base.detach().cpu())})

        # save checkpoint
        ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
        if accelerator.is_main_process:
            ensure_dir(ckpt)
            accelerator.save({"vismem_state": unwrapped.state_dict(), "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
            tokenizer.save_pretrained(ckpt)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Stage1 done.")

if __name__ == "__main__":
    main()
