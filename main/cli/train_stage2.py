from __future__ import annotations
import argparse
import os
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
from main.trainer.rewards import exact_match_reward
from main.cli.common import load_yaml, build_vismem_config
from main.trainer.stage2_invocation import compute_penalties
from main.trainer.paper_grpo import sample_paper_stage2_rollout

logger = get_logger("main.train_stage2")

ptype_w = 1.0
pneg_w = 1.0


def _pad_generated(generated, pad_id: int, device):
    max_len = max(int(ids.size(1)) for ids in generated)
    batch = torch.full((len(generated), max_len), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros((len(generated), max_len), device=device, dtype=torch.long)
    for i, ids in enumerate(generated):
        row = ids[0].to(device)
        batch[i, : row.numel()] = row
        mask[i, : row.numel()] = 1
    return batch, mask


def _repeat_prompt_inputs(inputs, repeat: int):
    out = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
        elif key in ("input_ids", "attention_mask"):
            out[key] = value.repeat(repeat, 1)
        elif key.endswith("grid_thw") and value.size(0) == 1:
            out[key] = value.repeat(repeat, 1)
        elif key.startswith("pixel_values"):
            out[key] = value.repeat((repeat,) + (1,) * (value.dim() - 1))
        elif value.size(0) == 1:
            out[key] = value.repeat((repeat,) + (1,) * (value.dim() - 1))
        else:
            out[key] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vismem_qwen25vl7b.yaml")
    ap.add_argument("--model_name_or_path", default=None)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--init_from", default=None, help="Stage1 checkpoint folder containing main.pt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--kl_beta", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--group_size", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
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
    dtype = to_torch_dtype(cfg_dict["model"].get("torch_dtype", "bfloat16"))
    device_map = resolve_device_map(cfg_dict["model"].get("device_map", "auto"), accelerator)
    trust = bool(cfg_dict["model"].get("trust_remote_code", True))

    base_model, tokenizer, processor = load_qwen25vl(model_name, torch_dtype=dtype, device_map=device_map,
                                                     trust_remote_code=trust)
    vismem = VisMemModel(base_model, tokenizer, processor, viscfg)
    vismem.move_aux_modules_to_device(accelerator.device)

    # load stage1
    if args.init_from is not None:
        state = torch.load(os.path.join(args.init_from, "main.pt"), map_location="cpu")
        vismem.load_state_dict(state["vismem_state"], strict=False)

    # Freeze memory formation; train only a small subset to learn invocation patterns
    for p in vismem.parameters():
        p.requires_grad = False

    # Unfreeze token embeddings
    emb = vismem.base_model.get_input_embeddings()
    emb.weight.requires_grad = True
    output_emb = getattr(vismem.base_model, "get_output_embeddings", lambda: None)()
    train_params = [emb.weight]
    if output_emb is not None and output_emb.weight.data_ptr() != emb.weight.data_ptr():
        output_emb.weight.requires_grad = True
        train_params.append(output_emb.weight)

    special_ids = torch.tensor(
        [vismem.short_invoke_id, vismem.short_end_id, vismem.long_invoke_id, vismem.long_end_id],
        device=vismem.device,
        dtype=torch.long,
    )
    end_ids = torch.tensor([vismem.short_end_id, vismem.long_end_id], device=vismem.device, dtype=torch.long)
    end_lr_mult = 0.1

    def grad_mask_hook(grad):
        g = torch.zeros_like(grad)
        g[special_ids] = grad[special_ids]
        g[end_ids] *= end_lr_mult
        return g

    emb.weight.register_hook(grad_mask_hook)
    if output_emb is not None and output_emb.weight.data_ptr() != emb.weight.data_ptr():
        output_emb.weight.register_hook(grad_mask_hook)

    lr = args.lr if args.lr is not None else float(train_cfg.get("stage2_lr", train_cfg.get("lr", 1e-5)))
    weight_decay = float(train_cfg.get("stage2_weight_decay", 0.0))
    opt = optim.AdamW(train_params, lr=lr, weight_decay=weight_decay)


    ref_model = None
    try:
        import copy
        ref_model = copy.deepcopy(vismem.base_model).eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    except Exception:
        ref_model = None

    ds = JsonlVLDataset(args.train_jsonl)
    ensure_dir(args.output_dir)
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 1))
    paper_aligned = bool(train_cfg.get("paper_aligned", False))
    if batch_size != 1:
        raise ValueError("Stage2 currently expects per-device batch_size=1 because each prompt samples a candidate group.")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_samples)
    vismem, opt, dl = accelerator.prepare(vismem, opt, dl)
    unwrapped = accelerator.unwrap_model(vismem)

    group_size = args.group_size if args.group_size is not None else int(train_cfg.get("group_size", 4))
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    temperature = args.temperature if args.temperature is not None else float(train_cfg.get("stage2_temperature", 0.7))
    top_p = args.top_p if args.top_p is not None else float(train_cfg.get("stage2_top_p", 0.9))
    max_new_tokens = int(train_cfg.get("max_new_tokens", 128))
    kl_beta = args.kl_beta if args.kl_beta is not None else float(train_cfg.get("kl_beta", 0.02))
    clip_ratio = float(train_cfg.get("clip_ratio", 0.2))
    penalty_alpha = float(train_cfg.get("penalty_alpha", 0.3))
    policy_epochs = int(train_cfg.get("paper_policy_epochs", 2 if paper_aligned else 1))

    vismem.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Stage2 epoch {epoch}", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            img = batch["images"][0]
            prompt = batch["prompts"][0]
            answer = batch["answers"][0]
            if answer is None:
                continue

            if paper_aligned:
                rollout = sample_paper_stage2_rollout(
                    unwrapped,
                    images=[img],
                    prompts=[prompt],
                    answers=[answer],
                    group_size=group_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    penalty_alpha=penalty_alpha,
                )
                for _ in range(policy_epochs):
                    with accelerator.accumulate(vismem):
                        loss, metrics = vismem(
                            stage="stage2_paper",
                            rollout=rollout,
                            clip_ratio=clip_ratio,
                            kl_beta=kl_beta,
                            ref_model=ref_model,
                        )
                        opt.zero_grad()
                        accelerator.backward(loss)
                        opt.step()
                reward_mean = metrics["raw_reward"]
            else:
                with accelerator.accumulate(vismem):
                    # Prepare prompt inputs
                    inputs = prepare_qwen_vl_inputs(processor, prompts=[prompt], images=[img])
                    inputs = move_to_device(inputs, accelerator.device)

                    preds, rev_preds, gen_ids_list = [], [], []
                    for _ in range(group_size):
                        pred_list, gen_ids = unwrapped.generate(
                            images=[img],
                            prompts=[prompt],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            enable_vismem=True,
                            return_token_ids=True,
                            skip_special_tokens=True,
                        )
                        preds.append(pred_list[0])
                        gen_ids_list.append(gen_ids.detach())

                        pred_rev_list = unwrapped.generate(
                            images=[img],
                            prompts=[prompt],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            enable_vismem=True,
                            return_token_ids=False,
                            skip_special_tokens=True,
                            reverse_mem_type=True,
                        )
                        rev_preds.append(pred_rev_list[0])

                    rewards_main = exact_match_reward(preds, [answer] * group_size)
                    rewards_rev = exact_match_reward(rev_preds, [answer] * group_size)
                    reward_mean = sum(rewards_main) / max(1, len(rewards_main))
                    rewards_eff = []
                    for r, r_rev in zip(rewards_main, rewards_rev):
                        pen = compute_penalties(float(r), float(r_rev), reward_mean)
                        rewards_eff.append(float(r) - ptype_w * pen["ptype"] - pneg_w * pen["pneg"])

                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    sampled_ids, sampled_mask = _pad_generated(gen_ids_list, pad_id, accelerator.device)
                    prompt_inputs = _repeat_prompt_inputs(inputs, group_size)
                    rewards = torch.tensor(rewards_eff, device=accelerator.device, dtype=torch.float32)
                    loss = vismem(
                        stage="stage2",
                        prompts_inputs=prompt_inputs,
                        sampled_ids=sampled_ids,
                        sampled_attention_mask=sampled_mask,
                        rewards=rewards,
                        ref_model=ref_model,
                        kl_beta=kl_beta,
                    )
                    opt.zero_grad()
                    accelerator.backward(loss)
                    opt.step()

            if accelerator.is_local_main_process:
                pbar.set_postfix({"reward": float(reward_mean), "loss": float(loss.detach().cpu())})

        ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
        if accelerator.is_main_process:
            ensure_dir(ckpt)
            accelerator.save({"vismem_state": unwrapped.state_dict(), "config": cfg_dict}, os.path.join(ckpt, "main.pt"))
            tokenizer.save_pretrained(ckpt)
        accelerator.wait_for_everyone()


    if accelerator.is_main_process:
        logger.info("Stage2 done.")

if __name__ == "__main__":
    main()
