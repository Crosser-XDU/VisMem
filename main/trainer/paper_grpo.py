from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch

from main.trainer.rewards import exact_match_reward
from main.trainer.stage2_invocation import compute_penalties
from main.utils.distributed import move_to_device
from main.utils.qwen_vl import extend_qwen_vl_inputs, prepare_qwen_vl_inputs


def group_advantages(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = scores - scores.mean()
    std = torch.sqrt(torch.mean(centered * centered))
    return centered / (std + eps)


def clipped_grpo_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
                      clip_ratio: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(logp - old_logp.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    obj = torch.minimum(ratio * advantages.detach(), clipped * advantages.detach())
    return -obj.mean()


def sampled_kl_from_logps(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    log_ratio = ref_logp.detach() - logp
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()


def _special_action_mask(vismem_model, ids: torch.LongTensor) -> torch.LongTensor:
    mask = torch.ones_like(ids, dtype=torch.long)
    for token_id in [
        vismem_model.short_invoke_id,
        vismem_model.short_end_id,
        vismem_model.long_invoke_id,
        vismem_model.long_end_id,
    ]:
        mask = mask.masked_fill(ids == token_id, 0)
    return mask


def _candidate_inputs(vismem_model, prompt: str, image: Any) -> Dict[str, Any]:
    inputs = prepare_qwen_vl_inputs(vismem_model.processor, prompts=[prompt], images=[image])
    return move_to_device(inputs, vismem_model.device)


def _sample_forced_step(max_new_tokens: int, mode: str) -> int:
    if mode == "start":
        return 0
    if mode == "random":
        return random.randrange(max(1, max_new_tokens))
    raise ValueError("stage1_force_position must be 'start' or 'random'")


def _sample_memory_type(memory_type: str) -> str:
    if memory_type == "random":
        return random.choice(["short", "long"])
    if memory_type in ("short", "long"):
        return memory_type
    raise ValueError("memory_type must be one of: short, long, random")


def _trajectory_logp(vismem_model, prompt: str, image: Any, ids: torch.LongTensor,
                     action_mask: Optional[torch.LongTensor] = None,
                     enable_vismem: bool = True, reverse_mem_type: bool = False):
    inputs = _candidate_inputs(vismem_model, prompt, image)
    return vismem_model.trajectory_logprobs(
        inputs,
        ids.to(vismem_model.device),
        action_mask=action_mask.to(vismem_model.device) if action_mask is not None else None,
        enable_vismem=enable_vismem,
        reverse_mem_type=reverse_mem_type,
    )


def _sampled_ref_kl(vismem_model, ref_model, prompt: str, image: Any, ids: torch.LongTensor,
                    action_mask: torch.LongTensor, logp: torch.Tensor) -> torch.Tensor:
    if ref_model is None:
        return logp.new_zeros(())
    with torch.no_grad():
        ref_inputs = _candidate_inputs(vismem_model, prompt, image)
        ref_input_ids = torch.cat([ref_inputs["input_ids"], ids.to(vismem_model.device)], dim=1)
        generated_attn = torch.ones_like(ids, device=vismem_model.device, dtype=ref_inputs["attention_mask"].dtype)
        ref_attn = torch.cat([ref_inputs["attention_mask"], generated_attn], dim=1)
        model_inputs = extend_qwen_vl_inputs(ref_inputs, ref_input_ids, ref_attn)
        out = ref_model(**model_inputs)
        labels = ref_input_ids.clone()
        labels[:, : ref_inputs["input_ids"].size(1)] = -100
        labels[:, ref_inputs["input_ids"].size(1):] = labels[:, ref_inputs["input_ids"].size(1):].masked_fill(
            action_mask.to(vismem_model.device) == 0, -100
        )
        log_probs = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
        next_labels = labels[:, 1:]
        valid = next_labels != -100
        safe_labels = next_labels.masked_fill(~valid, 0)
        ref_token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_count = valid.sum(dim=1).clamp_min(1)
        ref_logp = (ref_token_logp * valid).sum(dim=1) / token_count
    return sampled_kl_from_logps(logp, ref_logp)


def _loss_from_rollout(vismem_model, rollout: Dict[str, Any], clip_ratio: float,
                       kl_beta: float, ref_model=None):
    losses, kls = [], []
    prompt = rollout["prompt"]
    image = rollout["image"]
    for ids, mask, advantage, old_logp, token_count in zip(
        rollout["gen_ids"],
        rollout["masks"],
        rollout["advantages"],
        rollout["old_logps"],
        rollout["token_counts"],
    ):
        logp, fresh_count = _trajectory_logp(vismem_model, prompt, image, ids, mask, enable_vismem=True)
        logp = logp / fresh_count
        losses.append(clipped_grpo_loss(logp, old_logp, advantage.unsqueeze(0), clip_ratio=clip_ratio))
        kls.append(_sampled_ref_kl(vismem_model, ref_model, prompt, image, ids, mask, logp))

    loss = torch.stack(losses).mean()
    if ref_model is not None and kl_beta > 0:
        loss = loss + kl_beta * torch.stack(kls).mean()
    return loss, rollout["metrics"]


def sample_paper_stage1_rollout(vismem_model, images: List[Any], prompts: List[str], answers: List[str],
                                group_size: int, max_new_tokens: int, temperature: float, top_p: float,
                                memory_type: str = "random", force_position: str = "random"):
    if len(prompts) != 1:
        raise ValueError("paper_stage1_loss expects per-device batch_size=1.")
    prompt, image, answer = prompts[0], images[0], answers[0]

    base_pred = vismem_model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        enable_vismem=False,
        return_token_ids=False,
        skip_special_tokens=True,
    )[0]
    base_score = exact_match_reward([base_pred], [answer])[0]

    gen_ids, masks, scores = [], [], []
    for _ in range(group_size):
        mem_type = _sample_memory_type(memory_type)
        step = _sample_forced_step(max_new_tokens, force_position)
        pred, ids = vismem_model.generate(
            images=[image],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_vismem=True,
            return_token_ids=True,
            skip_special_tokens=True,
            forced_memories=[(step, mem_type)],
        )
        score = exact_match_reward([pred[0]], [answer])[0] - base_score
        gen_ids.append(ids.detach())
        masks.append(_special_action_mask(vismem_model, ids.detach()))
        scores.append(float(score))

    score_tensor = torch.tensor(scores, device=vismem_model.device, dtype=torch.float32)
    advantages = group_advantages(score_tensor)
    old_logps, token_counts = [], []
    for ids, mask in zip(gen_ids, masks):
        with torch.no_grad():
            old_logp, token_count = _trajectory_logp(vismem_model, prompt, image, ids, mask, enable_vismem=True)
        old_logps.append((old_logp / token_count).detach())
        token_counts.append(token_count.detach())
    return {
        "prompt": prompt,
        "image": image,
        "gen_ids": gen_ids,
        "masks": masks,
        "advantages": advantages.detach(),
        "old_logps": old_logps,
        "token_counts": token_counts,
        "metrics": {"reward": float(score_tensor.mean().detach().cpu()), "base": float(base_score)},
    }


def paper_stage1_loss(vismem_model, images: List[Any] | None = None, prompts: List[str] | None = None,
                      answers: List[str] | None = None, group_size: int = 16, max_new_tokens: int = 256,
                      temperature: float = 0.7, top_p: float = 0.9, memory_type: str = "random",
                      force_position: str = "random", clip_ratio: float = 0.2,
                      kl_beta: float = 0.015, ref_model=None, rollout: Dict[str, Any] | None = None):
    if rollout is None:
        rollout = sample_paper_stage1_rollout(
            vismem_model, images, prompts, answers, group_size, max_new_tokens, temperature, top_p,
            memory_type=memory_type, force_position=force_position
        )
    return _loss_from_rollout(vismem_model, rollout, clip_ratio=clip_ratio, kl_beta=kl_beta, ref_model=ref_model)


def sample_paper_stage2_rollout(vismem_model, images: List[Any], prompts: List[str], answers: List[str],
                                group_size: int, max_new_tokens: int, temperature: float, top_p: float,
                                penalty_alpha: float = 0.3):
    if len(prompts) != 1:
        raise ValueError("paper_stage2_loss expects per-device batch_size=1.")
    prompt, image, answer = prompts[0], images[0], answers[0]

    preds, rev_preds, gen_ids, masks = [], [], [], []
    for _ in range(group_size):
        pred, ids = vismem_model.generate(
            images=[image],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_vismem=True,
            return_token_ids=True,
            skip_special_tokens=True,
        )
        reverse_schedule = vismem_model.extract_invocation_schedule(ids.detach(), reverse_types=True)
        rev_pred = vismem_model.generate(
            images=[image],
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_vismem=True,
            return_token_ids=False,
            skip_special_tokens=True,
            forced_memories=reverse_schedule,
        )[0]
        preds.append(pred[0])
        rev_preds.append(rev_pred)
        gen_ids.append(ids.detach())
        masks.append(torch.ones_like(ids.detach(), dtype=torch.long))

    rewards = exact_match_reward(preds, [answer] * group_size)
    rewards_rev = exact_match_reward(rev_preds, [answer] * group_size)
    reward_mean = sum(rewards) / max(1, len(rewards))
    effective_rewards = []
    for reward, reward_rev in zip(rewards, rewards_rev):
        penalties = compute_penalties(float(reward), float(reward_rev), reward_mean)
        effective_rewards.append(float(reward) - penalty_alpha * (penalties["ptype"] + penalties["pneg"]))

    reward_tensor = torch.tensor(effective_rewards, device=vismem_model.device, dtype=torch.float32)
    advantages = group_advantages(reward_tensor)
    old_logps, token_counts = [], []
    for ids, mask in zip(gen_ids, masks):
        with torch.no_grad():
            old_logp, token_count = _trajectory_logp(vismem_model, prompt, image, ids, mask, enable_vismem=True)
        old_logps.append((old_logp / token_count).detach())
        token_counts.append(token_count.detach())
    return {
        "prompt": prompt,
        "image": image,
        "gen_ids": gen_ids,
        "masks": masks,
        "advantages": advantages.detach(),
        "old_logps": old_logps,
        "token_counts": token_counts,
        "metrics": {"reward": float(reward_tensor.mean().detach().cpu()), "raw_reward": float(reward_mean)},
    }


def paper_stage2_loss(vismem_model, images: List[Any] | None = None, prompts: List[str] | None = None,
                      answers: List[str] | None = None, group_size: int = 16, max_new_tokens: int = 256,
                      temperature: float = 0.7, top_p: float = 0.9, clip_ratio: float = 0.2,
                      kl_beta: float = 0.03, penalty_alpha: float = 0.3, ref_model=None,
                      rollout: Dict[str, Any] | None = None):
    if rollout is None:
        rollout = sample_paper_stage2_rollout(
            vismem_model, images, prompts, answers, group_size, max_new_tokens, temperature, top_p,
            penalty_alpha=penalty_alpha
        )
    return _loss_from_rollout(vismem_model, rollout, clip_ratio=clip_ratio, kl_beta=kl_beta, ref_model=ref_model)
