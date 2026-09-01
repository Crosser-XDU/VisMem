from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from main.utils.qwen_vl import extend_qwen_vl_inputs

@dataclass
class GRPOBatch:
    input_ids: torch.LongTensor          # (B, T)
    attention_mask: torch.LongTensor     # (B, T)
    labels: torch.LongTensor             # (B, T) with -100 for prompt positions

def sequence_logprobs(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    # gather label logp
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)

def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:

    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)  # (B,T)
    return kl.mean(dim=-1)

def grpo_loss_from_samples(model, prompts_inputs: Dict[str, Any], sampled_ids: torch.LongTensor, rewards: torch.Tensor,
                           ref_model=None, kl_beta: float = 0.02, sampled_attention_mask: Optional[torch.LongTensor] = None):
    # Build full input: prompt + sampled
    input_ids = torch.cat([prompts_inputs["input_ids"], sampled_ids], dim=1)
    prompt_mask = prompts_inputs.get("attention_mask", torch.ones_like(prompts_inputs["input_ids"], dtype=torch.long))
    if sampled_attention_mask is None:
        sampled_attention_mask = torch.ones_like(sampled_ids, dtype=torch.long)
    attn = torch.cat([prompt_mask, sampled_attention_mask], dim=1).to(input_ids.device)
    labels = input_ids.clone()
    labels[:, :prompts_inputs["input_ids"].size(1)] = -100
    labels[:, prompts_inputs["input_ids"].size(1):] = labels[:, prompts_inputs["input_ids"].size(1):].masked_fill(sampled_attention_mask == 0, -100)

    out = model.base_model(**extend_qwen_vl_inputs(prompts_inputs, input_ids, attn), output_hidden_states=False)
    logits = out.logits
    logp = sequence_logprobs(logits[:, :-1, :], labels[:, 1:])

    # Normalize rewards within the candidate group. A single candidate falls back to REINFORCE.
    if rewards.numel() > 1:
        std = rewards.std(unbiased=False)
        adv = rewards - rewards.mean()
        if bool(torch.isfinite(std).item()) and float(std.detach().cpu()) > 1e-6:
            adv = adv / (std + 1e-6)
    else:
        adv = rewards
    pg_loss = -(adv.detach() * logp).mean()

    if ref_model is None or kl_beta <= 0:
        return pg_loss

    with torch.no_grad():
        ref_out = ref_model(**extend_qwen_vl_inputs(prompts_inputs, input_ids, attn))
    kl = kl_divergence(logits[:, :-1, :], ref_out.logits[:, :-1, :])
    return pg_loss + kl_beta * kl.mean()


class SimpleGRPOTrainer:
    def __init__(self, model, ref_model=None, kl_beta: float = 0.02):
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta

    def loss_from_samples(self, prompts_inputs: Dict[str, Any], sampled_ids: torch.LongTensor, rewards: torch.Tensor,
                          sampled_attention_mask: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return grpo_loss_from_samples(
            self.model,
            prompts_inputs=prompts_inputs,
            sampled_ids=sampled_ids,
            sampled_attention_mask=sampled_attention_mask,
            rewards=rewards,
            ref_model=self.ref_model,
            kl_beta=self.kl_beta,
        )
