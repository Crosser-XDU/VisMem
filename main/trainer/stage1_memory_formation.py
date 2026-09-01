from __future__ import annotations
from typing import Dict, Any, Optional, List
import torch
import torch.nn.functional as F

from main.utils.qwen_vl import extend_qwen_vl_inputs

def _as_list(target_text):
    if isinstance(target_text, str):
        return [target_text]
    return list(target_text)


def stage1_loss(base_model, vismem_model, inputs: Dict[str, Any], target_text, mem_type: str = "short"):

    tokenizer = vismem_model.tokenizer
    device = vismem_model.device

    # Encode target
    targets = _as_list(target_text)
    tgt = tokenizer(targets, return_tensors="pt", padding=True, add_special_tokens=False)
    tgt_ids = tgt.input_ids.to(device)
    tgt_mask = tgt.attention_mask.to(device)

    with torch.no_grad():
        full_ids = torch.cat([inputs["input_ids"], tgt_ids], dim=1)
        prompt_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"], dtype=torch.long))
        attn = torch.cat([prompt_mask, tgt_mask], dim=1)
        labels = full_ids.clone()
        labels[:, :inputs["input_ids"].size(1)] = -100
        labels[:, inputs["input_ids"].size(1):] = labels[:, inputs["input_ids"].size(1):].masked_fill(tgt_mask == 0, -100)
        out = base_model(**extend_qwen_vl_inputs(inputs, full_ids, attn))
        logits = out.logits[:, :-1, :]
        loss_base = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)

    base_out = base_model(**inputs, output_hidden_states=True)
    hidden = base_out.hidden_states[-1]  # (B,T,D)
    # Build H
    H = hidden
    M = vismem_model.form_memory(H, mem_type=mem_type)

    # Feed
    if getattr(base_out, "hidden_states", None) is not None and len(base_out.hidden_states) > 0:
        emb = base_out.hidden_states[0].detach()
    else:
        emb = base_model.get_input_embeddings()(inputs["input_ids"]).detach()
    inp_embeds = torch.cat([emb, M, base_model.get_input_embeddings()(tgt_ids)], dim=1)
    prompt_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"], dtype=torch.long))
    mem_mask = torch.ones((inputs["input_ids"].size(0), M.size(1)), device=device, dtype=torch.long)
    attn2 = torch.cat([prompt_mask, mem_mask, tgt_mask], dim=1)

    labels2 = torch.cat([inputs["input_ids"], torch.full((inputs["input_ids"].size(0), M.size(1)), -100, device=device, dtype=torch.long), tgt_ids], dim=1)
    labels2[:, :inputs["input_ids"].size(1) + M.size(1)] = -100
    labels2[:, inputs["input_ids"].size(1) + M.size(1):] = labels2[:, inputs["input_ids"].size(1) + M.size(1):].masked_fill(tgt_mask == 0, -100)

    out2 = base_model(inputs_embeds=inp_embeds, attention_mask=attn2)
    logits2 = out2.logits[:, :-1, :]
    loss_mem = F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), labels2[:, 1:].reshape(-1), ignore_index=-100)

    # Optimize
    return loss_mem, loss_base
