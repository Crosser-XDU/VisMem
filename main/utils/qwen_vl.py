from __future__ import annotations
from typing import Any, List, Optional
from main.constants import ALL_SPECIAL_TOKENS
import torch


def add_tokens(tokenizer):
    special = {"additional_special_tokens": ALL_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special)
    return tokenizer


def init_token_embeddings(model, tokenizer, init_from_token: str | None = None, noise_std: float = 1e-3):
    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        return
    w_in = emb_layer.weight

    init_id = None
    if init_from_token is not None:
        init_id = tokenizer.convert_tokens_to_ids(init_from_token)
    if init_id is None or init_id == tokenizer.unk_token_id:
        init_id = tokenizer.eos_token_id
    if init_id is None:
        return

    with torch.no_grad():
        ref = w_in[init_id].detach().clone()
        for tok in ALL_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                continue
            w_in[tid].copy_(ref + torch.randn_like(ref) * noise_std)

        out_layer = getattr(model, "get_output_embeddings", lambda: None)()
        if out_layer is not None and out_layer.weight.data_ptr() != w_in.data_ptr():
            w_out = out_layer.weight
            for tok in ALL_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid == tokenizer.unk_token_id:
                    continue
                w_out[tid].copy_(w_in[tid])


def _model_class_for_config(config):
    """Resolve the concrete Qwen-VL class without forcing one model generation."""
    import transformers

    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_vl":
        model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
        if model_class is None:
            raise RuntimeError(
                "This Transformers installation does not support Qwen3-VL. "
                "Install transformers>=4.57.0, then retry."
            )
        return model_class
    if model_type == "qwen2_5_vl":
        model_class = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if model_class is not None:
            return model_class

    # Keep support for other image-text models registered with Transformers.
    model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForVision2Seq", None)
    if model_class is None:
        raise RuntimeError(
            f"No compatible vision-language AutoModel class was found for model_type={model_type!r}."
        )
    return model_class


def load_qwen_vl(
    model_name_or_path: str,
    torch_dtype=None,
    device_map="auto",
    trust_remote_code=True,
    local_files_only: bool = False,
):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(model_name_or_path, **load_kwargs)
    model_class = _model_class_for_config(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **load_kwargs)

    old_vocab = len(tokenizer)
    tokenizer = add_tokens(tokenizer)
    model = model_class.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **load_kwargs,
    )
    # resize embeddings after adding tokens
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    if len(tokenizer) > old_vocab:
        init_token_embeddings(model, tokenizer, init_from_token=None, noise_std=1e-3)

    return model, tokenizer, processor


def load_qwen25vl(*args, **kwargs):
    """Backward-compatible alias; new code should use :func:`load_qwen_vl`."""
    return load_qwen_vl(*args, **kwargs)


def _has_image_token(prompt: str) -> bool:
    return "<|vision_start|>" in prompt or "<image>" in prompt or "<|image_pad|>" in prompt


def prepare_qwen_vl_inputs(processor, prompts: List[str], images: Optional[List[Any]], **kwargs):
    use_template = images is not None and any(img is not None for img in images)
    texts = prompts
    if use_template and hasattr(processor, "apply_chat_template"):
        texts = []
        for prompt, image in zip(prompts, images):
            if image is not None and not _has_image_token(prompt):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(prompt)
    image_inputs = images if use_template else None
    return processor(text=texts, images=image_inputs, return_tensors="pt", padding=True, **kwargs)
