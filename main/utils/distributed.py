from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


class SingleProcessAccelerator:
    is_main_process = True
    is_local_main_process = True
    num_processes = 1
    process_index = 0
    local_process_index = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare(self, *args):
        return args[0] if len(args) == 1 else args

    def accumulate(self, _model):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        return None

    def save(self, obj: Any, path: str):
        torch.save(obj, path)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)


def build_accelerator(grad_accum: int = 1):
    try:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs
    except Exception:
        return SingleProcessAccelerator()

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(gradient_accumulation_steps=max(1, int(grad_accum)), kwargs_handlers=[kwargs])


def resolve_device_map(configured_device_map, accelerator):
    if getattr(accelerator, "num_processes", 1) > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(accelerator.local_process_index)
        return {"": accelerator.local_process_index}
    return configured_device_map


def move_to_device(batch: dict[str, Any], device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
