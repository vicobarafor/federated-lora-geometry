from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model


def apply_lora(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: List[str] | None = None,
):
    if target_modules is None:
        target_modules = ["q_lin", "k_lin", "v_lin"]

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, config)


def _get_lora_module(model):
    if hasattr(model, "encoder"):
        return model.encoder
    return model


def get_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    module = _get_lora_module(model)
    return {
        k: v.detach().cpu().clone()
        for k, v in module.state_dict().items()
        if "lora_" in k
    }


def load_lora_state_dict(
    model,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    module = _get_lora_module(model)
    current = module.state_dict()

    for k, v in state_dict.items():
        if k in current:
            current[k].copy_(v.to(current[k].device))


def subtract_state_dicts(
    new_sd: Dict[str, torch.Tensor],
    old_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: new_sd[k] - old_sd[k] for k in new_sd}


def add_delta_to_state_dict(
    base_sd: Dict[str, torch.Tensor],
    delta_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = deepcopy(base_sd)
    for k in out:
        out[k] = out[k] + delta_sd[k]
    return out


def average_deltas(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float] | None = None,
) -> Dict[str, torch.Tensor]:
    if not deltas:
        raise ValueError("No deltas provided.")

    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)

    result = {}
    keys = deltas[0].keys()

    for k in keys:
        result[k] = sum(w * d[k] for w, d in zip(weights, deltas))

    return result
