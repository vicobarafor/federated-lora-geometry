from __future__ import annotations

from typing import Dict, List

import torch

from src.models.lora_utils import average_deltas


def fedavg(client_updates: List[Dict]):
    total_examples = sum(u["num_examples"] for u in client_updates)
    weights = torch.tensor(
        [u["num_examples"] / total_examples for u in client_updates],
        dtype=torch.float32,
    )

    deltas = [u["delta"] for u in client_updates]
    aggregated_delta = average_deltas(deltas, weights.tolist())

    return {
        "aggregated_delta": aggregated_delta,
        "weights": weights,
        "scores": None,
        "alignments": None,
        "distances": None,
        "norms": None,
    }
