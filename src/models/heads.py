from __future__ import annotations

import torch
import torch.nn as nn


TASK_REGISTRY = {
    "sst2": {"num_labels": 2},
    "qnli": {"num_labels": 2},
}


class ClassificationHead(nn.Module):
    """Simple classification head."""

    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))
