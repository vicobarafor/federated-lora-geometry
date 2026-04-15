from __future__ import annotations

import torch
import torch.nn as nn


class FederatedClassifier(nn.Module):
    """Encoder + classification head wrapper."""

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    @property
    def hidden_size(self) -> int:
        return self.encoder.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        features = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.head(features)
        return logits
