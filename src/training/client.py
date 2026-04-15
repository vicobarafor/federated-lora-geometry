from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.models.lora_utils import (
    get_lora_state_dict,
    load_lora_state_dict,
    subtract_state_dicts,
)


class FederatedClient:
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        dataloader,
        lr: float,
        device: str,
        prox_mu: float = 0.0,
    ):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.prox_mu = float(prox_mu)

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )
        self.criterion = nn.CrossEntropyLoss()

    def _compute_prox_term(
        self,
        reference_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        current_state = get_lora_state_dict(self.model)
        prox_term = torch.zeros((), device=self.device)

        for name in current_state.keys():
            current = current_state[name].to(self.device)
            reference = reference_state[name].to(self.device)
            prox_term = prox_term + torch.sum((current - reference) ** 2)

        return prox_term

    def train_one_round(
        self,
        global_adapter_state: Dict[str, torch.Tensor],
        num_epochs: int,
    ):
        self.model.train()
        load_lora_state_dict(self.model, global_adapter_state)
        before_state = get_lora_state_dict(self.model)

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.criterion(logits, labels)

                if self.prox_mu > 0.0:
                    prox_term = self._compute_prox_term(before_state)
                    loss = loss + 0.5 * self.prox_mu * prox_term

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_examples += labels.size(0)

        after_state = get_lora_state_dict(self.model)
        delta = subtract_state_dicts(after_state, before_state)

        return {
            "client_id": self.client_id,
            "delta": delta,
            "num_examples": total_examples,
            "train_loss": total_loss / total_examples,
            "train_accuracy": total_correct / total_examples,
        }