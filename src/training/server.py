from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.geometry.metrics import summarize_geometry
from src.models.lora_utils import (
    add_delta_to_state_dict,
    get_lora_state_dict,
    load_lora_state_dict,
)
from src.utils.logging import append_jsonl


class FederatedServer:
    def __init__(
        self,
        model,
        clients: Dict[str, Any],
        aggregator,
        eval_loader,
        device: str,
        log_path: str,
        task_name: str,
    ):
        self.model = model
        self.clients = clients
        self.aggregator = aggregator
        self.eval_loader = eval_loader
        self.device = device
        self.log_path = log_path
        self.task_name = task_name.lower()
        self.criterion = nn.CrossEntropyLoss()

        self.global_adapter = get_lora_state_dict(self.model)

    def sample_clients(self, k: int) -> List[str]:
        return random.sample(list(self.clients.keys()), k)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        all_preds = []
        all_labels = []

        for batch in self.eval_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

            if self.task_name in {"mrpc", "qqp"}:
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())

        metrics = {
            "eval_loss": total_loss / total_examples,
            "eval_accuracy": total_correct / total_examples,
        }

        if self.task_name in {"mrpc", "qqp"}:
            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()
            metrics["eval_f1"] = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
            )

        return metrics

    def run(
        self,
        num_rounds: int,
        clients_per_round: int,
        local_epochs: int,
    ):
        for round_idx in range(num_rounds):
            selected = self.sample_clients(clients_per_round)
            updates = []

            for cid in selected:
                client = self.clients[cid]
                update = client.train_one_round(
                    global_adapter_state=self.global_adapter,
                    num_epochs=local_epochs,
                )
                updates.append(update)

            agg_out = self.aggregator(updates)
            aggregated_delta = agg_out["aggregated_delta"]

            self.global_adapter = add_delta_to_state_dict(self.global_adapter, aggregated_delta)
            load_lora_state_dict(self.model, self.global_adapter)

            geometry = summarize_geometry([u["delta"] for u in updates])
            eval_metrics = self.evaluate()

            log_entry = {
                "round": round_idx,
                "sampled_client_ids": selected,
                "num_clients_sampled": len(selected),
                "avg_train_loss": sum(u["train_loss"] for u in updates) / len(updates),
                "avg_train_acc": sum(u["train_accuracy"] for u in updates) / len(updates),
                "client_train_losses": [u["train_loss"] for u in updates],
                "client_train_accuracies": [u["train_accuracy"] for u in updates],
                "aggregation_weights": agg_out["weights"].detach().cpu().tolist(),
                **eval_metrics,
                **geometry,
            }

            if agg_out.get("scores") is not None:
                log_entry["aggregation_scores"] = agg_out["scores"].detach().cpu().tolist()
            if agg_out.get("raw_scores") is not None:
                log_entry["aggregation_raw_scores"] = agg_out["raw_scores"].detach().cpu().tolist()
            if agg_out.get("alignments") is not None:
                log_entry["client_alignments"] = agg_out["alignments"].detach().cpu().tolist()
            if agg_out.get("distances") is not None:
                log_entry["client_distances"] = agg_out["distances"].detach().cpu().tolist()
            if agg_out.get("norms") is not None:
                log_entry["client_norms"] = agg_out["norms"].detach().cpu().tolist()
            if agg_out.get("avg_cosines") is not None:
                log_entry["client_avg_cosines"] = agg_out["avg_cosines"].detach().cpu().tolist()

            append_jsonl(self.log_path, log_entry)

            message = (
                f"[Round {round_idx}] "
                f"train_loss={log_entry['avg_train_loss']:.4f} "
                f"train_acc={log_entry['avg_train_acc']:.4f} "
                f"eval_loss={log_entry['eval_loss']:.4f} "
                f"eval_acc={log_entry['eval_accuracy']:.4f} "
            )
            if "eval_f1" in log_entry:
                message += f"eval_f1={log_entry['eval_f1']:.4f} "
            message += f"mean_cos={log_entry['mean_pairwise_cosine']:.4f}"

            print(message)