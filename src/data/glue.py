from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


TASK_TO_INPUT_KEYS = {
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
}


@dataclass
class ClientData:
    client_id: str
    train_dataset: Dataset


class GlueTorchDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def load_glue_task_tokenized(task_name: str, model_name: str, max_length: int):
    if task_name not in TASK_TO_INPUT_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", task_name)

    key1, key2 = TASK_TO_INPUT_KEYS[task_name]

    def tokenize_fn(batch):
        if key2 is None:
            return tokenizer(
                batch[key1],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        return tokenizer(
            batch[key1],
            batch[key2],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    return tokenized


def _random_partition_indices(
    labels: np.ndarray,
    num_clients: int,
    train_subset_per_client: int,
    seed: int,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    total_needed = num_clients * train_subset_per_client
    if total_needed > len(indices):
        raise ValueError(
            f"Requested {total_needed} train examples, but only {len(indices)} available."
        )

    selected = indices[:total_needed]
    client_indices = []

    for i in range(num_clients):
        start = i * train_subset_per_client
        end = start + train_subset_per_client
        client_indices.append(selected[start:end].tolist())

    return client_indices


def _label_skew_partition_indices(
    labels: np.ndarray,
    num_clients: int,
    train_subset_per_client: int,
    dominant_label_ratio: float,
    seed: int,
) -> List[List[int]]:
    if not (0.5 <= dominant_label_ratio <= 1.0):
        raise ValueError("dominant_label_ratio must be in [0.5, 1.0].")

    rng = np.random.default_rng(seed)

    unique_labels = sorted(np.unique(labels).tolist())
    if unique_labels != [0, 1]:
        raise ValueError("Current label_skew partition only supports binary-label tasks.")

    label0_indices = np.where(labels == 0)[0]
    label1_indices = np.where(labels == 1)[0]

    rng.shuffle(label0_indices)
    rng.shuffle(label1_indices)

    ptr0 = 0
    ptr1 = 0
    client_indices = []

    for client_idx in range(num_clients):
        dominant_label = client_idx % 2
        dominant_count = int(round(train_subset_per_client * dominant_label_ratio))
        minority_count = train_subset_per_client - dominant_count

        if dominant_label == 0:
            if ptr0 + dominant_count > len(label0_indices) or ptr1 + minority_count > len(label1_indices):
                raise ValueError("Not enough examples to create label-skewed partition with requested settings.")
            chosen = list(label0_indices[ptr0:ptr0 + dominant_count]) + list(label1_indices[ptr1:ptr1 + minority_count])
            ptr0 += dominant_count
            ptr1 += minority_count
        else:
            if ptr1 + dominant_count > len(label1_indices) or ptr0 + minority_count > len(label0_indices):
                raise ValueError("Not enough examples to create label-skewed partition with requested settings.")
            chosen = list(label1_indices[ptr1:ptr1 + dominant_count]) + list(label0_indices[ptr0:ptr0 + minority_count])
            ptr1 += dominant_count
            ptr0 += minority_count

        rng.shuffle(chosen)
        client_indices.append(chosen)

    return client_indices


def build_client_subsets(
    tokenized_train,
    num_clients: int,
    train_subset_per_client: int,
    seed: int,
    partition_mode: str = "random",
    dominant_label_ratio: float = 0.9,
) -> List[ClientData]:
    labels = np.array(tokenized_train["label"])

    if partition_mode == "random":
        partitioned_indices = _random_partition_indices(
            labels=labels,
            num_clients=num_clients,
            train_subset_per_client=train_subset_per_client,
            seed=seed,
        )
    elif partition_mode == "label_skew":
        partitioned_indices = _label_skew_partition_indices(
            labels=labels,
            num_clients=num_clients,
            train_subset_per_client=train_subset_per_client,
            dominant_label_ratio=dominant_label_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode}")

    clients: List[ClientData] = []
    for i, indices in enumerate(partitioned_indices):
        client_subset = tokenized_train.select(indices)
        clients.append(
            ClientData(
                client_id=f"client_{i}",
                train_dataset=GlueTorchDataset(client_subset),
            )
        )

    return clients


def build_eval_dataset(tokenized_eval, eval_subset: int | None = None):
    if eval_subset is not None:
        eval_subset = min(eval_subset, len(tokenized_eval))
        tokenized_eval = tokenized_eval.select(list(range(eval_subset)))
    return GlueTorchDataset(tokenized_eval)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_federated_glue(
    task_name: str,
    model_name: str,
    max_length: int,
    num_clients: int,
    train_subset_per_client: int,
    eval_subset: int | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    partition_mode: str = "random",
    dominant_label_ratio: float = 0.9,
):
    tokenized = load_glue_task_tokenized(
        task_name=task_name,
        model_name=model_name,
        max_length=max_length,
    )

    train_clients = build_client_subsets(
        tokenized_train=tokenized["train"],
        num_clients=num_clients,
        train_subset_per_client=train_subset_per_client,
        seed=seed,
        partition_mode=partition_mode,
        dominant_label_ratio=dominant_label_ratio,
    )

    eval_dataset = build_eval_dataset(
        tokenized_eval=tokenized["validation"],
        eval_subset=eval_subset,
    )

    client_loaders = {
        client.client_id: make_dataloader(
            client.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for client in train_clients
    }

    eval_loader = make_dataloader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return client_loaders, eval_loader
