from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a parameter state dict into a single 1D CPU tensor."""
    flat_tensors = []
    for key in sorted(state_dict.keys()):
        flat_tensors.append(state_dict[key].detach().float().cpu().reshape(-1))
    return torch.cat(flat_tensors)


def stack_flattened_deltas(deltas: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Convert a list of adapter deltas into a matrix of shape [num_clients, dim]."""
    if not deltas:
        raise ValueError("No deltas provided.")
    return torch.stack([flatten_state_dict(delta) for delta in deltas], dim=0)


def pairwise_cosine_matrix(delta_matrix: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix between client updates."""
    normalized = F.normalize(delta_matrix, p=2, dim=1)
    return normalized @ normalized.T


def upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    """Return strictly upper-triangular values of a square matrix."""
    n = matrix.size(0)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    return matrix[mask]


def mean_update(delta_matrix: torch.Tensor) -> torch.Tensor:
    return delta_matrix.mean(dim=0)


def alignment_to_mean(delta_matrix: torch.Tensor) -> torch.Tensor:
    """Cosine alignment of each client update to the mean update."""
    mean_vec = mean_update(delta_matrix).unsqueeze(0)
    return F.cosine_similarity(delta_matrix, mean_vec, dim=1)


def update_norms(delta_matrix: torch.Tensor) -> torch.Tensor:
    return torch.norm(delta_matrix, p=2, dim=1)


def distances_to_centroid(delta_matrix: torch.Tensor) -> torch.Tensor:
    centroid = mean_update(delta_matrix).unsqueeze(0)
    return torch.norm(delta_matrix - centroid, p=2, dim=1)


def summarize_geometry(deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Compute round-level geometry summary statistics for client updates."""
    delta_matrix = stack_flattened_deltas(deltas)

    cos_matrix = pairwise_cosine_matrix(delta_matrix)
    pairwise_vals = upper_triangle_values(cos_matrix)

    alignments = alignment_to_mean(delta_matrix)
    norms = update_norms(delta_matrix)
    centroid_dists = distances_to_centroid(delta_matrix)

    summary = {
        "num_clients": float(delta_matrix.size(0)),
        "mean_pairwise_cosine": pairwise_vals.mean().item() if pairwise_vals.numel() > 0 else 1.0,
        "std_pairwise_cosine": pairwise_vals.std().item() if pairwise_vals.numel() > 1 else 0.0,
        "mean_alignment_to_mean": alignments.mean().item(),
        "std_alignment_to_mean": alignments.std().item() if alignments.numel() > 1 else 0.0,
        "mean_update_norm": norms.mean().item(),
        "std_update_norm": norms.std().item() if norms.numel() > 1 else 0.0,
        "mean_distance_to_centroid": centroid_dists.mean().item(),
        "std_distance_to_centroid": centroid_dists.std().item() if centroid_dists.numel() > 1 else 0.0,
    }
    return summary
