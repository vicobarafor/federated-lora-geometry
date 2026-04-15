import torch
import torch.nn.functional as F

from src.geometry.metrics import (
    alignment_to_mean,
    pairwise_cosine_matrix,
    stack_flattened_deltas,
)
from src.models.lora_utils import average_deltas


def _safe_zscore(x: torch.Tensor) -> torch.Tensor:
    std = x.std()
    if x.numel() <= 1 or std.item() < 1e-12:
        return torch.zeros_like(x)
    return (x - x.mean()) / (std + 1e-12)


def _softmax(scores: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return F.softmax(scores / temperature, dim=0)


def _shrink_to_uniform(weights: torch.Tensor, beta_uniform: float) -> torch.Tensor:
    beta_uniform = float(beta_uniform)
    beta_uniform = min(max(beta_uniform, 0.0), 1.0)

    n = weights.numel()
    uniform = torch.full_like(weights, 1.0 / n)
    return (1.0 - beta_uniform) * weights + beta_uniform * uniform


def compute_fedgeo_scores(
    deltas,
    alpha_align: float = 1.0,
    alpha_cosine: float = 0.5,
    alpha_norm: float = 0.0,
):
    delta_matrix = stack_flattened_deltas(deltas)

    align = alignment_to_mean(delta_matrix)
    cos_mat = pairwise_cosine_matrix(delta_matrix)

    n = cos_mat.size(0)
    if n > 1:
        avg_cos = (cos_mat.sum(dim=1) - torch.diag(cos_mat)) / (n - 1)
    else:
        avg_cos = torch.ones(1, dtype=delta_matrix.dtype, device=delta_matrix.device)

    norms = torch.norm(delta_matrix, p=2, dim=1)

    align_z = _safe_zscore(align)
    avg_cos_z = _safe_zscore(avg_cos)
    norm_z = _safe_zscore(norms)

    scores = (
        alpha_align * align_z
        + alpha_cosine * avg_cos_z
        + alpha_norm * norm_z
    )

    return {
        "scores": scores,
        "alignments": align,
        "avg_cosines": avg_cos,
        "norms": norms,
    }


def fedgeo_score(
    client_updates,
    alpha_align: float = 1.0,
    alpha_cosine: float = 0.5,
    alpha_norm: float = 0.0,
    temperature: float = 2.0,
    epsilon: float = 0.3,
):
    deltas = [u["delta"] for u in client_updates]

    geom = compute_fedgeo_scores(
        deltas=deltas,
        alpha_align=alpha_align,
        alpha_cosine=alpha_cosine,
        alpha_norm=alpha_norm,
    )

    scores = geom["scores"]
    soft = _softmax(scores, temperature=temperature)

    epsilon = float(epsilon)
    epsilon = min(max(epsilon, 0.0), 1.0)
    n = soft.numel()
    uniform = torch.full_like(soft, 1.0 / n)
    weights = (1.0 - epsilon) * uniform + epsilon * soft

    aggregated_delta = average_deltas(deltas, weights.tolist())

    return {
        "aggregated_delta": aggregated_delta,
        "weights": weights,
        "scores": scores,
        "alignments": geom["alignments"],
        "distances": None,
        "norms": geom["norms"],
        "avg_cosines": geom["avg_cosines"],
    }


def fedgeo_hybrid(
    client_updates,
    alpha_align: float = 1.0,
    alpha_cosine: float = 0.5,
    alpha_norm: float = 0.0,
    temperature: float = 2.0,
    lambda_geo: float = 0.2,
):
    deltas = [u["delta"] for u in client_updates]

    n = len(deltas)
    uniform_weights = [1.0 / n] * n
    delta_avg = average_deltas(deltas, uniform_weights)

    geom = compute_fedgeo_scores(
        deltas=deltas,
        alpha_align=alpha_align,
        alpha_cosine=alpha_cosine,
        alpha_norm=alpha_norm,
    )

    geo_weights = _softmax(geom["scores"], temperature=temperature)
    delta_geo = average_deltas(deltas, geo_weights.tolist())

    lambda_geo = float(lambda_geo)
    lambda_geo = min(max(lambda_geo, 0.0), 1.0)

    final_delta = {}
    for k in delta_avg.keys():
        final_delta[k] = (1.0 - lambda_geo) * delta_avg[k] + lambda_geo * delta_geo[k]

    return {
        "aggregated_delta": final_delta,
        "weights": geo_weights,
        "scores": geom["scores"],
        "alignments": geom["alignments"],
        "distances": None,
        "norms": geom["norms"],
        "avg_cosines": geom["avg_cosines"],
    }


def fedgeo_hybrid_robust(
    client_updates,
    alpha_align: float = 1.0,
    alpha_cosine: float = 0.5,
    alpha_norm: float = 0.0,
    temperature: float = 2.0,
    lambda_geo: float = 0.2,
    score_clip: float = 1.0,
    beta_uniform: float = 0.3,
):
    """Robust FedGeo v5.

    Steps:
      1. compute geometry scores
      2. clip scores to reduce outlier dominance
      3. softmax to get geometry weights
      4. shrink geometry weights toward uniform
      5. blend FedAvg update with robust geometry update
    """

    deltas = [u["delta"] for u in client_updates]

    n = len(deltas)
    uniform_weights = torch.full((n,), 1.0 / n, dtype=torch.float32)

    # FedAvg component
    delta_avg = average_deltas(deltas, uniform_weights.tolist())

    # Geometry component
    geom = compute_fedgeo_scores(
        deltas=deltas,
        alpha_align=alpha_align,
        alpha_cosine=alpha_cosine,
        alpha_norm=alpha_norm,
    )

    raw_scores = geom["scores"]

    score_clip = max(float(score_clip), 1e-6)
    clipped_scores = torch.clamp(raw_scores, min=-score_clip, max=score_clip)

    geo_weights = _softmax(clipped_scores, temperature=temperature)
    robust_geo_weights = _shrink_to_uniform(geo_weights, beta_uniform=beta_uniform)

    delta_geo = average_deltas(deltas, robust_geo_weights.tolist())

    lambda_geo = float(lambda_geo)
    lambda_geo = min(max(lambda_geo, 0.0), 1.0)

    final_delta = {}
    for k in delta_avg.keys():
        final_delta[k] = (1.0 - lambda_geo) * delta_avg[k] + lambda_geo * delta_geo[k]

    return {
        "aggregated_delta": final_delta,
        "weights": robust_geo_weights,
        "scores": clipped_scores,
        "raw_scores": raw_scores,
        "alignments": geom["alignments"],
        "distances": None,
        "norms": geom["norms"],
        "avg_cosines": geom["avg_cosines"],
    }
