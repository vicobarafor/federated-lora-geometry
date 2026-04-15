from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.utils.seed import set_seed
from src.utils.io import make_run_dir, copy_file
from src.data.glue import build_federated_glue
from src.models.encoder import EncoderWrapper
from src.models.heads import ClassificationHead
from src.models.classifier import FederatedClassifier
from src.models.lora_utils import apply_lora
from src.training.client import FederatedClient
from src.training.server import FederatedServer
from src.aggregation.fedavg import fedavg
from src.aggregation.fedgeo import fedgeo_score, fedgeo_hybrid, fedgeo_hybrid_robust


def build_aggregator(cfg):
    agg_name = cfg["aggregation"]["name"]

    if agg_name == "fedavg":
        return fedavg

    if agg_name == "fedgeo":
        return partial(
            fedgeo_score,
            alpha_align=cfg["aggregation"].get("alpha_align", 1.0),
            alpha_cosine=cfg["aggregation"].get("alpha_cosine", 0.5),
            alpha_norm=cfg["aggregation"].get("alpha_norm", 0.0),
            temperature=cfg["aggregation"].get("temperature", 2.0),
            epsilon=cfg["aggregation"].get("epsilon", 0.3),
        )

    if agg_name == "fedgeo_hybrid":
        return partial(
            fedgeo_hybrid,
            alpha_align=cfg["aggregation"].get("alpha_align", 1.0),
            alpha_cosine=cfg["aggregation"].get("alpha_cosine", 0.5),
            alpha_norm=cfg["aggregation"].get("alpha_norm", 0.0),
            temperature=cfg["aggregation"].get("temperature", 2.0),
            lambda_geo=cfg["aggregation"].get("lambda_geo", 0.2),
        )

    if agg_name == "fedgeo_hybrid_robust":
        return partial(
            fedgeo_hybrid_robust,
            alpha_align=cfg["aggregation"].get("alpha_align", 1.0),
            alpha_cosine=cfg["aggregation"].get("alpha_cosine", 0.5),
            alpha_norm=cfg["aggregation"].get("alpha_norm", 0.0),
            temperature=cfg["aggregation"].get("temperature", 2.0),
            lambda_geo=cfg["aggregation"].get("lambda_geo", 0.2),
            score_clip=cfg["aggregation"].get("score_clip", 1.0),
            beta_uniform=cfg["aggregation"].get("beta_uniform", 0.3),
        )

    raise ValueError(f"Unknown aggregation method: {agg_name}")


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["runtime"]["seed"])
    device = cfg["runtime"]["device"]

    run_dir = make_run_dir(
        cfg["output_root"],
        cfg["experiment_name"],
    )
    copy_file(config_path, run_dir / "config.yaml")

    encoder = EncoderWrapper(
        model_name=cfg["model"]["name"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
    )
    encoder = apply_lora(encoder, **cfg["lora"])

    head = ClassificationHead(
        hidden_dim=encoder.hidden_size,
        num_labels=cfg["task"]["num_labels"],
    )

    model = FederatedClassifier(encoder=encoder, head=head).to(device)

    client_loaders, eval_loader = build_federated_glue(
        task_name=cfg["task"]["name"],
        model_name=cfg["model"]["name"],
        max_length=cfg["data"]["max_length"],
        num_clients=cfg["data"]["num_clients"],
        train_subset_per_client=cfg["data"]["train_subset_per_client"],
        eval_subset=cfg["data"]["eval_subset"],
        batch_size=cfg["optimization"]["batch_size"],
        num_workers=cfg["runtime"]["num_workers"],
        seed=cfg["runtime"]["seed"],
        partition_mode=cfg["data"].get("partition_mode", "random"),
        dominant_label_ratio=cfg["data"].get("dominant_label_ratio", 0.9),
    )

    clients = {
        cid: FederatedClient(
            client_id=cid,
            model=model,
            dataloader=loader,
            lr=cfg["optimization"]["lr"],
            device=device,
        cfg["aggregation"].get("prox_mu", 0.0,
            prox_mu=cfg["aggregation"].get("prox_mu", 0.0),
        ),
        )
        for cid, loader in client_loaders.items()
    }

    aggregator = build_aggregator(cfg)

    server = FederatedServer(
        model=model,
        clients=clients,
        aggregator=aggregator,
        eval_loader=eval_loader,
        device=device,
        log_path=run_dir / "metrics.jsonl",
    )

    server.run(
        num_rounds=cfg["federation"]["num_rounds"],
        clients_per_round=cfg["federation"]["clients_per_round"],
        local_epochs=cfg["federation"]["local_epochs"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
