# When Federated LoRA Fails

Research code for studying instability in federated LoRA under heterogeneous data distributions, introducing early-round update geometry as a predictive signal of failure and proposing FedGeoX, a geometry-aware aggregation method.

## Overview

This repository accompanies our study of **federated LoRA under heterogeneity**. We analyze how geometric properties of client updates evolve during training and show that **early-round geometry strongly predicts downstream performance and instability**.

In particular, we study:
- the relationship between cosine similarity, alignment, and negative update conflict
- performance degradation under increasing label skew
- variability across random seeds
- scaling behavior as the number of clients increases
- FedGeoX, a geometry-aware aggregation method for federated LoRA

## Main contributions

- A research codebase for federated LoRA under heterogeneous client data
- Analysis showing that early-round update geometry predicts final performance
- Controlled experiments across heterogeneity, seeds, and scaling regimes
- FedGeoX, a geometry-aware aggregation approach
- Scripts for summarization, plotting, and instability analysis

## Repository structure

```text
.
├── analysis/            # Additional analysis utilities
├── configs/             # Experiment configurations
├── docs/                # Notes on benchmarks, methods, and roadmap
├── notebooks/           # Colab and exploratory notebooks
├── results/
│   ├── plots/           # Figures used in the paper
│   ├── summaries/       # Compact summary CSVs
│   └── summary/         # Additional compact outputs
├── scripts/             # Experiment runners and analysis scripts
├── src/                 # Core source code
├── tests/               # Tests
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
