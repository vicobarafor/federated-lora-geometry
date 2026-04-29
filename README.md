# When Federated LoRA Fails

Geometry dynamics and instability in federated parameter-efficient fine-tuning under heterogeneous data.

---

## Overview

This repository studies instability in federated LoRA under heterogeneous client distributions.

Key idea:

- Early geometry determines training outcome
- Alignment → good performance
- Conflict → failure

---

## Usage

Run:

python scripts/run_experiment.py --config configs/glue/<config>.yaml

---

## Results

### Main Comparison

![MRPC](results/plots/mrpc_comparison.png)
![SST2](results/plots/sst2_comparison.png)
![QQP](results/plots/qqp_comparison.png)

---

### Geometry Predicts Performance

![Alignment](results/plots/figure1_alignment_delta_vs_final_accuracy.png)
![Conflict](results/plots/figure2_max_conflict_vs_final_accuracy.png)
![Cosine](results/plots/figure5_cosine_delta_vs_final_accuracy.png)

---

## Performance Summary

| Task | Method | Accuracy |
|------|--------|----------|
| SST-2 | FedAvg | 0.8448 |
| SST-2 | FedGeo v5b | 0.8486 |
| MRPC | FedAvg | 0.6332 |
| MRPC | FedGeo v5b | 0.7353 |

---

## Reproducibility

1. Use configs in configs/
2. Fix seeds
3. Run multiple seeds

---

## Limitations

- High heterogeneity remains unstable
- Scaling increases variance

---

## Status

Active research codebase.