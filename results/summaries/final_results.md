# Final Results Summary

## Main conclusion
FedGeo v5b is the strongest geometry-aware method explored in this project so far.

## SST-2
| Method | Mean Accuracy | Std | Evidence |
|---|---:|---:|---|
| FedAvg | 0.8448 | 0.0136 | multi-seed |
| FedGeo v4 | 0.8463 | 0.0066 | multi-seed |
| FedGeo v5b | 0.8486 | 0.0068 | multi-seed |

## MRPC
| Method | Mean Accuracy | Std | Evidence |
|---|---:|---:|---|
| FedAvg | 0.6332 | 0.0873 | multi-seed |
| FedGeo v4 | 0.6176 | 0.1075 | multi-seed |
| FedGeo v5 | 0.7353 | - | single-run |
| FedGeo v5b | 0.7353 | - | single-run |

## QQP
| Method | Accuracy | Evidence |
|---|---:|---|
| FedAvg | 0.8050 | single-run |
| FedGeo v5b | 0.8050 | single-run |

## Key findings
- Raw geometry weighting was unstable.
- Hybrid geometry with FedAvg was much stronger than pure geometry weighting.
- Robust calibration with clipping and shrinkage produced the best overall method.
- FedGeo v5b is competitive across datasets and strongest on SST-2.
