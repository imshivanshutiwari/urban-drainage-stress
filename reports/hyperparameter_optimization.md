# Hyperparameter Optimization Report

**Generated**: 2026-01-10T23:15:59.099741
**Total Time**: 2664.8s

## Final Best Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_spatial_layers | 3 |
| num_temporal_layers | 1 |
| dropout | 0.1 |
| weight_decay | 0.001 |
| alpha (MSE) | 2.0 |
| beta (NLL) | 0.3 |
| gamma (smooth) | 0.2 |
| learning_rate | 0.001 |
| epochs | 50 |

## Final Metrics

| Metric | Value |
|--------|-------|
| Hybrid RMSE | 0.0205 |
| Base RMSE | 0.0296 |
| DL-only RMSE | 0.1646 |
| **Improvement** | **30.6%** |

## Constraints Check

- [x] Hybrid < Base: 0.0205 < 0.0296
- [x] DL-only > Hybrid: 0.1646 > 0.0205
- [x] Constraints satisfied: True

## Artifacts

- checkpoints/st_gnn_best_optimized.pt
- results/capacity_scaling_results.json
- results/temporal_depth_results.json
- results/regularization_results.json
- results/loss_weight_results.json
- results/final_training_summary.json

---

**This is the maximally optimized model under scientific constraints.**
