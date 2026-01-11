# ST-GNN Real Data Training Report

**Generated**: 2026-01-11T00:10:24.968827
**Data Source**: REAL Bayesian outputs (stress_map.geojson, uncertainty_map.geojson)

## Dataset

| Metric | Value |
|--------|-------|
| Nodes | 10000 |
| Timesteps | 50 |
| Train timesteps | 35 |
| Val timesteps | 15 |
| Features | 8 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_spatial_layers | 3 |
| num_temporal_layers | 1 |
| dropout | 0.1 |
| epochs | 50 |
| learning_rate | 0.001 |

## Results

| Metric | Value |
|--------|-------|
| Base RMSE | 0.051022 |
| **Hybrid RMSE** | **0.021071** |
| DL-only RMSE | 0.182180 |
| **Improvement** | **58.70%** |
| Training Time | 2376.8s |

## Constraints Check

- [x] Hybrid < Base: 0.021071 < 0.051022
- [x] DL-only > Hybrid: 0.182180 > 0.021071

---

**Trained on REAL data from Seattle urban drainage system.**
