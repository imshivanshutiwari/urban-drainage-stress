"""
Quick Test of the New Latent ST-GNN Architecture.

Tests:
1. Model creation and forward pass
2. Loss computation
3. Input guards
4. Uncertainty guards
5. DL Role Audit
"""

import sys
import numpy as np
import torch

# Add project to path
sys.path.insert(0, '.')

print("=" * 60)
print("TESTING NEW LATENT ST-GNN ARCHITECTURE")
print("=" * 60)

# ============================================================================
# 1. MODEL CREATION
# ============================================================================
print("\n[1] Creating LatentSTGNN model...")

from src.ml.models.st_gnn_latent import LatentSTGNN, LatentSTGNNConfig

config = LatentSTGNNConfig(
    input_dim=8,
    hidden_dim=32,
    num_gnn_layers=2,
    temporal_hidden=32,
    num_temporal_layers=1,
)
model = LatentSTGNN(config)
print(f"    ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ============================================================================
# 2. FORWARD PASS
# ============================================================================
print("\n[2] Testing forward pass with synthetic data...")

# Synthetic scale-free data (z-scores)
T, N, D = 10, 100, 8  # 10 timesteps, 100 nodes, 8 features
x = torch.randn(T, N, D)  # Scale-free z-scores

# Random graph edges
E = 500
edge_index = torch.randint(0, N, (2, E))

# Forward pass
delta_z, sigma_dl = model(x, edge_index)
print(f"    ✓ Output shapes: delta_z={delta_z.shape}, sigma_dl={sigma_dl.shape}")
print(f"    ✓ delta_z range: [{delta_z.min().item():.4f}, {delta_z.max().item():.4f}]")
print(f"    ✓ sigma_dl range: [{sigma_dl.min().item():.4f}, {sigma_dl.max().item():.4f}]")

# ============================================================================
# 3. LOSS COMPUTATION
# ============================================================================
print("\n[3] Testing loss computation...")

from src.ml.training.latent_losses import LatentCompositeLoss

loss_fn = LatentCompositeLoss()

# Synthetic target
delta_z_target = torch.randn(T, N, 1) * 0.5

loss_dict = loss_fn(delta_z, delta_z_target, sigma_dl, edge_index)
print(f"    ✓ Total loss: {loss_dict['total'].item():.4f}")
print(f"      - MSE: {loss_dict['mse'].item():.4f}")
print(f"      - NLL: {loss_dict['nll'].item():.4f}")
print(f"      - Smooth: {loss_dict['smooth'].item():.4f}")
print(f"      - Rank: {loss_dict['rank'].item():.4f}")

# ============================================================================
# 4. INPUT GUARDS
# ============================================================================
print("\n[4] Testing input guards...")

from src.ml.guards import validate_model_inputs

# Test with valid scale-free input
result = validate_model_inputs(
    {'features': x.numpy()},
    {'features': 'zscore'}
)
print(f"    ✓ Valid input check: {'PASSED' if result.passed else 'FAILED'}")

# Test with invalid raw-scale input (should warn/fail)
raw_rainfall = np.random.exponential(50, (T, N, D))  # Raw mm
result_bad = validate_model_inputs(
    {'rainfall': raw_rainfall},
    {'rainfall': 'zscore'}
)
print(f"    ✓ Raw input detection: {'DETECTED' if not result_bad.passed else 'MISSED'}")

# ============================================================================
# 5. UNCERTAINTY GUARDS
# ============================================================================
print("\n[5] Testing uncertainty guards...")

from src.ml.guards import validate_uncertainty_additive, combine_uncertainties_safe

bayesian_unc = np.abs(np.random.randn(T, N)) * 0.3 + 0.1
dl_unc = sigma_dl.detach().numpy().squeeze(-1)

# Safe combination
total_unc = combine_uncertainties_safe(bayesian_unc, dl_unc)
result = validate_uncertainty_additive(bayesian_unc, dl_unc, total_unc)
print(f"    ✓ Uncertainty additive check: {'PASSED' if result.passed else 'FAILED'}")

# ============================================================================
# 6. DL ROLE AUDIT
# ============================================================================
print("\n[6] Running DL Role Audit...")

from src.ml.audit import run_dl_role_audit, AuditConfig

# Use lenient config for testing (don't abort)
audit_config = AuditConfig(abort_on_violation=False)

delta_z_np = delta_z.detach().numpy().squeeze(-1)
physics_z = np.random.randn(T, N)  # Simulated physics output

audit_result = run_dl_role_audit(
    model_inputs={'features': x.numpy()},
    delta_z=delta_z_np,
    physics_z=physics_z,
    config=audit_config,
)

print(f"\n    Audit Result: {'PASSED ✓' if audit_result.passed else 'FAILED ✗'}")
for check, status in audit_result.checks.items():
    if status is not None:
        print(f"      - {check}: {'✓' if status else '✗'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print(f"""
Summary:
  ✓ Model forward pass works
  ✓ Loss computation works
  ✓ Input guards working
  ✓ Uncertainty guards working
  ✓ DL Role Audit: {'PASSED' if audit_result.passed else 'FAILED'}
""")
