# Urban Drainage Stress Inference System
## Complete Technical Documentation (From Scratch to Deployment)

**Author:** Shivanshu Tiwari  
**Date:** January 2026  
**Status:** ✅ Production Ready  

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Pipeline](#4-data-pipeline)
5. [Bayesian Inference Engine](#5-bayesian-inference-engine)
6. [Deep Learning Component](#6-deep-learning-component)
7. [Training Methodology](#7-training-methodology)
8. [Safety & Constraints](#8-safety--constraints)
9. [Deployment Pipeline](#9-deployment-pipeline)
10. [Results & Metrics](#10-results--metrics)
11. [Lessons Learned](#11-lessons-learned)
12. [Technical Glossary](#12-technical-glossary)

---

# 1. Executive Summary

## What This Project Does
This system predicts **urban drainage stress** (risk of flooding/pipe failure) by combining:
- **Physics-based Bayesian inference** (17 mathematical modules)
- **Deep Learning residual correction** (Latent ST-GNN)
- **Uncertainty-aware decision making** (refuses to decide when unsure)

## Key Innovation
The DL model is a **Structural Residual Learner** that:
- Only corrects 2% of the physics prediction (supportive, not dominant)
- Is 420x more uncertain on unseen cities (OOD detection)
- Never makes final decisions (only adjusts beliefs)

## Final Results
| Metric                    | Value       |
| ------------------------- | ----------- |
| OOD Calibrated            | ✅ True      |
| NYC > Seattle Uncertainty | 420x higher |
| DL Fraction               | 2.17%       |
| All Audits                | ✅ Passed    |

---

# 2. Problem Statement

## The Urban Flooding Challenge

Urban drainage systems are complex networks of pipes, channels, and storage facilities designed to manage stormwater. When these systems fail, the consequences include:
- Property damage
- Traffic disruption
- Public health hazards
- Environmental contamination

## Why Prediction is Hard

1. **Hidden Infrastructure**: Most pipes are underground and unmapped
2. **Non-linear Dynamics**: Small changes in rainfall can cause large stress spikes
3. **Spatial Correlation**: Failure in one area affects downstream areas
4. **Temporal Evolution**: Stress accumulates over time during storms
5. **Data Scarcity**: Few sensors, incomplete records

## Our Approach

Instead of predicting exact flood locations, we infer **latent stress** - a hidden variable representing the probability of system failure at each location.

```
Stress(x, t) = f(Rainfall, Terrain, Complaints, Infrastructure, ...)
```

---

# 3. System Architecture Overview

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA SOURCES                           │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│  Rainfall   │   Terrain   │  Complaints │  Historical Events  │
│   (mm/hr)   │    (DEM)    │   (311)     │   (Past Floods)     │
└──────┬──────┴──────┬──────┴──────┬──────┴──────────┬──────────┘
       │             │             │                 │
       ▼             ▼             ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              BAYESIAN INFERENCE ENGINE (17 Modules)             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Capacity │ │  Flow   │ │Upstream │ │Evidence │ │  Prior  │   │
│  │ Stress  │ │Direction│ │Accum.   │ │ Fusion  │ │ Update  │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       └──────────┬┴──────────┬┴──────────┬┴──────────┬┘        │
│                  ▼           ▼           ▼           ▼          │
│              ┌─────────────────────────────────────────┐        │
│              │     σ_Bayes (Physics Uncertainty)       │        │
│              │     Z_physics (Latent Stress)           │        │
│              └──────────────────┬──────────────────────┘        │
└─────────────────────────────────┼───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              DEEP LEARNING CORRECTION (Latent ST-GNN)           │
│                                                                 │
│   Input: Z-scored features (NEVER raw values)                   │
│                                                                 │
│   ┌──────────────────┐    ┌──────────────────┐                  │
│   │  Spatial GNN     │───▶│  Temporal GRU    │                  │
│   │  (Graph Conv)    │    │  (Causal Attn)   │                  │
│   └──────────────────┘    └────────┬─────────┘                  │
│                                    │                            │
│                    ┌───────────────┼───────────────┐            │
│                    ▼               ▼               │            │
│              ┌──────────┐   ┌──────────────┐      │            │
│              │   ΔZ     │   │   σ_DL       │      │            │
│              │ Residual │   │ Uncertainty  │      │            │
│              └────┬─────┘   └──────┬───────┘      │            │
└───────────────────┼────────────────┼──────────────┘            │
                    │                │                            │
                    ▼                ▼                            │
┌─────────────────────────────────────────────────────────────────┐
│              FUSION & DECISION LAYER                            │
│                                                                 │
│   Z_final = Z_physics + ΔZ  (latent space addition)             │
│   σ_final = √(σ_Bayes² + σ_DL²)  (uncertainty inflation)        │
│                                                                 │
│   IF σ_final > threshold:                                       │
│       Decision = NO_DECISION (refuse to guess)                  │
│   ELSE:                                                         │
│       Decision = Threshold(Z_final)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

# 4. Data Pipeline

## 4.1 Input Data Sources

### Rainfall Data
- **Source**: Weather stations, radar (NEXRAD)
- **Resolution**: Hourly, 1km grid
- **Units**: mm/hour
- **Processing**: Spatial interpolation (Kriging)

### Terrain Data (DEM)
- **Source**: USGS, LiDAR
- **Resolution**: 10m
- **Key Derivatives**:
  - Slope (gradient magnitude)
  - Aspect (gradient direction)
  - Upstream contributing area (flow accumulation)
  - Topographic Wetness Index (TWI)

### Complaint Data (311)
- **Source**: City 311 system
- **Types**: Flooding, ponding, sewer backup
- **Processing**: Kernel density estimation

### Infrastructure Data
- **Source**: City GIS, utility records
- **Features**: Pipe diameter, age, material, capacity

## 4.2 Feature Engineering

### Z-Score Normalization (CRITICAL)
The DL model NEVER sees raw values. Everything is Z-scored:

```python
# Z-score formula
z = (x - μ) / σ

# Where:
# x = raw value
# μ = mean of training data
# σ = standard deviation of training data
```

**Why Z-scoring matters:**
1. Makes features comparable across cities
2. Prevents the model from learning city-specific magnitudes
3. Enables transfer learning to new cities

### Temporal Features
```python
# Temporal gradients
∂Z/∂t = (Z[t] - Z[t-1]) / Δt

# Cumulative stress
Z_cumulative = Σ Z[t'] for t' < t
```

### Spatial Features
```python
# Spatial gradients
∂Z/∂x, ∂Z/∂y = np.gradient(Z)

# Upstream accumulation
upstream_stress = flow_accumulation(Z, DEM)
```

---

# 5. Bayesian Inference Engine

## 5.1 The 17 Core Modules

The physics-based inference engine contains 17 mathematical modules:

| #   | Module                   | Purpose                 | Key Equation                       |
| --- | ------------------------ | ----------------------- | ---------------------------------- |
| 1   | Capacity Stress          | Pipe capacity vs demand | S = Q_demand / Q_capacity          |
| 2   | Flow Accumulation        | Upstream catchment      | A = Σ upslope_area                 |
| 3   | Antecedent Moisture      | Soil wetness            | AMC = exp_decay(past_rainfall)     |
| 4   | Evidence Fusion          | Combine observations    | P(H                                | E) ∝ P(E   | H)·P(H) |
| 5   | Prior Update             | Bayesian update         | posterior = likelihood × prior     |
| 6   | Spatial Smoothing        | Reduce noise            | Z_smooth = gaussian_filter(Z)      |
| 7   | Temporal Decay           | Memory effects          | Z[t] = α·Z[t-1] + (1-α)·new        |
| 8   | Anisotropy               | Directional flow        | principal_axes(gradient_tensor)    |
| 9   | Hierarchical Aggregation | Multi-scale             | Z_region = mean(Z_cells)           |
| 10  | Complaint Integration    | Citizen reports         | P(stress                           | complaint) |
| 11  | Infrastructure Aging     | Pipe degradation        | capacity × (1 - age_factor)        |
| 12  | Seasonal Adjustment      | Winter/summer patterns  | seasonal_coefficient               |
| 13  | Storm Event Detection    | Peak identification     | local_maxima(rainfall)             |
| 14  | Recovery Modeling        | Post-storm decay        | exponential_recovery               |
| 15  | Uncertainty Propagation  | Error tracking          | σ_out = f(σ_in, model_uncertainty) |
| 16  | Tail Risk                | Extreme events          | EVT distribution fitting           |
| 17  | Calibration              | Reality check           | compare_to_observations            |

## 5.2 Bayes' Theorem

The core of probabilistic inference:

```
P(Stress | Observations) = P(Observations | Stress) × P(Stress)
                          ────────────────────────────────────────
                                    P(Observations)
```

In our notation:
- **P(Stress | Observations)**: Posterior - what we want
- **P(Observations | Stress)**: Likelihood - how well observations match hypothesis
- **P(Stress)**: Prior - what we believed before
- **P(Observations)**: Evidence - normalizing constant

## 5.3 Gaussian Process Regression

For spatial interpolation:

```python
# GP mean prediction
μ* = K(X*, X) × K(X, X)^(-1) × y

# GP variance (uncertainty)
σ²* = K(X*, X*) - K(X*, X) × K(X, X)^(-1) × K(X, X*)

# Kernel function (RBF)
K(x, x') = σ² × exp(-||x - x'||² / (2 × l²))
```

**Key parameters:**
- `σ²`: Signal variance (how much the function varies)
- `l`: Length scale (how far correlations extend)

---

# 6. Deep Learning Component

## 6.1 Architecture: Latent Spatio-Temporal GNN

### Why This Architecture?

| Challenge            | Solution                   |
| -------------------- | -------------------------- |
| Spatial correlations | Graph Neural Network       |
| Temporal dynamics    | GRU + Causal Attention     |
| Scale invariance     | Z-score inputs only        |
| OOD detection        | Distance-aware uncertainty |

### Model Components

```python
class LatentSTGNN(nn.Module):
    def __init__(self, config):
        # 1. Spatial Encoder (Graph Convolution)
        self.spatial_encoder = SpatialGNNEncoder(config)
        
        # 2. Temporal Encoder (GRU + Attention)
        self.temporal_encoder = CausalTemporalEncoder(config)
        
        # 3. Output Heads (ONLY 2 ALLOWED)
        self.residual_head = StructuralResidualHead(...)  # Predicts ΔZ
        self.uncertainty_head = EpistemicUncertaintyHead(...)  # Predicts σ_DL
```

## 6.2 Spatial GNN Encoder

### Graph Convolution Operation

```python
# Message passing
h_i^(l+1) = σ(W × Σ (1/√(d_i × d_j)) × h_j^(l) + b)

# Where:
# h_i: Node i's hidden state
# W: Learnable weight matrix
# d_i, d_j: Node degrees
# σ: Activation function (LeakyReLU)
```

### Why Graph Structure Matters

Urban drainage naturally forms a graph:
- **Nodes**: Grid cells or pipe junctions
- **Edges**: Flow connections (based on DEM or pipe network)
- **Edge weights**: Flow capacity or distance

```python
# k-NN graph construction
from scipy.spatial import cKDTree
tree = cKDTree(coordinates)
_, neighbors = tree.query(coordinates, k=8)  # 8 nearest neighbors

edge_index = []
for i, neigh in enumerate(neighbors):
    for j in neigh[1:]:  # Skip self
        edge_index.append([i, j])
```

## 6.3 Temporal Encoder

### Gated Recurrent Unit (GRU)

```python
# GRU equations
z_t = σ(W_z × [h_{t-1}, x_t])  # Update gate
r_t = σ(W_r × [h_{t-1}, x_t])  # Reset gate
h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])  # Candidate
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t  # Output
```

**Why GRU over LSTM?**
- Fewer parameters (25% smaller)
- Similar performance for our use case
- Faster training

### Causal Attention (No Future Leakage)

```python
# Causal mask ensures no future information leaks
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
attention_weights = attention_weights.masked_fill(mask, -inf)
```

**Critical for temporal forecasting**: The model must not "see" future values during training or inference.

## 6.4 Output Heads

### Residual Head (ΔZ Prediction)

```python
class StructuralResidualHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize for small outputs
        nn.init.normal_(self.net[-1].weight, std=0.1)
```

**Purpose**: Predict the difference between physics prediction and reality.

### Uncertainty Head (σ_DL Prediction)

```python
class EpistemicUncertaintyHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        self.base_net = nn.Sequential(...)
        self.distance_scale = nn.Parameter(torch.tensor(0.1))
        
        # Store training distribution for OOD detection
        self.register_buffer('train_mean', torch.zeros(input_dim))
        self.register_buffer('train_std', torch.ones(input_dim))
        
    def forward(self, x):
        base_sigma = F.softplus(self.base_net(x))
        
        # OOD detection: distance from training distribution
        z_score = (x - self.train_mean) / self.train_std
        distance = torch.sqrt((z_score ** 2).sum(dim=-1, keepdim=True))
        distance_sigma = F.softplus(self.distance_scale) * distance
        
        return base_sigma + distance_sigma + 0.01
```

**Key Innovation**: The uncertainty head stores the training distribution statistics and outputs higher uncertainty for inputs that are "far" from training data.

---

# 7. Training Methodology

## 7.1 Loss Function

### Composite Loss

```python
L_total = α × L_MSE + β × L_NLL + γ × L_smooth + η × L_rank

# Where:
# α = 10.0 (MSE weight - primary)
# β = 0.2  (NLL weight - reduced to avoid zero-collapse)
# γ = 0.1  (Smoothness weight)
# η = 0.2  (Rank preservation weight)
```

### Individual Loss Components

**1. MSE Loss (Scale-Free)**
```python
L_MSE = mean((ΔZ_pred - ΔZ_target)²)
```

**2. Negative Log-Likelihood (Calibrated Uncertainty)**
```python
L_NLL = 0.5 × mean(log(σ²) + (ΔZ_pred - ΔZ_target)² / σ²)
```

This loss encourages the model to:
- Predict larger σ when it makes errors
- Predict smaller σ when it's confident

**3. Graph Smoothness (Spatial Coherence)**
```python
L_smooth = mean((ΔZ_i - ΔZ_j)² for all edges (i,j))
```

Neighboring nodes should have similar predictions.

**4. Rank Preservation (Relative Ordering)**
```python
L_rank = -correlation(rank(ΔZ_pred), rank(ΔZ_target))
```

Even if magnitudes are wrong, preserve the ordering.

## 7.2 Training Configuration

```python
# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training details
epochs = 500
batch_size = 1  # Full graph per batch
gradient_clip = 1.0
```

## 7.3 Data Split (Temporal, NOT Random)

```python
# CRITICAL: Temporal split, no shuffling
num_timesteps = 50
train_idx = 40  # 80% train
val_idx = 10    # 20% val

train_data = data[:train_idx]
val_data = data[train_idx:]
```

**Why temporal split?**
- Prevents data leakage
- Simulates real-world deployment (predict future from past)
- Forces model to learn temporal patterns

## 7.4 OOD Calibration

After training, we calibrate the uncertainty head:

```python
# Store training distribution statistics
model.calibrate_ood(train_features, edge_index)

# This runs a forward pass and stores:
# - train_mean: mean of intermediate representations
# - train_std: std of intermediate representations
# - is_calibrated: True
```

---

# 8. Safety & Constraints

## 8.1 The Golden Rules

### Rule 1: DL is Supportive, Not Dominant
```python
# CONSTRAINT: DL contributes < 5% of final prediction
dl_fraction = abs(delta_z.mean()) / abs(physics_prediction.mean())
assert dl_fraction < 0.05, "DL is too dominant!"

# Actual result: 2.17% ✓
```

### Rule 2: Higher Uncertainty on OOD Data
```python
# CONSTRAINT: NYC uncertainty > Seattle uncertainty
assert nyc_uncertainty > seattle_uncertainty, "OOD detection failed!"

# Actual result: 420x higher ✓
```

### Rule 3: No Raw Scale Inputs
```python
# CONSTRAINT: All inputs must be Z-scored
def validate_inputs(x):
    assert abs(x.mean()) < 0.1, "Input not centered!"
    assert abs(x.std() - 1.0) < 0.1, "Input not normalized!"
```

### Rule 4: DL Never Makes Decisions
```python
# DL outputs are ALWAYS fused with physics
# Final decision is made by the decision engine, not DL
decision = decision_engine(physics_stress, dl_residual, physics_uncertainty, dl_uncertainty)
```

## 8.2 Guards & Audits

### Input Sanity Checks
```python
class InputSanityChecker:
    def check(self, x):
        # 1. Check for NaN/Inf
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        
        # 2. Check scale-free (Z-scored)
        assert abs(x.mean()) < 0.5
        assert 0.5 < x.std() < 2.0
        
        # 3. Check shape
        assert x.dim() == 3  # [T, N, D]
```

### Uncertainty Guards
```python
class UncertaintyGuard:
    def validate(self, sigma_dl, sigma_bayes):
        # DL uncertainty must be additive (never reduce total)
        sigma_total = torch.sqrt(sigma_bayes**2 + sigma_dl**2)
        assert (sigma_total >= sigma_bayes).all()
```

### DL Role Audit
```python
class DLRoleAudit:
    def run(self, model, data):
        checks = {
            'no_raw_scale_input': self._check_no_raw_scale(data),
            'bounded_latent_corrections': self._check_bounded(model, data),
            'no_dl_dominance': self._check_not_dominant(model, data),
        }
        return all(checks.values()), checks
```

---

# 9. Deployment Pipeline

## 9.1 Universal Deployment Pipeline

```python
class UniversalDeploymentPipeline:
    def run(self, raw_stress, city_name, sigma_bayes=None, sigma_dl=None):
        # Step 1: Latent Transform
        Z = self.latent_transform.to_latent(raw_stress)
        
        # Step 2: Shift Detection (is this OOD?)
        shift_score = self.shift_detector.detect(Z)
        
        # Step 3: Uncertainty Inflation
        sigma_final = self.uncertainty_inflator.inflate(
            sigma_bayes, sigma_dl, shift_score
        )
        
        # Step 4: Relative Decision (compare to reference)
        decision = self.decision_engine.decide(Z, sigma_final)
        
        return decision
```

## 9.2 Cross-City Transfer

The key innovation is that the model can transfer to new cities:

```python
# Train on Seattle
model = train(seattle_data)

# Deploy on NYC (never seen before)
nyc_prediction, nyc_uncertainty = model(nyc_data)

# Result: NYC uncertainty is 420x higher
# Model correctly says "I don't know" for NYC
```

## 9.3 Decision Logic

```python
def make_decision(stress, uncertainty, threshold=0.5):
    # High uncertainty -> refuse to decide
    if uncertainty > threshold:
        return "NO_DECISION"
    
    # Low stress -> low risk
    if stress < 0.3:
        return "LOW_RISK"
    
    # Medium stress -> medium risk
    if stress < 0.7:
        return "MEDIUM_RISK"
    
    # High stress -> high risk
    return "HIGH_RISK"
```

---

# 10. Results & Metrics

## 10.1 Final Training Results

| Metric         | Value | Meaning                     |
| -------------- | ----- | --------------------------- |
| Best Epoch     | 19    | Training converged quickly  |
| Best Val Loss  | -0.70 | Negative due to NLL term    |
| Final RMSE     | 0.028 | Low prediction error        |
| R² Score       | 0.18  | Modest but expected         |
| OOD Calibrated | True  | Uncertainty head calibrated |

## 10.2 Cross-City Uncertainty

| City               | Mean Uncertainty | Status             |
| ------------------ | ---------------- | ------------------ |
| Seattle (training) | 0.01             | Low - confident    |
| NYC (never seen)   | 4.2              | High - uncertain   |
| **Ratio**          | **420x**         | **✓ OOD detected** |

## 10.3 DL Contribution Analysis

```
Physics Magnitude:  0.8126 (98%)
DL Magnitude:       0.0177 (2%)
DL Fraction:        2.17%
```

The DL model makes **small corrections** to the physics baseline, exactly as designed.

## 10.4 Audit Results

| Check                      | Status |
| -------------------------- | ------ |
| No Raw Scale Input         | ✓ PASS |
| Bounded Latent Corrections | ✓ PASS |
| No DL Dominance            | ✓ PASS |
| All Audits                 | ✓ PASS |

---

# 11. Lessons Learned

## 11.1 Technical Lessons

### Lesson 1: NLL Loss Can Cause Zero-Collapse
**Problem**: Model outputs near-zero predictions with low uncertainty.
**Solution**: Reduce NLL weight (β) and increase MSE weight (α).

### Lesson 2: Spectral Normalization Alone Doesn't Fix OOD
**Problem**: Spectral norm constrains Lipschitz, but doesn't penalize OOD confidence.
**Solution**: Add distance-aware uncertainty that explicitly measures deviation from training distribution.

### Lesson 3: Weight Initialization Matters
**Problem**: Small weight init (std=0.01) causes vanishing gradients.
**Solution**: Increase to std=0.1 for faster learning.

### Lesson 4: Calibration Must Use Intermediate Features
**Problem**: Calibrating on raw inputs (dim=8) fails when head expects encoded features (dim=64).
**Solution**: Run forward pass to get intermediate embeddings, then calibrate.

## 11.2 Process Lessons

### Lesson 5: Local Training Can Be Slow
**Problem**: Training on CPU hangs the system.
**Solution**: Use Kaggle/Colab for GPU training.

### Lesson 6: Always Exclude Models from Git
**Problem**: Large .pt files slow down pushes and expose proprietary work.
**Solution**: Add `*.pt` to .gitignore.

### Lesson 7: Test on OOD Data Early
**Problem**: Discovered NYC uncertainty issue late in development.
**Solution**: Include OOD tests in the training script itself.

---

# 12. Technical Glossary

## A

**AdamW**: Adam optimizer with decoupled weight decay. Better than vanilla Adam for deep learning.

**Anisotropy**: Property of being directionally dependent. Flow in drainage is anisotropic (water flows downhill).

**Antecedent Moisture Condition (AMC)**: How wet the soil is before a storm. Affects how much runoff occurs.

## B

**Bayes' Theorem**: P(A|B) = P(B|A)×P(A)/P(B). Foundation of probabilistic inference.

**Batch Normalization**: Normalizes activations within a batch. Helps training stability.

## C

**Causal**: Respecting time ordering. Causal models don't use future information.

**Cross-Entropy**: Loss function for classification. -Σ y×log(p).

## D

**DEM (Digital Elevation Model)**: Grid of elevation values. Used to compute flow direction.

**Dropout**: Randomly zeroes neurons during training. Prevents overfitting.

## E

**Epistemic Uncertainty**: Uncertainty due to lack of knowledge (reducible with more data).

**Evidence Lower Bound (ELBO)**: Objective for variational inference.

## F

**Flow Accumulation**: Count of upstream cells that flow through each cell.

**F1 Score**: Harmonic mean of precision and recall. 2×P×R/(P+R).

## G

**Gaussian Process (GP)**: Non-parametric model that defines a distribution over functions.

**GRU (Gated Recurrent Unit)**: Simpler alternative to LSTM for sequence modeling.

**Graph Neural Network (GNN)**: Neural network that operates on graph-structured data.

## H

**Heteroscedastic**: Having non-constant variance. Our model predicts heteroscedastic uncertainty.

**Hyperparameter**: Parameter not learned during training (e.g., learning rate, hidden size).

## I

**Inference**: Using a trained model to make predictions on new data.

## K

**Kernel**: Function that measures similarity between data points. K(x, x').

**Kriging**: Gaussian process regression for spatial interpolation.

## L

**Latent Space**: Hidden representation space where the model operates.

**LeakyReLU**: Activation function. max(0.01×x, x). Prevents dead neurons.

**Lipschitz Constant**: Maximum rate of change of a function. Important for stability.

**Loss Function**: Objective that the model minimizes during training.

## M

**Message Passing**: GNN operation where nodes exchange information with neighbors.

**MSE (Mean Squared Error)**: Average squared difference between prediction and target.

## N

**NLL (Negative Log-Likelihood)**: Loss that measures how well uncertainty is calibrated.

**Normalization**: Scaling data to a standard range (e.g., Z-score, min-max).

## O

**OOD (Out-of-Distribution)**: Data that differs from training distribution. Must detect!

**Optuna**: Hyperparameter optimization framework.

**Overfitting**: Model memorizes training data but fails on new data.

## P

**Posterior**: Updated belief after seeing evidence. P(H|E).

**Prior**: Initial belief before seeing evidence. P(H).

## R

**Regularization**: Techniques to prevent overfitting (L2, dropout, etc.).

**Residual**: Difference between prediction and target. What DL learns.

**ReLU**: Rectified Linear Unit. max(0, x).

## S

**Softplus**: Smooth approximation of ReLU. log(1 + exp(x)). Always positive.

**Spectral Normalization**: Constraint on weight matrix to control Lipschitz constant.

**Stochastic Gradient Descent (SGD)**: Optimization algorithm using random mini-batches.

## T

**Temporal Split**: Dividing data by time, not randomly. Required for forecasting.

**Transfer Learning**: Using knowledge from one domain in another.

## U

**Uncertainty Quantification**: Estimating how confident the model is in its predictions.

## V

**Variational Inference**: Approximating intractable posteriors with simpler distributions.

## W

**Weight Decay**: L2 regularization on weights. Prevents large weights.

## Z

**Z-Score**: Standardized value. (x - μ) / σ. Mean 0, std 1.

---

# Appendix A: Key File Locations

| Purpose             | File Path                                       |
| ------------------- | ----------------------------------------------- |
| Main Model          | `src/ml/models/st_gnn_latent.py`                |
| Loss Functions      | `src/ml/training/latent_losses.py`              |
| Training Script     | `scripts/train_latent_model.py`                 |
| DL Role Audit       | `src/ml/audit/dl_role_audit.py`                 |
| Input Guards        | `src/ml/guards/input_sanity_checks.py`          |
| Deployment Pipeline | `src/pipeline/universal_deployment_pipeline.py` |
| Full System Run     | `run_complete_system.py`                        |
| Kaggle Instructions | `KAGGLE_INSTRUCTIONS.md`                        |

---

# Appendix B: Commands to Run

```bash
# 1. Test architecture
python scripts/test_latent_architecture.py

# 2. Train model (local - slow)
python scripts/train_latent_model.py

# 3. Run full system
python run_complete_system.py

# 4. Create Kaggle bundle
python scripts/prepare_kaggle_bundle.py
```

---

# Appendix C: Citation

```bibtex
@software{urban_drainage_stress_2026,
  title = {Urban Drainage Stress Inference System},
  author = {Shivanshu Tiwari},
  year = {2026},
  url = {https://github.com/imshivanshutiwari/urban-drainage-stress},
  note = {Bayesian + Deep Learning hybrid for flood risk assessment}
}
```

---

**End of Documentation**

*Generated: January 2026*
*Last Updated: 2026-01-11*
