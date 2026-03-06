# Experimental Settings and Hyperparameters

## Overview

This document details the complete experimental settings for the pretrain + DP finetuning experiments on ogbn-arxiv.

---

## 1. Dataset: ogbn-arxiv

| Property | Value |
|----------|-------|
| Task | Node classification (transductive) |
| Nodes | 169,343 |
| Edges | 1,166,243 (original) → 464,361 (after degree-bounded filtering) |
| Features | 128 dimensions |
| Classes | 40 |
| Training split | 90,941 nodes (53.7%) |
| Validation split | 29,799 nodes (17.6%) |
| Test split | 48,603 nodes (28.7%) |

**Preprocessing:**
- Add reverse edges (makes graph undirected)
- Degree-bounded sampling: max_degree = 5
- Add self-loops
- Edge normalization: inverse-degree

---

## 2. Model Architectures

### MLP (Pretraining)

```
Input (128) 
    ↓
Linear(128 → 100) + Tanh
    ↓
Linear(100 → 100) + Tanh  
    ↓
Linear(100 → 40)          (output)
```

| Property | Value |
|----------|-------|
| Layers | 3 |
| Hidden dim | 100 |
| Activation | tanh |
| Skip connections | No |
| Parameters | 27,040 |

### GCN (Baseline & Pretrained)

```
Input (128)
    ↓
┌─────────────────────────────────────┐
│  ENCODER (MLP)                      │
│  Linear(128 → 100) + Tanh           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MESSAGE PASSING (1 hop)            │
│  One-hop graph convolution          │
│  + Update: Linear(100 → 100) + Tanh │
│  + Skip connection                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  DECODER (MLP)                      │
│  Linear(100 → 40)                   │
└─────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| Encoder layers | 1 |
| Message passing steps | 1 |
| Decoder layers | 1 |
| Latent dimension | 100 |
| Activation | tanh |
| Max degree | 5 |
| Parameters | 27,040 |

---

## 3. Training Hyperparameters

### Pretraining (MLP)

| Parameter | Value |
|-----------|-------|
| Training data | Full graph (ALL nodes) |
| Epochs | 100 |
| Batch size | 10,000 |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Weight decay | 0.0 |
| Device | CUDA (GPU) |
| Seed | 0 |

**Pretraining Results:**
- Train: 58.12%
- Val: 60.63%
- Test: 59.88%
- Time: 7.4 seconds

### DP-GCN Finetuning

| Parameter | Value |
|-----------|-------|
| Training data | Training split only (90,941 nodes) |
| Batch size | 10,000 |
| Optimizer | Adam |
| Learning rate | 0.003 |
| Privacy accounting | Multi-term RDP accountant |
| Noise multiplier | 4.0 or 6.0 |
| L2 norm clip | 75th percentile (estimated from 10k samples) |
| Base sensitivity | 12.0 (2×(d+1) where d=5) |
| Device | CUDA (GPU) |
| Seed | 0 |

---

## 4. Experiment Configurations

### Experiment 1: 500 steps, noise=4.0

| Setting | Steps | Noise Mult | LR | Expected ε |
|---------|-------|-----------|-----|-----------|
| All methods | 500 | 4.0 | 0.003 | ~5.0 |

| Method | Transfer Strategy | Test Acc | ε |
|--------|------------------|----------|-----|
| Baseline | N/A | 51.68% | 4.98 |
| Pretrained | encoder_only | 56.40% | 4.98 |
| Pretrained | encoder_classifier | 57.19% | 4.98 |

**Improvement:** +5.51% (best result for this setting)

### Experiment 2: 1000 steps, noise=4.0

| Setting | Steps | Noise Mult | LR | Expected ε |
|---------|-------|-----------|-----|-----------|
| All methods | 1000 | 4.0 | 0.003 | ~7.3 |

| Method | Transfer Strategy | Test Acc | ε |
|--------|------------------|----------|-----|
| Baseline | N/A | 54.55% | 7.32 |
| Pretrained | encoder_classifier | 57.97% | 7.32 |

**Improvement:** +3.42% **(BEST OVERALL RESULT)**

### Experiment 3: 750 steps, noise=6.0

| Setting | Steps | Noise Mult | LR | Expected ε |
|---------|-------|-----------|-----|-----------|
| All methods | 750 | 6.0 | 0.003 | ~3.9 |

| Method | Transfer Strategy | Test Acc | ε |
|--------|------------------|----------|-----|
| Baseline | N/A | 51.64% | 3.93 |
| Pretrained | encoder_classifier | 57.11% | 3.93 |

**Improvement:** +5.47%

---

## 5. Transfer Strategies

### Strategy Comparison (500 steps, noise=4.0)

| Strategy | Transferred Parameters | Test Acc | Improvement |
|----------|----------------------|----------|-------------|
| encoder_only | 2 (encoder weight + bias) | 56.40% | +4.72% |
| encoder_classifier | 4 (encoder + classifier weights + biases) | 57.19% | +5.51% |

**Best strategy:** encoder_classifier

### Parameter Mapping

```
MLP (pretrained)          GCN (target)
├── layers.0.weight  →    encoder.layers.0.weight
├── layers.0.bias    →    encoder.layers.0.bias
├── layers.1.weight  →    (not transferred in encoder_classifier)
├── layers.1.bias    →    (not transferred in encoder_classifier)
├── layers.2.weight  →    decoder.layers.0.weight
└── layers.2.bias    →    decoder.layers.0.bias
```

---

## 6. Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.10.12 |
| PyTorch | 2.2.0+cu121 |
| PyTorch Geometric | 2.7.0 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 4090 (24GB) |
| dp-accounting | 0.6.0 |

---

## 7. Reproduction Commands

### Pretraining
```bash
python scripts/pretrain_mlp_arxiv.py \
    --config dpgcn_match \
    --epochs 100 \
    --device cuda \
    --seed 0
```

### Baseline DP-GCN
```bash
python scripts/finetune_dpgcn_with_pretrain.py \
    --no-pretrain \
    --noise-mult 4.0 \
    --lr 3e-3 \
    --epochs 1000 \
    --device cuda \
    --seed 0
```

### Pretrained DP-GCN (Best Config)
```bash
python scripts/finetune_dpgcn_with_pretrain.py \
    --pretrain-ckpt checkpoints/pretrain/mlp_dpgcn_match_ls100_nl3_ep100.pt \
    --transfer encoder_classifier \
    --noise-mult 4.0 \
    --lr 3e-3 \
    --epochs 1000 \
    --device cuda \
    --seed 0
```

---

## 8. Key Design Decisions

### Why latent_size=100?
- Matches reference DP-GCN implementation
- Balances model capacity with privacy noise impact

### Why noise_multiplier=4.0 or 6.0?
- 4.0: Moderate noise, good convergence (ε≈5-7)
- 6.0: Higher noise, allows more steps at lower ε (ε≈3.9)

### Why batch_size=10,000?
- Large batches reduce per-step privacy cost
- Allows more training steps within privacy budget

### Why max_degree=5?
- Limits sensitivity amplification
- Base sensitivity = 2×(5+1) = 12
- Trade-off between graph information and privacy

### Why 100 pretraining epochs?
- Sufficient to learn good feature representations
- Very fast (7.4 seconds)
- Diminishing returns beyond 100 epochs

### Why encoder_classifier transfer?
- Encoder: Provides good feature representations
- Classifier: Provides good decision boundary
- Both contribute to improved performance

---

## 9. Summary Statistics

| Metric | Value |
|--------|-------|
| Total experiments | 7 |
| Best baseline | 54.55% |
| Best pretrained | 57.97% |
| Best improvement | +5.51% |
| Average improvement | +4.8% |
| Pretraining overhead | 7.4s |

---

## 10. File Locations

### Configs
- `dp_gnn/configs/pretrain_mlp.py` - Pretraining configs
- `dp_gnn/configs/dpgcn.py` - DP-GCN configs

### Checkpoints
- `checkpoints/pretrain/mlp_dpgcn_match_ls100_nl3_ep100.pt` - Pretrained MLP

### Results
- `results/experiments/all_results.json` - Structured results
- `results/experiments/summary.txt` - Text summary

### Logs
- `results/experiments/pretrain_100ep.log`
- `results/experiments/baseline_nm4_*.log`
- `results/experiments/pretrained_nm4_*.log`
