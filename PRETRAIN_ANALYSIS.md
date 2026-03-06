# Technical Analysis: Pre-train + DP Finetuning Framework

## 1. Current Project Structure

```
dp-gnn-e2e/
├── dp_gnn/                    # Main package
│   ├── configs/               # Configurations (mlp.py, gcn.py, dpmlp.py, dpgcn.py)
│   ├── models.py              # MLP, GraphMLP, GCN model definitions
│   ├── train.py               # Training loop (non-DP and DP paths)
│   ├── optimizers.py          # DP-SGD: clip_by_norm, dp_aggregate
│   ├── privacy_accountants.py # RDP accounting (standard + multi-term)
│   ├── input_pipeline.py      # Graph preprocessing pipeline
│   ├── sampler.py             # Bernoulli edge sampling
│   ├── normalizations.py      # Edge normalization
│   └── dataset_readers.py     # OGB dataset loading
├── scripts/                   # Training scripts
├── tests/                     # Test suite (395 tests)
└── reference_repo/            # Original JAX/jraph reference
```

## 2. DPGCN Finetuning Backend Analysis

### 2.1 Model Architecture

**GCN (GraphConvolutionalNetwork):**
- **Encoder**: `MultiLayerPerceptron` with `latent_size` hidden units, `num_encoder_layers` layers
- **Core**: `num_message_passing_steps` hops of `OneHopGraphConvolution`
- **Decoder**: `MultiLayerPerceptron` with `num_decoder_layers` layers, final layer outputs `num_classes`

**Key observation for parameter transfer:**
- Encoder MLP structure: `[input_dim] → [latent_size] × num_encoder_layers`
- Decoder MLP structure: `[latent_size] → [latent_size] × (num_decoder_layers-1) → [num_classes]`

**MLP (GraphMultiLayerPerceptron):**
- Single MLP: `[input_dim] → [latent_size] × num_layers → [num_classes]`

### 2.2 Parameter Space Alignment

For parameter transfer to work:

| Component | GCN | MLP | Compatible? |
|-----------|-----|-----|-------------|
| Input layer | encoder.layers.0 | layers.0 | ✅ Same shape [input_dim, latent_size] |
| Hidden layers | encoder.layers[i] | layers[i] | ✅ Same shape if depths match |
| Output layer | decoder.layers[-1] | layers[-1] | ✅ Same shape [latent_size, num_classes] |
| Message passing | core_hops (skippable) | N/A | ❌ GCN-specific |

**Transfer strategy:**
1. **Encoder transfer**: MLP layers 0..(num_encoder_layers-1) → GCN encoder
2. **Classifier transfer**: MLP last layer → GCN decoder last layer
3. **Full transfer**: Requires MLP depth = encoder_layers + decoder_layers

### 2.3 Data Flow

```
Dataset (OGB) → add_reverse_edges → subsample_graph (max_degree) → 
compute_masks → convert_to_pyg_data → add_self_loops → normalize_edges → Data
```

### 2.4 DP Training Mechanism

1. **Subgraph extraction**: `get_subgraphs()` creates padded subgraph indices per node
2. **Clipping threshold estimation**: `estimate_clipping_thresholds()` from sample
3. **Per-example gradients**: 
   - MLP: `_clip_and_sum_mlp_vmap()` using torch.func.vmap
   - GCN: `_clip_and_sum_gcn_vmap()` using vectorized feature gathering
4. **Noise addition**: `dp_aggregate()` adds Gaussian noise scaled by `clip * base_sensitivity * noise_multiplier`
5. **Privacy accounting**: Multi-term DPSGD for GCN (accounts for neighbor amplification)

### 2.5 Baseline Reproduction Command

```bash
# Current DP-GCN baseline (55.8% at ε≤12)
python scripts/train_dpgcn_arxiv.py

# Config: noise_mult=4.0, lr=3e-3, latent=100, 1+1+1 layers, batch=10000
```

## 3. Reusable Modules for Pretraining

### 3.1 Fully Reusable
- `dataset_readers.py` - OGB dataset loading
- `input_pipeline.py` - Graph preprocessing
- `models.py` - MLP/GCN definitions (with modifications for checkpoint loading)
- `privacy_accountants.py` - DP accounting
- `optimizers.py` - Gradient clipping and noise

### 3.2 Partially Reusable
- `train.py` - Need to add:
  - Pretraining trainer (non-DP, full-batch or large-batch)
  - Checkpoint save/load
  - Parameter mapping MLP→GCN

### 3.3 New Modules Required
- `pretrain.py` - MLP pretraining pipeline
- `checkpoint_utils.py` - Save/load parameter mappings
- `transfer.py` - Parameter transfer strategies
- `pretrain_configs.py` - Pretraining-specific configs

## 4. Constraints for Clean Parameter Transfer

### 4.1 Dimension Constraints
```python
# For encoder transfer:
MLP.layers[i].weight.shape == GCN.encoder.layers[i].weight.shape
# Requires: mlp_hidden_dim == gcn_latent_size

# For classifier transfer:
MLP.layers[-1].weight.shape == GCN.decoder.layers[-1].weight.shape
# Requires: mlp_hidden_dim == gcn_latent_size
# And: mlp_num_classes == gcn_num_classes == 40 (for ogbn-arxiv)
```

### 4.2 Architecture Constraints
- MLP and GCN must use **same activation function** for transferred layers
- GCN encoder/decoder depths must be ≤ MLP depth for partial transfer
- Input/output dimensions must match (128 → latent_size → 40 for ogbn-arxiv)

### 4.3 Initialization Constraints
- Current code uses `_lecun_normal_init()` (Flax/JAX style truncated normal)
- Pretrained MLP must use same initialization for fair comparison

## 5. Implementation Plan Summary

### Stage 1: Pretraining Infrastructure
1. Create `pretrain.py` - Standard (non-DP) MLP training
2. Create `checkpoint_utils.py` - Parameter save/load
3. Create `transfer.py` - MLP→GCN parameter mapping

### Stage 2: Integration
1. Extend configs for pretraining settings
2. Modify `train.py` to support pretrained initialization
3. Add freeze/unfreeze capabilities

### Stage 3: Experiment Framework
1. Create `pretrain_mlp_arxiv.py` script
2. Create `finetune_dpgcn_from_pretrain.py` script
3. Create sweep scripts for hyperparameter search

### Stage 4: Validation
1. Unit tests for each module
2. End-to-end smoke tests
3. Baseline vs proposed comparison

## 6. Key Design Decisions

1. **Transfer scope**: Start with encoder-only (most impactful) + classifier transfer
2. **Pretraining mode**: Supervised on node labels (not self-supervised) to match downstream task
3. **Pretraining data**: Use full graph (not just train split) for maximum signal
4. **Freeze strategy**: Optional freezing of transferred layers for first N steps
5. **Checkpoint format**: Standard PyTorch state_dict for interoperability
