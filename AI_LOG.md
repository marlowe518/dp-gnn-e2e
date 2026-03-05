# AI_LOG (Persistent)

## [Session 1 — Prior Session: Full Reproduction]

**Goal:** Reproduce the reference JAX/jraph "Node-Level Differentially Private Graph Neural Networks" repository in PyTorch/PyG, unit-by-unit with independent tests.

**Context:**
- Branch: `main`
- Reference repo: `reference_repo/differentially_private_gnns/`
- Target: `dp_gnn/` package + `tests/`

**Work Log:**
- ✅ Implemented all 8 core modules in `dp_gnn/`:
  - `normalizations.py` — edge normalization (inverse-degree, inverse-sqrt-degree)
  - `models.py` — MLP, GraphMLP, OneHopGraphConvolution, GCN
  - `sampler.py` — in-degree bounded Bernoulli sampling
  - `dataset_readers.py` — DummyDataset (initially)
  - `input_pipeline.py` — reverse edges, subsampling, self-loops, normalization
  - `optimizers.py` — clip_by_norm, dp_aggregate (DP-SGD core)
  - `privacy_accountants.py` — multiterm + standard DPSGD accounting via dp_accounting
  - `train.py` — full training loop (non-DP and DP paths)
- ✅ Created test suite in `tests/`: 8 test files, 361+ tests all passing
- ✅ Created configs in `dp_gnn/configs/`: gcn.py, mlp.py, dpgcn.py, dpmlp.py (DummyDataset)
- ✅ Bug fixed: `torch.normal` NaN std when `noise_multiplier=0` and `l2_norms_threshold=inf` → added `isfinite` guard in `dp_aggregate`

---

## [Session 2 — 2026-03-05: ogbn-arxiv Integration + DP Completion]

**Goal:** Implement ogbn-arxiv data loading, match reference configs exactly, fix training loop correctness, add GPU support, optimize DP per-example gradients, and run end-to-end training on all 4 configs.

**Context:**
- Branch: `main` (single commit `b0da06e`)
- Environment: Python 3.10, PyTorch 2.2+cu121, PyG, RTX 4090 GPU available
- Dataset: ogbn-arxiv downloaded to `datasets/ogbn_arxiv/` (169,343 nodes, 1.17M edges, 128-dim features, 40 classes)
- Dependencies added: `ogb`, `pandas`, `scikit-learn`

### Work Log

#### 1. OGB Dataset Readers (`dp_gnn/dataset_readers.py`)
- ✅ Added `OGBTransductiveDataset` — reads raw CSV files (node-feat, node-label, edge, splits) matching reference exactly
- ✅ Added `OGBDisjointDataset` — filters inter-split edges (for DP-GCN), uses `np.vectorize` split membership check
- ✅ Updated `get_dataset()` dispatcher for `ogb*` and `*-disjoint` name patterns
- ✅ Downloaded ogbn-arxiv dataset via `ogb.nodeproppred.NodePropPredDataset`

#### 2. Configs Updated to Match Reference (`dp_gnn/configs/`)
- ✅ `gcn.py`: ogbn-arxiv, adam, lr=0.002, latent=256, 2 enc + 1 MP + 2 dec layers, batch=1000, 1000 steps
- ✅ `mlp.py`: ogbn-arxiv, adam, lr=0.003, latent=256, 2 layers, batch=1000, 10000 steps
- ✅ `dpgcn.py`: ogbn-arxiv-disjoint, adam, lr=3e-3, latent=100, 1+1+1, noise=2.0, batch=10000, 3000 steps
- ✅ `dpmlp.py`: ogbn-arxiv, adam, lr=0.003, latent=256, 1 layer, tanh, noise=3.0, batch=10000, 500 steps, max_eps=10

#### 3. Training Loop Fixes (`dp_gnn/train.py`)
- ✅ **Critical bug fix: node index mapping** — discovered that ogbn-arxiv training nodes are NOT contiguous (only 54% of {0..N_train-1} are train nodes). The reference's `compute_updates` indexes full-graph logits with training-relative indices, causing label-logit mismatch. Fixed by adding `train_indices` parameter to `compute_updates()` to map batch indices → global node IDs for logit indexing.
- ✅ Step counting aligned with reference: `range(initial_step, num_training_steps)`, privacy accountant called with `step+1`
- ✅ Evaluation: `evaluate_predictions` uses masked mean (matching reference's `jnp.where` + `jnp.sum/sum(mask)` pattern)
- ✅ Metrics logging with accuracy as percentage (matching reference)

#### 4. GPU Support (`dp_gnn/train.py`)
- ✅ Added `config.device` support (defaults to `'cpu'`)
- ✅ `data.to(device)`, `labels.to(device)`, `masks.to(device)`, `model.to(device)`
- ✅ Batch indices generated on CPU then moved to device

#### 5. DP Path GPU Fixes
- ✅ `make_subgraph_from_indices`: subgraph built on CPU from `data.x.detach().cpu()`, then `.to(device)` at end
- ✅ `estimate_clipping_thresholds`: `.detach().cpu().numpy()` for percentile computation
- ✅ `dp_aggregate` in `optimizers.py`: noise generated on CPU (CPU generator), then `.to(summed.device)`

#### 6. DP Per-Example Gradient Optimization (`dp_gnn/train.py`)
- ✅ **MLP fast path** (`_compute_per_example_grads_mlp_vmap`): Uses `torch.func.vmap` + `torch.func.grad` + `functional_call` for vectorized per-example gradients. ~6500x speedup over sequential loop (23s vs projected 42h for 500 steps, batch=10K).
- ✅ **GCN batched path** (`_compute_per_example_grads_gcn_batched`): Uses `torch_geometric.data.Batch` for single forward pass, then sequential backward for per-example gradients. Pre-allocates gradient storage.
- ✅ `compute_updates_for_dp` dispatches based on model type (`GraphMultiLayerPerceptron` → vmap, else → batched)

#### 7. Tests
- ✅ Updated `tests/test_train.py` and `tests/test_e2e.py` for new configs (DummyDataset for unit tests)
- ✅ Added `tests/test_ogbn_arxiv.py` — validates loading, splits, full pipeline for MLP and GCN
- ✅ All 395 tests pass: `python -m pytest tests/ -v` → 395 passed

#### 8. Training Scripts (`scripts/`)
- ✅ `train_mlp_arxiv.py` — non-DP MLP on ogbn-arxiv (CPU)
- ✅ `train_gcn_arxiv.py` — non-DP GCN on ogbn-arxiv (GPU)
- ✅ `train_dpmlp_arxiv.py` — DP-MLP on ogbn-arxiv (GPU)
- ✅ `train_dpgcn_arxiv.py` — DP-GCN on ogbn-arxiv-disjoint (GPU)

### Results

**Non-DP MLP** (1000 steps, CPU, ~14 min):
| Step | Train Acc | Val Acc | Test Acc |
|------|-----------|---------|----------|
| 500  | 55.2%     | 55.2%   | 53.4%    |
| 800  | 56.4%     | 55.6%   | **53.8%** (peak) |
| 999  | 57.2%     | 54.9%   | 52.0%    |
Reference: ~55% test accuracy (10K steps)

**Non-DP GCN** (1000 steps, GPU, 52 seconds):
| Step | Train Acc | Val Acc | Test Acc |
|------|-----------|---------|----------|
| 550  | 64.3%     | 65.0%   | 64.5%    |
| 850  | 66.7%     | 66.3%   | **65.8%** (peak) |
| 999  | 67.9%     | 66.7%   | 65.8%    |
Reference: ~68% test accuracy (1K steps, best hyperparams)

**DP-MLP** (500 steps, GPU, 23 seconds, noise=3.0):
| Step | Train Acc | Val Acc | Test Acc | Epsilon |
|------|-----------|---------|----------|---------|
| 100  | 18.1%     | 8.0%    | 6.3%     | 1.92    |
| 499  | 20.1%     | 11.8%   | 9.8%     | 4.43    |

**DP-GCN** (in progress at session end, batch=1000, noise=2.0):
- Step 20: epsilon=1.24, test_acc=5.9%
- Still running with 10K estimation samples, clipping thresholds estimated

### Unstaged Changes (at session end)
- `dp_gnn/train.py` — vmap optimization + GCN batched path (the main diff)
- `dp_gnn/configs/dpgcn.py`, `dp_gnn/configs/dpmlp.py` — updated to reference settings
- `scripts/train_dpgcn_arxiv.py`, `scripts/train_dpmlp_arxiv.py` — new DP training scripts

### Next Steps (Resume From Here)

1. **Commit all changes** — stage unstaged files (`dp_gnn/train.py`, `dp_gnn/configs/dpgcn.py`, `dp_gnn/configs/dpmlp.py`, `scripts/train_dp*.py`), commit everything. Consider adding `datasets/` and `__pycache__/` to `.gitignore`.
2. **Run full DP-GCN training** at reference batch_size=10000 for 3000 steps — the GCN batched path is slower than MLP vmap; may need further optimization or patience.
3. **Run full DP-MLP training** at reference batch_size=10000 for 500 steps with max_eps=10 — already done in 23s; could extend to confirm epsilon convergence.
4. **Hyperparameter sweep** — reference reports results across multiple configs (encoder/decoder layers, learning rates, noise multipliers). Run sweeps to match Table 1 numbers.
5. **Consider vmap for GCN** — the GCN per-example gradient path still uses sequential backward. Investigate `torch.func.vmap` over batched subgraphs if performance is insufficient.

### Checkpoint
- **branch:** `main`
- **last_good_commit:** `b0da06e` (reference repo only; all new code is staged but uncommitted)
- **tests:** 395 passed (`python -m pytest tests/ -v`)
