# Environment Setup

## System Requirements

- **OS:** Ubuntu 22.04 LTS (tested on kernel 6.8.0)
- **GPU:** NVIDIA GPU with CUDA 12.1 support (tested on RTX 4090, 24 GB VRAM)
- **NVIDIA Driver:** >= 530.x (tested with 570.211.01)
- **Python:** 3.10.x
- **Disk:** ~2 GB for dependencies + ~200 MB for ogbn-arxiv dataset

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.0+cu121 | Deep learning framework |
| `torch-geometric` | 2.7.0 | Graph neural network library |
| `torch-scatter` | 2.1.2+pt22cu121 | Scatter operations for PyG |
| `torch-sparse` | 0.6.18+pt22cu121 | Sparse operations for PyG |
| `torch-cluster` | 1.6.3+pt22cu121 | Clustering operations for PyG |
| `torch-spline-conv` | 1.2.2+pt22cu121 | Spline convolutions for PyG |
| `dp-accounting` | 0.6.0 | Google's differential privacy accounting (RDP) |
| `ml-collections` | 1.1.0 | Configuration management |
| `ogb` | 1.3.6 | Open Graph Benchmark dataset loading |
| `numpy` | 1.26.3 | Numerical computing |
| `scipy` | 1.15.3 | Scientific computing (hypergeom, logsumexp) |
| `scikit-learn` | 1.7.2 | StandardScaler for feature preprocessing |
| `pandas` | 2.3.3 | CSV reading for dataset files |
| `matplotlib` | 3.10.8 | Plotting (optional, for result visualization) |
| `pytest` | 9.0.2 | Testing framework |

## Quick Setup

```bash
# From project root
bash scripts/setup_env.sh
```

## Manual Setup

### 1. PyTorch (CUDA 12.1)

If PyTorch is not already installed:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 2. PyG Extensions (must match PyTorch + CUDA version)

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric
```

### 3. Remaining Dependencies

```bash
pip install dp-accounting ml-collections ogb \
    numpy scipy scikit-learn pandas matplotlib pytest
```

### 4. Dataset

The ogbn-arxiv dataset is downloaded automatically on first training run via OGB.
To pre-download:

```bash
python -c "from ogb.nodeproppred import NodePropPredDataset; NodePropPredDataset(name='ogbn-arxiv', root='datasets/')"
```

## Verification

```bash
# Run full test suite (395 tests, ~30s)
python -m pytest tests/ -v

# Quick smoke test
python -c "
import torch, torch_geometric, dp_accounting, ml_collections, ogb
print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'PyG {torch_geometric.__version__}')
print('All imports OK')
"
```

## Project Structure

```
dp-gnn-e2e/
├── dp_gnn/                    # Main package
│   ├── configs/               # ML-collections configs (gcn, mlp, dpgcn, dpmlp)
│   ├── models.py              # MLP, GraphMLP, GCN model definitions
│   ├── train.py               # Training loop (non-DP and DP paths)
│   ├── optimizers.py          # DP-SGD: clip_by_norm, dp_aggregate
│   ├── privacy_accountants.py # RDP accounting (standard + multi-term)
│   ├── input_pipeline.py      # Graph preprocessing pipeline
│   ├── sampler.py             # Bernoulli edge sampling with degree bounds
│   ├── normalizations.py      # Edge normalization (inv-degree, inv-sqrt-degree)
│   └── dataset_readers.py     # OGB dataset loading
├── scripts/                   # Training scripts
│   ├── setup_env.sh           # Environment setup script
│   ├── train_mlp_arxiv.py     # Non-DP MLP training
│   ├── train_gcn_arxiv.py     # Non-DP GCN training
│   ├── train_dpmlp_arxiv.py   # DP-MLP training
│   └── train_dpgcn_arxiv.py   # DP-GCN training
├── tests/                     # Test suite (395 tests)
├── reference_repo/            # Original JAX/jraph reference implementation
├── datasets/                  # Downloaded datasets (gitignored)
├── AI_LOG.md                  # Development session log
├── RESUME.md                  # Resume instructions
└── ENVIRONMENT.md             # This file
```

## Notes

- The PyG extension wheels (torch-scatter, etc.) are version-locked to the PyTorch + CUDA combination. If you upgrade PyTorch, you must reinstall these from the matching wheel URL.
- The `dp-accounting` package is Google's standalone DP library, distinct from `tensorflow-privacy`. The reference repo lists `tensorflow_privacy` in requirements.txt but actually imports `dp_accounting`.
- GPU is strongly recommended for DP training (per-example gradient computation is the bottleneck).
