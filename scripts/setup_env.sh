#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="2.2.0"
CUDA_TAG="cu121"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "=== DP-GNN Environment Setup ==="
echo "PyTorch ${TORCH_VERSION} + CUDA ${CUDA_TAG}"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: ${PY_VER}"

# Step 1: PyTorch
echo ""
echo "[1/4] Installing PyTorch ${TORCH_VERSION} (CUDA ${CUDA_TAG})..."
if python3 -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}')" 2>/dev/null; then
    echo "  -> Already installed, skipping."
else
    pip install torch==${TORCH_VERSION} torchvision==0.17.0 torchaudio==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/${CUDA_TAG}
fi

# Step 2: PyG extensions (must match torch + CUDA)
echo ""
echo "[2/4] Installing PyG extensions..."
if python3 -c "import torch_geometric" 2>/dev/null; then
    echo "  -> torch-geometric already installed, skipping extensions."
else
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
        -f "${PYG_WHEEL_URL}"
    pip install torch-geometric
fi

# Step 3: Core dependencies
echo ""
echo "[3/4] Installing core dependencies..."
pip install \
    dp-accounting==0.6.0 \
    ml-collections==1.1.0 \
    ogb==1.3.6 \
    "numpy>=1.24,<2.0" \
    "scipy>=1.10" \
    "scikit-learn>=1.3" \
    "pandas>=2.0" \
    "matplotlib>=3.7" \
    pytest

# Step 4: Download dataset
echo ""
echo "[4/4] Downloading ogbn-arxiv dataset..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -d "${PROJECT_ROOT}/datasets/ogbn_arxiv" ]; then
    echo "  -> Dataset already exists, skipping."
else
    python3 -c "
from ogb.nodeproppred import NodePropPredDataset
NodePropPredDataset(name='ogbn-arxiv', root='${PROJECT_ROOT}/datasets/')
print('  -> Downloaded successfully.')
"
fi

# Verification
echo ""
echo "=== Verification ==="
python3 -c "
import torch
import torch_geometric
import dp_accounting
import ml_collections
import ogb
import scipy
import sklearn
import pandas
import numpy

print(f'PyTorch:          {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:              {torch.cuda.get_device_name(0)}')
print(f'PyG:              {torch_geometric.__version__}')
print(f'dp-accounting:    installed')
print(f'NumPy:            {numpy.__version__}')
print(f'SciPy:            {scipy.__version__}')
print()
print('All dependencies OK.')
"

echo ""
echo "=== Setup Complete ==="
echo "Run tests:  python -m pytest tests/ -v"
echo "Train MLP:  python scripts/train_mlp_arxiv.py"
echo "Train GCN:  python scripts/train_gcn_arxiv.py"
