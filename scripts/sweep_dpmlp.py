"""DP-MLP hyperparameter sweep on ogbn-arxiv.

Sweeps num_layers x learning_rate (same axes as reference config).
Reports best test accuracy and corresponding epsilon.
"""

import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import torch
from dp_gnn.configs.dpmlp import get_config
from dp_gnn import train


def run_sweep():
    num_layers_options = [1, 2, 3, 4]
    lr_options = [1e-3, 2e-3, 3e-3, 5e-3]

    results = []
    for num_layers in num_layers_options:
        for lr in lr_options:
            config = get_config()
            config.device = 'cuda'
            config.num_layers = num_layers
            config.learning_rate = lr
            config.num_training_steps = 500
            config.evaluate_every_steps = 50

            label = f"layers={num_layers}, lr={lr}"
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"{'='*60}", flush=True)

            t0 = time.time()
            model = train.train_and_evaluate(config)
            elapsed = time.time() - t0
            print(f"Elapsed: {elapsed:.1f}s", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    print("\n\nSweep complete.")


if __name__ == '__main__':
    run_sweep()
