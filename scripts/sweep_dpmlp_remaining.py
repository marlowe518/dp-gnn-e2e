"""DP-MLP sweep: remaining configs (layers 2-4)."""

import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

import gc
import time
import torch
from dp_gnn.configs.dpmlp import get_config
from dp_gnn import train


def run_sweep():
    configs = [
        (2, 0.003), (2, 0.005),
        (3, 0.001), (3, 0.002), (3, 0.003), (3, 0.005),
        (4, 0.001), (4, 0.002), (4, 0.003), (4, 0.005),
    ]

    for num_layers, lr in configs:
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
        try:
            model = train.train_and_evaluate(config)
            elapsed = time.time() - t0
            print(f"Elapsed: {elapsed:.1f}s", flush=True)
            del model
        except torch.cuda.OutOfMemoryError:
            print(f"OOM for {label}, skipping", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n\nSweep complete.")


if __name__ == '__main__':
    run_sweep()
