"""Sweep noise_multiplier and learning_rate for DP-GCN on ogbn-arxiv."""
import gc
import sys
import time

sys.path.insert(0, '.')

import torch
from dp_gnn.configs.dpgcn import get_config
from dp_gnn import train

noise_mult_options = [0.5, 1.0, 2.0]
lr_options = [1e-3, 3e-3, 5e-3]

results = []

for noise_mult in noise_mult_options:
    for lr in lr_options:
        config = get_config()
        config.device = 'cuda'
        config.training_noise_multiplier = noise_mult
        config.learning_rate = lr
        config.max_training_epsilon = 12.0

        label = f'noise={noise_mult}, lr={lr}'
        print(f'\n{"="*60}')
        print(f'Running: {label}')
        print(f'{"="*60}')

        t0 = time.time()
        try:
            model = train.train_and_evaluate(config)
        except Exception as e:
            print(f'ERROR: {e}')
            results.append((label, 'ERROR', str(e)))
            continue
        finally:
            if 'model' in dir():
                del model
            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        results.append((label, elapsed))

print('\n\n' + '='*60)
print('SWEEP SUMMARY')
print('='*60)
for r in results:
    print(r)
