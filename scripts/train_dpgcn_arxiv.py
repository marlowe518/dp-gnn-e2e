"""Run DP-GCN training on ogbn-arxiv with the reference config."""
import sys
import time

sys.path.insert(0, '.')

from dp_gnn.configs.dpgcn import get_config
from dp_gnn import train

config = get_config()
config.device = 'cuda'

print('=== DP-GCN Training on ogbn-arxiv ===')
print(f'Config: lr={config.learning_rate}, noise_mult={config.training_noise_multiplier}, '
      f'batch_size={config.batch_size}, max_degree={config.max_degree}, '
      f'latent_size={config.latent_size}, steps={config.num_training_steps}')

t0 = time.time()
model = train.train_and_evaluate(config)
elapsed = time.time() - t0
print(f'\nTotal training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)')
