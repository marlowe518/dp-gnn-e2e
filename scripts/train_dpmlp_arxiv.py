"""Train DP-MLP on ogbn-arxiv with reference settings."""

import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

from dp_gnn.configs.dpmlp import get_config
from dp_gnn import train


def main():
    config = get_config()
    config.device = 'cuda'
    print(f"Config: model={config.model}, lr={config.learning_rate}, "
          f"layers={config.num_layers}, latent={config.latent_size}, "
          f"batch={config.batch_size}, steps={config.num_training_steps}, "
          f"noise={config.training_noise_multiplier}, "
          f"max_eps={config.max_training_epsilon}",
          flush=True)
    model = train.train_and_evaluate(config, workdir='/tmp/dp_gnn_dpmlp_arxiv')
    print("Training complete.", flush=True)


if __name__ == '__main__':
    main()
