"""Train non-DP MLP on ogbn-arxiv with reference settings."""

import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

from dp_gnn.configs.mlp import get_config
from dp_gnn import train


def main():
    config = get_config()
    config.num_training_steps = 1000
    config.evaluate_every_steps = 100
    print(f"Config: model={config.model}, lr={config.learning_rate}, "
          f"layers={config.num_layers}, latent={config.latent_size}, "
          f"batch={config.batch_size}, steps={config.num_training_steps}",
          flush=True)
    model = train.train_and_evaluate(config, workdir='/tmp/dp_gnn_mlp_arxiv')
    print("Training complete.", flush=True)


if __name__ == '__main__':
    main()
