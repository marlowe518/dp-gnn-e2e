"""Train non-DP GCN on ogbn-arxiv with reference settings on GPU."""

import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)

from dp_gnn.configs.gcn import get_config
from dp_gnn import train


def main():
    config = get_config()
    config.device = 'cuda'
    config.num_training_steps = 1000
    config.evaluate_every_steps = 50
    print(f"Config: model={config.model}, lr={config.learning_rate}, "
          f"latent={config.latent_size}, enc_layers={config.num_encoder_layers}, "
          f"mp_steps={config.num_message_passing_steps}, "
          f"dec_layers={config.num_decoder_layers}, "
          f"batch={config.batch_size}, steps={config.num_training_steps}",
          flush=True)
    model = train.train_and_evaluate(config, workdir='/tmp/dp_gnn_gcn_arxiv')
    print("Training complete.", flush=True)


if __name__ == '__main__':
    main()
