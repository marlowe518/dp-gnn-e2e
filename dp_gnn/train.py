"""Training pipeline for DP-GNN.

Reproduces the reference JAX training loop in PyTorch:
- Standard (non-DP) training: full-graph forward, batch loss over sampled nodes.
- DP training: per-example subgraph forward + gradient clipping + noise.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch_geometric.data import Data

from dp_gnn import input_pipeline
from dp_gnn import models
from dp_gnn import normalizations
from dp_gnn import optimizers as dp_optimizers
from dp_gnn import privacy_accountants

_SUBGRAPH_PADDING_VALUE = -1


# ---------------------------------------------------------------------------
# Subgraph extraction (for DP training)
# ---------------------------------------------------------------------------

def get_subgraphs(data: Data, pad_to: int) -> torch.Tensor:
    """Creates an array of padded subgraph indices [num_nodes, pad_to].

    For each node, the subgraph contains itself followed by its outgoing
    neighbors (deduped, order-preserving), padded to `pad_to`.
    Matches reference: uses np.unique(..., return_index=True) for order-preserving dedup.
    """
    num_nodes = data.num_nodes
    senders = data.edge_index[0].numpy()
    receivers = data.edge_index[1].numpy()

    outgoing = {u: [] for u in range(num_nodes)}
    for s, r in zip(senders, receivers):
        if s != r:
            outgoing[s].append(r)

    subgraphs = np.full((num_nodes, pad_to), _SUBGRAPH_PADDING_VALUE, dtype=np.int64)
    for node in range(num_nodes):
        indices = np.asarray([node] + outgoing[node])
        indices = indices[np.sort(np.unique(indices, return_index=True)[1])]
        indices = indices[:pad_to]
        subgraphs[node, :len(indices)] = indices

    return torch.tensor(subgraphs, dtype=torch.long)


def make_subgraph_from_indices(
    data: Data,
    subgraph_indices: torch.Tensor,
    adjacency_normalization: Optional[str],
) -> Data:
    """Constructs a subgraph Data object from subgraph_indices [pad_to].

    The root node is always index 0 in the subgraph.
    Subgraph construction is done on CPU, then moved to data's device.
    """
    device = data.x.device
    # Work on CPU for subgraph construction
    subgraph_indices = subgraph_indices.cpu()
    x_cpu = data.x.detach().cpu()

    valid_mask = (subgraph_indices != _SUBGRAPH_PADDING_VALUE)
    safe_indices = subgraph_indices.clone()
    safe_indices[~valid_mask] = 0

    subgraph_nodes = x_cpu[safe_indices]
    subgraph_nodes[~valid_mask] = 0.0

    padding_node_idx = len(subgraph_indices)
    subgraph_nodes = torch.cat(
        [subgraph_nodes, torch.zeros(1, subgraph_nodes.shape[1])], dim=0)

    subgraph_senders = torch.zeros_like(subgraph_indices)
    subgraph_receivers = torch.arange(len(subgraph_indices))

    subgraph_senders = torch.where(valid_mask, subgraph_senders,
                                   torch.full_like(subgraph_senders, padding_node_idx))
    subgraph_receivers = torch.where(valid_mask, subgraph_receivers,
                                     torch.full_like(subgraph_receivers, padding_node_idx))

    edge_index = torch.stack([subgraph_senders, subgraph_receivers])
    edge_attr = torch.ones(edge_index.size(1), 1)

    sub_data = Data(
        x=subgraph_nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    sub_data.num_nodes = subgraph_nodes.shape[0]

    sub_data = normalizations.normalize_edges_with_mask(
        sub_data, valid_mask, adjacency_normalization)

    # Move to the same device as the original data
    return sub_data.to(device)


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def create_model(config, input_dim: int) -> nn.Module:
    """Creates an MLP or GCN model based on config."""
    activation_map = {
        'relu': torch.relu,
        'tanh': torch.tanh,
        'selu': torch.selu,
    }
    activation = activation_map[config.activation_fn]

    if config.model == 'mlp':
        dimensions = [config.latent_size] * config.num_layers + [config.num_classes]
        return models.GraphMultiLayerPerceptron(
            dimensions=dimensions,
            activation=activation,
            input_dim=input_dim,
        )
    elif config.model == 'gcn':
        return models.GraphConvolutionalNetwork(
            latent_size=config.latent_size,
            num_classes=config.num_classes,
            num_message_passing_steps=config.num_message_passing_steps,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            activation=activation,
            input_dim=input_dim,
        )
    else:
        raise ValueError(f'Unsupported model: {config.model}.')


# ---------------------------------------------------------------------------
# Sensitivity & max terms
# ---------------------------------------------------------------------------

def compute_max_terms_per_node(config) -> int:
    if config.model == 'mlp':
        return 1
    d = config.max_degree
    k = config.num_message_passing_steps
    if k == 1:
        return d + 1
    if k == 2:
        return d ** 2 + d + 1
    raise ValueError('Not supported for num_message_passing_steps > 2.')


def compute_base_sensitivity(config) -> float:
    if config.model == 'mlp':
        return 1.0
    d = config.max_degree
    k = config.num_message_passing_steps
    if k == 1:
        return float(2 * (d + 1))
    if k == 2:
        return float(2 * (d ** 2 + d + 1))
    raise ValueError('Not supported for num_message_passing_steps > 2.')


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------

def compute_logits(model: nn.Module, data: Data) -> torch.Tensor:
    """Forward pass returning logits [num_nodes, num_classes]."""
    return model(data).x


def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean softmax cross-entropy loss. labels are one-hot [N, C].

    Matches reference: optax.softmax_cross_entropy + jnp.mean.
    """
    return F.cross_entropy(logits, labels)


def compute_updates(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    node_indices: torch.Tensor,
    train_indices: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Standard (non-DP) gradient computation over a batch of nodes.

    Args:
        model: The model.
        data: Full graph data.
        labels: Training labels [num_training_nodes, C].
        node_indices: Batch indices into [0, num_training_nodes).
        train_indices: Global node IDs of training nodes. If provided,
            maps node_indices to global IDs for logit indexing.
    """
    model.zero_grad()
    logits = compute_logits(model, data)
    if train_indices is not None:
        global_idx = train_indices[node_indices]
    else:
        global_idx = node_indices
    batch_logits = logits[global_idx]
    batch_labels = labels[node_indices]
    loss = compute_loss(batch_logits, batch_labels)
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()
             if p.grad is not None}
    return grads


def _compute_per_example_grads_mlp_vmap(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    node_indices: torch.Tensor,
    chunk_size: int = 2000,
) -> Dict[str, torch.Tensor]:
    """Vectorized per-example gradients for MLP using torch.func.vmap.

    Returns per-example gradients of shape [B, *param_shape].
    Processes in chunks to limit peak memory.
    Used for threshold estimation (small B).
    """
    device = data.x.device
    node_indices_cpu = node_indices.cpu() if node_indices.is_cuda else node_indices
    global_ids = subgraphs[node_indices_cpu, 0].to(device)
    batch_x = data.x[global_ids]
    batch_labels = labels[node_indices_cpu].to(device)

    mlp_module = model.mlp
    params = dict(mlp_module.named_parameters())

    def single_loss(params, x, label):
        logits = functional_call(mlp_module, params, (x,))
        return F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

    ft_per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))

    B = len(node_indices)
    chunks_grads = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_eg = ft_per_sample_grad(
            params, batch_x[start:end], batch_labels[start:end])
        chunks_grads.append(chunk_eg)

    per_example_grads = {}
    for mlp_name in chunks_grads[0]:
        full_name = f'mlp.{mlp_name}'
        per_example_grads[full_name] = torch.cat(
            [c[mlp_name] for c in chunks_grads], dim=0)

    return per_example_grads


def _clip_and_sum_mlp_vmap(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    node_indices: torch.Tensor,
    l2_norms_threshold: Dict[str, float],
    chunk_size: int = 500,
) -> Dict[str, torch.Tensor]:
    """Memory-efficient per-example clip-and-sum for MLP.

    Computes per-example grads in chunks, clips and sums within each
    chunk, then accumulates across chunks. Never materializes all B
    per-example grads simultaneously.
    """
    device = data.x.device
    node_indices_cpu = node_indices.cpu() if node_indices.is_cuda else node_indices
    global_ids = subgraphs[node_indices_cpu, 0].to(device)
    batch_x = data.x[global_ids]
    batch_labels = labels[node_indices_cpu].to(device)

    mlp_module = model.mlp
    params = dict(mlp_module.named_parameters())

    def single_loss(params, x, label):
        logits = functional_call(mlp_module, params, (x,))
        return F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

    ft_per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))

    B = len(node_indices)
    clipped_sum = None

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_eg = ft_per_sample_grad(
            params, batch_x[start:end], batch_labels[start:end])

        chunk_dict = {f'mlp.{k}': v for k, v in chunk_eg.items()}
        chunk_clipped = dp_optimizers.clip_by_norm(chunk_dict, l2_norms_threshold)

        if clipped_sum is None:
            clipped_sum = {k: v.sum(dim=0) for k, v in chunk_clipped.items()}
        else:
            for k in clipped_sum:
                clipped_sum[k] += chunk_clipped[k].sum(dim=0)

    return clipped_sum


def _compute_per_example_grads_gcn_vmap(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    sub_weights: torch.Tensor,
    node_indices: torch.Tensor,
    chunk_size: int = 500,
) -> Dict[str, torch.Tensor]:
    """Per-example gradients for GCN using vmap + weight-vector forward.

    Returns per-example gradients of shape [B, *param_shape].
    Used for threshold estimation.
    """
    device = data.x.device
    B = len(node_indices)
    node_indices_cpu = node_indices.cpu() if node_indices.is_cuda else node_indices
    batch_labels = labels[node_indices_cpu].to(device)
    activation = model.encoder.activation
    params = dict(model.named_parameters())

    def single_loss(params, sub_x_i, sub_w_i, label_i):
        logits = _gcn_forward_root_only(params, sub_x_i, sub_w_i, activation)
        return F.cross_entropy(logits.unsqueeze(0), label_i.unsqueeze(0))

    ft_per_sample_grad = vmap(grad(single_loss), in_dims=(None, 0, 0, 0))

    chunks_grads = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_indices = node_indices[start:end]

        sub_x = _gather_subgraph_features_batch(data.x, subgraphs, chunk_indices)
        chunk_w = sub_weights[chunk_indices.cpu()
                              if chunk_indices.is_cuda
                              else chunk_indices].to(device)
        chunk_labels = batch_labels[start:end]

        chunk_eg = ft_per_sample_grad(params, sub_x, chunk_w, chunk_labels)
        chunks_grads.append(chunk_eg)

        del sub_x, chunk_w
        torch.cuda.empty_cache()

    per_example_grads = {}
    for name in chunks_grads[0]:
        per_example_grads[name] = torch.cat(
            [c[name] for c in chunks_grads], dim=0)

    return per_example_grads


def _precompute_subgraph_weights(
    train_subgraphs: torch.Tensor,
    adjacency_normalization: Optional[str],
) -> torch.Tensor:
    """Vectorized weight vectors for root message aggregation.

    For each training node's subgraph, computes the weight vector used
    to aggregate neighbor features to the root node.  All valid edges
    in a subgraph share the same weight because the root is the only
    sender in the subgraph edge structure.

    Returns:
        weights: [N_train, pad_to + 1] (last col is padding node, always 0)
    """
    _N, pad_to = train_subgraphs.shape
    valid_mask = (train_subgraphs != _SUBGRAPH_PADDING_VALUE)
    num_valid = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

    weights = torch.zeros(_N, pad_to + 1)
    if adjacency_normalization is None:
        weights[:, :pad_to] = valid_mask.float()
    elif adjacency_normalization == 'inverse-degree':
        weights[:, :pad_to] = valid_mask.float() / num_valid
    elif adjacency_normalization == 'inverse-sqrt-degree':
        weights[:, :pad_to] = valid_mask.float() / num_valid.sqrt()
    else:
        raise ValueError(f'Unknown adj norm: {adjacency_normalization}')

    return weights


def _gather_subgraph_features_batch(
    data_x: torch.Tensor,
    subgraphs: torch.Tensor,
    node_indices: torch.Tensor,
) -> torch.Tensor:
    """Vectorized feature gathering for a batch of subgraphs.

    Returns:
        sub_x: [B, pad_to + 1, F] — last node per subgraph is zero-padding.
    """
    device = data_x.device
    node_indices_cpu = (node_indices.cpu()
                        if node_indices.is_cuda else node_indices)
    batch_sg = subgraphs[node_indices_cpu]

    valid_mask = (batch_sg != _SUBGRAPH_PADDING_VALUE)
    safe_sg = batch_sg.clone()
    safe_sg[~valid_mask] = 0

    sub_x = data_x[safe_sg.to(device)]
    sub_x = sub_x * valid_mask.to(device).unsqueeze(-1).float()

    B, pad_to, F = sub_x.shape
    padding = torch.zeros(B, 1, F, device=device, dtype=sub_x.dtype)
    sub_x = torch.cat([sub_x, padding], dim=1)
    return sub_x


def _get_layer_ids(params, prefix):
    """Extract sorted integer layer IDs from a parameter dict."""
    ids = set()
    for k in params:
        if k.startswith(prefix):
            rest = k[len(prefix):]
            lid = rest.split('.')[0]
            if lid.isdigit():
                ids.add(int(lid))
    return sorted(ids)


def _gcn_forward_root_only(params, sub_x, sub_w, activation):
    """Functional GCN forward pass optimised for root-node output.

    Only supports 1-hop message passing (num_message_passing_steps == 1).
    sub_x: [pad_to+1, F], sub_w: [pad_to+1]
    Returns root logits: [C]
    """
    h = sub_x

    for lid in _get_layer_ids(params, 'encoder.layers.'):
        w = params[f'encoder.layers.{lid}.weight']
        b = params[f'encoder.layers.{lid}.bias']
        h = h @ w.t() + b
        h = activation(h)

    h_root = sub_w @ h
    for hop_id in _get_layer_ids(params, 'core_hops.'):
        w = params[f'core_hops.{hop_id}.update_fn.layers.0.weight']
        b = params[f'core_hops.{hop_id}.update_fn.layers.0.bias']
        h_root_new = h_root @ w.t() + b
        h_root_new = activation(h_root_new)
        if h_root_new.shape == h_root.shape:
            h_root_new = h_root_new + h_root
        h_root = h_root_new

    h = h_root
    decoder_ids = _get_layer_ids(params, 'decoder.layers.')
    for i, lid in enumerate(decoder_ids):
        w = params[f'decoder.layers.{lid}.weight']
        b = params[f'decoder.layers.{lid}.bias']
        h = h @ w.t() + b
        if i < len(decoder_ids) - 1:
            h = activation(h)

    return h


def _clip_and_sum_gcn_vmap(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    sub_weights: torch.Tensor,
    node_indices: torch.Tensor,
    l2_norms_threshold: Dict[str, float],
    chunk_size: int = 500,
) -> Dict[str, torch.Tensor]:
    """Memory-efficient per-example clip-and-sum for GCN using vmap.

    Uses vectorized feature gathering and weight-vector-based message
    passing (root-only forward) to avoid building subgraphs one-by-one.
    """
    device = data.x.device
    B = len(node_indices)
    node_indices_cpu = node_indices.cpu() if node_indices.is_cuda else node_indices
    batch_labels = labels[node_indices_cpu].to(device)
    activation = model.encoder.activation

    params = dict(model.named_parameters())

    def single_loss(params, sub_x_i, sub_w_i, label_i):
        logits = _gcn_forward_root_only(params, sub_x_i, sub_w_i, activation)
        return F.cross_entropy(logits.unsqueeze(0), label_i.unsqueeze(0))

    ft_grad = vmap(grad(single_loss), in_dims=(None, 0, 0, 0))

    clipped_sum = None

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_indices = node_indices[start:end]

        sub_x = _gather_subgraph_features_batch(data.x, subgraphs, chunk_indices)
        chunk_w = sub_weights[chunk_indices.cpu()
                              if chunk_indices.is_cuda
                              else chunk_indices].to(device)
        chunk_labels = batch_labels[start:end]

        chunk_grads = ft_grad(params, sub_x, chunk_w, chunk_labels)
        chunk_clipped = dp_optimizers.clip_by_norm(chunk_grads, l2_norms_threshold)

        if clipped_sum is None:
            clipped_sum = {k: v.sum(dim=0) for k, v in chunk_clipped.items()}
        else:
            for k in clipped_sum:
                clipped_sum[k] += chunk_clipped[k].sum(dim=0)

        del sub_x, chunk_w, chunk_grads, chunk_clipped
        torch.cuda.empty_cache()

    return clipped_sum


def compute_updates_for_dp(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    node_indices: torch.Tensor,
    sub_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Per-example gradient computation for DP training.

    Dispatches to optimized vmap implementations for both MLP and GCN.
    """
    if isinstance(model, models.GraphMultiLayerPerceptron):
        return _compute_per_example_grads_mlp_vmap(
            model, data, labels, subgraphs, node_indices)
    else:
        return _compute_per_example_grads_gcn_vmap(
            model, data, labels, subgraphs, sub_weights,
            node_indices)


# ---------------------------------------------------------------------------
# Clipping threshold estimation
# ---------------------------------------------------------------------------

def estimate_clipping_thresholds(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    subgraphs: torch.Tensor,
    estimation_indices: torch.Tensor,
    l2_norm_clip_percentile: float,
    sub_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Estimates per-layer gradient clipping thresholds from a sample."""
    per_eg_grads = compute_updates_for_dp(
        model, data, labels, subgraphs, estimation_indices,
        sub_weights=sub_weights)

    thresholds = {}
    for name, grad in per_eg_grads.items():
        B = grad.shape[0]
        flat = grad.reshape(B, -1)
        norms = torch.linalg.norm(flat, dim=1)
        threshold = float(np.percentile(norms.detach().cpu().numpy(),
                                        l2_norm_clip_percentile))
        thresholds[name] = max(threshold, 1e-8)
    return thresholds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """Returns (loss, accuracy) on the masked subset.

    Matches reference: per-element softmax cross-entropy, masked mean.
    """
    with torch.no_grad():
        loss_per_node = F.cross_entropy(logits, labels, reduction='none')
        loss_masked = torch.where(mask, loss_per_node, torch.zeros_like(loss_per_node))
        loss = loss_masked.sum() / mask.float().sum()

        preds = logits.argmax(dim=-1)
        targets = labels.argmax(dim=-1)
        correct = (preds == targets).float()
        correct_masked = torch.where(mask, correct, torch.zeros_like(correct))
        accuracy = correct_masked.sum() / mask.float().sum()
    return loss.item(), accuracy.item()


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    train_loss, train_acc = evaluate_predictions(logits, labels, masks['train'])
    val_loss, val_acc = evaluate_predictions(logits, labels, masks['validation'])
    test_loss, test_acc = evaluate_predictions(logits, labels, masks['test'])
    return {
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(config, workdir: str = '/tmp/dp_gnn'):
    """Full training and evaluation loop.

    Matches reference step counting:
    - initial_step = 1 (after init)
    - loop: range(initial_step, config.num_training_steps)
    - privacy accountant called with step+1
    - evaluate when step % evaluate_every_steps == 0 or is_last_step
    """
    device = torch.device(getattr(config, 'device', 'cpu'))
    print(f'Using device: {device}')

    torch.manual_seed(config.rng_seed)
    rng = torch.Generator()
    rng.manual_seed(config.rng_seed)

    # Load dataset (on CPU first, then move)
    data, labels_int, masks = input_pipeline.get_dataset(config, rng)
    labels = F.one_hot(labels_int, config.num_classes).float()

    train_mask = masks['train']
    train_indices = torch.where(train_mask)[0]
    num_training_nodes = len(train_indices)
    train_labels = labels[train_indices]

    print(f'Num training nodes: {num_training_nodes}')
    print(f'Num total nodes: {data.num_nodes}')
    print(f'Num edges: {data.edge_index.size(1)}')
    print(f'Feature dim: {data.x.shape[1]}')

    # Subgraphs for DP (keep on CPU — indexed per batch)
    if config.differentially_private_training:
        print('Building subgraphs...')
        subgraphs = get_subgraphs(data, pad_to=config.pad_subgraphs_to)
        train_subgraphs = subgraphs[train_indices]
        del subgraphs
        print('Pre-computing subgraph weights...')
        sub_weights = _precompute_subgraph_weights(
            train_subgraphs, config.adjacency_normalization)
        print('Subgraphs built.')
    else:
        train_subgraphs = None
        sub_weights = None

    # Privacy accountant
    max_terms = compute_max_terms_per_node(config)
    training_privacy_accountant = privacy_accountants.get_training_privacy_accountant(
        config, num_training_nodes, max_terms)

    # Move data to device
    data = data.to(device)
    labels = labels.to(device)
    train_labels = train_labels.to(device)
    train_indices = train_indices.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    # Create model on device
    input_dim = data.x.shape[1]
    model = create_model(config, input_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params}')

    # Estimate clipping thresholds for DP
    dp_gen = torch.Generator()
    dp_gen.manual_seed(config.rng_seed + 1)

    if config.differentially_private_training:
        estimation_indices = train_indices[:config.num_estimation_samples]
        print(f'Estimating clipping thresholds from {len(estimation_indices)} samples...')
        l2_norms_threshold = estimate_clipping_thresholds(
            model, data, train_labels, train_subgraphs,
            estimation_indices, config.l2_norm_clip_percentile,
            sub_weights=sub_weights)
        base_sensitivity = compute_base_sensitivity(config)
        print(f'Clipping thresholds: {l2_norms_threshold}')
        print(f'Base sensitivity: {base_sensitivity}')
    else:
        l2_norms_threshold = None
        base_sensitivity = None

    # Optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate,
            momentum=config.momentum, nesterov=config.nesterov)
    else:
        raise ValueError(f'Unsupported optimizer: {config.optimizer}')

    # Privacy budget
    max_training_epsilon = None
    if config.differentially_private_training:
        max_training_epsilon = getattr(config, 'max_training_epsilon', None)

    # Initial metrics
    model.eval()
    with torch.no_grad():
        logits = compute_logits(model, data)
    init_metrics = compute_metrics(logits, labels, masks)
    init_metrics['epsilon'] = 0
    _log_metrics(0, init_metrics, postfix='_after_init')

    # Training loop - matches reference: range(initial_step, num_training_steps)
    initial_step = 1
    for step in range(initial_step, config.num_training_steps):
        model.train()

        # Sample batch indices (CPU generator, then move)
        step_gen = torch.Generator()
        step_gen.manual_seed(config.rng_seed * 1000 + step)
        batch_idx = torch.randint(
            num_training_nodes, (config.batch_size,), generator=step_gen)
        batch_idx = batch_idx.to(device)

        if config.differentially_private_training:
            if isinstance(model, models.GraphMultiLayerPerceptron):
                clipped_sum = _clip_and_sum_mlp_vmap(
                    model, data, train_labels, train_subgraphs, batch_idx,
                    l2_norms_threshold)
            else:
                clipped_sum = _clip_and_sum_gcn_vmap(
                    model, data, train_labels, train_subgraphs, sub_weights,
                    batch_idx, l2_norms_threshold)

            noisy_grads = {}
            for name, summed in clipped_sum.items():
                noise_std = (l2_norms_threshold[name] * base_sensitivity
                             * config.training_noise_multiplier)
                if noise_std > 0 and np.isfinite(noise_std):
                    noise = torch.normal(
                        mean=0.0, std=noise_std, size=summed.shape,
                        generator=dp_gen, dtype=summed.dtype,
                    ).to(summed.device)
                    noisy_grads[name] = summed + noise
                else:
                    noisy_grads[name] = summed

            optimizer.zero_grad()
            for name, p in model.named_parameters():
                if name in noisy_grads:
                    p.grad = noisy_grads[name] / config.batch_size
            optimizer.step()
        else:
            grads = compute_updates(model, data, train_labels, batch_idx,
                                    train_indices=train_indices)
            optimizer.zero_grad()
            for name, p in model.named_parameters():
                if name in grads:
                    p.grad = grads[name]
            optimizer.step()

        # Evaluate periodically (matches reference step counting)
        is_last_step = (step == config.num_training_steps - 1)
        if step % config.evaluate_every_steps == 0 or is_last_step:
            training_epsilon = training_privacy_accountant(step + 1)
            if max_training_epsilon is not None and training_epsilon >= max_training_epsilon:
                print(f'Privacy budget exhausted at step {step}, '
                      f'epsilon={training_epsilon:.4f}')
                break

            model.eval()
            with torch.no_grad():
                logits = compute_logits(model, data)
            step_metrics = compute_metrics(logits, labels, masks)
            step_metrics['epsilon'] = training_epsilon
            _log_metrics(step, step_metrics)

    return model


def _log_metrics(step: int, metrics: Dict[str, float], postfix: str = ''):
    """Logs metrics, formatting accuracy as percentage (matching reference)."""
    display = {}
    for key, val in metrics.items():
        display_key = key + postfix
        if 'accuracy' in key:
            display[display_key] = val * 100
        else:
            display[display_key] = val

    parts = [f'num_steps: {step:>3d}']
    for key, val in display.items():
        parts.append(f'{key}: {val:.4f}')
    print(', '.join(parts))
