"""Parameter transfer utilities for MLP to GCN initialization.

Maps pretrained MLP parameters to GCN components (encoder, decoder, or both)
for pretrain-finetune workflows.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dp_gnn.models import GraphConvolutionalNetwork, GraphMultiLayerPerceptron


def validate_transfer_compatibility(
    mlp_state: Dict[str, torch.Tensor],
    gcn: GraphConvolutionalNetwork,
    transfer_strategy: str,
) -> Tuple[bool, str]:
    """Validate that MLP state can be transferred to GCN.

    Args:
        mlp_state: MLP state_dict.
        gcn: Target GCN model.
        transfer_strategy: One of 'encoder_only', 'classifier_only',
            'encoder_classifier', 'full'.

    Returns:
        (is_compatible, message)
    """
    # Get MLP layer dimensions from state dict
    mlp_input_dim = None
    mlp_hidden_dim = None
    mlp_output_dim = None
    mlp_num_layers = 0

    for key in mlp_state.keys():
        if key.startswith('layers.') and key.endswith('.weight'):
            layer_idx = int(key.split('.')[1])
            weight = mlp_state[key]
            mlp_num_layers = max(mlp_num_layers, layer_idx + 1)

            if layer_idx == 0:
                mlp_input_dim = weight.shape[1]
                mlp_hidden_dim = weight.shape[0]
            elif layer_idx == mlp_num_layers - 1:
                mlp_output_dim = weight.shape[0]

    # Get GCN dimensions
    gcn_input_dim = gcn.encoder.layers[0].weight.shape[1]
    gcn_latent_dim = gcn.encoder.layers[0].weight.shape[0]
    gcn_output_dim = gcn.decoder.layers[-1].weight.shape[0]
    gcn_num_encoder_layers = len(gcn.encoder.layers)
    gcn_num_decoder_layers = len(gcn.decoder.layers)

    # Validate based on strategy
    if transfer_strategy == 'encoder_only':
        if mlp_hidden_dim != gcn_latent_dim:
            return False, (
                f"MLP hidden_dim ({mlp_hidden_dim}) != GCN latent_dim ({gcn_latent_dim})"
            )
        if mlp_input_dim != gcn_input_dim:
            return False, (
                f"MLP input_dim ({mlp_input_dim}) != GCN input_dim ({gcn_input_dim})"
            )
        if mlp_num_layers < gcn_num_encoder_layers:
            return False, (
                f"MLP num_layers ({mlp_num_layers}) < GCN encoder_layers ({gcn_num_encoder_layers})"
            )

    elif transfer_strategy == 'classifier_only':
        if mlp_output_dim != gcn_output_dim:
            return False, (
                f"MLP output_dim ({mlp_output_dim}) != GCN output_dim ({gcn_output_dim})"
            )
        if mlp_hidden_dim != gcn_latent_dim:
            return False, (
                f"MLP hidden_dim ({mlp_hidden_dim}) != GCN latent_dim ({gcn_latent_dim})"
            )

    elif transfer_strategy in ('encoder_classifier', 'full'):
        if mlp_hidden_dim != gcn_latent_dim:
            return False, (
                f"MLP hidden_dim ({mlp_hidden_dim}) != GCN latent_dim ({gcn_latent_dim})"
            )
        if mlp_input_dim != gcn_input_dim:
            return False, (
                f"MLP input_dim ({mlp_input_dim}) != GCN input_dim ({gcn_input_dim})"
            )
        if mlp_output_dim != gcn_output_dim:
            return False, (
                f"MLP output_dim ({mlp_output_dim}) != GCN output_dim ({gcn_output_dim})"
            )
        if transfer_strategy == 'full':
            total_gcn_layers = gcn_num_encoder_layers + gcn_num_decoder_layers
            if mlp_num_layers < total_gcn_layers:
                return False, (
                    f"MLP num_layers ({mlp_num_layers}) < total GCN layers ({total_gcn_layers})"
                )

    else:
        return False, f"Unknown transfer_strategy: {transfer_strategy}"

    return True, "Compatible"


def create_parameter_mapping(
    mlp_state: Dict[str, torch.Tensor],
    gcn: GraphConvolutionalNetwork,
    transfer_strategy: str,
) -> Dict[str, str]:
    """Create mapping from GCN parameter names to MLP parameter names.

    Args:
        mlp_state: MLP state_dict.
        gcn: Target GCN model.
        transfer_strategy: One of 'encoder_only', 'classifier_only',
            'encoder_classifier', 'full'.

    Returns:
        Dict mapping GCN param name -> MLP param name.
    """
    mapping = {}
    gcn_num_encoder_layers = len(gcn.encoder.layers)
    gcn_num_decoder_layers = len(gcn.decoder.layers)

    if transfer_strategy == 'encoder_only':
        # Map MLP layers 0..(encoder_layers-1) to GCN encoder
        for i in range(gcn_num_encoder_layers):
            mapping[f'encoder.layers.{i}.weight'] = f'layers.{i}.weight'
            mapping[f'encoder.layers.{i}.bias'] = f'layers.{i}.bias'

    elif transfer_strategy == 'classifier_only':
        # Map MLP last layer to GCN decoder last layer
        mlp_num_layers = sum(1 for k in mlp_state if k.startswith('layers.') and k.endswith('.weight'))
        mlp_last_idx = mlp_num_layers - 1
        gcn_last_idx = gcn_num_decoder_layers - 1
        mapping[f'decoder.layers.{gcn_last_idx}.weight'] = f'layers.{mlp_last_idx}.weight'
        mapping[f'decoder.layers.{gcn_last_idx}.bias'] = f'layers.{mlp_last_idx}.bias'

    elif transfer_strategy == 'encoder_classifier':
        # Map encoder layers
        for i in range(gcn_num_encoder_layers):
            mapping[f'encoder.layers.{i}.weight'] = f'layers.{i}.weight'
            mapping[f'encoder.layers.{i}.bias'] = f'layers.{i}.bias'

        # Map classifier (last layer)
        mlp_num_layers = sum(1 for k in mlp_state if k.startswith('layers.') and k.endswith('.weight'))
        mlp_last_idx = mlp_num_layers - 1
        gcn_last_idx = gcn_num_decoder_layers - 1
        mapping[f'decoder.layers.{gcn_last_idx}.weight'] = f'layers.{mlp_last_idx}.weight'
        mapping[f'decoder.layers.{gcn_last_idx}.bias'] = f'layers.{mlp_last_idx}.bias'

    elif transfer_strategy == 'full':
        # Map encoder layers
        for i in range(gcn_num_encoder_layers):
            mapping[f'encoder.layers.{i}.weight'] = f'layers.{i}.weight'
            mapping[f'encoder.layers.{i}.bias'] = f'layers.{i}.bias'

        # Map decoder layers (continuing from encoder)
        mlp_start_idx = gcn_num_encoder_layers
        for i in range(gcn_num_decoder_layers):
            mlp_idx = mlp_start_idx + i
            mapping[f'decoder.layers.{i}.weight'] = f'layers.{mlp_idx}.weight'
            mapping[f'decoder.layers.{i}.bias'] = f'layers.{mlp_idx}.bias'

    return mapping


def transfer_parameters(
    mlp_state: Dict[str, torch.Tensor],
    gcn: GraphConvolutionalNetwork,
    transfer_strategy: str = 'encoder_only',
    strict: bool = False,
) -> Tuple[GraphConvolutionalNetwork, Dict[str, List[str]]]:
    """Transfer MLP parameters to GCN.

    Args:
        mlp_state: MLP state_dict from pretrained model.
        gcn: Target GCN model (will be modified in-place).
        transfer_strategy: One of 'encoder_only', 'classifier_only',
            'encoder_classifier', 'full'.
        strict: If True, raise error on validation failure.
            If False, skip transfer and return info.

    Returns:
        (gcn, info_dict) where info_dict contains:
            - 'transferred': list of transferred param names
            - 'skipped': list of skipped param names
            - 'errors': list of error messages
    """
    info = {
        'transferred': [],
        'skipped': [],
        'errors': [],
    }

    # Validate compatibility
    is_compatible, message = validate_transfer_compatibility(
        mlp_state, gcn, transfer_strategy
    )

    if not is_compatible:
        info['errors'].append(message)
        if strict:
            raise ValueError(f"Transfer validation failed: {message}")
        return gcn, info

    # Create parameter mapping
    mapping = create_parameter_mapping(mlp_state, gcn, transfer_strategy)

    # Get GCN state dict
    gcn_state = gcn.state_dict()

    # Transfer parameters
    for gcn_name, mlp_name in mapping.items():
        if mlp_name not in mlp_state:
            info['errors'].append(f"MLP parameter not found: {mlp_name}")
            if strict:
                raise KeyError(f"MLP parameter not found: {mlp_name}")
            continue

        if gcn_name not in gcn_state:
            info['errors'].append(f"GCN parameter not found: {gcn_name}")
            if strict:
                raise KeyError(f"GCN parameter not found: {gcn_name}")
            continue

        # Check shape compatibility
        if mlp_state[mlp_name].shape != gcn_state[gcn_name].shape:
            msg = (
                f"Shape mismatch: {mlp_name} {mlp_state[mlp_name].shape} "
                f"vs {gcn_name} {gcn_state[gcn_name].shape}"
            )
            info['errors'].append(msg)
            if strict:
                raise ValueError(msg)
            continue

        # Transfer parameter
        gcn_state[gcn_name] = mlp_state[mlp_name].clone()
        info['transferred'].append(gcn_name)

    # Load updated state dict
    gcn.load_state_dict(gcn_state)

    return gcn, info


def get_transferable_parameters(
    gcn: GraphConvolutionalNetwork,
    transfer_strategy: str,
) -> List[str]:
    """Get list of GCN parameter names that would be transferred.

    Args:
        gcn: GCN model.
        transfer_strategy: Transfer strategy.

    Returns:
        List of GCN parameter names.
    """
    params = []

    if transfer_strategy in ('encoder_only', 'encoder_classifier', 'full'):
        for i in range(len(gcn.encoder.layers)):
            params.extend([f'encoder.layers.{i}.weight', f'encoder.layers.{i}.bias'])

    if transfer_strategy in ('classifier_only', 'encoder_classifier'):
        last_idx = len(gcn.decoder.layers) - 1
        params.extend([
            f'decoder.layers.{last_idx}.weight',
            f'decoder.layers.{last_idx}.bias',
        ])

    if transfer_strategy == 'full':
        # Add all decoder layers (not just last)
        for i in range(len(gcn.decoder.layers) - 1):  # Exclude last (already added above)
            params.extend([f'decoder.layers.{i}.weight', f'decoder.layers.{i}.bias'])

    return params


def freeze_parameters(
    gcn: GraphConvolutionalNetwork,
    parameter_names: List[str],
) -> None:
    """Freeze specified parameters in GCN.

    Args:
        gcn: GCN model.
        parameter_names: List of parameter names to freeze.
    """
    for name, param in gcn.named_parameters():
        if name in parameter_names:
            param.requires_grad = False


def unfreeze_all_parameters(gcn: GraphConvolutionalNetwork) -> None:
    """Unfreeze all parameters in GCN.

    Args:
        gcn: GCN model.
    """
    for param in gcn.parameters():
        param.requires_grad = True


def load_mlp_into_gcn(
    checkpoint_path: str,
    gcn: GraphConvolutionalNetwork,
    transfer_strategy: str = 'encoder_only',
    device: str = 'cpu',
    strict: bool = False,
) -> Tuple[GraphConvolutionalNetwork, Dict]:
    """Load MLP checkpoint and transfer to GCN.

    Convenience function that loads checkpoint and transfers in one call.

    Args:
        checkpoint_path: Path to MLP checkpoint.
        gcn: Target GCN model.
        transfer_strategy: Transfer strategy.
        device: Device to load checkpoint to.
        strict: If True, raise on validation failure.

    Returns:
        (gcn, info_dict)
    """
    from dp_gnn.checkpoint_utils import load_model_state

    mlp_state = load_model_state(checkpoint_path, device=device)
    return transfer_parameters(mlp_state, gcn, transfer_strategy, strict)
