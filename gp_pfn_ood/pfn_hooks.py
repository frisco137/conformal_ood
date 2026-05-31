import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

# Ensure parent directory is in path to import transformer modules from the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))
import transformer
import encoders
import positional_encodings

class PFNHookManager:
    """
    Manages loading a pre-trained PFN model, registering forward hooks on its
    TransformerEncoderLayers and MultiheadAttention modules, and caching activations
    and attention weights during a forward pass.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load the model
        print(f"Loading pre-trained PFN from {checkpoint_path}...")
        loaded_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(loaded_data, dict):
            print("Loaded a state dictionary. Initializing model architecture...")
            emsize = 256
            nhead = 4
            nhid = 512
            nlayers = 6
            seq_len = 100

            encoder_module = encoders.Linear(1, emsize)
            y_encoder_module = encoders.Linear(1, emsize)
            pos_encoder_module = positional_encodings.PositionalEncoding(emsize, seq_len * 2)

            self.model = transformer.TransformerModel(
                encoder=encoder_module,
                n_out=1,
                ninp=emsize,
                nhead=nhead,
                nhid=nhid,
                nlayers=nlayers,
                dropout=0.0,
                y_encoder=y_encoder_module,
                pos_encoder=pos_encoder_module
            )
            self.model.load_state_dict(loaded_data)
        else:
            self.model = loaded_data

        # Apply compatibility fixes for newer PyTorch versions
        self._apply_compatibility_patches()

        self.model.eval()
        self.model.to(device)

        # Dictionary to store list of detached tensor outputs per layer
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.attention_maps: Dict[int, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []

        # Set up instrumentation hooks
        self._register_hooks()

    def _apply_compatibility_patches(self):
        """
        Ensures compatibility with newer PyTorch versions by adding missing attributes
        norm_first and approximate to TransformerEncoderLayer and GELU modules if absent.
        """
        for m in self.model.modules():
            if m.__class__.__name__ == 'TransformerEncoderLayer':
                if not hasattr(m, 'norm_first'):
                    m.norm_first = False
            elif m.__class__.__name__ == 'GELU':
                if not hasattr(m, 'approximate'):
                    m.approximate = 'none'

    def _register_hooks(self):
        """
        Monkeypatches the MultiheadAttention modules to output attention maps,
        and registers forward hooks on both the layers and attention modules.
        """
        # Force MultiheadAttention to output attention weights by intercepting forward calls
        for m in self.model.modules():
            if m.__class__.__name__ == 'MultiheadAttention':
                orig_forward = m.forward
                def make_new_forward(orig):
                    def new_forward(query, key, value, *args, **kwargs):
                        # Force need_weights to True so attention maps are returned in output tuple
                        kwargs['need_weights'] = True
                        return orig(query, key, value, *args, **kwargs)
                    return new_forward
                m.forward = make_new_forward(orig_forward)

        # Register forward hooks for each encoder layer
        for idx, layer in enumerate(self.model.transformer_encoder.layers):
            self.activations[idx] = []
            self.attention_maps[idx] = []

            # Helper to bind layer index to hook function scope
            def make_layer_hook(l_idx: int):
                def layer_hook(module: nn.Module, input_args: Any, output_val: torch.Tensor):
                    # output_val has shape: [seq_len, batch_size, hidden_dim]
                    # We detach and move to CPU immediately to avoid memory accumulation
                    self.activations[l_idx].append(output_val.detach().cpu())
                return layer_hook

            h_layer = layer.register_forward_hook(make_layer_hook(idx))
            self.hooks.append(h_layer)

            # Helper to bind layer index to self-attention hook function scope
            def make_sa_hook(l_idx: int):
                def sa_hook(module: nn.Module, input_args: Any, output_val: Tuple[torch.Tensor, torch.Tensor]):
                    # output_val is a tuple: (attn_output, attn_output_weights)
                    # attn_output_weights has shape: [batch_size, seq_len, seq_len]
                    if isinstance(output_val, tuple) and len(output_val) > 1 and output_val[1] is not None:
                        self.attention_maps[l_idx].append(output_val[1].detach().cpu())
                return sa_hook

            h_sa = layer.self_attn.register_forward_hook(make_sa_hook(idx))
            self.hooks.append(h_sa)

    def clear_cache(self):
        """
        Clears all cached activations and attention maps.
        """
        for idx in self.activations:
            self.activations[idx] = []
            self.attention_maps[idx] = []

    def remove_hooks(self):
        """
        Removes all registered forward hooks.
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_aggregated_features(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Concatenates all batch lists into unified tensors and averages the activations
        over the sequence length dimension.

        Returns:
        - layer_activations: Dict of {layer_idx: Tensor of shape (total_samples, hidden_dim)}
        - layer_attention: Dict of {layer_idx: Tensor of shape (total_samples, seq_len, seq_len)}
        """
        layer_activations: Dict[int, torch.Tensor] = {}
        layer_attention: Dict[int, torch.Tensor] = {}

        for idx in self.activations:
            if len(self.activations[idx]) == 0:
                continue

            # Concatenate list of shape [seq_len, batch_size, hidden_dim] along the batch dimension (dim 1)
            # Resulting shape: [seq_len, total_samples, hidden_dim]
            merged_act = torch.cat(self.activations[idx], dim=1)

            # Compute the sequence-mean representation: average across the sequence length (dim 0)
            # Resulting shape: [total_samples, hidden_dim]
            layer_activations[idx] = torch.mean(merged_act, dim=0)

            # Concatenate list of shape [batch_size, seq_len, seq_len] along the batch dimension (dim 0)
            # Resulting shape: [total_samples, seq_len, seq_len]
            layer_attention[idx] = torch.cat(self.attention_maps[idx], dim=0)

        return layer_activations, layer_attention
