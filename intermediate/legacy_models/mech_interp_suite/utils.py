"""Shared utilities for the mechanistic interpretability suite.

Provides:
- HookManager: context-manager that registers forward hooks, captures
  activations, and auto-removes hooks on exit.
- generate_gaussian_data: creates synthetic Gaussian noise datasets for
  both classification and regression sanity checks.
- print_activation_summary: pretty-prints shape, dtype, mean, std of
  each captured activation tensor.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Hook Manager
# ---------------------------------------------------------------------------

class HookManager:
    """Context-manager that registers forward hooks on named modules and
    stores their output activations.

    Usage::

        targets = {"layer_name": model.some_layer, ...}
        with HookManager(targets) as hm:
            model(inputs)
        acts = hm.activations  # {"layer_name": output_tensor, ...}

    For modules that return tuples, only the first element is stored by
    default.  Set ``capture_index=None`` to store the raw return value.
    """

    def __init__(
        self,
        targets: Dict[str, nn.Module],
        capture_index: Optional[int] = 0,
        detach: bool = True,
    ):
        """
        Args:
            targets: mapping from human-readable name to the nn.Module to hook.
            capture_index: if the module's forward returns a tuple, store only
                ``output[capture_index]``.  Set to ``None`` to store the raw output.
            detach: whether to ``.detach().cpu()`` captured tensors (saves GPU
                memory during analysis).
        """
        self.targets = targets
        self.capture_index = capture_index
        self.detach = detach
        self.activations: Dict[str, torch.Tensor] = OrderedDict()
        self._handles: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str):
        """Create a closure that writes into ``self.activations[name]``."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and self.capture_index is not None:
                val = output[self.capture_index]
            else:
                val = output
            if isinstance(val, torch.Tensor) and self.detach:
                val = val.detach().cpu()
            self.activations[name] = val
        return hook_fn

    def __enter__(self):
        self.activations.clear()
        for name, module in self.targets.items():
            h = module.register_forward_hook(self._make_hook(name))
            self._handles.append(h)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False  # don't suppress exceptions


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_gaussian_data(
    n_train: int = 50,
    n_test: int = 10,
    n_features: int = 5,
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pure Gaussian noise data for sanity checking.

    Returns:
        X_train, y_train, X_test, y_test — all NumPy arrays.
        For classification: y values are integers in [0, n_classes).
        Regression targets are standard-normal floats.
    """
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
    y_test = rng.integers(0, n_classes, size=n_test).astype(np.int64)
    return X_train, y_train, X_test, y_test


def generate_gaussian_data_regression(
    n_train: int = 50,
    n_test: int = 10,
    n_features: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate pure Gaussian noise data for regression sanity checking."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_train = rng.standard_normal(n_train).astype(np.float32)
    y_test = rng.standard_normal(n_test).astype(np.float32)
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Pretty-print activations
# ---------------------------------------------------------------------------

def print_activation_summary(activations: Dict[str, torch.Tensor], title: str = ""):
    """Print a formatted summary table of captured activations."""
    sep = "=" * 100
    print(f"\n{sep}")
    if title:
        print(f"  ACTIVATION SUMMARY: {title}")
        print(sep)
    print(f"  {'Name':<45} {'Shape':<25} {'Dtype':<12} {'Mean':>10} {'Std':>10} {'NaN?':>6}")
    print("-" * 100)
    for name, tensor in activations.items():
        if isinstance(tensor, torch.Tensor):
            t = tensor.float()
            shape_str = str(tuple(tensor.shape))
            dtype_str = str(tensor.dtype).replace("torch.", "")
            mean_val = f"{t.mean().item():.4f}"
            std_val = f"{t.std().item():.4f}"
            has_nan = "YES" if torch.isnan(tensor).any() else "no"
        else:
            shape_str = str(type(tensor).__name__)
            dtype_str = "N/A"
            mean_val = "N/A"
            std_val = "N/A"
            has_nan = "N/A"
        print(f"  {name:<45} {shape_str:<25} {dtype_str:<12} {mean_val:>10} {std_val:>10} {has_nan:>6}")
    print(sep)


def validate_activations(activations: Dict[str, torch.Tensor], model_name: str) -> bool:
    """Validate that activations are sane (no NaN/Inf, expected types).

    Returns True if all checks pass, False otherwise.
    """
    all_ok = True
    for name, tensor in activations.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"  [WARN] {model_name}/{name}: not a tensor ({type(tensor).__name__})")
            continue
        if torch.isnan(tensor).any():
            print(f"  [FAIL] {model_name}/{name}: contains NaN!")
            all_ok = False
        if torch.isinf(tensor).any():
            print(f"  [FAIL] {model_name}/{name}: contains Inf!")
            all_ok = False
        if tensor.ndim == 0:
            print(f"  [WARN] {model_name}/{name}: scalar tensor (ndim=0)")
    return all_ok
