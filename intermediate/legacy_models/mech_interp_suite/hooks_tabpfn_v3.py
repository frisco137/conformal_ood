#!/usr/bin/env python3
"""Mechanistic interpretability hooks for TabPFN v3.

Demonstrates how to extract activations from every major stage of the
TabPFN v3 pipeline using PyTorch forward hooks.

Architecture overview (TabPFN v3):
  Stage 0: Cell Embedding       — x_embed (Linear: feature_group_size → embed_dim=128)
  Stage 1: Distribution Embedder — 3× InducedSelfAttentionBlock (128 inducing points, SSMax)
  Stage 2: Column Aggregator     — 3× TransformerBlock + CLS readout (4 CLS tokens, RoPE)
  Stage 3: ICL Transformer       — 24× ICLTransformerBlock (train-only KV, SSMax, dim=512)
  Decoder: ManyClassDecoder (attention, 6 heads) or MLP (regression)

Usage:
    python models/mech_interp_suite/hooks_tabpfn_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure models/ is importable
_models_dir = str(Path(__file__).resolve().parent.parent)
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
_project_dir = str(Path(__file__).resolve().parent.parent.parent)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

import torch
import numpy as np
from collections import OrderedDict

from mech_interp_suite.utils import (
    HookManager,
    generate_gaussian_data,
    print_activation_summary,
    validate_activations,
)


def get_inner_model(clf):
    """Extract the raw TabPFNV3 nn.Module from the sklearn-like wrapper.

    The wrapper stores the model at ``clf.model_`` after fit() is called.
    This returns the Architecture subclass (TabPFNV3 or TabPFNV2p6).
    """
    if not hasattr(clf, "models_"):
        raise RuntimeError(
            "The classifier has not been fitted yet. Call clf.fit() first."
        )
    return clf.models_[0]


def build_hook_targets(model) -> OrderedDict:
    """Build an OrderedDict of (name → nn.Module) for key layers.

    This maps human-readable names to the actual PyTorch modules inside
    the TabPFNV3 architecture so we can attach forward hooks to them.
    """
    targets = OrderedDict()

    # Stage 0: Cell embedding
    targets["stage0_cell_embed"] = model.x_embed

    # Stage 1: Feature Distribution Embedder (per-column induced self-attention)
    for i, layer in enumerate(model.feature_distribution_embedder.layers):
        targets[f"stage1_isab_{i}_cross_attn1"] = layer.cross_attn_block1
        targets[f"stage1_isab_{i}_cross_attn2"] = layer.cross_attn_block2

    # Stage 2: Column Aggregator (cross-feature transformer + CLS readout)
    for i, block in enumerate(model.column_aggregator.blocks):
        targets[f"stage2_col_agg_block_{i}_attn"] = block.attention
        targets[f"stage2_col_agg_block_{i}_mlp"] = block.mlp

    # Stage 3: ICL Transformer
    for i, block in enumerate(model.icl_blocks):
        targets[f"stage3_icl_block_{i}_attn"] = block.icl_attention
        targets[f"stage3_icl_block_{i}_mlp"] = block.mlp

    # Output norm
    targets["output_norm"] = model.output_norm

    # Decoder
    if hasattr(model, "many_class_decoder"):
        targets["decoder_many_class"] = model.many_class_decoder
    if hasattr(model, "output_projection"):
        targets["decoder_output_proj"] = model.output_projection

    return targets


def run_hooks_tabpfn_v3(device: str = "cpu"):
    """Load TabPFN v3 classifier, register hooks, run forward pass, print summary."""
    print("\n" + "=" * 80)
    print("  TabPFN v3 — Mechanistic Interpretability Hook Demo")
    print("=" * 80)

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_gaussian_data(
        n_train=50, n_test=10, n_features=5, n_classes=2, seed=42
    )
    print(f"\n  Data: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")

    # 2. Load model
    print("  Loading TabPFN v3 Classifier...")
    from models.tabpfn_v3 import get_classifier
    clf = get_classifier(device=device)
    clf.fit(X_train, y_train)

    # 3. Get inner model
    inner_model = get_inner_model(clf)
    print(f"  Inner model type: {type(inner_model).__name__}")

    # 4. Build targets and register hooks
    targets = build_hook_targets(inner_model)
    print(f"  Registered {len(targets)} hook targets")

    # 5. Run forward pass with hooks
    with HookManager(targets) as hm:
        _ = clf.predict_proba(X_test)

    # 6. Print summary
    print_activation_summary(hm.activations, title="TabPFN v3")

    # 7. Validate
    ok = validate_activations(hm.activations, "TabPFN v3")
    status = "PASSED" if ok else "FAILED"
    print(f"\n  Validation: {status}")
    print(f"  Total activations captured: {len(hm.activations)}")

    return hm.activations, ok


if __name__ == "__main__":
    activations, ok = run_hooks_tabpfn_v3()
    sys.exit(0 if ok else 1)
