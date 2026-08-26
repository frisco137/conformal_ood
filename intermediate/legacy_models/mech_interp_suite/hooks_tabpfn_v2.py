#!/usr/bin/env python3
"""Mechanistic interpretability hooks for TabPFN v2.

Demonstrates how to extract activations from every major component of the
TabPFN v2 (PerFeatureTransformer) pipeline using PyTorch forward hooks.

Architecture overview (TabPFN v2 — PerFeatureTransformer):
  Encoder Pipeline (encoder):
      [0] RemoveEmptyFeaturesEncoderStep
      [1] NanHandlingEncoderStep
      [2] NormalizeFeatureGroupsEncoderStep
      [3] FeatureTransformEncoderStep
      [4] NormalizeFeatureGroupsEncoderStep
      [5] LinearInputEncoderStep (Linear → embed_dim)
  Target Encoder (y_encoder):
      [0] NanHandlingEncoderStep
      [1] MulticlassClassificationTargetEncoderStep
      [2] LinearInputEncoderStep (Linear → embed_dim)
  Feature Positional Embedding (Linear)
  Transformer Encoder (transformer_encoder.layers) — 12 layer groups, each:
      [0] MultiHeadAttention — row-wise (features attend to each other within a row)
      [1] MultiHeadAttention — column-wise (cells attend to each other within a column)
      [2] MLP — 2-layer GELU feed-forward
      [3] ModuleList of 3 LayerNorms
  Decoder (decoder_dict.standard) — Sequential(Linear, GELU, Linear)

Usage:
    python models/mech_interp_suite/hooks_tabpfn_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    """Extract the raw PerFeatureTransformer nn.Module from the sklearn wrapper."""
    if not hasattr(clf, "models_"):
        raise RuntimeError("Call clf.fit() before accessing the inner model.")
    return clf.models_[0]


def build_hook_targets(model) -> OrderedDict:
    """Build hook targets for TabPFN v2 (PerFeatureTransformer) architecture."""
    targets = OrderedDict()

    # Encoder: the last step is LinearInputEncoderStep with a 'layer' (Linear)
    targets["encoder_linear"] = model.encoder[-1]  # LinearInputEncoderStep

    # Target encoder: the last step is also LinearInputEncoderStep
    targets["y_encoder_linear"] = model.y_encoder[-1]  # LinearInputEncoderStep

    # Feature positional embedding
    targets["feature_pos_embed"] = model.feature_positional_embedding_embeddings

    # Transformer encoder: 12 layer groups
    te_layers = list(model.transformer_encoder.layers.children())
    for gi, group in enumerate(te_layers):
        children = list(group.children())
        # [0] = row-wise MHA, [1] = col-wise MHA, [2] = MLP, [3] = LayerNorms
        if len(children) >= 3:
            targets[f"layer_{gi}_row_attn"] = children[0]   # MultiHeadAttention
            targets[f"layer_{gi}_col_attn"] = children[1]   # MultiHeadAttention
            targets[f"layer_{gi}_mlp"] = children[2]        # MLP
        if len(children) >= 4:
            # LayerNorms
            ln_list = list(children[3].children())
            for li, ln in enumerate(ln_list):
                targets[f"layer_{gi}_ln_{li}"] = ln

    # Decoder
    targets["decoder"] = model.decoder_dict["standard"]

    return targets


def run_hooks_tabpfn_v2(device: str = "cpu"):
    """Load TabPFN v2 classifier, register hooks, run forward pass, print summary."""
    print("\n" + "=" * 80)
    print("  TabPFN v2 — Mechanistic Interpretability Hook Demo")
    print("=" * 80)

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_gaussian_data(
        n_train=50, n_test=10, n_features=5, n_classes=2, seed=42
    )
    print(f"\n  Data: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")

    # 2. Load model
    print("  Loading TabPFN v2 Classifier...")
    from models.tabpfn_v2 import get_classifier
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
    print_activation_summary(hm.activations, title="TabPFN v2")

    # 7. Validate
    ok = validate_activations(hm.activations, "TabPFN v2")
    status = "PASSED" if ok else "FAILED"
    print(f"\n  Validation: {status}")
    print(f"  Total activations captured: {len(hm.activations)}")

    return hm.activations, ok


if __name__ == "__main__":
    activations, ok = run_hooks_tabpfn_v2()
    sys.exit(0 if ok else 1)
