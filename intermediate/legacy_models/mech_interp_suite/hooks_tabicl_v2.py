#!/usr/bin/env python3
"""Mechanistic interpretability hooks for TabICL v2.

Demonstrates how to extract activations from every major component of the
TabICL v2 pipeline using PyTorch forward hooks.

Architecture overview (TabICL v2):
  ColEmbedding (col_embedder):
      Feature grouping (circular shift, group_size=3)
      3× InducedSelfAttentionBlock (128 inducing points, SSMax qassmax-mlp-elementwise)
      Target-aware embedding (adds y-encoding to train rows)
  RowInteraction (row_interactor):
      4 CLS tokens prepended
      3× MultiheadAttentionBlock (pre-norm, RoPE, GELU FFN)
      Output: CLS tokens only → concatenated to dim=512
  ICLearning (icl_predictor):
      y_encoder: OneHot+Linear (classification) or Linear (regression)
      12× Encoder blocks (MultiheadAttentionBlock, pre-norm, SSMax, train-only KV)
      LayerNorm → MLP decoder

Usage:
    python models/mech_interp_suite/hooks_tabicl_v2.py
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
    """Extract the raw TabICL nn.Module from the sklearn wrapper.

    TabICLClassifier stores the model at ``clf.model_`` after fit().
    """
    if not hasattr(clf, "model_"):
        raise RuntimeError("Call clf.fit() before accessing the inner model.")
    return clf.model_


def build_hook_targets(model) -> OrderedDict:
    """Build hook targets for TabICL v2 architecture."""
    targets = OrderedDict()

    # Stage 1: Column Embedding
    targets["col_embedder"] = model.col_embedder

    # Column embedding internal: InducedSelfAttentionBlocks
    if hasattr(model.col_embedder, "blocks"):
        for i, block in enumerate(model.col_embedder.blocks):
            targets[f"col_isab_{i}_mab1"] = block.multihead_attn1
            targets[f"col_isab_{i}_mab2"] = block.multihead_attn2

    # Stage 2: Row Interaction
    targets["row_interactor"] = model.row_interactor

    # Row interaction internal: Encoder blocks
    if hasattr(model.row_interactor, "encoder"):
        encoder = model.row_interactor.encoder
        if hasattr(encoder, "blocks"):
            for i, block in enumerate(encoder.blocks):
                targets[f"row_block_{i}"] = block
                if hasattr(block, "attn"):
                    targets[f"row_block_{i}_attn"] = block.attn

    # Stage 3: ICL Predictor
    targets["icl_predictor"] = model.icl_predictor

    # ICL y_encoder
    targets["icl_y_encoder"] = model.icl_predictor.y_encoder

    # ICL Encoder blocks
    if hasattr(model.icl_predictor, "tf_icl"):
        icl_enc = model.icl_predictor.tf_icl
        if hasattr(icl_enc, "blocks"):
            for i, block in enumerate(icl_enc.blocks):
                targets[f"icl_block_{i}"] = block
                if hasattr(block, "attn"):
                    targets[f"icl_block_{i}_attn"] = block.attn

    # ICL output norm
    if hasattr(model.icl_predictor, "ln"):
        targets["icl_output_norm"] = model.icl_predictor.ln

    # Decoder
    targets["decoder"] = model.icl_predictor.decoder

    return targets


def run_hooks_tabicl_v2(device: str = "cpu"):
    """Load TabICL v2 classifier, register hooks, run forward pass, print summary."""
    print("\n" + "=" * 80)
    print("  TabICL v2 — Mechanistic Interpretability Hook Demo")
    print("=" * 80)

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_gaussian_data(
        n_train=50, n_test=10, n_features=5, n_classes=2, seed=42
    )
    print(f"\n  Data: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")

    # 2. Load model
    print("  Loading TabICL v2 Classifier...")
    from models.tabicl_v2 import get_classifier
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
    print_activation_summary(hm.activations, title="TabICL v2")

    # 7. Validate
    ok = validate_activations(hm.activations, "TabICL v2")
    status = "PASSED" if ok else "FAILED"
    print(f"\n  Validation: {status}")
    print(f"  Total activations captured: {len(hm.activations)}")

    return hm.activations, ok


if __name__ == "__main__":
    activations, ok = run_hooks_tabicl_v2()
    sys.exit(0 if ok else 1)
