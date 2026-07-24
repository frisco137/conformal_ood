#!/usr/bin/env python3
"""Mechanistic interpretability hooks for Google TabFM.

Demonstrates how to extract activations from every major component of the
Google TabFM pipeline using PyTorch forward hooks.

Architecture overview (Google TabFM v1.0.0):
  CellEmbedder:
      Feature grouping (circular shift, group_size=3)
      Fourier frequency expansion (32 frequencies, separate num/cat linear)
      y_embedder_lookup (Embedding for clf, MLP for reg), target-aware
  ColEmbedding (col_embedder):
      SetTransformer: 3× InducedSelfAttentionBlock (SwiGLU FFN, 256 inducing pts)
      Output: RMSNorm + Linear
  RowInteraction (row_interactor):
      8 CLS tokens prepended
      Encoder: 3× MultiheadAttentionBlock (RoPE, PerDimScale, q/k RMSNorm, SwiGLU)
      Output: full (output_full=True) → [B,T,HC,E]
  ColEmbedding_2 (col_embedder_2):
      Second round of SetTransformer (same architecture as col_embedder)
  RowInteraction_2 (row_interactor_2):
      Same encoder but output_full=False → CLS tokens only → ICL dim = 8 × 256 = 2048
  ICLearning (icl_predictor):
      Encoder: 24× MultiheadAttentionBlock (NO RoPE, PerDimScale, SwiGLU)
      y_encoder: OneHot+Linear (clf) or MLP (reg)
      Output: RMSNorm → MLP decoder

Usage:
    python models/mech_interp_suite/hooks_google_tabfm.py
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
    """Extract the raw TabFM nn.Module from the sklearn-like wrapper.

    TabFMClassifier stores the model at ``clf.model``.
    """
    if not hasattr(clf, "model"):
        raise RuntimeError("No model found on the classifier.")
    return clf.model


def build_hook_targets(model) -> OrderedDict:
    """Build hook targets for Google TabFM architecture."""
    targets = OrderedDict()

    # Cell embedding
    targets["cell_embedder"] = model.cell_embedder

    # ColEmbedding 1 — SetTransformer
    targets["col_embedder_out_w"] = model.col_embedder.out_w
    targets["col_embedder_ln_w"] = model.col_embedder.ln_w
    for i, block in enumerate(model.col_embedder.tf_col.blocks):
        targets[f"col1_isab_{i}_mab1"] = block.mab1
        targets[f"col1_isab_{i}_mab2"] = block.mab2

    # RowInteraction 1 — Encoder (with RoPE)
    for i, block in enumerate(model.row_interactor.tf_row.blocks):
        targets[f"row1_mab_{i}"] = block
        targets[f"row1_mab_{i}_attn"] = block.attn
    targets["row1_out_ln"] = model.row_interactor.out_ln

    # ColEmbedding 2 — second SetTransformer
    targets["col_embedder_2_out_w"] = model.col_embedder_2.out_w
    for i, block in enumerate(model.col_embedder_2.tf_col.blocks):
        targets[f"col2_isab_{i}_mab1"] = block.mab1
        targets[f"col2_isab_{i}_mab2"] = block.mab2

    # RowInteraction 2 — Encoder (CLS readout)
    for i, block in enumerate(model.row_interactor_2.tf_row.blocks):
        targets[f"row2_mab_{i}"] = block
        targets[f"row2_mab_{i}_attn"] = block.attn
    targets["row2_out_ln"] = model.row_interactor_2.out_ln

    # ICL Predictor
    targets["icl_y_encoder"] = model.icl_predictor.y_encoder
    for i, block in enumerate(model.icl_predictor.tf_icl.blocks):
        targets[f"icl_mab_{i}"] = block
        targets[f"icl_mab_{i}_attn"] = block.attn
    targets["icl_output_norm"] = model.icl_predictor.ln
    targets["icl_decoder"] = model.icl_predictor.decoder

    return targets


def run_hooks_google_tabfm(device: str = "cpu"):
    """Load TabFM classifier, register hooks, run forward pass, print summary."""
    print("\n" + "=" * 80)
    print("  Google TabFM — Mechanistic Interpretability Hook Demo")
    print("=" * 80)

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_gaussian_data(
        n_train=50, n_test=10, n_features=5, n_classes=2, seed=42
    )
    print(f"\n  Data: X_train={X_train.shape}, y_train={y_train.shape}, "
          f"X_test={X_test.shape}, y_test={y_test.shape}")

    # 2. Load model
    print("  Loading Google TabFM Classifier...")
    from models.google_tabfm import get_classifier
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
    print_activation_summary(hm.activations, title="Google TabFM")

    # 7. Validate
    ok = validate_activations(hm.activations, "Google TabFM")
    status = "PASSED" if ok else "FAILED"
    print(f"\n  Validation: {status}")
    print(f"  Total activations captured: {len(hm.activations)}")

    return hm.activations, ok


if __name__ == "__main__":
    activations, ok = run_hooks_google_tabfm()
    sys.exit(0 if ok else 1)
