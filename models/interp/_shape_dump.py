"""Empirical shape dump: load each model, hook every top-level block,
run ONE forward pass on a tiny synthetic tabular task, and print
module name, module class, output shape, and dtype for every hook point.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood")
from models import load, SPECS, ROSTER, SCALE_BAR


def dump_one(model_id: str, task: str, device: str = "cuda"):
    print(f"\n{'='*72}")
    print(f"MODEL: {model_id}  TASK: {task}  DEVICE: {device}")
    print(f"{'='*72}")

    s = SPECS[model_id]
    if task == "classification" and not s.supports_classification:
        print(f"  SKIPPED: no classifier available ({s.known_issues})")
        return
    if task == "regression" and not s.supports_regression:
        print(f"  SKIPPED: no regressor available ({s.known_issues})")
        return

    try:
        m = load(model_id, task=task, device=device)
    except Exception as exc:
        print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
        return

    print(f"  n_layers={m.n_layers}  d_model={m.d_model}  stream_view={m.stream_view.value}")
    print(f"  estimator type: {type(m.estimator).__name__}")

    # Fit on a tiny synthetic task
    rng = np.random.RandomState(42)
    n_train, n_test, d = 20, 5, 3
    X_train = rng.randn(n_train, d)
    X_test = rng.randn(n_test, d)
    if task == "regression":
        y_train = rng.randn(n_train)
    else:
        y_train = np.array([0, 1] * 10)  # binary

    m._fit(X_train, y_train)
    model = m._torch_model()

    # Print all named_modules (first 2 levels)
    print(f"\n  --- named_modules (first 100) ---")
    for i, (name, mod) in enumerate(model.named_modules()):
        if i >= 100:
            print(f"  ... ({sum(1 for _ in model.named_modules())} total modules)")
            break
        depth = name.count('.')
        if depth <= 2:
            print(f"    {name!r:50s}  {type(mod).__name__}")

    # Hook every top-level ICL block
    blocks = list(m._blocks()) if hasattr(m, '_blocks') else []
    hooks = []
    captured = {}

    def make_hook(block_idx):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                captured[f"block_{block_idx}"] = {
                    "class": type(module).__name__,
                    "shape": tuple(output.shape),
                    "dtype": str(output.dtype),
                }
            elif isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                captured[f"block_{block_idx}"] = {
                    "class": type(module).__name__,
                    "shape": tuple(output[0].shape),
                    "dtype": str(output[0].dtype),
                    "n_outputs": len(output),
                }
        return hook_fn

    for i, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    # Run forward pass
    try:
        from models.interp import CaptureSpec
        trace = m.run(X_train, y_train, X_test, capture=CaptureSpec(resid=True, attention=False))
        print(f"\n  --- Forward pass succeeded ---")
        print(f"  trace.pred shape: {np.asarray(trace.pred).shape}")
        print(f"  trace.layout: prefix={trace.layout.n_prefix} support={trace.layout.n_support} query={trace.layout.n_query}")
        for layer in sorted(trace.resid.keys()):
            print(f"  trace.resid[{layer}] shape: {trace.resid[layer].shape}")
    except Exception as exc:
        print(f"  FORWARD FAILED: {type(exc).__name__}: {exc}")

    # Print hooked outputs
    print(f"\n  --- Hooked block outputs ---")
    for name in sorted(captured.keys(), key=lambda x: int(x.split('_')[1])):
        info = captured[name]
        print(f"    {name:15s}  class={info['class']:30s}  shape={str(info['shape']):30s}  dtype={info['dtype']}")
        if 'n_outputs' in info:
            print(f"                    (tuple output with {info['n_outputs']} elements)")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Clean up
    del m, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_models = list(ROSTER) + list(SCALE_BAR)
    for model_id in all_models:
        for task in ["regression"]:
            dump_one(model_id, task, device)
    # Also try classification where available
    for model_id in all_models:
        s = SPECS[model_id]
        if s.supports_classification:
            dump_one(model_id, "classification", device)
