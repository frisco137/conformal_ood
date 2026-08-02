"""Phase 0 regression tests: the vendored TabSwift matches upstream, and the
architecture facts we rely on are the ones upstream actually ships.

Upstream: https://github.com/LAMDA-Tabular/TabSwift @ 8edf8f0b4225bc03e1f5db011912619cd92b798d
Path within that repo: TALENT/model/lib/tabswift
Checkpoint: HF LAMDA-Tabular/TabSwift :: swift.ckpt (public, one file, both tasks)
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from models import load
from models.registry import checkpoint_path

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "tabswift"

#: sha256 of HF LAMDA-Tabular/TabSwift :: swift.ckpt, read from the Hub API.
UPSTREAM_CKPT_SHA256 = "16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363"
UPSTREAM_CKPT_BYTES = 32_947_079
UPSTREAM_CKPT_PARAMS = 8_203_211


# ---------------------------------------------------------------------------
# 0.2 checkpoint identity and strict loading
# ---------------------------------------------------------------------------


def test_checkpoint_is_byte_identical_to_upstream():
    """The committed swift.ckpt is the upstream file, not a re-export."""
    path = checkpoint_path("tabswift")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == UPSTREAM_CKPT_SHA256
    assert path.stat().st_size == UPSTREAM_CKPT_BYTES


def test_checkpoint_loads_strict_with_no_missing_or_unexpected_keys():
    """strict=True must succeed. We never paper over a mismatch with strict=False."""
    from models.vendor.tabswift.model.tabswift import TabSwift

    ck = torch.load(checkpoint_path("tabswift"), map_location="cpu", weights_only=True)
    assert set(ck) == {"config", "state_dict"}

    model = TabSwift(**ck["config"])
    incompatible = model.load_state_dict(ck["state_dict"], strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert sum(p.numel() for p in model.parameters()) == UPSTREAM_CKPT_PARAMS


def test_one_checkpoint_serves_both_tasks():
    """Upstream ships ONE file for both heads; the wrappers must both point at it.

    Confirmed by HF config.json: {"task": ["classification", "regression"]}.
    """
    from models.vendor.tabswift import classifier, regressor

    for mod in (classifier, regressor):
        src = inspect.getsource(mod._load_model if hasattr(mod, "_load_model") else mod)
        assert 'repo_id = "LAMDA-Tabular/TabSwift"' in src
        assert 'filename = "swift.ckpt"' in src


# ---------------------------------------------------------------------------
# 0.3 early-exit heads
# ---------------------------------------------------------------------------


def test_checkpoint_has_no_early_exit_heads():
    """The shipped checkpoint carries no per-layer prediction / exit heads.

    The paper (s3.2, App. B.2) describes h_pred^(e) / h_exit^(e) attached to all
    24 layers by a post-training stage, but those weights are NOT in the public
    release. If a future checkpoint adds them this test fails loudly and we can
    wire up a native per-layer readout (which would beat our patch-based lens).
    """
    ck = torch.load(checkpoint_path("tabswift"), map_location="cpu", weights_only=True)
    keys = list(ck["state_dict"])
    banned = ("exit", "early", "h_pred", "h_exit", "pred_head", "aux", "halt", "ponder")
    assert not [k for k in keys if any(b in k.lower() for b in banned)]
    # every key is accounted for by the documented module tree
    assert all(k.startswith(("x_linear.", "icl_predictor.")) for k in keys)
    assert len(keys) == 353


# ---------------------------------------------------------------------------
# 0.6 / architecture invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_no_column_or_row_stage(task):
    """TabSwift is row-attention-only BY DESIGN. Do not "restore" these stages.

    Paper arXiv 2606.07345 s3.1 / Fig. 2 / Fig. 3a / Eq. 5: the contribution is a
    row-attention backbone plus register tokens plus element-wise SDPA-output
    gating. The upstream class docstring mentioning a column embedder and a row
    interactor is inherited from TabICL's framework and does not describe this
    model. The entire feature encoder is `x_linear`.
    """
    m = load("tabswift", task=task, device="cpu")
    model = m._torch_model()

    assert not hasattr(model, "col_embedder")
    assert not hasattr(model, "row_interactor")

    children = dict(model.named_children())
    assert set(children) == {"x_linear", "icl_predictor"}
    assert isinstance(children["x_linear"], torch.nn.Linear)
    assert children["x_linear"].in_features == model.max_dim
    assert children["x_linear"].out_features == model.embed_dim * model.row_num_cls


def test_forward_does_not_reach_dead_paths():
    """forward() must never dispatch to _train_forward / _inference_forward.

    _inference_forward pads to a hardcoded 500 (not max_dim) and references an
    undefined `d`, so reaching it is either wrong or a NameError.
    """
    from models.vendor.tabswift.model import tabswift as ts

    tree = ast.parse(textwrap.dedent(inspect.getsource(ts.TabSwift.forward)))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_train_forward" not in called
    assert "_inference_forward" not in called

    # and prove it at runtime: tripwire raises if either is ever entered
    tripped = []
    originals = {n: getattr(ts.TabSwift, n) for n in ("_train_forward", "_inference_forward")}
    for name in originals:
        setattr(ts.TabSwift, name, lambda self, *a, _n=name, **k: tripped.append(_n))
    try:
        m = load("tabswift", task="regression", device="cpu")
        rng = np.random.RandomState(0)
        m._fit(rng.randn(12, 3), rng.randn(12))
        m._run_forward(rng.randn(3, 3))
    finally:
        for name, fn in originals.items():
            setattr(ts.TabSwift, name, fn)  # restore, do NOT delete
    assert tripped == []


def test_dead_inference_forward_is_still_broken_upstream():
    """Documents WHY the dead path must stay dead: it references an undefined name."""
    from models.vendor.tabswift.model import tabswift as ts

    src = inspect.getsource(ts.TabSwift._inference_forward)
    assert "pad_len = 500 - H" in src  # hardcoded, ignores self.max_dim

    # `d` is neither a parameter nor assigned before its first read, so the
    # first `if d is not None` raises. Prove it by actually calling the method.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    assert "d" not in {a.arg for a in fn.args.args}

    model = ts.TabSwift(max_dim=100, proj_dim=100, embed_dim=8, row_num_cls=1, icl_num_blocks=1)
    with pytest.raises(NameError, match=r"\bd\b"):
        model._inference_forward(torch.zeros(1, 4, 3), torch.zeros(1, 2))


# ---------------------------------------------------------------------------
# 0.5 feature shuffle must be the identity at n_estimators == 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_features", [2, 3, 5, 9])
def test_feature_shuffler_identity_at_one_estimator(n_features):
    """[VENDOR PATCH V4] Upstream drops the early return, so latin-square
    permutations were applied even for a single ensemble member."""
    from models.vendor.tabswift.preprocessing import FeatureShuffler

    for method in ("latin", "shift", "random", "none"):
        patterns = FeatureShuffler(n_features, method, random_state=42).shuffle(1)
        assert patterns == [list(range(n_features))], (method, patterns)


@pytest.mark.parametrize("model_id", ["tabpfn_v2", "tabicl_v2", "tabswift"])
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_resolved_feature_order_is_identity_for_all_models(model_id, task):
    """0.5: assert the *resolved* pattern, not just the shuffler, across models."""
    m = load(model_id, task=task, device="cpu")
    rng = np.random.RandomState(0)
    X = rng.randn(24, 4)
    y = rng.randn(24) if task == "regression" else (np.arange(24) % 2)
    m._fit(X, y)

    gen = getattr(m.estimator, "ensemble_generator_", None)
    if gen is None:  # TabPFN has no EnsembleGenerator; it uses ensemble configs
        assert m.estimator.n_estimators_ == 1
        return
    patterns = getattr(gen, "feature_shuffle_patterns_", None) or getattr(gen, "feature_shuffles_")
    flat = [p for group in patterns.values() for p in group]
    assert len(flat) == 1, f"expected exactly one ensemble member, got {len(flat)}"
    assert list(flat[0]) == sorted(flat[0]), f"non-identity feature order: {flat[0]}"
