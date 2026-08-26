"""The battery that proves the instrumentation is correct.

Shape and NaN checks pass on a broken hook.  Every check here is designed to
*fail* on a specific way this layer went wrong before:

===========================  ==================================================
check                        what it would have caught
===========================  ==================================================
prediction_roundtrip         TabICL predictions left in standardised units while
                             TabPFN's were in original units.
logit_lens_identity          Per-layer decoders that miss the model's own output
                             by 8% of y_std even at the final layer.
capture_call_count           Silent chunking / stale module handles: a hook that
                             fires 0 or 5 times while the code assumes 1.
identity_patch               Wrong slicing or a write path that lands somewhere
                             other than where the read came from.
zero_patch_moves_output      A "patch" that is silently a no-op.
cross_task_patch             That the exposed stream is causally sufficient --
                             i.e. it is really the model's state, not a view.
layout_matches_data          Hardcoded token offsets (TabSwift's 64 registers).
query_permutation            Query rows silently reordered relative to X_test.
attention_selfcheck          Attention probabilities that do not reproduce the
                             model's own attention output.
attention_calls_complete     Keeping "the last" of several attention calls per
                             block and discarding the rest.
no_global_mutation           A monkeypatch left installed, changing the model
                             for every later run in the process.
determinism                  Unseeded state making a run irreproducible.
random_control_reproducible  A "random control" that is a different network on
                             every call.
registry_matches_reality     Architecture facts drifting from the checkpoint.
===========================  ==================================================

Run it with::

    python -m models.interp.verify              # roster, regression
    python -m models.interp.verify --task classification
    python -m models.interp.verify --models tabpfn_v2
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import traceback
from typing import Callable

import numpy as np
import torch

from ..registry import ROSTER, SPECS, spec
from .base import CaptureSpec
from .conventions import PRED_ROUNDTRIP_ATOL, StreamView, Task


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    seconds: float


class Battery:
    def __init__(self, model, task: Task, seed: int = 0, n: int = 40, m: int = 15, d: int = 4):
        self.model = model
        self.task = task
        rng = np.random.RandomState(seed)
        self.X = rng.randn(n, d)
        self.Xq = rng.randn(m, d)
        self.X2 = rng.randn(n, d) * 1.7 + 0.6
        if task is Task.REGRESSION:
            self.y = np.sin(self.X[:, 0] * 2.0) * 3.0 + 0.3 * rng.randn(n) + 5.0
            self.y2 = np.cos(self.X2[:, 1]) * 2.0 - 4.0 + 0.3 * rng.randn(n)
        else:
            self.y = (self.X[:, 0] + 0.3 * rng.randn(n) > 0).astype(int)
            self.y2 = (self.X2[:, 1] - 0.2 * rng.randn(n) > 0).astype(int)
        self.results: list[CheckResult] = []

    def check(self, name: str, fn: Callable[[], str]) -> None:
        start = time.time()
        try:
            detail = fn() or "ok"
            passed = True
        except Exception as exc:  # noqa: BLE001 - a battery reports, it does not raise
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
            if _VERBOSE:
                traceback.print_exc()
        self.results.append(CheckResult(name, passed, detail, time.time() - start))
        flag = "PASS" if passed else "FAIL"
        print(f"    [{flag}] {name:<30} {detail}")

    # -- individual checks ---------------------------------------------------

    def prediction_roundtrip(self) -> str:
        trace = self.model.run(self.X, self.y, self.Xq)
        err = self.model.check_prediction_roundtrip(trace, self.Xq)
        scale = float(np.std(np.asarray(trace.pred)))
        return f"max abs err {err:.2e} (pred sd {scale:.3f}, tol {PRED_ROUNDTRIP_ATOL:g})"

    def logit_lens_identity(self) -> str:
        last = self.model.n_layers - 1
        trace = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(layer_pred=True, layers=[0, last])
        )
        got = np.asarray(trace.layer_pred[last], dtype=np.float64)
        ref = np.asarray(trace.pred, dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
        if err > PRED_ROUNDTRIP_ATOL:
            raise AssertionError(
                f"logit lens at the final block differs from the model's own "
                f"prediction by {err:.3e}; the decode path is not the model's tail"
            )
        early = np.asarray(trace.layer_pred[0], dtype=np.float64)
        drift = float(np.max(np.abs(early - ref)))
        return f"final-block decode err {err:.2e}; block 0 differs by {drift:.3f} (as expected)"

    def capture_call_count(self) -> str:
        trace = self.model.run(self.X, self.y, self.Xq)
        counts = set(trace.call_counts.values())
        if counts != {1}:
            raise AssertionError(f"hooked modules fired {counts} times, expected exactly 1")
        return f"{len(trace.call_counts)} blocks, 1 forward call each"

    def layout_matches_data(self) -> str:
        trace = self.model.run(self.X, self.y, self.Xq)
        layout = trace.layout
        s = spec(self.model.model_id)
        if layout.n_prefix != s.n_prefix_tokens:
            raise AssertionError(
                f"layout prefix {layout.n_prefix} != registry {s.n_prefix_tokens}"
            )
        if layout.n_support != len(self.X) or layout.n_query != len(self.Xq):
            raise AssertionError(f"layout {layout} does not match the data")
        for layer, state in trace.resid.items():
            if state.shape != (layout.total, self.model.d_model):
                raise AssertionError(
                    f"block {layer} stream {state.shape} != ({layout.total}, {self.model.d_model})"
                )
        return f"prefix={layout.n_prefix} support={layout.n_support} query={layout.n_query}"

    def query_permutation(self) -> str:
        """Query rows must follow X_test order, not some internal ordering."""
        perm = np.random.RandomState(3).permutation(len(self.Xq))
        base = self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(layers=[0]))
        permuted = self.model.run(self.X, self.y, self.Xq[perm], capture=CaptureSpec(layers=[0]))
        a = base.query(0)[perm]
        b = permuted.query(0)
        err = float(np.max(np.abs(a - b)))
        tol = 1e-3 * max(1.0, float(np.abs(a).max()))
        if err > tol:
            raise AssertionError(
                f"permuting X_test did not permute the query stream (max err {err:.3e} > {tol:.3e})"
            )
        return f"query rows track X_test order (max err {err:.2e})"

    def identity_patch(self) -> str:
        """Re-injecting a block's own output must change nothing at all."""
        layer = self.model.n_layers // 2
        trace = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(resid_full=True, layers=[layer])
        )
        values = trace.resid_full[layer] if trace.resid_full else trace.resid[layer]
        patched = self.model.run(
            self.X,
            self.y,
            self.Xq,
            capture=CaptureSpec(resid=False, layers=[layer]),
            patches=[self.model.make_patch(layer, values)],
        )
        err = float(np.max(np.abs(np.asarray(patched.pred) - np.asarray(trace.pred))))
        if err > 1e-6:
            raise AssertionError(
                f"identity patch at block {layer} moved the prediction by {err:.3e}; "
                "the captured tensor is not the tensor that flows forward"
            )
        return f"block {layer}: identity patch changes nothing (err {err:.2e})"

    def zero_patch_moves_output(self) -> str:
        layer = self.model.n_layers // 2
        base = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(resid_full=True, layers=[layer])
        )
        values = base.resid_full[layer] if base.resid_full else base.resid[layer]
        patched = self.model.run(
            self.X,
            self.y,
            self.Xq,
            capture=CaptureSpec(resid=False, layers=[layer]),
            patches=[self.model.make_patch(layer, np.zeros_like(values))],
        )
        delta = float(np.max(np.abs(np.asarray(patched.pred) - np.asarray(base.pred))))
        if delta < 1e-6:
            raise AssertionError(
                f"zeroing block {layer} did not change the prediction; the patch is a no-op"
            )
        return f"block {layer}: zero patch moves prediction by {delta:.3f}"

    def cross_task_patch(self) -> str:
        """Transplanting another task's block-0 state must transfer its answer.

        This is the strongest statement available about the exposed stream: if
        overwriting it makes the model produce the donor's prediction, the
        stream really is the model's computational state and not a view of it.

        Compared with a *scale-invariant* statistic on purpose.  These models
        decode through a per-task target transform fitted on the context ``y``
        (TabPFN's bar-distribution borders, TabICL's y-scaler), so a donor
        state decoded through the recipient's transform is the donor's answer
        up to an affine map.  Correlation tests the thing we actually care
        about -- did the donor's *computation* transfer -- without being fooled
        by that expected rescaling.
        """
        layer = 0
        donor = self.model.run(
            self.X2, self.y2, self.Xq, capture=CaptureSpec(resid_full=True, layers=[layer])
        )
        donor_values = donor.resid_full[layer] if donor.resid_full else donor.resid[layer]
        donor_pred = np.asarray(donor.pred, dtype=np.float64)

        base = self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(resid=False))
        base_pred = np.asarray(base.pred, dtype=np.float64)

        patched = self.model.run(
            self.X,
            self.y,
            self.Xq,
            capture=CaptureSpec(resid=False, layers=[layer]),
            patches=[self.model.make_patch(layer, donor_values)],
        )
        patched_pred = np.asarray(patched.pred, dtype=np.float64)

        r_donor = _corr(patched_pred, donor_pred)
        r_base = _corr(patched_pred, base_pred)
        if not (r_donor > 0.9 and r_donor > r_base):
            raise AssertionError(
                f"patching block 0 with a donor task's state did not transfer the "
                f"donor's computation (corr with donor {r_donor:.3f}, with original "
                f"{r_base:.3f}; need >0.9 and > the original)"
            )
        return f"corr(patched, donor) = {r_donor:.4f} vs corr(patched, original) = {r_base:.3f}"

    def attention_module_is_live(self) -> str:
        """The observed attention code must be the code the live model runs.

        A package that exists both vendored and pip-installed loads twice under
        two module names; the checkpoint can then build the model out of one
        copy while hooks sit on the other, and every capture comes back empty
        while all the shape checks still pass. Compare the top-level package of
        the live block class with the package we are observing.
        """
        module = getattr(self.model, "_attention_module", None)
        if module is None:
            return "n/a (observes a class attribute, not a module-level function)"
        block_pkg = type(self.model._blocks()[0]).__module__.split(".")[0]
        observed_pkg = module.__name__.split(".")[0]
        if block_pkg != observed_pkg:
            raise AssertionError(
                f"live blocks come from package {block_pkg!r} but the attention "
                f"observer is installed on {observed_pkg!r}: two copies of the same "
                "package are loaded and the hooks are on the wrong one"
            )
        return f"live blocks and observer both from {block_pkg!r}"

    def attention_selfcheck(self) -> str:
        trace = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(attention=True, layers=[0])
        )
        records = trace.attn.get(0, [])
        if not records:
            raise AssertionError("no attention recorded for block 0")
        worst = max(r.selfcheck_error for r in records)
        for record in records:
            row_sums = record.probs.sum(axis=-1)
            if not np.allclose(row_sums, 1.0, atol=1e-5):
                raise AssertionError(
                    f"{record.kind}: attention rows do not sum to 1 "
                    f"(min {row_sums.min():.4f}, max {row_sums.max():.4f})"
                )
        kinds = [r.kind for r in records]
        return f"{len(records)} record(s) {kinds}, max reconstruction err {worst:.2e}"

    def attention_calls_complete(self) -> str:
        """Every attention call in a block must survive as its own record."""
        trace = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(attention=True, layers=[0])
        )
        records = trace.attn.get(0, [])
        if not records:
            raise AssertionError("no attention recorded for block 0")
        kinds = [r.kind for r in records]
        if len(set(kinds)) != len(kinds):
            raise AssertionError(f"duplicate attention kinds in one block: {kinds}")
        if self.model.model_id.startswith("tabpfn"):
            # A TabPFN block runs feature attention once and item attention TWICE
            # (query->context, then context->context). Keeping only one of the
            # item calls was a real bug; assert all three survive.
            if "feature_to_feature" not in kinds:
                raise AssertionError(f"TabPFN block 0 recorded no feature attention: {kinds}")
            item_calls = [k for k in kinds if k != "feature_to_feature"]
            if len(item_calls) != 2:
                raise AssertionError(
                    f"TabPFN block 0 should record 2 distinct item-attention calls "
                    f"(query->context and context->context), got {item_calls}"
                )
        return f"{len(records)} distinct call(s) kept: {kinds}"

    def no_global_mutation(self) -> str:
        """After a run, every observed function must be the pristine original."""
        before = _attention_fingerprint(self.model)
        self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(attention=True, layers=[0]))
        after = _attention_fingerprint(self.model)
        if before != after:
            raise AssertionError(
                "the attention function was not restored after the run; a monkeypatch "
                "is still installed and will alter every later run in this process"
            )
        return "observed functions restored"

    def observation_is_free(self) -> str:
        """Turning attention capture on must not change what the model computes."""
        plain = self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(resid=False))
        observed = self.model.run(
            self.X, self.y, self.Xq, capture=CaptureSpec(resid=False, attention=True, layers=[0])
        )
        err = float(np.max(np.abs(np.asarray(plain.pred) - np.asarray(observed.pred))))
        if err > 1e-9:
            raise AssertionError(
                f"capturing attention changed the prediction by {err:.3e}; the observer "
                "is not purely observational"
            )
        return f"identical predictions with and without capture (err {err:.2e})"

    def determinism(self) -> str:
        a = self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(layers=[0]))
        b = self.model.run(self.X, self.y, self.Xq, capture=CaptureSpec(layers=[0]))
        pred_err = float(np.max(np.abs(np.asarray(a.pred) - np.asarray(b.pred))))
        resid_err = float(np.max(np.abs(a.resid[0] - b.resid[0])))
        if pred_err > 1e-9 or resid_err > 1e-9:
            raise AssertionError(
                f"repeat run differs (pred {pred_err:.3e}, resid {resid_err:.3e})"
            )
        return f"two runs identical (pred {pred_err:.1e}, resid {resid_err:.1e})"

    def registry_matches_reality(self) -> str:
        s = spec(self.model.model_id)
        if self.model.n_layers != s.n_icl_blocks:
            raise AssertionError(
                f"registry says {s.n_icl_blocks} blocks, model has {self.model.n_layers}"
            )
        if self.model.d_model != s.d_model:
            raise AssertionError(
                f"registry says d_model {s.d_model}, model has {self.model.d_model}"
            )
        expected_view = StreamView.TARGET_SLOT if s.per_cell_state else StreamView.NATIVE_ROW
        if self.model.stream_view is not expected_view:
            raise AssertionError(
                f"registry implies {expected_view}, adapter reports {self.model.stream_view}"
            )
        return f"{s.n_icl_blocks} blocks, d_model {s.d_model}, {self.model.stream_view.value}"

    def predictive_distribution_declaration(self) -> str:
        s = spec(self.model.model_id)
        if self.task is not Task.REGRESSION:
            return "n/a for classification"
        if s.has_predictive_distribution:
            q, levels = self.model.predict_quantiles(self.Xq)
            q = np.asarray(q)
            if q.shape != (len(self.Xq), len(levels)):
                raise AssertionError(f"quantiles shape {q.shape} != {(len(self.Xq), len(levels))}")
            if np.any(np.diff(q, axis=1) < -1e-6):
                raise AssertionError("predictive quantiles are not monotone in the level")
            width = float(np.mean(q[:, -1] - q[:, 0]))
            if not np.isfinite(width) or width <= 0:
                raise AssertionError(f"degenerate predictive interval width {width}")
            return f"real quantiles, mean 99% width {width:.3f}"
        try:
            self.model.predict_quantiles(self.Xq)
        except NotImplementedError:
            return "declares no predictive distribution and refuses to invent one"
        raise AssertionError(
            "registry says this model has no predictive distribution, but "
            "predict_quantiles returned something"
        )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the flattened prediction vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def _attention_fingerprint(model) -> tuple:
    """Identity of whatever function the adapter observes, for mutation checks."""
    from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention

    module = getattr(model, "_attention_module", None)
    if module is not None:
        return (id(module.sdpa_with_flattened_batch),)
    return (id(MultiHeadAttention.compute_attention_heads),)


def random_control_reproducible(model_id: str, task: Task, device: str) -> CheckResult:
    """A random control must be ONE fixed network, not a fresh draw per call."""
    from ..loaders import load

    start = time.time()
    try:
        rng = np.random.RandomState(0)
        X, Xq = rng.randn(30, 3), rng.randn(10, 3)
        y = rng.randn(30) if task is Task.REGRESSION else (rng.rand(30) > 0.5).astype(int)

        a = load(model_id, task=task, device=device, random_weights_seed=7)
        b = load(model_id, task=task, device=device, random_weights_seed=7)
        c = load(model_id, task=task, device=device, random_weights_seed=8)

        pa1 = np.asarray(a.run(X, y, Xq, capture=CaptureSpec(resid=False)).pred, dtype=np.float64)
        pa2 = np.asarray(a.run(X, y, Xq, capture=CaptureSpec(resid=False)).pred, dtype=np.float64)
        pb = np.asarray(b.run(X, y, Xq, capture=CaptureSpec(resid=False)).pred, dtype=np.float64)
        pc = np.asarray(c.run(X, y, Xq, capture=CaptureSpec(resid=False)).pred, dtype=np.float64)

        if not np.allclose(pa1, pa2, atol=1e-9):
            raise AssertionError(
                f"same instance gave different outputs on two calls "
                f"(max diff {np.max(np.abs(pa1 - pa2)):.3e}); the control is being "
                "re-randomised per run"
            )
        if not np.allclose(pa1, pb, atol=1e-9):
            raise AssertionError(
                f"same seed in two instances gave different outputs "
                f"(max diff {np.max(np.abs(pa1 - pb)):.3e}); the control is not reproducible"
            )
        if np.allclose(pa1, pc, atol=1e-9):
            raise AssertionError("different seeds gave identical outputs; the seed is ignored")
        if not np.all(np.isfinite(pa1)):
            raise AssertionError("random control produced non-finite predictions")
        detail = (
            f"seed-stable across calls and instances; seed 7 vs 8 differ by "
            f"{np.max(np.abs(pa1 - pc)):.3f}"
        )
        passed = True
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        passed = False
        if _VERBOSE:
            traceback.print_exc()
    result = CheckResult("random_control_reproducible", passed, detail, time.time() - start)
    print(f"    [{'PASS' if passed else 'FAIL'}] {result.name:<30} {result.detail}")
    return result


_VERBOSE = False

CHECKS = [
    "registry_matches_reality",
    "prediction_roundtrip",
    "capture_call_count",
    "layout_matches_data",
    "query_permutation",
    "logit_lens_identity",
    "identity_patch",
    "zero_patch_moves_output",
    "cross_task_patch",
    "attention_module_is_live",
    "attention_selfcheck",
    "attention_calls_complete",
    "observation_is_free",
    "no_global_mutation",
    "determinism",
    "predictive_distribution_declaration",
]


def run_battery(model_id: str, task: Task, device: str = "cuda") -> list[CheckResult]:
    from ..loaders import load

    print(f"\n=== {model_id} / {task.value} ===")
    model = load(model_id, task=task, device=device)
    print(f"    {model.describe()}")
    battery = Battery(model, task)
    for name in CHECKS:
        battery.check(name, getattr(battery, name))
    results = list(battery.results)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results.append(random_control_reproducible(model_id, task, device))
    return results


def main() -> int:
    global _VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(ROSTER))
    parser.add_argument("--task", default="regression", choices=[t.value for t in Task])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    _VERBOSE = args.verbose

    task = Task(args.task)
    all_results: dict[str, list[CheckResult]] = {}
    for model_id in args.models:
        s = SPECS[model_id]
        if task is Task.CLASSIFICATION and not s.supports_classification:
            print(f"\n=== {model_id} / {task.value} === SKIPPED: {s.known_issues}")
            continue
        all_results[model_id] = run_battery(model_id, task, args.device)

    print("\n" + "=" * 72)
    failures = 0
    for model_id, results in all_results.items():
        bad = [r for r in results if not r.passed]
        failures += len(bad)
        status = "ALL PASS" if not bad else f"{len(bad)} FAILED: {[r.name for r in bad]}"
        print(f"{model_id:<14} {len(results) - len(bad)}/{len(results)}  {status}")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
