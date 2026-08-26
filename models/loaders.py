"""The one function that builds an instrumented model."""

from __future__ import annotations

from .interp.base import RandomWeightSpec
from .interp.conventions import Task
from .registry import SPECS, spec


def available(task: str | Task | None = None) -> list[str]:
    """Model ids that can serve ``task`` (or all of them when ``task`` is None)."""
    if task is None:
        return list(SPECS)
    task = Task(task)
    key = "supports_regression" if task is Task.REGRESSION else "supports_classification"
    return [mid for mid, s in SPECS.items() if getattr(s, key)]


def load(
    model_id: str,
    task: str | Task = Task.REGRESSION,
    device: str = "cuda",
    *,
    random_weights_seed: int | None = None,
    **kwargs,
):
    """Build an instrumented model.

    ``random_weights_seed`` produces the negative control: a randomly
    initialised clone of the same architecture. It is a *seed*, so the control
    is one fixed network, reproducible across calls and across runs.
    """
    s = spec(model_id)
    task = Task(task)
    if task is Task.CLASSIFICATION and not s.supports_classification:
        raise ValueError(f"{model_id} has no usable classifier: {s.known_issues}")
    if task is Task.REGRESSION and not s.supports_regression:
        raise ValueError(f"{model_id} has no usable regressor: {s.known_issues}")

    random_weights = (
        RandomWeightSpec(seed=random_weights_seed) if random_weights_seed is not None else None
    )

    if model_id in ("tabpfn_v2", "tabpfn_v2_5"):
        from .interp.adapters.tabpfn_v2 import TabPFNv2Interp, TabPFNv2_5Interp

        cls = TabPFNv2Interp if model_id == "tabpfn_v2" else TabPFNv2_5Interp
    elif model_id == "tabicl_v2":
        from .interp.adapters.tabicl_v2 import TabICLv2Interp as cls
    elif model_id == "tabswift":
        from .interp.adapters.tabswift import TabSwiftInterp as cls
    else:  # pragma: no cover - guarded by spec() above
        raise KeyError(model_id)

    return cls(task=task, device=device, random_weights=random_weights, **kwargs)
