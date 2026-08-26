"""Model layer: definitions, checkpoints, and instrumented access.

Two entry points, nothing else:

    from models import load, SPECS

    m = load("tabpfn_v2", task="regression", device="cuda")
    trace = m.run(X_train, y_train, X_test)

``SPECS`` is the single source of truth for architecture facts and
capabilities. See models/README.md.
"""

from .registry import ROSTER, SCALE_BAR, SPECS, UNAVAILABLE, ModelSpec, spec, summary_table
from .loaders import load, available

__all__ = [
    "load",
    "available",
    "SPECS",
    "ROSTER",
    "SCALE_BAR",
    "UNAVAILABLE",
    "ModelSpec",
    "spec",
    "summary_table",
]
