"""Frozen contracts shared by every interp adapter.

Everything in this file is a *promise* that `models.interp.verify` checks at
runtime.  If an adapter cannot honour one of these, it must raise rather than
silently return something else-shaped.

The three models in the roster are structurally different, so the value of this
layer is entirely in the conventions being identical across them.

--------------------------------------------------------------------------
1. Token layout
--------------------------------------------------------------------------
Every model consumes one sequence built from the support (train) rows followed
by the query (test) rows.  Some models prepend non-data tokens:

    [ prefix tokens ][ support rows ][ query rows ]
      n_prefix         n_support       n_query

``n_prefix`` is read from the loaded model's own config -- never hardcoded.
Currently: TabPFN v2 = 0, TabICL v2 = 0, TabSwift = 64 (register tokens).

--------------------------------------------------------------------------
2. Residual stream views
--------------------------------------------------------------------------
``Trace.resid[l]`` is the CANONICAL per-row stream: ``(n_prefix + n_support +
n_query, d_model)``, taken at the OUTPUT of block ``l``.

TabPFN v2 is a per-*cell* transformer: its native state is
``(items, feature_groups, d_model)``.  There is no per-row vector in the model.
We define the canonical row vector as the LAST feature slot, because that is
the target slot and the only slot the decoder reads
(``decoder(state[:, :, -1])``).  The native tensor is always available
unreduced as ``Trace.resid_full[l]``; nothing is thrown away.

``StreamView`` records which convention an adapter used so downstream code can
assert on it instead of guessing.

--------------------------------------------------------------------------
3. Layer indexing
--------------------------------------------------------------------------
``l`` indexes the interior blocks of the model's in-context-learning stack,
``0 <= l < n_layers``, and ``resid[l]`` is that block's OUTPUT.  ``n_layers`` is
the real depth of that stack (TabPFN v2 = 12, TabICL v2 = 12, TabSwift = 24).
Adapters must not truncate.  If you want fewer layers, slice downstream.

Stages that run *before* the ICL stack (TabICL's column embedder and row
interaction, TabSwift's column/row stages) are not part of this index.  They
are reachable via ``Trace.pre_icl`` when an adapter captures them.

--------------------------------------------------------------------------
4. Units
--------------------------------------------------------------------------
``Trace.pred`` is in the ORIGINAL units of the ``y`` handed to ``run()``, and
must equal the estimator's own ``.predict()`` to within
``PRED_ROUNDTRIP_ATOL``.  Adapters do not standardise ``y`` internally; each
model's own preprocessing owns that.  If an experiment wants standardised
targets it standardises the task before calling.

``Trace.layer_pred[l]`` is the logit-lens read of block ``l`` through the
model's own final norm + decoder + output transform, also in original units.
By construction ``layer_pred[n_layers - 1] == pred``; `verify` asserts it and
reports the residual as the measured decode floor.

--------------------------------------------------------------------------
5. Attention
--------------------------------------------------------------------------
Attention is recorded per *forward call*, never collapsed.  A module that runs
twice in a block (TabPFN's item attention runs once for query->support and once
for support->support) yields two records, each tagged with its call index and
q/k lengths.  Nothing is silently overwritten and no head or feature-group axis
is silently dropped: ``AttentionRecord.probs`` keeps ``(batch, heads, q, k)``
and any pooling is the caller's explicit choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Tolerance for "the trace's prediction is the model's prediction".
# Tight on purpose: anything looser hides a unit or layout bug.
PRED_ROUNDTRIP_ATOL = 1e-4
PRED_ROUNDTRIP_RTOL = 1e-4

# Tolerance for "recomputed attention probabilities reproduce the real output",
# as a RELATIVE error against the magnitude of that output.
#
# It has to be relative and dtype-aware: TabPFN runs its attention under
# autocast in float16, so the fused kernel's own rounding is worth ~1e-2
# relative on some calls. An absolute bound either rejects a correct observer
# or accepts a wrong one, depending on the scale of the activations.
ATTN_SELFCHECK_RTOL_HIGH_PRECISION = 1e-4  # float32 / float64
ATTN_SELFCHECK_RTOL_LOW_PRECISION = 5e-2  # float16 / bfloat16


def attn_selfcheck_rtol(dtype_name: str) -> float:
    """Relative tolerance appropriate to the dtype the model computed in."""
    return (
        ATTN_SELFCHECK_RTOL_LOW_PRECISION
        if any(tag in dtype_name for tag in ("float16", "bfloat16", "half"))
        else ATTN_SELFCHECK_RTOL_HIGH_PRECISION
    )


class StreamView(str, Enum):
    """How an adapter reduced its native state to the canonical row stream."""

    #: Model is natively per-row; canonical stream is the state itself.
    NATIVE_ROW = "native_row"
    #: Model is per-cell; canonical stream is the final (target) feature slot.
    TARGET_SLOT = "target_slot"


class Task(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass(frozen=True)
class TokenLayout:
    """Index ranges of the model's ICL sequence.

    ``n_prefix`` comes from the model config; ``n_support``/``n_query`` from the
    data handed to ``run()``.
    """

    n_prefix: int
    n_support: int
    n_query: int

    def __post_init__(self) -> None:
        for name in ("n_prefix", "n_support", "n_query"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @property
    def total(self) -> int:
        return self.n_prefix + self.n_support + self.n_query

    @property
    def prefix_slice(self) -> slice:
        return slice(0, self.n_prefix)

    @property
    def support_slice(self) -> slice:
        return slice(self.n_prefix, self.n_prefix + self.n_support)

    @property
    def query_slice(self) -> slice:
        return slice(self.n_prefix + self.n_support, self.total)

    def check(self, seq_len: int, where: str) -> None:
        """Fail loudly when a captured tensor does not match the declared layout."""
        if seq_len != self.total:
            raise AssertionError(
                f"{where}: captured sequence length {seq_len} != declared layout "
                f"{self.n_prefix}+{self.n_support}+{self.n_query}={self.total}. "
                "The hook is on the wrong module, the model chunked the forward "
                "pass, or the layout is stale."
            )
