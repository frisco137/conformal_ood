"""What a hooked forward pass returns."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import StreamView, Task, TokenLayout


@dataclass
class AttentionRecord:
    """Attention probabilities for one forward call of one attention module.

    ``probs`` is ``(batch, heads, q_len, k_len)`` in float64 and always sums to
    1 along the last axis.  Nothing is pooled: for TabPFN the batch axis carries
    the feature-group index, and pooling over it (or over heads) is the caller's
    explicit decision, not ours.

    ``kind`` disambiguates the calls a block makes:
      - ``"support_to_support"`` -- q and k are both the support block
      - ``"query_to_support"``   -- q is the query block, k the support block
      - ``"full"``               -- q and k span the whole sequence
    """

    layer: int
    module: str
    call_index: int
    kind: str
    probs: np.ndarray
    #: max |probs @ v - real_output|; the observer's own proof it reconstructed
    #: the attention the model actually computed.
    selfcheck_error: float
    #: the same, relative to the magnitude of the real output. This is the
    #: quantity that is actually checked, because models run attention in fp16.
    selfcheck_rel_error: float = 0.0
    #: dtype the model computed this attention in, so callers know the
    #: provenance of ``probs`` (recomputed in fp32 from the model's own q/k).
    compute_dtype: str = "unknown"

    @property
    def q_len(self) -> int:
        return self.probs.shape[-2]

    @property
    def k_len(self) -> int:
        return self.probs.shape[-1]

    def mean_over_heads(self) -> np.ndarray:
        """(batch, q, k) -- averaging heads only."""
        return self.probs.mean(axis=1)

    def mean_over_heads_and_batch(self) -> np.ndarray:
        """(q, k) -- the fully pooled view. For TabPFN this also pools feature
        groups, which is what the spectral paper's H-vector uses."""
        return self.probs.mean(axis=(0, 1))


@dataclass
class Trace:
    """Result of one instrumented forward pass.

    See ``models.interp.conventions`` for the guarantees on every field.
    """

    model_id: str
    task: Task
    layout: TokenLayout
    stream_view: StreamView
    n_layers: int
    d_model: int

    #: block index -> (total_tokens, d_model), canonical per-row stream
    resid: dict[int, np.ndarray] = field(default_factory=dict)
    #: block index -> model-native state (per-cell for TabPFN); never reduced
    resid_full: dict[int, np.ndarray] = field(default_factory=dict)
    #: block index -> attention records, one per forward call
    attn: dict[int, list[AttentionRecord]] = field(default_factory=dict)
    #: stage name -> array, for stages before the ICL stack
    pre_icl: dict[str, np.ndarray] = field(default_factory=dict)

    #: (n_query,) regression mean, or (n_query, n_classes) class probabilities.
    #: Original units. Equals the estimator's own predict()/predict_proba().
    pred: np.ndarray | None = None
    #: block index -> logit-lens read of that block, same units as ``pred``
    layer_pred: dict[int, np.ndarray] = field(default_factory=dict)
    #: (n_query, n_quantiles) predictive quantiles, original units (regression)
    quantiles: np.ndarray | None = None
    #: the quantile levels ``quantiles`` was evaluated at
    quantile_levels: np.ndarray | None = None
    #: (n_query,) predictive variance, original units squared (regression)
    pred_variance: np.ndarray | None = None

    #: diagnostics recorded by the run, surfaced so callers can assert on them
    call_counts: dict[str, int] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    # -- convenience views ---------------------------------------------------

    def support(self, layer: int) -> np.ndarray:
        return self.resid[layer][self.layout.support_slice]

    def query(self, layer: int) -> np.ndarray:
        return self.resid[layer][self.layout.query_slice]

    def attn_of_kind(self, layer: int, kind: str) -> AttentionRecord:
        matches = [r for r in self.attn.get(layer, []) if r.kind == kind]
        if len(matches) != 1:
            raise KeyError(
                f"layer {layer}: expected exactly one attention record of kind {kind!r}, "
                f"found {len(matches)} (available: {[r.kind for r in self.attn.get(layer, [])]})"
            )
        return matches[0]

    def has_attention(self) -> bool:
        return any(self.attn.values())

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Trace({self.model_id}, {self.task.value}, layers={self.n_layers}, "
            f"d_model={self.d_model}, layout={self.layout}, "
            f"resid={len(self.resid)}, attn_layers={sum(1 for v in self.attn.values() if v)})"
        )
