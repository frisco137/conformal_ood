"""TabSwift adapter.

Architecture, as measured from ``models/checkpoints/tabswift/swift.ckpt``
(``config`` block inside the checkpoint):

* ``icl_num_blocks = 24``, ``embed_dim = 192``, ``register_tokens = 64``,
  ``max_classes = 10``.  The stack is **24** blocks and the adapter exposes all
  24; nothing is truncated to 12.
* Sequence layout is ``[64 register tokens][support][query]``.  ``n_prefix`` is
  read from ``icl_predictor.register_tokens``, never hardcoded -- the model
  itself slices ``out[:, register_tokens + train_size:]``.
* Block outputs are ``(batch, tokens, d_model)``; natively per row.
* Column and row stages run before the ICL stack and are outside the layer index.

**No predictive distribution for regression.**  ``icl_predictor.reg_decoder``
ends in ``Linear(384, 1)`` -- a point estimate.  TabSwift therefore cannot
answer variance/uncertainty questions, and this adapter raises rather than
inventing a number.  (The previous implementation returned
``np.ones(m) * y_std**2 * 0.1``, a constant, which silently fed fabricated data
into the uncertainty experiments.)  Classification does yield real class
probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...registry import checkpoint_path
from ...vendor.tabswift.classifier import TabSwiftClassifier
from ...vendor.tabswift.model import attention as _tabswift_attention
from ...vendor.tabswift.regressor import TabSwiftRegressor
from ..base import RandomWeightSpec
from ..conventions import Task
from ._icl_stack import IclStackInterp


class TabSwiftInterp(IclStackInterp):
    model_id = "tabswift"
    _attention_module = _tabswift_attention
    _supports_ssmax = False

    def __init__(
        self,
        task: Task | str = Task.REGRESSION,
        device: str = "cuda",
        random_weights: RandomWeightSpec | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else checkpoint_path("tabswift")
        super().__init__(task=Task(task), device=device, random_weights=random_weights)
        self._warmup()

    def _build_estimator(self):
        cls = TabSwiftRegressor if self.task is Task.REGRESSION else TabSwiftClassifier
        return cls(model_path=str(self._checkpoint), n_estimators=1, device=self.device)

    def _warmup(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.randn(16, 3)
        y = rng.randn(16) if self.task is Task.REGRESSION else np.arange(16) % 2
        self._fit(X, y)
        predictor = self._torch_model().icl_predictor
        self._n_layers = len(predictor.tf_icl.blocks)
        self._d_model = int(predictor.ln.normalized_shape[0])

    def _torch_model(self):
        return self.estimator.model_

    def _blocks(self):
        return self._torch_model().icl_predictor.tf_icl.blocks

    def _n_prefix(self) -> int:
        return int(self._torch_model().icl_predictor.register_tokens)

    # -- output shape normalisation -----------------------------------------

    def _reference_prediction(self, X_test: np.ndarray) -> np.ndarray:
        """TabSwift's regressor returns ``(n, 1)``; flatten to the common shape."""
        if self.task is Task.CLASSIFICATION:
            return np.asarray(
                self.estimator.predict_proba(np.asarray(X_test, dtype=np.float64)),
                dtype=np.float64,
            )
        raw = np.asarray(
            self.estimator.predict(np.asarray(X_test, dtype=np.float64)), dtype=np.float64
        )
        return raw.reshape(-1)

    # -- capability declaration ---------------------------------------------

    def predict_quantiles(self, X_test: np.ndarray, levels=None):
        raise NotImplementedError(
            "TabSwift has no predictive distribution for regression: its regression "
            "head is Linear(d_model*2, 1), a point estimate. Any variance or "
            "quantile-width number for TabSwift would be fabricated. Use "
            "models.registry.SPECS['tabswift'].has_predictive_distribution to branch "
            "on this instead of asking."
        )
