"""TabICL v2 adapter.

Architecture, as measured from the loaded checkpoint
(``jingang/TabICL`` / ``tabicl-regressor-v2-20260212.ckpt``):

* ``icl_predictor.tf_icl.blocks`` = **12** blocks, ``d_model = 512``.
  The project plan's "8-block ICL stack, width 256" is TabICL **v1**; it does
  not describe the checkpoint this project loads.
* Block outputs are ``(batch, tokens, d_model)`` -- natively per row, so the
  canonical stream is the state itself (``StreamView.NATIVE_ROW``).
* ``n_prefix = 0``: the sequence is ``[support][query]`` and the model slices
  ``out[:, train_size:]``.
* The column embedder and row-interaction stages run *before* the ICL stack and
  are deliberately outside the layer index (see ``conventions``); SDPA calls
  they make are ignored by the attention observer.
* Installed as the pip package ``tabicl==2.1.1`` and hooked, never vendored:
  the checkpoint unpickles classes under the absolute module path
  ``tabicl._model.*``, so a vendored copy is shadowed by the installed one and
  hooks silently land on a module the live model never uses.
* Regression head is a 999-quantile decoder, so a genuine predictive
  distribution is available via the estimator's own
  ``predict(output_type="quantiles")``.
"""

from __future__ import annotations

import numpy as np

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl._model import attention as _tabicl_attention
from ..base import RandomWeightSpec
from ..conventions import Task
from ._icl_stack import IclStackInterp


class TabICLv2Interp(IclStackInterp):
    model_id = "tabicl_v2"
    _attention_module = _tabicl_attention
    _supports_ssmax = True

    def __init__(
        self,
        task: Task | str = Task.REGRESSION,
        device: str = "cuda",
        random_weights: RandomWeightSpec | None = None,
    ) -> None:
        super().__init__(task=Task(task), device=device, random_weights=random_weights)
        self._warmup()

    def _build_estimator(self):
        cls = TabICLRegressor if self.task is Task.REGRESSION else TabICLClassifier
        return cls(device=self.device, n_estimators=1, n_jobs=1, random_state=0)

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
        # TabICL prepends nothing inside the ICL stack; the 4 CLS tokens live in
        # the *row interaction* stage, which is upstream of this index.
        return 0

    # -- predictive distribution --------------------------------------------

    def predict_quantiles(self, X_test: np.ndarray, levels: np.ndarray | None = None):
        """Predictive quantiles in original y units, from the model's own head."""
        if self.task is not Task.REGRESSION:
            raise TypeError("predict_quantiles is regression-only")
        levels = np.asarray(
            levels if levels is not None else np.linspace(0.005, 0.995, 199), dtype=float
        )
        q = self.estimator.predict(
            np.asarray(X_test),
            output_type="quantiles",
            alphas=[float(a) for a in levels],
        )
        return np.asarray(q, dtype=np.float64), levels
