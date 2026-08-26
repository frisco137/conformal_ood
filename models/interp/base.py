"""The uniform interp API. One class, three very different models behind it."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .conventions import (
    PRED_ROUNDTRIP_ATOL,
    PRED_ROUNDTRIP_RTOL,
    StreamView,
    Task,
    TokenLayout,
)
from .hooks import PatchSpec
from .trace import Trace


@dataclass
class CaptureSpec:
    """What to record during a run. Everything is off by default because
    attention capture and logit-lens decoding both cost real time."""

    #: capture the per-block residual stream
    resid: bool = True
    #: also keep the model-native (unreduced) state
    resid_full: bool = False
    #: recover attention probabilities (delegating observer; no numerics change)
    attention: bool = False
    #: decode every block through the model's own head
    layer_pred: bool = False
    #: capture stages that run before the ICL stack
    pre_icl: bool = False
    #: restrict to these block indices; None means all
    layers: Sequence[int] | None = None

    def wanted_layers(self, n_layers: int) -> list[int]:
        if self.layers is None:
            return list(range(n_layers))
        bad = [l for l in self.layers if not 0 <= l < n_layers]
        if bad:
            raise IndexError(f"layers {bad} out of range for a {n_layers}-block stack")
        return sorted(set(self.layers))


@dataclass
class RandomWeightSpec:
    """A reproducible randomly-initialised clone.

    The previous implementation re-randomised on every call with the global RNG,
    so a "random control" was a *different network per task* and irreproducible
    across runs.  Here the seed is fixed at construction and re-applied
    identically after every ``fit`` (each estimator rebuilds ``model_`` inside
    ``fit``, so re-application is mandatory), which makes the control one fixed
    network.
    """

    seed: int
    #: Leave normalisation gains/biases at their trained values. Randomising a
    #: LayerNorm gain to N(0,1) does not produce "the same architecture with
    #: random weights", it produces a differently-conditioned network.
    preserve_norm_layers: bool = True


class InterpModel(abc.ABC):
    """Instrumented wrapper around one tabular foundation model.

    Subclasses implement the model-specific parts; everything callers touch is
    defined here and guaranteed by ``models.interp.verify``.
    """

    model_id: str

    def __init__(
        self,
        task: Task,
        device: str = "cuda",
        random_weights: RandomWeightSpec | None = None,
    ) -> None:
        self.task = Task(task)
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.random_weights = random_weights
        self.estimator = self._build_estimator()

    # -- subclass contract ---------------------------------------------------

    @abc.abstractmethod
    def _build_estimator(self):
        """Return the fitted-later sklearn estimator for ``self.task``."""

    @abc.abstractmethod
    def _torch_model(self) -> nn.Module:
        """The live torch module. Called *after* fit; never cached by callers."""

    @abc.abstractmethod
    def _block_resolvers(self) -> dict[str, callable]:
        """name -> zero-arg resolver for each hookable ICL block."""

    @abc.abstractmethod
    def _run_forward(self, X_test: np.ndarray):
        """Run the estimator's own predict path and return its raw output."""

    @abc.abstractmethod
    def _layout(self, n_support: int, n_query: int) -> TokenLayout:
        ...

    @property
    @abc.abstractmethod
    def n_layers(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def stream_view(self) -> StreamView:
        ...

    @abc.abstractmethod
    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        *,
        capture: CaptureSpec | None = None,
        patches: Sequence[PatchSpec] = (),
    ) -> Trace:
        """Fit on the context, predict the queries, and return a ``Trace``."""

    @abc.abstractmethod
    def make_patch(
        self,
        layer: int,
        values: np.ndarray | torch.Tensor,
        positions: Sequence[int] | slice | None = None,
        *,
        call_index: int | None = 0,
    ) -> PatchSpec:
        """Build a patch that writes ``values`` into block ``layer``'s output."""

    # -- shared machinery ----------------------------------------------------

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit and then (re-)apply the random-weight control.

        Order matters: every estimator here rebuilds ``model_`` from the
        checkpoint inside ``fit``, which would silently undo randomisation done
        at construction time.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train)
        if self.task is Task.CLASSIFICATION:
            y_train = y_train.astype(np.int64)
        else:
            y_train = y_train.astype(np.float64)
        self.estimator.fit(X_train, y_train)
        if self.random_weights is not None:
            self._apply_random_weights(self._torch_model(), self.random_weights)

    @staticmethod
    def _apply_random_weights(model: nn.Module, spec: RandomWeightSpec) -> None:
        """Deterministically re-initialise ``model`` in place.

        Uses a private generator seeded from ``spec.seed`` so the same spec
        always yields the same network, independent of global RNG state and of
        how many times it has been called.
        """
        generator = torch.Generator(device="cpu").manual_seed(spec.seed)
        norm_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.RMSNorm) if hasattr(nn, "RMSNorm") else (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
        )
        preserved: set[int] = set()
        if spec.preserve_norm_layers:
            for module in model.modules():
                if isinstance(module, norm_types):
                    preserved.update(id(p) for p in module.parameters(recurse=False))

        with torch.no_grad():
            for param in model.parameters():
                if id(param) in preserved:
                    continue
                sample = torch.empty(param.shape, dtype=torch.float32)
                if param.dim() > 1:
                    nn.init.xavier_normal_(sample, generator=generator)
                else:
                    # Biases and other 1-D parameters start at zero in a fresh
                    # init; a small normal keeps them non-degenerate without
                    # blowing up activations.
                    sample.normal_(0.0, 0.02, generator=generator)
                param.copy_(sample.to(device=param.device, dtype=param.dtype))

    def check_prediction_roundtrip(self, trace: Trace, X_test: np.ndarray) -> float:
        """Assert ``trace.pred`` is the estimator's own prediction.

        This single check is what makes a unit bug impossible to ship: if an
        adapter forgets to undo a scaler, or slices the wrong rows, the numbers
        stop matching and the run fails instead of silently producing a trace in
        the wrong units.
        """
        reference = self._reference_prediction(X_test)
        got = np.asarray(trace.pred, dtype=np.float64)
        ref = np.asarray(reference, dtype=np.float64)
        if got.shape != ref.shape:
            raise AssertionError(
                f"{self.model_id}: trace.pred shape {got.shape} != estimator prediction "
                f"shape {ref.shape}"
            )
        err = float(np.max(np.abs(got - ref))) if got.size else 0.0
        if not np.allclose(got, ref, atol=PRED_ROUNDTRIP_ATOL, rtol=PRED_ROUNDTRIP_RTOL):
            raise AssertionError(
                f"{self.model_id}: trace.pred does not reproduce the estimator's own "
                f"prediction (max abs err {err:.3e}). This is a units or layout bug."
            )
        return err

    def _reference_prediction(self, X_test: np.ndarray) -> np.ndarray:
        if self.task is Task.CLASSIFICATION:
            return self.estimator.predict_proba(np.asarray(X_test, dtype=np.float64))
        return np.asarray(
            self.estimator.predict(np.asarray(X_test, dtype=np.float64)), dtype=np.float64
        ).reshape(-1)

    # -- helpers for adapters ------------------------------------------------

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().to(torch.float64).cpu().numpy()

    def _new_trace(self, layout: TokenLayout) -> Trace:
        return Trace(
            model_id=self.model_id,
            task=self.task,
            layout=layout,
            stream_view=self.stream_view,
            n_layers=self.n_layers,
            d_model=self.d_model,
        )

    def describe(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "task": self.task.value,
            "device": self.device,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "stream_view": self.stream_view.value,
            "random_weights": None if self.random_weights is None else self.random_weights.seed,
        }
