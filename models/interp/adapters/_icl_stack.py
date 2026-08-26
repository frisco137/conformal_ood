"""Shared implementation for the two staged row-attention models.

TabICL v2 and TabSwift have the same interior shape: a stack of blocks whose
output is ``(batch, tokens, d_model)``, preceded by column/row stages that are
outside the ICL index, and followed by ``ln`` + a decoder.  Only the details
differ (prefix tokens, quantile head, checkpoint), so the run loop lives here.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np
import torch

from ..base import CaptureSpec, InterpModel
from ..conventions import StreamView, TokenLayout, attn_selfcheck_rtol
from ..hooks import ForwardCapture, ForwardPatch, FunctionObserver, PatchSpec
from ..trace import AttentionRecord, Trace
from ._sdpa import classify, observe_sdpa


class IclStackInterp(InterpModel):
    """Base for models whose interior is a flat ``(B, T, D)`` block stack."""

    #: module object that owns ``sdpa_with_flattened_batch``
    _attention_module = None
    #: whether that function takes an ``ssmax_layer`` argument
    _supports_ssmax = False

    # -- subclass contract ---------------------------------------------------

    @abc.abstractmethod
    def _blocks(self):
        """The ICL block ModuleList of the live model."""

    @abc.abstractmethod
    def _n_prefix(self) -> int:
        """Non-data tokens prepended inside the ICL stack, read from config."""

    # -- shared ---------------------------------------------------------------

    @property
    def stream_view(self) -> StreamView:
        return StreamView.NATIVE_ROW

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def d_model(self) -> int:
        return self._d_model

    def _block_resolvers(self) -> dict[str, callable]:
        return {f"block_{l}": (lambda l=l: self._blocks()[l]) for l in range(self.n_layers)}

    def _layout(self, n_support: int, n_query: int) -> TokenLayout:
        return TokenLayout(n_prefix=self._n_prefix(), n_support=n_support, n_query=n_query)

    def _run_forward(self, X_test: np.ndarray):
        return self._reference_prediction(X_test)

    @staticmethod
    def _write(output: torch.Tensor, values: torch.Tensor, positions) -> torch.Tensor:
        """Write into a ``(batch, tokens, d_model)`` block output."""
        new = output.clone()
        idx = slice(None) if positions is None else (
            positions
            if isinstance(positions, slice)
            else torch.as_tensor(list(positions), device=output.device)
        )
        vals = values.to(device=output.device, dtype=output.dtype)
        if vals.dim() == 2:
            new[:, idx, :] = vals
        elif vals.dim() == 3:
            new[:, idx, :] = vals if vals.shape[0] == new.shape[0] else vals[0]
        else:
            raise ValueError(f"patch values must have rank 2 or 3; got {vals.dim()}")
        return new

    def make_patch(
        self,
        layer: int,
        values: np.ndarray | torch.Tensor,
        positions: Sequence[int] | slice | None = None,
        *,
        call_index: int | None = 0,
    ) -> PatchSpec:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers} blocks")
        return PatchSpec(
            module=f"block_{layer}",
            values=torch.as_tensor(np.asarray(values)),
            positions=positions,
            call_index=call_index,
            writer=self._write,
        )

    # -- attention scoping ---------------------------------------------------

    def _attention_context(self, context: dict) -> list:
        """Pre/post hooks that tag SDPA calls with the ICL block they came from.

        Calls made by the column/row stages leave ``context['where'] is None``
        and are ignored, so the ICL index stays clean.
        """
        handles = []
        for layer_index, block in enumerate(self._blocks()):
            handles.append(
                block.register_forward_pre_hook(
                    lambda _m, _i, layer_index=layer_index: context.__setitem__("where", layer_index)
                )
            )
            handles.append(
                block.register_forward_hook(
                    lambda _m, _i, _o: context.__setitem__("where", None)
                )
            )
        return handles

    # -- the run -------------------------------------------------------------

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        *,
        capture: CaptureSpec | None = None,
        patches: Sequence[PatchSpec] = (),
    ) -> Trace:
        capture = capture or CaptureSpec()
        X_test = np.asarray(X_test)
        self._fit(X_train, y_train)

        layout = self._layout(len(X_train), len(X_test))
        trace = self._new_trace(layout)
        wanted = capture.wanted_layers(self.n_layers)
        resolvers = self._block_resolvers()

        attn_records: list[dict] = []
        context: dict[str, object] = {"where": None}

        def observe(args, kwargs, out, index):
            if context["where"] is None:
                return None
            payload = observe_sdpa(args, kwargs, out, supports_ssmax=self._supports_ssmax)
            payload["layer"] = context["where"]
            attn_records.append(payload)
            return {"layer": context["where"]}

        pre_handles: list = []
        if capture.pre_icl:
            pre_icl_cap = []
            def hook(m, args, kwargs):
                tensor = args[0] if len(args) > 0 else list(kwargs.values())[0]
                pre_icl_cap.append(self._to_numpy(tensor))
            pre_handles.append(resolvers["block_0"]().register_forward_pre_hook(hook, with_kwargs=True))
            
        capture_cm = ForwardCapture({f"block_{l}": resolvers[f"block_{l}"] for l in wanted})
        patch_cm = ForwardPatch(resolvers, patches) if patches else None

        try:
            if capture.attention:
                pre_handles.extend(self._attention_context(context))
            with capture_cm:
                observer_cm = (
                    FunctionObserver(self._attention_module, "sdpa_with_flattened_batch", observe)
                    if capture.attention
                    else None
                )
                if patch_cm is not None:
                    patch_cm.__enter__()
                try:
                    if observer_cm is not None:
                        with observer_cm:
                            pred = self._run_forward(X_test)
                    else:
                        pred = self._run_forward(X_test)
                finally:
                    if patch_cm is not None:
                        patch_cm.__exit__(None, None, None)
        finally:
            for handle in pre_handles:
                handle.remove()

        capture_cm.require_call_count(1, where=f"{self.model_id}.run")
        trace.call_counts = capture_cm.call_counts()
        trace.pred = np.asarray(pred)

        for layer in wanted:
            state = capture_cm.only(f"block_{layer}")
            if state.dim() != 3:
                raise AssertionError(
                    f"{self.model_id}: expected a rank-3 (B, T, D) state, got {tuple(state.shape)}"
                )
            if state.shape[0] != 1:
                raise AssertionError(
                    f"{self.model_id}: expected batch 1, got {state.shape[0]}"
                )
            layout.check(state.shape[1], f"{self.model_id} block_{layer}")
            val = self._to_numpy(state[0])
            if getattr(self.estimator, "inference_precision", None) == torch.float32:
                val = val.astype(np.float32)
            trace.resid[layer] = val
            if capture.resid_full:
                val_full = self._to_numpy(state[0])
                if getattr(self.estimator, "inference_precision", None) == torch.float32:
                    val_full = val_full.astype(np.float32)
                trace.resid_full[layer] = val_full


        if capture.pre_icl and 'pre_icl_cap' in locals() and pre_icl_cap:
            trace.pre_icl['0'] = pre_icl_cap[0]

        if capture.attention:
            for record in attn_records:
                probs = record["probs"]
                rtol = attn_selfcheck_rtol(record["compute_dtype"])
                if record["selfcheck_rel_error"] > rtol:
                    raise AssertionError(
                        f"{self.model_id} layer {record['layer']}: recomputed attention does "
                        f"not reproduce the model's output (relative err "
                        f"{record['selfcheck_rel_error']:.3e} > {rtol:g} for dtype "
                        f"{record['compute_dtype']}). The observer is wrong."
                    )
                layer = record["layer"]
                trace.attn.setdefault(layer, []).append(
                    AttentionRecord(
                        layer=layer,
                        module="icl_block_attention",
                        call_index=len(trace.attn.get(layer, [])),
                        kind=classify(probs.shape[-2], probs.shape[-1], layout),
                        probs=probs,
                        selfcheck_error=record["selfcheck_error"],
                        selfcheck_rel_error=record["selfcheck_rel_error"],
                        compute_dtype=record["compute_dtype"],
                    )
                )

        if capture.layer_pred:
            self._add_layer_predictions(trace, X_test, wanted, resolvers)

        if not patches:
            trace.notes["pred_roundtrip_err"] = self.check_prediction_roundtrip(trace, X_test)
        return trace

    def _add_layer_predictions(
        self, trace: Trace, X_test: np.ndarray, wanted: Sequence[int], resolvers: dict
    ) -> None:
        """Decode every block through the model's own ``ln`` + decoder + head,
        by overwriting the last block's output and re-running the real predict
        path. Identity at the last block, so ``layer_pred[-1] == pred``."""
        last = self.n_layers - 1
        needed = sorted(set(wanted))
        with ForwardCapture({f"block_{l}": resolvers[f"block_{l}"] for l in needed}) as cap:
            self._run_forward(X_test)
        cap.require_call_count(1, where=f"{self.model_id}.layer_pred capture")

        for layer in needed:
            spec = PatchSpec(
                module=f"block_{last}",
                values=cap.only(f"block_{layer}"),
                positions=None,
                call_index=0,
                writer=self._write,
            )
            with ForwardPatch(resolvers, [spec]) as patch:
                trace.layer_pred[layer] = np.asarray(self._run_forward(X_test))
            patch.require_applied(1)
