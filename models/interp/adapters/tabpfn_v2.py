"""TabPFN v2 adapter.

Architecture, as measured (not as assumed):

* ``PerFeatureTransformer``, ``transformer_encoder.layers`` = **12** blocks,
  ``d_model = 192``.  (The "24 blocks" in the project plan is TabPFN 2.5, a
  different checkpoint.)
* The state is per-**cell**: each block's output is
  ``(batch, items, feature_groups, d_model)``.  ``feature_groups`` depends on
  the input width and the preprocessing pipeline and is read off the tensor,
  never computed from ``d``.
* The model reads only the LAST feature slot to predict:
  ``transformer.py`` does ``encoder_out[:, single_eval_pos:, -1]`` before the
  decoder.  That slot is therefore the canonical per-row stream
  (``StreamView.TARGET_SLOT``); the unreduced tensor stays available.
* Each block runs **three** attention calls:
  ``self_attn_between_features`` once (q=k=feature groups, batch axis = items),
  then ``self_attn_between_items`` **twice** -- first query->support, then
  support->support (batch axis = feature groups).  Both item calls are kept as
  separate records; neither is allowed to overwrite the other.
* Prefix tokens are read from ``add_thinking_tokens``: **0** for v2, **64**
  for v2.5 (which is also 18 blocks, not 24).

Attention probabilities are recovered with a delegating observer around
``MultiHeadAttention.compute_attention_heads``: the original does the real
computation, we only recompute the probabilities from the same q/k and check
that ``probs @ v`` reproduces the original's output.  The function is restored
on exit, so nothing about the model is permanently altered.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.architectures.base.attention.full_attention import MultiHeadAttention
from tabpfn.constants import ModelVersion

from ..base import CaptureSpec, InterpModel, RandomWeightSpec
from ..conventions import StreamView, Task, TokenLayout, attn_selfcheck_rtol
from ..hooks import ForwardCapture, ForwardPatch, FunctionObserver, PatchSpec
from ..trace import AttentionRecord, Trace
from ._sdpa import classify

_ITEM_ATTN = "self_attn_between_items"
_FEATURE_ATTN = "self_attn_between_features"


class TabPFNv2Interp(InterpModel):
    model_id = "tabpfn_v2"
    _model_version = ModelVersion.V2

    def __init__(
        self,
        task: Task | str = Task.REGRESSION,
        device: str = "cuda",
        random_weights: RandomWeightSpec | None = None,
    ) -> None:
        super().__init__(task=Task(task), device=device, random_weights=random_weights)
        # One throwaway fit so the checkpoint is loaded and the architecture can
        # be measured rather than guessed. Also fails fast on a bad checkpoint.
        self._warmup()

    # -- construction --------------------------------------------------------

    def _build_estimator(self):
        cls = TabPFNRegressor if self.task is Task.REGRESSION else TabPFNClassifier
        return cls.create_default_for_version(
            self._model_version,
            device=self.device,
            n_estimators=1,  # one ensemble member => one forward pass to hook
            random_state=0,
            ignore_pretraining_limits=True,
        )

    def _warmup(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.randn(16, 3)
        y = rng.randn(16) if self.task is Task.REGRESSION else np.arange(16) % 2
        self._fit(X, y)
        model = self._torch_model()
        self._n_layers = len(model.transformer_encoder.layers)
        self._d_model = int(getattr(model, "emsize", getattr(model, "embedding_dim", getattr(model, "ninp", 192))))

    # -- model access --------------------------------------------------------

    def _torch_model(self):
        return self.estimator.model_

    def _blocks(self):
        return self._torch_model().transformer_encoder.layers

    def _block_resolvers(self) -> dict[str, callable]:
        return {f"block_{l}": (lambda l=l: self._blocks()[l]) for l in range(self.n_layers)}

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def stream_view(self) -> StreamView:
        return StreamView.TARGET_SLOT

    def _n_prefix(self) -> int:
        """Thinking tokens prepended to the item axis, read from the model.

        v2 has none; v2.5 prepends 64 (``AddThinkingTokens`` does
        ``cat([thinking, embedded], dim=1)`` and shifts ``single_eval_pos``),
        exactly like TabSwift's register tokens. Never hardcode this -- it is a
        per-checkpoint fact, and assuming 0 silently misaligns every row index.
        """
        adder = getattr(self._torch_model(), "add_thinking_tokens", None)
        return 0 if adder is None else int(adder.num_thinking_rows)

    def _layout(self, n_support: int, n_query: int) -> TokenLayout:
        return TokenLayout(n_prefix=self._n_prefix(), n_support=n_support, n_query=n_query)

    def _run_forward(self, X_test: np.ndarray):
        return self._reference_prediction(X_test)

    # -- patching ------------------------------------------------------------

    @staticmethod
    def _write(output: torch.Tensor, values: torch.Tensor, positions) -> torch.Tensor:
        """Write into a ``(batch, items, groups, d_model)`` block output.

        ``values`` with rank 2 ``(n_pos, d_model)`` writes the target slot only;
        rank 3 ``(n_pos, groups, d_model)`` writes the full cell state.
        """
        new = output.clone()
        idx = slice(None) if positions is None else (
            positions if isinstance(positions, slice) else torch.as_tensor(list(positions), device=output.device)
        )
        vals = values.to(device=output.device, dtype=output.dtype)
        if vals.dim() == 2:
            new[:, idx, -1, :] = vals
        elif vals.dim() == 3:
            new[:, idx, :, :] = vals
        elif vals.dim() == 4:
            new[:, idx, :, :] = vals[0] if vals.shape[0] == 1 else vals
        else:
            raise ValueError(f"TabPFN patch values must have rank 2, 3 or 4; got {vals.dim()}")
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

    # -- attention observation -----------------------------------------------

    def _attention_observer(self, records: list[dict]):
        """Delegating observer that recomputes attention probabilities.

        A pair of forward-pre hooks (installed by ``_attention_context``) tells
        us which layer and which attention module the current call belongs to.
        """
        context: dict[str, object] = {"where": None}

        def observe(args, kwargs, out, index):
            merged = dict(zip(("q", "k", "v", "kv", "qkv"), args))
            merged.update(kwargs)
            q, k, v = merged.get("q"), merged.get("k"), merged.get("v")
            kv, qkv = merged.get("kv"), merged.get("qkv")
            softmax_scale = merged.get("softmax_scale")
            if qkv is not None:
                q, k, v = qkv.unbind(dim=-3)
            elif kv is not None:
                k, v = kv.unbind(dim=-3)
            where = context["where"]
            if where is None:
                return None

            # See _sdpa._no_autocast: without disabling autocast the recomputation
            # is silently done in fp16 and the probabilities come out wrong.
            with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                nhead = q.shape[2]
                nhead_kv = k.shape[2]
                share = nhead // nhead_kv
                kb = MultiHeadAttention.broadcast_kv_across_heads(k, share)
                vb = MultiHeadAttention.broadcast_kv_across_heads(v, share)
                d_k = q.shape[-1]
                scale = float(softmax_scale) if softmax_scale is not None else 1.0 / np.sqrt(d_k)
                logits = torch.einsum("bqhd,bkhd->bqkh", q.float(), kb.float()) * scale
                probs = torch.softmax(logits, dim=2)
                recon = torch.einsum("bqkh,bkhd->bqhd", probs, vb.float())
                err = float((recon - out.float()).abs().max())
                scale = float(out.float().abs().max())
                # (batch, q, k, heads) -> (batch, heads, q, k)
                probs_np = probs.permute(0, 3, 1, 2).to(torch.float64).cpu().numpy()

            layer, module = where
            records.append(
                {
                    "layer": layer,
                    "module": module,
                    "probs": probs_np,
                    "selfcheck_error": err,
                    "selfcheck_rel_error": err / (scale + 1e-9),
                    "compute_dtype": str(q.dtype),
                }
            )
            return {"layer": layer}

        return observe, context

    def _attention_context(self, context: dict) -> list:
        handles = []
        for layer_index, block in enumerate(self._blocks()):
            for name in (_FEATURE_ATTN, _ITEM_ATTN):
                module = getattr(block, name, None)
                if module is None:
                    continue

                def pre_hook(_m, _inp, layer_index=layer_index, name=name):
                    context["where"] = (layer_index, name)

                handles.append(module.register_forward_pre_hook(pre_hook))
        return handles

    def _classify_attention(self, module: str, q_len: int, k_len: int, layout: TokenLayout) -> str:
        if module == _FEATURE_ATTN:
            return "feature_to_feature"
        return classify(q_len, k_len, layout)

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
        observe, context = self._attention_observer(attn_records)
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
                pre_handles = self._attention_context(context)
            with capture_cm:
                observer_cm = (
                    FunctionObserver(
                        MultiHeadAttention, "compute_attention_heads", observe, staticmethod_=True
                    )
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
            state = capture_cm.only(f"block_{layer}")  # (batch, items, groups, d)
            if state.dim() != 4:
                raise AssertionError(
                    f"{self.model_id}: expected a rank-4 per-cell state, got {tuple(state.shape)}"
                )
            if state.shape[0] != 1:
                raise AssertionError(
                    f"{self.model_id}: expected batch 1 (n_estimators=1), got {state.shape[0]}"
                )
            layout.check(state.shape[1], f"{self.model_id} block_{layer}")
            native = state[0]  # (items, groups, d_model)
            val = self._to_numpy(native[:, -1, :])
            if getattr(self.estimator, "inference_precision", None) == torch.float32:
                val = val.astype(np.float32)
            trace.resid[layer] = val
            if capture.resid_full:
                val_full = self._to_numpy(native)
                if getattr(self.estimator, "inference_precision", None) == torch.float32:
                    val_full = val_full.astype(np.float32)
                trace.resid_full[layer] = val_full

        if capture.pre_icl and 'pre_icl_cap' in locals() and pre_icl_cap:
            trace.pre_icl['0'] = pre_icl_cap[0]

        if capture.attention:
            for record in attn_records:
                probs = record["probs"]
                kind = self._classify_attention(
                    record["module"], probs.shape[-2], probs.shape[-1], layout
                )
                rtol = attn_selfcheck_rtol(record["compute_dtype"])
                if record["selfcheck_rel_error"] > rtol:
                    raise AssertionError(
                        f"{self.model_id} layer {record['layer']} {record['module']}: "
                        f"recomputed attention does not reproduce the model's output "
                        f"(relative err {record['selfcheck_rel_error']:.3e} > {rtol:g} for "
                        f"dtype {record['compute_dtype']}). The observer is wrong."
                    )
                trace.attn.setdefault(record["layer"], []).append(
                    AttentionRecord(
                        layer=record["layer"],
                        module=record["module"],
                        call_index=len(trace.attn.get(record["layer"], [])),
                        kind=kind,
                        probs=probs,
                        selfcheck_error=record["selfcheck_error"],
                        selfcheck_rel_error=record["selfcheck_rel_error"],
                        compute_dtype=record["compute_dtype"],
                    )
                )
            trace.notes["attn_batch_axis"] = (
                "item attention: batch axis = feature groups; "
                "feature attention: batch axis = items"
            )

        if capture.layer_pred:
            self._add_layer_predictions(trace, X_test, wanted, resolvers)

        if not patches:
            trace.notes["pred_roundtrip_err"] = self.check_prediction_roundtrip(trace, X_test)
        return trace

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
            quantiles=[float(a) for a in levels],
        )
        return np.stack([np.asarray(qi, dtype=np.float64) for qi in q], axis=1), levels

    def _add_layer_predictions(
        self, trace: Trace, X_test: np.ndarray, wanted: Sequence[int], resolvers: dict
    ) -> None:
        """Logit-lens read of every block, through the model's OWN tail.

        Rather than reimplement the decoder + bar-distribution + border
        translation (which is where a hand-rolled per-layer decoder silently
        drifts), we overwrite the LAST block's output with block ``l``'s state
        and re-run the estimator's real predict path.  At ``l = n_layers - 1``
        this is the identity, so ``layer_pred[-1] == pred`` by construction --
        and `verify` asserts exactly that.
        """
        last = self.n_layers - 1
        needed = sorted(set(wanted))
        with ForwardCapture({f"block_{l}": resolvers[f"block_{l}"] for l in needed}) as cap:
            self._run_forward(X_test)
        cap.require_call_count(1, where=f"{self.model_id}.layer_pred capture")

        for layer in needed:
            native = cap.only(f"block_{layer}")
            spec = PatchSpec(
                module=f"block_{last}",
                values=native,
                positions=None,
                call_index=0,
                writer=self._write,
            )
            with ForwardPatch(resolvers, [spec]) as patch:
                trace.layer_pred[layer] = np.asarray(self._run_forward(X_test))
            patch.require_applied(1)


class TabPFNv2_5Interp(TabPFNv2Interp):
    """TabPFN 2.5 -- same architecture family, different checkpoint.

    Present only as the *within-family scale bar* for cross-model alignment:
    it answers "how different do two checkpoints of the same architecture
    look?", which is the calibration any cross-model distance claim needs.
    Regressor only; the classifier repo is gated (see registry).
    """

    model_id = "tabpfn_v2_5"
    _model_version = ModelVersion.V2_5

    def _build_estimator(self):
        if self.task is not Task.REGRESSION:
            raise ValueError(
                "TabPFN v2.5 classifier weights are gated upstream (HTTP 401); "
                "only the regressor is available."
            )
        return super()._build_estimator()
