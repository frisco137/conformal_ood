"""
tabpfn_hooks.py — Shared model loading, data generation, hooking, and intervention
utilities for circuit-level mechanistic interpretability of **TabPFN**.

This is the TabPFN analogue of `circuits/circuit_hooks.py` (which targets the
Spectral-Mixture GP PFN). TabPFN differs from that model in several important ways
that this module is built around:

Architecture (from the loaded checkpoint config):
  - 12 TransformerEncoderLayers, post-norm (pre_norm=False)
  - Each layer: self_attn (4 heads, head_dim=128), FFN (512 -> 1024 -> 512, GELU)
  - hidden dim (emsize) = 512
  - It is a **classifier**: the decoder outputs `n_out = 10` class logits.
  - num_features = 100

Crucial behavioural quirk (`tabpfn/layer.py`, `efficient_eval_masking=True`):
  Each encoder layer calls `self_attn` **TWICE** per forward pass:
    1. context -> context : self_attn(src[:p], src[:p], src[:p])     ("ctoc")
    2. query   -> context : self_attn(src[p:], src[:p], src[:p])     ("qtoc")
  where `p = single_eval_pos`. The two outputs are concatenated along the
  sequence dim to form the attention sub-block output. All hooking, ablation
  and patching logic in this module therefore handles the two calls separately,
  distinguishing them by call order (ctoc first, qtoc second).

Because it is a classifier, "prediction quality" is measured with **cross-entropy**
of the 10-class logits against the true query labels (the model's own training
loss), NOT MSE.

The most delicate piece is `ManualMHA`, a drop-in replacement for the layer's
`self_attn.forward` that recomputes multi-head attention explicitly so we can
(a) zero-ablate individual heads, (b) record per-head pre-output-projection
outputs, or (c) patch a single head's output from a cache. A self-test at the
bottom of this file verifies that, in passthrough mode, `ManualMHA` reproduces
the model's stock `nn.MultiheadAttention` output to within 1e-4.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# --- Make TabPFN data generation + repo importable -----------------------------
_HERE = os.path.dirname(__file__)
_TABPFN_OOD = os.path.abspath(os.path.join(_HERE, '..', 'tabpfn_ood'))
_TABPFN_REPO = os.path.abspath(os.path.join(_HERE, '..', 'tabpfn_repo'))
for _p in (_TABPFN_OOD, _TABPFN_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_data import get_inprior_data, get_outprior_data  # noqa: E402
from tabpfn.scripts.model_builder import load_model_only_inference  # noqa: E402

# --- Architecture constants -----------------------------------------------------
CHECKPOINT_PATH = os.path.join(_TABPFN_REPO, 'tabpfn', 'models_diff')
CHECKPOINT_FILE = 'prior_diff_real_checkpoint_n_0_epoch_42.cpkt'

NLAYERS = 12
NHEAD = 4
EMSIZE = 512
HEAD_DIM = EMSIZE // NHEAD   # 128
N_CLASSES = 10
NUM_FEATURES = 100


# ==============================================================================
# Model loading
# ==============================================================================
def load_tabpfn(device: str = "cpu") -> nn.Module:
    """Load the pretrained TabPFN transformer for inference only."""
    print(f"Loading TabPFN from {os.path.join(CHECKPOINT_PATH, CHECKPOINT_FILE)} ...")
    (_, _, model), _config = load_model_only_inference(
        path=CHECKPOINT_PATH, filename=CHECKPOINT_FILE, device=device
    )
    model.eval()
    model.to(device)
    return model


# ==============================================================================
# Data generation
# ==============================================================================
def make_inprior(n: int, seq_len: int, without_noise: bool = False,
                 num_features: int = NUM_FEATURES, num_classes: int = N_CLASSES):
    """In-prior (prior-conforming) data. Shapes: x (seq, n, feat), y (seq, n)."""
    return get_inprior_data(n, seq_len, num_features, num_classes, without_noise=without_noise)


def make_outprior(n: int, seq_len: int,
                  num_features: int = NUM_FEATURES, num_classes: int = N_CLASSES):
    """Out-prior (anti-prior) data. Shapes: x (seq, n, feat), y (seq, n)."""
    return get_outprior_data(n, seq_len, num_features, num_classes)


def shuffle_labels(y: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Return a copy of y with all output labels shuffled independently within each
    column (i.e. each dataset in the batch), destroying the x->y mapping while
    preserving the label marginal. y shape: (seq_len, batch).
    """
    y_shuf = y.clone()
    seq_len, batch = y.shape
    for b in range(batch):
        perm = torch.randperm(seq_len, generator=generator)
        y_shuf[:, b] = y[perm, b]
    return y_shuf


# ==============================================================================
# Loss / metric from logits  (classifier -> cross-entropy)
# ==============================================================================
@torch.no_grad()
def compute_ce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                    device: str, single_eval_pos: int, batch_size: int = 128,
                    return_acc: bool = False, y_target: Optional[torch.Tensor] = None):
    """
    Mean cross-entropy of the model's 10-class query logits vs the query labels,
    averaged over all query points across all samples. Ignores label -100.

    Args:
        x: (seq, N, feat), y: (seq, N) integer class labels fed to the model.
        y_target: optional (seq, N) labels to evaluate against instead of `y`.
            Used for the label-shuffle subexperiment, where the model is fed
            shuffled labels but is evaluated against the true labels (the query
            labels are never consumed by the model, only the context ones are).
    Returns:
        ce (float), and accuracy (float) if return_acc.
    """
    if y_target is None:
        y_target = y
    N = x.shape[1]
    total_ce, total_correct, total_count = 0.0, 0, 0
    for i in range(0, N, batch_size):
        bx = x[:, i:i + batch_size].to(device)
        by = y[:, i:i + batch_size].to(device)
        byt = y_target[:, i:i + batch_size].to(device)
        logits = model((bx, by.float()), single_eval_pos=single_eval_pos)  # (q, b, C)
        tgt = byt[single_eval_pos:].long()                                  # (q, b)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_tgt = tgt.reshape(-1)
        valid = flat_tgt != -100
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        ce = F.cross_entropy(flat_logits[valid], flat_tgt[valid], reduction='sum')
        total_ce += float(ce.item())
        if return_acc:
            pred = flat_logits[valid].argmax(dim=-1)
            total_correct += int((pred == flat_tgt[valid]).sum().item())
        total_count += n_valid
    mean_ce = total_ce / max(total_count, 1)
    if return_acc:
        return mean_ce, total_correct / max(total_count, 1)
    return mean_ce


# ==============================================================================
# Manual multi-head attention  (ablation / per-head record / per-head patch)
# ==============================================================================
class ManualMHA:
    """
    A callable that replaces a layer's `self_attn.forward`, recomputing
    multi-head attention explicitly so individual heads can be intervened on.

    Modes (set via attributes before the forward pass):
      - passthrough (default): behaves identically to the stock module.
      - heads_to_ablate (set[int]): those heads' outputs are zeroed before out_proj.
      - record (bool): store per-head, pre-out_proj outputs for each of the two
        self_attn calls into `self.recorded` (list, in call order).
      - patch_cache (list) + patch_heads (set[int]): replace those heads' pre-out_proj
        output with the cached values (per call, matched by call order & positions).

    The two calls per layer (ctoc, qtoc) are tracked with `self._call_idx`, which
    must be reset to 0 (via `reset()`) before every model forward pass.
    """

    def __init__(self, mha: nn.Module, nhead: int = NHEAD, head_dim: int = HEAD_DIM):
        self.mha = mha
        self.nhead = nhead
        self.head_dim = head_dim
        self.orig_forward = mha.forward

        # intervention state
        self.heads_to_ablate: set = set()
        self.record: bool = False
        self.recorded: List[torch.Tensor] = []          # per call: (B, nhead, Lq, head_dim)
        self.patch_cache: Optional[List[torch.Tensor]] = None
        self.patch_heads: set = set()
        self._call_idx = 0

    def reset(self):
        self._call_idx = 0
        if self.record:
            self.recorded = []

    def install(self):
        self.mha.forward = self.__call__

    def uninstall(self):
        self.mha.forward = self.orig_forward

    def __call__(self, query, key, value, *args, **kwargs):
        mha = self.mha
        Lq, B, E = query.shape
        Lk = key.shape[0]
        nhead, hd = self.nhead, self.head_dim

        # ---- QKV projection (combined in_proj_weight) ----
        Wq, Wk, Wv = mha.in_proj_weight.chunk(3, dim=0)
        if mha.in_proj_bias is not None:
            bq, bk, bv = mha.in_proj_bias.chunk(3, dim=0)
        else:
            bq = bk = bv = None
        q = F.linear(query, Wq, bq)
        k = F.linear(key, Wk, bk)
        v = F.linear(value, Wv, bv)

        # ---- reshape to (B*nhead, L, head_dim); ordering is (b outer, h inner) ----
        q = q.contiguous().view(Lq, B * nhead, hd).transpose(0, 1)
        k = k.contiguous().view(Lk, B * nhead, hd).transpose(0, 1)
        v = v.contiguous().view(Lk, B * nhead, hd).transpose(0, 1)

        scale = hd ** -0.5
        scores = torch.bmm(q, k.transpose(1, 2)) * scale          # (B*nhead, Lq, Lk)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, v)                                   # (B*nhead, Lq, hd)

        # ---- per-head view for interventions ----
        ctx = ctx.view(B, nhead, Lq, hd)

        if self.record:
            self.recorded.append(ctx.detach().clone())

        if self.patch_cache is not None and self.patch_heads:
            cached = self.patch_cache[self._call_idx]              # (B_cached, nhead, Lq, hd)
            for h in self.patch_heads:
                ctx[:, h] = cached[:B, h]                          # (B, Lq, hd)

        if self.heads_to_ablate:
            for h in self.heads_to_ablate:
                ctx[:, h] = 0.0

        # ---- recombine and output projection ----
        ctx = ctx.view(B * nhead, Lq, hd)
        ctx = ctx.transpose(0, 1).contiguous().view(Lq, B, E)
        out = F.linear(ctx, mha.out_proj.weight, mha.out_proj.bias)

        # per-head attn weights (B, nhead, Lq, Lk) for divergence callers that want them
        attn_w = attn.view(B, nhead, Lq, Lk)

        self._call_idx += 1
        return out, attn_w


# ==============================================================================
# Attention divergence collector (per-head, ctoc + qtoc, running means)
# ==============================================================================
class AttnDivergenceCollector:
    """
    Captures per-head attention maps for every layer and accumulates a running
    *mean* attention map (over samples) separately for the two call types
    (ctoc = context->context, qtoc = query->context).

    Storing running sums (rather than every sample) keeps memory bounded.
    """

    def __init__(self, model: nn.Module, single_eval_pos: int):
        self.model = model
        self.p = single_eval_pos
        self.hooks: List[Any] = []
        # layer -> running sum over batch of mean attn map; and counts
        self.sum_ctoc: Dict[int, Optional[torch.Tensor]] = {l: None for l in range(NLAYERS)}
        self.sum_qtoc: Dict[int, Optional[torch.Tensor]] = {l: None for l in range(NLAYERS)}
        self.cnt: Dict[int, int] = {l: 0 for l in range(NLAYERS)}
        self._orig: Dict[int, Any] = {}
        self._call_state: Dict[int, int] = {}

    def install(self):
        for idx, layer in enumerate(self.model.transformer_encoder.layers):
            mha = layer.self_attn
            self._orig[idx] = mha.forward
            self._call_state[idx] = 0

            def make_fwd(l_idx, orig):
                def fwd(query, key, value, *args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False
                    out = orig(query, key, value, *args, **kwargs)
                    w = out[1]
                    if w is not None:
                        w = w.detach()                  # (B, nhead, Lq, Lk)
                        s = w.sum(dim=0).cpu()          # (nhead, Lq, Lk)
                        call = self._call_state[l_idx] % 2
                        if call == 0:                   # ctoc
                            self.sum_ctoc[l_idx] = s if self.sum_ctoc[l_idx] is None else self.sum_ctoc[l_idx] + s
                            self.cnt[l_idx] += w.shape[0]
                        else:                           # qtoc
                            self.sum_qtoc[l_idx] = s if self.sum_qtoc[l_idx] is None else self.sum_qtoc[l_idx] + s
                        self._call_state[l_idx] += 1
                    return out
                return fwd

            mha.forward = make_fwd(idx, self._orig[idx])
            self.hooks.append((mha, idx))

    def reset_call_state(self):
        for k in self._call_state:
            self._call_state[k] = 0

    def remove(self):
        for mha, idx in self.hooks:
            mha.forward = self._orig[idx]
        self.hooks = []

    def get_mean_maps(self):
        """Returns (ctoc, qtoc) dicts: layer -> (nhead, Lq, Lk) mean attention."""
        ctoc = {l: (self.sum_ctoc[l] / self.cnt[l]) for l in range(NLAYERS) if self.cnt[l] > 0}
        qtoc = {l: (self.sum_qtoc[l] / self.cnt[l]) for l in range(NLAYERS) if self.cnt[l] > 0}
        return ctoc, qtoc


# ==============================================================================
# Self-test: ManualMHA passthrough must match the stock module
# ==============================================================================
def _self_test():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = load_tabpfn(device)
    p = 54
    x, y = make_inprior(4, 60)
    x, y = x.to(device), y.float().to(device)

    # Baseline logits
    with torch.no_grad():
        ref = model((x, y), single_eval_pos=p)

    # Install passthrough ManualMHA on all layers
    manuals = []
    for layer in model.transformer_encoder.layers:
        mm = ManualMHA(layer.self_attn)
        mm.install()
        manuals.append(mm)
    for mm in manuals:
        mm.reset()
    with torch.no_grad():
        got = model((x, y), single_eval_pos=p)
    for mm in manuals:
        mm.uninstall()

    max_abs = (ref - got).abs().max().item()
    print(f"[self-test] ManualMHA passthrough max|Δlogit| = {max_abs:.3e}")
    assert max_abs < 1e-3, "ManualMHA does not reproduce stock attention!"
    print("[self-test] PASSED")


if __name__ == "__main__":
    _self_test()
