"""Shared attention observer for models that funnel through ``sdpa_with_flattened_batch``.

Both TabICL v2 and TabSwift route every attention call through one small
function that hands PyTorch's fused SDPA the already-projected, already-RoPE'd
q/k/v.  Fused SDPA never materialises the probability matrix, so we recompute it
from the same q/k the real kernel saw.

The recomputation is checked against the real output on every call
(``probs @ v`` must reproduce it).  That check is what makes this safe: if the
mask handling, the SSMax scaling, or the head layout were wrong, the error
blows up and the run fails instead of quietly returning plausible-looking
numbers.
"""

from __future__ import annotations

import numpy as np
import torch


def _no_autocast(device_type: str):
    """Disable autocast for the recomputation.

    Critical and easy to miss: these models run their forward pass inside
    ``torch.autocast``, and ``einsum``/``matmul``/``scaled_dot_product_attention``
    are all on the autocast cast-to-fp16 list.  Calling ``.float()`` on the
    inputs is NOT enough -- autocast casts them straight back, so the softmax
    is computed in fp16 and the recovered probabilities are wrong by up to 18%
    while still looking like a plausible attention matrix.
    """
    return torch.autocast(device_type=device_type, enabled=False)


def observe_sdpa(args: tuple, kwargs: dict, out: torch.Tensor, *, supports_ssmax: bool) -> dict:
    """Recompute attention probabilities for one ``sdpa_with_flattened_batch`` call.

    Returns ``{"probs": (batch, heads, q, k) float64, "selfcheck_error": float}``.
    """
    names = ["q", "k", "v", "attn_mask", "dropout_p"] + (["ssmax_layer"] if supports_ssmax else [])
    merged: dict = dict(zip(names, args))
    merged.update(kwargs)

    q, k, v = merged["q"], merged["k"], merged["v"]
    attn_mask = merged.get("attn_mask")
    ssmax_layer = merged.get("ssmax_layer")

    device_type = q.device.type
    with torch.no_grad(), _no_autocast(device_type):
        q_shape = q.shape
        qf = q.reshape(-1, *q.shape[-3:]).float()
        kf = k.reshape(-1, *k.shape[-3:]).float()
        vf = v.reshape(-1, *v.shape[-3:]).float()
        mask = attn_mask.reshape(-1, *attn_mask.shape[-3:]) if attn_mask is not None else None

        # Mirror the source: SSMax is applied to q *inside* the function, before SDPA.
        if supports_ssmax and ssmax_layer is not None:
            qf = ssmax_layer(qf, kf.size(-2)).float()

        scale = 1.0 / np.sqrt(qf.shape[-1])
        logits = torch.matmul(qf, kf.transpose(-2, -1)) * scale
        if mask is not None:
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, float("-inf"))
            else:
                logits = logits + mask.float()
        probs = torch.softmax(logits, dim=-1)
        recon = torch.matmul(probs, vf).view(q_shape)
        err = float((recon - out.float()).abs().max())
        scale = float(out.float().abs().max())
        probs_np = probs.to(torch.float64).cpu().numpy()

    return {
        "probs": probs_np,
        "selfcheck_error": err,
        "selfcheck_rel_error": err / (scale + 1e-9),
        "compute_dtype": str(q.dtype),
    }


def span_name(length: int, layout) -> str:
    """Name the token span an attention axis covers.

    Ordered most-specific-first, because spans coincide when ``n_prefix`` is 0
    (then ``context`` and ``support`` are the same length and "support" is the
    honest name).
    """
    candidates = [
        ("support", layout.n_support),
        ("query", layout.n_query),
        ("prefix", layout.n_prefix),
        ("context", layout.n_prefix + layout.n_support),
        ("all", layout.total),
    ]
    for name, size in candidates:
        if size and length == size:
            return name
    return f"len{length}"


def classify(q_len: int, k_len: int, layout) -> str:
    """Name an attention call from its q/k extents, e.g. ``query_to_context``."""
    return f"{span_name(q_len, layout)}_to_{span_name(k_len, layout)}"
