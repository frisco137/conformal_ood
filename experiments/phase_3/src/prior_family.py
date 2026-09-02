"""T1.1 -- the prior-family context generator.

Draws contexts from the released TabICL prior and returns them in the form the
audit requires: a latent `f`, a Gaussian channel `y = f + eps` with `sigma^2`
recorded, and standardised features.

WHY THIS IS ASSEMBLED FROM COMPONENTS RATHER THAN FROM PriorDataset
-------------------------------------------------------------------
See `provenance_prior.md`. In short: the regression setting the audited
checkpoint records (`max_classes: 0`) makes the shipped `PriorDataset` raise
`ValueError: low >= high` at `prior/_dataset.py:652`, and no `max_classes == 0`
branch exists in that file. The prior's *components* do support continuous
targets -- `Reg2Cls` with `num_classes=0` leaves `y` continuous and
standard-scaled -- so this module drives `MLPSCM` / `TreeSCM` -> `Reg2Cls`
directly, which is the composition `generate_dataset` performs.

That composition is OUR RECONSTRUCTION of the documented pipeline. It is not
verified to be the pre-training prior. Every claim built on it says
"prior-family", never "in-family".

THE NOISE, AND WHY IT IS ADDED RATHER THAN FOUND
------------------------------------------------
The prior's noise lives *inside* every SCM layer (`_mlp_scm.py:201-214`) and is
pushed through subsequent nonlinearities, so the SCM's output is not `f + eps`
with a known `sigma^2` and no exposed knob makes it so.

Assumption 1 needs a Gaussian channel. Theorem 5 needs only that `P` be *some*
prior on R^n, and the pushforward of the SCM prior onto the sampled locations is
such a prior. So:

    rung A   f := SCM output (standard-scaled);  y := f + eps,  eps ~ N(0, s2 I)
             Assumption 1 HOLDS. sigma^2 is ours, recorded per context.
    rung B   y := SCM output directly, internal layer noise only
             Assumption 1 VIOLATED. A2/A3 are uninterpretable, not failed.

`sigma = 1.0` by default, which matches the Phase 1 audit family (unit-variance
latent, unit noise) so that rung A and rung D differ in the *prior over f* and
not in the noise level.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from tabicl.prior._dataset import SCMPrior                       # noqa: E402
from tabicl.prior._mlp_scm import MLPSCM                          # noqa: E402
from tabicl.prior._prior_config import (                          # noqa: E402
    DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP,
)
from tabicl.prior._reg2cls import Reg2Cls                         # noqa: E402
from tabicl.prior._tree_scm import TreeSCM                        # noqa: E402

__all__ = ["generate_prior_context", "PRIOR_DEFAULTS"]

#: Fixed choices, recorded so the generator is reproducible and auditable.
PRIOR_DEFAULTS = {
    "n": 100,                 # matches rung D and the Phase 2 12x5 convention
    "min_features": 2,
    "max_features": 20,       # prior default is 100; capped so d stays comparable
    "mix_probs": (0.7, 0.3),  # mlp_scm / tree_scm, the prior's own DEFAULT_FIXED_HP
    "sigma": 1.0,             # terminal noise sd for rung A
}


def _draw_params(rng_seed, n, d_min, d_max, prior_type):
    """One draw of the prior's own sampled hyperparameters, plus our fixed fields."""
    sp = SCMPrior(batch_size=1, min_features=d_min, max_features=d_max,
                  max_classes=10, max_seq_len=max(n, 128),
                  prior_type="mlp_scm", fixed_hp=DEFAULT_FIXED_HP,
                  sampled_hp=DEFAULT_SAMPLED_HP, n_jobs=1, device="cpu")
    grp = {k: (v() if callable(v) else v) for k, v in sp.hp_sampling().items()}
    d = int(round(np.random.uniform(d_min, d_max)))
    return {**DEFAULT_FIXED_HP, **grp,
            "seq_len": n, "train_size": max(2, n // 2),
            "max_features": d_max, "num_features": d,
            "prior_type": prior_type,
            "num_classes": 0,          # the continuous switch; see module docstring
            "device": "cpu"}, d, grp


def generate_prior_context(seed, rung="A", n=None, sigma=None,
                           d_min=None, d_max=None, max_tries=25):
    """Draw one prior-family context.

    Returns (X, y, f, meta). X is (n, d) standardised on its own rows, y is (n,),
    f is the latent (identical to y on rung B, where no terminal noise is added).

    rung 'A' : y = f + eps, eps ~ N(0, sigma^2 I).  Assumption 1 holds.
    rung 'B' : y = f, the SCM's native output.      Assumption 1 violated.
    """
    cfg = PRIOR_DEFAULTS
    n = cfg["n"] if n is None else n
    sigma = cfg["sigma"] if sigma is None else sigma
    d_min = cfg["min_features"] if d_min is None else d_min
    d_max = cfg["max_features"] if d_max is None else d_max

    rs = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    last = None
    for attempt in range(max_tries):
        prior_type = "mlp_scm" if rs.random() < cfg["mix_probs"][0] else "tree_scm"
        params, d, grp = _draw_params(seed + attempt, n, d_min, d_max, prior_type)
        cls = MLPSCM if prior_type == "mlp_scm" else TreeSCM
        try:
            with torch.no_grad():
                Xr, yr = cls(**params)()
                Xo, yo = Reg2Cls(params)(Xr.detach(), yr.detach())
        except Exception as e:                       # degenerate draw; resample
            last = f"{type(e).__name__}: {e}"
            continue

        X = np.asarray(Xo, float)[:, :d]
        f = np.asarray(yo, float).ravel()
        # Reg2Cls emits max_features columns, zero-padded past d; keep the live ones
        if X.shape[1] == 0 or not np.isfinite(X).all() or not np.isfinite(f).all():
            last = "non-finite or empty draw"
            continue
        keep = X.std(axis=0) > 1e-8
        if keep.sum() < 1:
            last = "all features constant"
            continue
        X = X[:, keep]
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)      # standardise on context rows only
        if f.std() < 1e-6:
            last = "constant target"
            continue

        if rung == "A":
            y = f + sigma * rs.randn(n)
            s2 = float(sigma ** 2)
        elif rung == "B":
            y = f.copy()
            s2 = None                                # not a Gaussian channel
        else:
            raise ValueError(f"unknown rung {rung!r}")

        meta = {"seed": seed, "rung": rung, "prior_type": prior_type,
                "n": int(n), "d_drawn": int(d), "d_kept": int(X.shape[1]),
                "sigma2": s2, "attempts": attempt + 1,
                "f_std": float(f.std()), "y_std": float(np.std(y)),
                "num_layers": int(grp.get("num_layers", -1)),
                "hidden_dim": int(grp.get("hidden_dim", -1)),
                "is_causal": bool(grp.get("is_causal", False)),
                "scm_noise_std": float(grp.get("noise_std", np.nan)),
                "sampling": str(grp.get("sampling", "?")),
                "assumption1": rung == "A"}
        return X, y, f, meta

    raise RuntimeError(f"no valid prior-family context after {max_tries} tries "
                       f"(seed {seed}, rung {rung}); last: {last}")


if __name__ == "__main__":
    print("prior_family self-test -- 6 contexts per rung")
    for rung in ("A", "B"):
        print(f"\n rung {rung}")
        for s in range(6):
            X, y, f, m = generate_prior_context(1000 + s, rung=rung)
            snr = f.var() / (np.var(y - f) + 1e-12) if rung == "A" else float("inf")
            print(f"   seed {m['seed']}  {m['prior_type']:<8} d={m['d_kept']:<3} "
                  f"n={m['n']}  f_std={m['f_std']:.3f}  y_std={m['y_std']:.3f}  "
                  f"sigma2={m['sigma2']}  SNR={snr:.2f}  tries={m['attempts']}")
