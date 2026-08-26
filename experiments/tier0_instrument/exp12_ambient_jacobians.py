"""Experiments 1 and 2 — one measurement pass, full ambient Jacobians saved.

Both experiments need the AMBIENT Jacobian:
  Exp 1 (A3)        needs its diagonal, plus the predictive variance channel
  Exp 2 (profiling) needs the whole matrix

so it is measured once, per model per seed, and written to
exp12_ambient_jacobians.npz. Chunk 0 found that no Jacobian had ever been
persisted in this project, which is why every earlier recompute needed GPU time.
That ends here.

Configurations, and why:
  TabICL v2  h=1e-3, no dither. float32 output, effective quantum ~3e-6.
  TabSwift   h=1e-1, no dither. Coarse outputs; 1e-3 is inside the
             derivative-collapse regime (RESULTS_REPORT.md 4.3).
  TabPFN v2  h=1e-1, dither halfwidth delta, N=10, AND a no-dither pass.
             t=1e-1 per Chunk 3: the quantisation artifact falls as 1/t, from
             asym 0.0969 at 1e-2 to 0.0097 at 1e-1 for identical cost.

Variance channel (Exp 1 only):
  TabICL   output_type="quantiles" on 9999 uniform levels in [1e-4, 1-1e-4],
           integrated by core/metrics.py::extract_variance. Validated against
           the exact GP at 0.03% worst case (exp1_extractor_validation.json).
           NOT output_type="variance", which is raw_quantiles.var(dim=-1)
           (tabicl/_model/tabicl.py:582) -- a quantile-grid spread, ruled out by
           the E2.5 required fixes.
  TabPFN   FullSupportBarDistribution.variance(logits) = mean_of_square - mean^2
           (tabpfn/architectures/base/bar_distribution.py:606, :412): the
           integrated second moment with half-normal tails on the outer bins.
  TabSwift NOT COMPUTABLE -- Linear(384,1) point head.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules['analytics'] = _MockAnalytics()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from models import load
from experiments.core.context import generate_audit_context
from experiments.core.metrics import extract_variance

SEEDS = [42, 100, 200, 300, 400]
DELTA = 6.87e-4
LEVELS = np.linspace(1e-4, 1 - 1e-4, 9999)
HERE = Path(__file__).resolve().parent
NPZ = HERE / "exp12_ambient_jacobians.npz"

RUNS = [
    ("tabicl_v2",          "tabicl_v2",          1e-3, False, 1),
    ("tabswift",           "tabswift",           1e-1, False, 1),
    ("tabpfn_v2_nodither", "tabpfn_v2",          1e-1, False, 1),
    ("tabpfn_v2_dither",   "tabpfn_v2",          1e-1, True, 10),
]


def ambient_jacobian(predict, y, h, dither, N, seed):
    n = len(y)
    J = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        if dither:
            rng = np.random.RandomState(seed + i)
            noise = rng.uniform(-DELTA, DELTA, (N, n))
            mp = np.mean([np.asarray(predict(y + noise[k] + e)).ravel() for k in range(N)], axis=0)
            mm = np.mean([np.asarray(predict(y + noise[k] - e)).ravel() for k in range(N)], axis=0)
        else:
            mp = np.asarray(predict(y + e)).ravel()
            mm = np.asarray(predict(y - e)).ravel()
        J[:, i] = (mp - mm) / (2 * h)
    return J


def variance_channel(model_id, est, X):
    if model_id == "tabicl_v2":
        Q = np.asarray(est.predict(X, output_type="quantiles", alphas=list(LEVELS)))
        return np.asarray(extract_variance(Q, LEVELS)).ravel()
    if model_id == "tabpfn_v2":
        full = est.predict(X, output_type="full")
        crit, lg = full["criterion"], full["logits"]
        lg = lg if torch.is_tensor(lg) else torch.as_tensor(lg)
        lg = lg.to(next(crit.buffers()).device).float()
        return crit.variance(lg).detach().cpu().numpy().ravel()
    return None


def main():
    store, meta = {}, {}
    for tag, model_id, h, dither, N in RUNS:
        print(f"\n{'='*78}\n{tag}:  h={h:.0e}, dither={dither}"
              f"{f', halfwidth=delta, N={N}' if dither else ''}\n{'='*78}", flush=True)
        model = load(model_id, task="regression", device="cuda")
        meta[tag] = {"model_id": model_id, "h": h, "dither": dither,
                     "N": N if dither else 1,
                     "dither_halfwidth": DELTA if dither else None, "seeds": SEEDS}
        for s in SEEDS:
            t0 = time.time()
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)

            def predict(y_eval):
                model.estimator.fit(X, y_eval)
                if model_id == "tabpfn_v2":
                    return model.estimator.predict(X, output_type="mean")
                return model.estimator.predict(X)

            J = ambient_jacobian(predict, y, h, dither, N, s)
            model.estimator.fit(X, y)
            s2 = variance_channel(model_id, model.estimator, X)

            store[f"{tag}__J__{s}"] = J
            store[f"{tag}__y__{s}"] = y
            if s2 is not None:
                store[f"{tag}__s2__{s}"] = s2
            np.savez_compressed(NPZ, **store)
            json.dump(meta, open(HERE / "exp12_ambient_meta.json", "w"), indent=2)

            d = np.diag(J)
            msg = (f"  seed {s:<5} {time.time()-t0:6.1f}s  "
                   f"J_ii in [{d.min():+.5f}, {d.max():+.5f}]  "
                   f"cv_J {d.std()/abs(d.mean()):.5f}  ||J||_F {np.linalg.norm(J):.4f}")
            if s2 is not None:
                msg += f"  s2 in [{s2.min():.5f}, {s2.max():.5f}]"
            print(msg, flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"\nSaved {len(store)} arrays to {NPZ}", flush=True)


if __name__ == "__main__":
    main()
