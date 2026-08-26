"""E1.1 — shift equivariance and scale homogeneity, forward passes only.

    shift:  m(y + c*1) =? m(y) + c*1     shift_err = ||lhs - rhs||_2 / ||m(y)||_2
    scale:  m(c*y)     =? c * m(y)       scale_err = ||lhs - rhs||_2 / ||c*m(y)||_2

No derivatives, no projection, no dithering. Dither is a device for smoothing a
quantised output before finite differencing; these are single forward passes, so
it is off for every model including TabPFN.

Contexts: generate_audit_context(n=100, d=5, sigma=1.0, seed=s), seeds
[42, 100, 200, 300, 400].
"""
import sys
import json
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
from experiments.core.surrogates import ExactGP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk1_control_validation import wrap, GP_SIGMA

SEEDS = [42, 100, 200, 300, 400]
SHIFTS = [0.1, 0.5, 1.0]
SCALES = [0.5, 2.0]
HERE = Path(__file__).resolve().parent


def measure(predict, y):
    m0 = np.asarray(predict(y)).ravel()
    n0 = np.linalg.norm(m0)
    out = {"shift": {}, "scale": {}, "norm_m0": float(n0)}
    for c in SHIFTS:
        lhs = np.asarray(predict(y + c)).ravel()
        out["shift"][str(c)] = float(np.linalg.norm(lhs - m0 - c) / n0)
    for c in SCALES:
        lhs = np.asarray(predict(c * y)).ravel()
        out["scale"][str(c)] = float(np.linalg.norm(lhs - c * m0) / np.linalg.norm(c * m0))
    return out


def run_model(model_id):
    model = load(model_id, task="regression", device="cuda")
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            if model_id == "tabpfn_v2":
                return model.estimator.predict(X, output_type="mean")
            return model.estimator.predict(X)

        r = measure(predict, y); r["seed"] = s
        rows.append(r)
        print(f"    seed {s:<5} shift " +
              "  ".join(f"c={c}: {r['shift'][str(c)]:.6e}" for c in SHIFTS), flush=True)
        print(f"              scale " +
              "  ".join(f"c={c}: {r['scale'][str(c)]:.6e}" for c in SCALES), flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def run_control(kind):
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        predict = wrap(egp.predict) if kind == "wrapped" else (lambda yy: egp.predict(yy))
        r = measure(predict, y); r["seed"] = s
        rows.append(r)
        print(f"    seed {s:<5} shift " +
              "  ".join(f"c={c}: {r['shift'][str(c)]:.6e}" for c in SHIFTS))
        print(f"              scale " +
              "  ".join(f"c={c}: {r['scale'][str(c)]:.6e}" for c in SCALES))
    return rows


def summarise(tag, rows):
    print(f"\n  {tag}  means over 5 seeds")
    for c in SHIFTS:
        v = np.array([r["shift"][str(c)] for r in rows])
        print(f"    shift c={c:<5} mean {v.mean():.6e}   max {v.max():.6e}")
    for c in SCALES:
        v = np.array([r["scale"][str(c)] for r in rows])
        print(f"    scale c={c:<5} mean {v.mean():.6e}   max {v.max():.6e}")


def main():
    out = {"_config": {"seeds": SEEDS, "shifts": SHIFTS, "scales": SCALES,
                       "dither": False, "n": 100, "d": 5, "context_sigma": 1.0}}
    for tag, fn in [("exactgp_wrapped", lambda: run_control("wrapped")),
                    ("exactgp_unwrapped", lambda: run_control("unwrapped")),
                    ("tabicl_v2", lambda: run_model("tabicl_v2")),
                    ("tabpfn_v2", lambda: run_model("tabpfn_v2")),
                    ("tabswift", lambda: run_model("tabswift"))]:
        print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)
        out[tag] = fn()
        summarise(tag, out[tag])
        json.dump(out, open(HERE / "e11_wrapper_form.json", "w"), indent=2)
    print(f"\nSaved: {HERE / 'e11_wrapper_form.json'}")


if __name__ == "__main__":
    main()
