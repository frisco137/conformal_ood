"""CHUNK 2, part 1 -- OpenML dataset selection and split construction.

SELECTION RULE, fixed before any result is looked at
----------------------------------------------------
A dataset is admitted iff ALL of:

  R1  listed on OpenML with status "active" and a declared default target
  R2  200 <= n_rows <= 2000        (need n_ctx=100 plus calibration plus test)
  R3  2 <= n_features <= 50        after dropping the target
  R4  every feature numeric        (no categorical encoding is introduced here;
                                    encoding choices would be a free parameter)
  R5  zero missing values
  R6  target numeric, non-constant, with >= 20 distinct values
      (excludes ordinal/count targets that are classification in disguise)
  R7  n_rows >= 200 after dropping duplicate rows

Candidates are ordered by OpenML id ascending and the first `N_TARGET` that pass
are taken. Ordering by id is arbitrary but fixed in advance and independent of
anything measured later; no dataset is dropped after results are seen. Every
candidate that fails is recorded with the rule it failed, in
`chunk2_datasets.json -> rejected`.

SPLITS
------
Per dataset, 5 splits with seeds [42, 100, 200, 300, 400]. Each split shuffles
the rows and takes
    context      first  n_ctx = 100          (what the model is fitted on)
    calibration  next   n_cal  <= 100        (split conformal only)
    test         next   n_test <= 100        (everything is scored here)
Features are standardised with statistics from the CONTEXT ROWS ONLY. Targets
are left untouched -- the models handle targets themselves, and standardising
them would change the very preprocessing path the audit characterised.

Writes chunk2_datasets.json (manifest) and chunk2_data.npz (arrays).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))              # repo root
warnings.filterwarnings("ignore")

from experiments.phase_2.src.paths import RESULTS, ARRAYS      # noqa: E402
from experiments.phase_2.src.estimators import jsonable        # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
N_CTX = 100
N_CAL_MAX = 100
#: capped at 50 to keep the appended-query pass affordable: each test point
#: costs two forward passes, and the full grid is 3 models x 12 datasets x
#: 5 splits. 50 x 5 x 12 = 3000 scored points per model. Stated, not silent.
N_TEST_MAX = 50
N_TARGET = 12                  # aim inside the brief's 8-15 band
N_MIN, N_MAX = 200, 2000
D_MIN, D_MAX = 2, 50
MIN_TARGET_UNIQUE = 20


def candidates():
    import openml
    dl = openml.datasets.list_datasets(output_format="dataframe")
    dl = dl[(dl["status"] == "active")]
    dl = dl[(dl["NumberOfInstances"] >= N_MIN) & (dl["NumberOfInstances"] <= N_MAX)]
    dl = dl[(dl["NumberOfFeatures"] >= D_MIN + 1) & (dl["NumberOfFeatures"] <= D_MAX + 1)]
    dl = dl[(dl["NumberOfMissingValues"] == 0)]
    dl = dl[(dl["NumberOfSymbolicFeatures"] <= 1)]     # target may be listed here
    # NumberOfClasses is 0 for regression targets on OpenML
    dl = dl[(dl["NumberOfClasses"].fillna(0) == 0)]
    return dl.sort_values("did")


def try_load(did):
    """Return (X, y, name) or raise with the rule that failed."""
    import openml
    ds = openml.datasets.get_dataset(int(did), download_data=True,
                                     download_qualities=False,
                                     download_features_meta_data=True)
    if ds.default_target_attribute is None:
        raise ValueError("R1 no default target")
    target = ds.default_target_attribute
    if "," in target:
        raise ValueError("R1 multiple default targets")
    X, y, cat, names = ds.get_data(target=target, dataset_format="dataframe")
    if y is None:
        raise ValueError("R1 target not returned")
    if any(cat):
        raise ValueError("R4 categorical feature present")
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] < D_MIN or X.shape[1] > D_MAX:
        raise ValueError(f"R3 d={X.shape[1]} outside [{D_MIN},{D_MAX}]")
    if not np.issubdtype(np.asarray(y).dtype, np.number):
        raise ValueError("R6 target not numeric")
    Xv = np.asarray(X, float)
    yv = np.asarray(y, float).ravel()
    keep = np.isfinite(Xv).all(1) & np.isfinite(yv)
    Xv, yv = Xv[keep], yv[keep]
    if not keep.all():
        raise ValueError("R5 non-finite entries present")
    _, uniq_idx = np.unique(Xv, axis=0, return_index=True)
    if len(uniq_idx) < N_MIN:
        raise ValueError(f"R7 only {len(uniq_idx)} distinct rows")
    if len(yv) < N_MIN or len(yv) > N_MAX:
        raise ValueError(f"R2 n={len(yv)} outside [{N_MIN},{N_MAX}]")
    if np.std(yv) == 0:
        raise ValueError("R6 constant target")
    if len(np.unique(yv)) < MIN_TARGET_UNIQUE:
        raise ValueError(f"R6 target has {len(np.unique(yv))} distinct values")
    # drop zero-variance features; they carry no information and break scaling
    keep_c = np.std(Xv, axis=0) > 0
    Xv = Xv[:, keep_c]
    if Xv.shape[1] < D_MIN:
        raise ValueError("R3 too few non-constant features")
    return Xv, yv, ds.name


def make_splits(X, y, seed):
    n = len(y)
    idx = np.random.RandomState(seed).permutation(n)
    ctx = idx[:N_CTX]
    rest = idx[N_CTX:]
    half = len(rest) // 2
    cal = rest[:min(half, N_CAL_MAX)]
    test = rest[half:half + N_TEST_MAX]
    mu, sd = X[ctx].mean(0), X[ctx].std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    return {"ctx": ctx, "cal": cal, "test": test,
            "X_ctx": Z[ctx], "y_ctx": y[ctx],
            "X_cal": Z[cal], "y_cal": y[cal],
            "X_test": Z[test], "y_test": y[test]}


def main():
    print("=" * 78)
    print("CHUNK 2.1  dataset selection")
    print("=" * 78)
    print(f"  rule: active, {N_MIN}<=n<={N_MAX}, {D_MIN}<=d<={D_MAX}, all-numeric,")
    print(f"        no missing, numeric target with >={MIN_TARGET_UNIQUE} distinct values,")
    print(f"        >={N_MIN} distinct rows; ordered by OpenML id; first {N_TARGET} taken.")

    cand = candidates()
    print(f"  candidates passing the listing-level filters: {len(cand)}")

    accepted, rejected, store = [], [], {}
    for did in cand["did"].tolist():
        if len(accepted) >= N_TARGET:
            break
        t0 = time.time()
        try:
            X, y, name = try_load(did)
        except Exception as e:
            rejected.append({"did": int(did), "reason": f"{type(e).__name__}: {e}"[:200]})
            continue
        rec = {"did": int(did), "name": name, "n": int(len(y)), "d": int(X.shape[1]),
               "target_mean": float(np.mean(y)), "target_sd": float(np.std(y)),
               "target_min": float(np.min(y)), "target_max": float(np.max(y)),
               "target_n_unique": int(len(np.unique(y))),
               "load_seconds": round(time.time() - t0, 2)}
        accepted.append(rec)
        store[f"{did}__X"] = X
        store[f"{did}__y"] = y
        print(f"  ACCEPT did={did:<6} {name[:34]:<34} n={rec['n']:<5} d={rec['d']:<3} "
              f"target sd {rec['target_sd']:.4g}")

    print(f"\n  accepted {len(accepted)}, rejected {len(rejected)} "
          f"(reasons in chunk2_datasets.json -> rejected)")

    print("\n" + "=" * 78)
    print("CHUNK 2.2  splits")
    print("=" * 78)
    split_meta = {}
    for rec in accepted:
        did = rec["did"]
        X, y = store[f"{did}__X"], store[f"{did}__y"]
        per_seed = {}
        for s in SEEDS:
            sp = make_splits(X, y, s)
            for k in ["X_ctx", "y_ctx", "X_cal", "y_cal", "X_test", "y_test"]:
                store[f"{did}__{s}__{k}"] = sp[k]
            per_seed[s] = {"n_ctx": int(len(sp["y_ctx"])), "n_cal": int(len(sp["y_cal"])),
                           "n_test": int(len(sp["y_test"]))}
        split_meta[did] = per_seed
        z = per_seed[SEEDS[0]]
        print(f"  did={did:<6} n_ctx={z['n_ctx']} n_cal={z['n_cal']} n_test={z['n_test']}")

    np.savez_compressed(ARRAYS / "chunk2_data.npz", **store)
    out = {"_rule": {"n_range": [N_MIN, N_MAX], "d_range": [D_MIN, D_MAX],
                     "min_target_unique": MIN_TARGET_UNIQUE, "n_target": N_TARGET,
                     "order": "OpenML id ascending", "seeds": SEEDS,
                     "n_ctx": N_CTX, "n_cal_max": N_CAL_MAX, "n_test_max": N_TEST_MAX,
                     "feature_scaling": "StandardScaler fitted on context rows only",
                     "target_scaling": "none"},
           "accepted": accepted, "rejected": rejected, "splits": split_meta}
    with open(RESULTS / "chunk2_datasets.json", "w") as f:
        json.dump(jsonable(out), f, indent=2)
    print(f"\nSaved: {RESULTS/'chunk2_datasets.json'}  and  {ARRAYS/'chunk2_data.npz'}")


if __name__ == "__main__":
    main()
