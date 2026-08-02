# Changes to vendored upstream code

`models/vendor/` holds upstream source that cannot be pip-installed here. It is
otherwise a clean checkout. Every deviation is listed below and marked in the
source with a `[VENDOR PATCH <id>]` comment.

Only **TabSwift** is vendored. TabPFN and TabICL are installed packages accessed
purely through hooks — see `models/README.md` for why TabICL in particular must
never be vendored.

---

## tabswift

**Upstream:** <https://github.com/LAMDA-Tabular/TabSwift>
**Commit:** `8edf8f0b4225bc03e1f5db011912619cd92b798d` (2026-07-27, "Merge pull request #2 from Kekssuchti/main")
**Path in that repo:** `TALENT/model/lib/tabswift`
**Checkpoint:** HF `LAMDA-Tabular/TabSwift` :: `swift.ckpt` — **public**, one file, both tasks.

The vendored tree is byte-identical to upstream apart from the five patches
below and an added `LICENSE` (copied from the upstream repository root, which
does not ship a licence file inside the library directory).

Re-verify identity at any time:

```bash
git clone https://github.com/LAMDA-Tabular/TabSwift /tmp/ts
cd /tmp/ts && git checkout 8edf8f0b4225bc03e1f5db011912619cd92b798d
diff -r --exclude=LICENSE --exclude=__pycache__ \
  /tmp/ts/TALENT/model/lib/tabswift <repo>/models/vendor/tabswift
```

### Patch table

| id | file | what | changes numerics? |
|---|---|---|---|
| V1 | `preprocessing.py` | sklearn ≥1.6 `validate_data` shim, 10 call sites | **no** |
| V2 | `regressor.py`, `classifier.py` | same shim, 4 call sites | **no** |
| V3 | `regressor.py`, `classifier.py` | removed a stray `print(avg.shape)` | **no** |
| V4 | `preprocessing.py` | added the missing `return` in `FeatureShuffler.shuffle` | **YES — see below** |
| V5 | `model/tabswift.py` | comment block only | **no** |

---

### V1 — `preprocessing.py`: scikit-learn ≥ 1.6 compatibility (10 call sites)

**Upstream:** `preprocessing.py:167, 196, 254, 303, 358, 379, 438, 482, 717, 946`
(`X = self._validate_data(X)` and variants).

scikit-learn 1.6 removed `BaseEstimator._validate_data` in favour of the
module-level `validate_data(estimator, X, y, ...)`. Upstream still calls the
removed method, so on the scikit-learn installed here (1.7.2) every `fit` raises
`AttributeError: 'EnsembleGenerator_Reg' object has no attribute '_validate_data'`.

Added a `_validate_data_compat(estimator, X, y, **kw)` shim that prefers the new
API and falls back to the old one, and rewrote the 10 call sites to use it.

*Why here and not at runtime:* a previous version of this repo monkeypatched
`sklearn.base.BaseEstimator._validate_data` globally at import time, which
changed input validation for **every** estimator in the process, including the
other two foundation models and any downstream scikit-learn code. Patching the
vendored source keeps the blast radius to TabSwift.

**Numerics: unchanged.** `validate_data()` is the same routine under a new name.

---

### V2 — `regressor.py:309-318`, `classifier.py:321-330` and the two `predict` sites

Same root cause as V1, in the two estimator classes. Upstream:

```python
if OLD_SKLEARN:
    # Workaround for compatibility with scikit-learn prior to v1.6
    X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
else:
    X, y = self._validate_data(X, y, dtype=None)
```

Rewritten to call `_validate_data_compat(self, ...)`.

> **Note on the previous vendoring.** The earlier copy replaced these calls with
> `check_X_y` / `check_array` instead. That is *not* equivalent:
> `validate_data()` also records `n_features_in_` and `feature_names_in_` on the
> estimator, which upstream relies on for the `reset=False` feature-count
> consistency check in `predict`. Under `check_X_y` that check silently did
> nothing, so a predict-time feature-count mismatch would have gone undetected.
> Restored to upstream semantics here.
> Covered by `models/tests/test_tabswift_upstream.py` (asserts `n_features_in_`
> is set after `fit`).

**Numerics: unchanged** for well-formed input; strictly more validation.

---

### V3 — `regressor.py:485`, `classifier.py:491`: removed a debug `print`

Upstream ships a bare `print(avg.shape)` in the `predict_proba` path, which
writes a line to stdout on every prediction and floods the log during hooked
experiment loops. Replaced with a comment.

**Numerics: unchanged** — the print had no effect on the return value. This is a
convenience patch, not a necessity.

---

### V4 — `preprocessing.py:553-572`: missing `return` in `FeatureShuffler.shuffle`

> **⚠ THIS PATCH CHANGES NUMERICS. It is the only one that does.**

**Upstream (bug):**

```python
# No shuffling
if self.method == "none" or n_estimators == 1:
    shuffle_patterns = [feature_indices]

# Generate permutations based on method
if self.method == "shift":
    ...
elif self.method == "latin":
    shuffle_patterns = self._latin_squares()
```

The `if` that handles `n_estimators == 1` assigns the identity pattern and then
falls through into the second `if`/`elif` chain, which overwrites it. With the
default `feat_shuffle_method="latin"`, a run that asked for a **single** ensemble
member still had its feature columns permuted.

TabICL's equivalent shuffler does have the `return`
(`site-packages/tabicl/_sklearn/preprocessing.py:816-819`), which is why the two
libraries disagreed.

**Fix:** added `return shuffle_patterns` inside that branch.

**Measured effect** (40 context rows, 8 query rows, 5 features, `random_state=42`,
fp32, TabSwift regression):

```
BEFORE (upstream bug) feature pattern: {'none': [[4, 3, 0, 1, 2]]}
AFTER  (patch V4)     feature pattern: {'none': [[0, 1, 2, 3, 4]]}
before preds: [2.9043 4.4609 6.9688 2.6973 2.2715 3.9297 3.9277 6.5   ]
after  preds: [2.9219 3.4707 6.793  2.7852 2.3418 4.1914 3.1348 5.5   ]
max |delta| = 1.000000   mean |delta| = 0.424561   y sd = 2.2128
delta as fraction of y sd: 45.191%
```

**Why we take the change rather than living with upstream.** The permutation is
deterministic given `random_state`, so upstream behaviour was reproducible — but
it means the model does not see the caller's columns in the order they were
passed. Any per-feature attribution, feature-ablation or column-alignment result
computed on TabSwift would be misattributed to the wrong column. Neither ordering
is "more correct" for accuracy purposes; identity is the only one that makes
feature indices mean what the caller thinks they mean.

Covered by `models/tests/test_tabswift_upstream.py::test_feature_shuffler_identity_at_one_estimator`
and `::test_resolved_feature_order_is_identity_for_all_models`.

---

### V5 — `model/tabswift.py`: comment block above `class TabSwift`

Comment only; no code changed. Records two things that a future reader will
otherwise "fix" incorrectly:

1. `_train_forward` and `_inference_forward` are unreachable, and
   `_inference_forward` is broken upstream (pads to a hardcoded 500 instead of
   `self.max_dim`, and reads an undefined local `d` → `NameError`).
2. TabSwift has **no column embedder and no row interactor** and this is by
   design (paper §3.1, Fig. 2, Fig. 3a, Eq. 5). The upstream class docstring
   describing three stages is inherited from TabICL's framework and is wrong for
   this model.

**Numerics: unchanged.**

---

## Corrections to earlier documentation

The previous vendoring carried an **undocumented** edit that changed which
checkpoint the library downloads:

```python
-        repo_id = "LAMDA-Tabular/TabSwift"      # upstream
-        filename = "swift.ckpt"
+        repo_id = "pretrain-models/tabswift"    # previous vendored copy
+        filename = "tabswift-regressor.ckpt"    # (and -classifier.ckpt)
```

`pretrain-models/tabswift` **does not exist** (the Hub returns
`RepositoryNotFoundError: 401`). The 401 was then recorded in `registry.py` and
`README.md` as evidence that "upstream is gated" and that the committed
`swift.ckpt` was "the only available copy". That provenance story was wrong in
every part:

* upstream is `LAMDA-Tabular/TabSwift` and is **public** (Apache-2.0);
* it ships **one** checkpoint, `swift.ckpt`, not two;
* the committed file is **byte-identical** to it
  (`sha256 16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363`,
  32,947,079 bytes), so nothing was ever lost;
* the Hub `config.json` states `"task": ["classification", "regression"]`,
  confirming one backbone serving both heads.

This edit has been reverted; the vendored copy now points at upstream, and
auto-download works again.

## Re-vendoring

To refresh from a newer upstream, replace the directory with a clean checkout,
re-apply V1–V5 above, then run:

```bash
python -m pytest models/tests/
python -m models.interp.verify --models tabswift
python -m models.interp.verify --models tabswift --task classification
```

All must pass. The battery checks the block count, the 64-token register offset,
and the layout against the loaded checkpoint, so an upstream architecture change
fails loudly instead of silently shifting every row index.
