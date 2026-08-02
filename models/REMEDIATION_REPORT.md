# `models/` remediation — phase reports

Companion to `models/ARCHITECTURE_REPORT.md`. One section per phase, appended in
order. Every claim carries file:line and a verbatim snippet; anything not
verifiable is marked UNKNOWN with the resolution step.

Environment: `.venv/bin/python` 3.10, torch 2.11.0+cu130 (CUDA available),
tabpfn 8.0.8, tabicl 2.1.1, scikit-learn 1.7.2, numpy 2.2.6.

---

# PHASE 0 — TabSwift: replace the vendored copy with upstream

**Status: complete. All acceptance criteria pass.**

Commits: `91e3d4b` (0.1), `467955c` (0.2/0.3), `cfed11d` (0.5/0.6).

## Headline

The previous vendoring carried an **undocumented edit that invented a false
provenance story**, and that story had been propagated into `registry.py`, the
README, and my own architecture report. Upstream is public, ships exactly one
checkpoint, and our committed file is byte-identical to it. Open Question #1 of
the architecture report is now closed.

## 0.1 — Upstream fetch and re-vendor

**Upstream:** `https://github.com/LAMDA-Tabular/TabSwift`
**Commit SHA:** `8edf8f0b4225bc03e1f5db011912619cd92b798d` (2026-07-27,
"Merge pull request #2 from Kekssuchti/main")
**Path in repo:** `TALENT/model/lib/tabswift`

The library directory was re-vendored as a clean checkout. Verification that
only the documented patches remain:

```
$ diff -r --exclude=LICENSE --exclude=__pycache__ /tmp/ts/TALENT/model/lib/tabswift models/vendor/tabswift
IDENTICAL   (before patches were re-applied)
```

### Behavioural differences between the OLD vendored copy and upstream

`diff -rq` reported three differing files. `model/` — `tabswift.py`,
`learning.py`, `layers.py`, `attention.py`, `encoders.py`, `inference.py` — was
already byte-identical, so **no model math had been altered**. The three
wrapper files differed as follows.

**(a) Checkpoint source — undocumented, behaviour-changing.**
`regressor.py:173-174` / `classifier.py:183-184`, upstream:

```python
        repo_id = "LAMDA-Tabular/TabSwift"
        filename = "swift.ckpt"
```

old vendored copy:

```python
        repo_id = "pretrain-models/tabswift"
        filename = "tabswift-regressor.ckpt"     # and "-classifier.ckpt"
```

This was not in `PATCHES.md`. It changed *which* file the library downloads, and
because that repo does not exist, every auto-download attempt 401'd. The 401 was
then written into `registry.py` and `README.md` as proof that upstream was gated.
Reverted.

**(b) `_validate_data` — documented in spirit, but implemented differently from
`preprocessing.py` and behaviourally weaker.** Upstream `regressor.py:308-312`:

```python
        if OLD_SKLEARN:
            # Workaround for compatibility with scikit-learn prior to v1.6
            X, y = self._validate_data(X, y, dtype=None, cast_to_ndarray=False)
        else:
            X, y = self._validate_data(X, y, dtype=None)
```

The old copy replaced this with `check_X_y(X, y, dtype=None)`. That is **not**
equivalent: `validate_data()` also records `n_features_in_` / `feature_names_in_`
on the estimator, which upstream relies on for the `reset=False` feature-count
consistency check in `predict`. Under `check_X_y` that check silently did
nothing. Now uses the same `_validate_data_compat` shim as `preprocessing.py`
(patch V2), restoring upstream semantics.

Measured after the fix:

```
regression      n_features_in_ set on estimator: True -> 3
classification  n_features_in_ set on estimator: True -> 3
```

**(c) `print(avg.shape)` removal** — as documented, behaviour-neutral (patch V3).

**(d) `preprocessing.py`** — differed only by the documented sklearn shim.

### Final patch set

Five patches, all marked `[VENDOR PATCH <id>]` in source and tabulated in
`models/vendor/PATCHES.md`. Only **V4** changes numerics.

| id | file | what | numerics |
|---|---|---|---|
| V1 | `preprocessing.py` | sklearn ≥1.6 `validate_data` shim, 10 sites | no |
| V2 | `regressor.py`, `classifier.py` | same shim, 4 sites | no |
| V3 | `regressor.py`, `classifier.py` | drop stray `print` | no |
| V4 | `preprocessing.py` | missing `return` in `FeatureShuffler.shuffle` | **YES** |
| V5 | `model/tabswift.py` | comment block only | no |

## 0.2 — Checkpoints: one or two?

**Verdict: ONE.** Upstream ships a single `swift.ckpt` used by both the regressor
and the classifier wrapper — the same two lines quoted in 0.1(a) appear in both
files.

Hub metadata, read live:

```
LAMDA-Tabular/TabSwift: OK -> ['.gitattributes', 'README.md', 'config.json', 'swift.ckpt']
    swift.ckpt  size=32947079  sha=16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363
pretrain-models/tabswift: RepositoryNotFoundError: 401 ... Repository Not Found
```

`config.json` from the Hub:

```json
{
"model_type": "tabswift",
"architecture": "TabSwift",
"task": ["classification", "regression"],
"format_version": "1.0"
}
```

This is the model card confirming the paper's design (§3.1 prediction heads,
§4.2 unified targets): **one backbone, two heads, trained jointly**.

**Identity with our committed file:**

```
$ sha256sum models/checkpoints/tabswift/swift.ckpt
16e324177be2ab9e2bac15e5edf7867329e6595e134a5c4c9b6d79d3b657b363
$ stat -c%s models/checkpoints/tabswift/swift.ckpt
32947079
```

Identical hash and size to the Hub file. **`swift.ckpt` is the upstream
checkpoint**, not "neither" and not one of two variants.

**Strictness — reported explicitly, not papered over:**

```python
model = TabSwift(**ck["config"])
incompatible = model.load_state_dict(ck["state_dict"], strict=True)
assert incompatible.missing_keys == []
assert incompatible.unexpected_keys == []
```

Result: **strict=True succeeds, 0 missing, 0 unexpected, 8,203,211 parameters**,
353 state-dict entries. Covered by
`models/tests/test_tabswift_upstream.py::test_checkpoint_loads_strict_with_no_missing_or_unexpected_keys`.

Top-level keys: `['config', 'state_dict']`. Config:

```python
{'max_classes': 10, 'max_dim': 100, 'embed_dim': 192, 'proj_dim': 100,
 'col_num_blocks': 3, 'col_nhead': 4, 'col_num_inds': 128, 'row_num_blocks': 3,
 'row_nhead': 8, 'row_num_cls': 1, 'row_rope_base': 100000.0,
 'icl_num_blocks': 24, 'icl_nhead': 4, 'ff_factor': 2, 'dropout': 0.0,
 'activation': 'gelu', 'norm_first': True, 'use_headwise_gate': False,
 'use_elementwise_gate': True, 'register_tokens': 64}
```

Note `col_num_blocks`/`row_num_blocks`/`col_nhead` are present in the config but
**unused** — `TabSwift.__init__` never constructs those stages (§0.6).

## 0.3 — Early-exit heads: **NOT PRESENT**

**Verdict: the public checkpoint contains no per-layer prediction or exit heads.**

Every one of the 353 state-dict entries is accounted for by the documented module
tree; no key matches `exit|early|h_pred|h_exit|pred_head|aux|halt|ponder`:

```
=== distinct key patterns ===
  x_linear.weight                                      x1   (192, 100)
  x_linear.bias                                        x1   (192,)
  icl_predictor.tf_icl.blocks.{i}.linear1.weight       x24  (384, 192)
  ... 14 patterns x 24 blocks = 336 ...
  icl_predictor.ln.weight / .bias                      x2
  icl_predictor.y_encoder.weight / .bias               x2
  icl_predictor.y_encoder_reg.weight / .bias           x2
  icl_predictor.decoder.{i}.weight / .bias             x4
  icl_predictor.reg_decoder.{i}.weight / .bias         x4
  icl_predictor.tf_icl.register_token_values           x1   (64, 192)

=== search for exit / per-layer prediction head weights ===
  matches: NONE
```

2 + 336 + 2 + 2 + 2 + 4 + 4 + 1 = 353. ✓

The paper (§3.2, App. B.2) describes `h_pred^(e)` / `h_exit^(e)` attached to all
24 layers via a post-training stage, but **those weights were not released**.
The vendored code has no machinery to run them either — there is no reference to
per-layer heads anywhere in `model/learning.py`.

**Consequence for Phase 2.7:** `Trace.layer_pred_native` is **not implementable**.
The patch-based logit lens remains the only per-layer readout, for all four
models. This is a downgrade to the plan and it is not recoverable from our side;
resolution would require the authors to release the post-trained checkpoint.

Covered by `::test_checkpoint_has_no_early_exit_heads`, which fails loudly if a
future checkpoint adds them.

## 0.4 — Target standardization: **upstream omits it too**

**Verdict: case (a) — do NOT patch the vendor; enforce at the harness in 1.4.**

Upstream `regressor.py:314-318`, verbatim:

```python
        # check_classification_targets(y)
        # self.scaler = StandardScaler()
        # # self.scaler.fit(y)
        # y = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        # print(y)
```

The commenting-out is upstream's, not ours. The inverse transform in `predict` is
likewise commented out upstream. So **TabSwift genuinely ingests raw y and emits
raw y**, while the paper (§4.2) says the regression target is standardized during
pretraining. That is a real train/inference mismatch in the released wrapper, and
it is exactly the scale confound flagged in the architecture report.

No vendor patch applied. This is now a hard requirement for Phase 1.4:
harness-level y standardization fitted on the context, on by default for
regression, recorded in `run_meta`.

## 0.5 — Feature shuffle at `n_estimators=1`: **bug is upstream**

**Verdict: present upstream; patched as V4 with an assertion added.**

Upstream `preprocessing.py:538-546`, verbatim — note the missing `return`:

```python
        # No shuffling
        if self.method == "none" or n_estimators == 1:
            shuffle_patterns = [feature_indices]

        # Generate permutations based on method
        if self.method == "shift":
```

The identity assignment falls through into the second `if`/`elif` chain and is
overwritten by `elif self.method == "latin": shuffle_patterns = self._latin_squares()`.
TabICL's equivalent does return
(`site-packages/tabicl/_sklearn/preprocessing.py:816-819`), which is why the two
libraries disagreed.

**Measured before/after** (40 context rows, 8 query rows, 5 features,
`random_state=42`, fp32, regression):

```
BEFORE (upstream bug) feature pattern: {'none': [[4, 3, 0, 1, 2]]}
AFTER  (patch V4)     feature pattern: {'none': [[0, 1, 2, 3, 4]]}
before preds: [2.9043 4.4609 6.9688 2.6973 2.2715 3.9297 3.9277 6.5   ]
after  preds: [2.9219 3.4707 6.793  2.7852 2.3418 4.1914 3.1348 5.5   ]
max |delta| = 1.000000   mean |delta| = 0.424561   y sd = 2.2128
delta as fraction of y sd: 45.191%
```

**A 45%-of-σ shift.** Neither ordering is more accurate a priori, but only the
identity makes feature indices mean what the caller thinks they mean, which is a
precondition for every feature-level analysis we plan.

**Assertion added for all three models** (0.5's explicit requirement):
`::test_resolved_feature_order_is_identity_for_all_models` checks the *resolved*
pattern from the fitted `ensemble_generator_`, both tasks:

```
tabpfn_v2   -> no EnsembleGenerator; asserts estimator.n_estimators_ == 1
tabicl_v2   -> feature_shuffles_    == identity
tabswift    -> feature_shuffle_patterns_ == identity
```

## 0.6 — Dead code: **upstream ships it; left in place, marked and tested**

Upstream `model/tabswift.py` is byte-identical to our copy, and it contains both
dead methods. Confirmed present upstream:

```
131:    def _train_forward(
172:        if d is not None and len(d.unique()) == 1 and d[0] == H:
186:    def _inference_forward(
255:        pad_len = 500 - H
268:        if d is not None and len(d.unique()) == 1 and d[0] == H:
290:    def forward(
```

Per the rules, not deleted. Added patch V5, a comment block above
`class TabSwift`, recording that they are unreachable, that `_inference_forward`
pads to a hardcoded 500 instead of `self.max_dim`, and that it reads an undefined
local `d`.

Two tests, one static and one dynamic:
* `::test_forward_does_not_reach_dead_paths` — AST-walks `forward` to prove
  neither is called, then installs a runtime tripwire on both and runs a real
  `fit`+`predict`, asserting neither fires.
* `::test_dead_inference_forward_is_still_broken_upstream` — asserts
  `pad_len = 500 - H` is present and that calling `_inference_forward` raises
  `NameError` on `d`.

## Acceptance — Phase 0

| criterion | status | evidence |
|---|---|---|
| clean upstream checkout at a recorded SHA plus a short documented patch list | **PASS** | `8edf8f0b`, 5 patches in `PATCHES.md`, `diff -r` clean before patching |
| every checkpoint loads with `strict=True` | **PASS** | 0 missing, 0 unexpected, 8,203,211 params |
| verdict: one checkpoint or two | **PASS** | **one**, `swift.ckpt`, `task: [classification, regression]` |
| verdict: exit heads present | **PASS** | **absent**; 353/353 keys accounted for |
| verdict: y-scaler upstream | **PASS** | **absent upstream**; harness must enforce (1.4) |
| verdict: shuffle bug upstream | **PASS** | **present upstream**; patched V4 + cross-model assertion |
| `verify --models tabswift` both tasks | **PASS** | 17/17 and 17/17 |
| test asserts exact module tree, NO col_embedder / row_interactor, with paper citation | **PASS** | `::test_no_column_or_row_stage`, both tasks |

Full test run:

```
$ python -m pytest models/tests/ -q
18 passed in 3.92s

$ python -m models.interp.verify
tabpfn_v2      17/17  ALL PASS
tabicl_v2      17/17  ALL PASS
tabswift       17/17  ALL PASS

$ python -m models.interp.verify --models tabswift --task classification
tabswift       17/17  ALL PASS
```

## Corrections to the architecture report

Phase 0 falsifies three statements in `models/ARCHITECTURE_REPORT.md`:

1. **§A.1 / §A.2 / Open Question #1** — "upstream HF repo returns HTTP 401
   (gated)" and "which of the two upstream files `swift.ckpt` actually is:
   UNKNOWN". Both wrong. Upstream is public, ships one file, and our copy is it.
   Open Question #1 is **closed**.
2. **§A.1** — "2 documented patches". There were three patch groups, one
   undocumented and behaviour-changing (the repo id) and one weaker than
   documented (`check_X_y`). Now five, all documented.
3. **§A.2 caveat** — "this repo runs the *classifier* checkpoint (or whichever
   one `swift.ckpt` is) for both tasks". There is only ever one checkpoint;
   there is no ambiguity to resolve.

The architecture report's substantive architecture claims (no column/row stage,
24 blocks, 64 registers, one backbone two heads, no predictive distribution for
regression) are all **confirmed** by the upstream source and the paper.

## UNKNOWNS after Phase 0

1. **Whether the released `swift.ckpt` is the pre- or post-trained checkpoint.**
   The paper's early-exit stage (§3.2) produces additional heads; those are
   absent here. Whether the *backbone* weights are also pre-post-training, or
   post-trained with the heads simply stripped, is not determinable from the
   file — there is no training metadata, only `config` and `state_dict`.
   **Resolution:** ask the authors, or compare against a future release. Impact:
   low for our purposes; it only matters if we want to reproduce the paper's
   early-exit accuracy numbers, which we cannot do anyway without the heads.
2. **Why the config carries `col_num_blocks`, `row_num_blocks`, `col_nhead`,
   `col_num_inds`, `row_rope_base`.** These are read into attributes
   (`model/tabswift.py:101-107`) and never used. Most likely inherited from the
   TabICL-derived framework. **Resolution:** none needed; they are inert. Flagged
   only so nobody treats them as evidence that the stages exist.
