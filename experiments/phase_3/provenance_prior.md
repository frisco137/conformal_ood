# T0.1 — Prior and checkpoint provenance

**Deliverable for plan §2, T0.1.** Answered from installed source and the checkpoint on disk, not
from memory. Every claim carries a file path and line number into
`.venv/lib/python3.10/site-packages/tabicl/` (package version **2.1.1**).

**Date:** 2026-09-02 · **Verdict: PARTIAL** → see §5.

---

## 1. The audited object, pinned

| field | value |
|---|---|
| package | `tabicl` **2.1.1** (`importlib.metadata.version`) |
| HF repo | `jingang/TabICL` (`_sklearn/regressor.py:298`) |
| checkpoint file | `tabicl-regressor-v2-20260212.ckpt` |
| local path | `~/.cache/huggingface/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/` |
| **sha256** | `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a` |
| size | 114 324 594 bytes |

**Naming.** Plan §1 asks whether "TabICL v2" is what the release calls it. **It is** — the checkpoint
filename is `tabicl-regressor-v2-20260212.ckpt` and the classifier counterpart is
`tabicl-classifier-v2-20260212.ckpt`, against a `v1` dated `20250208`. **No rename is needed**;
Phase 1–2's "TabICL v2" is correct and stays.

### Regression-head configuration, read from the checkpoint's own `config` block

```
max_classes      0        <- the regression switch
num_quantiles    999
embed_dim        128
icl_num_blocks   12       icl_nhead 8
col_num_blocks   3        col_nhead 8   col_num_inds 128   col_target_aware true
row_num_blocks   3        row_nhead 8   row_num_cls 4      row_rope_base 100000
ff_factor 2   dropout 0.0   activation gelu   norm_first true   bias_free_ln true
```

**`max_classes: 0` is the regression setting.** `_model/learning.py:324` and `:455` branch on
`if self.max_classes == 0:` into the regression path, and `:259` documents
*"For regression (max_classes=0): out_dim = num_quantiles"*.

**Note on the quantile grid.** The head emits **999** quantiles. Phase 1–2 requested 9999 `alphas`
via `predict(output_type="quantiles", alphas=...)`; that is a *request* grid interpolated from the
head, not the head width. `_sklearn/regressor.py:480-503` shows `alphas` is a caller-supplied list.
This does not invalidate the Phase 1 variance channel — its extractor was validated against the exact
GP's closed form at 0.31% — but the paper should say "999-quantile head, integrated on a
9999-point grid" rather than implying a 9999-wide head.

---

## 2. Does the released prior sampler emit continuous targets?

**Yes — the components do. The public API does not.**

The prior generates a *regression* target and then optionally discretises it:

- `prior/_mlp_scm.py` and `prior/_tree_scm.py` emit **continuous** `y` from the SCM. Measured:
  100 unique values out of 100 rows, both `mlp_scm` and `tree_scm`.
- `prior/_reg2cls.py:281-296` — `Reg2Cls.__init__` sets `self.class_assigner = None` **iff
  `num_classes == 0`**.
- `prior/_reg2cls.py:322-327` — with no class assigner, `y` is only standard-scaled and returned
  continuous.

Verified directly (both SCM types, `num_classes=0`, `n=100`, `d=5`):

```
mlp_scm    X (100, 10)   y unique = 100   mean -0.0000  std 1.0000   CONTINUOUS
tree_scm   X (100, 10)   y unique = 100   mean +0.0000  std 1.0000   CONTINUOUS
```

*(`Reg2Cls.forward` calls `.numpy()` on a grad-tracking tensor at `_reg2cls.py:39`; it must be fed
detached tensors or run under `torch.no_grad()`. Not a defect for our use.)*

**But the shipped public entry point cannot reach that path.** `PriorDataset` samples the class
count per dataset at `prior/_dataset.py:650-654`:

```python
if np.random.random() > 0.5:
    ds_num_classes = np.random.randint(2, self.max_classes + 1)
else:
    ds_num_classes = 2
```

This is spread into `params` **after** `**self.fixed_hp` (`_dataset.py:658-668`), so it overrides any
`num_classes` a caller supplies. And constructing `PriorDataset(max_classes=0)` — the regression
setting — **crashes**:

```
ValueError: low >= high        # np.random.randint(2, 0+1) at _dataset.py:652
```

**There is no `max_classes == 0` branch anywhere in `prior/_dataset.py`.** Confirmed by enumerating
every occurrence of `max_classes` in that file (lines 64, 94, 108, 417, 472, 491, 652, 741, 770, 782,
829, 860, 920, 941, 956, 981, 1063): all are pass-through or the `randint` above.

---

## 3. Was the audited regression checkpoint trained on *that* prior?

**Cannot be verified. The evidence points both ways and the decisive branch is missing from the
release.**

**For:** the training driver passes **one** `max_classes` to both the model and the prior:

```
train/_run.py:172   "max_classes": self.config.max_classes,     -> model config
train/_run.py:236   max_classes=self.config.max_classes,        -> PriorDataset(...)
```

The checkpoint records `max_classes: 0`. So the run that produced it had `config.max_classes = 0`,
and that same `0` was handed to `PriorDataset`.

**Against:** `PriorDataset(max_classes=0)` **raises** in the released code. The code path that trained
this checkpoint therefore **is not the code path that shipped**. Either the regression checkpoint was
trained against an unreleased prior branch, or against a `_dataset.py` carrying a `max_classes == 0`
case that was removed or never published.

**No training-run metadata is in the checkpoint.** It contains exactly two top-level keys,
`state_dict` and `config`, and `config` is the *model* architecture only — no prior version, no
sampler hyperparameters, no data provenance, no commit.

---

## 4. Which knobs are exposed

From `PriorDataset.__init__` (`prior/_dataset.py:913-934`) and
`prior/_prior_config.py`:

| knob | exposed | where |
|---|---|---|
| DAG structure — `is_causal`, `num_causes`, `y_is_effect`, `in_clique`, `num_layers` | **yes**, as sampling distributions | `_prior_config.py` `DEFAULT_SAMPLED_HP` |
| mechanism family — `prior_type` ∈ {`mlp_scm`, `tree_scm`, `mix_scm`} | **yes** | `_dataset.py:928` |
| tree mechanism — `tree_model`, `tree_depth_lambda`, `tree_n_estimators_lambda` | **yes** | `DEFAULT_FIXED_HP` |
| MLP mechanism — `mlp_activations`, `hidden_dim`, `init_std`, `block_wise_dropout`, `mlp_dropout_prob` | **yes** | `DEFAULT_SAMPLED_HP` |
| feature distributions — `sampling` ∈ {`normal`, `mixed`, `uniform`}, `pre_sample_cause_stats` | **yes** | `DEFAULT_SAMPLED_HP` |
| `n` (`seq_len`), `d` (`num_features`) | **yes** | `min/max_seq_len`, `min/max_features` |
| categorical fraction — `cat_prob`, `max_categories` | **yes** | `DEFAULT_FIXED_HP` |
| **noise magnitude** — `noise_std`, `pre_sample_noise_std` | **yes** | `DEFAULT_SAMPLED_HP`, log-scaled trunc-norm, mean in `[1e-4, 0.3]` |
| **noise *mechanism*** | **NO — and this is the one that matters** | see below |
| **continuous targets** | **not via `PriorDataset`**; reachable only by calling `MLPSCM`/`TreeSCM` + `Reg2Cls(num_classes=0)` directly | §2 |

### The noise mechanism is not terminal, and is not controllable

`_mlp_scm.py:201-214` — noise is injected **inside every SCM layer**:

```python
noise_layer = GaussianNoise(noise_std)
return nn.Sequential(activation, linear_layer, noise_layer)
```

`_tree_scm.py:180-188` documents the same design. So the SCM's output `y` carries noise that has been
pushed through subsequent nonlinear layers. **It is not `f + ε` with a known `σ²`**, and there is no
knob that makes it so.

**Consequence for T1.1, and it is the design point the plan already anticipated.** Assumption 1
requires a Gaussian channel `y = f + ε`, `ε ~ N(0, σ²I)`. The construction is therefore:

- take the SCM's continuous output as the **latent `f`** — its distribution is the pushforward of the
  released prior onto the sampled locations, which is *some* prior on `ℝⁿ`, which is all Theorem 5
  requires;
- **add** terminal `ε ~ N(0, σ²I)` with `σ²` recorded per context.

That is **rung A**. Leaving the SCM output as `y` directly, with only its internal layer noise, is
**rung B**, where Assumption 1 is violated and A2/A3 are uninterpretable rather than failed —
exactly as the plan's ladder specifies.

---

## 5. Verdict — **PARTIAL**

Against the plan's branch conditions (§2, T0.1):

| condition | status |
|---|---|
| continuous targets | **yes**, from released components (`MLPSCM`/`TreeSCM` + `Reg2Cls(num_classes=0)`) |
| prior provably matches the checkpoint | **NO.** `PriorDataset(max_classes=0)` raises; no training metadata in the checkpoint |
| noise mechanism controllable | **NO** natively — it is per-layer, not terminal. Made controllable by *adding* terminal Gaussian noise, which is a construction of ours, not a knob of theirs |

This is neither PASS nor FAIL. It is the plan's **PARTIAL** branch:

> *PARTIAL (prior available but not provably the training prior) → run Tier 1 anyway, and rename the
> rung everywhere from in-family to **prior-family**.*

### Binding consequences for every downstream claim

1. **The word "in-family" is not used anywhere in Phase 3.** Rung A is **prior-family**.
2. **Every claim about rung A carries this sentence:** *"contexts drawn from the released TabICL
   prior, which we could not verify is the exact pre-training prior for this checkpoint."*
3. **A second, sharper caveat is required and the plan did not anticipate it.** The released
   `PriorDataset` cannot produce the regression setting at all, so rung A is built from the prior's
   *components* rather than from its public sampler. The generator reproduces the documented
   composition — SCM → `Reg2Cls` — but that composition is our reconstruction. State it as such.
4. **Rung A's terminal Gaussian noise is ours, not theirs.** The prior's own noise is internal to the
   SCM layers. Rung B exists precisely to show what the native noise mechanism does.

**T1.7 (nano-PFN) is not triggered.** That branch fires on FAIL. It remains a stretch item under its
own time box, and PARTIAL raises its value: a model whose prior is known *exactly* by construction is
the only way to close the gap this document opens.
