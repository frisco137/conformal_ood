"""Single source of truth for which models exist, what they are, and what they can do.

Every architectural number here was read off a loaded checkpoint, not copied
from a paper or a plan. ``models.interp.verify`` re-checks them at runtime, so
a silent upstream change fails a test instead of corrupting an experiment.

Nothing else in the repository should hardcode a layer count, a width, a
checkpoint filename, or a capability flag -- import it from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
VENDOR_DIR = MODELS_DIR / "vendor"


@dataclass(frozen=True)
class ModelSpec:
    """What a model is and what it is allowed to be asked."""

    model_id: str
    display_name: str
    #: how the weights are obtained
    source: str
    #: exact checkpoint identifier (HF filename, or path relative to models/)
    checkpoint: str
    #: where the modelling code lives
    code: str

    # -- architecture (verified) --------------------------------------------
    n_icl_blocks: int
    d_model: int
    #: non-data tokens prepended inside the ICL stack
    n_prefix_tokens: int
    #: True when the residual stream is per-cell rather than per-row
    per_cell_state: bool

    # -- capabilities --------------------------------------------------------
    supports_regression: bool = True
    supports_classification: bool = True
    #: True when the regression head emits a distribution (quantiles/bins),
    #: False when it emits a point estimate. Gate every uncertainty experiment
    #: on this instead of synthesising a variance.
    has_predictive_distribution: bool = True

    notes: str = ""
    known_issues: tuple[str, ...] = field(default_factory=tuple)


SPECS: dict[str, ModelSpec] = {
    "tabpfn_v2": ModelSpec(
        model_id="tabpfn_v2",
        display_name="TabPFN v2",
        source="pip package `tabpfn` (>=7.1.0), ModelVersion.V2",
        checkpoint="Prior-Labs/TabPFN-v2-{reg,clf} :: tabpfn-v2-{regressor,classifier}.ckpt",
        code="site-packages/tabpfn (unmodified; hooks only)",
        n_icl_blocks=12,
        d_model=192,
        n_prefix_tokens=0,
        per_cell_state=True,
        has_predictive_distribution=True,
        notes=(
            "PerFeatureTransformer: state is (items, feature_groups, d_model). The "
            "decoder reads only the last feature slot, which is the canonical row "
            "stream. Item attention runs twice per block (query->support, then "
            "support->support). 12 blocks, not 24 -- 24 is TabPFN 2.5."
        ),
    ),
    "tabpfn_v2_5": ModelSpec(
        model_id="tabpfn_v2_5",
        display_name="TabPFN v2.5",
        source="pip package `tabpfn` (>=7.1.0), ModelVersion.V2_5",
        checkpoint="tabpfn-v2.5-regressor-v2.5_default.ckpt (local HF cache)",
        code="site-packages/tabpfn (unmodified; hooks only)",
        n_icl_blocks=18,
        d_model=192,
        n_prefix_tokens=64,
        per_cell_state=True,
        supports_classification=False,
        has_predictive_distribution=True,
        notes=(
            "Present only to provide the within-family scale bar for cross-model "
            "alignment (how different do two checkpoints of the SAME family look?). "
            "NOT a drop-in for v2: it has 18 blocks (not 12, and not the 24 the "
            "project plan assumes) and prepends 64 'thinking tokens' to every "
            "sequence, so row indices are offset by 64 relative to v2."
        ),
        known_issues=(
            "The classifier checkpoint repo (Prior-Labs/TabPFN-v2.5-clf) returns HTTP "
            "401: gated. Only the regressor is usable, and only because it is already "
            "in the local HF cache.",
        ),
    ),
    "tabicl_v2": ModelSpec(
        model_id="tabicl_v2",
        display_name="TabICL v2",
        source="pip package `tabicl` (==2.1.1); weights from HF `jingang/TabICL`",
        checkpoint="tabicl-{regressor,classifier}-v2-20260212.ckpt",
        code="site-packages/tabicl (unmodified; hooks only)",
        n_icl_blocks=12,
        d_model=512,
        n_prefix_tokens=0,
        per_cell_state=False,
        has_predictive_distribution=True,
        notes=(
            "Staged: column embedder -> row interaction (CLS + RoPE) -> 12-block ICL "
            "stack of width 512. Only the ICL stack is in the layer index. The plan's "
            "'8 blocks, width 256' describes TabICL v1, not this checkpoint. "
            "Regression head emits 999 quantiles."
        ),
        known_issues=(
            "Must NOT be vendored: the checkpoint unpickles classes under the "
            "absolute module path `tabicl._model.*`, so a vendored copy is silently "
            "shadowed by the installed package and any hook placed on it never fires.",
        ),
    ),
    "tabswift": ModelSpec(
        model_id="tabswift",
        display_name="TabSwift",
        source=(
            "vendored `models/vendor/tabswift` @ upstream "
            "github.com/LAMDA-Tabular/TabSwift 8edf8f0b; weights committed in-repo "
            "and byte-identical to HF `LAMDA-Tabular/TabSwift` :: swift.ckpt"
        ),
        checkpoint="checkpoints/tabswift/swift.ckpt",
        code="models/vendor/tabswift (5 documented patches, see vendor/PATCHES.md)",
        n_icl_blocks=24,
        d_model=192,
        n_prefix_tokens=64,
        per_cell_state=False,
        has_predictive_distribution=False,
        notes=(
            "24 ICL blocks preceded by 64 register tokens: the sequence is "
            "[64 registers][support][query]. Regression head is Linear(384, 1): a "
            "POINT ESTIMATE, so there is no predictive variance to measure."
        ),
        known_issues=(
            "Regression gives no predictive distribution; uncertainty questions "
            "(plan Q4/Q7) are not answerable for this model and must be reported as "
            "N/A, never synthesised.",
            "ONE checkpoint serves both tasks (HF config.json declares "
            "task=[classification, regression]); reg/clf differ only in the head "
            "(`reg_decoder`/`y_encoder_reg` vs `decoder`/`y_encoder`). This is the "
            "only model in the roster where a reg<->clf transplant is a "
            "same-weights experiment.",
            "The public checkpoint contains NO early-exit / per-layer prediction "
            "heads, despite paper s3.2 and App. B.2 describing them: all 353 "
            "state-dict entries are accounted for by x_linear + 24 blocks + ln + "
            "the two y encoders + the two decoders + register_token_values.",
        ),
    ),
}

#: Models the project's headline comparison runs on.
ROSTER: tuple[str, ...] = ("tabpfn_v2", "tabicl_v2", "tabswift")

#: Additional checkpoints used only for the within-family alignment scale bar.
SCALE_BAR: tuple[str, ...] = ("tabpfn_v2_5",)

#: Requested by the plan but not obtainable here; recorded so their absence is a
#: documented fact rather than a silent omission.
UNAVAILABLE: dict[str, str] = {
    "tabpfn_v1": (
        "No checkpoint on disk and none reachable: the v1 interface downloads "
        "`models_diff/prior_diff_real_checkpoint*`, which is not cached, and the v1 "
        "codebase predates the sklearn API this environment provides."
    ),
    "tabpfn_v2_5_classifier": "HF repo Prior-Labs/TabPFN-v2.5-clf is gated (HTTP 401).",
    "tabpfn_v2_6_classifier": "HF repo Prior-Labs/TabPFN-v2.6-clf is gated (HTTP 401).",
    "mitra": "Not vendored; no checkpoint available in this environment.",
    "nanotabpfn": "Not vendored; would need training from scratch to be meaningful.",
}


def spec(model_id: str) -> ModelSpec:
    try:
        return SPECS[model_id]
    except KeyError:
        raise KeyError(
            f"Unknown model {model_id!r}. Known: {sorted(SPECS)}. "
            f"Deliberately unavailable: {sorted(UNAVAILABLE)}"
        ) from None


def checkpoint_path(model_id: str) -> Path:
    """Absolute path to a checkpoint that lives inside this repo."""
    s = spec(model_id)
    if not s.checkpoint.startswith("checkpoints/"):
        raise ValueError(
            f"{model_id} does not ship an in-repo checkpoint; it is fetched by "
            f"{s.source}"
        )
    path = MODELS_DIR / s.checkpoint
    if not path.exists():
        raise FileNotFoundError(
            f"{model_id} checkpoint missing at {path}. It is tracked in git because "
            "upstream is gated; restore it with `git checkout -- models/checkpoints`."
        )
    return path


# ---------------------------------------------------------------------------
# Checkpoint manifest
# ---------------------------------------------------------------------------

MANIFEST_PATH = CHECKPOINT_DIR / "MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> dict:
    """Record every in-repo checkpoint with its size and SHA-256."""
    entries = {}
    for path in sorted(CHECKPOINT_DIR.rglob("*.ckpt")):
        entries[str(path.relative_to(CHECKPOINT_DIR))] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {"checkpoints": entries, "specs": {k: asdict(v) for k, v in SPECS.items()}}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_manifest() -> None:
    """Fail if an in-repo checkpoint no longer matches its recorded hash."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{MANIFEST_PATH} missing; run registry.write_manifest()")
    manifest = json.loads(MANIFEST_PATH.read_text())
    for relative, entry in manifest["checkpoints"].items():
        path = CHECKPOINT_DIR / relative
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {relative} listed in manifest but missing")
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise AssertionError(
                f"checkpoint {relative} hash mismatch: manifest {entry['sha256'][:12]}, "
                f"file {actual[:12]}"
            )


def summary_table() -> str:
    """A one-glance view of the roster, for READMEs and run logs."""
    header = f"{'model':<14}{'blocks':>7}{'d_model':>9}{'prefix':>8}{'cell':>6}{'clf':>5}{'dist':>6}"
    rows = [header, "-" * len(header)]
    for model_id, s in SPECS.items():
        rows.append(
            f"{model_id:<14}{s.n_icl_blocks:>7}{s.d_model:>9}{s.n_prefix_tokens:>8}"
            f"{'yes' if s.per_cell_state else 'no':>6}"
            f"{'yes' if s.supports_classification else 'no':>5}"
            f"{'yes' if s.has_predictive_distribution else 'no':>6}"
        )
    return "\n".join(rows)
