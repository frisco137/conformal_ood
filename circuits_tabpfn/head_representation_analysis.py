"""
head_representation_analysis.py — Dimensionality-reduction analysis of
per-head attention vectors for the most ablation-critical TabPFN heads.

Target heads (0-indexed, ranked by ablation delta-CE from experiment1_results.json):
  L7H1, L11H2, L7H3, L8H2, L10H1, L7H2, L7H0

For each target head × {ctoc, qtoc}, three subexperiments are run:

  subexp1  basic           Fit PCA / t-SNE / UMAP on 500 ID + 500 OOD, plot same set.
  subexp2  fit_id_ood      Fit on 500 ID + 500 OOD, apply transform to 100 new ID + 100 OOD.
  subexp3  fit_id_only     Fit on 500 ID only,       apply transform to 100 new ID + 100 OOD.

  Note — t-SNE produces no reusable transform, so in subexp2/3 it is run on the
  COMBINED fit+eval set; fit points are shown as a faint background and eval points
  as the foreground.

Finally, a delta-CE-weighted mean of the 7 head vectors is computed (weight =
delta_CE_h / sum(delta_CE), where delta_CE comes from the zero-ablation results)
and the same 3 subexperiments are run on that fused vector.

Entry point for future work:
  The core subroutine `run_dim_reduction_suite` takes any (n_fit, D) and
  (n_eval, D) numpy arrays; swap in a different vector extraction method and
  call the same routine.

Outputs → results/head_representation/
  <name>_subexp1_basic.png
  <name>_subexp2_fit_id_ood.png
  <name>_subexp3_fit_id_only.png
  summary.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tabpfn_hooks import (
    load_tabpfn, make_inprior, make_outprior,
    ManualMHA, NHEAD, HEAD_DIM,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: target heads + ablation weights
# ──────────────────────────────────────────────────────────────────────────────

TARGET_HEADS = [
    (7, 0), (7, 1), (7, 2), (7, 3), (8, 2), (10, 1), (11, 2),
]

# delta_CE values from results/experiment1_results.json head_ablation array
DELTA_CE = {
    (7,  0): 0.006109746093750035,
    (7,  1): 1.698201337890625,
    (7,  2): 0.015964980468750056,
    (7,  3): 0.13749270507812494,
    (8,  2): 0.044397929687500115,
    (10, 1): 0.018926064453125058,
    (11, 2): 0.182504091796875,
}
_TOTAL_W = sum(DELTA_CE.values())

TARGET_LAYERS = sorted({l for l, _ in TARGET_HEADS})

RESULTS_DIR = os.path.join(_HERE, 'results', 'head_representation')

plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

_ID_COLOR  = '#3b82f6'   # blue
_OOD_COLOR = '#ef4444'   # red

# ──────────────────────────────────────────────────────────────────────────────
# ManualMHA management
# ──────────────────────────────────────────────────────────────────────────────

def install_recording_manuals(model):
    """Install ManualMHA in record mode on TARGET_LAYERS. Returns {layer_idx: ManualMHA}."""
    manuals = {}
    for l_idx in TARGET_LAYERS:
        layer = model.transformer_encoder.layers[l_idx]
        mm = ManualMHA(layer.self_attn)
        mm.record = True
        mm.install()
        manuals[l_idx] = mm
    return manuals


def uninstall_manuals(manuals):
    for mm in manuals.values():
        mm.uninstall()


# ──────────────────────────────────────────────────────────────────────────────
# Vector collection
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_head_vectors(model, make_data_fn, n_tables, device, single_eval_pos,
                         manuals, batch_size=32):
    """
    Run n_tables forward passes (in batches) and record per-head attention vectors.

    The vector for head h in layer l on call type t is the mean of the per-head
    pre-out_proj output across all sequence positions:
      ctx[t] shape from ManualMHA: (B, nhead, Lq, head_dim)
      → mean over Lq → (B, nhead, head_dim)
      → head h column → (B, head_dim)

    Args:
        make_data_fn: callable(n) → (x, y) with x (seq, n, feat), y (seq, n)

    Returns:
        dict: (layer, head, call_type) → np.ndarray (n_tables, HEAD_DIM)
        where call_type ∈ {'ctoc', 'qtoc'}
    """
    # heads per layer (for faster lookup)
    heads_by_layer = {}
    for l, h in TARGET_HEADS:
        heads_by_layer.setdefault(l, []).append(h)

    accum = {(l, h, t): []
             for l, h in TARGET_HEADS
             for t in ('ctoc', 'qtoc')}

    collected = 0
    with tqdm(total=n_tables, desc="  collecting", leave=False) as pbar:
        while collected < n_tables:
            n = min(batch_size, n_tables - collected)
            x, y = make_data_fn(n)
            x = x.to(device)
            y = y.float().to(device)

            for mm in manuals.values():
                mm.reset()

            model((x, y), single_eval_pos=single_eval_pos)

            for l_idx, mm in manuals.items():
                # recorded[0]=ctoc (B, nhead, Lq_ctx, hd)
                # recorded[1]=qtoc (B, nhead, Lq_qry, hd)
                for call_idx, call_type in enumerate(('ctoc', 'qtoc')):
                    ctx = mm.recorded[call_idx]          # (B, nhead, Lq, hd)
                    ctx_mean = ctx.mean(dim=2)            # (B, nhead, hd)
                    for h in heads_by_layer.get(l_idx, []):
                        accum[(l_idx, h, call_type)].append(
                            ctx_mean[:, h, :].cpu().numpy()  # (B, hd)
                        )

            collected += n
            pbar.update(n)

    return {k: np.vstack(v) for k, v in accum.items()}


def weighted_mean_vectors(vecs_dict, call_type):
    """
    Compute delta-CE-weighted mean across TARGET_HEADS for a given call_type.
    Returns np.ndarray (n_tables, HEAD_DIM).
    """
    result = None
    for l, h in TARGET_HEADS:
        v = vecs_dict[(l, h, call_type)] * (DELTA_CE[(l, h)] / _TOTAL_W)
        result = v if result is None else result + v
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core dim-reduction subroutine  (reusable entry point for future analyses)
# ──────────────────────────────────────────────────────────────────────────────

def run_dim_reduction_suite(fit_vecs, fit_labels, eval_vecs, eval_labels,
                            out_path, title, skip_tsne=False):
    """
    Fit PCA and UMAP on fit_vecs; transform eval_vecs and plot side-by-side.
    t-SNE is run on the combined fit+eval set (it cannot transform new data);
    fit points are shown as a faint background, eval points as the foreground.

    When fit_vecs IS eval_vecs (subexp1 — no train/test split), the "fit"
    points are omitted from the background and only the single dataset is shown.

    Args:
        fit_vecs:   (N_fit,  D) float32 array used to fit transforms.
        fit_labels: (N_fit,)    int array, 0 = ID, 1 = OOD.
        eval_vecs:  (N_eval, D) float32 array to project/evaluate.
        eval_labels:(N_eval,)   int array, 0 = ID, 1 = OOD.
        out_path:   path to save the figure.
        title:      suptitle string.
        skip_tsne:  if True, replace the t-SNE panel with a blank note (faster).
    """
    is_same = fit_vecs is eval_vecs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    pca.fit(fit_vecs)
    e_pca = pca.transform(eval_vecs)
    ax = axes[0]
    if not is_same:
        f_pca = pca.transform(fit_vecs)
        ax.scatter(f_pca[fit_labels == 0, 0], f_pca[fit_labels == 0, 1],
                   c=_ID_COLOR,  alpha=0.15, s=6,  marker='o', zorder=1)
        ax.scatter(f_pca[fit_labels == 1, 0], f_pca[fit_labels == 1, 1],
                   c=_OOD_COLOR, alpha=0.15, s=6,  marker='o', zorder=1)
    ax.scatter(e_pca[eval_labels == 0, 0], e_pca[eval_labels == 0, 1],
               c=_ID_COLOR,  alpha=0.75, s=18, marker='o',  label='ID', zorder=3)
    ax.scatter(e_pca[eval_labels == 1, 0], e_pca[eval_labels == 1, 1],
               c=_OOD_COLOR, alpha=0.75, s=18, marker='^',  label='OOD', zorder=3)
    var = pca.explained_variance_ratio_
    ax.set_title(f'PCA  (PC1={var[0]:.1%}, PC2={var[1]:.1%})')
    ax.set_xlabel('PC 1');  ax.set_ylabel('PC 2')
    ax.legend(loc='best', fontsize=8, markerscale=1.2)

    # ── t-SNE ────────────────────────────────────────────────────────────────
    ax = axes[1]
    if skip_tsne:
        ax.text(0.5, 0.5, 't-SNE skipped\n(--skip_tsne)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('t-SNE')
    else:
        if is_same:
            combined      = fit_vecs
            combined_lbl  = fit_labels
            is_eval_mask  = np.ones(len(fit_vecs), dtype=bool)
        else:
            combined      = np.vstack([fit_vecs, eval_vecs])
            combined_lbl  = np.concatenate([fit_labels, eval_labels])
            is_eval_mask  = np.array([False] * len(fit_vecs) + [True] * len(eval_vecs))

        perp = int(min(30, max(5, len(combined) // 20)))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                    n_iter=300, method='barnes_hut')
        xy = tsne.fit_transform(combined)

        if not is_same:
            bg = ~is_eval_mask
            ax.scatter(xy[bg & (combined_lbl == 0), 0], xy[bg & (combined_lbl == 0), 1],
                       c=_ID_COLOR,  alpha=0.12, s=6,  marker='o', zorder=1)
            ax.scatter(xy[bg & (combined_lbl == 1), 0], xy[bg & (combined_lbl == 1), 1],
                       c=_OOD_COLOR, alpha=0.12, s=6,  marker='o', zorder=1)
        fg = is_eval_mask
        ax.scatter(xy[fg & (combined_lbl == 0), 0], xy[fg & (combined_lbl == 0), 1],
                   c=_ID_COLOR,  alpha=0.75, s=18, marker='o',  label='ID', zorder=3)
        ax.scatter(xy[fg & (combined_lbl == 1), 0], xy[fg & (combined_lbl == 1), 1],
                   c=_OOD_COLOR, alpha=0.75, s=18, marker='^',  label='OOD', zorder=3)
        note = '' if is_same else '  (fit=bg, eval=fg)'
        ax.set_title(f't-SNE perp={perp}{note}')
        ax.set_xlabel('Component 1');  ax.set_ylabel('Component 2')
        ax.legend(loc='best', fontsize=8, markerscale=1.2)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
    reducer.fit(fit_vecs)
    e_umap = reducer.transform(eval_vecs)
    ax = axes[2]
    if not is_same:
        f_umap = reducer.transform(fit_vecs)
        ax.scatter(f_umap[fit_labels == 0, 0], f_umap[fit_labels == 0, 1],
                   c=_ID_COLOR,  alpha=0.15, s=6,  marker='o', zorder=1)
        ax.scatter(f_umap[fit_labels == 1, 0], f_umap[fit_labels == 1, 1],
                   c=_OOD_COLOR, alpha=0.15, s=6,  marker='o', zorder=1)
    ax.scatter(e_umap[eval_labels == 0, 0], e_umap[eval_labels == 0, 1],
               c=_ID_COLOR,  alpha=0.75, s=18, marker='o',  label='ID', zorder=3)
    ax.scatter(e_umap[eval_labels == 1, 0], e_umap[eval_labels == 1, 1],
               c=_OOD_COLOR, alpha=0.75, s=18, marker='^',  label='OOD', zorder=3)
    ax.set_title('UMAP')
    ax.set_xlabel('Component 1');  ax.set_ylabel('Component 2')
    ax.legend(loc='best', fontsize=8, markerscale=1.2)

    fig.suptitle(title, fontweight='bold', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Three-subexperiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_three_subexps(fit_id, fit_ood, eval_id, eval_ood,
                      results_dir, name_prefix, title_prefix,
                      skip_tsne=False):
    """
    Run all 3 subexperiments for a given set of per-sample vectors.

    Args:
        fit_id:   (n_fit,  D) ID   vectors for fitting transforms.
        fit_ood:  (n_fit,  D) OOD  vectors for fitting transforms.
        eval_id:  (n_eval, D) new ID  vectors to evaluate on.
        eval_ood: (n_eval, D) new OOD vectors to evaluate on.
        results_dir:   output directory.
        name_prefix:   filename prefix (e.g. "L7H1_ctoc").
        title_prefix:  human-readable prefix for plot suptitle.
        skip_tsne:     forward to run_dim_reduction_suite.
    """
    n_fit  = len(fit_id)
    n_eval = len(eval_id)

    fit_all    = np.vstack([fit_id,  fit_ood])
    fit_all_lbl = np.array([0] * n_fit + [1] * n_fit)
    eval_all    = np.vstack([eval_id, eval_ood])
    eval_all_lbl = np.array([0] * n_eval + [1] * n_eval)
    fit_id_lbl  = np.zeros(n_fit, dtype=int)

    # subexp1 — basic: fit==eval (no split), visualise the fit distribution itself
    run_dim_reduction_suite(
        fit_all, fit_all_lbl,
        fit_all, fit_all_lbl,
        os.path.join(results_dir, f"{name_prefix}_subexp1_basic.png"),
        f"{title_prefix}  |  subexp1: basic ({n_fit} ID + {n_fit} OOD, no split)",
        skip_tsne=skip_tsne,
    )

    # subexp2 — fit on ID+OOD, eval on new ID+OOD
    run_dim_reduction_suite(
        fit_all, fit_all_lbl,
        eval_all, eval_all_lbl,
        os.path.join(results_dir, f"{name_prefix}_subexp2_fit_id_ood.png"),
        f"{title_prefix}  |  subexp2: fit={n_fit}×(ID+OOD) → eval {n_eval} new ID+OOD",
        skip_tsne=skip_tsne,
    )

    # subexp3 — fit on ID only, eval on new ID+OOD
    run_dim_reduction_suite(
        fit_id, fit_id_lbl,
        eval_all, eval_all_lbl,
        os.path.join(results_dir, f"{name_prefix}_subexp3_fit_id_only.png"),
        f"{title_prefix}  |  subexp3: fit={n_fit} ID only → eval {n_eval} new ID+OOD",
        skip_tsne=skip_tsne,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Dim-reduction analysis of per-head attention vectors for critical TabPFN heads."
    )
    ap.add_argument("--n_fit",          type=int,   default=500,
                    help="Tables per class for fitting transforms (default 500).")
    ap.add_argument("--n_eval",         type=int,   default=100,
                    help="Tables per class for evaluation (default 100).")
    ap.add_argument("--seq_len",        type=int,   default=500)
    ap.add_argument("--single_eval_pos",type=int,   default=450)
    ap.add_argument("--batch_size",     type=int,   default=32,
                    help="Tables per forward pass during collection.")
    ap.add_argument("--skip_tsne",      action="store_true",
                    help="Skip t-SNE (much faster; useful for a quick check).")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_tabpfn(device)
    manuals = install_recording_manuals(model)

    seq = args.seq_len
    p   = args.single_eval_pos

    make_id  = lambda n: make_inprior(n, seq)
    make_ood = lambda n: make_outprior(n, seq)

    print(f"\n[1/4] Collecting FIT vectors: {args.n_fit} ID tables ...")
    fit_id = collect_head_vectors(model, make_id,  args.n_fit, device, p, manuals, args.batch_size)

    print(f"[2/4] Collecting FIT vectors: {args.n_fit} OOD tables ...")
    fit_ood = collect_head_vectors(model, make_ood, args.n_fit, device, p, manuals, args.batch_size)

    print(f"[3/4] Collecting EVAL vectors: {args.n_eval} ID tables ...")
    eval_id = collect_head_vectors(model, make_id,  args.n_eval, device, p, manuals, args.batch_size)

    print(f"[4/4] Collecting EVAL vectors: {args.n_eval} OOD tables ...")
    eval_ood = collect_head_vectors(model, make_ood, args.n_eval, device, p, manuals, args.batch_size)

    uninstall_manuals(manuals)
    print("Collection complete.\n")

    # ── Per-head analysis ──────────────────────────────────────────────────────
    total_runs = len(TARGET_HEADS) * 2 + 2   # 7 heads × 2 call types + 2 for weighted mean
    run_idx = 0
    for l, h in TARGET_HEADS:
        for call_type in ('ctoc', 'qtoc'):
            run_idx += 1
            key = (l, h, call_type)
            name  = f"L{l}H{h}_{call_type}"
            title = f"L{l}H{h} ({call_type})  δCE={DELTA_CE[(l,h)]:.4f}"
            print(f"[{run_idx}/{total_runs}] {name} — running 3 subexps ...")
            run_three_subexps(
                fit_id[key], fit_ood[key],
                eval_id[key], eval_ood[key],
                RESULTS_DIR, name, title,
                skip_tsne=args.skip_tsne,
            )

    # ── Weighted-mean analysis ─────────────────────────────────────────────────
    for call_type in ('ctoc', 'qtoc'):
        run_idx += 1
        name  = f"weighted_mean_{call_type}"
        title = f"δCE-weighted mean ({call_type})  [{', '.join(f'L{l}H{h}' for l,h in TARGET_HEADS)}]"
        print(f"[{run_idx}/{total_runs}] {name} — running 3 subexps ...")
        run_three_subexps(
            weighted_mean_vectors(fit_id,  call_type),
            weighted_mean_vectors(fit_ood, call_type),
            weighted_mean_vectors(eval_id,  call_type),
            weighted_mean_vectors(eval_ood, call_type),
            RESULTS_DIR, name, title,
            skip_tsne=args.skip_tsne,
        )

    # ── JSON summary ───────────────────────────────────────────────────────────
    summary = {
        "config": vars(args),
        "target_heads": [f"L{l}H{h}" for l, h in TARGET_HEADS],
        "delta_ce_weights": {f"L{l}H{h}": DELTA_CE[(l, h)] for l, h in TARGET_HEADS},
        "total_delta_ce": _TOTAL_W,
        "normalized_weights": {
            f"L{l}H{h}": round(DELTA_CE[(l, h)] / _TOTAL_W, 6)
            for l, h in TARGET_HEADS
        },
        "subexperiments": {
            "subexp1_basic":       f"fit+plot {args.n_fit} ID + {args.n_fit} OOD (no split)",
            "subexp2_fit_id_ood":  f"fit on {args.n_fit} ID+OOD → eval {args.n_eval} new ID+OOD",
            "subexp3_fit_id_only": f"fit on {args.n_fit} ID → eval {args.n_eval} new ID+OOD",
        },
        "note_tsne": (
            "skipped (--skip_tsne)" if args.skip_tsne
            else "run on combined fit+eval; fit=background, eval=foreground"
        ),
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nAll done. {total_runs * 3} plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
