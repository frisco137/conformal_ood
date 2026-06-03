"""
attention_divergence_ablation.py — Experiment 1 for TabPFN
==========================================================

Combines the two "discovery" analyses from the GP-PFN pipeline
(`circuits/attention_divergence.py` + `circuits/head_ablation.py`), adapted to
TabPFN (12 layers x 4 heads = 48 heads, classifier, two-call attention).

ID  = in-prior data  (prior-conforming, level-2 "In-Prior")
OOD = out-prior data (anti-prior,        level-2 "Out-Prior")

--- Part A: Attention divergence -------------------------------------------------
For every head we compute the Jensen-Shannon Divergence (JSD) between the mean
ID and mean OOD attention maps. Because TabPFN runs self-attention twice per
layer, we report JSD **separately** for:
    * ctoc : context -> context attention  (450 x 450)
    * qtoc : query   -> context attention  ( 50 x 450)

--- Part B: Zero-ablation study (impact on ID prediction quality) ----------------
For every head and every MLP we zero-ablate the component and measure the
increase in cross-entropy (ΔCE) on **in-distribution (in-prior) query points**.
Heads are ablated with `ManualMHA` (zeroing the head's output before out_proj);
MLPs are ablated by zeroing the `linear2` output (the FFN residual contribution).

Outputs (written to circuits_tabpfn/results/):
  - attn_jsd_heatmap_ctoc.png / attn_jsd_heatmap_qtoc.png  (12x4 JSD heatmaps)
  - ablation_delta_ce_bar.png      (ranked ΔCE for all 48 heads + 12 MLPs)
  - ablation_ce_heatmap.png        (12x4 head ΔCE heatmap)
  - cross_reference_jsd_vs_ablation.png  (JSD vs ΔCE scatter, ctoc & qtoc)
  - experiment1_results.json
"""

import os
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

from tabpfn_hooks import (
    load_tabpfn, make_inprior, make_outprior, compute_ce_loss,
    ManualMHA, AttnDivergenceCollector,
    NLAYERS, NHEAD,
)

plt.rcParams.update({
    'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


# --------------------------------------------------------------------------------
def jsd_mean_maps(mean_id: torch.Tensor, mean_ood: torch.Tensor) -> float:
    """Row-wise JSD (squared Jensen-Shannon) between two mean attention maps (Lq, Lk)."""
    a = mean_id.numpy()
    b = mean_ood.numpy()
    vals = []
    for i in range(a.shape[0]):
        p = a[i] + 1e-12
        q = b[i] + 1e-12
        p = p / p.sum()
        q = q / q.sum()
        vals.append(jensenshannon(p, q) ** 2)
    return float(np.mean(vals))


def collect_mean_attention(model, x, y, single_eval_pos, device, batch_size):
    """Run the divergence collector over a dataset; return (ctoc, qtoc) mean-map dicts."""
    col = AttnDivergenceCollector(model, single_eval_pos)
    col.install()
    N = x.shape[1]
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="  attn", leave=False):
            bx = x[:, i:i + batch_size].to(device)
            by = y[:, i:i + batch_size].float().to(device)
            col.reset_call_state()
            _ = model((bx, by), single_eval_pos=single_eval_pos)
    ctoc, qtoc = col.get_mean_maps()
    col.remove()
    return ctoc, qtoc


# --------------------------------------------------------------------------------
def ablate_head_ce(model, layer_idx, head_idx, x, y, single_eval_pos, device, batch_size):
    """ΔCE-ready: CE on ID data with one head zero-ablated in one layer."""
    mha = model.transformer_encoder.layers[layer_idx].self_attn
    mm = ManualMHA(mha)
    mm.heads_to_ablate = {head_idx}
    mm.install()
    ce = compute_ce_loss(model, x, y, device, single_eval_pos, batch_size)
    mm.uninstall()
    return ce


def ablate_mlp_ce(model, layer_idx, x, y, single_eval_pos, device, batch_size):
    """CE on ID data with one layer's MLP (FFN) zero-ablated."""
    layer = model.transformer_encoder.layers[layer_idx]

    def zero_hook(module, inp, out):
        return torch.zeros_like(out)

    h = layer.linear2.register_forward_hook(zero_hook)
    ce = compute_ce_loss(model, x, y, device, single_eval_pos, batch_size)
    h.remove()
    return ce


# --------------------------------------------------------------------------------
def plot_jsd_heatmap(matrix, title, path):
    fig, ax = plt.subplots(figsize=(6, 9), dpi=200)
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    ax.set_xlabel("Head Index", fontweight='bold')
    ax.set_ylabel("Layer Index", fontweight='bold')
    ax.set_xticks(range(NHEAD)); ax.set_yticks(range(NLAYERS))
    ax.set_title(title, fontweight='bold', fontsize=12)
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for l in range(NLAYERS):
        for h in range(NHEAD):
            val = matrix[l, h]
            ax.text(h, l, f"{val:.3f}", ha='center', va='center', fontsize=7,
                    color='white' if val > vmax * 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.7, label='JSD')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--single_eval_pos", type=int, default=450)
    ap.add_argument("--batch_size", type=int, default=50)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = load_tabpfn(device)
    p = args.single_eval_pos

    print(f"Generating {args.n_samples} in-prior (ID) and out-prior (OOD) samples "
          f"(seq_len={args.seq_len}, ctx={p})...")
    x_id, y_id = make_inprior(args.n_samples, args.seq_len, without_noise=False)
    x_ood, y_ood = make_outprior(args.n_samples, args.seq_len)

    # ===== Part A: Attention divergence =====
    print("\n--- Part A: collecting mean attention maps ---")
    print(" ID:")
    ctoc_id, qtoc_id = collect_mean_attention(model, x_id, y_id, p, device, args.batch_size)
    print(" OOD:")
    ctoc_ood, qtoc_ood = collect_mean_attention(model, x_ood, y_ood, p, device, args.batch_size)

    print("--- Computing JSD per head (ctoc & qtoc) ---")
    jsd_ctoc = np.zeros((NLAYERS, NHEAD))
    jsd_qtoc = np.zeros((NLAYERS, NHEAD))
    for l in range(NLAYERS):
        for h in range(NHEAD):
            jsd_ctoc[l, h] = jsd_mean_maps(ctoc_id[l][h], ctoc_ood[l][h])
            jsd_qtoc[l, h] = jsd_mean_maps(qtoc_id[l][h], qtoc_ood[l][h])

    plot_jsd_heatmap(jsd_ctoc, "Attention JSD (ID vs OOD)\ncontext→context",
                     os.path.join(RESULTS_DIR, "attn_jsd_heatmap_ctoc.png"))
    plot_jsd_heatmap(jsd_qtoc, "Attention JSD (ID vs OOD)\nquery→context",
                     os.path.join(RESULTS_DIR, "attn_jsd_heatmap_qtoc.png"))

    # ===== Part B: Zero-ablation =====
    print("\n--- Part B: baseline CE on ID query points ---")
    baseline_ce = compute_ce_loss(model, x_id, y_id, device, p, args.batch_size)
    print(f"Baseline ID CE: {baseline_ce:.6f}")

    head_results = []
    print("--- Per-head ablation ---")
    for l in range(NLAYERS):
        for h in tqdm(range(NHEAD), desc=f"  L{l}", leave=False):
            ce = ablate_head_ce(model, l, h, x_id, y_id, p, device, args.batch_size)
            head_results.append({"component": f"L{l}H{h}", "layer": l, "head": h,
                                 "type": "head", "ablated_ce": ce,
                                 "delta_ce": float(ce - baseline_ce)})

    mlp_results = []
    print("--- Per-MLP ablation ---")
    for l in tqdm(range(NLAYERS), desc="  MLP", leave=False):
        ce = ablate_mlp_ce(model, l, x_id, y_id, p, device, args.batch_size)
        mlp_results.append({"component": f"L{l}_MLP", "layer": l, "type": "mlp",
                            "ablated_ce": ce, "delta_ce": float(ce - baseline_ce)})

    all_results = head_results + mlp_results
    all_sorted = sorted(all_results, key=lambda r: r["delta_ce"], reverse=True)

    # ----- Plot: ΔCE bar chart -----
    fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    names = [r["component"] for r in all_sorted]
    deltas = [r["delta_ce"] for r in all_sorted]
    bar_colors = ['#ef4444' if r["type"] == "mlp" else '#3b82f6' for r in all_sorted]
    ax.bar(range(len(names)), deltas, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel("ΔCE (ablated − baseline)", fontweight='bold')
    ax.set_title("TabPFN: Impact of Component Ablation on In-Prior Prediction (Cross-Entropy)",
                 fontweight='bold', fontsize=13, pad=12)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(handles=[Patch(color='#3b82f6', label='Attention Head'),
                       Patch(color='#ef4444', label='MLP')], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_delta_ce_bar.png"), bbox_inches='tight')
    plt.close()

    # ----- Plot: ΔCE head heatmap -----
    ce_matrix = np.zeros((NLAYERS, NHEAD))
    for r in head_results:
        ce_matrix[r["layer"], r["head"]] = r["delta_ce"]
    fig, ax = plt.subplots(figsize=(6, 9), dpi=200)
    im = ax.imshow(ce_matrix, cmap='Reds', aspect='auto')
    ax.set_xlabel("Head Index", fontweight='bold'); ax.set_ylabel("Layer Index", fontweight='bold')
    ax.set_xticks(range(NHEAD)); ax.set_yticks(range(NLAYERS))
    ax.set_title("Head Ablation Impact (ΔCE on ID)", fontweight='bold', fontsize=12)
    vmax = ce_matrix.max() if ce_matrix.max() > 0 else 1.0
    for l in range(NLAYERS):
        for h in range(NHEAD):
            val = ce_matrix[l, h]
            ax.text(h, l, f"{val:.3f}", ha='center', va='center', fontsize=7,
                    color='white' if val > vmax * 0.6 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.7, label='ΔCE')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_ce_heatmap.png"), bbox_inches='tight')
    plt.close()

    # ----- Plot: cross-reference JSD vs ablation (ctoc & qtoc) -----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    for ax, jsd_mat, label in zip(axes, [jsd_ctoc, jsd_qtoc], ["ctoc", "qtoc"]):
        for r in head_results:
            l, h = r["layer"], r["head"]
            ax.scatter(jsd_mat[l, h], r["delta_ce"], s=60, zorder=5,
                       edgecolors='white', linewidth=0.5, color=plt.cm.viridis(l / NLAYERS))
            ax.annotate(f"L{l}H{h}", (jsd_mat[l, h], r["delta_ce"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=6)
        # correlation
        xs = [jsd_mat[r["layer"], r["head"]] for r in head_results]
        ys = [r["delta_ce"] for r in head_results]
        corr = float(np.corrcoef(xs, ys)[0, 1]) if np.std(xs) > 0 and np.std(ys) > 0 else 0.0
        ax.set_xlabel(f"Attention JSD ({label})", fontweight='bold')
        ax.set_ylabel("ΔCE from Ablation", fontweight='bold')
        ax.set_title(f"JSD ({label}) vs Ablation Impact  (Pearson r = {corr:.3f})",
                     fontweight='bold', fontsize=12)
        ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.suptitle("Cross-Reference: Attention Divergence vs Functional Importance",
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_reference_jsd_vs_ablation.png"), bbox_inches='tight')
    plt.close()

    # ----- JSON -----
    summary = {
        "config": vars(args),
        "baseline_ce_id": baseline_ce,
        "jsd_ctoc": jsd_ctoc.tolist(),
        "jsd_qtoc": jsd_qtoc.tolist(),
        "head_ablation": head_results,
        "mlp_ablation": mlp_results,
        "top5_components_by_delta_ce": all_sorted[:5],
    }
    with open(os.path.join(RESULTS_DIR, "experiment1_results.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nExperiment 1 complete! Results in", RESULTS_DIR)
    print("Top 5 components by ΔCE:")
    for r in all_sorted[:5]:
        print(f"  {r['component']}: ΔCE = {r['delta_ce']:+.6f}")


if __name__ == "__main__":
    main()
