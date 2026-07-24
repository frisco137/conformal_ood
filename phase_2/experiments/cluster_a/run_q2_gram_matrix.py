import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure phase_2 package is importable
sys.path.insert(0, os.path.abspath("."))

from phase_2.src.models.tabpfn_wrapper import TabPFNWrapper
from phase_2.src.models.tabicl_wrapper import TabICLWrapper
from phase_2.src.utils.gp_generator import sample_task
from phase_2.src.probing.gram_probes import evaluate_gram_decodability, compute_attention_gram_alignment

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q2: Gram Matrix Materialization ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_a", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_a", exist_ok=True)
    
    n_tasks = 15
    gp_tasks = [sample_task(n=100, d=1, kernel_type="rbf", l=1.0, seed=5000+i, task_mode="gp") for i in range(n_tasks)]
    
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    gram_results = []
    attn_alignments = {m_name: {l: [] for l in range(12)} for m_name in models}
    
    for m_name, m_wrapper in models.items():
        print(f"Processing {m_name}...")
        
        support_embs = {l: [] for l in range(12)}
        true_grams = []
        
        for task in gp_tasks:
            art = m_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
            for l in range(12):
                support_embs[l].append(art["support_embs"][l])
            true_grams.append(task["G"])
            
            if "item_attn" in art and len(art["item_attn"]) > 0:
                alignment = compute_attention_gram_alignment(art["item_attn"], task["G"])
                for l, score in alignment.items():
                    attn_alignments[m_name][l].append(score)
                    
        for l in range(12):
            r2_gram = evaluate_gram_decodability(support_embs[l], true_grams)
            attn_cos_sim = np.mean(attn_alignments[m_name][l]) if len(attn_alignments[m_name][l]) > 0 else np.nan
            
            gram_results.append({
                "Model": m_name,
                "Layer": l,
                "R2_Gram_Entry": max(0.0, r2_gram),
                "Attn_Gram_CosSim": attn_cos_sim
            })
            
    df_gram = pd.DataFrame(gram_results)
    df_gram.to_csv("phase_2/results/data/cluster_a/q2_gram_matrix.csv", index=False)
    print("\nSaved Gram matrix decodability results to phase_2/results/data/cluster_a/q2_gram_matrix.csv")
    print(df_gram.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_gram[df_gram["Model"] == m_name]
        ax.plot(df_sub["Layer"], df_sub["R2_Gram_Entry"], 'o-', label=f"{m_name} ($K_{{ij}}$ Probing $R^2$)", color=color, linewidth=2)
        
    ax.set_title("Q2: Pairwise Gram Matrix $K(x_i, x_j)$ Decodability Across Layers", fontsize=12, fontweight='bold')
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Pairwise Gram Entry Decoding $R^2$", fontsize=11)
    ax.set_xticks(range(12))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_a/q2_gram_decodability.pdf")
    plt.savefig("phase_2/results/figures/cluster_a/q2_gram_decodability.png", dpi=300)
    plt.close()
    
    print("\nQ2 Experiment completed successfully!")

if __name__ == "__main__":
    main()
