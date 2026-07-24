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
from phase_2.src.adaptation.incremental_rank import evaluate_incremental_update

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q6: Batch vs Incremental Update Mechanics ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_b", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_b", exist_ok=True)
    
    n_tasks = 10
    tasks = [sample_task(n=100, d=1, kernel_type="rbf", l=1.0, seed=9000+i, task_mode="gp") for i in range(n_tasks)]
    
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    q6_results = []
    
    for m_name, m_wrapper in models.items():
        print(f"Evaluating single-token insertion for {m_name}...")
        
        ranks = {l: [] for l in range(12)}
        ginis = {l: [] for l in range(12)}
        
        for task in tasks:
            res = evaluate_incremental_update(m_wrapper, task)
            for l in range(12):
                ranks[l].append(res["eff_ranks"][l])
                ginis[l].append(res["gini_localities"][l])
                
        for l in range(12):
            q6_results.append({
                "Model": m_name,
                "Layer": l,
                "Delta_H_Effective_Rank": float(np.mean(ranks[l])),
                "Delta_H_Gini_Locality": float(np.mean(ginis[l]))
            })
            
    df_q6 = pd.DataFrame(q6_results)
    df_q6.to_csv("phase_2/results/data/cluster_b/q6_batch_vs_incremental.csv", index=False)
    print("\nSaved Q6 batch vs incremental results to phase_2/results/data/cluster_b/q6_batch_vs_incremental.csv")
    print(df_q6.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Effective Rank Plot
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_q6[df_q6["Model"] == m_name]
        axes[0].plot(df_sub["Layer"], df_sub["Delta_H_Effective_Rank"], 'o-', label=m_name, color=color, linewidth=2)
        
    axes[0].set_title(r"Activation Update Matrix Rank $\text{rank}_{\text{eff}}(\Delta H^{(\ell)})$", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Layer Index", fontsize=11)
    axes[0].set_ylabel(r"Effective Numerical Rank", fontsize=11)
    axes[0].set_xticks(range(12))
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # Gini Locality Plot
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_q6[df_q6["Model"] == m_name]
        axes[1].plot(df_sub["Layer"], df_sub["Delta_H_Gini_Locality"], 's--', label=m_name, color=color, linewidth=2)
        
    axes[1].set_title(r"Update Spatial Locality Index (Gini Coefficient)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Layer Index", fontsize=11)
    axes[1].set_ylabel(r"Gini Locality Index (0 = Diffuse, 1 = Localized)", fontsize=11)
    axes[1].set_xticks(range(12))
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_b/q6_batch_vs_incremental.pdf")
    plt.savefig("phase_2/results/figures/cluster_b/q6_batch_vs_incremental.png", dpi=300)
    plt.close()
    
    print("\nQ6 Experiment completed successfully!")

if __name__ == "__main__":
    main()
