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
from phase_2.src.utils.metrics import compute_effective_rank
from phase_2.src.probing.primal_dual_probes import evaluate_layer_probes
from phase_2.src.patching.loo_ablation import evaluate_loo_empirical_dependence

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q1: Primal vs Dual Solution Representation ===")
    
    # 1. Initialize Wrappers
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    # Setup data directories
    os.makedirs("phase_2/results/data/cluster_a", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_a", exist_ok=True)
    
    # -------------------------------------------------------------
    # Q1.1: Probing \hat{w} (Primal) vs \alpha (Dual) Across Layers
    # -------------------------------------------------------------
    print("\n--- Running Q1.1: Linear Probing for Primal w* and Dual alpha* ---")
    
    # Generate tasks
    n_tasks = 20
    gp_tasks = [sample_task(n=100, d=1, kernel_type="rbf", seed=1000+i, task_mode="gp") for i in range(n_tasks)]
    linear_tasks = [sample_task(n=100, d=5, seed=2000+i, task_mode="linear") for i in range(n_tasks)]
    
    # Collect activations across models
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    probe_results = []
    
    for m_name, m_wrapper in models.items():
        print(f"Extracting activations for {m_name}...")
        
        # 1. Dual alpha* probing on GP tasks
        gp_support_embs = {l: [] for l in range(12)}
        gp_alphas = []
        for task in gp_tasks:
            art = m_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
            for l in range(12):
                gp_support_embs[l].append(art["support_embs"][l])
            gp_alphas.append(task["alpha_star"])
            
        # 2. Primal w* probing on Linear tasks
        lin_support_embs = {l: [] for l in range(12)}
        lin_wstars = []
        for task in linear_tasks:
            art = m_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
            for l in range(12):
                lin_support_embs[l].append(art["support_embs"][l])
            lin_wstars.append(task["w_star"])
            
        # Fit probes layer by layer
        for l in range(12):
            r2_dual = evaluate_layer_probes(gp_support_embs[l], gp_alphas)
            r2_primal = evaluate_layer_probes(lin_support_embs[l], lin_wstars)
            
            probe_results.append({
                "Model": m_name,
                "Layer": l,
                "R2_Dual_alpha": max(0.0, r2_dual),
                "R2_Primal_w": max(0.0, r2_primal)
            })
            
    df_probe = pd.DataFrame(probe_results)
    df_probe.to_csv("phase_2/results/data/cluster_a/q1_primal_dual_probing.csv", index=False)
    print("Saved probing results to phase_2/results/data/cluster_a/q1_primal_dual_probing.csv")
    print(df_probe.to_string())
    
    # -------------------------------------------------------------
    # Q1.2: Leave-One-Context-Out (LOO) Empirical Influence Test
    # -------------------------------------------------------------
    print("\n--- Running Q1.2: LOO Influence Analysis ---")
    loo_results = []
    
    test_task = sample_task(n=50, d=1, kernel_type="rbf", seed=999, task_mode="gp")
    for m_name, m_wrapper in models.items():
        res = evaluate_loo_empirical_dependence(m_wrapper, test_task)
        loo_results.append({
            "Model": m_name,
            "R2_Dual_Influence": res["r2_dual_influence"],
            "R2_Primal_Influence": res["r2_primal_influence"]
        })
        
    df_loo = pd.DataFrame(loo_results)
    df_loo.to_csv("phase_2/results/data/cluster_a/q1_loo_influence.csv", index=False)
    print("\nSaved LOO influence results to phase_2/results/data/cluster_a/q1_loo_influence.csv")
    print(df_loo.to_string())
    
    # -------------------------------------------------------------
    # Q1.3: Effective Dimensionality Scaling (Varying n and d)
    # -------------------------------------------------------------
    print("\n--- Running Q1.3: Effective Dimensionality Scaling vs n and d ---")
    rank_results = []
    
    d_list = [1, 5, 10, 20]
    n_fixed = 100
    
    for d in d_list:
        task = sample_task(n=n_fixed, d=d, seed=3000+d, task_mode="gp")
        for m_name, m_wrapper in models.items():
            art = m_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
            # Measure rank at Layer 10 (interior solution layer)
            H10 = art["support_embs"][10]
            eff_rank = compute_effective_rank(H10)
            rank_results.append({
                "Model": m_name,
                "Varying_Parameter": "d",
                "d": d,
                "n": n_fixed,
                "Effective_Rank": eff_rank
            })
            
    n_list = [50, 100, 200, 400]
    d_fixed = 1
    
    for n in n_list:
        task = sample_task(n=n, d=d_fixed, seed=4000+n, task_mode="gp")
        for m_name, m_wrapper in models.items():
            art = m_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
            H10 = art["support_embs"][10]
            eff_rank = compute_effective_rank(H10)
            rank_results.append({
                "Model": m_name,
                "Varying_Parameter": "n",
                "d": d_fixed,
                "n": n,
                "Effective_Rank": eff_rank
            })
            
    df_rank = pd.DataFrame(rank_results)
    df_rank.to_csv("phase_2/results/data/cluster_a/q1_effective_rank_scaling.csv", index=False)
    print("\nSaved rank scaling results to phase_2/results/data/cluster_a/q1_effective_rank_scaling.csv")
    print(df_rank.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figures
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for idx, m_name in enumerate(["TabPFN_v2", "TabICL_v2"]):
        df_sub = df_probe[df_probe["Model"] == m_name]
        axes[idx].plot(df_sub["Layer"], df_sub["R2_Dual_alpha"], 'o-', label=r'Dual ($\alpha^*$) $R^2$', color='#1f77b4', linewidth=2)
        axes[idx].plot(df_sub["Layer"], df_sub["R2_Primal_w"], 's--', label=r'Primal ($\hat{w}$) $R^2$', color='#ff7f0e', linewidth=2)
        axes[idx].set_title(f"{m_name}: Solution Probing $R^2$", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Layer Index", fontsize=11)
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].set_xticks(range(12))
        axes[idx].legend(fontsize=10)
        
    axes[0].set_ylabel("Probe Decoding $R^2$", fontsize=11)
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_a/q1_primal_vs_dual_probing.pdf")
    plt.savefig("phase_2/results/figures/cluster_a/q1_primal_vs_dual_probing.png", dpi=300)
    plt.close()
    plt.close()
    
    print("\nQ1 Experiment completed successfully!")

if __name__ == "__main__":
    main()
