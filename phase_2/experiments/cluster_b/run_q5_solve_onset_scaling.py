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
from phase_2.src.adaptation.solve_onset import evaluate_solve_onset_scaling

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q5: Construct -> Solve Boundary Movement (l_0) ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_b", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_b", exist_ok=True)
    
    n_values = [20, 50, 100, 200, 400]
    d_values = [1, 5, 10, 20, 50]
    
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    all_onset_results = []
    
    for m_name, m_wrapper in models.items():
        print(f"Evaluating solve onset scaling for {m_name}...")
        res = evaluate_solve_onset_scaling(m_wrapper, n_values, d_values, seed_base=8000)
        for r in res:
            r["Model"] = m_name
            all_onset_results.append(r)
            
    df_q5 = pd.DataFrame(all_onset_results)
    df_q5.to_csv("phase_2/results/data/cluster_b/q5_solve_onset_scaling.csv", index=False)
    print("\nSaved Q5 solve onset results to phase_2/results/data/cluster_b/q5_solve_onset_scaling.csv")
    print(df_q5.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    
    # Plot vs n
    df_n = df_q5[df_q5["Varying_Parameter"] == "n"]
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_n[df_n["Model"] == m_name]
        axes[0].plot(df_sub["n"], df_sub["solve_onset_layer_l0"], 'o-', label=m_name, color=color, linewidth=2)
        
    axes[0].set_title("Solve Onset Layer $l_0$ vs. Context Size $n$", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Context Size $n$ (Fixed $d=1$)", fontsize=11)
    axes[0].set_ylabel("Solve Onset Layer $l_0$", fontsize=11)
    axes[0].set_yticks(range(12))
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)
    
    # Plot vs d
    df_d = df_q5[df_q5["Varying_Parameter"] == "d"]
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_d[df_d["Model"] == m_name]
        axes[1].plot(df_sub["d"], df_sub["solve_onset_layer_l0"], 's--', label=m_name, color=color, linewidth=2)
        
    axes[1].set_title("Solve Onset Layer $l_0$ vs. Feature Dimension $d$", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Feature Dimension $d$ (Fixed $n=100$)", fontsize=11)
    axes[1].set_yticks(range(12))
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_b/q5_solve_onset_scaling.pdf")
    plt.savefig("phase_2/results/figures/cluster_b/q5_solve_onset_scaling.png", dpi=300)
    plt.close()
    
    print("\nQ5 Experiment completed successfully!")

if __name__ == "__main__":
    main()
