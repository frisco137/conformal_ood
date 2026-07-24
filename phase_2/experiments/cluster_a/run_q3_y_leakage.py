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
from phase_2.src.patching.label_perturbation import run_label_perturbation_experiment

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q3: Label Leakage into Kernel Construction ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_a", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_a", exist_ok=True)
    
    n_tasks = 10
    tasks = [sample_task(n=100, d=1, kernel_type="rbf", l=1.0, seed=6000+i, task_mode="gp") for i in range(n_tasks)]
    
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    y_leakage_results = []
    
    for m_name, m_wrapper in models.items():
        print(f"Running label perturbation tests on {m_name}...")
        
        k_shifts = {l: [] for l in range(12)}
        s_shifts = {l: [] for l in range(12)}
        
        for task in tasks:
            res = run_label_perturbation_experiment(m_wrapper, task)
            for l in range(12):
                k_shifts[l].append(res["kernel_shifts"][l])
                s_shifts[l].append(res["solution_shifts"][l])
                
        for l in range(12):
            y_leakage_results.append({
                "Model": m_name,
                "Layer": l,
                "Relative_Kernel_Shift": float(np.mean(k_shifts[l])),
                "Relative_Solution_Shift": float(np.mean(s_shifts[l]))
            })
            
    df_q3 = pd.DataFrame(y_leakage_results)
    df_q3.to_csv("phase_2/results/data/cluster_a/q3_y_leakage.csv", index=False)
    print("\nSaved y-leakage results to phase_2/results/data/cluster_a/q3_y_leakage.csv")
    print(df_q3.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for idx, m_name in enumerate(["TabPFN_v2", "TabICL_v2"]):
        df_sub = df_q3[df_q3["Model"] == m_name]
        axes[idx].plot(df_sub["Layer"], df_sub["Relative_Kernel_Shift"], 'o-', label=r'Kernel Representation Shift ($\Delta K$)', color='#2ca02c', linewidth=2)
        axes[idx].plot(df_sub["Layer"], df_sub["Relative_Solution_Shift"], 's--', label=r'Solution Shift ($\Delta \alpha$)', color='#d62728', linewidth=2)
        axes[idx].set_title(f"{m_name}: Label Perturbation Sensitivity", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Layer Index", fontsize=11)
        axes[idx].grid(True, linestyle='--', alpha=0.6)
        axes[idx].set_xticks(range(12))
        axes[idx].legend(fontsize=10)
        
    axes[0].set_ylabel("Relative Shift Norm", fontsize=11)
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_a/q3_y_leakage.pdf")
    plt.savefig("phase_2/results/figures/cluster_a/q3_y_leakage.png", dpi=300)
    plt.close()
    
    print("\nQ3 Experiment completed successfully!")

if __name__ == "__main__":
    main()
