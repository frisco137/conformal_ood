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
from phase_2.src.alignment.cka_procrustes import evaluate_cross_model_alignment

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q8: Cross-Model Linear State Alignment (CKA & Procrustes) ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_d", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_d", exist_ok=True)
    
    n_tasks = 15
    batch_tasks = [sample_task(n=100, d=1, kernel_type="rbf", l=1.0, seed=11000+i, task_mode="gp") for i in range(n_tasks)]
    
    print("Evaluating cross-model representation alignment across layers...")
    q8_results = evaluate_cross_model_alignment(pfn_wrapper, icl_wrapper, batch_tasks)
    
    df_q8 = pd.DataFrame(q8_results)
    df_q8.to_csv("phase_2/results/data/cluster_d/q8_cross_model_alignment.csv", index=False)
    print("\nSaved Q8 cross-model alignment results to phase_2/results/data/cluster_d/q8_cross_model_alignment.csv")
    print(df_q8.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    ax.plot(df_q8["Layer"], df_q8["Cross_Model_CKA"], 'o-', label="Cross-Model Linear CKA (TabPFN vs. TabICL)", color='#d62728', linewidth=2.5)
    ax.plot(df_q8["Layer"], df_q8["Cross_Model_Alignment_R2"], 's--', label="Cross-Model Procrustes $R^2$", color='#9467bd', linewidth=2.0)
    
    ax.plot(df_q8["Layer"], df_q8["TabPFN_Within_Family_CKA"], '^:', label="TabPFN Within-Family CKA ($L_l$ vs $L_{l-1}$)", color='#1f77b4', alpha=0.7)
    ax.plot(df_q8["Layer"], df_q8["TabICL_Within_Family_CKA"], 'v:', label="TabICL Within-Family CKA ($L_l$ vs $L_{l-1}$)", color='#ff7f0e', alpha=0.7)
    
    ax.set_title("Q8: Cross-Model Representation Alignment Across Layers", fontsize=12, fontweight='bold')
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Linear CKA / Alignment $R^2$", fontsize=11)
    ax.set_xticks(range(12))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_d/q8_cross_model_alignment.pdf")
    plt.savefig("phase_2/results/figures/cluster_d/q8_cross_model_alignment.png", dpi=300)
    plt.close()
    
    print("\nQ8 Experiment completed successfully!")

if __name__ == "__main__":
    main()
