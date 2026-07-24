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
from phase_2.src.uncertainty.variance_dissociation import sample_adversarial_dissociation_task, fit_uncertainty_dissociation_regression

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q7: Uncertainty Computation vs. Density Heuristic ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_c", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_c", exist_ok=True)
    
    task_types = ["dense_uninformative", "sparse_informative"]
    n_tasks = 10
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    
    q7_results = []
    
    for t_type in task_types:
        print(f"\n--- Processing Task Type: {t_type} ---")
        tasks = [sample_adversarial_dissociation_task(n=100, d=1, task_type=t_type, seed=10000+i) for i in range(n_tasks)]
        
        for m_name, m_wrapper in models.items():
            print(f"Running uncertainty regression on {m_name} ({t_type})...")
            
            full_r2_list = []
            gp_r2_list = []
            nn_r2_list = []
            beta_gp_list = []
            beta_nn_list = []
            
            for task in tasks:
                res = fit_uncertainty_dissociation_regression(m_wrapper, task)
                full_r2_list.append(res["full_r2"])
                gp_r2_list.append(res["r2_gp_alone"])
                nn_r2_list.append(res["r2_nn_alone"])
                beta_gp_list.append(res["beta_gp_std"])
                beta_nn_list.append(res["beta_nn_std"])
                
            q7_results.append({
                "Model": m_name,
                "Task_Type": t_type,
                "Full_Regression_R2": float(np.mean(full_r2_list)),
                "R2_GP_Variance_Alone": max(0.0, float(np.mean(gp_r2_list))),
                "R2_NN_Distance_Alone": max(0.0, float(np.mean(nn_r2_list))),
                "Beta_GP_Std": float(np.mean(beta_gp_list)),
                "Beta_NN_Std": float(np.mean(beta_nn_list))
            })
            
    df_q7 = pd.DataFrame(q7_results)
    df_q7.to_csv("phase_2/results/data/cluster_c/q7_uncertainty_dissociation.csv", index=False)
    print("\nSaved Q7 uncertainty dissociation results to phase_2/results/data/cluster_c/q7_uncertainty_dissociation.csv")
    print(df_q7.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Bar plot comparing R^2 of GP Variance alone vs NN Distance alone
    models_labels = []
    r2_gp_vals = []
    r2_nn_vals = []
    
    for row in q7_results:
        label = f"{row['Model']}\n({row['Task_Type'].split('_')[0]})"
        models_labels.append(label)
        r2_gp_vals.append(row["R2_GP_Variance_Alone"])
        r2_nn_vals.append(row["R2_NN_Distance_Alone"])
        
    x = np.arange(len(models_labels))
    width = 0.35
    
    ax.bar(x - width/2, r2_gp_vals, width, label=r'True GP Variance ($\sigma^2_{\text{GP}}$) $R^2$', color='#1f77b4')
    ax.bar(x + width/2, r2_nn_vals, width, label=r'NN Distance ($d_{\text{NN}}$) $R^2$', color='#ff7f0e')
    
    ax.set_title("Q7: Predictive Uncertainty Predictor Dissociation ($R^2$)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_labels, fontsize=10)
    ax.set_ylabel(r"Uncertainty Decoding $R^2$", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_c/q7_uncertainty_dissociation.pdf")
    plt.savefig("phase_2/results/figures/cluster_c/q7_uncertainty_dissociation.png", dpi=300)
    plt.close()
    
    print("\nQ7 Experiment completed successfully!")

if __name__ == "__main__":
    main()
