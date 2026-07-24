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
from phase_2.src.adaptation.noise_inference import evaluate_noise_adaptation

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("muted")

def main():
    print("=== Running Phase 2 - Q4: Adaptive Noise Inference vs Baked-In Regularizer ===")
    
    print("Loading model wrappers...")
    pfn_wrapper = TabPFNWrapper(device="cuda")
    icl_wrapper = TabICLWrapper(device="cuda")
    
    os.makedirs("phase_2/results/data/cluster_b", exist_ok=True)
    os.makedirs("phase_2/results/figures/cluster_b", exist_ok=True)
    
    sigma_true_list = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
    models = {"TabPFN_v2": pfn_wrapper, "TabICL_v2": icl_wrapper}
    
    all_rows = []
    
    for m_name, m_wrapper in models.items():
        print(f"Running noise adaptation for {m_name}...")
        res = evaluate_noise_adaptation(m_wrapper, sigma_true_list, seed_base=7000)
        for r in res:
            r["Model"] = m_name
            all_rows.append(r)
            
    df_q4 = pd.DataFrame(all_rows)
    df_q4.to_csv("phase_2/results/data/cluster_b/q4_noise_adaptation.csv", index=False)
    print("\nSaved Q4 noise adaptation results to phase_2/results/data/cluster_b/q4_noise_adaptation.csv")
    
    # Calculate regression slopes per model and layer
    slopes = []
    for (m_name, l), group in df_q4.groupby(["Model", "Layer"]):
        x = group["sigma2_true"].values
        y = group["implied_sigma2_gcv"].values
        slope, intercept = np.polyfit(x, y, 1)
        slopes.append({
            "Model": m_name,
            "Layer": l,
            "Adaptation_Slope": slope,
            "Intercept": intercept
        })
        
    df_slopes = pd.DataFrame(slopes)
    print("\nImplied Regularizer Adaptation Slopes d(sigma_hat^2) / d(sigma_true^2):")
    print(df_slopes.to_string())
    
    # -------------------------------------------------------------
    # Plotting Figure
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m_name, color in zip(["TabPFN_v2", "TabICL_v2"], ["#1f77b4", "#ff7f0e"]):
        df_sub = df_slopes[df_slopes["Model"] == m_name]
        ax.plot(df_sub["Layer"], df_sub["Adaptation_Slope"], 'o-', label=f"{m_name} (Slope $d\\hat{{\\sigma}}^2 / d\\sigma_{{true}}^2$)", color=color, linewidth=2)
        
    ax.axhline(1.0, color='gray', linestyle=':', label='Ideal Calibrated Slope (1.0)')
    ax.axhline(0.0, color='red', linestyle='--', label='Fixed Baked-in Slope (0.0)')
    ax.set_title("Q4: Noise Adaptation Slope Across Layers", fontsize=12, fontweight='bold')
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel(r"Implied Noise Adaptation Slope $\frac{d\hat{\sigma}^2}{d\sigma_{\text{true}}^2}$", fontsize=11)
    ax.set_xticks(range(12))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("phase_2/results/figures/cluster_b/q4_noise_adaptation.pdf")
    plt.savefig("phase_2/results/figures/cluster_b/q4_noise_adaptation.png", dpi=300)
    plt.close()
    
    print("\nQ4 Experiment completed successfully!")

if __name__ == "__main__":
    main()
