import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, compute_reduced_jacobian, get_metrics

def main():
    print("--- Section 2: TabPFN v2 Un-Voiding & Sweep ---")
    seeds = [42, 100, 200, 300, 400]
    
    # TabPFN sweep at amplitudes above the noise floor
    ts_tabpfn = [1e-2, 3e-2, 1e-1]
    results_pfn = {t: {"asym": [], "negeig": []} for t in ts_tabpfn}
    
    for seed in seeds:
        print(f"\nTabPFN Seed {seed}")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        for t in ts_tabpfn:
            J = compute_reduced_jacobian("tabpfn_v2", X, y, Q, t, dither=True, N=10, delta=6.87e-4)
            a, n_eig, _ = get_metrics(J)
            results_pfn[t]["asym"].append(a)
            results_pfn[t]["negeig"].append(n_eig)
            print(f"  t={t:.1e}: asym={a:.4f}, negeig={n_eig:.4f}")
            
    print("\nTabPFN Sweep Summary:")
    for t in ts_tabpfn:
        a_mean, a_std = np.mean(results_pfn[t]["asym"]), np.std(results_pfn[t]["asym"])
        n_mean, n_std = np.mean(results_pfn[t]["negeig"]), np.std(results_pfn[t]["negeig"])
        print(f"t={t:.1e}: asym={a_mean:.3f} +/- {a_std:.3f}, negeig={n_mean:.3f} +/- {n_std:.3f}")

    print("\n--- Section 3: TabSwift Un-Voiding & Sweep ---")
    ts_swift = [1e-1, 1.0, 10.0] # Within A4 radius
    results_swift = {t: {"asym": [], "negeig": []} for t in ts_swift}
    
    for seed in seeds:
        print(f"\nTabSwift Seed {seed}")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        for t in ts_swift:
            J = compute_reduced_jacobian("tabswift", X, y, Q, t, dither=False)
            a, n_eig, _ = get_metrics(J)
            results_swift[t]["asym"].append(a)
            results_swift[t]["negeig"].append(n_eig)
            print(f"  t={t:.1e}: asym={a:.4f}, negeig={n_eig:.4f}")
            
    print("\nTabSwift Sweep Summary:")
    for t in ts_swift:
        a_mean, a_std = np.mean(results_swift[t]["asym"]), np.std(results_swift[t]["asym"])
        n_mean, n_std = np.mean(results_swift[t]["negeig"]), np.std(results_swift[t]["negeig"])
        print(f"t={t:.1e}: asym={a_mean:.3f} +/- {a_std:.3f}, negeig={n_mean:.3f} +/- {n_std:.3f}")

if __name__ == "__main__":
    main()
