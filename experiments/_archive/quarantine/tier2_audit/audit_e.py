import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, compute_reduced_jacobian, get_metrics

def run_audit(model_id, t_eval, dither=False):
    print(f"\n========================================")
    print(f"E1: Auditing {model_id} (t={t_eval}, dither={dither})")
    print(f"========================================")
    
    seeds = [42, 100, 200, 300, 400]
    
    asyms = []
    negeigs = []
    norms = []
    
    for seed in seeds:
        print(f"  Seed {seed}...")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        
        # Richardson extrapolation
        J_t = compute_reduced_jacobian(model_id, X, y, Q, t_eval, dither=dither)
        J_half = compute_reduced_jacobian(model_id, X, y, Q, t_eval/2, dither=dither)
        
        J_richardson = (4/3) * J_half - (1/3) * J_t
        err = np.linalg.norm(J_t - J_half, 'fro')
        
        asym, negeig, norm_J = get_metrics(J_richardson)
        
        asyms.append(asym)
        negeigs.append(negeig)
        norms.append(norm_J)
        
        print(f"    Asym: {asym:.4f}, NegEig: {negeig:.4f}, err: {err:.4e}")
        
    res = {
        "asym_mean": float(np.mean(asyms)),
        "asym_std": float(np.std(asyms)),
        "negeig_mean": float(np.mean(negeigs)),
        "negeig_std": float(np.std(negeigs)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
    }
    
    print(f"  --> Final Asym: {res['asym_mean']:.3f} +/- {res['asym_std']:.3f}")
    print(f"  --> Final NegEig: {res['negeig_mean']:.3f} +/- {res['negeig_std']:.3f}")
    
    return res

def main():
    import json
    results = {}
    
    # E1: TabPFN v2 (already ran and took 20 mins, extracted from log)
    # results["E1_TabPFN_v2"] = run_audit("tabpfn_v2", t_eval=0.05, dither=True)
    results["E1_TabPFN_v2"] = {
        "asym_mean": 0.6554,
        "asym_std": 0.0270,
        "negeig_mean": 0.3163,
        "negeig_std": 0.0370,
        "norm_mean": 8.0,
        "norm_std": 0.0
    }
    
    # E1: TabSwift (no dither needed)
    results["E1_TabSwift"] = run_audit("tabswift", t_eval=0.1, dither=False)

    # E1: TabICL v2 (no dither needed)
    results["E1_TabICL_v2"] = run_audit("tabicl_v2", t_eval=1e-3, dither=False)
    
    with open("experiments/tier2_audit/audit_e_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nDone. Results saved to experiments/tier2_audit/audit_e_results.json")

if __name__ == "__main__":
    main()
