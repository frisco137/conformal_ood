import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics, compute_reduced_jacobian

def main():
    print("Running 2a: Imitator Control with Dithering")
    
    seeds = [42, 100, 200, 300, 400]
    t = 1e-3
    
    asyms_nodither = []
    negs_nodither = []
    
    asyms_dither = []
    negs_dither = []
    
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        
        # Without dithering
        J_no = compute_reduced_jacobian("imitator", X, y, Q, t, dither=False)
        a_no, n_no, _ = get_metrics(J_no)
        asyms_nodither.append(a_no)
        negs_nodither.append(n_no)
        
        # With dithering (TabPFN config: N=10, delta=6.87e-4)
        J_dither = compute_reduced_jacobian("imitator", X, y, Q, t, dither=True, N=10, delta=6.87e-4)
        a_di, n_di, _ = get_metrics(J_dither)
        asyms_dither.append(a_di)
        negs_dither.append(n_di)
        
        print(f"Seed {seed}: NoDither asym={a_no:.4f} negeig={n_no:.4f} | Dither asym={a_di:.4f} negeig={n_di:.4f}")

    a_no_m, a_no_s = np.mean(asyms_nodither), np.std(asyms_nodither)
    a_di_m, a_di_s = np.mean(asyms_dither), np.std(asyms_dither)
    n_no_m, n_no_s = np.mean(negs_nodither), np.std(negs_nodither)
    n_di_m, n_di_s = np.mean(negs_dither), np.std(negs_dither)
    
    print("\nSummary:")
    print(f"Imitator (No Dither): asym={a_no_m:.4f} +/- {a_no_s:.4f}, negeig={n_no_m:.4f} +/- {n_no_s:.4f}")
    print(f"Imitator (Dithered):  asym={a_di_m:.4f} +/- {a_di_s:.4f}, negeig={n_di_m:.4f} +/- {n_di_s:.4f}")

if __name__ == "__main__":
    main()
