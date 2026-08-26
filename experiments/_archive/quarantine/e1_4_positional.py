import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.metrics import central_jacobian, asym

import tabicl
from tabicl import TabICLRegressor

# Monkeypatch to avoid reloading model weights on every fit() call inside central_jacobian
original_load = TabICLRegressor._load_model
def fast_load(self):
    if hasattr(self, 'model_'):
        return
    original_load(self)
TabICLRegressor._load_model = fast_load

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def main():
    print("=" * 40)
    print("Testing Model: tabicl_v2 (ROPE ZEROED)")
    print("=" * 40)
    
    n_seeds = 5
    seeds = [42, 100, 200, 300, 400]
    
    results = {
        "asym_raw": [],
        "asym_zeroed": []
    }
    
    for seed in seeds:
        print(f"  Running seed {seed}...")
        X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        # Original Model
        model_raw = load("tabicl_v2", task="regression", device="cuda")
        # trigger first load
        model_raw.estimator.fit(X, y)
        def model_func_raw(y_eval):
            trace = model_raw.run(X, y_eval, X)
            return trace.pred
            
        J_raw = central_jacobian(model_func_raw, y_std, h=1e-3)
        Q = get_Q(y_std)
        asym_raw = asym(Q.T @ J_raw @ Q)
        
        # Zero ROPE
        model_zero = load("tabicl_v2", task="regression", device="cuda")
        model_zero.estimator.fit(X, y)
        # Zero out the rope parameter permanently for this instance
        model_zero.estimator.model_.row_interactor.tf_row.rope = None
        
        def model_func_zero(y_eval):
            trace = model_zero.run(X, y_eval, X)
            return trace.pred
            
        J_zero = central_jacobian(model_func_zero, y_std, h=1e-3)
        asym_zero = asym(Q.T @ J_zero @ Q)
        
        results["asym_raw"].append(asym_raw)
        results["asym_zeroed"].append(asym_zero)
        
    mean_raw = np.mean(results["asym_raw"])
    mean_zero = np.mean(results["asym_zeroed"])
    
    print(f"\nE1.4 Results:")
    print(f"  Original Asym: {mean_raw:.4f}")
    print(f"  ROPE-zeroed Asym: {mean_zero:.4f}")
    
    # Write summary
    report = f"""# E1.4 Positional-encoding zeroing

**Model**: TabICL v2
**Description**: We zeroed out the ROPE (Rotary Position Embeddings) in the `RowInteraction` stage of TabICL v2. The ICL transformer itself operates over the set of samples and does not natively employ positional embeddings (making it permutation equivariant across samples). We test whether the ROPE embeddings applied to the *feature* dimension play a role in inducing the measured Jacobian asymmetry.

## Results
- **Original Q-Projected Asymmetry**: {mean_raw:.4f}
- **ROPE-zeroed Q-Projected Asymmetry**: {mean_zero:.4f}

## Conclusion
"""
    if abs(mean_raw - mean_zero) < 0.05:
        report += "The asymmetry persists practically unchanged. This confirms that the non-Bayesian asymmetry in TabICL v2 is an inherent structural property of its multi-head attention mechanism and is **not** an artifact of positional embeddings."
    else:
        report += "The asymmetry significantly changes when positional encodings are zeroed. Positional encodings contribute heavily to the observed structural violations."
        
    out_path = Path(__file__).resolve().parent / "e1_4_report.md"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {out_path}")

if __name__ == '__main__':
    main()
