import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.metrics import central_jacobian

def main():
    print("=" * 40)
    print("E2.5: Cross-channel Law (cv_s <= cv_J)")
    print("=" * 40)
    
    seeds = [42, 100, 200, 300, 400]
    
    results_cv_s = []
    results_cv_J = []
    
    for seed in seeds:
        X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        
        model = load("tabicl_v2", task="regression", device="cuda")
        model.estimator.fit(X, y)
        s2 = model.estimator.predict(X, output_type='variance')
        
        def model_func(y_eval):
            trace = model.run(X, y_eval, X)
            return trace.pred
            
        J = central_jacobian(model_func, y, h=1e-3)
        J_diag = np.diag(J)
        
        cv_s = np.std(s2) / np.mean(s2)
        cv_J = np.std(J_diag) / np.mean(J_diag)
        
        results_cv_s.append(cv_s)
        results_cv_J.append(cv_J)
        
        print(f"  Seed {seed}: cv_s={cv_s:.4f}, cv_J={cv_J:.4f}  (Passes cv_s <= cv_J: {cv_s <= cv_J})")
        
    mean_cv_s = np.mean(results_cv_s)
    mean_cv_J = np.mean(results_cv_J)
    
    report = f"""# E2.5 Cross-channel Law

**Model**: TabICL v2
**Description**: The cross-channel law for exact Bayesian inference states that `s²ᵢ = σ²(1 + Jᵢᵢ)`. Since variance is positive, this imposes a parameter-free inequality on their coefficients of variation: `cv_s ≤ cv_J`. We test this condition directly.

## Results
- Mean `cv_s` (Predictive Variance): {mean_cv_s:.4f}
- Mean `cv_J` (Jacobian Diagonal): {mean_cv_J:.4f}

## Conclusion
"""
    if mean_cv_s <= mean_cv_J:
        report += "The model PASSES the parameter-free cross-channel inequality, demonstrating consistency between its predictive uncertainty and its label sensitivity."
    else:
        report += "The model FAILS the parameter-free cross-channel inequality, demonstrating a structural contradiction between its predictive uncertainty and its label sensitivity."
        
    out_path = Path(__file__).resolve().parent / "e2_5_report.md"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {out_path}")

if __name__ == '__main__':
    main()
