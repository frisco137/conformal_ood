import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.context import generate_audit_context

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def run_psd_precheck(model_id, t_eval):
    print(f"\n" + "="*40)
    print(f"B1: Cheap PSD Pre-check ({model_id})")
    print("="*40)
    
    seed = 42
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    model = load(model_id, task="regression", device="cuda")
    
    Q = get_Q(y)
    n = len(y)
    
    # 30 random directions in span{1, u}^perp
    np.random.seed(seed+2)
    directions = np.random.randn(30, n - 2)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    
    # Map back to R^n
    vs = directions @ Q.T
    
    neg_count = 0
    min_val = float('inf')
    
    print(f"Testing 30 random directions with t = {t_eval}...")
    for i, v in enumerate(vs):
        y_plus = y + t_eval * v
        y_minus = y - t_eval * v
        
        model.estimator.fit(X, y_plus)
        m_plus = model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
        
        model.estimator.fit(X, y_minus)
        m_minus = model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
        
        deriv = float(np.sum(v * (m_plus.flatten() - m_minus.flatten())) / (2 * t_eval))
        
        if deriv < min_val:
            min_val = deriv
        if deriv < 0:
            neg_count += 1
            
    print(f"Minimum directional derivative: {min_val:.4e}")
    print(f"Number of negative probes: {neg_count} / 30")
    
    return {"min_deriv": float(min_val), "neg_count": int(neg_count)}

def main():
    import json
    
    results = {}
    # For TabPFN, use t=0.05 to clear the Delta=6.87e-4 staircase
    results["B1_TabPFN"] = run_psd_precheck("tabpfn_v2", 0.05)
    # For TabSwift, use t=0.1
    results["B1_TabSwift"] = run_psd_precheck("tabswift", 0.1)
    
    with open("experiments/tier2_audit/diagnostic_b_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nDone. Results saved to experiments/tier2_audit/diagnostic_b_results.json")

if __name__ == "__main__":
    main()
