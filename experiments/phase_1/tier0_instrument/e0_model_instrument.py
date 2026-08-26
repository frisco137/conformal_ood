import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models import load
from experiments.phase_1.core.context import generate_context
from experiments.phase_1.core.metrics import central_jacobian, asym, negeig

def run_tests_for_model(model_name):
    print(f"\n{'='*40}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*40}")
    
    X, y, f = generate_context(n=50, d=5, seed=42)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    print("Loading model...")
    model = load(model_name, task="regression", device="cuda")
    
    def model_func(y_eval):
        trace = model.run(X, y_eval, X)
        return trace.pred
        
    print("\n--- E0.1 Determinism ---")
    pred1 = model_func(y_std)
    pred2 = model_func(y_std)
    max_diff = np.max(np.abs(pred1 - pred2))
    
    if max_diff < 1e-9:
        print(f"PASS: Deterministic (max_diff = {max_diff:.2e})")
    else:
        print(f"FAIL: Nondeterministic (max_diff = {max_diff:.2e})")
        
    print("\n--- E0.3 Step-size Plateau ---")
    h_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    J_best = None
    
    for h in h_values:
        J = central_jacobian(model_func, y_std, h=h)
        a = asym(J)
        e = negeig(J)
        if h == 1e-3:
            J_best = J
        print(f"h={h:.0e} | asym: {a:.4e} | negeig: {e:.4e}")
        
    print("\n--- E0.6 Row-sum Residual (at h=1e-3) ---")
    if J_best is not None:
        ones = np.ones(len(y_std))
        r1 = np.linalg.norm(J_best @ ones - ones) / np.linalg.norm(ones)
        print(f"r_1 = {r1:.4e}")
        if r1 > 0.1:
            print(f"FAIL: Jacobian is heavily corrupted (r_1 > 0.1).")
        else:
            print(f"PASS: Instrument noise bound established.")

def main():
    models = ["tabicl_v2", "tabpfn_v2", "tabswift"]
    for m in models:
        try:
            run_tests_for_model(m)
        except Exception as e:
            print(f"Error running {m}: {e}")

if __name__ == "__main__":
    main()
