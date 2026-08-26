import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.core.context import generate_context
from experiments.core.metrics import central_jacobian, asym, negeig
from experiments.core.surrogates import (
    ExactGP, Ridge, NadarayaWatson, OneNN, HierarchicalGP, Imitator
)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-12:
        return 0.0
    return 1 - (ss_res / ss_tot)

def main():
    print("Running E0.2 Analytic Control Battery...\n")
    X, y, f = generate_context(n=50, d=5, seed=42)
    
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    surrogates = [
        ("Exact GP", ExactGP(X, sigma=0.3)),
        ("Ridge", Ridge(X, lam=0.1)),
        ("Nadaraya-Watson", NadarayaWatson(X, lengthscale=1.0)),
        ("1-NN", OneNN(X)),
        ("Hierarchical GP", HierarchicalGP(X, sigma=0.3)),
        ("Imitator", Imitator(X, epsilon=0.1))
    ]
    
    all_passed = True
    
    for name, model in surrogates:
        m_y = model.predict(y_std)
        r2 = r2_score(y_std, m_y)
        
        print(f"--- {name} ---")
        print(f"R^2 on context: {r2:.4f}")
        
        J_fd = central_jacobian(model.predict, y_std, h=1e-4)
        J_cf = model.jacobian(y_std)
        
        # relative error
        norm_cf = np.linalg.norm(J_cf, ord='fro')
        if norm_cf < 1e-12:
            err = np.linalg.norm(J_fd - J_cf, ord='fro')
        else:
            err = np.linalg.norm(J_fd - J_cf, ord='fro') / norm_cf
            
        print(f"Jacobian relative err: {err:.2e}")
        
        if err > 1e-5:
            print(f"FAIL: Finite difference does not match closed form!")
            all_passed = False
            
        a = asym(J_cf)
        e = negeig(J_cf)
        
        if name == "Imitator":
            sym_J = (J_cf + J_cf.T) / 2
            print("Imitator min eigval:", np.min(np.linalg.eigvalsh(sym_J)))
        
        print(f"asym: {a:.4e} | negeig: {e:.4e}")
        print()
        
    if all_passed:
        print("E0.2 PASS: All analytic controls match closed forms to high precision.")
    else:
        print("E0.2 FAIL: One or more models did not match.")
        sys.exit(1)

if __name__ == "__main__":
    main()
