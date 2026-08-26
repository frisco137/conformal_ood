import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.context import generate_audit_context
from experiments.core.metrics import central_jacobian
from experiments.core.surrogates import ExactGP, Ridge, NadarayaWatson, OneNN

import tabicl
from tabicl import TabICLRegressor

# Monkeypatch to avoid reloading model weights on every fit() call inside central_jacobian
original_load = TabICLRegressor._load_model
def fast_load(self):
    if hasattr(self, 'model_'):
        return
    original_load(self)
TabICLRegressor._load_model = fast_load

def main():
    print("=" * 40)
    print("E2.6: Mechanism Surrogate Fitting")
    print("=" * 40)
    
    seed = 42
    print(f"  Running seed {seed}...")
    X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    
    model = load("tabicl_v2", task="regression", device="cuda")
    model.estimator.fit(X, y)
    
    # Get model's map and Jacobian on this context
    m_y = model.estimator.predict(X, output_type='mean')
    
    def model_func(y_eval):
        trace = model.run(X, y_eval, X)
        return trace.pred
        
    J_model = central_jacobian(model_func, y, h=1e-3)
    
    print("\nFitting mechanisms to model's outputs (minimizing ||s(y) - m(y)||)...")
    
    # 1. ExactGP
    def obj_gp(params):
        sigma, ls = params
        if sigma <= 0 or ls <= 0: return np.inf
        gp = ExactGP(X, sigma=sigma, lengthscale=ls)
        return np.linalg.norm(gp.predict(y) - m_y)
        
    res_gp = minimize(obj_gp, [0.5, 1.0], bounds=[(1e-3, 10.0), (1e-3, 10.0)])
    gp = ExactGP(X, sigma=res_gp.x[0], lengthscale=res_gp.x[1])
    err_val_gp = np.linalg.norm(gp.predict(y) - m_y) / np.linalg.norm(m_y)
    err_jac_gp = np.linalg.norm(gp.jacobian(y) - J_model, 'fro') / np.linalg.norm(J_model, 'fro')
    print(f"  ExactGP       (sigma={res_gp.x[0]:.3f}, ls={res_gp.x[1]:.3f}) -> Value Error: {err_val_gp:.4f}, Jac Error: {err_jac_gp:.4f}")
    
    # 2. Ridge
    def obj_ridge(params):
        lam = params[0]
        if lam <= 0: return np.inf
        ridge = Ridge(X, lam=lam)
        return np.linalg.norm(ridge.predict(y) - m_y)
        
    res_ridge = minimize(obj_ridge, [1.0], bounds=[(1e-4, 100.0)])
    ridge = Ridge(X, lam=res_ridge.x[0])
    err_val_ridge = np.linalg.norm(ridge.predict(y) - m_y) / np.linalg.norm(m_y)
    err_jac_ridge = np.linalg.norm(ridge.jacobian(y) - J_model, 'fro') / np.linalg.norm(J_model, 'fro')
    print(f"  Ridge         (lam={res_ridge.x[0]:.3f})          -> Value Error: {err_val_ridge:.4f}, Jac Error: {err_jac_ridge:.4f}")
    
    # 3. Nadaraya-Watson
    def obj_nw(params):
        ls = params[0]
        if ls <= 0: return np.inf
        nw = NadarayaWatson(X, lengthscale=ls)
        return np.linalg.norm(nw.predict(y) - m_y)
        
    res_nw = minimize(obj_nw, [1.0], bounds=[(1e-3, 10.0)])
    nw = NadarayaWatson(X, lengthscale=res_nw.x[0])
    err_val_nw = np.linalg.norm(nw.predict(y) - m_y) / np.linalg.norm(m_y)
    err_jac_nw = np.linalg.norm(nw.jacobian(y) - J_model, 'fro') / np.linalg.norm(J_model, 'fro')
    print(f"  NW            (ls={res_nw.x[0]:.3f})           -> Value Error: {err_val_nw:.4f}, Jac Error: {err_jac_nw:.4f}")
    
    # 4. 1-NN
    nn = OneNN(X)
    err_val_nn = np.linalg.norm(nn.predict(y) - m_y) / np.linalg.norm(m_y)
    err_jac_nn = np.linalg.norm(nn.jacobian(y) - J_model, 'fro') / np.linalg.norm(J_model, 'fro')
    print(f"  1-NN                                   -> Value Error: {err_val_nn:.4f}, Jac Error: {err_jac_nn:.4f}")
    
    report = f"""# E2.6 Mechanism Surrogate Fitting

**Model**: TabICL v2
**Description**: We fit published candidate mechanisms to TabICL v2's outputs on a test context. For each mechanism, we optimized its hyperparameters to minimize the Euclidean distance between its predictions and TabICL's predictions. We then evaluate whether the surrogate's Jacobian matches TabICL's Jacobian. A true mechanistic explanation must exhibit low derivative error bounded by the value-fit error.

## Results
- **ExactGP**: Value Error = {err_val_gp:.2%}, Jacobian Error = {err_jac_gp:.2%}
- **Ridge**: Value Error = {err_val_ridge:.2%}, Jacobian Error = {err_jac_ridge:.2%}
- **Nadaraya-Watson**: Value Error = {err_val_nw:.2%}, Jacobian Error = {err_jac_nw:.2%}
- **1-NN**: Value Error = {err_val_nn:.2%}, Jacobian Error = {err_jac_nn:.2%}

## Conclusion
"""
    if min(err_jac_gp, err_jac_ridge, err_jac_nw, err_jac_nn) > 0.5:
        report += "The Jacobian of TabICL v2 differs massively (over 50% relative error) from all fitted surrogates, even when the value-level fit appears plausible. This confirms that TabICL v2 is not implementing any of these standard smoothers or linear solvers. Its internal mechanism structurally departs from standard algorithms."
    else:
        report += "A mechanism closely matches TabICL's Jacobian, identifying its internal algorithm."
        
    out_path = Path(__file__).resolve().parent / "e2_6_report.md"
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport written to {out_path}")

if __name__ == '__main__':
    main()
