import sys
import os
import json
import numpy as np
import scipy.linalg
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to allow importing models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.synthetic import generate_audit_context
from experiments.core.surrogates import ExactGP, HierarchicalGP, Imitator
from experiments.tier2_audit.posthog import mock_posthog
mock_posthog()

def get_Q(y):
    n = len(y)
    ones = np.ones((n, 1)) / np.sqrt(n)
    U, _, _ = np.linalg.svd(ones, full_matrices=True)
    return U[:, 1:]

def profile_jacobian(J, mode="column"):
    n = J.shape[0]
    A = []
    b = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(J[i, j]) > 1e-10 and abs(J[j, i]) > 1e-10:
                row = np.zeros(n)
                row[i] = 1
                row[j] = -1 if mode == "column" else 1
                A.append(row)
                target = np.log(abs(J[j, i])) - np.log(abs(J[i, j]))
                if mode == "row":
                    target = np.log(abs(J[i, j])) - np.log(abs(J[j, i]))
                b.append(target)
    if len(A) == 0:
        return 0.0
    A = np.array(A)
    b = np.array(b)
    s_log, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    s = np.exp(s_log)
    S = np.diag(s)
    
    if mode == "column":
        J_prof = J @ S
    else:
        J_prof = S @ J
        
    norm_J = np.linalg.norm(J_prof, 'fro')
    norm_anti = np.linalg.norm((J_prof - J_prof.T) / 2, 'fro')
    return norm_anti / norm_J if norm_J > 0 else 0.0

def compute_metrics(J):
    norm_J = np.linalg.norm(J, 'fro')
    J_anti = (J - J.T) / 2
    norm_anti = np.linalg.norm(J_anti, 'fro')
    A1 = norm_anti / norm_J if norm_J > 0 else 0
    
    J_sym = (J + J.T) / 2
    eigs = np.linalg.eigvalsh(J_sym)
    neg_eigs = eigs[eigs < -1e-10]
    
    negfrac = np.sum(neg_eigs) / np.sum(eigs) if len(neg_eigs) > 0 and np.sum(eigs) != 0 else 0.0
    
    norm_J2 = np.linalg.norm(J, 2)
    negeig = abs(min(0.0, np.min(eigs))) / norm_J2 if norm_J2 > 0 else 0.0
    
    return A1, negeig, negfrac, J_sym, J_anti

def run_model(model_id, t, dither=False, N_dither=10, eps=1e-3, seeds=[42, 100, 200, 300, 400], is_quantized_gp=False):
    res = {'A1': [], 'negeig': [], 'negfrac': [], 'col_asym': [], 'row_asym': [], 'curvature': []}
    
    print(f"Loading {model_id}...")
    model = None
    if model_id not in ["exactgp", "hierarchicalgp", "imitator", "quantized_exactgp"]:
        model = load(model_id, task="regression", device="cuda")
        
    for s in seeds:
        print(f"  Seed {s}...")
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y)
        
        # Internal predict function
        if model_id == "exactgp":
            est = ExactGP(np.zeros(len(y)), sigma=0.5)
            def base_predict(y_eval): return est.predict(y_eval).flatten()
        elif model_id == "hierarchicalgp":
            est = HierarchicalGP(np.zeros(len(y)))
            def base_predict(y_eval): return est.predict(y_eval).flatten()
        elif model_id == "imitator":
            est = Imitator(np.zeros(len(y)), sigma=0.5)
            def base_predict(y_eval): return est.predict(y_eval).flatten()
        elif model_id == "quantized_exactgp":
            est = ExactGP(np.zeros(len(y)), sigma=0.5)
            def base_predict(y_eval): 
                # TabPFN bar distribution uses 1000 bins, ~0.01 bucket size
                return np.round(est.predict(y_eval).flatten() / 0.01) * 0.01
        else:
            def base_predict(y_eval):
                model.estimator.fit(X, y_eval)
                if model_id == "tabpfn_v2":
                    return model.estimator.predict(X, output_type="mean").flatten()
                return model.estimator.predict(X).flatten()
                
        def predict(y_eval):
            if not dither:
                return base_predict(y_eval)
            np.random.seed(s) # deterministic jitter per context
            preds = []
            for _ in range(N_dither):
                jitter = np.random.uniform(-eps, eps, size=y_eval.shape)
                preds.append(base_predict(y_eval + jitter))
            return np.mean(preds, axis=0)

        n, m_dim = Q.shape
        J_cols = []
        C_cols = []
        
        m_base = predict(y)
        
        for j in range(m_dim):
            q_j = Q[:, j]
            m_plus = predict(y + t * q_j)
            m_minus = predict(y - t * q_j)
            
            J_cols.append((m_plus - m_minus) / (2 * t))
            C_cols.append(np.linalg.norm(m_plus - 2 * m_base + m_minus) / (t**2 * np.linalg.norm(m_base)))
            
        J_Q = np.column_stack(J_cols)
        J = Q.T @ J_Q
        mean_curv = np.mean(C_cols)
        
        A1, negeig, negfrac, _, _ = compute_metrics(J)
        
        res['A1'].append(A1)
        res['negeig'].append(negeig)
        res['negfrac'].append(negfrac)
        res['col_asym'].append(profile_jacobian(J, "column"))
        res['row_asym'].append(profile_jacobian(J, "row"))
        res['curvature'].append(mean_curv)
        
    return res

def main():
    final_results = {}
    
    # 1. ExactGP (t=1e-3, Dither=False)
    final_results['exactgp'] = run_model("exactgp", t=1e-3, dither=False)
    
    # 2. HierarchicalGP (t=1e-3, Dither=False)
    final_results['hierarchicalgp'] = run_model("hierarchicalgp", t=1e-3, dither=False)
    
    # 3. Targeted Imitator (t=1e-3, Dither=False)
    # Note: Imitator was fixed in phase 2 to not have hardcoded seed, but we just measure it.
    final_results['imitator'] = run_model("imitator", t=1e-3, dither=False)
    
    # 4. TabICL v2 (t=1e-3, Dither=False)
    final_results['tabicl_v2'] = run_model("tabicl_v2", t=1e-3, dither=False)
    
    # 5. TabPFN v2 (t=1e-2, Dither=True, N=10)
    final_results['tabpfn_v2'] = run_model("tabpfn_v2", t=1e-2, dither=True, N_dither=10, eps=1e-3)
    
    # 6. TabSwift (t=1e-1, Dither=False) - FIX 4
    final_results['tabswift'] = run_model("tabswift", t=1e-1, dither=False)
    
    # 7. Quantized ExactGP control (t=1e-2, Dither=True, N=10) - FIX 3
    final_results['quantized_exactgp'] = run_model("quantized_exactgp", t=1e-2, dither=True, N_dither=10, eps=1e-3)
    
    with open('results_final_fixed.json', 'w') as f:
        json.dump(final_results, f, indent=2)
        
    print("ALL DONE. SAVED TO results_final_fixed.json")

if __name__ == "__main__":
    main()
