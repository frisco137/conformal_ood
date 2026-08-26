import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP
from experiments.tier2_audit.audit_phase1_metrics import get_metrics_canonical, get_Q

class TargetedImitator:
    def __init__(self, X, y, seed=42):
        self.egp = ExactGP(X, sigma=0.5)
        J_gp = self.egp.jacobian(y)
        norm_J = np.linalg.norm(J_gp, 'fro')
        
        np.random.seed(seed)
        M = np.random.randn(len(y), len(y))
        M_anti = (M - M.T) / 2
        target_anti_norm = np.sqrt(0.09 / 0.91) * norm_J
        self.M_anti = M_anti * (target_anti_norm / np.linalg.norm(M_anti, 'fro'))
        
        v = np.random.randn(len(y))
        v /= np.linalg.norm(v)
        eigs_gp = np.linalg.eigvalsh((J_gp + J_gp.T) / 2)
        lam_min = np.min(eigs_gp)
        norm_J_2 = np.linalg.norm(J_gp, 2)
        
        # We want the *projected* negeig to be ~0.1, but we approximate it on the full J here.
        # c = 0.15 * norm_J_2 + lam_min (using 0.15 to account for Q projection losing some mass)
        c = 0.8 * norm_J_2 + max(0, lam_min)
        self.v = v
        self.c = c
        
    def inner_imitator(self, u_eval):
        return self.egp.predict(u_eval) + self.M_anti @ u_eval - (self.c / 2) * np.dot(self.v, u_eval) * self.v

    def predict(self, y_eval):
        mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
        return mean_y + std_y * self.inner_imitator((y_eval - mean_y) / std_y)
        
    def get_analytic_jacobian(self, y_eval):
        return self.egp.jacobian(y_eval) + self.M_anti - self.c * np.outer(self.v, self.v)

def compute_jacobian_imitator(model, y, Q, t, dither=False, N=10, delta=6.87e-4, seed=42):
    n, m_dim = Q.shape
    J_hat = np.zeros((m_dim, m_dim))
    
    for j in range(m_dim):
        q_j = Q[:, j]
        if dither:
            m_plus_sum = np.zeros(n)
            m_minus_sum = np.zeros(n)
            np.random.seed(seed + j)
            noise = np.random.uniform(-delta/2, delta/2, (N, n))
            for k in range(N):
                e = noise[k]
                m_plus_sum += model.predict(y + e + t * q_j)
                m_minus_sum += model.predict(y + e - t * q_j)
            v_j = (m_plus_sum - m_minus_sum) / (2 * t * N)
        else:
            m_plus = model.predict(y + t * q_j).flatten()
            m_minus = model.predict(y - t * q_j).flatten()
            v_j = (m_plus - m_minus) / (2 * t)
            
        J_hat[:, j] = (Q.T @ v_j).flatten()
    return J_hat

def main():
    seeds = [42, 100, 200, 300, 400]
    t = 1e-3
    
    results = {
        "analytic": {"asym": [], "negeig": [], "negfrac": []},
        "measured_nodither": {"asym": [], "negeig": [], "negfrac": []},
        "measured_dither": {"asym": [], "negeig": [], "negfrac": []},
    }
    
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        
        model = TargetedImitator(X, y, seed=seed)
        
        J_analytic_full = model.get_analytic_jacobian(y)
        J_analytic_proj = Q.T @ J_analytic_full @ Q
        a, ne, nf, _ = get_metrics_canonical(J_analytic_proj)
        results["analytic"]["asym"].append(a)
        results["analytic"]["negeig"].append(ne)
        results["analytic"]["negfrac"].append(nf)
        
        J_nodither = compute_jacobian_imitator(model, y, Q, t, dither=False, seed=seed)
        a_no, ne_no, nf_no, _ = get_metrics_canonical(J_nodither)
        results["measured_nodither"]["asym"].append(a_no)
        results["measured_nodither"]["negeig"].append(ne_no)
        results["measured_nodither"]["negfrac"].append(nf_no)
        
        J_dither = compute_jacobian_imitator(model, y, Q, t, dither=True, N=10, seed=seed)
        a_di, ne_di, nf_di, _ = get_metrics_canonical(J_dither)
        results["measured_dither"]["asym"].append(a_di)
        results["measured_dither"]["negeig"].append(ne_di)
        results["measured_dither"]["negfrac"].append(nf_di)
        
    for k, data in results.items():
        print(f"--- {k} ---")
        for mk in ['asym', 'negeig', 'negfrac']:
            mean = np.mean(data[mk])
            std = np.std(data[mk])
            print(f"  {mk}: {mean:.4f} +/- {std:.4f}")
            
    with open("experiments/tier2_audit/results_phase2_imitator.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
