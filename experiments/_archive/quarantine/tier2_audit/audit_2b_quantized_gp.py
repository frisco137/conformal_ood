import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics
from experiments.phase_1.core.surrogates import ExactGP, Imitator

# 2b. Quantized GP and Imitator
def compute_reduced_jacobian_quantized(model_id, X, y, Q, t, dither=False, N=10, delta=6.87e-4):
    n, m_dim = Q.shape
    
    if model_id == "exact_gp":
        egp = ExactGP(X, sigma=0.5)
        def raw_predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * egp.predict((y_eval - mean_y) / std_y)
    elif model_id == "imitator":
        egp = ExactGP(X, sigma=0.5)
        a = np.random.RandomState(42).randn(n)
        a = a - np.mean(a)
        epsilon = 0.01
        y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        phase = np.dot(a, y_std) / epsilon
        def inner_imitator(u_eval):
            return egp.predict(u_eval) + epsilon * np.sin(np.dot(a, u_eval) / epsilon - phase - np.pi) * a
        def raw_predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * inner_imitator((y_eval - mean_y) / std_y)
            
    def predict(y_eval):
        # Quantize the output with step delta
        out = raw_predict(y_eval)
        return np.round(out / delta) * delta
        
    J_hat = np.zeros((m_dim, m_dim))
    
    for j in range(m_dim):
        q_j = Q[:, j]
        
        if dither:
            m_plus_sum = np.zeros(n)
            m_minus_sum = np.zeros(n)
            # Uniform noise in [-delta/2, delta/2]
            np.random.seed(42 + j)
            noise = np.random.uniform(-delta/2, delta/2, (N, n))
            for k in range(N):
                e = noise[k]
                m_plus_sum += predict(y + e + t * q_j)
                m_minus_sum += predict(y + e - t * q_j)
                
            v_j = (m_plus_sum - m_minus_sum) / (2 * t * N)
        else:
            m_plus = predict(y + t * q_j).flatten()
            m_minus = predict(y - t * q_j).flatten()
            v_j = (m_plus - m_minus) / (2 * t)
            
        J_hat[:, j] = (Q.T @ v_j).flatten()
        
    return J_hat

def validate_model(model_id):
    seeds = [42, 100, 200, 300, 400]
    t = 1e-3
    delta = 6.87e-4
    
    nodither_negs = []
    dither_negs = []
    
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        
        J_no = compute_reduced_jacobian_quantized(model_id, X, y, Q, t, dither=False, delta=delta)
        _, n_no, _ = get_metrics(J_no)
        nodither_negs.append(n_no)
        
        J_di = compute_reduced_jacobian_quantized(model_id, X, y, Q, t, dither=True, N=10, delta=delta)
        _, n_di, _ = get_metrics(J_di)
        dither_negs.append(n_di)
        
    n_no_m, n_no_s = np.mean(nodither_negs), np.std(nodither_negs)
    n_di_m, n_di_s = np.mean(dither_negs), np.std(dither_negs)
    
    print(f"{model_id.upper()}:")
    print(f"  No Dither: negeig={n_no_m:.4f} +/- {n_no_s:.4f}")
    print(f"  Dithered:  negeig={n_di_m:.4f} +/- {n_di_s:.4f}")

def main():
    print("Running 2b: Quantized Surrogate validation")
    validate_model("exact_gp")
    validate_model("imitator")

if __name__ == "__main__":
    main()
