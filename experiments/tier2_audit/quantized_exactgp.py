import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from experiments.core.surrogates import ExactGP
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

def main():
    s = 42
    print(f"Running Quantized ExactGP for seed {s}")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
    Q = get_Q(y)
    
    t = 1e-2
    N = 10
    
    model = ExactGP(X, sigma=0.5)
    
    # 1000 bins mimicking TabPFN's BarDistribution
    def predict(y_eval):
        pred = model.predict(y_eval).flatten()
        return np.round(pred * 1000) / 1000.0

    n, m_dim = Q.shape
    J_cols = []
    C_cols = []
    
    m_base = predict(y)
    
    for j in range(m_dim):
        q_j = Q[:, j]
        m_plus = np.zeros(n)
        m_minus = np.zeros(n)
        np.random.seed(42 + j)
        noise = np.random.uniform(-6.87e-4, 6.87e-4, (N, n))
        for k in range(N):
            e = noise[k]
            m_plus += predict(y + e + t * q_j)
            m_minus += predict(y + e - t * q_j)
        m_plus /= N
        m_minus /= N
            
        J_cols.append((m_plus - m_minus) / (2 * t))
        C_cols.append(np.linalg.norm(m_plus - 2 * m_base + m_minus) / (t**2 * np.linalg.norm(m_base)))
        
    J_Q = np.column_stack(J_cols)
    J = Q.T @ J_Q
    
    mean_curv = np.mean(C_cols)
    norm_J = np.linalg.norm(J, ord='fro')
    
    print("--- C.4 / C.5 Results ---")
    print(f"Quantized ExactGP (t={t}, N={N} dither)")
    print(f"Measured Curvature: {mean_curv:.6e}")
    print(f"Measured ||J||_F: {norm_J:.6e}")
    
    # Analytic Unquantized Baseline for ||J||_F
    def predict_unquant(y_eval):
        return model.predict(y_eval).flatten()
        
    m_base_unq = predict_unquant(y)
    J_cols_unq = []
    C_cols_unq = []
    for j in range(m_dim):
        q_j = Q[:, j]
        m_plus = predict_unquant(y + t * q_j)
        m_minus = predict_unquant(y - t * q_j)
        J_cols_unq.append((m_plus - m_minus) / (2 * t))
        C_cols_unq.append(np.linalg.norm(m_plus - 2 * m_base_unq + m_minus) / (t**2 * np.linalg.norm(m_base_unq)))
        
    J_Q_unq = np.column_stack(J_cols_unq)
    J_unq = Q.T @ J_Q_unq
    norm_J_unq = np.linalg.norm(J_unq, ord='fro')
    curv_unq = np.mean(C_cols_unq)
    print(f"Analytic ExactGP (t={t}, no dither)")
    print(f"Analytic Curvature: {curv_unq:.6e}")
    print(f"Analytic ||J||_F: {norm_J_unq:.6e}")

if __name__ == "__main__":
    main()
