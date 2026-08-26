import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.surrogates import ExactGP
from experiments.phase_1.core.context import generate_audit_context

class MockAnalytics:
    def __getattr__(self, name): return lambda *args, **kwargs: None
sys.modules['analytics'] = MockAnalytics()

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def get_quantized_egp(seed, X):
    class QuantizedMockWrapper:
        def __init__(self):
            self.egp = ExactGP(X, sigma=0.5)
            self.delta = 6.87e-4 # Approx TabPFN output scale quantization
        
        def predict(self, y_eval):
            # Normal ExactGP
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            out = mean_y + std_y * self.egp.predict((y_eval - mean_y) / std_y)
            # Quantize
            return np.round(out / self.delta) * self.delta

    class MockModel:
        def __init__(self):
            self.estimator = QuantizedMockWrapper()
            
    return MockModel()

def get_jacobian_and_curvature(X, y, Q, t, dither=False, N=10):
    model = get_quantized_egp(42, X)
    def predict(y_eval):
        return model.estimator.predict(y_eval).flatten()

    n, m_dim = Q.shape
    J_cols = []
    C_cols = []
    
    m_base = predict(y)
    
    for j in range(m_dim):
        q_j = Q[:, j]
        if dither:
            m_plus = np.zeros(n)
            m_minus = np.zeros(n)
            np.random.seed(42 + j)
            noise = np.random.uniform(-6.87e-4/2, 6.87e-4/2, (N, n)) # dither size matched to delta
            for k in range(N):
                e = noise[k]
                m_plus += predict(y + e + t * q_j)
                m_minus += predict(y + e - t * q_j)
            m_plus /= N
            m_minus /= N
        else:
            m_plus = predict(y + t * q_j).flatten()
            m_minus = predict(y - t * q_j).flatten()
            
        J_cols.append((m_plus - m_minus) / (2 * t))
        C_cols.append(np.linalg.norm(m_plus - 2 * m_base + m_minus) / (t**2 * np.linalg.norm(m_base)))
        
    J_Q = np.column_stack(J_cols)
    J = Q.T @ J_Q
    mean_curv = np.mean(C_cols)
    
    return J, mean_curv

def main():
    print("--- C.4 Quantized ExactGP Check ---")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    
    # We test it at t=1e-2, N=10 (same as TabPFN)
    t = 1e-2
    print(f"t = {t}, N=10, Dither=True")
    J, curv = get_jacobian_and_curvature(X, y, Q, t, dither=True, N=10)
    norm_J = np.linalg.norm(J)
    
    print(f"Quantized ExactGP Curvature: {curv:.4e}")
    print(f"Quantized ExactGP ||J||_F:   {norm_J:.4f}")
    
    # Analytic GP norm for reference
    class AnalyticMockWrapper:
        def __init__(self):
            self.egp = ExactGP(X, sigma=0.5)
        def predict(self, y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * self.egp.predict((y_eval - mean_y) / std_y)
    class MockModel2:
        def __init__(self):
            self.estimator = AnalyticMockWrapper()
            
    m2 = MockModel2()
    def p2(y_eval): return m2.estimator.predict(y_eval).flatten()
    m_base = p2(y)
    J_cols = []
    for j in range(Q.shape[1]):
        q_j = Q[:, j]
        m_plus = p2(y + t * q_j)
        m_minus = p2(y - t * q_j)
        J_cols.append((m_plus - m_minus)/(2*t))
    J2 = Q.T @ np.column_stack(J_cols)
    print(f"Analytic ExactGP ||J||_F:    {np.linalg.norm(J2):.4f}")
    
    out = {
        'quantized_egp_curvature': float(curv),
        'quantized_egp_norm': float(norm_J),
        'analytic_egp_norm': float(np.linalg.norm(J2))
    }
    with open('experiments/tier2_audit/results_c4.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
