import sys
import numpy as np
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP, Imitator

def get_metrics_canonical(J):
    norm_J = np.linalg.norm(J, 'fro')
    if norm_J < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    
    J_sym = (J + J.T) / 2
    J_anti = (J - J.T) / 2
    
    asym = np.linalg.norm(J_anti, 'fro') / norm_J
    
    eigs = np.linalg.eigvalsh(J_sym)
    
    negfrac = np.sum(eigs < -1e-10) / len(eigs)
    
    min_eig = np.min(eigs)
    norm_J_2 = np.linalg.norm(J, 2)
    if min_eig < -1e-10:
        negeig_mag = -min_eig / norm_J_2
    else:
        negeig_mag = 0.0
        
    return float(asym), float(negeig_mag), float(negfrac), float(norm_J)

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

class StrongImitator:
    def __init__(self, X, y):
        self.X = X
        self.egp = ExactGP(X, sigma=0.5)
        n = len(y)
        np.random.seed(42)
        M = np.random.randn(n, n)
        self.M_anti = (M - M.T) / 2
        self.M_anti = self.M_anti * (3.0 / np.linalg.norm(self.M_anti, 'fro'))
        
        self.a = np.random.RandomState(42).randn(n)
        self.a = self.a - np.mean(self.a)
        self.epsilon = 0.5 
        y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        self.phase = np.dot(self.a, y_std) / self.epsilon
        
    def inner_imitator(self, u_eval):
        return self.egp.predict(u_eval) + self.M_anti @ u_eval + self.epsilon * np.sin(np.dot(self.a, u_eval) / self.epsilon - self.phase - np.pi) * self.a

    def predict(self, y_eval):
        mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
        return mean_y + std_y * self.inner_imitator((y_eval - mean_y) / std_y)

_MODEL_CACHE = {}

def get_model_predict_fn(model_id, X, y, delta=6.87e-4):
    n = len(y)
    if model_id == "exact_gp_unwrapped":
        egp = ExactGP(X, sigma=0.5)
        return egp.predict
    elif model_id == "exact_gp_wrapped":
        egp = ExactGP(X, sigma=0.5)
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * egp.predict((y_eval - mean_y) / std_y)
        return predict
    elif model_id == "quantized_exact_gp_wrapped":
        egp = ExactGP(X, sigma=0.5)
        def raw_predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * egp.predict((y_eval - mean_y) / std_y)
        def predict(y_eval):
            out = raw_predict(y_eval)
            return np.round(out / delta) * delta
        return predict
    elif model_id == "imitator_wrapped":
        egp = ExactGP(X, sigma=0.5)
        a = np.random.RandomState(42).randn(n)
        a = a - np.mean(a)
        epsilon = 0.01
        y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        phase = np.dot(a, y_std) / epsilon
        def inner_imitator(u_eval):
            return egp.predict(u_eval) + epsilon * np.sin(np.dot(a, u_eval) / epsilon - phase - np.pi) * a
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * inner_imitator((y_eval - mean_y) / std_y)
        return predict
    elif model_id == "strong_imitator_wrapped":
        model = StrongImitator(X, y)
        return model.predict
    else:
        if model_id not in _MODEL_CACHE:
            _MODEL_CACHE[model_id] = load(model_id, task="regression", device="cuda")
        model = _MODEL_CACHE[model_id]
        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
        return predict

def compute_jacobian(predict_fn, y, Q, t, dither=False, N=10, delta=6.87e-4, seed=42):
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
                m_plus_sum += predict_fn(y + e + t * q_j)
                m_minus_sum += predict_fn(y + e - t * q_j)
            v_j = (m_plus_sum - m_minus_sum) / (2 * t * N)
        else:
            m_plus = predict_fn(y + t * q_j).flatten()
            m_minus = predict_fn(y - t * q_j).flatten()
            v_j = (m_plus - m_minus) / (2 * t)
            
        J_hat[:, j] = (Q.T @ v_j).flatten()
    return J_hat

def run_evaluation(model_id, t, dither=False, is_richardson=False, N=10):
    seeds = [42, 100, 200, 300, 400]
    out = {"asym": [], "negeig": [], "negfrac": []}
    
    for seed in seeds:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
        Q = get_Q(y)
        predict_fn = get_model_predict_fn(model_id, X, y)
        
        if is_richardson:
            J_t = compute_jacobian(predict_fn, y, Q, t, dither, N, seed=seed)
            J_half = compute_jacobian(predict_fn, y, Q, t/2, dither, N, seed=seed)
            J = (4/3) * J_half - (1/3) * J_t
        else:
            J = compute_jacobian(predict_fn, y, Q, t, dither, N, seed=seed)
            
        a, ne, nf, _ = get_metrics_canonical(J)
        out["asym"].append(a)
        out["negeig"].append(ne)
        out["negfrac"].append(nf)
        
    return out

def main():
    configs = [
        ("exact_gp_unwrapped", 1e-3, False, True),
        ("exact_gp_wrapped", 1e-3, False, True),
        ("quantized_exact_gp_wrapped", 1e-3, False, False),
        ("quantized_exact_gp_wrapped", 1e-3, True, False),
        ("imitator_wrapped", 1e-3, False, False),
        ("imitator_wrapped", 1e-3, True, False),
        ("strong_imitator_wrapped", 1e-3, False, False),
        ("strong_imitator_wrapped", 1e-3, True, False),
        ("tabicl_v2", 1e-3, False, True),
        ("tabpfn_v2", 1e-2, True, False),
        ("tabswift", 1e-1, False, False)
    ]
    
    results = {}
    for model_id, t, dither, is_richardson in configs:
        key = f"{model_id}_t{t}_dither{dither}_richardson{is_richardson}"
        print(f"Running: {key}")
        res = run_evaluation(model_id, t, dither, is_richardson)
        results[key] = res
        print(f"  asym: {res['asym']}")
        print(f"  negeig: {res['negeig']}")
        print(f"  negfrac: {res['negfrac']}")
        
    with open("experiments/tier2_audit/results_phase1.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
