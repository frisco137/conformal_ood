import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

_MODEL_CACHE = {}

from experiments.phase_1.core.surrogates import ExactGP, Imitator

def compute_reduced_jacobian(model_id, X, y, Q, t, dither=False, N=10, delta=6.87e-4):
    n, m_dim = Q.shape
    
    if model_id == "exact_gp":
        egp = ExactGP(X, sigma=0.5)
        def predict(y_eval):
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
        def predict(y_eval):
            mean_y, std_y = np.mean(y_eval), np.std(y_eval) + 1e-8
            return mean_y + std_y * inner_imitator((y_eval - mean_y) / std_y)
    else:
        if model_id not in _MODEL_CACHE:
            _MODEL_CACHE[model_id] = load(model_id, task="regression", device="cuda")
        model = _MODEL_CACHE[model_id]
        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
            
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

def get_metrics(J):
    """Canonical (asym, negeig, ||J||_F).

    CHUNK 1 FIX. This function previously returned
        asym   = ||(J - J^T)/2||_F / ||J||_F      (half the canonical value)
        negeig = #{lam < -1e-10} / dim            (the COUNT FRACTION, i.e.
                                                   negfrac, under the name negeig)
    Every caller of this function therefore reported negfrac as negeig:
    audit_e.py, audit_1_sweep_match.py, audit_2a_imitator.py,
    audit_2b_quantized_gp.py, audit_2c_to_2g_pipeline_stress.py,
    audit_4_gaps.py, rebuttal_2_3_unvoid.py, rebuttal_4_strong_imitator.py.
    The affected saved artifacts are audit_e_results.json and
    instrument_c_d_results.json; their "negeig" fields are retracted.
    """
    from experiments.phase_1.core.metrics import asym as _asym, negeig as _negeig
    norm_J = float(np.linalg.norm(J, 'fro'))
    if norm_J < 1e-12:
        return 0.0, 0.0, norm_J
    return float(_asym(J)), float(_negeig(J)), norm_J

def validate_model(model_id, t=1e-3, dither=False):
    print(f"\n" + "="*40)
    print(f"Validation: {model_id} (t={t}, dither={dither})")
    print("="*40)
    
    seed = 42
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    Q = get_Q(y)
    
    # C4: Richardson extrapolation
    J_t = compute_reduced_jacobian(model_id, X, y, Q, t, dither=dither)
    J_half = compute_reduced_jacobian(model_id, X, y, Q, t/2, dither=dither)
    
    J_richardson = (4/3) * J_half - (1/3) * J_t
    err = np.linalg.norm(J_t - J_half, 'fro')
    
    asym, negeig, norm_J = get_metrics(J_richardson)
    
    print(f"Norm(J): {norm_J:.4f}")
    print(f"Richardson Error ||J(t) - J(t/2)||: {err:.4e}")
    print(f"Asymmetry: {asym:.4f}")
    print(f"NegEig: {negeig:.4f}")
    
    return {
        "norm_J": float(norm_J),
        "err": float(err),
        "asym": float(asym),
        "negeig": float(negeig)
    }

def main():
    import json
    results = {}
    
    # D1: ExactGP (Expected: asym ~ 0, negeig ~ 0)
    results["D1_ExactGP"] = validate_model("exact_gp", t=1e-3, dither=False)
    
    # D2: Imitator (Expected: negeig > 0)
    results["D2_Imitator"] = validate_model("imitator", t=1e-3, dither=False)
    
    # D3: TabICL v2 (Expected: asym ~ 0.58, negeig ~ 0.176)
    results["D3_TabICL"] = validate_model("tabicl_v2", t=1e-3, dither=False)
    
    with open("experiments/tier2_audit/instrument_c_d_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nDone. Results saved to experiments/tier2_audit/instrument_c_d_results.json")
    
if __name__ == "__main__":
    main()
