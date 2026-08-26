import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models import load
from experiments.phase_1.core.context import generate_context
from experiments.phase_1.core.metrics import central_jacobian, asym, negeig
from experiments.phase_1.core.surrogates import ExactGP, Imitator

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
    report_lines = ["# Tier 1: Preprocessing Confounds Report\n"]
    
    print("=" * 40)
    print("Tier 1: Preprocessing Confounds")
    print("=" * 40)
    
    X, y, f = generate_context(n=50, d=5, seed=42)
    y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    print("Loading TabICL v2...")
    model = load("tabicl_v2", task="regression", device="cuda")
    
    def model_func(y_eval):
        trace = model.run(X, y_eval, X)
        return trace.pred
        
    print("\n--- E1.1 Preprocessing identification ---")
    m_y = model_func(y_std)
    
    shift_errs = []
    for c in [1.0, 5.0, -2.0]:
        m_shifted = model_func(y_std + c)
        err = np.max(np.abs(m_shifted - (m_y + c))) / (np.max(np.abs(m_y + c)) + 1e-8)
        shift_errs.append(err)
        
    scale_errs = []
    for c in [0.5, 2.0, 10.0]:
        m_scaled = model_func(c * y_std)
        err = np.max(np.abs(m_scaled - c * m_y)) / (np.max(np.abs(c * m_y)) + 1e-8)
        scale_errs.append(err)
        
    max_shift_err = max(shift_errs)
    max_scale_err = max(scale_errs)
    
    pass_e11 = max_shift_err < 1e-3 and max_scale_err < 1e-3
    status = "PASS" if pass_e11 else "FAIL"
    res = f"E1.1 {status}: shift_err={max_shift_err:.2e}, scale_err={max_scale_err:.2e}"
    print(res)
    report_lines.append(f"- **E1.1 Preprocessing Identification**: {status} (shift_err={max_shift_err:.2e}, scale_err={max_scale_err:.2e})")
    
    print("\n--- E1.3 Ensemble-of-one ---")
    n_est = getattr(model.estimator, 'n_estimators', None)
    status_e13 = "PASS" if n_est == 1 else ("FAIL" if n_est is not None else "UNKNOWN")
    res_e13 = f"E1.3 {status_e13}: n_estimators = {n_est}"
    print(res_e13)
    report_lines.append(f"- **E1.3 Ensemble-of-One**: {status_e13} (n_estimators={n_est})")

    print("\n--- E1.5 M0 radial/tangential gate (Euler defect) ---")
    h = 1e-4
    m_plus = model_func(y_std + h * y_std)
    m_minus = model_func(y_std - h * y_std)
    D_y = (m_plus - m_minus) / (2 * h)
    
    e_y = D_y[0] - m_y[0]
    e_ratio = abs(e_y) / (abs(m_y[0]) + 1e-8)
    
    print(f"E1.5 Euler defect ratio = {e_ratio:.4e}")
    report_lines.append(f"- **E1.5 Euler Defect**: e_ratio = {e_ratio:.4e}")
    
    print("\n--- E1.2 Projection validation ---")
    # For instrument validation, use a noisy context so the GP prediction deviates from y,
    # which creates a large raw asymmetry confound.
    X_noisy, y_noisy, _ = generate_context(n=50, d=5, sigma=0.5, seed=100)
    y_noisy_std = (y_noisy - np.mean(y_noisy)) / (np.std(y_noisy) + 1e-8)
    
    egp = ExactGP(X_noisy, sigma=0.5)
    
    def W(g, y_in):
        mean_y = np.mean(y_in)
        std_y = np.std(y_in)
        if std_y < 1e-8: std_y = 1e-8
        u = (y_in - mean_y) / std_y
        return mean_y + std_y * g(u)
        
    def normalized_egp(y_eval):
        return W(egp.predict, y_eval)
        
    J_norm_egp = central_jacobian(normalized_egp, y_noisy_std, h=1e-4)
    raw_asym_egp = asym(J_norm_egp)
    
    Q = get_Q(y_noisy_std)
    J_red_egp = Q.T @ J_norm_egp @ Q
    red_asym_egp = asym(J_red_egp)
    
    status_egp = "PASS" if raw_asym_egp > 0.01 and red_asym_egp < 0.02 else "FAIL"
    res_egp = f"E1.2 (a) ExactGP {status_egp}: raw_asym={raw_asym_egp:.4f} (confound), red_asym={red_asym_egp:.4e} (resolved)"
    print(res_egp)
    report_lines.append(f"- **E1.2 (a) Projection on GP**: {status_egp} (raw_asym={raw_asym_egp:.4f}, red_asym={red_asym_egp:.4e})")
    
    # Custom Imitator that guarantees a negative eigenvalue in the projected subspace
    np.random.seed(40)
    a = np.random.randn(50)
    # Make a orthogonal to the standardizer subspace to ensure it survives Q
    a -= np.dot(a, np.ones(50))/50 * np.ones(50)
    v2 = y_noisy_std - np.mean(y_noisy_std)
    a -= np.dot(a, v2)/np.dot(v2, v2) * v2
    a /= np.linalg.norm(a)
    
    epsilon = 0.01
    # We want cos(a.T y / epsilon) to be negative at y_noisy_std. 
    # Let's shift it so it's exactly -1.
    phase = np.dot(a, y_noisy_std) / epsilon
    
    def inner_imitator(u_eval):
        # We subtract phase to make cos(0 - np.pi) = -1
        return egp.predict(u_eval) + epsilon * np.sin(np.dot(a, u_eval) / epsilon - phase - np.pi) * a
        
    def normalized_imitator(y_eval):
        return W(inner_imitator, y_eval)
        
    J_norm_im = central_jacobian(normalized_imitator, y_noisy_std, h=1e-4)
    J_red_im = Q.T @ J_norm_im @ Q
    red_negeig_im = negeig(J_red_im)
    
    A_floor_norm = red_asym_egp
    thresh_b = max(10 * A_floor_norm, 1e-6)
    
    status_im = "PASS" if red_negeig_im >= thresh_b else "FAIL"
    res_im = f"E1.2 (b) Imitator {status_im}: red_negeig={red_negeig_im:.4e} (threshold: {thresh_b:.2e})"
    print(res_im)
    report_lines.append(f"- **E1.2 (b) Projection on Imitator**: {status_im} (red_negeig={red_negeig_im:.4e} >= {thresh_b:.2e})")
    
    out_path = Path(__file__).resolve().parent / "tier1_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport written to {out_path}")

if __name__ == '__main__':
    main()
