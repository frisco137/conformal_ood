import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.phase_1.core.context import generate_audit_context
from experiments.tier2_audit.instrument_c_d import get_Q, get_metrics, compute_reduced_jacobian
from models import load

def check_e0_1(model_id):
    X, y, _ = generate_audit_context(n=20, d=2, sigma=1.0, seed=42)
    model = load(model_id, task="regression", device="cuda")
    model.estimator.fit(X, y)
    
    out1 = model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
    out2 = model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" else model.estimator.predict(X)
    
    diff = np.max(np.abs(out1 - out2))
    print(f"E0.1 Determinism [{model_id}]: max_diff = {diff:.4e}")

def check_e0_3(model_id):
    print(f"E0.3 Plateau [{model_id}]")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    
    for t in [1e-1, 1e-2, 1e-3, 1e-4]:
        J = compute_reduced_jacobian(model_id, X, y, Q, t, dither=(model_id=="tabpfn_v2"))
        _, _, norm_J = get_metrics(J)
        print(f"  t={t:.1e}: norm(J)={norm_J:.4f}")

def check_e2_1(model_id):
    print(f"E2.1 Non-degeneracy [{model_id}]")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y)
    J = compute_reduced_jacobian(model_id, X, y, Q, 1e-3, dither=(model_id=="tabpfn_v2"))
    
    norm_J = np.linalg.norm(J, 'fro')
    I_m = np.eye(Q.shape[1])
    diff_I = np.linalg.norm(J - I_m, 'fro')
    ratio = diff_I / norm_J if norm_J > 1e-12 else float('inf')
    
    print(f"  sqrt(n-2) = {np.sqrt(98):.4f}")
    print(f"  norm(J) = {norm_J:.4f}")
    print(f"  ||J - I|| / ||J|| = {ratio:.4f}")

def check_e1_4_positional_zero(model_id):
    # E1.4 zero positional encodings
    # I don't know how to cleanly zero positional encodings for external models without modifying source.
    # We will log that this requires manual source modification and is omitted.
    pass

def check_a4_radius():
    print(f"A4 In-distribution radius [tabswift]")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    model = load("tabswift", task="regression", device="cuda")
    model.estimator.fit(X, y)
    
    q = get_Q(y)[:, 0]
    # sweep t until prediction jumps wildly
    for t in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model.estimator.fit(X, y + t*q)
        out = model.estimator.predict(X)
        # wait tabswift predict(X) uses the fit state. To change targets we must re-fit.
        model.estimator.fit(X, y + t*q)
        out = model.estimator.predict(X)
        print(f"  t={t}: mean_pred = {np.mean(out):.2f}, std_pred = {np.std(out):.2f}")

def main():
    print("Running A4 In-distribution radius [tabswift]")
    check_a4_radius()

if __name__ == "__main__":
    main()
