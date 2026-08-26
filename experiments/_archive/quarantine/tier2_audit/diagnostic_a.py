import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.phase_1.core.context import generate_audit_context
from sklearn.metrics import root_mean_squared_error

def get_Q(y):
    n = len(y)
    v1 = np.ones(n)
    v2 = y - np.mean(y)
    A = np.random.randn(n, n)
    A[:, 0] = v1
    A[:, 1] = v2
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]

def a2_tabpfn_staircase():
    print("="*40)
    print("A2: TabPFN v2 Staircase Measurement")
    print("="*40)
    
    seed = 42
    X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    model = load("tabpfn_v2", task="regression", device="cuda")
    
    Q = get_Q(y)
    q1 = Q[:, 0]
    
    # We want to scan m(y + t q1)
    ts = np.linspace(-0.05, 0.05, 501)
    
    y_base = y.copy()
    model.estimator.fit(X, y_base)
    base_pred = model.estimator.predict(X, output_type="mean")
    
    preds = []
    print("Sweeping t from -0.05 to 0.05 (501 points)...")
    for t in ts:
        y_pert = y_base + t * q1
        model.estimator.fit(X, y_pert)
        p = model.estimator.predict(X, output_type="mean")
        preds.append(p)
        
    preds = np.array(preds) # shape: (501, n)
    
    # Calculate jump sizes
    jumps = np.abs(np.diff(preds, axis=0))
    median_jump = np.median(jumps[jumps > 1e-8]) if np.any(jumps > 1e-8) else 0.0
    max_jump = np.max(jumps)
    
    print(f"Median jump size (effective step Δ): {median_jump:.4e}")
    print(f"Max jump size: {max_jump:.4e}")
    if median_jump < 1e-5:
        print("Trace is essentially smooth.")
    else:
        print("Trace exhibits distinct quantization jumps.")
        
    return {"median_jump": float(median_jump), "max_jump": float(max_jump)}

def a4_tabswift_radius():
    print("\n" + "="*40)
    print("A4: TabSwift In-Distribution Radius")
    print("="*40)
    
    seed = 42
    X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    
    # Generate X_test, y_test for true RMSE check
    np.random.seed(seed + 1)
    X_test = np.random.randn(50, 5)
    f_test = 0.5 * X_test[:, 0] + np.sin(X_test[:, 1]) - 0.2 * X_test[:, 2]**2
    y_test = f_test + np.random.randn(50) * 1.0
    
    model = load("tabswift", task="regression", device="cuda")
    model.estimator.fit(X, y)
    base_pred = model.estimator.predict(X_test)
    base_rmse = root_mean_squared_error(y_test, base_pred)
    print(f"Base RMSE (t=0): {base_rmse:.4f}")
    
    Q = get_Q(y)
    q1 = Q[:, 0]
    
    ts = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 2.0, 5.0, 10.0]
    
    radii = {}
    print("Sweeping t to find RMSE degradation...")
    for t in ts:
        y_pert = y + t * q1
        model.estimator.fit(X, y_pert)
        p = model.estimator.predict(X_test)
        rmse = root_mean_squared_error(y_test, p)
        degrad = (rmse - base_rmse) / base_rmse
        print(f"  t = {t:.4f} -> RMSE = {rmse:.4f} (Degradation: {degrad:+.2%})")
        radii[float(t)] = {"rmse": float(rmse), "degrad": float(degrad)}
        
    return radii

def a5_determinism(model_id):
    print(f"\n" + "="*40)
    print(f"A5: Determinism Check ({model_id})")
    print("="*40)
    
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    model = load(model_id, task="regression", device="cuda")
    
    # Run 1
    np.random.seed(0)
    model.estimator.fit(X, y)
    if model_id == "tabpfn_v2":
        p1 = model.estimator.predict(X, output_type="mean")
    else:
        p1 = model.estimator.predict(X)
        
    # Run 2
    np.random.seed(0)
    model.estimator.fit(X, y)
    if model_id == "tabpfn_v2":
        p2 = model.estimator.predict(X, output_type="mean")
    else:
        p2 = model.estimator.predict(X)
        
    diff = np.max(np.abs(p1 - p2))
    print(f"Max absolute difference between 2 identical runs: {diff:.4e}")
    return {"max_diff": float(diff)}

def a6_units(model_id):
    print(f"\n" + "="*40)
    print(f"A6: Units Check ({model_id})")
    print("="*40)
    
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    model = load(model_id, task="regression", device="cuda")
    model.estimator.fit(X, y)
    
    if model_id == "tabpfn_v2":
        m = model.estimator.predict(X, output_type="mean")
    else:
        m = model.estimator.predict(X)
        
    mean_y = np.mean(y)
    std_y = np.std(y)
    mean_m = np.mean(m)
    std_m = np.std(m)
    mean_diff = np.mean(np.abs(m - y))
    
    print(f"mean(y): {mean_y:.4f}, std(y): {std_y:.4f}")
    print(f"mean(m): {mean_m:.4f}, std(m): {std_m:.4f}")
    print(f"mean|m_i - y_i|: {mean_diff:.4f}")
    
    # Check forward pass dtype
    import torch
    dtype_found = None
    if model_id == "tabpfn_v2":
        dtype_found = model.estimator.models_[0].__class__.__name__ # Just approx
        print("DType checks require hooks, assuming standard for TabPFN.")
    
    return {
        "mean_y": float(mean_y), "std_y": float(std_y),
        "mean_m": float(mean_m), "std_m": float(std_m),
        "mean_abs_diff": float(mean_diff)
    }

def main():
    import json
    
    results = {}
    results["A2_TabPFN"] = a2_tabpfn_staircase()
    results["A4_TabSwift"] = a4_tabswift_radius()
    results["A5_TabPFN"] = a5_determinism("tabpfn_v2")
    results["A5_TabSwift"] = a5_determinism("tabswift")
    results["A6_TabPFN"] = a6_units("tabpfn_v2")
    results["A6_TabSwift"] = a6_units("tabswift")
    
    with open("experiments/tier2_audit/diagnostic_a_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nDone. Results saved to experiments/tier2_audit/diagnostic_a_results.json")

if __name__ == "__main__":
    main()
