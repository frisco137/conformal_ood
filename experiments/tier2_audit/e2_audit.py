import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import load
from experiments.core.context import generate_audit_context
from experiments.core.metrics import central_jacobian, asym, negeig

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
    report_lines = ["# Tier 2: Core Structural Audit Report\n"]
    
    for model_name in ["tabicl_v2", "tabpfn_v2", "tabswift"]:
        print("=" * 40)
        print(f"Testing Model: {model_name}")
        print("=" * 40)
        report_lines.append(f"## Model: {model_name}\n")
        
        # We average over 5 seeds as requested by standing rules
        n_seeds = 5
        seeds = [42, 100, 200, 300, 400]
        
        results = {
            "non_deg_ratio": [],
            "norm_J": [],
            "max_J": [],
            "negeig": [],
            "asym": []
        }
        
        for seed in seeds:
            print(f"  Running seed {seed}...")
            X, y, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
            y_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
            
            model = load(model_name, task="regression", device="cuda")
            
            def model_func(y_eval):
                trace = model.run(X, y_eval, X)
                return trace.pred
            
            # Step size handling
            h = 1e-3
            if model_name == "tabswift":
                h = 1e-1  # Due to derivative collapse at small h
            
            J = central_jacobian(model_func, y_std, h=h)
            
            # E2.1 Non-degeneracy
            norm_J = np.linalg.norm(J, ord='fro')
            norm_J_diff = np.linalg.norm(J - np.eye(100), ord='fro')
            non_deg_ratio = norm_J_diff / (norm_J + 1e-8)
            max_J = np.max(np.abs(J))
            
            results["non_deg_ratio"].append(non_deg_ratio)
            results["norm_J"].append(norm_J)
            results["max_J"].append(max_J)
            
            if model_name == "tabicl_v2":
                Q = get_Q(y_std)
                J_eval = Q.T @ J @ Q
                red_asym = asym(J_eval)
                red_neg = negeig(J_eval)
                results["asym"].append(red_asym)
                results["negeig"].append(red_neg)
            else:
                # Raw Jacobian since Q is mathematically invalid
                raw_asym = asym(J)
                raw_neg = negeig(J)
                results["asym"].append(raw_asym)
                results["negeig"].append(raw_neg)
                
        # Aggregate results
        mean_non_deg = np.mean(results["non_deg_ratio"])
        mean_norm_J = np.mean(results["norm_J"])
        mean_max_J = np.mean(results["max_J"])
        
        mean_asym = np.mean(results["asym"])
        spread_asym = np.max(results["asym"]) - np.min(results["asym"])
        
        mean_negeig = np.mean(results["negeig"])
        spread_negeig = np.max(results["negeig"]) - np.min(results["negeig"])
        
        print(f"\nE2.1 Non-degeneracy:")
        print(f"  norm(J-I)/norm(J) = {mean_non_deg:.4f} (Pass: >0.3)")
        print(f"  norm(J) = {mean_norm_J:.2f} (Expected ~10)")
        print(f"  max|J| = {mean_max_J:.4f}")
        
        report_lines.append(f"- **E2.1 Non-degeneracy**: Ratio = {mean_non_deg:.4f}, Norm = {mean_norm_J:.2f}, Max = {mean_max_J:.4f}")
        if mean_non_deg > 0.3 and mean_norm_J > 1.0:
            report_lines.append("  - Status: **PASS** (Model listens to context and isn't trivial 1-NN)")
        else:
            report_lines.append("  - Status: **FAIL / VACUOUS**")

        if model_name != "tabicl_v2":
            report_lines.append("- *(Note: TabPFN and TabSwift failed shift-equivariance (E0.6), so Q-projection is invalid. Reporting raw metrics.)*")
            
        print(f"\nE2.2 Positivity (A2):")
        print(f"  negeig = {mean_negeig:.4e} +/- {spread_negeig/2:.4e}")
        
        F = 1e-11 if model_name == "tabicl_v2" else 0.05
        thresh_pass_a2 = max(3*F, 0.02)
        thresh_fail_a2 = max(10*F, 0.05)
        
        status_a2 = "PASS" if mean_negeig <= thresh_pass_a2 else ("FAIL" if mean_negeig > thresh_fail_a2 else "INCONCLUSIVE")
        report_lines.append(f"- **E2.2 Positivity (A2)**: {status_a2} (negeig = {mean_negeig:.4e} ± {spread_negeig/2:.4e})")
        
        print(f"\nE2.3 Symmetry (A1):")
        print(f"  asym = {mean_asym:.4f} +/- {spread_asym/2:.4f}")
        
        thresh_pass_a1 = max(3*F, 0.05)
        thresh_fail_a1 = max(10*F, 0.15)
        
        status_a1 = "PASS" if mean_asym <= thresh_pass_a1 else ("FAIL" if mean_asym > thresh_fail_a1 else "INCONCLUSIVE")
        report_lines.append(f"- **E2.3 Symmetry (A1)**: {status_a1} (asym = {mean_asym:.4f} ± {spread_asym/2:.4f})")
        
        if model_name == "tabicl_v2" and status_a1 == "FAIL":
            print("\nE2.4 Profiling heteroscedastic scaling...")
            n_red = J_eval.shape[0]
            eqs = []
            b = []
            for i in range(n_red):
                for j in range(i+1, n_red):
                    if J_eval[i, j] > 1e-6 and J_eval[j, i] > 1e-6:
                        row = np.zeros(n_red)
                        row[i] = 1
                        row[j] = -1
                        eqs.append(row)
                        b.append(np.log(J_eval[i, j] / J_eval[j, i]))
            
            if len(eqs) > 0:
                eqs = np.array(eqs)
                b = np.array(b)
                d_log, _, _, _ = np.linalg.lstsq(eqs, b, rcond=None)
                d = np.exp(d_log)
                J_diag = J_eval @ np.diag(d)
                asym_prof = asym(J_diag)
                print(f"  asym_prof = {asym_prof:.4f}")
                status_prof = "RECOVERED" if asym_prof <= thresh_pass_a1 else "FAIL"
                report_lines.append(f"- **E2.4 Profiling**: {status_prof} (asym_prof = {asym_prof:.4f})")
            else:
                report_lines.append("- **E2.4 Profiling**: FAIL (Not enough positive entries to profile)")
        
        report_lines.append("\n")

    out_path = Path(__file__).resolve().parent / "tier2_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport written to {out_path}")

if __name__ == '__main__':
    main()
