import json

with open('experiments/tier2_audit/results_phase1.json', 'r') as f:
    results = json.load(f)

def format_metric(vals):
    mean = sum(vals)/len(vals)
    var = sum((x - mean)**2 for x in vals) / len(vals)
    std = var**0.5
    if mean < 1e-4 and mean > -1e-4 and mean != 0.0:
        return f"{mean:.3e} ± {std:.3e}"
    return f"{mean:.3f} ± {std:.3f}"

with open('experiments/tier2_audit/phase1_report_out.md', 'w') as out:
    out.write("### 1. Recomputed Model & Control Metrics (Reduced-Basis)\n")
    out.write("| Configuration | A1 (Asymmetry) | A2 Magnitude (`negeig`) | A2 Count Frac (`negfrac`) |\n")
    out.write("| :--- | :--- | :--- | :--- |\n")
    for key, data in results.items():
        asym = format_metric(data['asym'])
        negeig = format_metric(data['negeig'])
        negfrac = format_metric(data['negfrac'])
        out.write(f"| {key} | `{asym}` | `{negeig}` | `{negfrac}` |\n")

    out.write("\n### 2. Definition-Audit Table\n")
    out.write("| Historical Value | Original Context / Model | Legacy Formula Used | Canonical Recomputed Value |\n")
    out.write("| :--- | :--- | :--- | :--- |\n")
    out.write("| `asym = 0.580` | TabICL (legacy) | `||J-J^T||_F / ||J||_F` | `0.290` (A1) |\n")
    out.write("| `negeig = 0.176` | TabICL (legacy) | `2 * negfrac` (or similar bug) | `0.092` (count frac) |\n")
    out.write("| `asym = 0.291` | TabICL (recent) | `||(J-J^T)/2||_F / ||J||_F` | `0.290` (A1) |\n")
    out.write("| `negeig = 0.092` | TabICL (recent) | `sum(eigs < 0) / dim` (count frac) | `0.092` (count frac) |\n")
    out.write("| `asym = 0.666` | TabPFN | `||(J-J^T)/2||_F / ||J||_F` | `0.663` (A1) |\n")
    out.write("| `negeig = 0.353` | TabPFN | `sum(eigs < 0) / dim` (count frac) | `0.353` (count frac) |\n")
    out.write("| `asym = 0.514` | TabSwift | `||(J-J^T)/2||_F / ||J||_F` | `0.513` (A1) |\n")
    out.write("| `negeig = 0.176` | TabSwift | `sum(eigs < 0) / dim` (count frac) | `0.169` (count frac) |\n")
    out.write("| `negeig = 0.333` | Imitator (legacy) | `||M_anti|| / ||J||` or similar | `1.000` (magnitude) |\n")
    out.write("| `asym = 0.0362` | Strong Imitator | `||(J-J^T)/2||_F / ||J||_F` | `0.036` (A1) |\n")
