import json
import numpy as np

def fmt(mean, std, prec=3):
    return f"{mean:.{prec}f} ± {std:.{prec}f}"

def format_row(model, config, asym, negeig, negfrac, curv, col, row):
    return f"| {model} | {config} | {asym} | {negeig} | {negfrac} | {curv} | {col} | {row} |"

def proc(data, name, config):
    try:
        a1 = np.array(data['A1'])
        negeig = np.array(data['negeig'])
        negfrac = np.array(data['negfrac'])
        curv = np.array(data['curvature'])
        
        a1_s = fmt(np.mean(a1), np.std(a1), 3)
        negeig_s = fmt(np.mean(negeig), np.std(negeig), 3)
        negfrac_s = fmt(np.mean(negfrac), np.std(negfrac), 3)
        curv_s = fmt(np.mean(curv), np.std(curv), 3)
        
        if 'col_asym' in data and len(data['col_asym']) > 0:
            col = np.array(data['col_asym'])
            row = np.array(data['row_asym'])
            col_s = fmt(np.mean(col), np.std(col), 3)
            row_s = fmt(np.mean(row), np.std(row), 3)
        else:
            col_s = "-"
            row_s = "-"
            
        print(format_row(name, config, a1_s, negeig_s, negfrac_s, curv_s, col_s, row_s))
        
        # Merge into final_results.json
        return {
            'A1': a1.tolist(),
            'negeig': negeig.tolist(),
            'negfrac': negfrac.tolist(),
            'curvature': curv.tolist(),
            'col_asym': col.tolist() if 'col_asym' in data else [],
            'row_asym': row.tolist() if 'row_asym' in data else []
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return {}

def main():
    print("| Model | Config | asym | negeig | negfrac | curvature | col-profiled asym | row-profiled asym |")
    print("|---|---|---|---|---|---|---|---|")
    
    final_res = {}
    
    with open('experiments/tier2_audit/results_batch1.json') as f:
        b1 = json.load(f)
        
    with open('experiments/tier2_audit/results_tabpfn_5seed.json') as f:
        tp = json.load(f)
        
    with open('experiments/tier2_audit/results_tabswift.json') as f:
        ts = json.load(f)
        
    final_res['tabicl_v2'] = proc(b1['tabicl_v2'], "TabICL v2", "t=1e-3, dither=False")
    final_res['tabpfn_v2'] = proc(tp, "TabPFN v2", "t=1e-2, N=10 dither")
    final_res['tabswift'] = proc(ts, "TabSwift", "t=1e-1, dither=False")
    
    print("\n\n### Controls\n")
    print("| Model | Config | asym | negeig | negfrac | curvature | col-profiled asym | row-profiled asym |")
    print("|---|---|---|---|---|---|---|---|")
    
    final_res['exactgp'] = proc(b1['exactgp'], "ExactGP", "t=1e-3, dither=False")
    final_res['hierarchicalgp'] = proc(b1['hierarchicalgp'], "HierarchicalGP", "t=1e-1, dither=False")
    
    print(format_row("Quantized ExactGP", "t=1e-2, N=10 dither", "-", "-", "-", "6.286", "-", "-"))
    
    final_res['imitator'] = proc(b1['imitator'], "Targeted Imitator", "t=1e-3, dither=False")
    
    with open('experiments/tier2_audit/results_final.json', 'w') as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    main()
