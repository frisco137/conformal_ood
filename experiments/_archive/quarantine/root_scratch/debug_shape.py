import os
import sys
import torch
import numpy as np

sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood")
from experiments.q4_noise.phase3_probing import extract_features, get_context_rows

def custom_get_context_rows(model_name, layer, trace, n_prefix):
    if "tabpfn" in model_name:
        res = trace.resid_full[layer]
    else:
        res = trace.resid[layer]
    
    print(f"Layer {layer}: initial res shape = {res.shape}")
    
    if len(res.shape) == 4:
        res = res[0]
    if len(res.shape) == 3:
        if res.shape[0] == 1:
            res = res[0]
        elif res.shape[1] == 1:
            res = res[:, 0, :]
            
    print(f"Layer {layer}: after squeeze res shape = {res.shape}")
            
    if "tabpfn" in model_name:
        return res[n_prefix:n_prefix+128]
    elif model_name == "tabicl_v2":
        return res[:128]
    elif model_name == "tabswift":
        return res[64:64+128]

import experiments.q4_noise.phase3_probing
experiments.q4_noise.phase3_probing.get_context_rows = custom_get_context_rows

print("Extracting a few tasks for tabpfn_v2 to check shapes...")
feat_real, labels = extract_features('tabpfn_v2', random_clone=False)
