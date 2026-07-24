import os
import sys
import copy
import numpy as np
import torch

# Custom model paths
sys.path.insert(0, os.path.abspath("balef_repo/FoundationModels/TabPFN_v2/src"))
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("models"))
sys.path.append(os.path.abspath("prior_sampler"))
sys.path.append(os.path.abspath("solve_experiments"))

from tabicl_v2 import get_regressor as get_tabicl_regressor
from harness import solve_gcv_weights

class TabICLWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.clf = get_tabicl_regressor(device=self.device, n_estimators=1, n_jobs=1)
        self.clf.fit(np.random.randn(10, 2), np.random.randn(10))
        
        # Load finetuned decoders
        model_icl = self.clf.model_
        icl_dec_base = model_icl.icl_predictor.decoder
        self.tabicl_decoders = [copy.deepcopy(icl_dec_base).to(self.device) for _ in range(12)]
        icl_weights_path = "results/extras/tabicl_v2_reg_decoders.pt"
        if os.path.exists(icl_weights_path):
            icl_weights = torch.load(icl_weights_path, map_location=self.device)
            for l in range(12):
                self.tabicl_decoders[l].load_state_dict(icl_weights[l])

    def extract_task_artifacts(self, X_train, y_train, X_test, K_g=None):
        """
        Runs model inference on task and extracts layer-wise representations,
        predictions, and GCV-solved dual weights \alpha.
        """
        n = X_train.shape[0]
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-20
        y_scaled = (y_train - y_mean) / y_std
        
        self.clf.fit(X_train, y_scaled)
        model_icl = self.clf.model_
        
        activations = {}
        handles = []
        icl_blocks = list(model_icl.icl_predictor.tf_icl.blocks)
        
        for idx in range(12):
            def make_hook(layer_idx):
                def hook(mod, inp, out):
                    activations[f"layer_{layer_idx}"] = out.detach()
                return hook
            handles.append(icl_blocks[idx].register_forward_hook(make_hook(idx)))
            
        self.clf.predict(X_test)
        for h in handles:
            h.remove()
            
        support_embs = {}
        query_embs = {}
        predictions = {}
        alphas = {}
        
        for l in range(12):
            h_act = activations[f"layer_{l}"].to(self.device)
            h_norm = model_icl.icl_predictor.ln(h_act)
            
            # Extract representations
            h_np = h_norm.detach().cpu().numpy()
            if h_np.ndim == 3:
                if h_np.shape[1] == 1:
                    h_np = h_np[:, 0, :]
                elif h_np.shape[0] == 1:
                    h_np = h_np[0, :, :]
                else:
                    h_np = h_np[:, 0, :]
                    
            support_embs[l] = h_np[:n]
            query_embs[l] = h_np[n:]
            
            # Predict
            logits = self.tabicl_decoders[l](h_norm)
            dist = model_icl.quantile_dist(logits)
            preds_scaled = dist.quantiles.mean(dim=-1).squeeze(0).detach().cpu().numpy()
            preds_final = self.clf.y_scaler_.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            
            mu_unscaled = preds_final[n:]
            predictions[l] = mu_unscaled
            
            if K_g is not None:
                alpha_l, _, _ = solve_gcv_weights(K_g, mu_unscaled)
                alphas[l] = alpha_l
                
        return {
            "support_embs": support_embs,
            "query_embs": query_embs,
            "predictions": predictions,
            "alphas": alphas,
            "y_scaled": y_scaled,
            "y_mean": y_mean,
            "y_std": y_std
        }
