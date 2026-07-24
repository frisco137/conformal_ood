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

from tabpfn_v2.regressor import TabPFNRegressor
import tabpfn_v2.model.multi_head_attention as mha
from harness import solve_gcv_weights

# Global container for attention hooking
ATTN_HOOK_DATA = {
    "current_layer": None,
    "in_items_attn": False,
    "cache": {}  # layer -> (n, n) attention matrix
}

orig_compute_attention_heads = mha.MultiHeadAttention.compute_attention_heads

@staticmethod
def patched_compute_attention_heads(q, k, v, kv, qkv, dropout_p=None, softmax_scale=None):
    if qkv is not None:
        q_proj, k_proj, v_proj = qkv.unbind(dim=-3)
    elif kv is not None:
        k_proj, v_proj = kv.unbind(dim=-3)
        q_proj = q
    else:
        q_proj, k_proj, v_proj = q, k, v
        
    nhead = q_proj.shape[2]
    nhead_kv = k_proj.shape[2]
    share_kv_across_n_heads = nhead // nhead_kv
    
    k_proj = mha.MultiHeadAttention.broadcast_kv_across_heads(k_proj, share_kv_across_n_heads)
    v_proj = mha.MultiHeadAttention.broadcast_kv_across_heads(v_proj, share_kv_across_n_heads)
    
    d_k = q_proj.shape[-1]
    logits = torch.einsum("b q h d, b k h d -> b q k h", q_proj, k_proj)
    scale = softmax_scale if softmax_scale is not None else 1.0 / np.sqrt(d_k)
    logits *= scale
    
    ps = torch.softmax(logits, dim=2)
    
    curr_layer = ATTN_HOOK_DATA.get("current_layer", None)
    if ATTN_HOOK_DATA.get("in_items_attn", False) and curr_layer is not None:
        # Check if support-support attention (Q == K)
        if ps.shape[1] == ps.shape[2]:
            avg_ps = ps.mean(dim=-1).detach().cpu().numpy()
            ATTN_HOOK_DATA["cache"][curr_layer] = avg_ps[0] # (n, n)
            
    attention_head_outputs = torch.einsum("b q k h, b k h d -> b q h d", ps, v_proj)
    return attention_head_outputs

mha.MultiHeadAttention.compute_attention_heads = patched_compute_attention_heads

def register_attention_hooks(model):
    hooks = []
    for l in range(12):
        def make_pre_hook(layer_idx):
            def pre_hook(module, inp):
                ATTN_HOOK_DATA["current_layer"] = layer_idx
                ATTN_HOOK_DATA["in_items_attn"] = True
            return pre_hook
            
        def make_post_hook(layer_idx):
            def post_hook(module, inp, out):
                ATTN_HOOK_DATA["in_items_attn"] = False
            return post_hook
            
        self_attn = model.transformer_encoder.layers[l].self_attn_between_items
        hooks.append(self_attn.register_forward_pre_hook(make_pre_hook(l)))
        hooks.append(self_attn.register_forward_hook(make_post_hook(l)))
    return hooks

class TabPFNWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        layers_info = [(l, {
            "w_attn_between_features_on_query": 1.0,
            "w_attn_between_features_on_support": 1.0,
            "w_attn_between_items_on_query": 1.0,
            "w_attn_between_items_on_support": 1.0,
            "w_mlp_on_support": 1.0,
            "w_mlp_on_query": 1.0,
        }) for l in range(12)]
        
        self.clf = TabPFNRegressor(
            n_estimators=1,
            ignore_pretraining_limits=True,
            device=self.device,
            layers_info=layers_info,
            n_jobs=1
        )
        self.clf.fit(np.random.randn(10, 2), np.random.randn(10))
        self.clf.model_.to(self.device)
        
        # Load finetuned decoders
        pfn_dec_base = self.clf.get_decoder()
        self.tabpfn_decoders = [copy.deepcopy(pfn_dec_base).to(self.device) for _ in range(12)]
        pfn_weights_path = "results/extras/tabpfn_v2_reg_decoders.pt"
        if os.path.exists(pfn_weights_path):
            pfn_weights = torch.load(pfn_weights_path, map_location=self.device)
            for l in range(12):
                self.tabpfn_decoders[l].load_state_dict(pfn_weights[l])
        self.clf.finetuned_decoders = self.tabpfn_decoders

    def extract_task_artifacts(self, X_train, y_train, X_test, K_g=None):
        """
        Runs model inference on task and extracts layer-wise representations,
        predictions, GCV-solved dual weights \alpha, and item attention matrices.
        """
        n = X_train.shape[0]
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-20
        y_scaled = (y_train - y_mean) / y_std
        
        ATTN_HOOK_DATA["cache"].clear()
        
        self.clf.fit(X_train, y_scaled)
        self.clf.finetuned_decoders = self.tabpfn_decoders
        hooks = register_attention_hooks(self.clf.model_)
        self.clf.predict(X_test)
        for h in hooks:
            h.remove()
            
        embeddings = self.clf.get_all_layers_embeddings()
        preds_all, _ = self.clf.get_all_layers_predictions(decoder_type="finetuned")
        
        support_embs = {}
        query_embs = {}
        predictions = {}
        alphas = {}
        item_attn = {}
        
        for l in range(12):
            emb = embeddings[l]
            # Convert embedding to numpy (shape: batch_size, seq_len, hidden) or (seq_len, hidden)
            if isinstance(emb, torch.Tensor):
                emb_np = emb.detach().cpu().numpy()
            else:
                emb_np = np.array(emb)
                
            if emb_np.ndim == 4:
                if emb_np.shape[0] == 1:
                    emb_np = emb_np[0] # (seq_len, num_feat, hidden)
                elif emb_np.shape[1] == 1:
                    emb_np = emb_np[:, 0]
            elif emb_np.ndim == 3:
                if emb_np.shape[1] == 1:
                    emb_np = emb_np[:, 0, :] # (seq_len, hidden)
                elif emb_np.shape[0] == 1:
                    emb_np = emb_np[0, :, :] # (seq_len, hidden)
                else:
                    emb_np = emb_np[:, 0, :]
                    
            # support vs query tokens
            support_embs[l] = emb_np[:n]
            query_embs[l] = emb_np[n:]
            
            # Predictions (unscaled)
            mu_scaled = preds_all[l][n:]
            mu_unscaled = mu_scaled * y_std + y_mean
            predictions[l] = mu_unscaled
            
            if K_g is not None:
                alpha_l, _, _ = solve_gcv_weights(K_g, mu_unscaled)
                alphas[l] = alpha_l
                
            if l in ATTN_HOOK_DATA["cache"]:
                item_attn[l] = ATTN_HOOK_DATA["cache"][l]
                
        return {
            "support_embs": support_embs,
            "query_embs": query_embs,
            "predictions": predictions,
            "alphas": alphas,
            "item_attn": item_attn,
            "y_scaled": y_scaled,
            "y_mean": y_mean,
            "y_std": y_std
        }
