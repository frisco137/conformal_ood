"""Detailed architecture dump for all three models."""
import sys
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood")
import numpy as np
import torch
from models import load

def dump_tabpfn():
    print("=" * 72)
    print("TabPFN v2 DETAILED INTERNALS")
    print("=" * 72)
    m = load('tabpfn_v2', task='regression', device='cuda')
    rng = np.random.RandomState(42)
    m._fit(rng.randn(16, 3), rng.randn(16))
    model = m._torch_model()
    
    block0 = model.transformer_encoder.layers[0]
    print('\n--- Block 0 full structure ---')
    for name, mod in block0.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:45s}  {type(mod).__name__}')
    
    # Linear layers in block 0
    print('\n--- Block 0 Linear layers ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f'  {name}: Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})')
    
    # Attention details
    print('\n--- Block 0 attention details ---')
    item_attn = block0.self_attn_between_items
    feat_attn = block0.self_attn_between_features
    print(f'  self_attn_between_items: {type(item_attn).__name__}')
    for name, mod in item_attn.named_modules():
        if name.count('.') == 0 and name:
            print(f'    {name}: {type(mod).__name__}')
    print(f'  self_attn_between_features: {type(feat_attn).__name__}')
    for name, mod in feat_attn.named_modules():
        if name.count('.') == 0 and name:
            print(f'    {name}: {type(mod).__name__}')
    
    # n_heads
    print(f'\n  item_attn num_heads = {getattr(item_attn, "num_heads", "UNKNOWN")}')
    print(f'  feat_attn num_heads = {getattr(feat_attn, "num_heads", "UNKNOWN")}')
    
    # LayerNorm in block
    print('\n--- Block 0 norms ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            print(f'  {name}: LayerNorm({mod.normalized_shape})')

    # Decoder
    print('\n--- Decoder structure ---')
    for name, mod in model.named_modules():
        if name.startswith('decoder') or name.startswith('criterion'):
            if name.count('.') <= 2:
                print(f'  {name!r:50s}  {type(mod).__name__}')
                if isinstance(mod, torch.nn.Linear):
                    print(f'    Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})')
    
    # Encoder chain
    print('\n--- Encoder (x) chain ---')
    for name, mod in model.encoder.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:50s}  {type(mod).__name__}')
            if isinstance(mod, torch.nn.Linear):
                print(f'    Linear({mod.in_features}, {mod.out_features})')
    
    # y encoder
    print('\n--- y_encoder chain ---')
    for name, mod in model.y_encoder.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:50s}  {type(mod).__name__}')
            if isinstance(mod, torch.nn.Linear):
                print(f'    Linear({mod.in_features}, {mod.out_features})')
    
    # Check for global final norm
    print('\n--- All top-level norms ---')
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and 'transformer_encoder.layers' not in name:
            print(f'  {name!r}: LayerNorm({mod.normalized_shape})')

    del m
    torch.cuda.empty_cache()

def dump_tabicl():
    print("\n" + "=" * 72)
    print("TabICL v2 DETAILED INTERNALS")
    print("=" * 72)
    m = load('tabicl_v2', task='regression', device='cuda')
    rng = np.random.RandomState(42)
    m._fit(rng.randn(16, 3), rng.randn(16))
    model = m._torch_model()
    predictor = model.icl_predictor
    
    # ICL block 0
    block0 = predictor.tf_icl.blocks[0]
    print('\n--- ICL Block 0 structure ---')
    for name, mod in block0.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:45s}  {type(mod).__name__}')
    
    print('\n--- ICL Block 0 Linear layers ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f'  {name}: Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})')
    
    # Attention
    print('\n--- ICL Block 0 attention ---')
    attn = block0.attn if hasattr(block0, 'attn') else block0.self_attn
    print(f'  attn type: {type(attn).__name__}')
    print(f'  num_heads: {getattr(attn, "num_heads", "?")}')
    print(f'  embed_dim: {getattr(attn, "embed_dim", "?")}')
    head_dim = getattr(attn, 'head_dim', getattr(attn, 'embed_dim', 0) // getattr(attn, 'num_heads', 1))
    print(f'  head_dim: {head_dim}')
    
    # Norms in block
    print('\n--- ICL Block 0 norms ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            print(f'  {name}: LayerNorm({mod.normalized_shape})')
    
    # Decoders
    print('\n--- ICL Predictor decoders ---')
    for name, mod in predictor.named_modules():
        if isinstance(mod, torch.nn.Linear) and 'blocks' not in name:
            print(f'  {name}: Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})')
    
    # Final LN
    print(f'\n  predictor.ln exists: {hasattr(predictor, "ln")}')
    if hasattr(predictor, 'ln'):
        print(f'  predictor.ln: {type(predictor.ln).__name__}({predictor.ln.normalized_shape})')
    
    # Check RoPE
    print('\n--- RoPE in ICL blocks ---')
    for name, mod in predictor.tf_icl.named_modules():
        if 'rope' in name.lower() or 'rotary' in name.lower():
            print(f'  {name}: {type(mod).__name__}')
    
    # Column embedder structure
    print('\n--- Column embedder ---')
    for name, mod in model.col_embedder.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:50s}  {type(mod).__name__}')
    
    # Row interactor structure
    print('\n--- Row interactor ---')
    for name, mod in model.row_interactor.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:50s}  {type(mod).__name__}')
    
    # Check for CLS tokens
    print('\n--- Register/CLS params ---')
    for name, param in model.named_parameters():
        if 'cls' in name.lower() or 'register' in name.lower():
            print(f'  {name}: {param.shape}')
    
    # Check quantile_dist
    print(f'\n  quantile_dist: {type(model.quantile_dist).__name__}')
    for name, mod in model.quantile_dist.named_modules():
        if name:
            print(f'    {name}: {type(mod).__name__}')

    del m
    torch.cuda.empty_cache()

def dump_tabswift():
    print("\n" + "=" * 72)
    print("TabSwift DETAILED INTERNALS")
    print("=" * 72)
    m = load('tabswift', task='regression', device='cuda')
    rng = np.random.RandomState(42)
    m._fit(rng.randn(16, 3), rng.randn(16))
    model = m._torch_model()
    predictor = model.icl_predictor
    
    # Config from checkpoint
    print(f'\n  max_classes = {model.max_classes}')
    print(f'  max_dim = {model.max_dim}')
    print(f'  embed_dim = {model.embed_dim}')
    print(f'  icl_num_blocks = {model.icl_num_blocks}')
    print(f'  icl_nhead = {model.icl_nhead}')
    print(f'  ff_factor = {model.ff_factor}')
    print(f'  norm_first = {model.norm_first}')
    print(f'  register_tokens = {predictor.register_tokens}')
    print(f'  row_num_cls = {model.row_num_cls}')
    icl_dim = model.embed_dim * model.row_num_cls
    print(f'  icl_dim (embed_dim * row_num_cls) = {icl_dim}')
    
    # x_linear
    print(f'\n  x_linear: Linear({model.x_linear.in_features}, {model.x_linear.out_features})')
    
    # ICL block 0
    block0 = predictor.tf_icl.blocks[0]
    print('\n--- ICL Block 0 structure ---')
    for name, mod in block0.named_modules():
        if name.count('.') <= 1:
            print(f'  {name!r:45s}  {type(mod).__name__}')
    
    print('\n--- ICL Block 0 Linear layers ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f'  {name}: Linear({mod.in_features}, {mod.out_features}, bias={mod.bias is not None})')
    
    # Attention
    print(f'\n  block0.attn num_heads = {block0.attn.num_heads}')
    print(f'  block0.attn embed_dim = {block0.attn.embed_dim}')
    head_dim = block0.attn.embed_dim // block0.attn.num_heads
    print(f'  head_dim = {head_dim}')
    
    # Norms
    print('\n--- Block 0 norms ---')
    for name, mod in block0.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            print(f'  {name}: LayerNorm({mod.normalized_shape})')
    
    # Decoders
    print('\n--- Decoders ---')
    print(f'  clf decoder: {predictor.decoder}')
    print(f'  reg_decoder: {predictor.reg_decoder}')
    print(f'  y_encoder: Linear({predictor.y_encoder.in_features}, {predictor.y_encoder.out_features})')
    print(f'  y_encoder_reg: Linear({predictor.y_encoder_reg.in_features}, {predictor.y_encoder_reg.out_features})')
    
    # Final LN
    print(f'\n  predictor.ln: {type(predictor.ln).__name__}({predictor.ln.normalized_shape})')
    
    # Register token values
    print(f'  register_token_values shape: {predictor.tf_icl.register_token_values.shape}')

    del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    dump_tabpfn()
    dump_tabicl()
    dump_tabswift()
