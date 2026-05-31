from pfn_hooks import PFNHookManager
import torch

checkpoint_path = "tabpfn_ood/checkpoints/"
filename = "tabpfn.cpkt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, filename=filename, device=device)

print("success")