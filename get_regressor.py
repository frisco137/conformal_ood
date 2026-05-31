import torch
from tabpfn import TabPFNRegressor

def save_full_tabpfn_model(output_path='tabpfn_full_model.pt'):
    print("Loading TabPFNRegressor...")
    reg = TabPFNRegressor(device='cpu')

    # Fit on dummy data to ensure the underlying PyTorch model is initialized
    dummy_X = [[0.0], [1.0]]
    dummy_y = [0, 1]
    reg.fit(dummy_X, dummy_y)

    pt_model = getattr(reg, 'model_', getattr(reg, 'model', None))

    if pt_model is None:
        print("Failed to find the underlying PyTorch model.")
        return

    print(f"Model found: {type(pt_model).__name__}")

    # CRITICAL CHANGE: Save the entire object, NOT the state_dict
    torch.save(pt_model, output_path)
    print(f"Successfully saved full TabPFN model object to: {output_path}")

if __name__ == '__main__':
    save_full_tabpfn_model('tabpfn_full_model.pt')