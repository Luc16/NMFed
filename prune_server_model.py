# train_with_torchao.py
import torch
import torch.nn as nn
import time
from torchvision.models import resnet18, ResNet18_Weights
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

def run_torchao_proof_of_concept():
    """
    Demonstrates training a model with torchao's dynamic 2:4 sparsity.
    """
    print("--- TorchAO Dynamic Sparse Training Proof of Concept ---")
    
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0)):
        print("\nWARNING: Your GPU does not have CUDA compute capability 8.0+ (Ampere architecture).")
        print("         `torchao` will function correctly, but you will NOT see a training speedup.")
        print("         Training will likely be SLOWER than dense training.\n")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 1. Load a pre-trained ResNet18 model
    print("Loading pre-trained ResNet18 model...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # 2. Adapt the model for a 10-class problem
    num_ftrs = model.fc.in_features
    
    # --- CHANGE 1: Pad the output dimension to be a multiple of 128 ---
    # The torchao sparse kernel requires dimensions to be multiples of 128.
    # We change the output from 10 to 128 to meet this requirement.
    padded_out_features = 128
    model.fc = nn.Linear(num_ftrs, padded_out_features)
    
    model.to(device)
    model.half() 
    print(f"Model adapted. Final layer output padded to {padded_out_features} features.")

    # 3. Create the config dictionary for the swap function
    print("\nBuilding config to swap nn.Linear layers...")
    sparse_config = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  - Targeting layer for swap: '{name}'")
            sparse_config[name] = SemiSparseLinear
            
    # 4. Use torchao to swap nn.Linear layers
    print("Swapping nn.Linear layers with torchao.sparsity.training.SemiSparseLinear...")
    swap_linear_with_semi_sparse_linear(model, config=sparse_config)
    
    print("\nVerification after swap:")
    for name, module in model.named_modules():
        if isinstance(module, SemiSparseLinear):
            print(f"  - Layer '{name}' is now a SemiSparseLinear layer.")
    
    # 5. Prepare for training
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    dummy_input = torch.randn(16, 3, 224, 224).to(device).half()
    dummy_target = torch.randint(0, 10, (16,)).to(device)

    # 6. Run a simple training loop
    print("\nStarting a short training loop (5 steps)...")
    model.train()
    start_time = time.time()
    for i in range(5):
        optimizer.zero_grad()
        
        # The model now outputs a tensor of shape 
        output = model(dummy_input)
        
        # --- CHANGE 2: Slice the output to get only the 10 classes we need ---
        # We only use the first 10 outputs for our loss calculation.
        sliced_output = output[:, :10]
        
        loss = criterion(sliced_output, dummy_target)
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}/5, Loss: {loss.item():.4f}")
    end_time = time.time()
    
    print(f"\nTraining loop completed in {end_time - start_time:.2f} seconds.")
    print("\n--- Proof of Concept Complete ---")
    print("This demonstrates that a model converted with `torchao` can successfully execute a training loop.")

if __name__ == "__main__":
    run_torchao_proof_of_concept()
