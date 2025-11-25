import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Import torchao specific tools
from torchao.sparsity.training import swap_linear_with_semi_sparse_linear

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.001  # Lower LR is usually better for fine-tuning pretrained models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # Recommended for Ampere+ sparsity

# Check hardware compatibility
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    if cap < (8, 0):
        print(f"WARNING: Your GPU capability is {cap}. "
              "TorchAO 2:4 sparsity requires Ampere (8.0) or newer for speedups.")
else:
    print("WARNING: No CUDA detected. This benchmark requires an NVIDIA GPU.")

# ---------------------------------------------------------
# 1. Data Setup (CIFAR-10)
# ---------------------------------------------------------
def get_dataloader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Standard ImageNet normalization is often used with pretrained models, 
        # but CIFAR-10 stats are fine too. Sticking to CIFAR stats here.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True # Ensure shapes are consistent for kernels
    )

# ---------------------------------------------------------
# 2. Model Builder (Pretrained + Padding)
# ---------------------------------------------------------
def build_model(sparse=False):
    print(f"[Setup] Loading ResNet18 with ImageNet weights (sparse={sparse})...")
    
    # 1. Load Pretrained Weights
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    
    # 2. Pad output to 128 (Constraint for 2:4 kernel efficiency)
    # Note: This resets the final layer weights, which is correct for fine-tuning on new classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 128)
    
    # 3. Move to device/dtype
    model = model.to(DEVICE).to(dtype=DTYPE)
    
    # 4. Apply Sparsity if requested
    if sparse:
        print("[Setup] Swapping Linear layers for SemiSparseLinear (2:4)...")
        # Swap nn.Linear -> SemiSparseLinear
        swap_linear_with_semi_sparse_linear(model)
        
    return model

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_benchmark(model, name):
    print(f"\n--- Starting Benchmark: {name} ---")
    
    # Optimizer (SGD is standard, but AdamW is also popular for fine-tuning)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    loader = get_dataloader()
    model.train()
    
    # Warmup
    print("Warming up CUDA...")
    iter_loader = iter(loader)
    for _ in range(10):
        try:
            inputs, targets = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            inputs, targets = next(iter_loader)
            
        inputs, targets = inputs.to(DEVICE).to(dtype=DTYPE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)[:, :10] # Slice 128 -> 10
        loss = criterion(outputs.float(), targets)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Training for {EPOCHS} epochs...")
    start_time = time.time()
    total_samples = 0
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE).to(dtype=DTYPE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            raw_out = model(inputs) 
            
            # Slice padding (128 -> 10) for loss calculation
            outputs = raw_out[:, :10]
            
            loss = criterion(outputs.float(), targets)
            loss.backward()
            optimizer.step()
            
            total_samples += inputs.size(0)
            
        torch.cuda.synchronize()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f} | Time: {time.time() - epoch_start:.2f}s")

    end_time = time.time()
    total_time = end_time - start_time
    avg_throughput = total_samples / total_time
    
    print(f"--- {name} Result ---")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Throughput: {avg_throughput:.2f} samples/sec")
    return total_time

# ---------------------------------------------------------
# 4. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Run Dense Baseline
    dense_model = build_model(sparse=False)
    dense_time = train_benchmark(dense_model, "Dense Pretrained ResNet18")
    
    # Cleanup
    del dense_model
    torch.cuda.empty_cache()
    
    # 2. Run Sparse 2:4
    sparse_model = build_model(sparse=True)
    sparse_time = train_benchmark(sparse_model, "Sparse (2:4) Pretrained ResNet18")
    
    # 3. Compare
    print("\n================ SUMMARY ================")
    print(f"Dense Time : {dense_time:.2f} s")
    print(f"Sparse Time: {sparse_time:.2f} s")
    
    if sparse_time < dense_time:
        speedup = dense_time / sparse_time
        print(f"Result: Sparse is {speedup:.2f}x FASTER")
    else:
        print("Result: Sparse is SLOWER (Check GPU compatibility or model size overhead)")
    print("=========================================")
