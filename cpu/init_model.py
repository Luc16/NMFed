import torch
import torchvision.models as models
import torch.nn as nn
import os
from sparsity import apply_2_4_sparsity
import argparse

def main():
    parse = argparse.ArgumentParser(description="Initialize and prune a ResNet18 model with 2:4 sparsity.")
    parse.add_argument('--no-sparsity', type=bool, default=False, help='If set to True, do not apply sparsity.')
    parse.add_argument('--method', type=str, default='topk', help='Sparsity method: topk, random, stochastic, grad')
    args = parse.parse_args()

    method = args.method

    print("[1/3] Loading Pretrained ResNet18 Backbone...")
    base = models.resnet18(weights=None)

    # New CIFAR model
    model = models.resnet18(weights=None)

    # Modify layers for CIFAR
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)

    # Load pretrained weights except incompatible layers
    state = base.state_dict()

    # Remove keys that won't match
    remove_keys = ["conv1.weight", "fc.weight", "fc.bias"]
    for k in remove_keys:
        if k in state:
            del state[k]

    print("Loading pretrained backbone...")
    model.load_state_dict(state, strict=False)

    if not args.no_sparsity:
        print("[2/3] Applying 2:4 Sparsity (Offline)...")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                
                # basic check
                if module.weight.numel() % 4 == 0:
                    apply_2_4_sparsity(module, method=method)
                    print(f"   → Pruned: {name:20s} | Shape: {tuple(module.weight.shape)}")

    print("[3/3] Saving Initial Model...")
    os.makedirs("models", exist_ok=True)

    file_path = f"models/initial_sparse_none_{method}_model.pt" if not args.no_sparsity \
                else "models/initial_dense_none_model.pt"

    torch.save(model.state_dict(), file_path)

    print(f"Done. Model saved to {file_path}")

if __name__ == "__main__":
    main()
