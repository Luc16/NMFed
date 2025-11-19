import torch
import torchvision.models as models
import torch.nn as nn
import os
from sparsity import apply_2_4_sparsity

def main():
    print("[1/3] Loading ResNet18...")
    model = models.resnet18(pretrained=False)
    # Adjust for CIFAR-10 (smaller images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)

    print("[2/3] Applying 2:4 Sparsity (Offline)...")
    # Apply pruning to valid layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 2:4 requires size divisible by 4
            if module.weight.numel() % 4 == 0:
                apply_2_4_sparsity(module)
                print(f"   -> Pruned: {name} | Shape: {module.weight.shape}")

    print("[3/3] Saving Pruned Model...")
    # We save the state_dict. This contains 'weight_orig' and 'weight_mask'.
    # This mask is now PERMANENT for the lifecycle of the FL experiment.
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/initial_sparse_model.pt")
    print("Done. Model saved to 'models/initial_sparse_model.pt'")

if __name__ == "__main__":
    main()
