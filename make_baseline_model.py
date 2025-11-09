# make_baseline_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18_padded128_dense():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # ou None, se preferir
    m.fc = nn.Linear(m.fc.in_features, 128)
    return m

if __name__ == "__main__":
    m = build_resnet18_padded128_dense()
    torch.save(m.state_dict(), "baseline_model.pt")
    print("baseline_model.pt salvo.")
