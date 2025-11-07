import io, hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def build_resnet18_padded128():
    m = resnet18(weights=None)              # sem pré-treino
    m.fc = nn.Linear(m.fc.in_features, 128) # padding p/ sparsidade 2:4
    return m

def make_cifar10_loaders(data_dir: str, batch_size: int, num_workers: int = 2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),             # adapta CIFAR (32x32) para ResNet18 (224x224)
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True,  transform=train_tf)
    val_ds   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.digest().hex()

def sd_to_bytes(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()

def bytes_to_sd(b: bytes, device="cpu"):
    return torch.load(io.BytesIO(b), map_location=device)

@torch.no_grad()
def evaluate_top1(model: nn.Module, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[:, :10].float()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def train_locally(model: nn.Module, train_loader, steps: int = 200, device="cpu"):
    model.to(device).train()
    opt  = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    crit = nn.CrossEntropyLoss()
    seen = 0
    it = iter(train_loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)[:, :10].float()
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        sched.step()
        seen += x.size(0)
    model.eval()
    return model, seen