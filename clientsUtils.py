import io, hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def build_resnet18_padded128():
    """
    Constrói uma ResNet-18 com camada final de 128-dimensões.
    Se uma GPU Ampere+ (cc >= 8.0) estiver disponível, troca automaticamente
    as camadas nn.Linear por SemiSparseLinear (2:4) do torchao.
    """
    
    # 1. Verifica o hardware (lógica copiada de prune_server_model.py)
    use_cuda = torch.cuda.is_available()
    cc = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
    has_ampere = use_cuda and (cc >= (8, 0))

    if not has_ampere:
        print("\n[Client WARNING]")
        print("  - Cliente sem GPU Ampere (cc>=8.0). O kernel 2:4 do torchao é apenas CUDA Ampere+.")
        print("  - O treinamento local será feito de forma DENSA (nn.Linear padrão).\n")

    # 2. Constrói a arquitetura base
    m = resnet18(weights=None)              # sem pré-treino
    m.fc = nn.Linear(m.fc.in_features, 128) # padding p/ sparsidade 2:4

    # 3. (Condicional) Troca nn.Linear -> SemiSparseLinear
    if has_ampere:
        print("\n[Client] GPU Ampere detectada. Trocando nn.Linear por SemiSparseLinear...")
        sparse_config = {}
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - Alvo para troca: '{name}'")
                sparse_config[name] = SemiSparseLinear

        swap_linear_with_semi_sparse_linear(m, config=sparse_config)
        print("[Client] Troca de camadas concluída.")
    
    return m

def make_cifar10_loaders(data_dir, batch_size):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_val = transforms.ToTensor()

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm_train)
    val_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm_val)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=0,            # <- evita multi-process no Windows/Ray
        pin_memory=False,         # <- desnecessário sem GPU
        persistent_workers=False, # <- garante que não fica preso
    )

    train_loader = DataLoader(train_set, shuffle=True,  **loader_args)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_args)
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
    model.to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()
    seen = 0
    it = iter(train_loader)
    for i in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        seen += x.size(0)
        if (i+1) % 20 == 0:
            print(f"[train] step {i+1}/{steps} loss={loss.item():.4f} seen={seen}")
    return model, seen