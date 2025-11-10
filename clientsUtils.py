# clientsUtils.py
import io
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# CIFAR-10 stats
# ------------------------------------------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ------------------------------------------------------------
# Modelo base: ResNet-18 com saída 128 (padded)
# (A troca para SemiSparseLinear é feita no client.py)
# ------------------------------------------------------------
def build_resnet18_padded128():
    """
    Constrói uma ResNet-18 sem pesos pré-treinados e troca a fc para 128 saídas.
    OBS: NÃO faz swap para SemiSparseLinear aqui — isso é feito no client.py,
    já no device/dtype correto, para evitar incompatibilidades de versão do torchao.
    """
    m = resnet18(weights=None)               # sem pré-treino (pesos vêm do servidor)
    m.fc = nn.Linear(m.fc.in_features, 128)  # saída "padded" 128 (CIFAR-10 usa [:10])
    return m

# ------------------------------------------------------------
# DataLoaders de CIFAR-10
# ------------------------------------------------------------
def make_cifar10_loaders(data_dir, batch_size):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tfm_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=tfm_train)
    val_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm_val)

    # Heurística simples para acelerar quando há GPU
    use_cuda = torch.cuda.is_available()
    loader_args = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=(2 if use_cuda else 0),
        pin_memory=use_cuda,
        persistent_workers=(True if use_cuda and 2 > 0 else False),
    )
    train_loader = DataLoader(train_set, **loader_args)

    loader_args_val = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=(2 if use_cuda else 0),
        pin_memory=use_cuda,
        persistent_workers=(True if use_cuda and 2 > 0 else False),
    )
    val_loader = DataLoader(val_set, **loader_args_val)

    return train_loader, val_loader

# ------------------------------------------------------------
# Serialização dos pesos
# ------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.digest().hex()

def sd_to_bytes(sd):
    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()

def bytes_to_sd(b: bytes, device="cpu"):
    return torch.load(io.BytesIO(b), map_location=device)

# ------------------------------------------------------------
# Avaliação (Top-1)
# - Converte inputs para o MESMO dtype do modelo (BF16/FP16/FP32)
# - Usa autocast moderno (sem FutureWarning)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_top1(model, loader, device):
    model.eval()
    correct, total = 0, 0
    use_amp = (device != "cpu")
    target_dtype = next(model.parameters()).dtype  # p.ex. torch.bfloat16 no H100

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # inputs no mesmo dtype do modelo (evita "Input type vs weight type")
        if use_amp:
            x = x.to(dtype=target_dtype)

        with amp.autocast("cuda", enabled=use_amp, dtype=target_dtype):
            logits = model(x)              # [B, 128]
            # Se seu classificador usa só [:10] em outro lugar, mantenha aqui inteiro.
            preds = logits.float().argmax(1)  # argmax em FP32
        correct += (preds == y).sum().item()
        total   += y.size(0)

    return correct / max(1, total)

# ------------------------------------------------------------
# Treino local
# - Converte inputs para dtype do modelo
# - Autocast moderno
# - Loss em FP32 (estável), sem GradScaler para BF16
# ------------------------------------------------------------
def train_locally(model, loader, steps, device):
    model.train()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    use_amp = (device != "cpu")
    target_dtype = next(model.parameters()).dtype
    seen = 0

    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_amp:
            x = x.to(dtype=target_dtype)

        opt.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=use_amp, dtype=target_dtype):
            logits = model(x)
            loss = crit(logits.float(), y)  # perda em FP32

        loss.backward()
        opt.step()
        seen += x.size(0)

    return model, seen
