import os
import io
import hashlib
import torch
import grpc
from concurrent import futures  # thread pool padrão do gRPC

import comunication_pb2 as pb
import comunication_pb2_grpc as rpc

import threading
from clientsUtils import sd_to_bytes, bytes_to_sd, build_resnet18_padded128
import torch.nn as nn
from torchvision.models import resnet18

MIN_UPDATES_PER_ROUND = 4                 # quando receber N updates, agrega
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Estado global do servidor (modelo e versão)
# NEW
global_state = None                       # dict de tensores (state_dict)
global_version = 1                        # string/inteiro de versão do global
pending_updates = []                      # lista de tuplas (state_dict, peso)
lock = threading.Lock()   

# Tamanho do pedaço de bytes enviado por mensagem (2 MB é um valor bom)
CHUNK_SIZE = 2 * 1024 * 1024

def fedavg(weighted_states):
    """
    weighted_states: lista de (state_dict, peso)
    retorna: novo state_dict = soma( sd[k] * (peso/soma_pesos) )
    """
    total = sum(w for _, w in weighted_states)
    if total <= 0:
        total = 1.0

    keys = list(weighted_states[0][0].keys())
    new_sd = {}
    for k in keys:
        acc = None
        for sd, w in weighted_states:
            part = sd[k] * (w / total)
            acc = part if acc is None else acc + part
        new_sd[k] = acc
    return new_sd


def build_resnet18_padded128():
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 128)  # mesma FC usada nos clientes
    return m

def tensor_dtype_str(model):
    """
    Descobre o dtype dos parâmetros do modelo (só para informar no header).
    """
    for p in model.parameters():
        return str(p.dtype)
    return "float32"

def file_sha256(path):
    """
    Calcula o hash SHA-256 de um arquivo (para o cliente validar integridade).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

def serialize_state_dict_to_path(model, path):
    """
    Serializa APENAS o state_dict (pesos) para disco.
    - Vantagem: formato simples e leve para PyTorch-to-PyTorch.
    - Obs.: o cliente precisa reconstruir a mesma arquitetura para dar load.
    """
    sd = model.state_dict()
    torch.save(sd, path)

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

class ModelDistributorService(rpc.ModelDistributorServicer):
    def __init__(self, model, arch_name="resnet18"):
        # Set model to eval mode for distribution
        self.arch_name = arch_name
        self.model = model.eval()
        self.global_version = "1.0.0"
        global global_state, global_version
        global_state = model.state_dict()
        global_version = 1
        print(f"[server] Estado global inicial carregado. Versão {global_version}")

    def GetModel(self, request, context):
        try:
            print(f"[GetModel] pedido de {request.name}")
            buf = io.BytesIO()
            raw = sd_to_bytes(global_state)  # usa o modelo agregado

            print(f"[GetModel] state_dict serializado: {len(raw)} bytes")

            hdr = pb.ModelHeader(
                model_name = request.name,
                version    = str(global_version),
                arch       = self.arch_name,
                total_size = len(raw),
                sha256     = sha256_bytes(raw),
            )
            print("[GetModel] enviando header…")
            yield pb.ModelStream(header=hdr)

            sent = 0
            for i in range(0, len(raw), CHUNK_SIZE):
                chunk = raw[i:i+CHUNK_SIZE]
                sent += len(chunk)
                if i == 0 or sent == len(raw) or sent % (10*1024*1024) == 0:
                    print(f"[GetModel] enviando… {sent}/{len(raw)} bytes")
                yield pb.ModelStream(chunk=pb.ModelChunk(data=chunk))

            print("[GetModel] stream concluída")
        except Exception as e:
            print(f"[GetModel] ERRO: {e}")
            context.abort(grpc.StatusCode.INTERNAL, f"GetModel failed: {e}")

    # def SubmitUpdate(self, request, context):
    #     global pending_updates, global_state, global_version

    #     print(f"[SubmitUpdate] from={request.client_id} "
    #           f"bytes={len(request.state_bytes)} samples={request.num_samples} "
    #           f"base_version={request.base_version}")
    #     # apenas confirma recebimento; sem FedAvg por enquanto
    #     return pb.ModelUpdateAck(
    #         ok=True,
    #         server_version=self.global_version,
    #         msg="received (no aggregation yet)"
    #     )


    def test_global_model(global_state, device="cpu"):
        """
        Testa se o modelo agregado (global_state) é válido.
        - Faz forward com batch fake
        - (Opcional) avalia top1 no CIFAR-10
        """
        print("\n[TESTE] Iniciando validação do modelo global agregado…")

        try:
            # 1) Reconstrói arquitetura
            model = build_resnet18_padded128().to(device)
            model.load_state_dict(global_state, strict=True)
            model.eval()
            print("[TESTE] state_dict carregado com sucesso ✓")

            # 2) Faz forward com batch dummy
            x = torch.randn(8, 3, 224, 224).to(device)
            with torch.no_grad():
                y = model(x)
            print(f"[TESTE] forward OK — output shape: {tuple(y.shape)}")

            # 3) (opcional) test CIFAR-10 accuracy
            try:
                from clientsUtils import make_cifar10_loaders, evaluate_top1
                _, val_loader = make_cifar10_loaders("/root/data", batch_size=64)
                acc = evaluate_top1(model, val_loader, device=device)
                print(f"[TESTE] CIFAR10 top-1 accuracy ≈ {acc:.2f}%")
            except Exception as e:
                print(f"[TESTE] CIFAR10 não disponível ou falhou: {e}")

            print("[TESTE] Modelo global parece consistente ✅")
            return True

        except Exception as e:
            print(f"[TESTE] Falhou ao validar modelo global ❌: {e}")
            return False

    def SubmitUpdate(self, request, context):
        global pending_updates, global_state, global_version

        try:
            print(f"[SubmitUpdate] from={request.client_id} "
                  f"bytes={len(request.state_bytes)} samples={request.num_samples} "
                  f"base_version={getattr(request, 'base_version', '?')}")

            # NEW: recuperar state_dict do update
            sd = bytes_to_sd(request.state_bytes, device=DEVICE)
            weight = int(request.num_samples) if request.num_samples > 0 else 1

            # NEW: acumular e, se atingiu o mínimo, agregar
            with lock:
                pending_updates.append((sd, weight))

                if len(pending_updates) >= MIN_UPDATES_PER_ROUND:
                    print(f"[server] atingiu {len(pending_updates)} updates — agregando (FedAvg)…")
                    new_sd = fedavg(pending_updates)
                    global_state = new_sd
                    self.model.load_state_dict(global_state, strict=True)
                    test_global_model(global_state, device=DEVICE)

                    global_version += 1
                    pending_updates.clear()
                    print(f"[server] nova versão global: {global_version}")

            return pb.ModelUpdateAck(
                ok=True,
                server_version=str(global_version),
                msg="update recebido (agregação automática quando atingir o mínimo)"
            )

        except Exception as e:
            print("[server] erro no SubmitUpdate:", e)
            return pb.ModelUpdateAck(ok=False, server_version=str(global_version), msg=str(e))

# The 'server(model)' function has been replaced by the main block below
if __name__ == "__main__":
    SAVED_MODEL_PATH = "pruned_model.pt"
    
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Model file not found at {SAVED_MODEL_PATH}")
        print("Please run 'prune_server_model.py' first to generate the model file.")
    else:
        print(f"Loading model from {SAVED_MODEL_PATH}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_resnet18_padded128().to(device)
        print(f"Model loaded successfully and moved to {device}.")
        
        sd = torch.load(SAVED_MODEL_PATH, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        
        # Start the gRPC server
        grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=8),
            options=[
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),  # 128 MB
                ("grpc.max_send_message_length",    128 * 1024 * 1024),
            ],
        )
        rpc.add_ModelDistributorServicer_to_server(
            ModelDistributorService(model), grpc_server
        )

        grpc_server.add_insecure_port("[::]:50051")
        grpc_server.start()

        print("ModelDistributor gRPC server rodando em 0.0.0.0:50051")
        grpc_server.wait_for_termination()
