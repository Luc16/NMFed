import os
import io
import hashlib
import torch
import grpc
from concurrent import futures  # thread pool padrão do gRPC

import comunication_pb2 as pb
import comunication_pb2_grpc as rpc

# --- Imports Added for Loading ---
# We need to import the class definitions so torch.load() can 
# reconstruct the model object, even if we don't use them directly.
import torch.nn as nn
from torchvision.models import resnet18
# try:
#     from torchao.sparsity.training import SemiSparseLinear
# except ImportError:
#     print("WARNING: torchao library not found.")
#     print("         Loading the model may fail if it uses SemiSparseLinear layers.")
# # --- End Added Imports ---


# Tamanho do pedaço de bytes enviado por mensagem (2 MB é um valor bom)
CHUNK_SIZE = 2 * 1024 * 1024

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
        self.model = model.eval()
        self.arch_name = arch_name
        self.global_version = "1.0.0"

    def GetModel(self, request, context):
        # This server serializes the state_dict on-the-fly for each request
        # It does NOT use the pruned_model.pt file directly
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        raw = buf.getvalue()

        hdr = pb.ModelHeader(
            model_name = request.name, # Fixed: use request.model_name
            format =     request.global_version,
            version =    request.version,
            arch =       self.arch_name,
            total_size = len(raw),
            sha256 =     sha256_bytes(raw),
        )
        yield pb.ModelStream(header=hdr)

        for i in range(0, len(raw), CHUNK_SIZE):
            yield pb.ModelStream(chunk=pb.ModelChunk(data=raw[i:i+CHUNK_SIZE]))

    def SubmitUpdate(self, request, context):
        print(f"[SubmitUpdate] from={request.client_id} "
              f"bytes={len(request.state_bytes)} samples={request.num_samples} "
              f"base_version={request.base_version}")
        # apenas confirma recebimento; sem FedAvg por enquanto
        return pb.ModelUpdateAck(
            ok=True,
            server_version=self.global_version,
            msg="received (no aggregation yet)"
        )
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
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))

        rpc.add_ModelDistributorServicer_to_server(
            ModelDistributorService(model), grpc_server
        )

        grpc_server.add_insecure_port("[::]:50051")
        grpc_server.start()

        print("ModelDistributor gRPC server rodando em 0.0.0.0:50051")
        grpc_server.wait_for_termination()
