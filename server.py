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
try:
    from torchao.sparsity.training import SemiSparseLinear
except ImportError:
    print("WARNING: torchao library not found.")
    print("         Loading the model may fail if it uses SemiSparseLinear layers.")
# --- End Added Imports ---


# Tamanho do pedaço de bytes enviado por mensagem (2 MB é um valor bom)
CHUNK_SIZE = 2 * 1024 * 1024

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

class ModelDistributorService(rpc.ModelDistributorServicer):
    def __init__(self, model, arch_name="resnet18"):
        # Set model to eval mode for distribution
        self.model = model.eval()
        self.arch_name = arch_name

    def GetModel(self, request, context):
        # This server serializes the state_dict on-the-fly for each request
        # It does NOT use the pruned_model.pt file directly
        tmp_path = "/tmp/mode.pt"
        serialize_state_dict_to_path(self.model, tmp_path)

        total_size = os.path.getsize(tmp_path)
        sha256 = file_sha256(tmp_path)

        hdr = pb.ModelHeader(
            model_name = request.model_name, # Fixed: use request.model_name
            format =     request.format,
            version =    request.version,
            arch =       self.arch_name,
            dtype =      tensor_dtype_str(self.model),
            total_size = total_size,
            sha256 =     sha256,
        )
        yield pb.ModelStream(header=hdr)

        try:
            with open(tmp_path, "rb") as f:
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield pb.ModelStream(chunk=pb.ModelChunk(data=data))
        finally:
            # Clean up the temporary file after streaming
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# The 'server(model)' function has been replaced by the main block below
if __name__ == "__main__":
    SAVED_MODEL_PATH = "pruned_model.pt"
    
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Model file not found at {SAVED_MODEL_PATH}")
        print("Please run 'prune_server_model.py' first to generate the model file.")
    else:
        print(f"Loading model from {SAVED_MODEL_PATH}...")
        
        # Load the model onto the CPU first
        # torch.load needs the class definitions (like SemiSparseLinear)
        # to be imported in this script to work.
        model = torch.load(SAVED_MODEL_PATH, map_location=torch.device('cpu'))
        
        # Determine device and move model there
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded successfully and moved to {device}.")

        # Start the gRPC server
        grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))

        rpc.add_ModelDistributorServicer_to_server(
            ModelDistributorService(model), grpc_server
        )

        grpc_server.add_insecure_port("[::]:50051")
        grpc_server.start()

        print("ModelDistributor gRPC server rodando em 0.0.0.0:50051")
        grpc_server.wait_for_termination()
