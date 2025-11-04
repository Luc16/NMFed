import os
import io
import hashlib
import torch
import grpc
from concurrent import futures  # thread pool padrão do gRPC

import comunication_pb2 as pb
import comunication_pb2_grpc as rpc

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
        self.model = model.eval()
        self.arch_name = arch_name

    def GetModel(self, request, context):
        tmp_path = "/tmp/mode.pt"
        serialize_state_dict_to_path(self.model, tmp_path)

        total_size = os.path.getsize(tmp_path)
        sha256 = file_sha256(tmp_path)

        hdr = pb.ModelHeader(
            model_name = request.name,
            format =     request.format,
            version =    request.version,
            arch =       self.arch_name,
            dtype =      tensor_dtype_str(self.model),
            total_size = total_size,
            sha256 =     sha256,
        )
        yield pb.ModelStream(header=hdr)

        with open(tmp_path, "rb") as f:
            while True:
                data = f.read(CHUNK_SIZE)
                if not data:
                    break
                yield pb.ModelStream(chunk=pb.ModelChunk(data=data))

def server(model):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))

    rpc.add_ModelDistributorServicer_to_server(
        ModelDistributorService(model), server
    )

    server.add_insecure_port("[::]:50051")
    server.start()

    print("ModelDistributor gRPC server rodando em 0.0.0.0:50051")
    server.wait_for_termination()
