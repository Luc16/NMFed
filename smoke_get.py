# smoke_get.py
import io, hashlib, grpc, torch
from comunication_pb2 import ModelRequest
import comunication_pb2_grpc as rpc

def sha256_bytes(b: bytes):
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def main(host="127.0.0.1:50051"):
    with grpc.insecure_channel(host) as ch:
        stub = rpc.ModelDistributorStub(ch)
        header, buf = None, bytearray()
        for part in stub.GetModel(ModelRequest(name="resnet18-ibn-v1", version="any")):
            if part.WhichOneof("part") == "header":
                header = part.header
                print("[header]")
                print("  model_name :", header.model_name)
                print("  version    :", header.version)
                print("  arch       :", header.arch)
                print("  total_size :", header.total_size)
                print("  sha256     :", header.sha256)
            else:
                buf.extend(part.chunk.data)

    raw = bytes(buf)
    calc = sha256_bytes(raw)
    print("received bytes:", len(raw), "sha256:", calc)
    print("checksum OK?" , calc == header.sha256)

    # tenta desserializar como state_dict
    sd = torch.load(io.BytesIO(raw), map_location="cpu")
    print("state_dict keys:", len(sd.keys()))

if __name__ == "__main__":
    main()
