# ray_clients_cifar.py
import io, time, argparse
import re

import ray
import grpc
import torch
from comunication_pb2 import ModelRequest, ModelUpdateRequest
import comunication_pb2_grpc as rpc

from clientsUtils import (
    build_resnet18_padded128, make_cifar10_loaders,
    sd_to_bytes, bytes_to_sd, sha256_bytes,
    train_locally, evaluate_top1,
)

def fetch_model(host: str, timeout_s: float = 300.0):
    print(f"[client] conectando a {host}…")
    ch = grpc.insecure_channel(host)
    grpc.channel_ready_future(ch).result(timeout=timeout_s)
    print("[client] canal pronto, requisitando modelo…")

    stub = rpc.ModelDistributorStub(ch)
    header, buf = None, bytearray()
    total = None

    for part in stub.GetModel(ModelRequest(name="resnet18-ibn-v1", version="any"), timeout=timeout_s):
        which = part.WhichOneof("part")
        if which == "header":
            header = part.header
            total = header.total_size
            print(f"[client] header: version={header.version} arch={header.arch} size={total} sha256={header.sha256[:12]}…")
        else:
            buf.extend(part.chunk.data)
            if total and (len(buf)==total or len(buf) % (10*1024*1024) == 0):
                print(f"[client] recebido {len(buf)}/{total} bytes ({100*len(buf)/total:.1f}%)")
    if header is None:
        raise RuntimeError("no header")
    print("[client] download OK")
    return header, bytes(buf)

@ray.remote
class Client:
    def __init__(self, client_id: str, server_host: str, data_dir: str, batch_size: int, steps: int):
        self.client_id  = client_id
        self.server_host= server_host
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.steps      = steps
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        # cada ator prepara seus loaders; se usar o mesmo data_dir, torchvision reuseia os arquivos baixados
        self.train_loader, self.val_loader = make_cifar10_loaders(self.data_dir, self.batch_size)

    def one_round(self):
        # 1) baixa modelo global
        header, raw = fetch_model(self.server_host)
        if sha256_bytes(raw) != header.sha256:
            return {"client": self.client_id, "ok": False, "msg": "checksum mismatch"}

        # 2) reconstrói e carrega pesos
        model = build_resnet18_padded128()
        print("Ola")

        model.load_state_dict(bytes_to_sd(raw, device="cpu"), strict=True)

        # 3) avalia antes
        acc_before = evaluate_top1(model, self.val_loader, device=self.device)

        # 4) fine-tune local com CIFAR-10
        model, seen = train_locally(model, self.train_loader, steps=self.steps, device=self.device)

        # 5) avalia depois (opcional)
        acc_after = evaluate_top1(model, self.val_loader, device=self.device)

        print("Oi")

        # 6) envia update com base_version
        with grpc.insecure_channel(
            self.server_host,
            options=[
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),
                ("grpc.max_send_message_length",    128 * 1024 * 1024),
            ],
        ) as ch:
            stub = rpc.ModelDistributorStub(ch)
            ack = stub.SubmitUpdate(ModelUpdateRequest(
                client_id=self.client_id,
                num_samples=seen,
                state_bytes=sd_to_bytes(model.state_dict()),
                base_version=header.version,
            ))

        return {
            "client": self.client_id,
            "ok": bool(ack.ok),
            "msg": ack.msg,
            "server_version": ack.server_version,
            "seen": int(seen),
            "acc_before": float(acc_before),
            "acc_after": float(acc_after),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_host", default="127.0.0.1:50051")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)     # batches por round/cliente
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--data_dir", default="/root/data")   # cache do CIFAR-10
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)
    actors = [Client.remote(f"client-{i+1}", args.server_host, args.data_dir, args.batch_size, args.steps)
              for i in range(args.clients)]

    # Roda alguns rounds; servidor faz FedAvg quando atingir o min_updates_per_round
    for r in range(3):
        print(f"\n=== ROUND {r+1} ===")
        outs = ray.get([a.one_round.remote() for a in actors])
        for o in outs:
            print(o)
        time.sleep(0.5)

    ray.shutdown()

if __name__ == "__main__":
    main()
