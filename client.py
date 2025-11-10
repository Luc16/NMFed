# client.py
import io, time, argparse, sys, math, datetime, os, re
from contextlib import redirect_stdout, redirect_stderr

import ray
import grpc
import torch
from torch import amp

from comunication_pb2 import ModelRequest, ModelUpdateRequest
import comunication_pb2_grpc as rpc

from clientsUtils import (
    build_resnet18_padded128, make_cifar10_loaders,
    sd_to_bytes, bytes_to_sd, sha256_bytes,
    train_locally, evaluate_top1,
)

# ------------------ util de log ------------------
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data); f.flush()
    def flush(self):
        for f in self.files: f.flush()

def make_log_path(client_id: str, round_idx: int = 0):
    LOG_DIR = "logs_server"
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"log_client_{client_id}_round{round_idx}_{ts}.txt"
    return os.path.join(os.getcwd(), LOG_DIR, fname)

# ------------------ download do modelo ------------------
def fetch_model(host: str, timeout_s: float = 300.0):
    print(f"[client] conectando a {host}…")
    ch = grpc.insecure_channel(host)
    grpc.channel_ready_future(ch).result(timeout=timeout_s)
    print("[client] canal pronto, requisitando modelo…")

    stub = rpc.ModelDistributorStub(ch)
    header, buf, total = None, bytearray(), None

    for part in stub.GetModel(ModelRequest(name="resnet18-ibn-v1", version="any"), timeout=timeout_s):
        which = part.WhichOneof("part")
        if which == "header":
            header = part.header
            total = header.total_size
            print(f"[client] header: version={header.version} arch={header.arch} size={total} sha256={header.sha256[:12]}…")
        else:
            buf.extend(part.chunk.data)
            if total and (len(buf) == total or len(buf) % (10*1024*1024) == 0):
                print(f"[client] recebido {len(buf)}/{total} bytes ({100*len(buf)/total:.1f}%)")

    if header is None:
        raise RuntimeError("no header")
    print("[client] download OK")
    return header, bytes(buf)

# ------------------ ator Ray ------------------
@ray.remote(num_gpus=1)
class Client:
    def __init__(self, client_id: str, server_host: str, data_dir: str, batch_size: int, steps: int):
        self.client_id   = client_id
        self.server_host = server_host
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.steps       = steps

        # device via Ray
        ray_gpu_ids = ray.get_gpu_ids()
        if ray_gpu_ids:
            torch.cuda.set_device(0)              # Ray expõe as GPUs alocadas como [0..]
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        print(f"[Client {client_id}] device={self.device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
        self.train_loader, self.val_loader = make_cifar10_loaders(self.data_dir, self.batch_size)

    def one_round(self, round_idx: int = 0):
        log_path = make_log_path(self.client_id, round_idx)
        orig_out, orig_err = sys.stdout, sys.stderr
        with open(log_path, "w", buffering=1, encoding="utf-8") as lf:
            sys.stdout = Tee(orig_out, lf)
            sys.stderr = Tee(orig_err, lf)
            try:
                # 1) baixa pesos
                t_fetch0 = time.time()
                header, raw = fetch_model(self.server_host)
                t_fetch1 = time.time()
                download_time_s = t_fetch1 - t_fetch0
                model_bytes_download = len(raw)

                if sha256_bytes(raw) != header.sha256:
                    return {"client": self.client_id, "ok": False, "msg": "checksum mismatch", "log_path": log_path}

                # 2) constrói modelo e carrega pesos (CPU -> GPU -> dtype -> troca fc)
                model = build_resnet18_padded128()                  # deve bater com 'arch' do servidor
                sd = bytes_to_sd(raw, device="cpu")                 # SEMPRE CPU
                model.load_state_dict(sd, strict=True)

                if self.device.startswith("cuda"):
                    model.to("cuda:0")
                else:
                    model.to("cpu")

                use_ampere = (self.device.startswith("cuda") and torch.cuda.get_device_capability(0) >= (8, 0))
                if use_ampere:
                    # dtype BF16 recomendado no H100
                    model = model.to(dtype=torch.bfloat16)

                    # TorchAO 0.14.1: trocar a fc MANUALMENTE e garantir mesmo device/dtype
                    from torchao.sparsity.training import SemiSparseLinear
                    fc_old = model.fc
                    target_dev  = next(model.parameters()).device
                    target_dtype= next(model.parameters()).dtype

                    fc_new = SemiSparseLinear(
                        fc_old.in_features, fc_old.out_features,
                        bias=(fc_old.bias is not None)
                    ).to(device=target_dev, dtype=target_dtype)

                    with torch.no_grad():
                        fc_new.weight.copy_(fc_old.weight.to(device=target_dev, dtype=target_dtype))
                        if fc_old.bias is not None:
                            fc_new.bias.copy_(fc_old.bias.to(device=target_dev, dtype=target_dtype))

                    model.fc = fc_new
                    print("[client] fc -> SemiSparseLinear (device:", target_dev, ", dtype:", target_dtype, ")")
                else:
                    print("[client] Sem Ampere: mantendo camadas densas (FP32).")

                dbg_p = next(model.parameters())
                print(f"[dbg] model device={dbg_p.device} dtype={dbg_p.dtype}")

                # 3) avalia antes (autocast API nova)
                use_amp = self.device.startswith("cuda")
                with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    acc_before = evaluate_top1(model, self.val_loader, device=self.device)
                print(f"[client {self.client_id}] acc_before={acc_before:.4f}")

                # 4) treino local
                t_train0 = time.time()
                steps_total = int(self.steps)
                steps_per_epoch = max(1, len(self.train_loader))
                epochs = max(1, math.ceil(steps_total / steps_per_epoch))

                steps_left = steps_total
                seen_total = 0
                acc_per_epoch = []

                csv_path = make_log_path(self.client_id, round_idx).replace(".txt", ".csv")
                with open(csv_path, "w", encoding="utf-8") as cf:
                    cf.write("epoch, steps_this, seen_ep, acc, epoch_time\n")

                for ep in range(1, epochs + 1):
                    steps_this = min(steps_left, steps_per_epoch)
                    t0 = time.time()
                    model, seen_ep = train_locally(model, self.train_loader, steps=steps_this, device=self.device)
                    t1 = time.time()
                    epoch_time = t1 - t0
                    seen_total += int(seen_ep)

                    with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                        acc_ep = evaluate_top1(model, self.val_loader, device=self.device)

                    acc_per_epoch.append(float(acc_ep))
                    print(f"[client {self.client_id}] epoch {ep}/{epochs} steps_this={steps_this} seen_ep={seen_ep} acc={acc_ep:.4f}")

                    with open(csv_path, "a", encoding="utf-8") as cf:
                        cf.write(f"{ep}, {steps_this}, {int(seen_ep)}, {float(acc_ep):.6f}, {epoch_time:.4f}\n")

                    steps_left -= steps_this
                    if steps_left <= 0: break

                t_train1 = time.time()
                train_time_s = t_train1 - t_train0
                steps_per_s = (steps_total / train_time_s) if train_time_s > 0 else float("inf")
                samples_per_s = (seen_total / train_time_s) if train_time_s > 0 else float("inf")

                # 5) avalia depois
                with amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    acc_after = acc_per_epoch[-1] if acc_per_epoch else evaluate_top1(model, self.val_loader, device=self.device)
                print(f"[client {self.client_id}] acc_after={acc_after:.4f}")

                # 6) envia update
                update_bytes_blob = sd_to_bytes(model.state_dict())
                update_bytes_size = len(update_bytes_blob)
                t_up0 = time.time()
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
                        num_samples=seen_total,
                        state_bytes=update_bytes_blob,
                        base_version=header.version,
                    ))
                t_up1 = time.time()
                upload_time_s = t_up1 - t_up0

                return {
                    "client": self.client_id,
                    "ok": bool(ack.ok),
                    "msg": ack.msg,
                    "server_version": ack.server_version,
                    "seen": int(seen_total),
                    "acc_before": float(acc_before),
                    "acc_after": float(acc_after),
                    "acc_per_epoch": acc_per_epoch,
                    "log_path": log_path,
                    "model_bytes_download": int(model_bytes_download),
                    "download_time_s": float(download_time_s),
                    "update_bytes_upload": int(update_bytes_size),
                    "upload_time_s": float(upload_time_s),
                    "train_time_s": float(train_time_s),
                    "steps_per_s": float(steps_per_s),
                    "samples_per_s": float(samples_per_s),
                }
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err

# ------------------ launcher ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_host", default="127.0.0.1:50051")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--data_dir", default="/root/data")
    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)
    actors = [
        Client.remote(f"client-{i+1}", args.server_host, args.data_dir, args.batch_size, args.steps)
        for i in range(args.clients)
    ]

    for r in range(3):
        print(f"\n=== ROUND {r+1} ===")
        outs = ray.get([a.one_round.remote(round_idx=r+1) for a in actors])
        for o in outs: print(o)
        time.sleep(0.5)

    ray.shutdown()

if __name__ == "__main__":
    main()
