import grpc
host="63.141.33.85:22067"
ch = grpc.insecure_channel(host)
try:
    grpc.channel_ready_future(ch).result(timeout=10)
    print("gRPC pronto em", host)
except Exception as e:
    print("gRPC falhou:", e)