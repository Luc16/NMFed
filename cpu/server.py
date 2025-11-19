import grpc
from concurrent import futures
import io
import copy
import pandas as pd
import fl_pb2
import fl_pb2_grpc
import os
import time
import threading
from sparsity import TwoFourSparsifier # Needed for pickle to recognize class
import argparse

import torch
# Max message size (100MB) to handle model weights
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
OPTIONS = [
                ("grpc.max_receive_message_length", 128 * 1024 * 1024),  # 128 MB
                ("grpc.max_send_message_length",    128 * 1024 * 1024),
            ]

class FLServiceImpl(fl_pb2_grpc.FederatedServiceServicer):
    def __init__(self, initial_model_path, num_clients_per_round, output_csv):
        self.lock = threading.Lock()
        
        # Load initial sparse model
        print(f"Loading model from {initial_model_path}")
        self.global_state = torch.load(initial_model_path)
        
        self.num_clients_required = num_clients_per_round
        self.current_round = 1
        self.updates_buffer = [] # Stores tuples of (state_dict, num_samples)
        self.stop_flag = False
        self.csv_path = output_csv
        
        # Initialize CSV
        with open(self.csv_path, 'w') as f:
            f.write("Round,Client_ID,Train_Acc,Test_Acc\n")

    def GetGlobalModel(self, request, context):
        with self.lock:
            # Serialize state_dict to bytes
            buffer = io.BytesIO()
            torch.save(self.global_state, buffer)
            buffer.seek(0)
            
            return fl_pb2.ModelPacket(
                data=buffer.read(),
                round=self.current_round,
                stop=self.stop_flag
            )

    def SendUpdate(self, request, context):
        # Deserialize
        buffer = io.BytesIO(request.data)
        client_state = torch.load(buffer)
        
        with self.lock:
            print(f"Received update from Client {request.client_id} for Round {request.round}")
            
            # Reject stale updates
            if request.round != self.current_round:
                print(f"Stale update from Client {request.client_id}. Expected Round {self.current_round}, got {request.round}")
                return fl_pb2.Ack(success=False, message="Stale round")

            # Log metrics
            with open(self.csv_path, 'a') as f:
                f.write(f"{request.round},{request.client_id},{request.train_acc},{request.test_acc}\n")

            self.updates_buffer.append((client_state, request.num_samples))

            # Check if we can aggregate
            if len(self.updates_buffer) >= self.num_clients_required:
                self._aggregate_and_step()
                
            return fl_pb2.Ack(success=True, message="Accepted")

    def _aggregate_and_step(self):
        print(f"Aggregating Round {self.current_round}...")

        total_samples = sum(samples for _, samples in self.updates_buffer)

        # Copy structure from the first client
        first_state, _ = self.updates_buffer[0]
        avg_state = copy.deepcopy(first_state)

        # Zero floating-point tensors
        for key, tensor in avg_state.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                avg_state[key] = torch.zeros_like(tensor)

        # FedAvg aggregation
        for state, samples in self.updates_buffer:
            weight = samples / total_samples
            for key, tensor in state.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point():
                    avg_state[key] += tensor * weight

        self.global_state = avg_state
        self.updates_buffer = []
        self.current_round += 1
        print(f"Round {self.current_round} Started.")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=OPTIONS)
    file_name = "results.csv"
    counter = 1
    while os.path.isfile(file_name):
        file_name = f"results{counter}.csv"
        counter += 1


    parse = argparse.ArgumentParser(description="Initialize and prune a ResNet18 model with 2:4 sparsity.")
    parse.add_argument('--no-sparsity', type=bool, default=False, help='If set to True, do not apply sparsity.')
    args = parse.parse_args()

    model_path = "models/initial_sparse_model.pt" if not args.no_sparsity \
                else "models/initial_dense_model.pt"
    # Hardcoded configuration for demo; ideally args
    fl_pb2_grpc.add_FederatedServiceServicer_to_server(
        FLServiceImpl(model_path, num_clients_per_round=4, output_csv=file_name), 
        server
    )
    server.add_insecure_port('[::]:50051')
    print("Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
