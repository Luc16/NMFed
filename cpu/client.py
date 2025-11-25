import argparse
import ray
import grpc
import fl_pb2
import fl_pb2_grpc
import io
import numpy as np
import time
import os
from sparsity import apply_2_4_sparsity
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Max message size for gRPC
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
OPTIONS = [
    ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ("grpc.max_send_message_length", 128 * 1024 * 1024),
]

@ray.remote(num_gpus=0) # Share GPU among clients
class FLClientActor:
    def __init__(self, client_id, data_indices, server_address, epochs, batch_size, no_sparsity=False, pruning_method='topk'):
        self.client_id = client_id
        self.server_address = server_address
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Prepare Data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        
        # Load full dataset (Download=False because main() downloads it)
        # We assume the data is on a shared file system or local node
        try:
            full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        except RuntimeError:
            # Fallback: If data isn't found (e.g. distinct nodes), try downloading
            full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        # Split into train/test for this client
        split_idx = int(len(data_indices) * 0.8)
        self.train_set = Subset(full_dataset, data_indices[:split_idx])
        self.test_set = Subset(full_dataset, data_indices[split_idx:])
        
        # 2. Prepare Model Architecture
        self.model = torchvision.models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)
        
        # CRITICAL: Apply sparsity hooks to create buffers (this will be overwritten by the server model)
        if not no_sparsity:
            for name, module in self.model.named_modules():
                 if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if module.weight.numel() % 4 == 0:
                        apply_2_4_sparsity(module, method=pruning_method)
        
        self.model.to(self.device)

    def run_round(self):
        # 1. Connect to Server
        channel = grpc.insecure_channel(self.server_address, options=OPTIONS)
        stub = fl_pb2_grpc.FederatedServiceStub(channel)
        
        try:
            # 2. Get Global Model
            response = stub.GetGlobalModel(fl_pb2.Empty())
            if response.stop:
                return False
            
            # Deserialize and Load
            buffer = io.BytesIO(response.data)
            state_dict = torch.load(buffer)
            self.model.load_state_dict(state_dict)
            
            print(f"[Client {self.client_id}] Training Round {response.round}...")
            
            # 3. Train (SGD on weight_orig)
            train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            self.model.train()
            for epoch in range(self.epochs):
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(imgs)
                    loss = criterion(output, labels)
                    loss.backward()
                    
                    # Optimizer updates 'weight_orig'. 
                    # 'weight' is recomputed as weight_orig * weight_mask in forward pass.
                    optimizer.step()

            # 4. Evaluate
            train_acc = self._evaluate(self.train_set)
            test_acc = self._evaluate(self.test_set)
            
            # 5. Send Update
            out_buffer = io.BytesIO()
            torch.save(self.model.state_dict(), out_buffer)
            out_buffer.seek(0)
            
            update = fl_pb2.ModelUpdate(
                client_id=self.client_id,
                round=response.round,
                data=out_buffer.read(),
                num_samples=len(self.train_set),
                train_acc=train_acc,
                test_acc=test_acc
            )
            
            stub.SendUpdate(update)
            return True
            
        except Exception as e:
            print(f"Client {self.client_id} Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out = self.model(imgs)
                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--server", type=str, default="localhost:50051")
    parser.add_argument('--no-sparsity', type=bool, default=False, help='If set to True, do not apply sparsity.')
    parser.add_argument('--method', type=str, default='topk', help='Sparsity method: topk, random, stochastic, grad')
    args = parser.parse_args()

    ray.init()

    # FIX: Download Data Once in Main Driver
    print("Downloading CIFAR-10 dataset once...")
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Partition Data
    print("Partitioning Data...")
    total_images = 50000
    indices = np.arange(total_images)
    np.random.shuffle(indices)
    client_data_splits = np.array_split(indices, args.num_clients)

    # Instantiate Actors
    print(f"Initializing {args.num_clients} Clients...")
    clients = []
    for i in range(args.num_clients):
        c = FLClientActor.remote(
            client_id=i,
            data_indices=client_data_splits[i],
            server_address=args.server,
            epochs=args.epochs,
            batch_size=args.batch_size,
            no_sparsity=args.no_sparsity,
            pruning_method=args.method
        )
        clients.append(c)

    # Orchestration Loop
    print(f"Starting training for {args.rounds} rounds...")
    for r in range(args.rounds):
        print(f"--- Global Round {r+1} ---")
        
        # Trigger all clients
        futures = [c.run_round.remote() for c in clients]
        
        # Wait for completion
        results = ray.get(futures)
        
        # Simple synchronization
        time.sleep(2) 

    print("Training finished.")

if __name__ == "__main__":
    main()
