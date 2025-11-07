# train_with_torchao.py
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchao.sparsity.training import (
    SemiSparseLinear,
    swap_linear_with_semi_sparse_linear,
)


def run_torchao_proof_of_concept():
    """
    Proof-of-concept de treino com sparsidade dinâmica 2:4 do torchao.
    - Em GPUs Ampere+ (cc >= 8.0): troca nn.Linear -> SemiSparseLinear e roda em FP16.
    - Em CPU (ou GPU < Ampere): pula a troca (treino denso) porque o kernel 2:4 é só CUDA.
    """
    print("--- TorchAO Dynamic Sparse Training Proof of Concept ---")

    use_cuda = torch.cuda.is_available()
    cc = torch.cuda.get_device_capability(0) if use_cuda else (0, 0)
    has_ampere = use_cuda and (cc >= (8, 0))

    if not has_ampere:
        print("\nWARNING:")
        print("  - Sem GPU Ampere (cc>=8.0) disponível. O kernel 2:4 do torchao só roda em CUDA Ampere+.")
        print("  - Vou pular a troca para SemiSparseLinear e treinar densamente (sem speedup de sparsidade).\n")

    # Dispositivo e dtype
    device = torch.device("cuda" if use_cuda else "cpu")
    # GPU Ampere: FP16; CPU: BF16 é ok p/ testes, mas FP32 é o mais compatível. Vamos usar BF16 aqui.
    dtype = torch.float16 if has_ampere else torch.bfloat16 if not use_cuda else torch.float32

    # 1) Carrega ResNet18 pré-treinada
    print("Loading pre-trained ResNet18 model...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # 2) Adapta para problema de 10 classes
    num_ftrs = model.fc.in_features

    # Exigência do kernel 2:4: dimensões múltiplas de 128 na camada Linear.
    # Em vez de 10, inflamos para 128 e depois "fatiamos" no loss.
    padded_out_features = 128
    model.fc = nn.Linear(num_ftrs, padded_out_features)

    # Move para device/dtype desejado
    model = model.to(device).to(dtype)
    print(f"Model adapted. Final layer output padded to {padded_out_features} features.")

    # 3) (Opcional) Troca nn.Linear -> SemiSparseLinear (apenas se houver CUDA Ampere)
    if has_ampere:
        print("\nBuilding config to swap nn.Linear layers...")
        sparse_config = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"  - Targeting layer for swap: '{name}'")
                sparse_config[name] = SemiSparseLinear

        print("Swapping nn.Linear layers with torchao.sparsity.training.SemiSparseLinear...")
        swap_linear_with_semi_sparse_linear(model, config=sparse_config)

        print("\nVerification after swap:")
        for name, module in model.named_modules():
            if isinstance(module, SemiSparseLinear):
                print(f"  - Layer '{name}' is now a SemiSparseLinear layer.")
    else:
        print("\nCPU or non-Ampere GPU detected: skipping SemiSparseLinear swap.")

    # # 4) Otimizador / loss
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.CrossEntropyLoss()
    #
    # # 5) Dados dummy
    # dummy_input = torch.randn(16, 3, 224, 224, device=device, dtype=dtype)
    # dummy_target = torch.randint(0, 10, (16,), device=device)
    #
    # # 6) Treininho rápido
    # print("\nStarting a short training loop (5 steps)...")
    # model.train()
    # start_time = time.time()
    # for i in range(5):
    #     optimizer.zero_grad()
    #     output = model(dummy_input)
    #
    #     # Saída tem 128; usamos só 10 para o loss
    #     sliced_output = output[:, :10].to(torch.float32)  # CrossEntropyLoss espera float32 em geral
    #     loss = criterion(sliced_output, dummy_target)
    #
    #     loss.backward()
    #     optimizer.step()
    #     print(f"  Step {i+1}/5, Loss: {loss.item():.4f}")
    # end_time = time.time()
    #
    # print(f"\nTraining loop completed in {end_time - start_time:.2f} seconds.")
    # print("\n--- Proof of Concept Complete ---")
    # print("O modelo com (ou sem) torchao executou um loop de treino de ponta a ponta.")
    
    # Return model on CPU for easier saving/loading
    return model.cpu()

if __name__ == "__main__":
    # Run the process to get the trained/pruned model
    model = run_torchao_proof_of_concept()
    
    SAVED_MODEL_PATH = "pruned_model.pt"
    print(f"\nSaving model to {SAVED_MODEL_PATH}...")
    
    # Save the entire model object using torch.save
    # This makes loading easier, but requires the class definitions
    # (like SemiSparseLinear) to be available in the environment when loading.
    torch.save(model.state_dict(), SAVED_MODEL_PATH)
    
    print(f"Model saved successfully.")
    print(f"To run the server, you can now create a new script that loads this file and calls server().")
    
    # We no longer start the server here
    # server(model)
