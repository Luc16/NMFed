import torch
import torch.nn.utils.prune as prune
import warnings

class TwoFourSparsifier(prune.BasePruningMethod):
    """
    Custom pruning method to enforce 2:4 structured sparsity.
    
    Args:
        method (str): Strategy to select the 2 weights to keep.
            - 'topk': (Default) Keep largest magnitude weights.
            - 'random': Keep 2 random weights (Control).
            - 'stochastic': Sample 2 weights based on probability of magnitude.
            - 'grad': Keep weights with highest Sensitivity (|W * Grad|).
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, method='topk'):
        self.method = method
        valid_methods = ['topk', 'random', 'stochastic', 'grad']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        
        # 1. Shape Validation (must be divisible by 4)
        if t.numel() % 4 != 0:
            return mask
            
        shape_orig = t.shape
        t_reshaped = t.reshape(-1, 4)
        
        # 2. Calculate Scores based on Method
        # We want to select indices based on a "score" tensor
        
        if self.method == 'random':
            # Generate random noise as scores
            scores = torch.rand_like(t_reshaped)
            _, indices = torch.topk(scores, 2, dim=1)
            
        elif self.method == 'stochastic':
            # Convert weights to probabilities and sample
            w_abs = t_reshaped.abs()
            # Add epsilon to prevent division by zero
            probs = w_abs / (w_abs.sum(dim=1, keepdim=True) + 1e-6)
            # Sample 2 indices based on probability (no replacement)
            indices = torch.multinomial(probs, 2, replacement=False)
            
        elif self.method == 'grad':
            # Sensitivity = |Weight * Gradient|
            # Check if gradients exist
            if t.grad is not None:
                grad_reshaped = t.grad.reshape(-1, 4)
                scores = (t_reshaped * grad_reshaped).abs()
                _, indices = torch.topk(scores, 2, dim=1)
            else:
                # Fallback if user forgot to run backward()
                warnings.warn("Pruning method 'grad' selected, but t.grad is None. Falling back to 'topk'.")
                scores = t_reshaped.abs()
                _, indices = torch.topk(scores, 2, dim=1)
                
        else: # Default: 'topk'
            scores = t_reshaped.abs()
            _, indices = torch.topk(scores, 2, dim=1)

        # 3. Create and Apply Mask
        new_mask = torch.zeros_like(t_reshaped)
        new_mask.scatter_(1, indices, 1)
        
        return new_mask.view(shape_orig)

def apply_2_4_sparsity(module, method='topk'):
    """
    Helper to apply the 2:4 sparsifier to a layer.
    Args:
        module: The PyTorch layer (e.g., nn.Linear)
        method: 'topk', 'random', 'stochastic', or 'grad'
    """
    TwoFourSparsifier.apply(module, 'weight', method=method)
    return module
