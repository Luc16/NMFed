import torch
import torch.nn.utils.prune as prune

class TwoFourSparsifier(prune.BasePruningMethod):
    """
    Custom pruning method to enforce 2:4 structured sparsity.
    Selects the top-2 weights by magnitude in every block of 4.
    """
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # Ensure tensor shape is compatible (divisible by 4)
        if t.numel() % 4!= 0:
            return mask
            
        # Reshape to [N, 4] to process blocks of 4
        shape_orig = t.shape
        t_reshaped = t.abs().reshape(-1, 4)
        
        # Find top 2 indices in each block
        _, indices = torch.topk(t_reshaped, 2, dim=1)
        
        # Create mask
        new_mask = torch.zeros_like(t_reshaped)
        new_mask.scatter_(1, indices, 1)
        
        return new_mask.view(shape_orig)

def apply_2_4_sparsity(module):
    """Helper to apply the 2:4 sparsifier to a layer."""
    TwoFourSparsifier.apply(module, 'weight')
    return module
