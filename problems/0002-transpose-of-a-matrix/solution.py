import torch

def transpose_matrix(a) -> torch.Tensor:
    return torch.einsum("ij -> ji", a)