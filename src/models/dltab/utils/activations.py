# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn.functional as F

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Optional
from torch  import nn

# Custom GeGLU activation -------------------------------------------------------------------------------------------------#
class GEGLU(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Custom Torch implemetation of Gated Exponential Linear Unit (GeGLU) Activation Function.
    ________________________________________________________________________________________________________________________
    References:
    - "GLU Variants Improve Transformer" - Shazeer, 2020.
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[self.dim] % 2 != 0:
            raise RuntimeError(
                f"GeGLU requires even dimension for splitting, "
                f"but got shape {x.shape} at dim={self.dim}"
            )
        
        x, gates = x.chunk(2, dim=self.dim)
        return x * F.gelu(gates)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"

# Activation functions ----------------------------------------------------------------------------------------------------#
def select_activation(activation: str):
    """Selector of activation functions based on string identifier."""
    activations = {
        'relu'      : nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'tanh'      : nn.Tanh,
        'sigmoid'   : nn.Sigmoid,
        'elu'       : nn.ELU,
        'gelu'      : nn.GELU,
        'silu'      : nn.SiLU,
        'geglu'     : GEGLU
    }
    
    activation_lower = activation.lower()
    if activation_lower not in activations:
        raise ValueError(f"Activation function '{activation}' is not supported. Choose from {list(activations.keys())}.")   
    
    return activations[activation_lower]()

#---------------------------------------------------------------------------------------------------------------------------#