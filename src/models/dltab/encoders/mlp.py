# Modules -----------------------------------------------------------------------------------------------------------------#
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Optional
from torch  import nn

# Internal functions and utilities ----------------------------------------------------------------------------------------#
from src.models.dltab.utils.activations import select_activation

# MuliLayer Perceptron Regressor ------------------------------------------------------------------------------------------#
class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_layers:List[int]=[64, 32], out_features: int = 1,
                 activation    : str             = 'relu', 
                 dropout       : Optional[float] = 0.2, 
                 normalization : str             = 'batch',
                 bias          : bool            = True):
        """
        ___________________________________________________________________________________________________________________
        MultiLayer Perceptron Regressor for DLTabularRegressor.
        ___________________________________________________________________________________________________________________
        Parameters:
        -> in_features   (int)  : Mandatory. Number of input features.
        -> hidden_layers (List) : Optional. List with number of units per hidden layer.
        -> out_features  (int)  : Optional. Number of output features. Default is 1 (direct regression).
        -> activation    (str)  : Optional. Activation function to use ('relu', 'tanh', 'selu', etc.). Default is 'relu'.
        -> dropout       (float): Optional. Dropout rate between 0 and 1. If None, no dropout is applied. Default is 0.2.
        -> normalization (str)  : Optional. Normalization type ('batch', 'layer', or None). Default is 'batch'.
        -> bias          (bool) : Optional. Whether to include bias terms in Linear layers. Default is True.
        ___________________________________________________________________________________________________________________
        Notes:
        - The network is built dynamically based on the provided hidden_layers list.
        - Normalization layers are added after each Linear layer if specified.
        - Dropout is applied after activation functions if specified.
        - Implementation in PyTorch.
        ___________________________________________________________________________________________________________________
        """
        super().__init__()
        
        # Input validation ------------------------------------------------------------------------------------------------#
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")

        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        
        if dropout is not None and not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        
        # If using batch_norm, bias in Linear layers is redundant
        use_bias = bias and not (normalization == 'batch')
        
        # Store configuration ---------------------------------------------------------------------------------------------#
        self.in_features   = in_features
        self.hidden_layers = hidden_layers
        self.out_features  = out_features
        
        # Build MLP layers ------------------------------------------------------------------------------------------------#
        layers = []  # Use regular list, not nn.ModuleList
        prev   = in_features
        
        # Iterate hidden layers of the net
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h, bias=use_bias))
            
            # Flexible normalization layers
            if   (normalization == 'batch'): layers.append(nn.BatchNorm1d(h))
            elif (normalization == 'layer'): layers.append(nn.LayerNorm(h))
            
            layers.append(select_activation(activation))
            
            if dropout is not None and dropout > 0.0:
                if activation == 'selu':
                    layers.append(nn.AlphaDropout(dropout))
                else:
                    layers.append(nn.Dropout(dropout))
            
            prev = h
        
        # Append prediction layer
        layers.append(nn.Linear(prev, out_features, bias=use_bias))
        
        # Compile sequential model 
        self.mlp = nn.Sequential(*layers)

        # Initialize weights based on activation function -----------------------------------------------------------------#
        self._initialize_weights(activation=activation)

    def _initialize_weights(self, activation: str = 'relu') -> None:
        """Initialize layer weights using appropriate initialization based on activation function."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation == 'selu':
                    # LeCun normal initialization for self-normalizing networks
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def __repr__(self) -> str:
        """String representation of the Regressor."""
        return (f"{self.__class__.__name__}\n("
                f"in_features   ={self.in_features},\n"
                f"hidden_layers ={self.hidden_layers},\n"
                f"out_features  ={self.out_features})\n")
        
#--------------------------------------------------------------------------------------------------------------------------#