# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn

# External functions and utilities ----------------------------------------------------------------------------------------#

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.atte.layers.attention import MultiHeadAttentionHandler

# Normalization module ----------------------------------------------------------------------------------------------------#
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn   = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Custom feed-foward neural network ---------------------------------------------------------------------------------------#
class FeedForward(nn.Module):
    def __init__(self, dim:int, embed_dim:int, dropout:float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
                        nn.Linear(dim, embed_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim, dim),
                        nn.Dropout(dropout)
                    )
    def forward(self, x):
        return self.net(x)

# Linear layer for regression ---------------------------------------------------------------------------------------------#
class TokenRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.norm         = nn.LayerNorm(input_dim)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.output_layer(self.norm(x))

# Transformer Module with multi head attention ----------------------------------------------------------------------------#
class Transformer(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Transformer module with multi-head self-attention and feed-forward layers.
    ________________________________________________________________________________________________________________________
    Parameters:
        head_dim    (int):    Dimension of each attention head.
        num_heads   (int):    Number of attention heads.
        attn_layers (int):    Number of transformer layers (default: 1).
        dropout     (float):  Dropout rate (default: 0.0).
    ________________________________________________________________________________________________________________________
    Returns:
        Transformer instance (nn.Module)
    ________________________________________________________________________________________________________________________
    Example:
        >>> model = Transformer(head_dim=64, num_heads=4, attn_layers=2, dropout=0.1)
        >>> out = model(x, mask, causal_mask=False)
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, head_dim: int, num_heads: int, attn_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # Compute input dimension for each layer
        self.input_dim   = head_dim * num_heads
        self.attn_layers = attn_layers
        self.dropout     = dropout

        # Build transformer layers: each layer has (attention, feed-forward)
        self.layers = nn.ModuleList([])
        for _ in range(self.attn_layers):
            self.layers.append(nn.ModuleList([
                # PreNorm applies LayerNorm before the function
                PreNorm(self.input_dim, MultiHeadAttentionHandler(input_dim = self.input_dim, 
                                                                  head_dim  = head_dim, 
                                                                  num_heads = num_heads, 
                                                                  dropout   = dropout)),

                PreNorm(self.input_dim, FeedForward(self.input_dim, 2 * self.input_dim, 
                                                    dropout = self.dropout))
                ]))

    def get_input_dim(self) -> int:
        """Return the input dimension expected by the first attention layer."""
        return self.layers[0][0].fn.get_input_dim()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Forward pass for the transformer.
        ____________________________________________________________________________________________________________________
        Parameters:
            x           (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
            mask        (torch.Tensor): Attention mask tensor.
            causal_mask (bool):         Whether to apply causal (autoregressive) masking.
        ____________________________________________________________________________________________________________________
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, input_dim).
        ____________________________________________________________________________________________________________________
        """
        for idx, (attn, ff) in enumerate(self.layers):
            # Self-attention with residual connection
            x = attn(x=x, mask=mask, causal_mask=causal_mask) + x
            # Feed-forward with residual connection
            x = ff(x) + x
        return x

#--------------------------------------------------------------------------------------------------------------------------#