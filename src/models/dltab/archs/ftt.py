# Modules -----------------------------------------------------------------------------------------------------------------#
import math
import torch
import torch.nn as nn

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Optional

# Internal functions and utilities ----------------------------------------------------------------------------------------#
from src.models.dltab.utils.activations import select_activation

_GLU_ACTIVATIONS = {"glu", "reglu", "geglu"}

# Implementation of a transformer encoder layer ---------------------------------------------------------------------------#
class CustomTEL(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Pre-norm / Post-norm Transformer Encoder Layer with support for standard and GLU-type activations.
    ________________________________________________________________________________________________________________________
    Reference:
    - Original Implementation:
      https://github.com/OpenTabular/DeepTab/blob/main/deeptab/arch_utils/transformer_utils.py
    ________________________________________________________________________________________________________________________
    """
    # Initialize the class ------------------------------------------------------------------------------------------------#
    def __init__(self, d_model : int, n_heads : int, dim_ff : int, dropout : float, activation_layer: str,
                 layer_norm_eps : float,
                 norm_first     : bool,
                 bias           : bool):
        super().__init__()
        # Characteristics of the encoder
        self.norm_first = norm_first
        self.activation = select_activation(activation_layer)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, bias=bias, batch_first=True)

        # Feed-forward network. For GLU-type activations linear1 projects to 2*dim_ff
        ff_inner     = dim_ff * 2 if activation_layer in _GLU_ACTIVATIONS else dim_ff
        self.linear1 = nn.Linear(d_model, ff_inner, bias=bias)
        self.linear2 = nn.Linear(dim_ff,  d_model,  bias=bias)

        # Norms and dropout
        self.norm1    = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2    = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # [Helper] Feed-forward block -----------------------------------------------------------------------------------------#
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.activation(self.linear1(x))))

    # Forward pass of the CustomTEL ---------------------------------------------------------------------------------------#
    def forward(self, src              : torch.Tensor,
                      src_mask         : torch.Tensor = None,
                      src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:

        # Pre-norm (norm_first=True)
        if self.norm_first:
            normed = self.norm1(src)
            src    = src + self.dropout1(self.self_attn(normed, normed, normed,
                                                        attn_mask        = src_mask,
                                                        key_padding_mask = src_key_padding_mask)[0])
            src    = src + self._ff_block(self.norm2(src))

        # Post-norm (norm_first=False)
        else:
            src = self.norm1(src + self.dropout1(self.self_attn(src, src, src,
                                                                attn_mask        = src_mask,
                                                                key_padding_mask = src_key_padding_mask)[0]))
            src = self.norm2(src + self._ff_block(src))

        return src


# Numerical Feature Tokenizer ---------------------------------------------------------------------------------------------#
class NumericalTokenizer(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Maps a flat numerical feature vector to a sequence of d_token-dimensional token embeddings,
    prepending a learnable [CLS] token.
    ________________________________________________________________________________________________________________________
    For each feature i:  token_i = x_i * W_i + b_i   (W_i, b_i ∈ R^{d_token})
    CLS token:           W_CLS * 1.0 = W_CLS         (a learned embedding with no feature-scaling)
    ________________________________________________________________________________________________________________________
    Reference:
    - "Revisiting Deep Learning Models for Tabular Data" — Gorishniy et al., NeurIPS 2021.
      https://arxiv.org/abs/2106.11959
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, in_features: int, d_token: int, token_bias: bool = True):
        super().__init__()
        # Row 0 is the CLS embedding; rows 1..in_features are per-feature weight vectors
        self.weight = nn.Parameter(torch.empty(in_features + 1, d_token))
        self.bias   = nn.Parameter(torch.empty(in_features, d_token)) if token_bias else None
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        """Total number of output tokens: in_features + 1 (the [CLS] token)."""
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (batch, in_features) — raw numerical features.
        Returns:
            tokens (Tensor): (batch, in_features+1, d_token) — [CLS] first, then feature tokens.
        """
        ones   = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_aug  = torch.cat([ones, x], dim=1)             
        tokens = self.weight[None] * x_aug[:, :, None]   

        if self.bias is not None:
            tokens[:, 1:] = tokens[:, 1:] + self.bias

        return tokens


# FTTransformer Regressor -------------------------------------------------------------------------------------------------#
class FTTRegressor(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Custom Feature Tokenizer + Transformer (FT-Transformer) Regressor for tabular numerical data.
    ________________________________________________________________________________________________________________________
    Architecture:
    1. Numerical Tokenizer  : each feature is embedded as a d_token-dimensional vector; a [CLS] token is prepended.
    2. Transformer Encoder  : L stacked CustomTEL layers (pre-norm or post-norm) with optional GLU-type activations.
    3. Regression Head      : a linear layer applied to the final [CLS] token → scalar / multi-output prediction.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> in_features    (int)   : Mandatory. Number of numerical input features.
    -> out_features   (int)   : Optional. Number of output targets. Default is 1.
    -> d_token        (int)   : Optional. Token (embedding) dimension. Must be divisible by n_heads. Default is 64.
    -> n_layers       (int)   : Optional. Number of transformer encoder layers. Default is 3.
    -> n_heads        (int)   : Optional. Number of attention heads. Default is 8.
    -> d_ffn_factor   (float) : Optional. FFN hidden dim ratio: dim_ff = int(d_token * d_ffn_factor).
                                Default is 4/3, giving dim_ff ≈ 85 for d_token=64.
    -> dropout        (float) : Optional. Dropout rate applied to attention weights, FFN output, and
                                residual connections. Default is 0.0.
    -> activation     (str)   : Optional. FFN activation ('relu', 'gelu', 'reglu', 'geglu', etc.). Default is 'reglu'.
    -> norm_first     (bool)  : Optional. Pre-norm (True) or post-norm (False) variant. Default is True.
    -> token_bias     (bool)  : Optional. Learnable per-feature bias in the tokenizer. Default is True.
    -> layer_norm_eps (float) : Optional. Epsilon for all LayerNorm layers. Default is 1e-5.
    -> bias           (bool)  : Optional. Bias in Linear layers (attention projections and FFN). Default is True.
    ________________________________________________________________________________________________________________________
    Notes:
    - GLU-type activations ('glu', 'reglu', 'geglu') double the FFN inner projection before gating, so the
      effective post-activation width remains dim_ff = int(d_token * d_ffn_factor).
    - With norm_first=True (pre-norm), an additional LayerNorm is applied to the [CLS] token before the head,
      which is the standard practice for pre-norm transformers.
    - Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021.
      https://arxiv.org/abs/2106.11959
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, in_features : int, out_features : int = 1, d_token : int = 64, n_layers : int = 3, 
                 n_heads        : int   = 8,
                 d_ffn_factor   : float = 4 / 3,
                 dropout        : float = 0.0,
                 activation     : str   = 'reglu',
                 norm_first     : bool  = True,
                 token_bias     : bool  = True,
                 layer_norm_eps : float = 1e-5,
                 bias           : bool  = True):
        super().__init__()

        # Input validation ------------------------------------------------------------------------------------------------#
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if d_token <= 0 or d_token % n_heads != 0:
            raise ValueError(f"d_token must be positive and divisible by n_heads, "
                             f"got d_token={d_token}, n_heads={n_heads}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        # Store configuration ---------------------------------------------------------------------------------------------#
        self.in_features  = in_features
        self.out_features = out_features

        # Tokenizer: (batch, in_features) → (batch, in_features+1, d_token) -----------------------------------------------#
        self.tokenizer = NumericalTokenizer(in_features, d_token, token_bias)

        # Transformer encoder stack ---------------------------------------------------------------------------------------#
        dim_ff = max(1, int(d_token * d_ffn_factor))
        self.layers = nn.ModuleList([
            CustomTEL(d_model = d_token, n_heads = n_heads, dim_ff = dim_ff, dropout = dropout, 
                      activation_layer = activation, 
                      layer_norm_eps   = layer_norm_eps,
                      norm_first       = norm_first, 
                      bias             = bias)
            for _ in range(n_layers)
        ])

        # Final LayerNorm before head (pre-norm style only) -----------------------------------------------------------#
        self.final_norm = nn.LayerNorm(d_token, eps=layer_norm_eps) if norm_first else None

        # Regression head on the [CLS] token ------------------------------------------------------------------------------#
        self.head = nn.Linear(d_token, out_features, bias=True)

        # Initialize weights ----------------------------------------------------------------------------------------------#
        self._initialize_weights()

    # Initialize weights --------------------------------------------------------------------------------------------------#
    def _initialize_weights(self) -> None:
        """Xavier uniform for Linear layers; ones/zeros for LayerNorm. Tokenizer uses kaiming (set at construction)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Forward pass --------------------------------------------------------------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (batch, in_features) — numerical input features.
        Returns:
            out (Tensor): (batch, out_features) — regression predictions.
        """
        # Tokenize → (batch, in_features+1, d_token)
        x = self.tokenizer(x)

        # Transformer encoding
        for layer in self.layers:
            x = layer(x)

        # Extract [CLS] token (position 0) → (batch, d_token)
        x = x[:, 0]

        # Final LayerNorm (pre-norm only)
        if self.final_norm is not None:
            x = self.final_norm(x)

        # Regression head → (batch, out_features)
        return self.head(x)

#--------------------------------------------------------------------------------------------------------------------------#