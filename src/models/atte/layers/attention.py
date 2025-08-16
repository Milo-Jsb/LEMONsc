# Modules -----------------------------------------------------------------------------------------------------------------#
import math
import torch

import torch.nn            as nn
import torch.nn.functional as F

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.nn import init

# Multi-Head attention module ---------------------------------------------------------------------------------------------#
class MultiHeadAttention(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Multi-head attention module 
    ________________________________________________________________________________________________________________________
    Parameters:
    ________________________________________________________________________________________________________________________
    Returns:

    ________________________________________________________________________________________________________________________
    Notes:
        -> Original implementation: https://github.com/alercebroker/ATAT/blob/main/layers/mha.py
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, input_qdim:int, input_kdim:int, input_vdim:int, embed_dim:int, num_heads:int, output_dim:int = 16,
                 which_linear = nn.Linear, dropout = 0.0,
                 is_q_proj:bool      = True, 
                 is_k_proj:bool      = True, 
                 is_v_proj:bool      = True, 
                 is_output_proj:bool = True,
                 qk_same_length:bool = False):
        
        super().__init__()
        # Embedding dimension must be divisible by number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads." 
        
        # Set embedding dimension, number of attention heads, and the head dimension
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        
        # Set input dimension for Query, Key, Value and output
        self.input_qdim = input_qdim
        self.input_kdim = input_kdim
        self.input_vdim = input_vdim
        self.output_dim = output_dim
        
        # Flags for linear projection
        self.is_q_proj      = is_q_proj
        self.is_k_proj      = is_k_proj
        self.is_v_proj      = is_v_proj
        self.is_output_proj = is_output_proj
        
        # If all projections are enabled and input dims are equal, project Q, K, V together for efficiency
        self.proj_together  = is_q_proj  and is_k_proj and is_v_proj and \
                              input_qdim == input_kdim == input_vdim and \
                              qk_same_length

        if self.proj_together:
            # Joint projection for Q, K, V
            self.qkv_proj = which_linear(input_qdim, 3*embed_dim)
        
        # Otherwise, use separate projections for Q, K, V
        else:
            if is_q_proj: self.q_proj = which_linear(input_qdim, embed_dim)
            if is_k_proj: self.k_proj = which_linear(input_kdim, embed_dim)
            if is_v_proj: self.v_proj = which_linear(input_vdim, embed_dim)
        
        # Output projection (optional)
        if is_output_proj:
            self.o_proj   = which_linear(embed_dim, self.output_dim)

        # Dropout for attention weights
        self.dropout_layer = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        """ 
        Orthogonal initialization for all linear and embedding layers. Transformer-style, Original Transformer 
        initialization, see PyTorch documentation
        """
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                init.orthogonal_(module.weight)

    def scaled_dot_product(self, q, k, v, mask=None, causal_mask = False):
        # Compute scaled dot-product attention
        d_k = q.size()[-1] 
        
        attn_logits = torch.matmul(q, k.transpose(-2, -1))  # [Batch, Head, Qlen, Klen]
        attn_logits = attn_logits / math.sqrt(d_k)          # Scale by sqrt(d_k)

        # Apply mask if provided (e.g., for padding)
        if mask is not None:
            mask = mask.unsqueeze(1).permute(0, 1, 3, 2)           
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        # Apply causal mask for autoregressive attention if needed
        if causal_mask:
            i, j        = attn_logits.shape[-2:]
            mask_aux    = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
            attn_logits = attn_logits.masked_fill(mask_aux, -9e15)

        # Numerical stability: subtract max before softmax
        attn_logits = attn_logits - attn_logits.amax(dim = -1, keepdim = True).detach()
        attention   = F.softmax(attn_logits, dim=-1)
        attention   = self.dropout_layer(attention)
        values      = torch.matmul(attention, v)
        
        return values, attention

    def forward(self, value, key = None, query = None, mask=None, return_attention=False,
                                                reshape = True, causal_mask = False):
        # value: (batch_size, seq_len, v_dim)
        bs, sl, v_dim = value.size() # batch_size, source length, vdim
        
        # Target length (for output sequence)
        tl = sl if query is None else query.size(1)
        
        if self.proj_together:
            # Joint projection for Q, K, V
            tl  = sl
            qkv = self.qkv_proj(value)
            # Separate Q, K, V from linear output
            qkv = qkv.reshape(bs, sl, self.num_heads, 3*self.head_dim)
            qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Separate projections for Q, K, V (if enabled)
            q  = self.q_proj(query).reshape(bs, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) \
                    if self.is_q_proj else query.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            k  = self.k_proj(key).reshape(bs, sl, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  \
                    if self.is_k_proj else key.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            v  = self.v_proj(value).reshape(bs, sl, self.num_heads, self.head_dim).permute(0, 2, 1, 3) \
                    if self.is_v_proj else value.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Compute attention and weighted values
        values, attention = self.scaled_dot_product(q, k, v, mask = mask, causal_mask = causal_mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]

        # Output projection or reshape
        if self.is_output_proj:
            o  = self.o_proj(values.reshape(bs, tl, self.embed_dim)) 
        else:
            o  = values.reshape(bs, tl, -1) if reshape else values

        # Optionally return attention weights (for interpretability/visualization)
        if return_attention:
            return o, attention.permute(0, 2, 1, 3)
        else:
            return o, None

# Multi-Head Attention handler for easier call ----------------------------------------------------------------------------#
class MultiHeadAttentionHandler(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Multi-head attention handler 
    ________________________________________________________________________________________________________________________
    Parameters:
    ________________________________________________________________________________________________________________________
    Returns:

    ________________________________________________________________________________________________________________________
    Notes:
        -> Original implementation: https://github.com/alercebroker/ATAT/blob/main/layers/mha.py
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, input_dim, head_dim, num_heads, which_linear = nn.Linear, dropout = 0.0, **kwargs):
        super().__init__()
        
        self.input_dim  = input_dim
        self.embed_dim  = head_dim * num_heads
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self.mod_dim    = self.head_dim  

        # Main multi-head attention block (self-attention)
        self.mha_lc = MultiHeadAttention(input_qdim=input_dim, input_kdim=input_dim, input_vdim=input_dim, 
                                         embed_dim    = self.embed_dim, 
                                         num_heads    = num_heads, 
                                         which_linear = which_linear, 
                                         dropout      = dropout,
                                         output_dim   = input_dim,
                                         is_q_proj = True, 
                                         is_k_proj = True,
                                         is_v_proj = True, 
                                         is_output_proj = True)
                

    def get_input_dim(self):
        # Returns the embedding dimension used by the handler
        return self.embed_dim
    def get_last_dim(self):
        # Returns the embedding dimension used by the handler
        return self.embed_dim

    def forward(self, emb_x, mask=None, return_attention=False, causal_mask = False):
        # Pass input through the multi-head attention block (self-attention)
        emb, attention = self.mha_lc(emb_x, key = emb_x, query = emb_x, mask = mask, 
                                     return_attention = return_attention,
                                     causal_mask = causal_mask)
        return emb

#--------------------------------------------------------------------------------------------------------------------------#