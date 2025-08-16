# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Any, Dict, List

# Quantile Transformation module implementation in pytorch ----------------------------------------------------------------#
class QuantileNormalizer(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Efficient quantile-based normalizer for tabular features.
    ________________________________________________________________________________________________________________________
    Parameters:
        -> X_train     (array-like, shape (n_samples, n_features)) : Training data used to compute quantiles.
        -> n_quantiles (int, default=1000)                         : Number of quantiles to use for the empirical CDF.
    ________________________________________________________________________________________________________________________
    Returns:
        -> QuantileNormalizer instance : Quantile normalizer module
    ________________________________________________________________________________________________________________________
    Example:
        >>> normalizer = QuantileNormalizer(X_train)
        >>> x_norm     = normalizer(x)  
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, X_train: torch.Tensor, n_quantiles: int = 1000) -> None:
        super().__init__()
        
        assert X_train.ndim == 2, "X_train must be 2D (n_samples, n_features)"
        
        # Number of features
        self.n_features = X_train.shape[1]

        # Quantiles uniformly spaced in [0, 1]
        quantiles = torch.linspace(0, 1, n_quantiles, dtype=X_train.dtype, device=X_train.device)
        q_values  = torch.quantile(X_train, quantiles, dim=0)

        # Register buffers
        self.register_buffer("quantiles", quantiles)  
        self.register_buffer("q_values", q_values)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Normalize input features to [0, 1] using empirical quantiles.
        ____________________________________________________________________________________________________________________
        Parameters:
            -> x (torch.Tensor, shape (batch, n_features)) : Input features.
        ____________________________________________________________________________________________________________________
        Returns:
            -> x_norm (torch.Tensor, shape (batch, n_features)) : Normalized features in [0, 1] (in [0, 1])
        ____________________________________________________________________________________________________________________
        """
        assert x.ndim == 2 and x.shape[1] == self.n_features, f"Input must be (batch, {self.n_features})"
        
        # Move to the same device and dtype as the quantiles
        x = x.to(self.q_values.device, dtype=self.q_values.dtype)

        # Vectorized search for quantile interval (1, batch, n_features)
        x_exp  = x.unsqueeze(0)  
        q_vals = self.q_values.unsqueeze(1)  
        mask   = (x_exp >= q_vals).float()
        rank   = mask.sum(dim=0) - 1  
        rank   = rank.clamp(0, self.q_values.shape[0] - 2).long()

        # Gather lower and upper quantile values (batch, n_features)
        batch_idx = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        feat_idx  = torch.arange(x.shape[1], device=x.device)
        
        low    = self.q_values[rank, feat_idx]
        high   = self.q_values[rank + 1, feat_idx]
        q_low  = self.quantiles[rank]
        q_high = self.quantiles[rank + 1]

        # Linear interpolation
        t      = (x - low) / (high - low + 1e-9)
        x_norm = q_low + t * (q_high - q_low)
        return x_norm 

# Quantile Feature Tokenizer implementation -------------------------------------------------------------------------------#
class QuantileFeatureTokenizer(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Quantile Feature Tokenizer (QFT) for tabular data.
    ________________________________________________________________________________________________________________________
    Parameters:
        -> X_train      (array-like, shape (n_samples, n_features)) : Training data for quantile normalization.
        -> embedding_dim (int, default=32)                          : Size of the embedding vector for each feature.
        -> n_quantiles   (int, default=1000)                        : Number of quantiles for normalization.
    ________________________________________________________________________________________________________________________
    Returns:
        -> QuantileFeatureTokenizer instance : Quantile feature tokenizer module
    ________________________________________________________________________________________________________________________
    Notes:
        -> This tokenizer follows the descriptions found on Cabrera-Vives, et al. 2024.
    ________________________________________________________________________________________________________________________
    Example:
        >>> tokenizer = QuantileFeatureTokenizer(X_train, embedding_dim=16)
        >>> tokens    = tokenizer(x)  
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, X_train: torch.Tensor, embedding_dim: int = 32, n_quantiles: int = 1000) -> None:
        super().__init__()
        
        # Number of features
        self.n_features = X_train.shape[1]

        # Size of the embedding vector for each feature
        self.embedding_dim = embedding_dim

        # Quantile normalizer
        self.normalizer = QuantileNormalizer(X_train, n_quantiles=n_quantiles)
        self.W          = nn.Parameter(torch.randn(self.n_features, embedding_dim))
        self.b          = nn.Parameter(torch.zeros(self.n_features, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Tokenize input features.
        ____________________________________________________________________________________________________________________
        Parameters:
            -> x (torch.Tensor, shape (batch, n_features)) : Input features.
        ____________________________________________________________________________________________________________________
        Returns:
            -> tokens (torch.Tensor, shape (batch, n_features, embedding_dim)) : Tokenized feature embeddings.
        ____________________________________________________________________________________________________________________
        """
        assert x.ndim == 2 and x.shape[1] == self.n_features, f"Input must be (batch, {self.n_features})"
        
        # Move to the same device and dtype as the weights
        x = x.to(self.W.device, dtype=self.W.dtype)

        # Normalize the features
        x_q = self.normalizer(x)  # (batch, n_features) in [0, 1]

        # Tokenize the features (batch, n_features, embedding_dim)
        tokens = x_q.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        
        return tokens  

# ----------------------------------------------------------------------------------------------------------------------#
class QuantileCategoricalEmbedding(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Quantile-based categorical embedding module for tabular data.
    ________________________________________________________________________________________________________________________
    Parameters:
        -> categories    (Dict[str, List[str]]) : Dictionary mapping variable names to possible category values.
        -> embedding_dim (int, default=8)      : Dimension of the learnable embedding for each category.
        -> quantiles     (List[float])         : List of quantiles to compute for each category (default: [0.25, 0.5, 0.75]).
    ________________________________________________________________________________________________________________________
    Returns:
        -> QuantileCategoricalEmbedding instance : Module for categorical embedding with quantile statistics.
    ________________________________________________________________________________________________________________________
    Example:
        >>> categories = {'color': ['red', 'green', 'blue'], 'type': ['A', 'B']}
        >>> module = QuantileCategoricalEmbedding(categories, embedding_dim=8)
        >>> module.fit({'color': ['red', 'blue'], 'type': ['A', 'B']}, y=[1.0, 2.0])
        >>> X = {'color': torch.tensor([0, 2]), 'type': torch.tensor([1, 0])}
        >>> out = module(X)
    ________________________________________________________________________________________________________________________
    Notes:
        -> Call fit() with your data before using forward().
        -> Output is the concatenation of learned embedding and quantile statistics for each variable.
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, categories: Dict[str, List[str]], embedding_dim: int = 8, quantiles: List[float] = [0.25, 0.5, 0.75]
                ) -> None:
        super().__init__()
        
        # Quantiles and embedding 
        self.quantiles     = quantiles
        self.embedding_dim = embedding_dim
        
        # Dicts to store relevant info
        self.cat2idx: Dict[str, Dict[str, int]] = {}
        self.embeddings                         = nn.ModuleDict()
        
        # Build mapping and embedding for each categorical variable
        for var, cats in categories.items():
            self.cat2idx[var]    = {cat: i for i, cat in enumerate(cats)}
            self.embeddings[var] = nn.Embedding(len(cats), embedding_dim)
        
        # Quantile statistics will be stored as a dict of tensors after fit()
        self.quantile_stats: Dict[str, torch.Tensor] = {}

    def fit(self, X: Dict[str, List[Any]], y: Any) -> None:
        """
        ____________________________________________________________________________________________________________________
        Compute quantile statistics for each category in each variable using torch only.
        ____________________________________________________________________________________________________________________
        Parameters:
            -> X (Dict[str, List[Any] or torch.Tensor]) : Dictionary of variable to list/tensor of category values (length = n_samples).
            -> y (list or torch.Tensor)                : Target values (length = n_samples).
        ____________________________________________________________________________________________________________________
        Returns:
            None. Updates internal quantile_stats buffer.
        ____________________________________________________________________________________________________________________
        """
        # Convert y to torch tensor if is not already
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = y.float()
        # Bring tensor to device
        device = y.device

        # iterate around categories
        for var, cats in self.cat2idx.items():
            
            # Convert X[var] to tensor of indices
            x_var = X[var]
            if not isinstance(x_var, torch.Tensor):
                x_var = torch.tensor([self.cat2idx[var][cat] for cat in x_var], dtype=torch.long, device=device)
            else:
                x_var = x_var.to(device)
            
            # Compute number of categories and quantiles, prealocate the matrix
            n_cats = len(cats)
            n_q    = len(self.quantiles)
            mat    = torch.zeros(n_cats, n_q, dtype=torch.float32, device=device)

            for idx in range(n_cats):
                mask = (x_var == idx)
                if mask.any():
                    # Compute quantiles using torch.quantile, asociate idx to quantile
                    vals     = y[mask]
                    qs       = torch.quantile(vals, torch.tensor(self.quantiles, dtype=vals.dtype, 
                                              device=vals.device)).to(torch.float32)
                    mat[idx] = qs
                else:
                    mat[idx] = torch.zeros(n_q, dtype=torch.float32, device=device)
            
            # Retrieve the quantile_stats
            self.quantile_stats[var] = mat

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Forward pass: concatenate learned embedding and quantile statistics for each variable.
        ____________________________________________________________________________________________________________________
        Parameters:
            -> X (Dict[str, torch.Tensor]) : Dictionary {var_name: tensor of category indices, shape (batch_size,)}
        ____________________________________________________________________________________________________________________
        Returns:
            -> out (torch.Tensor, shape (batch_size, total_embedding_dim)) : Concatenated embeddings and quantile stats.
        ____________________________________________________________________________________________________________________
        """
        outputs = []
        for var, idxs in X.items():
            # Get embedding for each category index and quantile statistics for each category index
            emb       = self.embeddings[var](idxs) 
            quant_mat = self.quantile_stats[var].to(idxs.device)
            quant     = quant_mat[idxs]  
            outputs.append(torch.cat([emb, quant], dim=-1))
        
        # Concatenate all variables along the feature dimension (batch_size, embbeding_dim)
        tokens = torch.cat(outputs, dim=-1)
        
        return tokens

# ----------------------------------------------------------------------------------------------------------------------#
