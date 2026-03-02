# Modules -----------------------------------------------------------------------------------------------------------------#
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Optional
from torch  import nn

# Oblivious Decision Stump Tree -------------------------------------------------------------------------------------------#
class ODST(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Oblivious Decision Stump Tree (ODST).
    ________________________________________________________________________________________________________________________
    A single differentiable oblivious decision tree used as the building block of the NODE model.
    Each depth level applies the same learnable split (feature selector + threshold) to all nodes at
    that level, which is the 'oblivious' property that enables efficient vectorised computation.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> in_features (int) : Number of input features.
    -> num_trees   (int) : Number of independent trees in this layer.
    -> depth       (int) : Depth of each tree (number of split levels). Number of leaves = 2 ** depth.
    ________________________________________________________________________________________________________________________
    References:
    - Popov et al., "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data", ICLR 2020.
      https://arxiv.org/abs/1909.06312
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, in_features: int, num_trees: int = 128, depth: int = 6):
        super().__init__()

        # Input validation ------------------------------------------------------------------------------------------------#
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_trees <= 0:
            raise ValueError(f"num_trees must be positive, got {num_trees}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.in_features = in_features
        self.num_trees   = num_trees
        self.depth       = depth
        num_leaves       = 2 ** depth

        # Learnable parameters --------------------------------------------------------------------------------------------#
        # Feature selector weights per tree per split level: softmax over in_features selects the 'effective' feature
        self.feature_selectors = nn.Parameter(torch.empty(num_trees, depth, in_features))

        # Split thresholds per tree per split level
        self.thresholds = nn.Parameter(torch.empty(num_trees, depth))

        # Leaf response values per tree
        self.leaf_responses = nn.Parameter(torch.empty(num_trees, num_leaves))

        # Pre-compute binary leaf codes (static buffer) -------------------------------------------------------------------#
        # bit_masks[l, d] == 1 if depth-level d should turn right to reach leaf l, else 0
        leaf_indices = torch.arange(num_leaves)                                       # (num_leaves,)
        bit_masks    = ((leaf_indices.unsqueeze(1) >> torch.arange(depth)) & 1).float()  # (num_leaves, depth)
        self.register_buffer('bit_masks', bit_masks)

        # Weight initialisation -------------------------------------------------------------------------------------------#
        nn.init.normal_(self.feature_selectors, std=0.1)
        nn.init.normal_(self.thresholds,        std=0.1)
        nn.init.normal_(self.leaf_responses,    std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Forward pass through one ODST layer.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> x (torch.Tensor) : Input tensor of shape (batch, in_features).
        ____________________________________________________________________________________________________________________
        Returns:
        -> output (torch.Tensor) : Per-tree regression outputs of shape (batch, num_trees).
        ____________________________________________________________________________________________________________________
        """
        # Soft feature selection: weighted combination of input features for each split
        gates = torch.softmax(self.feature_selectors, dim=-1)      # (num_trees, depth, in_features)
        x_proj = torch.einsum('bi,tdi->btd', x, gates)             # (batch, num_trees, depth)

        # Soft binary split decisions via sigmoid
        h = torch.sigmoid(x_proj - self.thresholds)                # (batch, num_trees, depth)

        # Leaf probability: for leaf l, p_l = prod_d (h_d if bit_d(l)==1 else 1-h_d)
        h_exp  = h.unsqueeze(3)                                    # (batch, num_trees, depth, 1)
        bm_exp = self.bit_masks.T.unsqueeze(0).unsqueeze(0)        # (1, 1, depth, num_leaves)
        h_for_leaf  = h_exp * bm_exp + (1.0 - h_exp) * (1.0 - bm_exp)  # (batch, num_trees, depth, num_leaves)
        leaf_probs  = h_for_leaf.prod(dim=2)                       # (batch, num_trees, num_leaves)

        # Weighted sum of leaf responses
        output = (leaf_probs * self.leaf_responses.unsqueeze(0)).sum(dim=-1)  # (batch, num_trees)
        return output

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}\n("
                f"in_features={self.in_features}, "
                f"num_trees  ={self.num_trees}, "
                f"depth      ={self.depth})\n")


# NODE Regressor ----------------------------------------------------------------------------------------------------------#
class NODERegressor(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Neural Oblivious Decision Ensembles (NODE) Regressor.
    ________________________________________________________________________________________________________________________
    Stacks multiple ODST layers, each operating on the original input features (dense skip-connection
    style).  All layer outputs are concatenated and projected to the prediction head.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> in_features  (int)           : Mandatory. Number of input features.
    -> num_trees    (int)           : Optional. Number of trees per NODE layer. Default is 128.
    -> depth        (int)           : Optional. Depth of each tree. Number of leaves = 2 ** depth. Default is 6.
    -> num_layers   (int)           : Optional. Number of stacked ODST layers. Default is 1.
    -> out_features (int)           : Optional. Number of output features (1 for direct regression). Default is 1.
    -> dropout      (Optional[float]): Optional. Dropout applied to each layer's output. Default is 0.0.
    ________________________________________________________________________________________________________________________
    Notes:
    - Input is normalised with BatchNorm1d before being passed to the ODST stack.
    - All ODST layers receive the original (normalised) input, mirroring the dense connection design
      proposed by Popov et al.
    - Implementation in PyTorch.
    ________________________________________________________________________________________________________________________
    References:
    - Popov et al., "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data", ICLR 2020.
      https://arxiv.org/abs/1909.06312
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, in_features  : int,
                 num_trees    : int            = 128,
                 depth        : int            = 6,
                 num_layers   : int            = 1,
                 out_features : int            = 1,
                 dropout      : Optional[float] = 0.0):
        super().__init__()

        # Input validation ------------------------------------------------------------------------------------------------#
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_trees <= 0:
            raise ValueError(f"num_trees must be positive, got {num_trees}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if dropout is not None and not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Store configuration ---------------------------------------------------------------------------------------------#
        self.in_features  = in_features
        self.num_trees    = num_trees
        self.depth        = depth
        self.num_layers   = num_layers
        self.out_features = out_features

        # Input normalisation layer ---------------------------------------------------------------------------------------#
        self.input_bn = nn.BatchNorm1d(in_features)

        # Stack of ODST layers (each sees the normalised input) -----------------------------------------------------------#
        self.layers = nn.ModuleList([
            ODST(in_features=in_features, num_trees=num_trees, depth=depth)
            for _ in range(num_layers)
        ])

        # Optional dropout after each layer -------------------------------------------------------------------------------#
        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0.0) else None

        # Output head: project concatenated tree outputs to prediction target ---------------------------------------------#
        # Each ODST layer produces (batch, num_trees); after num_layers we have (batch, num_trees * num_layers)
        self.output_head = nn.Linear(num_trees * num_layers, out_features)

        # Weight initialisation -------------------------------------------------------------------------------------------#
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ____________________________________________________________________________________________________________________
        Forward pass through the NODE model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> x (torch.Tensor) : Input tensor of shape (batch, in_features).
        ____________________________________________________________________________________________________________________
        Returns:
        -> output (torch.Tensor) : Predicted values of shape (batch, out_features).
        ____________________________________________________________________________________________________________________
        """
        x_norm = self.input_bn(x)

        layer_outputs = []
        for layer in self.layers:
            out = layer(x_norm)                          # (batch, num_trees)
            if self.dropout is not None:
                out = self.dropout(out)
            layer_outputs.append(out)

        combined = torch.cat(layer_outputs, dim=-1)      # (batch, num_trees * num_layers)
        return self.output_head(combined)                # (batch, out_features)

    def __repr__(self) -> str:
        """String representation of the Regressor."""
        return (f"{self.__class__.__name__}\n("
                f"in_features  ={self.in_features},\n"
                f"num_trees    ={self.num_trees},\n"
                f"depth        ={self.depth},\n"
                f"num_layers   ={self.num_layers},\n"
                f"out_features ={self.out_features})\n")

#--------------------------------------------------------------------------------------------------------------------------#
