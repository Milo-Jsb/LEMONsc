# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import warnings
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Optional
from torch  import nn
from entmax import sparsemax, entmax15

# Torch module for data-aware initialization ------------------------------------------------------------------------------#
class ModuleWithInit(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Wrapper for a Torch Module that performs data-aware initialization. 
    ________________________________________________________________________________________________________________________
    References:
    - Original Implementation:
      https://github.com/yandex-research/rtdl-revisiting-models/tree/main/lib/node
    - Class created by:
      https://github.com/OpenTabular/DeepTab/blob/master/deeptab/arch_utils/data_aware_initialization.py
    ________________________________________________________________________________________________________________________
    """
    # Initialize the class ------------------------------------------------------------------------------------------------#
    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool   = None

    # Abstract function to be inherited by the subclass -------------------------------------------------------------------#
    def initialize(self, *args, **kwargs):
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement the 'initialize' method in the subclass.")

    # Define callable -----------------------------------------------------------------------------------------------------#
    def __call__(self, *args, **kwargs):
        
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool             = True
        
        return super().__call__(*args, **kwargs)

# Oblivious Decision Stump Tree -------------------------------------------------------------------------------------------#
class ODST(ModuleWithInit):
    """
    ________________________________________________________________________________________________________________________
    Oblivious Decision Stump Tree (ODST). Modifications included for stability and efficiency.
    ________________________________________________________________________________________________________________________
    References:
    - Popov et al., "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data", ICLR 2020.
      https://arxiv.org/abs/1909.06312
    - Implementation inspired by:
      https://github.com/OpenTabular/DeepTab/blob/master/deeptab/
    ________________________________________________________________________________________________________________________
    """
    # Initialization ------------------------------------------------------------------------------------------------------#
    def __init__(self, in_features: int, num_trees: int = 128, depth: int = 6, gate: str = "sparsemax", 
                 dynamic_gate : bool          = False, 
                 seed         : Optional[int] = None):
        """
        ____________________________________________________________________________________________________________________
        Initialize an ODST layer.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> in_features  (int)           : Number of input features.
        -> num_trees    (int)           : Number of trees in the ensemble. Default is 128.
        -> depth        (int)           : Depth of each tree. Number of leaves = 2 ** depth. Default is 6.
        -> gate         (str)           : Gate function for soft feature selection: 'softmax', 'sparsemax', 'entmax15'.
                                          Default is 'sparsemax'. 'sparsemax' and 'entmax15' produce sparse weights,
                                          each split to focus on a small number of features (~ hard decision trees).
                                          'softmax' produces dense weights, blending all features smoothly.
        -> dynamic_gate (bool)          : Optional context network for dynamic gating.
        -> seed         (Optional[int]) : Random seed for reproducibility. Default is None.
        ____________________________________________________________________________________________________________________
        Notes:
            -> The number of leaves per tree grows exponentially with depth (2^depth). Be mindful
               of memory usage when increasing depth, especially with many trees.
            -> The gate choice affects the sparsity of feature selection: 'sparsemax' and 'entmax15' encourage more 
               discrete feature choices.        
            -> We follow the same initialization scheme as the original paper for feature selectors and leaf responses, 
               but thresholds and log_temperatures are initialized differently for stability.
            -> Dynamic gating introduces a lightweight context network that produces an additive modulation over the 
               feature_selectors, conditioned on the current input. Zero-initialized so the model starts as a standard 
               static-gate ODST and learns to deviate. Experimental; not present in the original NODE paper.
        ____________________________________________________________________________________________________________________
        """
        ModuleWithInit.__init__(self)

        # Input validation ------------------------------------------------------------------------------------------------#
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_trees <= 0:
            raise ValueError(f"num_trees must be positive, got {num_trees}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if gate not in {"softmax", "sparsemax", "entmax15"}:
            raise ValueError(f"gate must be one of 'softmax', 'sparsemax', 'entmax15', got {gate!r}")

        # Initialize parameters -------------------------------------------------------------------------------------------#
        self.in_features = in_features
        self.num_trees   = num_trees
        self.depth       = depth
        self.gate        = gate
        self.dym_gate    = dynamic_gate
        self._init_seed  = seed
        
        # By definition of a binary tree, the number of leaves is 2 raised to the power of the depth
        num_leaves = 2 ** depth

        # Context network: maps input to additive modulation over feature_selectors (zero-init → static gate at t=0) --#
        if self.dym_gate: self.context_net = nn.Linear(in_features, in_features, bias=False)
        
        # Learnable parameters --------------------------------------------------------------------------------------------#
        
        # Feature selector weights per tree per split level: gate function over in_features selects the 'effective' feature
        self.feature_selectors = nn.Parameter(torch.empty(num_trees, depth, in_features))

        # Split thresholds per tree per split level
        self.thresholds = nn.Parameter(torch.empty(num_trees, depth))

        # Learnable temperature (controls sigmoid sharpness), prevents early saturation and improves gradient flow
        self.log_temperatures = nn.Parameter(torch.zeros(num_trees, depth))
        
        # Leaf response values per tree
        self.leaf_responses = nn.Parameter(torch.empty(num_trees, num_leaves))
        

        # Weight initialization (data-agnostic fallback, overridden by initialize() on first forward pass) ----------------#
        nn.init.uniform_(self.feature_selectors, 0, 1)
        nn.init.uniform_(self.thresholds, -1.0, 1.0)         
        nn.init.normal_(self.leaf_responses, std=1.0)        
        if self.dym_gate: nn.init.zeros_(self.context_net.weight)
    
    # Data-aware initialization of thresholds and log-temperatures -------------------------------------------------------#
    def initialize(self, x: torch.Tensor, eps: float = 1e-6) -> None:
        """
        ____________________________________________________________________________________________________________________
        Data-aware initialization of thresholds and log-temperatures using the first batch of data.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> x   (torch.Tensor) : Input tensor of shape (batch, in_features).
        -> eps (float)        : Small value to avoid log(0) in temperature init. Default is 1e-6.
        ____________________________________________________________________________________________________________________
        Notes:
            -> Thresholds are initialized at random percentiles of the projected feature values, placing each
               split point within the actual data distribution (not a blind uniform range).
            -> Temperatures are set so all first-batch samples fall in the linear region of sigmoid, ensuring
               non-zero gradients at initialization.
            -> A Beta(1,1) distribution is used for percentile sampling, which is equivalent to uniform random
               percentiles over [0, 100] — matching the paper's intent without extra hyperparameters.
            -> Runs under torch.no_grad() to avoid polluting the autograd graph during initialization.
            -> Initialize log-temperatures so all first-batch samples fall in the linear region of sigmoid:
                    temperature = percentile of |x_proj - threshold| at the 95th percentile (i.e. most of the distribution, 
                    avoiding extreme outliers).
               This ensures |logit| <= 1 for all batch samples at init, keeping sigmoid derivatives near their peak
        ____________________________________________________________________________________________________________________
        """
        # Input validation and warning for small batch size (stability risk) ----------------------------------------------#
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2-dimensional (batch, in_features), got shape {tuple(x.shape)}")
        if x.shape[0] < 1000:
            warnings.warn(
                f"Data-aware initialization is performed on only {x.shape[0]} samples (< 1000). "
                "This may cause instability. Use a larger first batch for better initialization.",
                UserWarning, stacklevel=2
            )

        # Start initialization outside the gradient tracking ---------------------------------------------------------------#
        with torch.no_grad():

            # Compute gates using the current (data-agnostic) feature_selectors
            if self.gate == "sparsemax":
                gates = sparsemax(self.feature_selectors, dim=-1)
            elif self.gate == "entmax15":
                gates = entmax15(self.feature_selectors, dim=-1)
            else:
                gates = torch.softmax(self.feature_selectors, dim=-1)

            # Project input features using gates to get per-split feature values 
            gates_flat = gates.reshape(self.num_trees * self.depth, self.in_features)
            x_proj     = torch.matmul(x, gates_flat.T).reshape(x.shape[0], self.num_trees, self.depth)

            # Initialize thresholds at random percentiles of the projected feature distribution Beta(1,1)
            if self._init_seed is not None:
                rng           = np.random.default_rng(self._init_seed)
                percentiles_q = 100.0 * rng.beta(1.0, 1.0, size=(self.num_trees, self.depth))
            
            # If no seed is provided, use numpy's global random state (which may be seeded elsewhere)
            else:
                percentiles_q = 100.0 * np.random.beta(1.0, 1.0, size=(self.num_trees, self.depth))

            # x_proj_np: (num_trees * depth, batch) — each row is one split's projected values across the batch
            x_proj_np = x_proj.cpu().float().numpy().transpose(1, 2, 0).reshape(self.num_trees * self.depth, -1)

            # Get the threshold values by computing the specified percentile for each split's projected values.
            n_splits  = self.num_trees * self.depth
            batch_sz  = x_proj_np.shape[1]
            sorted_proj = np.sort(x_proj_np, axis=-1)                                      
            
            frac_idx = (percentiles_q.flatten() / 100.0 * (batch_sz - 1)).clip(0, batch_sz - 1)
            lo_idx   = frac_idx.astype(np.int32)
            hi_idx   = np.minimum(lo_idx + 1, batch_sz - 1)
            
            frac      = (frac_idx - lo_idx).astype(np.float32)
            split_idx = np.arange(n_splits)
            
            threshold_values = (sorted_proj[split_idx, lo_idx] * (1.0 - frac) +
                                sorted_proj[split_idx, hi_idx] * frac
                                ).astype(np.float32).reshape(self.num_trees, self.depth)

            # Update thresholds with the computed values (in-place, no autograd tracking)
            self.thresholds.data[...] = torch.as_tensor(threshold_values, dtype=x.dtype, device=x.device)

            # Initialize log-temperatures
            abs_diff = (x_proj - self.thresholds).abs().cpu().float().numpy()

            # use max spread: all samples stay in linear region
            temperatures = np.percentile(abs_diff.transpose(1, 2, 0).reshape(self.num_trees * self.depth, -1),
                                         q    = 95.0,   
                                         axis = -1,
                                         ).reshape(self.num_trees, self.depth).astype(np.float32)

            # Update log_temperatures with the computed values (in-place, no autograd tracking)
            self.log_temperatures.data[...] = torch.log(torch.as_tensor(temperatures, dtype=x.dtype, device=x.device) + eps)

    # Forward pass of the ODST layer --------------------------------------------------------------------------------------#
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
        Notes:
            -> The logic goes as follows:
                    1. Soft feature selection: the gate function (softmax / sparsemax / entmax15) produces weights
                       over in_features for each (tree, depth) split, then projects the input to a scalar per split.
                       Two execution paths depending on dynamic_gate:

                       Static  (dynamic_gate=False):
                        |_gates     : (num_trees, depth, in_features)  — shared across all samples
                        |_x_proj    : (batch, num_trees, depth)         — via matmul + reshape

                       Dynamic (dynamic_gate=True):
                        |_modulation: (batch, in_features)              — additive shift from context_net(x)
                        |_gates     : (batch, num_trees, depth, in_features) — per-sample gates
                        |_x_proj    : (batch, num_trees, depth)         — via einsum 'bf,btdf->btd'

                    2. Temperature-scaled split logits: log_temperatures is mapped through exp(4·tanh(x/4)),
                       which keeps temperature in (e^-4, e^4) with smooth non-zero gradients everywhere.
                       Unlike clamp+exp, gradients are never killed at the boundary:
                        |_temperature : (num_trees, depth)
                        |_logits      : (batch, num_trees, depth)
                    3. Soft binary routing: sigmoid gives p_right at each node; 1 - sigmoid gives p_left:
                        |_p_right : (batch, num_trees, depth)
                        |_p_left  : (batch, num_trees, depth)
                    4. Log-space vectorized leaf construction: log-probabilities are computed for each split,
                       then aggregated over all 2^depth leaf paths via two einsums against a fixed bit-mask
                       matrix (leaf_bits). This avoids D serial cat iterations and is immune to float32
                       underflow at depth >= 6 where product accumulation reaches ~1e-9:
                        |_leaf_bits  : (num_leaves, depth)   — precomputed binary path mask, device-cached
                        |_log_pr/pl  : (batch, num_trees, depth)
                        |_leaf_probs : (batch, num_trees, num_leaves)
                    5. Leaf aggregation: element-wise product of leaf_probs and leaf_responses, summed over leaves,
                       yields the per-tree scalar output:
                        |_output : (batch, num_trees)
        ____________________________________________________________________________________________________________________
        """
        # Dynamic gating logic --------------------------------------------------------------------------------------------#
        if self.dym_gate:
            
            modulation  = self.context_net(x)
            dym_weights = self.feature_selectors.unsqueeze(0) + modulation[:, None, None, :]
            
            if self.gate == "sparsemax":
                gates = sparsemax(dym_weights, dim=-1)
            elif self.gate == "entmax15":
                gates = entmax15(dym_weights, dim=-1)
            else:
                gates = torch.softmax(dym_weights, dim=-1)
            
            # gates: (batch, T, D, F) → einsum en lugar de matmul plano
            x_proj = torch.einsum('bf,btdf->btd', x, gates)

        else:
            # Soft feature selection: weighted combination of input features for each split, then project to get split logits
            if self.gate == "sparsemax":
                gates = sparsemax(self.feature_selectors, dim=-1)
            elif self.gate == "entmax15":
                gates = entmax15(self.feature_selectors, dim=-1)
            else:
                gates = torch.softmax(self.feature_selectors, dim=-1)
        
            gates_flat = gates.reshape(self.num_trees * self.depth, self.in_features)
            x_proj     = torch.matmul(x, gates_flat.T)
            x_proj     = x_proj.reshape(x.shape[0], self.num_trees, self.depth) 
        
        # Temperature scaling: exp(4·tanh(x/4)) keeps temperature in (e^-4, e^4) with smooth non-zero gradients.
        # Unlike clamp+exp, gradients are never killed at the boundary — critical during hyperparameter search.
        temperature = torch.exp(4.0 * torch.tanh(self.log_temperatures / 4.0))

        # Compute split logits (scaled by temperature) for sharper splits and better gradient flow
        logits = (x_proj - self.thresholds) * temperature

        # Compute probabilities
        p_right = torch.sigmoid(logits)
        p_left  = 1 - p_right

        # Log-space vectorized leaf construction: avoids D serial iterations and float32 underflow at depth >= 6.
        # leaf_bits: (num_leaves, depth) binary path mask — bit d of leaf index l indicates right (1) or left (0).
        leaf_idx  = torch.arange(2 ** self.depth, device=x.device)
        leaf_bits = ((leaf_idx.unsqueeze(-1) >> torch.arange(self.depth, device=x.device)) & 1).to(x.dtype)

        log_pr = torch.log(p_right.clamp(min=1e-6))  # (batch, num_trees, depth)
        log_pl = torch.log(p_left.clamp(min=1e-6))   # (batch, num_trees, depth)

        # Aggregate log-probabilities over all 2^depth paths via einsum against the bit-mask
        log_leaf_probs = (torch.einsum('btd,ld->btl', log_pr, leaf_bits) +
                          torch.einsum('btd,ld->btl', log_pl, 1.0 - leaf_bits))
        leaf_probs = torch.exp(log_leaf_probs)        # (batch, num_trees, num_leaves)

        # Leaf aggregation: element-wise product of leaf_probs and leaf_responses, summed over leaves
        output = (leaf_probs * self.leaf_responses.unsqueeze(0)).sum(dim=-1)

        return output

    # String representation of the ODST layer -----------------------------------------------------------------------------#
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}\n("
                f"in_features ={self.in_features}, "
                f"num_trees   ={self.num_trees}, "
                f"depth       ={self.depth}, "
                f"gate        ={self.gate}, "
                f"dynamic_gate={self.dym_gate})\n")

# NODE Regressor ----------------------------------------------------------------------------------------------------------#
class NODERegressor(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    Neural Oblivious Decision Ensembles (NODE) Regressor. Modifications included for stability and efficiency.
    ________________________________________________________________________________________________________________________
    Stacks multiple ODST layers with dense skip connections following Popov et al. Each layer receives
    the original normalized input concatenated with all previous layer outputs (growing context), allowing
    each layer to condition on learned representations from prior layers. All layer outputs are concatenated
    and projected through a linear head to produce the final prediction.
    ________________________________________________________________________________________________________________________
    References:
    - Popov et al., "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data", ICLR 2020.
      https://arxiv.org/abs/1909.06312
    - Implementation inspired by:
      https://github.com/OpenTabular/DeepTab/blob/master/deeptab/
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, in_features: int, num_trees: int = 128, depth: int = 6, num_layers: int = 1, out_features: int  = 1,
                 dropout      : Optional[float] = 0.0, 
                 gate         : str             = "sparsemax",
                 dynamic_gate : bool            = False,
                 seed         : Optional[int]   = None):
        """
        ____________________________________________________________________________________________________________________
        Initialize NODE Regressor
        ____________________________________________________________________________________________________________________
        -> in_features  (int)             : Number of input features (dimensionality of input data).
        -> num_trees    (int)             : Number of trees in each ODST layer.
        -> depth        (int)             : Depth of each tree in the ODST layer.
        -> num_layers   (int)             : Number of ODST layers to stack (default 1, i.e. single layer)
        -> out_features (int)             : Output layer of the model (default to one, single point regression)
        -> dropout      (Optional[float]) : Dropout rate applied after each ODST layer (default 0.0)
        -> gate         (str)             : Gate function for soft feature selection in each ODST layer.
                                            One of 'softmax', 'sparsemax', 'entmax15'. Default is 'sparsemax'.
        -> dynamic_gate (bool)            : If True, enables input-conditioned (dynamic) gate weights on the first
                                            ODST layer only, via a shared linear context network (F x F). Layers
                                            k > 0 always use static gates to avoid the O(B·T·D·F_k) memory cost
                                            that grows with the dense-stacked context dimension. Default is False.
        -> seed         (Optional[int])   : Random seed for reproducibility. Default is None.
        ____________________________________________________________________________________________________________________
        Notes:
            -> LayerNorm is applied on the input features to stabilise training, as suggested in the original paper.
            -> Dense stacking: layer k receives [x_norm, out_0, ..., out_{k-1}] as input, so in_features grows by
               num_trees per layer. This allows each layer to condition on prior learned representations.
            -> Context is built via torch.cat (never in-place) to preserve autograd graph integrity.
            -> Only tree outputs (not raw features) feed the output head.
            -> Dropout is applied after each ODST layer to regularize the model and prevent overfitting.
            -> The gate choice is shared across all ODST layers.
            -> Dynamic gating is applied only to the first ODST layer (k=0), where in_features = F_orig is small
               and the gates tensor (batch, T, D, F_orig) is memory-bounded. For k > 0, in_features grows by T
               per layer (dense stacking), making the full per-sample gates tensor prohibitively large. Layer 0
               operates on raw physical features, which is also where regime-aware feature selection is most
               meaningful — subsequent layers already condition on learned tree representations.
        ____________________________________________________________________________________________________________________
        """
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
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if gate not in {"softmax", "sparsemax", "entmax15"}:
            raise ValueError(f"gate must be one of 'softmax', 'sparsemax', 'entmax15', got {gate!r}")
        if depth > 8:
            warnings.warn(
                f"depth={depth} produces {2**depth} leaves per tree. "
                f"Total leaf parameters per layer: {num_trees * 2**depth}. "
                "Consider whether this is intended.",
                UserWarning, stacklevel=2
            )

        # Store configuration ---------------------------------------------------------------------------------------------#
        self.in_features  = in_features
        self.num_trees    = num_trees
        self.depth        = depth
        self.num_layers   = num_layers
        self.out_features = out_features
        self.gate         = gate
        self.dym_gate     = dynamic_gate

        # Input normalization layer ---------------------------------------------------------------------------------------#
        self.input_norm = nn.LayerNorm(in_features)
        
        # Dense-stacked ODST layers: layer k receives in_features + k * num_trees inputs ----------------------------------#
        # Dynamic gating is restricted to k=0: F_0 = in_features (small), F_k grows by num_trees per layer,
        # making the per-sample gates tensor (batch, T, D, F_k) prohibitively large for k > 0.
        self.layers = nn.ModuleList([
            ODST(in_features=in_features + k * num_trees, num_trees=num_trees, depth=depth, gate=gate, 
                 dynamic_gate = (self.dym_gate and k == 0),
                 seed         = seed + k if seed is not None else None)
            for k in range(num_layers)])

        # Per-layer normalization applied to each ODST output before extending context ------------------------------------#
        self.ctx_norms = nn.ModuleList([nn.LayerNorm(num_trees) for _ in range(num_layers)])

        # Optional dropout after each layer -------------------------------------------------------------------------------#
        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0.0) else None

        # Output head: project concatenated tree outputs to prediction target ---------------------------------------------#
        self.output_head = nn.Linear(num_trees * num_layers, out_features)

        # Weight initialization -------------------------------------------------------------------------------------------#
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    # Forward pass of the NODE Regressor ----------------------------------------------------------------------------------#
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
        Notes:
            -> Context is built iteratively using torch.cat (no in-place writes) to keep the autograd graph intact.
               In-place writes on a tensor that has already entered the computation graph cause version counter
               mismatches and break gradient computation.
            -> ctx starts as x_norm and grows by one normalized tree-output block per layer:
                    ctx_0 = x_norm
                    ctx_1 = cat([x_norm, LayerNorm(out_0)])
                    ctx_k = cat([ctx_{k-1}, LayerNorm(out_{k-1})])
               Each block is normalized independently before concatenation (ctx_norms) to keep all partitions
               at unit scale and prevent later ODST layers from being dominated by unbounded leaf responses.
            -> The output head and the context receive different versions of each layer's output:
                    head  ← out            (raw, with dropout if enabled)
                    ctx   ← LayerNorm(out) (normalized, no dropout — preserves information flow)
            -> outputs accumulates the raw per-layer outputs; torch.cat(outputs, dim=-1) feeds the output head.
            -> The output head receives only tree outputs, not raw input features.
        ____________________________________________________________________________________________________________________
        """

        # Normalize input
        x_norm = self.input_norm(x)

        # ctx grows each iteration: [x_norm] → [x_norm, out_0] → [x_norm, out_0, out_1] → ...
        ctx     = x_norm
        outputs = []

        for i, layer in enumerate(self.layers):

            # Layer k receives [x_norm, out_0, ..., out_{k-1}]
            out = layer(ctx)

            # Normalize out previous to extending the context
            out_normed = self.ctx_norms[i](out)

            # Apply dropout only to the tensor going into the output head, NOT to the context.
            out_for_head = self.dropout(out) if self.dropout is not None else out

            # Append results to outputs list for final concatenation (no in-place modification)
            outputs.append(out_for_head)

            # Extend context with the CLEAN output (no dropout) to preserve information across layers
            ctx = torch.cat([ctx, out_normed], dim=-1)

        # Output head receives concatenation of all tree outputs (no raw features)
        return self.output_head(torch.cat(outputs, dim=-1))

    # Property to check if all layers are initialized ---------------------------------------------------------------------#    
    @property
    def is_initialized(self) -> bool:
        return all(bool(layer._is_initialized_tensor.item()) for layer in self.layers)
    
    # String representation of the NODE Regressor -------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """String representation of the Regressor."""
        return (f"{self.__class__.__name__}\n("
                f"in_features  ={self.in_features},\n"
                f"num_trees    ={self.num_trees},\n"
                f"depth        ={self.depth},\n"
                f"num_layers   ={self.num_layers},\n"
                f"out_features ={self.out_features},\n"
                f"gate         ={self.gate},\n"
                f"dynamic_gate ={self.dym_gate})\n")

#--------------------------------------------------------------------------------------------------------------------------#
