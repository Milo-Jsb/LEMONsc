# Avoid circular imports by defining shared types and dataclasses in a separate module ------------------------------------#
from __future__ import annotations

# Modules -----------------------------------------------------------------------------------------------------------------#
import torch

# External funtions and utilities -----------------------------------------------------------------------------------------#
from typing      import Dict, Any, Optional, Literal, TYPE_CHECKING
from dataclasses import dataclass, field

# Prevent circular imports while maintaining type hints
if TYPE_CHECKING:
    from optuna.samplers import BaseSampler
    from optuna.study    import Study

# Definitions -------------------------------------------------------------------------------------------------------------#
ModelType  = Literal["elasticnet", "svr", "rf", "lightgbm", "xgboost", "mlp"]
DeviceType = Literal["cpu", "cuda"]

# Configuration dataclass -------------------------------------------------------------------------------------------------#
@dataclass
class SpaceSearchConfig:
    """
    ________________________________________________________________________________________________________________________
    Configuration settings for SpaceSearch optimization
    ________________________________________________________________________________________________________________________
    Parameters:
    - model_type      : Type of model to optimize (e.g., "elasticnet", "svr", "rf", "lightgbm", "xgboost", "mlp")
    - n_jobs          : Number of parallel jobs for optimization (default: 10)
    - n_trials        : Number of optimization trials to run (default: 100)
    - device          : Device to use for computation ("cpu" or "cuda", default: "cuda" if available)
    - verbose         : Whether to print verbose output during optimization (default: True)
    - seed            : Random seed for reproducibility (default: 42)
    - sampler         : Optional custom Optuna sampler (default: None, uses TPESampler)
    - storage         : Optional storage URL for Optuna study persistence (default: None)
    - load_if_exists  : Whether to load existing study if it exists (default: False)
    - huber_delta     : Delta parameter for Huber loss (default: 1.0)
    - max_epochs      : Maximum epochs for deep learning models (default: 100)
    - dl_patience     : Patience for early stopping in deep learning models (default: 10)
    - dl_architecture : Optional dictionary of parameters to create a deep learning model (default: None)
    ________________________________________________________________________________________________________________________
    """
    model_type      : ModelType
    n_jobs          : int                      = 10                
    n_trials        : int                      = 100               
    device          : DeviceType               = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    verbose         : bool                     = True              
    seed            : int                      = 42                
    sampler         : Optional[BaseSampler]    = None              
    storage         : Optional[str]            = None              
    load_if_exists  : bool                     = False             
    huber_delta     : float                    = 1.0               
    max_epochs      : Optional[int]            = 100               
    dl_patience     : Optional[int]            = 10                
    dl_loss_fn      : str                      = 'huber'           
    dl_architecture : Optional[Dict[str, Any]] = None              
    
    def __post_init__(self):
        """Validate configuration parameters"""
        # Normalize model_type to lowercase and strip whitespace
        if isinstance(self.model_type, str):
            self.model_type = self.model_type.lower().strip()
        
        # Validate positive integers
        if self.n_jobs <= 0:
            raise ValueError(f"n_jobs must be positive, got {self.n_jobs}")
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        
        # Validate optional positive integers
        if self.max_epochs is not None and self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.dl_patience is not None and self.dl_patience <= 0:
            raise ValueError(f"dl_patience must be positive, got {self.dl_patience}")
        
        # Validate huber_delta
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {self.huber_delta}")
        
        # Validate device
        if self.device not in ["cpu", "cuda"]:
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{self.device}'")

# Results dataclass -------------------------------------------------------------------------------------------------------#
@dataclass
class SpaceSearchResult:
    """Results from a SpaceSearch optimization study"""
    best_params : Dict[str, Any]                  
    best_score  : float
    study       : Study
    n_trials    : int
    output_dir  : str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format (excludes study object)"""
        return {
            'best_params' : self.best_params, 
            'best_score'  : self.best_score, 
            'n_trials'    : self.n_trials,
            'output_dir'  : self.output_dir
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary of the results"""
        return (
            f"SpaceSearch Results:\n"
            f"  Best Score: {self.best_score:.6f}\n"
            f"  Trials Run: {self.n_trials}\n"
            f"  Best Params: {self.best_params}\n"
            f"  Output Dir: {self.output_dir}"
        )
    
#--------------------------------------------------------------------------------------------------------------------------#