# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import copy
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib import Path
from typing  import Optional, Union, Dict, Any
from loguru  import logger

# Custom imports ----------------------------------------------------------------------------------------------------------#
from src.models.dltab.utils.handling import _to_cpu_state_dict

# CheckpointManager Class -------------------------------------------------------------------------------------------------#
class CheckpointManager:
    """
    ________________________________________________________________________________________________________________________
    CheckpointManager: Handles saving and loading of model checkpoints
    ________________________________________________________________________________________________________________________
    Responsibilities:
    -> Save complete model state (model, optimizer, scheduler, metadata)
    -> Load model state from checkpoints
    -> Manage checkpoint files (directories, validation)
    -> Handle device transitions (CPU/GPU) during save/load
    ________________________________________________________________________________________________________________________
    """
    
    def __init__(self, verbose: bool = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the CheckpointManager.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> verbose (bool) : Whether to log checkpoint operations
        ____________________________________________________________________________________________________________________
        """
        self.verbose = verbose
    
    def save(self, path: str, model: torch.nn.Module, model_type: str, model_params: dict,
             optimizer_name: str, optimizer_params: dict, is_fitted: bool,
             in_features: int = 5,
             feature_names: Optional[list] = None, history: Optional[dict] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scheduler: Optional[Any] = None, scheduler_name: Optional[str] = None,
             scheduler_params: Optional[dict] = None) -> None:
        """
        ____________________________________________________________________________________________________________________
        Save model and training state to disk.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path             (str)       : Path where to save the checkpoint
        -> model            (nn.Module) : PyTorch model to save
        -> model_type       (str)       : Type of model (e.g., 'mlp')
        -> model_params     (dict)      : Model hyperparameters
        -> optimizer_name   (str)       : Name of optimizer used
        -> optimizer_params (dict)      : Optimizer hyperparameters
        -> is_fitted        (bool)      : Whether model is fitted
        -> in_features      (int)       : Number of input features for the model
        -> feature_names    (list)      : Optional. Feature names used
        -> history          (dict)      : Optional. Training history
        -> optimizer        (Optimizer) : Optional. Optimizer instance
        -> scheduler        (Scheduler) : Optional. LR scheduler instance
        -> scheduler_name   (str)       : Optional. Name of scheduler
        -> scheduler_params (dict)      : Optional. Scheduler hyperparameters
        ____________________________________________________________________________________________________________________
        Raises:
        -> TypeError  : If path is not a string
        -> ValueError : If save operation fails
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not is_fitted and self.verbose:
            logger.warning("Saving model although `is_fitted` is False")
        
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare save data
            save_data = self._prepare_save_data(model=model, model_type=model_type, model_params=model_params, 
                                                optimizer_name   = optimizer_name,
                                                optimizer_params = optimizer_params,
                                                is_fitted        = is_fitted,
                                                in_features      = in_features,
                                                feature_names    = feature_names,
                                                history          = history,
                                                optimizer        = optimizer,
                                                scheduler        = scheduler,
                                                scheduler_name   = scheduler_name,
                                                scheduler_params = scheduler_params)
            
            # Save using torch
            torch.save(save_data, path)
            
            if self.verbose:
                logger.success(f"Model saved successfully to: {path}")
        
        except (OSError, IOError, TypeError, RuntimeError) as e:
            raise ValueError(f"Error saving model: {e}") from e
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        ____________________________________________________________________________________________________________________
        Load checkpoint data from disk.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path (str) : Path to the saved checkpoint file
        ____________________________________________________________________________________________________________________
        Returns:
        -> checkpoint (dict) : Dictionary containing all saved data
        ____________________________________________________________________________________________________________________
        Raises:
        -> TypeError        : If path is not a string
        -> FileNotFoundError : If checkpoint file doesn't exist
        -> ValueError       : If load operation fails
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load data from checkpoint (mapped to CPU first)
            checkpoint = torch.load(path, map_location='cpu')
            
            if self.verbose:
                logger.success(f"Checkpoint loaded successfully from: {path}")
            
            return checkpoint
        
        except (OSError, IOError, KeyError, EOFError, RuntimeError) as e:
            raise ValueError(f"Error loading checkpoint: {e}") from e
    
    def _prepare_save_data(self, model: torch.nn.Module, model_type: str, model_params: dict,
                          optimizer_name: str, optimizer_params: dict, is_fitted: bool,
                          in_features: int, feature_names: Optional[list], history: Optional[dict],
                          optimizer: Optional[torch.optim.Optimizer],
                          scheduler: Optional[Any], scheduler_name: Optional[str],
                          scheduler_params: Optional[dict]) -> Dict[str, Any]:
        """
        ____________________________________________________________________________________________________________________
        Prepare complete save data dictionary.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model, optimizer, scheduler : Model and training components
        -> Other parameters            : Metadata and hyperparameters
        ____________________________________________________________________________________________________________________
        Returns:
        -> save_data (dict) : Complete data dictionary ready for saving
        ____________________________________________________________________________________________________________________
        """
        # Base save data
        save_data = {
            "model_type"       : model_type,
            "model_params"     : model_params.copy(),
            "model_state_dict" : _to_cpu_state_dict(model.state_dict()),
            "optimizer_name"   : optimizer_name,
            "optimizer_params" : optimizer_params.copy(),
            "is_fitted"        : is_fitted,
            "in_features"      : in_features,
            "feature_names"    : feature_names,
            "device"           : 'cpu',  # Always save as CPU for portability
            "history"          : history.copy() if history else {},
        }
        
        # Save optimizer state if available
        if optimizer is not None:
            save_data["optimizer_state_dict"] = self._cpu_optimizer_state(optimizer)
        
        # Save scheduler state and params if available
        if scheduler is not None:
            save_data["scheduler_name"]       = scheduler_name
            save_data["scheduler_params"]     = scheduler_params
            save_data["scheduler_state_dict"] = self._cpu_scheduler_state(scheduler)
        
        return save_data
    
    def _cpu_optimizer_state(self, optimizer: torch.optim.Optimizer) -> dict:
        """
        ____________________________________________________________________________________________________________________
        Move optimizer state to CPU for saving.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> optimizer (Optimizer) : Optimizer instance
        ____________________________________________________________________________________________________________________
        Returns:
        -> opt_state_dict (dict) : Optimizer state with all tensors on CPU
        ____________________________________________________________________________________________________________________
        """
        # Deep copy to avoid modifying original optimizer
        opt_sd = copy.deepcopy(optimizer.state_dict())
        
        if "state" in opt_sd and isinstance(opt_sd["state"], dict):
            for st_key, st_val in opt_sd["state"].items():
                for inner_k, inner_v in list(st_val.items()):
                    if isinstance(inner_v, torch.Tensor):
                        st_val[inner_k] = inner_v.cpu()
        
        return opt_sd
    
    def _cpu_scheduler_state(self, scheduler: Any) -> dict:
        """
        ____________________________________________________________________________________________________________________
        Move scheduler state to CPU for saving.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> scheduler (Scheduler) : Learning rate scheduler instance
        ____________________________________________________________________________________________________________________
        Returns:
        -> sched_state_dict (dict) : Scheduler state with all tensors on CPU
        ____________________________________________________________________________________________________________________
        """
        # Deep copy to avoid modifying original scheduler
        sched_sd = copy.deepcopy(scheduler.state_dict())
        
        if isinstance(sched_sd, dict):
            for k, v in list(sched_sd.items()):
                if isinstance(v, torch.Tensor):
                    sched_sd[k] = v.cpu()
        
        return sched_sd

#--------------------------------------------------------------------------------------------------------------------------#
