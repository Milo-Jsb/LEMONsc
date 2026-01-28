# Modules -----------------------------------------------------------------------------------------------------------------#
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Dict, Optional, Tuple, Union, Any
from torch.nn                 import Module
from torch.optim              import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Optimizer Registry ------------------------------------------------------------------------------------------------------#
OPTIMIZER_REGISTRY = {
    'adam' : torch.optim.Adam,
    'sgd'  : torch.optim.SGD
    }

# Scheduler Registry ------------------------------------------------------------------------------------------------------#
SCHEDULER_REGISTRY = {
    'step'   : torch.optim.lr_scheduler.StepLR,
    'cosine' : torch.optim.lr_scheduler.CosineAnnealingLR,
    }

# Optimizer Selection -----------------------------------------------------------------------------------------------------#
def select_optimizer(name: str,model: Module, optimizer_params: Optional[Dict[str, Any]] = None,
                     scheduler_name  : Optional[str]            = None,
                     scheduler_params: Optional[Dict[str, Any]] = None,
                     ) -> Union[Optimizer, Tuple[Optimizer, _LRScheduler]]:
    """
    _______________________________________________________________________________________________________________________
    Select and initialize an optimizer and optional learning rate scheduler. (Call for pytorch elements)
    _______________________________________________________________________________________________________________________
    Parameters:
    -> name             str                      : Name of the optimizer (e.g., 'adam', 'sgd')
    -> model            Module                   : PyTorch model whose parameters will be optimized.
    -> optimizer_params Optional[Dict[str, Any]] : Dictionary of optimizer-specific parameters.
    -> scheduler_name   Optional[str]            : Name of the scheduler (e.g., 'step', 'cosine').
    -> scheduler_params Optional[Dict[str, Any]] : Dictionary of scheduler-specific parameters.
    _______________________________________________________________________________________________________________________
    Returns:
    -> optimizer              [Optimizer]                      : If sheduler_name and scheduler_params are None 
    -> (optimizer, scheduler) [Tuple[Optimizer, _LRScheduler]] : If scheduler_name and scheduler_params are provided
    _______________________________________________________________________________________________________________________    
    Raises:
        ValueError: If optimizer or scheduler name is not supported
        TypeError  : If required parameters are missing
    _______________________________________________________________________________________________________________________   
    """
    # Validate inputs -----------------------------------------------------------------------------------------------------# 
    optimizer_lower = name.lower()
    if optimizer_lower not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{name}' is not supported. Choose from {list(OPTIMIZER_REGISTRY.keys())}.")
    
    
    # Create optimizer ----------------------------------------------------------------------------------------------------#
    try:
        optimizer = OPTIMIZER_REGISTRY[optimizer_lower](model.parameters(), **optimizer_params)
    
    except TypeError as e:
        raise TypeError(f"Error creating optimizer '{name}': {e}. "f"Check that all parameters are provided.") from e
    
    # Create scheduler if requested ---------------------------------------------------------------------------------------#
    if (scheduler_name is not None) and (scheduler_params is not None):
        scheduler = create_scheduler(optimizer, scheduler_name, scheduler_params)
        return optimizer, scheduler
    
    return optimizer

# Create a scheduler given optimizer and parameters -----------------------------------------------------------------------#
def create_scheduler(optimizer: Optimizer, scheduler_name: str, scheduler_params: Dict[str, Any]) -> _LRScheduler:
    """
    ________________________________________________________________________________________________________________________
    Create a learning rate scheduler.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> optimizer        (torch.Optimizer) : Optimizer instance
    -> scheduler_name   (str)             : Name of the scheduler
    -> scheduler_params (Dict[str, Any])  : Scheduler configuration with 'name' key and scheduler-specific params
    ________________________________________________________________________________________________________________________
    Returns:
    -> _LRScheduler : Learning rate scheduler instance
    ________________________________________________________________________________________________________________________
    Raises:
        ValueError: If scheduler name is missing or not supported
        TypeError: If required scheduler parameters are missing
    ________________________________________________________________________________________________________________________
    """
    # Validate scheduler name ---------------------------------------------------------------------------------------------#
    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{scheduler_name}' is not supported. Choose from {list(SCHEDULER_REGISTRY.keys())}.")

    # Create scheduler ----------------------------------------------------------------------------------------------------#
    try:
        scheduler = SCHEDULER_REGISTRY[scheduler_name](optimizer, **scheduler_params)
    except TypeError as e:
        raise TypeError(f"Error creating scheduler '{scheduler_name}': {e}. Check that all parameters are provided.") from e
    
    return scheduler

#--------------------------------------------------------------------------------------------------------------------------#