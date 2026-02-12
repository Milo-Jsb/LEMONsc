# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib import Path
from typing  import Optional
from loguru  import logger

# Constants ---------------------------------------------------------------------------------------------------------------#
LOG_RETENTION_DAYS = "10 days"
LOG_ROTATION_SIZE  = "10 MB"

# Logger configuration (only if not already configured) ------------------------------------------------------------------#
def _setup_logger(log_file: Optional[str] = None) -> None:
    """
    ________________________________________________________________________________________________________________
    Setup logger configuration if not already done.
    ________________________________________________________________________________________________________________
    Parameters:
    -> log_file (str) : Optional. Path to log file. If None, only console logging.
    ________________________________________________________________________________________________________________
    Notes:
        - Only configures logger if not already configured
        - Adds console handler (stdout)
        - Adds file handler if log_file provided
    ________________________________________________________________________________________________________________
    """
    # Only setup if no handlers exist
    if not logger._core.handlers:
        logger.remove()
        logger.add(
            sink=sys.stdout, 
            level="INFO", 
            format="<level>{level}: {message}</level>"
        )
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                level="INFO",
                format="{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
                rotation=LOG_ROTATION_SIZE,
                retention=LOG_RETENTION_DAYS,
                encoding="utf-8"
            )

# Convert state_dict to CPU (for saving/loading across devices) -----------------------------------------------------------#
def _to_cpu_state_dict(state_dict: dict) -> dict:
    """
    ________________________________________________________________________________________________________________
    Convert a state_dict with tensors to CPU.
    ________________________________________________________________________________________________________________
    Parameters:
    -> state_dict (dict) : Model state dictionary potentially containing GPU tensors
    ________________________________________________________________________________________________________________
    Returns:
    -> dict : State dictionary with all tensors moved to CPU
    ________________________________________________________________________________________________________________
    """
    cpu_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        else:
            cpu_state[k] = v
    return cpu_state

#--------------------------------------------------------------------------------------------------------------------------#