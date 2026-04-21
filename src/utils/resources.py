# Modules -----------------------------------------------------------------------------------------------------------------#
import random
import torch
import numpy as np

# Helper function to check GPU availability -------------------------------------------------------------------------------#
def check_gpu_available() -> bool: 
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    
    except ImportError:
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            return True
        
        except (ImportError, Exception):
            return False

# Manual seeder for reproducibility ---------------------------------------------------------------------------------------#
def set_numpy_torch_seed(seed_num: int, idx: int = 0, disable_deterministic: bool = False, disable_benchmark: bool = False
                         ) -> int:
    """
    Set seed for Python, NumPy, and PyTorch (CPU + all CUDA devices) for reproducibility.
    """
    # Define unique seed by combining base seed with index (useful for multiple runs or workers)
    seed = seed_num + idx
    
    # Python stdlib (used by some libraries internally)
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # All CUDA devices (covers single-GPU and multi-GPU)
    torch.cuda.manual_seed_all(seed)
    # cuDNN: disable benchmark mode and enforce deterministic algorithms
    torch.backends.cudnn.deterministic = not disable_deterministic
    torch.backends.cudnn.benchmark     = not disable_benchmark
    
    # Return the final seed used (useful for logging or debugging)
    return seed
#--------------------------------------------------------------------------------------------------------------------------#