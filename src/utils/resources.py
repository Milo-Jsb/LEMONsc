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
        
#--------------------------------------------------------------------------------------------------------------------------#