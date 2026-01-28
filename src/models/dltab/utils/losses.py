# Modules -----------------------------------------------------------------------------------------------------------------#
import torch.nn as nn

# Loss function selector --------------------------------------------------------------------------------------------------#
def get_loss_function(name: str, reduction: str = "mean", **kwargs) -> nn.Module:
    """Select and instantiate a loss function based on its name."""
    
    # Mapping of loss function names to their corresponding classes
    loss_map = {
        "mse"       : nn.MSELoss,
        "l1"        : nn.L1Loss,
        "smooth_l1" : nn.SmoothL1Loss,
        "huber"     : nn.HuberLoss,
    }
    
    # Name validation  
    name_lower = name.lower()
    if name_lower not in loss_map:
        available = ', '.join(sorted(loss_map.keys()))
        raise ValueError(f"Unknown loss function '{name}'. Available options: {available}")

    # Select the loss class
    loss_class = loss_map[name_lower]
    
    # Build the loss function with appropriate parameters
    try:
        return loss_class(reduction=reduction, **kwargs)
    
    except TypeError as e:
        raise TypeError(f"Invalid parameters for {name} loss function. Error: {str(e)}")

#--------------------------------------------------------------------------------------------------------------------------#