# Modules -----------------------------------------------------------------------------------------------------------------#
from torch import nn

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Optional

# Normalization types that make Linear bias redundant ---------------------------------------------------------------------#
_BATCH_NORM_TYPES = {"batch"}

# Selector of normalization layers ----------------------------------------------------------------------------------------#
def select_normalization(norm_type: str, dim: int) -> nn.Module:
    """Selector of normalization layers based on string identifier and feature dimension."""
    normalizations = {
        "batch" : lambda: nn.BatchNorm1d(dim),
        "layer" : lambda: nn.LayerNorm(dim),
        "rms"   : lambda: nn.RMSNorm(dim),
    }

    norm_lower = norm_type.lower()
    if norm_lower not in normalizations:
        raise ValueError(f"Normalization '{norm_type}' is not supported. Choose from {list(normalizations.keys())} or None.")

    return normalizations[norm_lower]()

# Helper: whether Linear bias is useful given normalization ---------------------------------------------------------------#
def bias_with_norm(bias: bool, norm_type: Optional[str]) -> bool:
    """Returns False when BatchNorm follows the Linear layer, making bias redundant."""
    return bias and norm_type not in _BATCH_NORM_TYPES

#---------------------------------------------------------------------------------------------------------------------------#
