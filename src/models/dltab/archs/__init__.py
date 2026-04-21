# Architectures available for DLTabular -----------------------------------------------------------------------------------#
from .mlp  import MLPRegressor
from .node import NODERegressor

# Define what is exported when using 'from dltab.archs import *'
__all__ = ["MLPRegressor", "NODERegressor"]

#--------------------------------------------------------------------------------------------------------------------------#