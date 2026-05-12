# Architectures available for DLTabular -----------------------------------------------------------------------------------#
from .mlp  import MLPRegressor
from .node import NODERegressor
from .ftt  import FTTRegressor

# Define what is exported when using 'from dltab.archs import *'
__all__ = ["MLPRegressor", "NODERegressor", "FTTRegressor"]

#--------------------------------------------------------------------------------------------------------------------------#