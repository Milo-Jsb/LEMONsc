# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import yaml
import json

import numpy  as np
import pandas as pd

# Path Management for dataset preparation ---------------------------------------------------------------------------------#
class PathManagerExperiment:
    """Centralized path management for the pipeline of feature building in the experiment."""
    def __init__(self, root_dir: str, dataset: str, exp_name: str, out_dir: str, fig_dir: str):
        self.data_path       = f"{root_dir}{dataset}/simulations/"
        self.out_path        = f"{out_dir}{exp_name}/{dataset}/"
        self.out_figs        = f"{fig_dir}{exp_name}/{dataset}/"
        self.stratified_path = f"{root_dir}{dataset}/"
        
        # Create directories
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.out_figs, exist_ok=True)
    
    def get_stratified_file_path(self, env_type: str) -> str:
        """Get path for stratified simulation files."""
        return os.path.join(self.stratified_path, f"{env_type}_simulations.txt")
    
# Listing directories -----------------------------------------------------------------------------------------------------#
def list_all_directories(directory: str):
    """Recursively list all directory names inside the specified directory path."""
    all_dirs = []
    
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))
    return all_dirs

# Safe Load of JSONfiles --------------------------------------------------------------------------------------------------#
def load_json_file(json_path: str, description: str, verbose: bool= False):
    """Helper function: Load a JSON existing file from a given path."""
    
    if os.path.exists(json_path):
        if verbose: print(f"Loading {description} from: {json_path}")
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        if verbose: print(f"Warning: {description} file not found: {json_path}")
        return None
    
#--------------------------------------------------------------------------------------------------------------------------#
def load_yaml_dict(path: str, verbose: bool = False):
    """
    ________________________________________________________________________________________________________________________
    Load search space configuration from a YAML file.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> path    (str)  : Mandatory. Path to the YAML configuration file.
    -> verbose (bool) : Optional. Enable verbose logging for debugging purposes.
    ________________________________________________________________________________________________________________________
    Returns:
    -> dict : Dictionary containing the search space configuration loaded from the YAML file.
    ________________________________________________________________________________________________________________________
    Raises:
    -> FileNotFoundError, yaml.YAMLError, TypeError
    ________________________________________________________________________________________________________________________
    Notes:
    -> Supports tuple construction in YAML using !tuple tag.
    -> Verbose logging provides insights into loading process.
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Tuple constructor if needed -----------------------------------------------------------------------------------------#
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor('!tuple', tuple_constructor)

    # Verbose logging -----------------------------------------------------------------------------------------------------#
    if verbose: print(f"Loading configuration from: {path}")
    
    # Load YAML file ------------------------------------------------------------------------------------------------------#
    try:
        with open(path, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        
        if verbose: print(f"Successfully loaded configuration with {len(yaml_dict)} parameters")
        return yaml_dict
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise
    
#--------------------------------------------------------------------------------------------------------------------------#