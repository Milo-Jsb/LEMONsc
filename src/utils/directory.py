# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import yaml
import json

import numpy  as np
import pandas as pd

# Listing directories -----------------------------------------------------------------------------------------------------#
def list_all_directories(directory: str):
    """
    ________________________________________________________________________________________________________________________
    Recursively list all directory names inside the specified directory path.
    ________________________________________________________________________________________________________________________
    """
    all_dirs = []
    
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))
    return all_dirs

# Safe Load of JSONfiles --------------------------------------------------------------------------------------------------#
def load_json_file(json_path: str, description: str, verbose: bool= False):
    """
    ________________________________________________________________________________________________________________________
    Helper function: Load a JSON existing file from a given path.
    ________________________________________________________________________________________________________________________
    """
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
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    if verbose: print(f"Loading configuration from: {path}")
    
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