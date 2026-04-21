# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing      import List, Optional, Tuple, Union, Dict, Any

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory import load_json_file

# [Helper] Extract ordered column names from a loaded JSON mapping dict ---------------------------------------------------#
def _cols_from_mapping(mapping: Dict[str, Any]) -> List[str]:
    """This function assumes the structure of the JSON file with a variable called column"""
    # Check that the key column exist for all variables
    for k in range(1, len(mapping) + 1):
        if "column" not in mapping[str(k)]:
            raise ValueError(f"Missing 'column' key for variable {k} in the mapping.")
    
    # return the column names in order of the variable keys (1, 2, ..., n)    
    return [mapping[str(k)]["column"] for k in range(1, len(mapping) + 1)]

# [Helper] Gather initial conditions from imbh-history.dat files in MOCCA simulations and the header columns --------------#
def _mocca_imbh_history_init_conds_header_columns(file_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Read all leading '#' lines from an imbh-history.dat file. Parse key=value pairs as initial conditions"""
    # Initialize empty dictionary for initial conditions and list for header lines ----------------------------------------#
    init_conds   = {}
    header_lines = []
    
    # Read the file and gather all leading '#' lines until the first non-header line --------------------------------------#
    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#'):
                header_lines.append(stripped)
            else:
                break
            
    # Check that we found header lines ------------------------------------------------------------------------------------#
    if not header_lines:
        raise ValueError(f"No header lines found in {file_path}")

    # Last '#' line in the leading block is the column header
    columns = header_lines[-1][1:].strip().split()

    # All preceding lines are init conditions (key = value format) --------------------------------------------------------#
    for hline in header_lines[:-1]:
        
        # Remove leading '#' and split on the first '=' to get key and value
        parts = hline[1:].strip().split('=', 1)
        
        # Only consider lines that have a key=value format
        if len(parts) == 2:
            key, value      = parts[0].strip(), parts[1].strip()
            init_conds[key] = value

    return init_conds, columns

# [Helper] Load MOCCA system.dat files and retrieve information for posterior processing ----------------------------------#
def _load_mocca_system_data(file_path: str, mapping_path: str, 
                            verbose: bool=True) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Helper function that load the evolution of the initial conditions of the system from a single MOCCA simulation"""
    # Validate input path -------------------------------------------------------------------------------------------------#
    if not os.path.exists(file_path):
        print(f"Warning: System file not found: {file_path}")
        return None, None

    # Load column mapping dictionary, assuming correct placement or files -------------------------------------------------#
    system_dict  = load_json_file(mapping_path, "system columns mapping", verbose)
    column_names = _cols_from_mapping(system_dict)

    # Load the system evolution data with pandas, using the column names from the mapping ---------------------------------#
    system_df = pd.read_csv(file_path, sep="\s+", header=None, names=column_names)
    system_df = system_df.apply(pd.to_numeric, errors='coerce')
    
    # Verbose output ------------------------------------------------------------------------------------------------------#
    if verbose:
        print(f"Successfully loaded {len(system_df)} rows of system evolution data with {len(column_names)} columns")

    return system_df, system_dict

#--------------------------------------------------------------------------------------------------------------------------#