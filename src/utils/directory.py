# Modules -----------------------------------------------------------------------------------------------------------------#
import os
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
def _load_json_file(json_path: str, description: str, verbose: bool= False):
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
def _parse_mocca_survey_imbh_history_init_conds(file_path: str):
    """
    ________________________________________________________________________________________________________________________
    Helper function: Load initial conditions from a single IMBH-HISTORY.DAT file into a dictionary, with respective column
    names for the dataframe.
    ________________________________________________________________________________________________________________________
    """
    init_conds = {}
    with open(file_path, 'r') as f:
        for i in range(86):
            line = f.readline().strip()
            if line.startswith('#'):
                parts = line[1:].strip().split('=', 1)
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    init_conds[key] = value
        columns_line = f.readline().strip()
        if not columns_line.startswith('#'):
            raise ValueError("Expected column headers at line 86")
        columns = columns_line[1:].split()
    return init_conds, columns

#--------------------------------------------------------------------------------------------------------------------------#
def _load_mocca_survey_system_data(system_path:str, verbose:bool =False):
    """
    ________________________________________________________________________________________________________________________
    Helper function: Load the evolution of the initial conditions of the sistem from a single MOCCA-Survey simulation, load column
    description into a dictionary.
    ________________________________________________________________________________________________________________________
    """
    if not os.path.exists(system_path):
        print(f"Warning: System file not found: {system_path}")
        return None, None

    with open('./rawdata/moccasurvey/mapping_dicts/system_columns.json', 'r') as f:
        system_dict = json.load(f)

    column_names = [system_dict[str(i)]['column'] for i in range(1, len(system_dict) + 1) if str(i) in system_dict]

    system_df = pd.read_csv(system_path, sep="\s+", header=None, names=column_names)
    system_df = system_df.apply(pd.to_numeric, errors='coerce')
    
    if verbose:
        print(f"Successfully loaded {len(system_df)} rows of system evolution data with {len(column_names)} columns")

    return system_df, system_dict

#--------------------------------------------------------------------------------------------------------------------------#
def load_mocca_survey_imbh_history(file_path: str, init_conds_sim: bool= False, col_description: bool= False, 
                            stellar_map: bool= False, 
                            init_conds_evo: bool= False,
                            verbose: bool = False):
    """
    ________________________________________________________________________________________________________________________
    Load MOCCA-Survey IMBH history data from simulation files.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> file_path       (str)  : Mandatory. Path to the IMBH history data file (imbh-history.dat)
    -> init_conds_sim  (bool) : Optional. Extract initial conditions from first 86 lines (imbh-history.dat).
    -> col_description (bool) : Optional. Load column descriptions from mapping file. Need to exits.
    -> stellar_map     (bool) : Optional. Load stellar type mappings from mapping file. Need to exits.
    -> init_conds_evo  (bool) : Optional. Load evolution of init conds of the system and their description. (system.dat)
    ________________________________________________________________________________________________________________________
    Returns:
    -> [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]
        
        sim_df       (pandas.DataFrame or None) : Main simulation data starting from line 87.
        init_conds   (dict or None)             : Initial simulation conditions (if init_conds_sim=True).
        col_dict     (dict or None)             : Column descriptions (if col_description=True).
        stellar_dict (dict or None)             : Stellar type mappings (if stellar_map=True).

        system_df    (pandas.DataFrame or None) : Evolution of the init conds in the system (if init_conds_evo=True).
        system_dict  (dict or None)             : Column description for dataframe (if init_conds_evo=True).

    ________________________________________________________________________________________________________________________
    Raises:
    -> FileNotFoundError, ValueError, JSONDecodeError
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    imbh_path   = f"{file_path}imbh-history.dat"
    system_path = f"{file_path}system.dat"

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Folder not found: {file_path}")
    
    # Initialize return values
    sim_df       = None 
    init_conds   = None
    stellar_dict = None
    col_dict     = None
    system_df    = None
    system_dict  = None
    
    if verbose: print(f"Loading IMBH history from: {imbh_path}")
    
    try:
        # Load optional JSON mapping files --------------------------------------------------------------------------------#
        if col_description: col_dict = _load_json_file( "./rawdata/moccasurvey/mapping_dicts/imbh_history.json", 
                                                        "description for imbh-history.dat")
        if stellar_map: stellar_dict = _load_json_file( "./rawdata/moccasurvey/mapping_dicts/stellar_types.json",
                                                        "stellar type mapping dictionary")
            
        # Parse initial conditions and columns for the dataframe ----------------------------------------------------------#
        if init_conds_sim:
            init_conds, columns = _parse_mocca_survey_imbh_history_init_conds(imbh_path)
        else:
            with open(imbh_path, 'r') as f:
                for _ in range(86): f.readline()
                columns = f.readline().strip()[1:].split()
        
        # Load main simulation data ---------------------------------------------------------------------------------------#
        if verbose: print("Loading main simulation data with pandas...")
        sim_df = pd.read_csv(imbh_path,
                            skiprows  = 87,
                            header    = None,
                            names     = columns,
                            sep       = r'\s+', 
                            index_col = False,
                            engine    = 'python')
        if verbose: print(f"Loaded {len(sim_df)} rows of simulation data")

        # Load system.dat file if requested
        if init_conds_evo: system_df, system_dict = _load_mocca_survey_system_data(system_path)

    except Exception as e:
        print(f"Error: {e}")
        raise

    return [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]
#--------------------------------------------------------------------------------------------------------------------------#
