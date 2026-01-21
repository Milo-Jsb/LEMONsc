# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Tuple, Optional, List, Dict, Union, Any
from dataclasses    import dataclass, field

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory     import load_json_file
from src.processing.format   import time_preparation, target_preparation, apply_noise
from src.processing.features import compute_physical_parameters, determine_formation_channel

# Default configuration of the experiments --------------------------------------------------------------------------------#
DEFAULT_FEATURE_NAMES = [                                     # - Features names to retrieve (IN ORDER)
        "t",                                                  #    - Time evolution of the simulation 
        "t_coll", "t_relax", "t_cc", "t_cross",               #    - Timescales
        "N",                                                  #    - Number of objects
        "M_tot", "M_mean", "M_max", "M_crit",                 #    - Mass-related 
        "M_loss",                                             #    - Mass loss channels
        "safnum",                                             #    - Safranov Number 
        "rho(R_h)", "rho(R_c)",                               #    - Density (half mass radius[num], core radius[mass])
        "R_*", "R_h", "R_core", "R_tid",                      #    - Radii
        "z",                                                  #    - Metallicity
        "fracbin",                                            #    - Fraction of binaries of the model
        "type_sim"]                                           #    - Categorical feature: type of simulation 
                                                              #       (FAST, SLOW)

DEFAULT_EXPECTED_ORDER = [                                    # - Actual column names of the tabular features in the 
        "tcoll", "trelax", "tcc", "tcross",                   #   expected order
        "n", 
        "m_tot", "m_mean", "m_max", "mcrit", 
        "m_loss_tot",
        "safnum",
        "rho_half", "rohut",
        "stellar_radius", "rh", "cr", "rtid",
        "z", 
        "fracbin"] 

@dataclass
class MoccaSurveyExperimentConfig:
    """
    ________________________________________________________________________________________________________________________
    Base Configuration elements for MOCCA-Survey Dataset preparation and Experiments.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> feature_names        : List of feature names to retrieve (in order)
    -> expected_order       : List of actual column names of the tabular features in the expected order
    -> target_name          : Target name (mass evolution of the IMBH)
    -> min_points_threshold : Minimum number of points per simulation to be used
    -> requires_temp_evol   : Whether some features require time-evolution data
    -> sample_window        : Whether to sample a window of the simulation or use full data
    -> retain_order         : Whether to retain the order of the simulation points
    -> mapping_dics_dir     : Directory path to mapping dictionaries
    -> time_column_imbh     : Name of the time column in the IMBH history data
    -> time_column_system   : Name of the time column in the system data
    -> mass_column_imbh     : Name of the mass column in the IMBH history data
    ________________________________________________________________________________________________________________________
    """
    feature_names        : List[str]      = field(default_factory=lambda: DEFAULT_FEATURE_NAMES.copy())
    expected_order       : List[str]      = field(default_factory=lambda: DEFAULT_EXPECTED_ORDER.copy())
    target_name          : str            = "M_MMO"                
    min_points_threshold : int            = 1000                   
    requires_temp_evol   : bool           = False                 
    sample_window        : bool           = True                   
    retain_order         : bool           = True                  
    mapping_dics_dir     : str            = "./rawdata/moccasurvey/mapping_dicts/"
    time_column_imbh     : str            = "time[Myr]"  
    time_column_system   : Optional[str]  = "tphys"
    mass_column_imbh     : str            = "massNew[Msun](10)"
    
def_config = MoccaSurveyExperimentConfig()

# Load MOCCASURVEY simulations files and retrieve information for posterior processing ------------------------------------#
def load_moccasurvey_imbh_history(file_path: str, init_conds_sim: bool= False, col_description: bool= False, 
                                  stellar_map    : bool                        = False, 
                                  init_conds_evo : bool                        = False,
                                  verbose        : bool                        = False,
                                  config         : MoccaSurveyExperimentConfig = def_config
                                  )-> Tuple[List[Optional[Union[pd.DataFrame, Dict[str, Any]]]], 
                                            List[Optional[Union[pd.DataFrame, Dict[str, Any]]]]]:
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
        # Load optional JSON mapping files (assuming correct placement of files) ------------------------------------------#
        if col_description: col_dict = load_json_file( f"{config.mapping_dics_dir}imbh_history.json", 
                                                        "description for imbh-history.dat")
        if stellar_map: stellar_dict = load_json_file( f"{config.mapping_dics_dir}stellar_types.json",
                                                        "stellar type mapping dictionary")
            
        # Parse initial conditions and columns for the dataframe ----------------------------------------------------------#
        if init_conds_sim:
            init_conds, columns = __parse_moccasurvey_imbh_history_init_conds(imbh_path)
        else:
            with open(imbh_path, 'r') as f:
                for _ in range(86): f.readline()
                columns = f.readline().strip()[1:].split()
        
        # Load main simulation data ---------------------------------------------------------------------------------------#
        if verbose: print("Loading main simulation data with pandas...")
        sim_df = pd.read_csv(imbh_path, skiprows= 87, header= None, names= columns, sep= r'\s+', index_col= False,
                             engine = 'python')
        if verbose: print(f"Loaded {len(sim_df)} rows of simulation data")

        # Load system.dat file if requested
        if init_conds_evo: system_df, system_dict = __load_moccasurvey_system_data(system_path)

    except Exception as e:
        print(f"Error: {e}")
        raise

    return [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]

# Process a single MOCCA simulation file ----------------------------------------------------------------------------------#
def process_single_mocca_simulation(imbh_df: pd.DataFrame, system_df: pd.DataFrame, meta_dict:Optional[Dict[str, Any]], 
                                    points_per_sim   : Optional[Union[int, float]], 
                                    config           : MoccaSurveyExperimentConfig = def_config, 
                                    environment      : Optional[str]               = None,
                                    augment          : bool                        = False,
                                    noise            : bool                        = False, 
                                    n_virtual        : int                         = 1
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function to preprocess one simulation elements from moccasurvey dataset"""

    # Validate the number of input points and determine sampling size only if augmentation is enabled
    n_points    = len(imbh_df)
    sample_size = None
    
    # If points_per_sim is an integer, should be greater than the number of points in the simulation
    if isinstance(points_per_sim, int):
        # Fixed number of points
        sample_size = points_per_sim
        if n_points < sample_size:
            raise ValueError(f"Simulation has {n_points} points, requires at least {sample_size}")
    
    # If points_per_sim is a float, should be between 0 and 1 as a proportional sampling
    elif isinstance(points_per_sim, float):
        if not (0 < points_per_sim <= 1):
            raise ValueError(f"points_per_sim must be between 0 and 1 when float, got {points_per_sim}")
        sample_size = int(n_points * points_per_sim)
        if sample_size < 1:
            raise ValueError(f"Proportional sampling resulted in 0 points (n_points={n_points}, ratio={points_per_sim})")
    
    # Otherwise, raise error
    else:
        raise TypeError(f"points_per_sim must be int or float, got {type(points_per_sim)}")
    
    # Sort input dataframes and calculate the total evolution time 
    imbh_df   = imbh_df.sort_values(config.time_column_imbh).reset_index(drop=True)
    system_df = system_df.sort_values(config.time_column_system).reset_index(drop=True)
    
    # Ensure both dataframes have the same length by dropping rows with NaN in either
    if len(imbh_df) != len(system_df):
        min_len   = min(len(imbh_df), len(system_df))
        imbh_df   = imbh_df.iloc[:min_len]
        system_df = system_df.iloc[:min_len]
    
    # Drop rows where either dataframe has NaN to maintain alignment
    valid_mask = ~(imbh_df.isna().any(axis=1) | system_df.isna().any(axis=1))
    imbh_df    = imbh_df[valid_mask].reset_index(drop=True)
    system_df  = system_df[valid_mask].reset_index(drop=True)
    
    # Update n_points after cleaning
    n_points = len(imbh_df)
    
    # Validate after cleaning
    if n_points == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Re-validate sample_size after cleaning NaN values
    if sample_size is not None and n_points < sample_size:
        raise ValueError(
            f"After cleaning NaN values, simulation has {n_points} valid points, "
            f"but requires at least {sample_size} for sampling. "
            f"This simulation has insufficient clean data for the requested sample size."
        )

    # This creates a sampled of points from a given number of elements
    def sample_window() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        # If sample_size is None, use all available points
        if sample_size is None or sample_size >= n_points:
            sidx = np.arange(n_points)
            df_sampled     = imbh_df.copy()
            system_sampled = system_df.copy()
            
            # Reset physical time to zero
            df_sampled[config.time_column_imbh]       -= df_sampled[config.time_column_imbh].iloc[0]
            system_sampled[config.time_column_system] -= system_sampled[config.time_column_system].iloc[0]
            
            return df_sampled, system_sampled, sidx
        
        # Validate we can actually sample
        if n_points < sample_size:
            raise ValueError(f"Cannot sample {sample_size} points from {n_points} available points")
        
        # choose an end point randomly within the allowed window
        start = np.random.randint(0, n_points - sample_size + 1)
        end   = start + sample_size
 
        # Create index array
        sidx = np.arange(start, end)

        # Sample the dataframes (no need for dropna since already cleaned)
        df_sampled     = imbh_df.iloc[sidx].copy()
        system_sampled = system_df.iloc[sidx].copy()
        
        # Reset physical time to zero for the sampled window
        df_sampled[config.time_column_imbh]       -= df_sampled[config.time_column_imbh].iloc[0]
        system_sampled[config.time_column_system] -= system_sampled[config.time_column_system].iloc[0]
        
        return df_sampled, system_sampled, sidx

    # Create the output of the simulation in point-to-point format 
    def make_features(df_sampled, system_sampled, iconds_dict, noise, environment, config):

        # Compute cluster features
        cluster_feats = compute_mocca_cluster_features(system_sampled, df_sampled, iconds_dict, noise, 
                                                       sim_env        = environment,
                                                       temp_evol      = config.requires_temp_evol)
        
        # Get temporal values
        time_evol = time_preparation(time_evolution = df_sampled[config.time_column_imbh])
        m_evol    = target_preparation(mass_evolution = df_sampled[config.mass_column_imbh])
        
        # Include noise if needed
        t = apply_noise(time_evol) if noise else time_evol
        m = apply_noise(m_evol) if noise else m_evol

        # Convert categorical variable to numeric codes
        fchannel_codes = {"FAST": 0, "SLOW": 1}
        type_sim_code  = fchannel_codes.get(cluster_feats["type_sim"], np.nan)

        # Define numeric features names in the expected order
        expected_order  = config.expected_order    
        feature_columns = [t]

        # Handle temporal vs static features
        for k in expected_order:
            feat_val = cluster_feats.get(k, np.nan)
            if config.requires_temp_evol and isinstance(feat_val, np.ndarray):
                # Use time-evolving values
                feature_columns.append(feat_val)
            else:
                # Repeat scalar value for all timesteps
                feature_columns.append(np.full(len(t), feat_val))
        
        # Add categorical feature (repeated for all timesteps)
        feature_columns.append(np.full(len(t), type_sim_code))
        
        sim_features = np.column_stack(feature_columns)
        target_mass  = m
        
        return sim_features, target_mass

    # List to store the information
    feats_all, targs_all, idxs_all = [], [], []

    # Determine number of iterations based on augmentation setting
    iterations = n_virtual if augment else 1
    
    for _ in range(iterations):
        
        if config.sample_window:
            df_sampled, system_sampled, sidx = sample_window()
        else:
            df_sampled     = imbh_df.copy()
            system_sampled = system_df.copy()
            sidx           = np.arange(n_points)
            
        # Retrieve features and targets
        feats, targs = make_features(df_sampled, system_sampled, meta_dict, noise, environment, config)
        
        # Append all
        feats_all.append(feats)
        targs_all.append(targs)
        idxs_all.append(sidx)

    return np.vstack(feats_all), np.concatenate(targs_all), np.concatenate(idxs_all)

# Retrieve elements from the mocca simulation and compute relevant features -----------------------------------------------#
def compute_mocca_cluster_features(system_df: pd.DataFrame, imbh_df: pd.DataFrame, 
                                   iconds_dict : Optional[Dict[str, Any]] = None, 
                                   noise       : bool                     = False, 
                                   sim_env     : Optional[str]            = None,
                                   temp_evol   : bool                     = False
                                   ) -> Dict[str, Union[float, str]]:
    """
    ________________________________________________________________________________________________________________________
    Compute relevant cluster features from a MOCCA simulation data 
    ________________________________________________________________________________________________________________________
    Parameters:
    -> system_df   (pd.DataFrame)             : DataFrame with system evolution data. Mandatory.
    -> imbh_df     (pd.DataFrame)             : DataFrame with IMBH formation
    -> iconds_dict (Optional[Dict[str, Any]]) : Dictionary with initial conditions. Optional.
    -> noise       (bool)                     : Whether to apply noise to the features. Default is False.
    -> sim_env     (Optional[str])            : Predefined simulation environment type. If None, it will be determined.
                                                Default is None.
    -> temp_evol   (bool)                     : Whether to compute features as temporal evolution (arrays) or single
                                                values. Default is False
    ________________________________________________________________________________________________________________________
    Returns:
        - features_dict (Dict[str, Union[float, str]]) : Dictionary with computed cluster features.
    ________________________________________________________________________________________________________________________
    Notes:
        - compute_mocca_cluster_features() is design to work with the output of MOCCA simulations. It extracts key cluster
          properties from the system evolution data and computes derived features using established physical formulas.
          Any changes in the MOCCA output format may require adjustments to this function.
        
        - If temp_evol is True, features will be returned as arrays representing their evolution over for the full 
          simulation, meaning an output with dimensions:
            - Dictionary[key]    : np.ndarray of len() = n_dimensions, where each element inside correspond to the full
              temporal evolution of that feature, for each simulation, which is not necessary an homogeneous array, as some 
              features are scalar and not all simulations have the same temporal resolution or evolution time.
        
        - If temp_evol is False, features will be returned as single values representing the initial state of the cluster.
          The output will have the following dimensions:
            - Dictionary[key]    : np.ndarray of len() = n_dimensions, where each element inside correspond to the initial 
              value of that feature. In here all arrays share the same dimentions so its homogeneous, and the relevant 
              leghtn of each array its given by the number of simulations processed.
    ________________________________________________________________________________________________________________________
    """
    # Extract base values from the simulation data ------------------------------------------------------------------------#
    try: 
        if temp_evol:
            
            n_total    = system_df["nt"].values - system_df["nbb"].values + (2*system_df["nbb"].values)
            m_loss_tot = system_df["sloses"].values 

                         
            base_values = {
                'tau'         : np.max(system_df["tphys"].values) - system_df["tphys"].values,
                'rh'          : system_df["r_h"].values,
                'v_disp'      : system_df["vc"].values,
                'm_tot'       : system_df["smt"].values,
                'n'           : n_total,
                'm_mean'      : system_df["atot"].values,
                'm_max'       : system_df["smsm"].values,
                "m_loss_tot"  : m_loss_tot,
                'rohut'       : system_df["rohut"].values,
                'cr'          : system_df["rc"].values,
                'rtid'        : system_df["rtid"].values
            }
            
        else:
            n_total    = system_df["nt"].iloc[0] - system_df["nbb"].iloc[0] + (2*system_df["nbb"].iloc[0])
            m_loss_tot = system_df["sloses"].iloc[-1]                         
                         
            base_values = {
                'tau'         : system_df["tphys"].iloc[-1],
                'rh'          : system_df["r_h"].iloc[0],
                'v_disp'      : system_df["vc"].iloc[0],
                'm_tot'       : system_df["smt"].iloc[0],
                'n'           : n_total,
                'm_mean'      : system_df["atot"].iloc[0],
                'm_max'       : system_df["smsm"].iloc[0],
                "m_loss_tot"  : m_loss_tot,
                'rohut'       : system_df["rohut"].iloc[0],
                'cr'          : system_df["rc"].iloc[0],
                'rtid'        : system_df["rtid"].iloc[0]
            }

        if iconds_dict:
            z_val       = float(iconds_dict["zini"])
            fracbin_val = float(iconds_dict.get("fracb", 0.0))

            if temp_evol:
                # Repeat scalar values to match temporal dimension
                n_timesteps            = len(system_df)
                base_values["z"]       = np.full(n_timesteps, z_val)
                base_values["fracbin"] = np.full(n_timesteps, fracbin_val)
            else:
                base_values["z"]       = z_val
                base_values["fracbin"] = fracbin_val 

    except Exception as e:
        raise ValueError(f"Error extracting base cluster features: {e}")
    
    # Apply noise if required, no consideration for "n" As it represents a count of elements in the cluster.
    if noise:
        for key in base_values:
            if key != "n":
                base_values[key] = apply_noise(base_values[key], sigma=0.01)

    # Compute derived features using physical formulas -------------------------------------------------------------------#
    derived_values = compute_physical_parameters(
                        time_values      = base_values["tau"],
                        rh_values        = base_values['rh'],
                        v_disp_values    = base_values['v_disp'],
                        n_values         = base_values['n'],
                        m_mean_values    = base_values['m_mean'],
                        m_max_values     = base_values['m_max'],
                        comp_stellar_val = temp_evol)
            
    # Type of the environment of the simulation (repeat if the simulation its already classified)
    if (sim_env is not None):  env = sim_env
    
    # Determine the formation channel based on initial core density and time of IMBH formation if first seen
    else: env = determine_formation_channel(imbh_df, mass_colum_name="massNew[Msun](10)", time_column_name="time[Myr]")

   
    return {**base_values, **derived_values, 'type_sim': env}

# [Helper] Text to Dictionary IMBH file  ----------------------------------------------------------------------------------#
def __parse_moccasurvey_imbh_history_init_conds(file_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """Helper function that load initial conditions from a single IMBH-HISTORY.DAT file into a dictionary, with respective 
    column names for the dataframe"""
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

# [Helper] Text to Dictionary System file ----------------------------------------------------------------------------------#
def __load_moccasurvey_system_data(system_path:str, verbose:bool=False, config:MoccaSurveyExperimentConfig=def_config
                                   ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Helper function that load the evolution of the initial conditions of the system from a single MOCCA-Survey 
    simulation, load column description into a dictionary"""
    
    if not os.path.exists(system_path):
        print(f"Warning: System file not found: {system_path}")
        return None, None

    # Load column mapping dictionary, assuming correct placement or files
    with open(config.mapping_dics_dir + 'system_columns.json', 'r') as f:
        system_dict = json.load(f)

    column_names = [system_dict[str(i)]['column'] for i in range(1, len(system_dict) + 1) if str(i) in system_dict]

    system_df = pd.read_csv(system_path, sep="\s+", header=None, names=column_names)
    system_df = system_df.apply(pd.to_numeric, errors='coerce')
    
    if verbose:
        print(f"Successfully loaded {len(system_df)} rows of system evolution data with {len(column_names)} columns")

    return system_df, system_dict

#--------------------------------------------------------------------------------------------------------------------------#