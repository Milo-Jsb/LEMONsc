# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Tuple, Optional, List, Dict, Union, Any
from dataclasses    import dataclass, field

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory                        import load_json_file
from src.processing.format                      import time_preparation, target_preparation, apply_noise
from src.processing.features                    import compute_physical_parameters
from src.processing.constructors.utils._formats import _mocca_imbh_history_init_conds_header_columns, _load_mocca_system_data
from src.processing.constructors.utils._config   import MoccaSurveyExperimentConfig
    
# Initialize default configuration instance for use in functions that require it but not provided explicitly
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
            init_conds, columns = _mocca_imbh_history_init_conds_header_columns(imbh_path)
        else:
            _, columns = _mocca_imbh_history_init_conds_header_columns(imbh_path)
        
        # Load main simulation data ---------------------------------------------------------------------------------------#
        if verbose: print("Loading main simulation data with pandas...")
        sim_df = pd.read_csv(imbh_path, skiprows= 87, header= None, names= columns, sep= r'\s+', index_col= False,
                             engine = 'python')
        if verbose: print(f"Loaded {len(sim_df)} rows of simulation data")

        # Load system.dat file if requested
        if init_conds_evo: system_df, system_dict = _load_mocca_system_data(system_path, 
                                                                            config.mapping_dics_dir + "system_columns.json", 
                                                                            verbose)

    except Exception as e:
        print(f"Error: {e}")
        raise

    return [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]

# Process a single MOCCA survey simulation file ----------------------------------------------------------------------------------#
def process_single_moccasurvey_simulation(imbh_df: pd.DataFrame, system_df: pd.DataFrame, meta_dict:Optional[Dict[str, Any]], 
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
    def sample_window(reset_time_window: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        # If sample_size is None, use all available points
        if sample_size is None or sample_size >= n_points:
            sidx           = np.arange(n_points)
            df_sampled     = imbh_df.copy()
            system_sampled = system_df.copy()
            
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
        
        # Reset physical time to zero for the sampled window if requested
        if reset_time_window:
            df_sampled[config.time_column_imbh]       -= df_sampled[config.time_column_imbh].iloc[0]
            system_sampled[config.time_column_system] -= system_sampled[config.time_column_system].iloc[0]
        
        return df_sampled, system_sampled, sidx

    # Create the output of the simulation in point-to-point format 
    def make_features(df_sampled, system_sampled, iconds_dict, noise, environment, config):

        # Compute cluster features
        cluster_feats = compute_moccasurvey_cluster_features(system_sampled, df_sampled, iconds_dict, noise, 
                                                             sim_env        = environment,
                                                             temp_evol      = config.requires_temp_evol,
                                                             mass_column    = config.mass_column_imbh,
                                                             time_column    = config.time_column_imbh)
        
        # Get temporal window and mass evolution for the target variable   
        time_evol = time_preparation(time_evolution = df_sampled[config.time_column_imbh])
        m_evol    = target_preparation(mass_evolution = df_sampled[config.mass_column_imbh])
        
        # Include noise if needed
        t = apply_noise(time_evol) if noise else time_evol
        m = apply_noise(m_evol) if noise else m_evol

        # Extract initial processed mass of the MMO and store in cluster features
        cluster_feats["m_mmo_0"] = m[0]

        # Convert categorical variable to numeric codes using config class_labels (supports N classes)
        class_labels   = getattr(config, 'class_labels', ["FAST", "SLOW"])
        fchannel_codes = {label: float(i) for i, label in enumerate(class_labels)}
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
            df_sampled, system_sampled, sidx = sample_window(config.reset_time_window)
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
def compute_moccasurvey_cluster_features(system_df: pd.DataFrame, imbh_df: Optional[pd.DataFrame] = None,
                                         iconds_dict : Optional[Dict[str, Any]] = None, 
                                         noise       : bool                     = False,
                                         sim_env     : Optional[str]            = None,
                                         temp_evol   : bool                     = False,
                                         mass_column : Optional[str]            = None,
                                         time_column : Optional[str]            = None
                                         ) -> Dict[str, Union[float, str]]:
    """
    ________________________________________________________________________________________________________________________
    Compute relevant cluster features from a MOCCA simulation data 
    ________________________________________________________________________________________________________________________
    Parameters:
    -> system_df   (pd.DataFrame)             : DataFrame with system evolution data. Mandatory.
    -> imbh_df     (pd.DataFrame)             : DataFrame with IMBH formation history. Optional.
    -> iconds_dict (Optional[Dict[str, Any]]) : Dictionary with initial conditions. Optional.
    -> noise       (bool)                     : Whether to apply noise to the features. Default is False.
    -> sim_env     (Optional[str])            : Pre-assigned environment label (e.g. 'FAST', 'SLOW', or any
                                                category from classify_simulations_by_mass). If None, the
                                                formation channel is determined automatically from imbh_df
                                                using determine_formation_channel().
    -> temp_evol   (bool)                     : Whether to compute features as temporal evolution (arrays) or single
                                                values. Default is False
    -> mass_column (Optional[str])            : Name of the mass column in imbh_df. Used for automatic channel
                                                determination when sim_env is None.
    -> time_column (Optional[str])            : Name of the time column in imbh_df. Used for automatic channel
                                                determination when sim_env is None.
    ________________________________________________________________________________________________________________________
    Returns:
        - features_dict (Dict[str, Union[float, str]]) : Dictionary with computed cluster features, including
          'type_sim' with the assigned environment/category label.
    ________________________________________________________________________________________________________________________
    """
    # Extract base values from the simulation data ------------------------------------------------------------------------#
    try: 
        if temp_evol:
            
            n_total = system_df["nt"].values - system_df["nbb"].values + (2*system_df["nbb"].values)
            f_bin   = system_df["nbb"].values / system_df["nt"].values   
                      
            base_values = {
                'tau'    : np.max(system_df["tphys"].values) - system_df["tphys"].values,
                'rh'     : system_df["r_h"].values,
                'v_disp' : system_df["vc"].values,
                'm_tot'  : system_df["smt"].values,
                'n'      : n_total,
                'm_mean' : system_df["atot"].values,
                'm_max'  : system_df["smsm"].values,
                'rohut'  : system_df["rohut"].values,
                'cr'     : system_df["rc"].values,
                'rtid'   : system_df["rtid"].values,
                'r70%'   : system_df["r70%"].values,
                'fbin'   : f_bin
            }
            
        else:
            n_total    = system_df["nt"].iloc[0] - system_df["nbb"].iloc[0] + (2*system_df["nbb"].iloc[0])
            f_bin      = system_df["nbb"].iloc[0] / system_df["nt"].iloc[0]
             
            base_values = {
                'tau'    : system_df["tphys"].iloc[-1],
                'rh'     : system_df["r_h"].iloc[0],
                'v_disp' : system_df["vc"].iloc[0],
                'm_tot'  : system_df["smt"].iloc[0],
                'n'      : n_total,
                'm_mean' : system_df["atot"].iloc[0],
                'm_max'  : system_df["smsm"].iloc[0],
                'rohut'  : system_df["rohut"].iloc[0],
                'cr'     : system_df["rc"].iloc[0],
                'rtid'   : system_df["rtid"].iloc[0],
                'r70%'   : system_df["r70%"].iloc[0],
                'fbin'   : f_bin
            }
        
        if iconds_dict:
            z_val = float(iconds_dict["zini"])
            
            # Repeat scalar values to match temporal dimension
            if temp_evol:
                n_timesteps      = len(system_df)
                base_values["z"] = np.full(n_timesteps, z_val)
                
            else:
                base_values["z"] = z_val

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

    # Determine simulation category label
    if sim_env is not None:
        type_sim = sim_env
    else:
        type_sim = np.nan

    return {**base_values, **derived_values, "type_sim": type_sim}

#--------------------------------------------------------------------------------------------------------------------------#