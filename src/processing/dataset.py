
# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json
import traceback

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Tuple, Optional, List, Dict, Union, Any
from loguru._logger import Logger
from dataclasses    import dataclass, field

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory       import load_json_file
from src.processing.format     import time_preparation, target_preparation, apply_noise
from src.processing.features   import compute_cluster_features
from src.processing.filters    import safe_downsampling_of_points

# Default configuration of the experiments --------------------------------------------------------------------------------#
DEFAULT_FEATURE_NAMES = [                             # - Features names to retrieve (IN ORDER)
        "t", "t_coll", "t_relax", "t_cc", "t_cross",  #     - Time-relared 
        "M_tot", "M_mean", "M_max", "M_crit",         #     - Mass-related 
        "safnum",                                     #     - Safranov Number 
        "rho(R_h)",                                   #     - Density
        "R_*", "R_h", "R_core",                       #     - Radii
        "z",                                          #     - Metallicity
        "fracbin",                                    #     - Fraction of binaries of the model
        "type_sim"]                                   #     - Categorical feature: type of simulation (FAST, SLOW, STEADY)

@dataclass
class MoccaSurveyExperimentConfig:
    feature_names      : List[str] = field(default_factory=lambda: DEFAULT_FEATURE_NAMES.copy())
    target_name        : str  = "M_MMO"                # - Target name (mass evolution of the IMBH)
    requires_temp_evol : bool = True                   # - Flag to use temporal evolution of cluster features
    min_sim_points     : int  = 1000                   # - Minimum number of points per simulation to be used
 
def_config = MoccaSurveyExperimentConfig()

# Helper functions --------------------------------------------------------------------------------------------------------#
def __parse_moccasurvey_imbh_history_init_conds(file_path: str):
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

#--------------------------------------------------------------------------------------------------------------------------#
def __load_moccasurvey_system_data(system_path:str, verbose:bool =False):
    """Helper function that load the evolution of the initial conditions of the system from a single MOCCA-Survey 
    simulation, load column description into a dictionary"""
    
    if not os.path.exists(system_path):
        print(f"Warning: System file not found: {system_path}")
        return None, None

    # Load column mapping dictionary, assuming correct placement or files
    with open('./rawdata/moccasurvey/mapping_dicts/system_columns.json', 'r') as f:
        system_dict = json.load(f)

    column_names = [system_dict[str(i)]['column'] for i in range(1, len(system_dict) + 1) if str(i) in system_dict]

    system_df = pd.read_csv(system_path, sep="\s+", header=None, names=column_names)
    system_df = system_df.apply(pd.to_numeric, errors='coerce')
    
    if verbose:
        print(f"Successfully loaded {len(system_df)} rows of system evolution data with {len(column_names)} columns")

    return system_df, system_dict

#--------------------------------------------------------------------------------------------------------------------------#
def process_single_simulation(imbh_df: pd.DataFrame, system_df: pd.DataFrame, meta_dict:Optional[Dict[str, Any]], 
                             points_per_sim   : Optional[Union[int, float]], 
                             config           : MoccaSurveyExperimentConfig = def_config, 
                             environment      : Optional[str] = None,
                             augment          : bool = False,
                             noise            : bool = False, 
                             n_virtual        : int  = 1):
    """Helper function to preprocess one simulation elements from moccasurvey dataset"""

    # Validate the number of input points and determine sampling size only if augmentation is enabled
    n_points    = len(imbh_df)
    sample_size = None
    
    if augment:
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
    imbh_df   = imbh_df.sort_values("time[Myr]").reset_index(drop=True)
    system_df = system_df.sort_values("tphys").reset_index(drop=True)
    
    # Ensure both dataframes have the same length by dropping rows with NaN in either
    # This maintains temporal alignment between IMBH history and system evolution
    if len(imbh_df) != len(system_df):
        min_len = min(len(imbh_df), len(system_df))
        imbh_df = imbh_df.iloc[:min_len]
        system_df = system_df.iloc[:min_len]
    
    # Drop rows where either dataframe has NaN to maintain alignment
    valid_mask = ~(imbh_df.isna().any(axis=1) | system_df.isna().any(axis=1))
    imbh_df = imbh_df[valid_mask].reset_index(drop=True)
    system_df = system_df[valid_mask].reset_index(drop=True)
    
    # Update n_points after cleaning
    n_points = len(imbh_df)
    
    # Validate after cleaning
    if n_points == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Re-validate and adjust sample_size if augmentation is enabled
    if augment and sample_size is not None:
        if n_points < sample_size:
            raise ValueError(
                f"After cleaning NaN values, simulation has {n_points} valid points, "
                f"but requires at least {sample_size} for sampling. "
                f"This simulation has insufficient clean data for augmentation."
            )

    # This creates a sampled of points from a given number of elements (only used when augment=True)
    def sample_window() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        if sample_size is None:
            raise ValueError("sample_window called but sample_size not defined. This should only be called when augment=True.")
        
        # choose an end point randomly within the allowed window
        end = np.random.randint(sample_size , n_points)
 
        # Create index array
        sidx = np.arange(0, end)

        # Sample the dataframes (no need for dropna since already cleaned)
        df_sampled     = imbh_df.iloc[sidx].copy()
        system_sampled = system_df.iloc[sidx].copy()
        
        return df_sampled, system_sampled, sidx

    # Create the output of the simulation in point-to-point format 
    def make_features(df_sampled, system_sampled, iconds_dict, noise, environment, config):

        # Compute cluster features
        cluster_feats = compute_cluster_features(system_sampled, df_sampled, iconds_dict, noise, 
                                                 sim_env        = environment,
                                                 temp_evol      = config.requires_temp_evol)
        
        # Get temporal values
        time_evol = time_preparation(time_evolution = df_sampled["time[Myr]"])
        m_evol    = target_preparation(mass_evolution = df_sampled["massNew[Msun](10)"])
        
        # Include noise if needed
        t = apply_noise(time_evol) if noise else time_evol
        m = apply_noise(m_evol) if noise else m_evol

        # Convert categorical variable to numeric codes
        fchannel_codes = {"FAST": 0, "SLOW": 1, "STEADY": 2}
        type_sim_code  = fchannel_codes.get(cluster_feats["type_sim"], np.nan)
       
         # Build features array
        feature_columns = [t]
       
        # Define numeric features names in the expected order
        expected_order = ["tcoll", "trelax", "tcc", "tcross", "m_tot", "m_mean", "m_max", "mcrit", "safnum",
                          "rho_half", "stellar_radius", "rh", "cr",
                          "z", "fracbin"]
        # Handle temporal vs static features
        for k in expected_order:
            feat_val = cluster_feats.get(k, np.nan)
            if config.requires_temp_evol and isinstance(feat_val, np.ndarray):
                # Use time-evolving values
                feature_columns.append(feat_val)
            else:
                # Repeat scalar value for all timesteps
                feature_columns.append(np.full(len(t), feat_val))
        
        # Add categorical feature (always repeated)
        feature_columns.append(np.full(len(t), type_sim_code))
        
        sim_features = np.column_stack(feature_columns)
        
        return sim_features, m

    # List to store the information
    feats_all, targs_all, idxs_all = [], [], []

    # Determine number of iterations based on augmentation setting
    iterations = n_virtual if augment else 1
    
    for _ in range(iterations):
        # Use sampling window only if augmentation is enabled, otherwise use full dataset
        if augment:
            df_sampled, system_sampled, sidx = sample_window()
        else:
            df_sampled, system_sampled, sidx = imbh_df.copy(), system_df.copy(), np.arange(n_points)
        
        # Retrieve features and targets
        feats, targs = make_features(df_sampled, system_sampled, meta_dict, noise, environment, config)
        
        # Append all
        feats_all.append(feats)
        targs_all.append(targs)
        idxs_all.append(sidx)

    return np.vstack(feats_all), np.concatenate(targs_all), np.concatenate(idxs_all)

# Load MOCCASURVEY simulations files and retrieve information for posterior processing ------------------------------------#
def load_moccasurvey_imbh_history(file_path: str, init_conds_sim: bool= False, col_description: bool= False, 
                                  stellar_map    : bool= False, 
                                  init_conds_evo : bool= False,
                                  verbose        : bool= False):
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
        if col_description: col_dict = load_json_file( "./rawdata/moccasurvey/mapping_dicts/imbh_history.json", 
                                                        "description for imbh-history.dat")
        if stellar_map: stellar_dict = load_json_file( "./rawdata/moccasurvey/mapping_dicts/stellar_types.json",
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
        sim_df = pd.read_csv(imbh_path,
                            skiprows  = 87,
                            header    = None,
                            names     = columns,
                            sep       = r'\s+', 
                            index_col = False,
                            engine    = 'python')
        if verbose: print(f"Loaded {len(sim_df)} rows of simulation data")

        # Load system.dat file if requested
        if init_conds_evo: system_df, system_dict = __load_moccasurvey_system_data(system_path)

    except Exception as e:
        print(f"Error: {e}")
        raise

    return [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]

# Retrieve a partition of input / target values for a ML-Experiment -------------------------------------------------------#
def moccasurvey_dataset(simulations_path: List[str], experiment_config: MoccaSurveyExperimentConfig = def_config,
                        simulations_type : Optional[List[str]] = None,
                        augmentation     : bool = False, 
                        logger           : Optional[Logger] = None, 
                        test_partition   : bool = False,
                        noise            : bool = True,
                        points_per_sim   : Optional[Union[int, float]] = None, 
                        n_virtual        : int = 1,
                        downsampled      : bool = False):
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment, supporting different experiment types.
    ________________________________________________________________________________________________________________________
    Parameters:
        simulations_path (list)             : List of simulation paths.
        augmentation     (bool)             : Whether to augment data based on the time evolution of the initial conditions.
        logger           (Optional[Logger]) : If given, print relevant comments in the console.
        test_partition   (bool)             : If True, store simulation paths for test data tracking.
        noise            (bool)             : Implement gaussian noise to data (not included for test partition)
        points_per_sim   (Union[int, float]) : Number of points to sample per simulation. Only used when augmentation=True.
                                               If int, uses fixed number of points. If float (between 0 and 1), uses 
                                               proportional sampling (size = len(imbh_df) * points_per_sim).
        n_virtual        (int)              : Number of virtual simulations to sample per real simulation if augmentation is 
                                              enabled.
        downsampled      (bool)             : If perform a downsampled of the dataset by accumulation points in a 2D 
                                              histogram
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple: ([features, feature_names], [targets, target_names]) or 
               ([features, feature_names], [targets, target_names], simulation_paths) if test_partition=True
    ________________________________________________________________________________________________________________________
    """
    # Retrieve configuration and prelocated values
    features, targets, simulation_paths = [], [], [] if test_partition else None
    ignored, processed = 0, 0

    # Run elements per simulation
    for idx, path in enumerate(simulations_path):

        try:
            imbh, system     = load_moccasurvey_imbh_history(f"{path}/", init_conds_sim=True, init_conds_evo=True)
            imbh_df          = imbh[0].drop_duplicates("time[Myr]").sort_values("time[Myr]")
            iconds_imbh_dict = imbh[1]
            system_df        = system[0].sort_values("tphys")
            env_type         = simulations_type[idx] if simulations_type else None

            # Use merge_asof to align system_df to imbh_df times
            matched_system_df = pd.merge_asof(imbh_df[["time[Myr]"]], system_df,
                                              left_on   = "time[Myr]",
                                              right_on  = "tphys",
                                              direction = "nearest")
            
            # Ignore short simulations and raise warning
            if len(imbh_df) <= def_config.min_sim_points:
                ignored += 1
                if logger: logger.warning(f"Ignored {path} (too few points: {len(imbh_df)})")
                continue

            processed += 1

            # Retrieve single simulation data
            feats, targs, idxs = process_single_simulation(imbh_df=imbh_df, system_df=matched_system_df, meta_dict= iconds_imbh_dict,
                                                            config         = experiment_config, 
                                                            points_per_sim = points_per_sim, 
                                                            environment    = env_type,
                                                            augment        = augmentation and not test_partition,
                                                            noise          = noise and not test_partition,
                                                            n_virtual      = n_virtual
                                                            )

            features.append(feats)
            targets.append(targs)
            
            if test_partition: simulation_paths.extend([path] * len(idxs))

        except Exception as e:
            ignored += 1
            if logger: logger.warning(f"Error in '{path}': {e}\n{traceback.format_exc()}")
            continue
    
    # Stack the features and the target given the selected features
    X = np.vstack(features) if features else np.empty((0, len(experiment_config.feature_names)))
    y = np.concatenate(targets) if targets else np.empty((0,))

    # Perform downsampled if desired:
    if downsampled: 
        # Initialize empty lists to collect results
        X_parts, y_parts = [], []
        
        # Process each category with safety checks
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW'), (2, 'STEADY')]:
            mask = X[:, -1] == channel_code
            if np.any(mask):
                X_channel, y_channel = safe_downsampling_of_points(X[mask], y[mask], logger)
                if X_channel is not None and len(X_channel) > 0:
                    X_parts.append(X_channel)
                    y_parts.append(y_channel)
                else:
                    if logger: logger.warning(f"No data after downsampling for {channel_name} channel")
            else:
                if logger: logger.warning(f"No data found for {channel_name} channel")
        
        # Only concatenate if we have results
        if X_parts:
            X = np.concatenate(X_parts)
            y = np.concatenate(y_parts)
        else:
            if logger: logger.error("No data remaining after downsampling all channels")

    if logger:
        logger.info(f"Processed {processed}, Ignored {ignored}, Shape: {X.shape}, Targets: {experiment_config.target_name}")

    if test_partition:
        
        if downsampled and logger:
            logger.warning("Downsampling with test_partition=True shouldn't be used. Review the configuration")

        return [X, experiment_config.feature_names], [y, experiment_config.target_name], simulation_paths

    return [X, experiment_config.feature_names], [y, experiment_config.target_name]

#--------------------------------------------------------------------------------------------------------------------------#