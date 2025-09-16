
# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json
import traceback

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Tuple, Optional, List, Dict, Union
from loguru._logger import Logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory       import __load_json_file
from src.utils.phyfactors      import critical_mass, relaxation_time, core_collapse_time,  collision_time, rho_at_rh
from src.processing.format     import time_preparation, target_preparation, __get_target_name, filter_and_downsample_hist2d

# Default configuration of the experiments --------------------------------------------------------------------------------#
DEFAULT_CONFIG = {
    "point_mass": {                                       # Experiment name
        "feature_names": [                                # - Features names to retrieve (IN ORDER)
            "t", "t_coll", "t_relax", "t_cc",             #     - Time-relared 
            "M_tot", "M_mean", "M_max", "M_crit",         #     - Mass-related 
            "rho(R_h)",                                   #     - Density
            "R_h", "R_core",                              #     - Radii
            "type_sim"],                                  #     - Categorical environment of the simulation                 
        
        "base_target":                                    # - Target name of the experiment
            "M_MMO",
        
        "requires_time_diff": False                       # - Flag to difference between steptimes
    },
    
    "delta_mass": {
        "feature_names": [
            "t", "delta_t", "t_coll", "t_relax", "t_cc", 
            "M_tot", "M_mean", "M_max", "M_crit", 
            "rho(R_h)", 
            "R_h", "R_core"],

        "base_target": 
            "delta_M_MMO",
        
        "requires_time_diff": True
    },
}

# HELPER FUNCTIONS --------------------------------------------------------------------------------------------------------#
def __parse_moccasurvey_imbh_history_init_conds(file_path: str):
    """Helper function that ñoad initial conditions from a single IMBH-HISTORY.DAT file into a dictionary, with respective 
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
    """Helper function that load the evolution of the initial conditions of the sistem from a single MOCCA-Survey 
    simulation, load column description into a dictionary"""
    
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
def __apply_noise(value: Union[float, np.ndarray, List[float]], sigma: float = 0.01) -> Union[float, np.ndarray]:
    """Helper function that creates a gaussian noise for a single tabular value or array of values (Default to 1sigma)"""
    value_array  = np.array(value)
    noised_array = value_array + np.random.normal(loc=0, scale=value_array * sigma)
    return np.maximum(noised_array, 0)

#--------------------------------------------------------------------------------------------------------------------------#
def compute_cluster_features(system_df: pd.DataFrame, imbh_df: pd.DataFrame, tau: float, apply_noise: bool = False,
                             sim_env: Optional[str] = None
                            ) -> Dict[str, Union[float, str]]:
    """Helper function that creates the cluster of initial conditions and physical values required from moccasurvey"""
    
    def get_value(value, apply_noise):
        return __apply_noise(value) if apply_noise else value
    
    # Initial conditions of interest
    rh     = get_value(system_df["r_h"].iloc[0], apply_noise)  
    v_disp = get_value(system_df["vc"].iloc[0], apply_noise)
    m_tot  = get_value(system_df["smt"].iloc[0], apply_noise)
    n      = get_value(system_df["nt"].iloc[0], apply_noise)
    m_mean = get_value(system_df["atot"].iloc[0], apply_noise)
    m_max  = get_value(system_df["smsm"].iloc[0], apply_noise)
    cr     = get_value(system_df["rc"].iloc[0], apply_noise)

    # Type of the environment of the simulation (repeat if the simulation its already classified)
    if (sim_env is not None):  env = sim_env
    
    # Determine the formation channel based on initial core density and time of IMBH formation if first seen
    else: env, _ = determine_formation_channel(system_df, imbh_df, None)

    output = dict(rh=rh, v_disp=v_disp, m_tot=m_tot, n=n, m_mean=m_mean, m_max=m_max, cr=cr, type_sim=env,
            mcrit    = critical_mass(hm_radius=rh, mass_per_star=m_mean, cluster_age=tau, v_disp=v_disp).value,
            trelax   = relaxation_time(n_stars=n, hm_radius=rh, v_disp=v_disp).value,
            tcc      = core_collapse_time(m_mean=m_mean, m_max=m_max, n_stars=n, hm_radius=rh, v_disp=v_disp).value,
            tcoll    = collision_time(hm_radius=rh, n_stars=n, mass_per_star=m_mean, v_disp=v_disp).value,
            rho_half = rho_at_rh(n_stars=n, hm_radius=rh).value
            )
        
    return output

#--------------------------------------------------------------------------------------------------------------------------#
def __prepare_timeseries_inputs(time_series: pd.Series, norm_factor: Optional[float], 
                                return_diff: bool
                                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Helper function to unify the time evolution output of a single similation for a given experiment"""
    
    time_evol = time_preparation(time_series, norm_factor, return_diff=False)
    t_diff = None
    if return_diff:
        _, t_diff = time_preparation(time_series, norm_factor, return_diff=True)
    
    return time_evol, t_diff

#--------------------------------------------------------------------------------------------------------------------------#
def determine_formation_channel(system_df, imbh_df, n_virtual: Optional[int]=None):
    """Helper function to classify a simulation given the time of IMBH formation and core density"""
    i_density = system_df["roc"].iloc[0]
    mass_time = imbh_df[imbh_df['massNew[Msun](10)'] > 100].iloc[0]['time[Myr]']
    
    if (i_density >= 1e7) and (mass_time <= 50):
        chform  = "FAST" 
        virsims = n_virtual 
    
    elif (i_density <= 1e6) and (mass_time >= 500):
        chform = "SLOW" 
        virsims = n_virtual 
    
    else:
        chform = "HYBRID"
        virsims = n_virtual 
    
    return chform, virsims

#--------------------------------------------------------------------------------------------------------------------------#
def __safe_downsamapling_of_points(feats, mass, logger):
    # Safety check: ensure we have data to downsample
    if len(feats) == 0:
        if logger: logger.warning("No data available for downsampling")
    else:
        time_feat = np.log10(feats[:,0] + 1)
        mass_feat = mass 
        
        # Remove any inf/nan values that might have been created
        valid_mask = np.isfinite(time_feat) & np.isfinite(mass_feat)
        if not np.all(valid_mask):
            if logger: logger.warning(f"Removing {np.sum(~valid_mask)} points with inf/nan values")
            time_feat  = time_feat[valid_mask]
            mass_feat  = mass_feat[valid_mask]
            feats_temp = feats[valid_mask]
            mass_temp  = mass[valid_mask]
        else:
            feats_temp = feats
            mass_temp  = mass
        
        # Only proceed if we still have data
        if len(time_feat) > 0:

            H1, xedges, yedges = np.histogram2d(time_feat, mass_feat, bins=[200, 200])
            idxs = filter_and_downsample_hist2d(time_feat, mass_feat, H1, xedges, yedges, 
                                                min_count = 10, 
                                                max_count = 150,
                                                seed      = 42)   
            
            if len(idxs) > 0:
                return feats_temp[idxs], mass_temp[idxs]
            
            else:
                if logger: logger.error("Downsampling resulted in empty dataset")
        else:
            if logger: logger.error("No valid data points remaining after filtering inf/nan values")


#--------------------------------------------------------------------------------------------------------------------------#
def process_single_simulation(imbh_df: pd.DataFrame, system_df: pd.DataFrame, config: Dict, experiment_type: str,
                               norm_target      : bool, 
                               log10_target     : bool,
                               time_norm_factor : Optional[float], 
                               points_per_sim   : Union[int, float], 
                               environment      : Optional[str] = None,
                               augment          : bool = False,
                               apply_noise      : bool = False, 
                               n_virtual        : int  = 1):
    """Helper function to preprocess one simulation elements from moccasurvey dataset"""

    # Validate the number of input points and determine sampling size
    n_points = len(imbh_df)
    
    # If points_per_sim is an interger, should be greater than the number of points in the simulation
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

    # This creates a sampled of points from a given number of elements
    def sample_window(max_start_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:

        # restrict start index to the early portion of the sequence
        max_start = int(n_points * max_start_frac)
        start     = np.random.randint(0, max_start)

        # choose an end point randomly within the allowed window
        end = np.random.randint(sample_size , n_points)
 
        # Create index array
        sidx = np.arange(start, end)

        # Sample the dataframes
        df_sampled     = imbh_df.iloc[sidx].copy()
        system_sampled = system_df.iloc[sidx].copy()
        
        return df_sampled, system_sampled, sidx

    # Create the output of the simulation in point-to-point format 
    def make_features(df_sampled, system_sampled, apply_noise, environment):
        # Separate numeric and categorical features
        numeric_feats_names   = ["tcoll", "trelax", "tcc", "m_tot", "m_mean", "m_max", "mcrit", "rho_half", "rh", "cr"]
        categorical_feat_name = "type_sim"

        tau  = df_sampled["time[Myr]"].max() - df_sampled["time[Myr]"].min()
        cluster_feats = compute_cluster_features(system_sampled, df_sampled, tau, apply_noise, sim_env=environment)
        
        # Get temporal values
        time_evol, t_diff = __prepare_timeseries_inputs(time_series = df_sampled["time[Myr]"], 
                                                        norm_factor = time_norm_factor, 
                                                        return_diff = config["requires_time_diff"])
        m_evol = target_preparation(mass_evolution= df_sampled["massNew[Msun](10)"], 
                                    time_evolution= df_sampled["time[Myr]"], 
                                    norm_factor = cluster_feats["m_tot"] if norm_target else None,
                                    target_type = experiment_type, 
                                    log10_scale = log10_target)
        
        # Include noise if needed
        t = __apply_noise(time_evol) if apply_noise else time_evol
        m = __apply_noise(m_evol) if apply_noise else m_evol

        # Convert categorical variable to numeric codes
        fchannel_codes = {"FAST": 0, "SLOW": 1, "HYBRID": 2}
        type_sim_code  = fchannel_codes.get(cluster_feats[categorical_feat_name], np.nan)
        if t_diff is not None:
            sim_features = np.column_stack([
                t,
                t_diff, 
                *[np.full(len(t), cluster_feats[k]) for k in numeric_feats_names],
                np.full(len(t), type_sim_code)])
        
        else: 
            sim_features = np.column_stack([
                t,
                *[np.full(len(t), cluster_feats[k]) for k in numeric_feats_names],
                np.full(len(t), type_sim_code)])
        
        return sim_features, m

    # List to store the information
    feats_all, targs_all, idxs_all = [], [], []

    iterations = n_virtual if augment else 1
    for _ in range(iterations):
        df_sampled, system_sampled, sidx = sample_window() if augment else (imbh_df.copy(), system_df.copy(), np.arange(n_points))
        feats, targs = make_features(df_sampled, system_sampled, apply_noise, environment)
        feats_all.append(feats)
        targs_all.append(targs)
        idxs_all.append(sidx)

    return np.vstack(feats_all), np.concatenate(targs_all), np.concatenate(idxs_all)

# END OF HELPER FUNCTIONS -------------------------------------------------------------------------------------------------#

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
        # Load optional JSON mapping files --------------------------------------------------------------------------------#
        if col_description: col_dict = __load_json_file( "./rawdata/moccasurvey/mapping_dicts/imbh_history.json", 
                                                        "description for imbh-history.dat")
        if stellar_map: stellar_dict = __load_json_file( "./rawdata/moccasurvey/mapping_dicts/stellar_types.json",
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
def moccasurvey_dataset(simulations_path: List[str], experiment_type: str, experiment_config: Dict = DEFAULT_CONFIG,
                         simulations_type : Optional[List[str]] = None,
                         augmentation     : bool = False, 
                         norm_target      : bool = False, 
                         log10_target     : bool = False,
                         logger           : Optional[Logger] = None, 
                         test_partition   : bool = False,
                         time_norm_factor : Optional[float] = None, 
                         time_return_diff : bool = False,
                         apply_noise      : bool = True,
                         points_per_sim   : Union[int, float] = 500, 
                         n_virtual        : int = 1,
                         downsampled      : bool = False):
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment, supporting different experiment types.
    ________________________________________________________________________________________________________________________
    Parameters:
        simulations_path (list)             : List of simulation paths.
        experiment_type  (str)              : Type of experiment ("point_mass", "delta_mass", "mass_rate").
        augmentation     (bool)             : Whether to augment data based on the time evolution of the initial conditions.
        norm_target      (bool)             : Whether to normalize the target by some initial condition.
        log10_target     (bool)             : Whether to use a logarithmic scale.
        logger           (Optional[Logger]) : If given, print relevant comments in the console.
        test_partition   (bool)             : If True, store simulation paths for test data tracking.
        time_norm_factor (Optional[float])  : Normalization factor for time_preparation.
        time_return_diff (bool)             : Whether to return time differences in time_preparation.
        apply_noise      (bool)             : Implement gaussian noise to data (not included for test partition)
        points_per_sim   (Union[int, float]) : Number of points to sample per simulation. If int, uses fixed number of points.
                                               If float (between 0 and 1), uses proportional sampling 
                                               (size = len(imbh_df) * points_per_sim).
        n_virtual        (int)              : Number of virtual simulations to sample per real simulation if augmentation is 
                                              enabled.
        downsampled      (bool)             : If performa a downsampled of the dataset by accumulation points in a 2D 
                                              histogram
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple: ([features, feature_names], [targets, target_names]) or 
               ([features, feature_names], [targets, target_names], simulation_paths) if test_partition=True
    ________________________________________________________________________________________________________________________
    """
    # Experiment validation -----------------------------------------------------------------------------------------------#
    if experiment_type not in experiment_config:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'")
    
    # Retrieve configuration and prelocated values
    config = experiment_config[experiment_type]
    features, targets, simulation_paths = [], [], [] if test_partition else None
    ignored, processed = 0, 0

    # Run elements per simulation
    for idx, path in enumerate(simulations_path):

        try:
            imbh, system = load_moccasurvey_imbh_history(f"{path}/", init_conds_sim=False, init_conds_evo=True)
            imbh_df   = imbh[0].drop_duplicates("time[Myr]").sort_values("time[Myr]")
            system_df = system[0].sort_values("tphys")
            env_type  = simulations_type[idx] if simulations_type else None

            # Use merge_asof to align system_df to imbh_df times
            matched_system_df = pd.merge_asof(imbh_df[["time[Myr]"]], system_df,
                                              left_on   = "time[Myr]",
                                              right_on  = "tphys",
                                              direction = "nearest")
            
            # Ignore short simulations and raise warning
            if len(imbh_df) <= 1000:
                ignored += 1
                if logger: logger.warning(f"Ignored {path} (too few points: {len(imbh_df)})")
                continue

            processed += 1

             # Define categorical environment of the simulation
            _, virsims = determine_formation_channel(matched_system_df, imbh_df, n_virtual)

            # Retrieve single simulation data
            feats, targs, idxs = process_single_simulation(imbh_df=imbh_df, system_df=matched_system_df, config=config, 
                                                            experiment_type  = experiment_type,
                                                            norm_target      = norm_target, 
                                                            log10_target     = log10_target, 
                                                            time_norm_factor = time_norm_factor,
                                                            points_per_sim   = points_per_sim, 
                                                            environment      = env_type,
                                                            augment          = augmentation and not test_partition,
                                                            apply_noise      = apply_noise and not test_partition,
                                                            n_virtual        = virsims
                                                            )

            features.append(feats)
            targets.append(targs)
            
            if test_partition: simulation_paths.extend([path] * len(idxs))

        except Exception as e:
            ignored += 1
            if logger: logger.warning(f"Error in '{path}': {e}\n{traceback.format_exc()}")
            continue
    
    # Stack the features and the target given the selected features
    X = np.vstack(features) if features else np.empty((0, len(config["feature_names"])))
    y = np.concatenate(targets) if targets else np.empty((0,))
    target_names = __get_target_name(config["base_target"], norm_target, log10_target)

    # Perform downsampled if desired:
    if downsampled: 
        # Initialize empty lists to collect results
        X_parts, y_parts = [], []
        
        # Process each category with safety checks
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW'), (2, 'HYBRID')]:
            mask = X[:, 11] == channel_code
            if np.any(mask):
                X_channel, y_channel = __safe_downsamapling_of_points(X[mask], y[mask], logger)
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
        logger.info(f"Processed {processed}, Ignored {ignored}, Shape: {X.shape}, Targets: {target_names}")

    if test_partition:
        
        if downsampled and logger:
            logger.warning("Downsampling with test_partition=True shouldn't be used. Review the configuration")
        
        return [X, config["feature_names"]], [y, target_names], simulation_paths
    
    return [X, config["feature_names"]], [y, target_names]

#--------------------------------------------------------------------------------------------------------------------------#