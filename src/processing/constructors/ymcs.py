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
        "rho(R_h)",                                           #    - Density (half mass radius[num])
        "R_*", "R_h", "R_core",                               #    - Radii
        "z",                                                  #    - Metallicity
        "type_sim"]                                           #    - Categorical feature: type of simulation 
                                                              #      (FAST, SLOW)

DEFAULT_EXPECTED_ORDER = [                                    # - Actual column names of the tabular features in the 
        "tcoll", "trelax", "tcc", "tcross",                   #   expected order
        "n", 
        "m_tot", "m_mean", "m_max", "mcrit", 
        "m_loss_tot",
        "safnum",
        "rho_half",
        "stellar_radius", "rh", "cr",,
        "z"] 

@dataclass
class VMSExperimentConfig:
    """
    ________________________________________________________________________________________________________________________
    Base Configuration elements for VMS Dataset preparation and Experiments.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> feature_names        : List of feature names to retrieve (in order)
    -> expected_order       : List of actual column names of the tabular features in the expected order
    ________________________________________________________________________________________________________________________
    """
    feature_names             : List[str] = field(default_factory=lambda: DEFAULT_FEATURE_NAMES.copy())
    expected_order            : List[str] = field(default_factory=lambda: DEFAULT_EXPECTED_ORDER.copy())
    target_name               : str       = "M"                
    min_points_threshold      : int       = 1000                   
    requires_temp_evol        : bool      = False                 
    sample_window             : bool      = True
    mapping_dics_dir_nbody    : str       = "./rawdata/ymcs/V2025a/mapping_dicts/nbody/"
    time_column_nbody_avmass  : str       = "time[NB]"
    time_column_nbody_coll    : str       = "time[NB]"  
    time_column_nbody_esc     : str       = "time[NB]"
    time_column_nbody_pos_vel : str       = "time[NB]"
    time_column_nbody_rlagr   : str       = "time[NB]"
    time_column_nbody_vms     : str       = "time[NB]"   
    mass_column_nbody_vms     : str       = "mass[Msun]"
    
def_config = VMSExperimentConfig()

# Load VMS NBODY Vergara 2025a simulation files and retrieve information for posterior processing -------------------------#
def load_vms_nbody_history(file_path: str, avmass: bool= False, collisions: bool = False, 
                           escapers          : bool                = False, 
                           position_velocity : bool                = False,
                           rlagrange         : bool                = False,
                           config            : VMSExperimentConfig = def_config
                           )-> Tuple[List[Optional[Union[pd.DataFrame, Dict[str, Any]]]], 
                                     List[Optional[Union[pd.DataFrame, Dict[str, Any]]]]]:
    """
    ________________________________________________________________________________________________________________________
    Load VMS NBODY Vergara 2025a history data from simulation files.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> file_path         (str)                 : Mandatory. Path to the VMS NBODY history data file (imbh-history.dat)
    -> avmass            (bool)                : Optional. Load average mass data.
    -> collisions        (bool)                : Optional. Load collision data.
    -> escapers          (bool)                : Optional. Load escaper data.
    -> position_velocity (bool)                : Optional. Load position and velocity data.
    -> rlagrange         (bool)                : Optional. Load rlagrange data.
    -> config            (VMSExperimentConfig) : Optional. Configuration for the VMS experiment.
    ________________________________________________________________________________________________________________________
    Returns:
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


# Retrieve elements from the mocca simulation and compute relevant features -----------------------------------------------#
def compute_vms_nbody_features(system_df: pd.DataFrame, imbh_df: pd.DataFrame, 
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
    ________________________________________________________________________________________________________________________
    Returns:
        - features_dict (Dict[str, Union[float, str]]) : Dictionary with computed cluster features.
    ________________________________________________________________________________________________________________________
    Notes:
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
    else: env = determine_formation_channel(imbh_df, mass_column_name="massNew[Msun](10)", time_column_name="time[Myr]")

   
    return {**base_values, **derived_values, 'type_sim': env}
