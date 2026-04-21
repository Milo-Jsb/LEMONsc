# Modules -----------------------------------------------------------------------------------------------------------------#
import os 
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing      import List, Optional, Tuple, Union, Dict, Any

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory                        import load_json_file
from src.processing.constructors.utils._config  import YMCsNBODYSims, YMCsMOCCASims
from src.processing.constructors.utils._formats import _cols_from_mapping, _mocca_imbh_history_init_conds_header_columns, _load_mocca_system_data

# Set configuration -------------------------------------------------------------------------------------------------------#
default_mocca = YMCsMOCCASims()
default_nbody = YMCsNBODYSims()

# [Helper] Load simulation files from MOCCA-based YMCs simulation ---------------------------------------------------------#
def _load_ymcs_mocca_imbh_history(file_path: str, init_conds_sim: bool = False, col_description: bool = False, 
                                  stellar_map    : bool            = False, 
                                  init_conds_evo : bool            = False,
                                  verbose        : bool            = False,
                                  config         : YMCsMOCCASims   = default_mocca
                                  ) -> Tuple[List[Optional[Union[pd.DataFrame, Dict[str, Any]]]], 
                                             List[Optional[Union[pd.DataFrame, Dict[str, Any]]]]]:
    """
    ________________________________________________________________________________________________________________________
    Load YMCs MOCCA IMBH history data from simulation files (imbh-history.dat).
    ________________________________________________________________________________________________________________________
    Parameters:
    -> file_path       (str)          : Mandatory. Path to the simulation folder (must end with '/').
    -> init_conds_sim  (bool)         : Optional. Extract initial conditions from '#' header lines.
    -> col_description (bool)         : Optional. Load column descriptions from mapping file.
    -> stellar_map     (bool)         : Optional. Load stellar type mappings from mapping file.
    -> init_conds_evo  (bool)         : Optional. Load evolution of init conds (system.dat).
    -> verbose         (bool)         : Optional. Enable verbose logging.
    -> config          (YMCsMOCCASims): Optional. Configuration with directory paths.
    ________________________________________________________________________________________________________________________
    Returns:
    -> [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]
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
        if col_description: col_dict = load_json_file(f"{config.mapping_dics_dir}imbh_history_columns.json", 
                                                       "description for imbh-history.dat")
        if stellar_map: stellar_dict = load_json_file(f"{config.mapping_dics_dir}stellar_types.json",
                                                       "stellar type mapping dictionary")
            
        # Parse header: variable-length init conditions + column names from last '#' line ---------------------------------#
        if init_conds_sim:
            init_conds, columns = _mocca_imbh_history_init_conds_header_columns(imbh_path)
        else:
            _, columns = _mocca_imbh_history_init_conds_header_columns(imbh_path)
        
        # Load data: comment='#' skips the header and the repeated column-header lines in the body ------------------------#
        if verbose: print("Loading main simulation data with pandas...")
        sim_df = pd.read_csv(imbh_path, comment='#', header=None, names=columns, sep=r'\s+', index_col=False,
                             engine='python')
        if verbose: print(f"Loaded {len(sim_df)} rows of simulation data")

        # Load system.dat if requested
        if init_conds_evo: system_df, system_dict = _load_mocca_system_data(system_path, verbose, config)

    except Exception as e:
        print(f"Error: {e}")
        raise

    return [sim_df, init_conds, col_dict, stellar_dict], [system_df, system_dict]










































# [Helper] Create imbh_df from ymcs nbody file ----------------------------------------------------------------------------#
def _create_imbh_df_from_ymcs_nbody(vms_df: pd.DataFrame, time_col: str, mass_col: str, 
                                    nb_to_myr       : float         = 8.559778667345e-04,
                                    rename_mass_col : Optional[str] = None, 
                                    rename_time_col : Optional[str] = None
                                    ) -> pd.DataFrame:
    """This function takes the direct output of the nbody files and maps them into a dataframe following the repo format"""
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(vms_df, pd.DataFrame):
        raise TypeError("vms_df must be a pandas DataFrame")
    if time_col not in vms_df.columns:
        raise KeyError(f"time column '{time_col}' not found in DataFrame")
    if mass_col not in vms_df.columns:
        raise KeyError(f"mass column '{mass_col}' not found in DataFrame")

    # Copy only the relevant columns --------------------------------------------------------------------------------------#
    imbh_df = vms_df[[time_col, mass_col]].copy()
    
    # Rescale time column in-place on the original column name (NB units to Myr) -----------------------------------------#
    if nb_to_myr is not None:
        imbh_df[time_col] = imbh_df[time_col] * nb_to_myr
    
    # Rename columns if requested (single rename pass avoids duplicate column names) --------------------------------------#
    rename_map = {k: v for k, v in {time_col: rename_time_col, mass_col: rename_mass_col}.items() if v}
    if rename_map:
        imbh_df = imbh_df.rename(columns=rename_map)

    out_time_col = rename_time_col if rename_time_col else time_col
    out_mass_col = rename_mass_col if rename_mass_col else mass_col

    return imbh_df[[out_time_col, out_mass_col]]

# [Helper] Create system_df from ymcs nbody file --------------------------------------------------------------------------#
def _create_system_df_from_ymcs_nbody():
    """This function takes the direct output of the nbody files and maps them into a dataframe following the repo format"""

    pass

# [Helper] Load YMCs NBODY simulations files and retrieve information for posterior processing ----------------------------#
def _load_ymcs_nbody_imbh_history(file_path: str, vms_sim: bool, add_sim_files: bool, init_conds_sim: bool= False, 
                                  col_description: bool          = False,  
                                  verbose        : bool          = False,
                                  config         : YMCsNBODYSims = default_nbody
                                  )-> Tuple[List[Optional[Union[pd.DataFrame, Dict[str, Any]]]], 
                                            List[Optional[Union[pd.DataFrame, Dict[str, Any]]]]]:
    """
    ________________________________________________________________________________________________________________________
    Load Young Massive Clusters NBODY IMBH history data from simulation files.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> file_path        (str)                  : Mandatory. Path to the simulation folder (must end with '/').
    -> vms_sim          (bool)                 : Mandatory. Whether the simulation contains a VMS/IMBH 
                                                 (vms.dat).
    -> add_sim_files    (bool)                 : Mandatory. Whether to load additional simulation files
                                                 (pos_vel.dat, coll.dat, avmass.dat, esc.dat, pos_vel.dat, rlagr.dat).
    -> init_conds_sim   (bool)                 : Optional. Whether to load initial conditions from iconds.dat.
    -> col_description  (bool)                 : Optional. Whether to load JSON column descriptions into col_dicts.
    -> verbose          (bool)                 : Optional. Enable verbose logging.
    -> config           (YMCsExperimentConfig) : Optional. Configuration object with directory paths.
    ________________________________________________________________________________________________________________________
    Returns:
    
    # TODO Standarized the output to contain same output as original moccasurvey function, create helpers to create 
           necesary files
    
    
    -> [sim_df, init_conds], [system_dicts, col_dicts]

        sim_df       (pd.DataFrame or None) : VMS/IMBH mass/radius/age evolution (vms.dat), or None if vms_sim=False.
        init_conds   (dict or None)         : Initial cluster conditions from iconds.dat (if init_conds_sim=True).

        system_dicts (dict)                 : Additional DataFrames keyed by name when add_sim_files=True:
                                              "coll"    -> collision history (coll.dat)
                                              "avmass"  -> average Lagrangian masses (avmass.dat)
                                              "esc"     -> escaper events (esc.dat)
                                              "pos_vel" -> VMS/IMBH position & velocity (pos_vel.dat)
                                              "rlagr"   -> Lagrangian radii (rlagr.dat)
        col_dicts    (dict)                 : JSON column descriptions keyed by name (if col_description=True).
    ________________________________________________________________________________________________________________________
    Raises:
    -> TypeError, FileNotFoundError, JSONDecodeError
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Folder not found: {file_path}")

    imbh_path   = f"{file_path}vms.dat"
    coll_path   = f"{file_path}coll.dat"
    avmass_path = f"{file_path}avmass.dat"
    escps_path  = f"{file_path}esc.dat"
    posvel_path = f"{file_path}pos_vel.dat"
    rlagr_path  = f"{file_path}rlagr.dat"
    iconds_path = f"{file_path}iconds.dat"

    # Initialize return values --------------------------------------------------------------------------------------------#
    sim_df       = None 
    init_conds   = None
    col_dicts    = {}
    system_dicts = {}
    
    if verbose: print(f"Loading NBODY simulation from: {file_path}")
    
    # Load files as expected by YMCs nbody format -------------------------------------------------------------------------#
    try:
        # Load optional JSON mapping files (assuming correct placement of files) ------------------------------------------#
        if col_description: 
            col_dicts = {
                "imbh_cols"    : load_json_file(f"{config.mapping_dics_dir}vms_columns.json",
                                                "description for vms.dat"),
                "coll_cols"    : load_json_file(f"{config.mapping_dics_dir}coll_columns.json",
                                                "description for coll.dat"),
                "avmass_cols"  : load_json_file(f"{config.mapping_dics_dir}avmass_columns.json",
                                                "description for avmass.dat"),
                "esc_cols"     : load_json_file(f"{config.mapping_dics_dir}esc_columns.json",
                                                "description for esc.dat"),
                "pos_vel_cols" : load_json_file(f"{config.mapping_dics_dir}pos_vel_columns.json",
                                                "description for pos_vel.dat"),
                "rlagr_cols"   : load_json_file(f"{config.mapping_dics_dir}rlagr_columns.json",
                                                "description for rlagr.dat")
                        }
            if init_conds_sim:
                col_dicts["iconds_cols"] = load_json_file(f"{config.mapping_dics_dir}iconds_columns.json",
                                                          "description for iconds.dat")

        # Load main VMS/IMBH simulation data (vms.dat has no header rows) -------------------------------------------------#
        if vms_sim and os.path.exists(imbh_path):
            
            # Verbose
            if verbose: print(f"Loading VMS/IMBH history from: {imbh_path}")
            
            # Mapping of columns, get from previous variable if exist, else load from JSON file
            vms_mapping = (col_dicts.get("imbh_cols") or load_json_file(f"{config.mapping_dics_dir}vms_columns.json", 
                                                                        "col mapping for vms.dat"))
            
            # Get column names and load the data in pandas
            columns = _cols_from_mapping(vms_mapping)
            sim_df  = pd.read_csv(imbh_path, skiprows=0, header=None, names=columns, sep=r'\s+', index_col=False, 
                                  engine='python')
            # Verbose
            if verbose: print(f"Loaded {len(sim_df)} rows of VMS/IMBH data")

        # Load initial conditions if requested (iconds.dat is a single-row whitespace-separated file) ---------------------#
        if init_conds_sim and os.path.exists(iconds_path):
            
            # Verbose
            if verbose: print(f"Loading initial conditions from: {iconds_path}")
            
            # Mapping of columns, get from previous variable if exist, else load from JSON file
            iconds_mapping = (col_dicts.get("iconds_cols") or
                              load_json_file(f"{config.mapping_dics_dir}iconds.json", "col mapping for iconds.dat"))
            
            # Get column names and load the data as a dictionary
            iconds_cols = _cols_from_mapping(iconds_mapping)
            with open(iconds_path, 'r') as f:
                values = f.readline().strip().split()
            init_conds = {col: float(val) for col, val in zip(iconds_cols, values)}

        # Load additional simulation files if requested -------------------------------------------------------------------#
        if add_sim_files:
            if verbose: print("Loading additional simulation files...")

            # coll.dat – collision events
            if os.path.exists(coll_path):
                coll_mapping = (col_dicts.get("coll_cols") or
                                load_json_file(f"{config.mapping_dics_dir}coll_columns.json", "col mapping for coll.dat"))
                
                system_dicts["coll"] = pd.read_csv(coll_path, header=None, names= _cols_from_mapping(coll_mapping), 
                                                   sep       = r'\s+', 
                                                   index_col = False, 
                                                   engine    = 'python')
                
                if verbose: print(f"Loaded {len(system_dicts['coll'])} collision events")

            # avmass.dat – average stellar mass inside each Lagrangian shell
            if os.path.exists(avmass_path):
                avmass_mapping = (col_dicts.get("avmass_cols") or
                                  load_json_file(f"{config.mapping_dics_dir}avmass_columns.json", "col mapping for avmass.dat"))
                
                system_dicts["avmass"] = pd.read_csv(avmass_path, header=None, names= _cols_from_mapping(avmass_mapping),
                                                     sep       = r'\s+', 
                                                     index_col = False, 
                                                     engine    = 'python')
                
                if verbose: print(f"Loaded {len(system_dicts['avmass'])} avmass rows")

            # esc.dat – escaping stars
            if os.path.exists(escps_path):
                esc_mapping = (col_dicts.get("esc_cols") or
                               load_json_file(f"{config.mapping_dics_dir}esc_columns.json", "col mapping for esc.dat"))
                
                system_dicts["esc"] = pd.read_csv(escps_path, header=None, names= _cols_from_mapping(esc_mapping),
                                                  sep       = r'\s+', 
                                                  index_col = False, 
                                                  engine    = 'python')
                
                if verbose: print(f"Loaded {len(system_dicts['esc'])} escaper events")

            # pos_vel.dat – VMS/IMBH position and velocity
            if vms_sim and os.path.exists(posvel_path):
                posvel_mapping = (col_dicts.get("pos_vel_cols") or
                                  load_json_file(f"{config.mapping_dics_dir}pos_vel_columns.json", "col mapping for pos_vel.dat"))
                
                system_dicts["pos_vel"] = pd.read_csv(posvel_path, header=None, names= _cols_from_mapping(posvel_mapping),
                                                      sep       = r'\s+', 
                                                      index_col = False, 
                                                      engine    = 'python')
                if verbose: print(f"Loaded {len(system_dicts['pos_vel'])} pos_vel rows")

            # rlagr.dat – Lagrangian radii
            if os.path.exists(rlagr_path):
                rlagr_mapping = (col_dicts.get("rlagr_cols") or
                                 load_json_file(f"{config.mapping_dics_dir}rlagr_columns.json", "col mapping for rlagr.dat"))
                
                system_dicts["rlagr"] = pd.read_csv(rlagr_path, header=None, names=_cols_from_mapping(rlagr_mapping),
                                                    sep       = r'\s+', 
                                                    index_col = False, 
                                                    engine    = 'python')
                
                if verbose: print(f"Loaded {len(system_dicts['rlagr'])} rlagr rows")

    except Exception as e:
        print(f"Error: {e}")
        raise
    
    # Transform to expected format for posterior processing ---------------------------------------------------------------#
    
    
    
    return [sim_df, init_conds], [system_dicts, col_dicts]

#--------------------------------------------------------------------------------------------------------------------------#