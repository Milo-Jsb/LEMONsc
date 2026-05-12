# Modules -----------------------------------------------------------------------------------------------------------------#

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing      import List, Optional
from dataclasses import dataclass, field

# Default features names to retrieve for the moccasurvey dataset ----------------------------------------------------------#
DEFAULT_MOCCASURVEY_FEATURE_NAMES = [                         # - Features names to retrieve (IN ORDER)
        "t",                                                  #    - Time evolution of the simulation 
        "t_coll", "t_relax", "t_cc", "t_cross",               #    - Timescales
        "N",                                                  #    - Number of objects
        "M_tot", "M_mean", "M_max", "M_crit", "M_MMO_0",      #    - Mass-related 
        "safnum",                                             #    - Safranov Number 
        "rho(R_h)", "rho(R_c)",                               #    - Density (half mass radius[num], core radius[mass])
        "R_*", "R_h", "R_core", "R_tid", "Rlg70",             #    - Radii
        "z",                                                  #    - Metallicity
        "fbin",                                               #    - Fraction of binaries of the model
        "type_sim"]                                           #    - Categorical feature: type of simulation 
                                                              #      (FAST, SLOW)

DEFAULT_MOCCASURVEY_EXPECTED_ORDER = [                        # - Actual column names of the tabular features in the 
        "tcoll", "trelax", "tcc", "tcross",                   #   expected order
        "n", 
        "m_tot", "m_mean", "m_max", "mcrit", "m_mmo_0",
        "safnum",
        "rho_half", "rohut",
        "stellar_radius", "rh", "cr", "rtid", "r70%",
        "z", 
        "fbin"] 

# Configuration class for the moccasurvey dataset processing --------------------------------------------------------------#
@dataclass
class MoccaSurveyExperimentConfig:
    """
    ________________________________________________________________________________________________________________________
    Base Configuration elements for MOCCA-Survey Dataset preparation and Experiments.
    ________________________________________________________________________________________________________________________
    Parameters:
    ........................................................................................................................
    -> mapping_dics_dir : Directory path to mapping dictionaries
    ........................................................................................................................
    -> feature_names        : List of feature names to retrieve (in order)
    -> expected_order       : List of actual column names of the tabular features in the expected order
    ........................................................................................................................
    -> target_name          : Target name (mass evolution of the IMBH) to be used for supervised learning
    -> time_column_imbh     : Name of the time column in the IMBH history data
    -> time_column_system   : Name of the time column in the system data
    -> mass_column_imbh     : Name of the mass column in the IMBH history data
    ........................................................................................................................
    -> min_points_threshold    : Minimum number of points per simulation to be used
    -> requires_temp_evol      : Whether some features require time-evolution data
    -> sample_window           : Whether to sample a window of the simulation or use full data
    -> reset_time_window       : Whether to reset time to zero for the sampled window (if sample_window=True)
    -> downsample_max_count    : Maximum number of points to keep per simulation after downsampling
    -> downsample_auto_bins    : Whether to automatically determine the number of bins for downsampling based on the data
    -> histogram_bins          : Number of bins to use if downsample_auto_bins is False
    -> eps_target              : Small epsilon value to add to the target during downsampling to avoid zero values
                                (if downsample_target_scale is "ratio" or "log_ratio")
    -> downsample_target_scale : Scale applied to the target axis during 2D-histogram downsampling.
                                 - "ratio"     : M_MMO / M_tot  (default, linear normalized ratio)
                                 - "identity"  : M_MMO          (raw mass in Msun, no normalization)
                                 - "log_ratio" : log10(M_MMO / M_tot)  (logarithmic normalized ratio)
    ________________________________________________________________________________________________________________________
    """
    # Directory
    mapping_dics_dir : str = "./rawdata/moccasurvey/mapping_dicts/"
    
    # Features names
    feature_names      : List[str] = field(default_factory=lambda: DEFAULT_MOCCASURVEY_FEATURE_NAMES.copy())
    expected_order     : List[str] = field(default_factory=lambda: DEFAULT_MOCCASURVEY_EXPECTED_ORDER.copy())
    
    # Specific name descriptions
    target_name        : str           = "M_MMO"
    time_column_imbh   : str           = "time[Myr]"  
    time_column_system : Optional[str] = "tphys"
    mass_column_imbh   : str           = "massNew[Msun](10)"
    
    # Configuration for data processing                
    min_points_threshold    : int  = 1000                   
    requires_temp_evol      : bool = False                 
    sample_window           : bool = False  
    reset_time_window       : bool = False                 
    
    # Donwsampling specifics
    downsample_max_count    : int   = 100
    downsample_auto_bins    : bool  = True
    downsample_norm_column  : str   = "M_tot"
    downsample_target_scale : str   = "ratio"
    histogram_bins          : int   = 200
    eps_target              : float = 0
    eps_feats               : float = 1
    
# Default configuration for a YMCs experiment -----------------------------------------------------------------------------#
@dataclass
class YMCsExperimentConfig:
    """
    ________________________________________________________________________________________________________________________
    Base Configuration elements for YMCs (Young Massive Clusters) NBODY Dataset preparation and Experiments.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> mapping_dics_dir : Directory path to the NBODY mapping dictionaries (JSON column description files).
    -> pub_dir          : Directory name for publication outputs (e.g., "V2025a/").
    -> code             : Code name for the simulations (Base config not specified, as is set by subclasses).
    -> time_column_imbh : Name of the time column in the transformed IMBH history DataFrame (Myr).
    -> target_name      : Name of the target column in the transformed IMBH history DataFrame (M_MMO).
    ________________________________________________________________________________________________________________________
    """
    # Directory
    mapping_dics_dir  : str           = "./rawdata/ymcs/V2025a/mapping_dicts/"
    pub_dir           : str           = "V2025a/"
    code              : Optional[str] = None
    # Features and targets
    time_column_imbh : str = "time[Myr]"
    target_name      : str = "M_MMO"

# Specific configuration for YMCs Nbody simulations, inheriting from base experiment config -------------------------------#
@dataclass
class YMCsNBODYSims(YMCsExperimentConfig):
    """
    ________________________________________________________________________________________________________________________
    Specific configuration for YMCs NBODY Simulations, inheriting from the base YMCsExperimentConfig.
    ________________________________________________________________________________________________________________________
    Parameters (non-present in base config):
    -> time_column_imbh : Name of the time column in the original NBODY vms.dat file (NB units).
    -> column_mass_vms : Name of the mass column in the original NBODY vms.dat file (Msun).
    -> nbody_to_myrs   : Conversion factor from NB time units to Myr.
    ________________________________________________________________________________________________________________________
    """
    mapping_dics_dir  : str   = field(default="./rawdata/ymcs/V2025a/mapping_dicts/nbody/", init=False)
    code              : str   = field(default="nbody", init=False)      
    time_column_imbh  : str   = "time[NB]"
    column_mass_vms   : str   = "mass[Msun]"
    nbody_to_myrs     : float = 8.559778667345e-04

# Specific configuration for YMCs MOCCA simulations, inheriting from base experiment config -------------------------------#    
@dataclass
class YMCsMOCCASims(YMCsExperimentConfig):
    """
    ________________________________________________________________________________________________________________________
    Specific configuration for YMCs MOCCA Simulations, inheriting from the base YMCsExperimentConfig.
    ________________________________________________________________________________________________________________________
    Parameters (non-present in base config):
    -> time_column_imbh : Name of the time column in the original MOCCA vms.dat file (Myr units).
    -> mass_column_imbh : Name of the mass column in the original MOCCA vms.dat file (Msun).
    ________________________________________________________________________________________________________________________
    """
    mapping_dics_dir   : str = field(default="./rawdata/ymcs/V2025a/mapping_dicts/mocca/", init=False)
    code               : str = field(default="mocca", init=False)      
    time_column_imbh   : str = "time[Myr]"
    time_column_system : str = "tphys"
    mass_column_imbh   : str = "massNew[Msun]"
