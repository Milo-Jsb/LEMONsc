# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from collections    import defaultdict
from typing         import Optional, Tuple, Union, List, Dict, Any
from dataclasses    import dataclass
from loguru._logger import Logger

# Default config ----------------------------------------------------------------------------------------------------------#
@dataclass
class FilterConfig:
    bingrid          : int   = 200   # Number of bins for 2D histogramming
    min_acc          : int   = 10    # Minimum accumulation per bin
    max_acc          : int   = 150   # Maximum accumulation per bin
    randseed         : int   = 42    # Random seed for reproducibility
    sim_pos          : int   = 0     # Position of simulation data in physical_params entries
    Mcrit_pos        : int   = 8     # Position of M_crit in simulation data
    Mtot_pos         : int   = 5     # Position of M_total in simulation data
    M_loss_total_pos : int   = 9     # Position of M_loss_total in simulation data
    Mmmo_pos         : int   = -1    # Position of M_mmo in mmo_mass entries
    k_thres          : float = 0.5   # Multiplier for IQR to set outlier threshold

def_config = FilterConfig()

# Perform a safe downsampling of points based on 2D histogramming ---------------------------------------------------------#
def safe_downsampling_of_points(feats: np.ndarray, mass: np.ndarray, logger) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    ________________________________________________________________________________________________________________________
    Perform a safe downsampling of points based on 2D histogramming.
    ________________________________________________________________________________________________________________________
    Parameters:
        - feats  (np.ndarray)    : Array of feature points. Mandatory.
        - mass   (np.ndarray)    : Array of mass values corresponding to the feature points. Mandatory.
        - logger (logging.Logger): Logger for logging messages. Mandatory.
    ________________________________________________________________________________________________________________________
    Returns:
        - downsampled_feats (np.ndarray) : Downsampled array of feature points.
        - downsampled_mass  (np.ndarray) : Downsampled array of mass values.
    ________________________________________________________________________________________________________________________
    """
    # Safety check: ensure we have data to downsample
    if len(feats) == 0:
        if logger: logger.warning("No data available for downsampling")
        return None
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

            H1, xedges, yedges = np.histogram2d(time_feat, mass_feat, bins=[def_config.bingrid, def_config.bingrid])
            idxs = filter_and_downsample_hist2d(time_feat, mass_feat, H1, xedges, yedges, 
                                                min_count = def_config.min_acc, 
                                                max_count = def_config.max_acc,
                                                seed      = def_config.randseed)   

            if len(idxs) > 0:
                return feats_temp[idxs], mass_temp[idxs]
            
            else:
                if logger: logger.error("Downsampling resulted in empty dataset")
                return None
        else:
            if logger: logger.error("No valid data points remaining after filtering inf/nan values")
            return None

# Filter and downsize a dataset based in a 2D histogram -------------------------------------------------------------------#
def filter_and_downsample_hist2d(x: np.ndarray, y: np.ndarray, H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, 
                                 min_count: int           = def_config.min_acc, 
                                 max_count: int           = def_config.max_acc, 
                                 seed     : Optional[int] = def_config.randseed
                                ) -> np.ndarray:
    """
    ________________________________________________________________________________________________________________________
    Filter and downsample a dataset based on 2D histogram binning.
    ________________________________________________________________________________________________________________________
    Parameters:
        - x          (np.ndarray)        : X-coordinates of the points. Mandatory.
        - y          (np.ndarray)        : Y-coordinates of the points. Mandatory.
        - H          (np.ndarray)        : 2D histogram array. Mandatory.
        - xedges     (np.ndarray)        : X-axis bin edges. Mandatory.
        - yedges     (np.ndarray)        : Y-axis bin edges. Mandatory.
        - min_count  (int)               : Minimum count threshold. Points in bins with fewer points are discarded. 
        - max_count  (int)               : Maximum count threshold. Points in bins with more points are downsampled.
        - seed       (Optional[int])     : Random seed for reproducible downsampling. 
    ________________________________________________________________________________________________________________________
    Returns:
        - selected_indices (np.ndarray) : Array of indices corresponding to the selected points.
    ________________________________________________________________________________________________________________________
    Notes:
        Points in bins with accumulation < min_count are discarded. For bins with accumulation > max_count, random 
        downsampling is performed. Between min_count and max_count, all points are kept.
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if min_count < 0:
        raise ValueError("min_count must be non-negative.")
    if max_count <= 0:
        raise ValueError("max_count must be positive.")
    if min_count >= max_count:
        raise ValueError("min_count must be less than max_count.")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Assign each point to their respective bin----------------------------------------------------------------------------#
    x_bin = np.digitize(x, xedges) - 1
    y_bin = np.digitize(y, yedges) - 1

    valid = (x_bin >= 0) & (x_bin < H.shape[0]) & (y_bin >= 0) & (y_bin < H.shape[1])

    # Group points by bin
    bin_points = defaultdict(list)
    for i in np.where(valid)[0]:
        bin_id = (x_bin[i], y_bin[i])
        bin_points[bin_id].append(i)

    selected_indices = []

    for bin_id, indices in bin_points.items():
        count = len(indices)
        if count < min_count:
            continue 
        elif count > max_count:
            sampled = np.random.choice(indices, size=max_count, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)

    selected_indices = np.array(selected_indices)
    return selected_indices

# Filter simulations based on the efficiency and  mass ratio relationship -------------------------------------------------#
def efficiency_mass_ratio_relation(mmo_mass        : Union[List[float], np.ndarray], 
                                   physical_params : Union[List[float], np.ndarray], 
                                   path_list       : List[str],
                                   labels_list     : List[str],
                                   logger          : Logger,
                                   config          : FilterConfig = def_config
                                   ) -> Dict[str, Any]:   
    """
    ________________________________________________________________________________________________________________________
    Identify outlier simulations based on the relationship between efficiency (epsilon) and mass ratio.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> mmo_mass        (Union[List[float], np.ndarray]) : List or array of MMO masses for each simulation. Mandatory.
    -> physical_params (Union[List[float], np.ndarray]) : List or array of physical parameters for each simulation. 
                                                          Mandatory.
    -> path_list       (List[str])                      : List of simulation paths. Mandatory.
    -> labels_list     (List[str])                      : List of simulation labels. Mandatory.
    -> logger          (Logger)                         : Logger for logging messages. Mandatory.
    -> config          (FilterConfig)                   : Configuration parameters for filtering. Is set by default.
    ________________________________________________________________________________________________________________________
    Returns:
    -> output (Dict[str, Any]) : Dictionary containing valid simulations and outliers based on the filtering criteria.
    ________________________________________________________________________________________________________________________
    Notes:
    
        The function computes the mass ratio 
        
            mass_ratio = M_total_initial / M_crit_initial
        
        and efficiency,  
        
            epsilon = M_mmo_final / M_stellar_final = M_mmo_final / (M_total_initial - M_loss_total)
        
        for each simulation, fits a curve based on Vergara et al.(2025), and calculates the perpendicular distance of each 
        point to this curve. Simulations that fall beyond a threshold (defined as median + k*IQR of distances) are 
        classified as outliers for further explorations. Outliers are not removed from the dataset, but flagged for review.
    
    Reference: Vergara et al. (2025),  10.48550/arXiv.2508.14260
    ________________________________________________________________________________________________________________________
    """
    # Compute elements of interest from the input elements
    init_totmass    = np.array([entry[config.sim_pos][config.Mtot_pos] for entry in physical_params])
    init_mcrit      = np.array([entry[config.sim_pos][config.Mcrit_pos] for entry in physical_params])
    final_bhmass    = np.array([mmo[config.Mmmo_pos] for mmo in mmo_mass])
    total_mass_loss = np.array([entry[config.sim_pos][config.M_loss_total_pos] for entry in physical_params])
    
    mass_ratio = init_totmass / init_mcrit
    epsilon    = final_bhmass / (init_totmass - total_mass_loss)
    
    # Define fit from Vergara et al. (2025)
    def V2025_epsilon_BH(m_ratio: np.ndarray, k: float=4.63, x0: float=4.0, a: float=-0.1):
        X = np.log(m_ratio)
        return (1 + np.exp(-k * (X - x0)))**a
    
    # Compute perpendicular distance to the curve Work in log-space for x to match the functional form
    log_mass_ratio = np.log(mass_ratio)
    
    # Normalize coordinates to make distances comparable
    x_mean, x_std = np.mean(log_mass_ratio), np.std(log_mass_ratio)
    y_mean, y_std = np.mean(epsilon), np.std(epsilon)
    
    log_mass_ratio_norm = (log_mass_ratio - x_mean) / x_std
    epsilon_norm        = (epsilon - y_mean) / y_std
    
    # Sample points along the curve for distance calculation
    log_mr_min, log_mr_max = log_mass_ratio.min(), log_mass_ratio.max()
    log_mr_range = log_mr_max - log_mr_min
    log_mr_curve = np.linspace(log_mr_min - 0.2 * log_mr_range, log_mr_max + 0.2 * log_mr_range, 1000)
    mr_curve     = np.exp(log_mr_curve)
    eps_curve    = V2025_epsilon_BH(mr_curve)
    
    # Normalize curve coordinates
    log_mr_curve_norm = (log_mr_curve - x_mean) / x_std
    eps_curve_norm    = (eps_curve - y_mean) / y_std
    
    # Calculate perpendicular distance for each point
    dist = np.zeros(len(mass_ratio))
    for i in range(len(mass_ratio)):
        # Compute Euclidean distance from point to all curve points
        dx = log_mr_curve_norm - log_mass_ratio_norm[i]
        dy = eps_curve_norm - epsilon_norm[i]
        distances = np.sqrt(dx**2 + dy**2)
        # Take minimum distance
        dist[i] = np.min(distances)
 
    logger.info("Filtering simulations based on perpendicular distance to fit curve (median and IQR)...")
        
    # Set up a threshold based on median + k*IQR
    med = np.median(dist)
    iqr = np.percentile(dist, 75) - np.percentile(dist, 25)    
    
    threshold = med + config.k_thres * iqr

    # Masks for good/bad simulations
    mask_good = dist <= threshold
    mask_bad  = ~mask_good
    
    # Built output dictionary (with physical values, not normalized)
    output = {
        "valid_sims": {
            "paths"      : path_list,
            "labels"     : labels_list,
            "mass_ratio" : mass_ratio,
            "epsilon"    : epsilon,
        },
        "outliers": {
            "paths"      : [p for p, m in zip(path_list, mask_bad) if m],
            "labels"     : [l for l, m in zip(labels_list, mask_bad) if m],
            "mass_ratio" : mass_ratio[mask_bad],
            "epsilon"    : epsilon[mask_bad],
        }
            }
    
    # Log summary information
    logger.info(f"Efficiency vs Mass Ratio Filter Results:")
    logger.info(f"Detected Outliers: {mask_bad.sum()} of {len(path_list)} simulations.")

    # If thresholding was applied, log the threshold values
    if threshold is not None:
        logger.info(f"Threshold = {threshold:.4f} (median={med:.4f}, IQR={iqr:.4f})")
    
    # Else, the user provided existing files
    else:
        logger.info(f"Loaded from existing files (median={med:.4f}, IQR={iqr:.4f})")

    return output
#--------------------------------------------------------------------------------------------------------------------------#