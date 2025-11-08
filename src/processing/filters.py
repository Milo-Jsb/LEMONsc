# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from collections  import defaultdict
from typing       import Optional, Tuple
from dataclasses  import dataclass

# Default config ----------------------------------------------------------------------------------------------------------#
@dataclass
class FilterConfig:
    bingrid : int = 200
    min_acc : int = 10
    max_acc : int = 150
    randseed: int = 42

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
                                 min_count: int = def_config.min_acc, 
                                 max_count: int = def_config.max_acc, 
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

#--------------------------------------------------------------------------------------------------------------------------#