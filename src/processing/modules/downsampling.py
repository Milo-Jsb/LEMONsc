# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Optional, Tuple
from loguru import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.filters import filter_and_downsample_hist2d

# Perfom Downsampling in a full dataset if needed -------------------------------------------------------------------------#
class DownsamplingProcessor:
    """
    ________________________________________________________________________________________________________________________
    Optimized downsampling with optinal channel separation for LEMONsc experiments.
    ________________________________________________________________________________________________________________________
    Config options recognised:
    -> downsample_max_count  (int)  : Maximum points per bin (excess is randomly dropped).
    -> downsample_auto_bins  (bool) : If True (default), bin counts are chosen automatically using the Freedman-Diaconis 
                                      rule. If False, the fixed value in `histogram_bins` is used for both axes
                                      (useful when reproducibility or visual consistency matters).
    -> histogram_bins        (int)  : Fixed bin count used when downsample_auto_bins=False.
    ________________________________________________________________________________________________________________________
    Notes:
    -> DonwsamplingProcessor is agnostic to the working scale of the variables, and this should be checked prior to 
       applying the algorithm.
    ________________________________________________________________________________________________________________________
    """
    # Initialize with config ----------------------------------------------------------------------------------------------#
    def __init__(self, config):
        self.config = config
    
    # [Helper] Freedman-Diaconis binning method ---------------------------------------------------------------------------#
    def _freedman_diaconis_bins(self, data: np.ndarray) -> int:
        """Compute optimal bin count using the Freedman-Diaconis rule."""
        
        # Obtain number of data points and IQR
        n, iqr = len(data), np.percentile(data, 75) - np.percentile(data, 25)
        
        # Sturges fallback when IQR is zero (e.g., for small datasets or low variance)
        if iqr == 0: return max(1, int(np.ceil(np.log2(n))) + 1)
        
        # compute normalization factor and data range
        h           = 2.0 * iqr * (n ** (-1.0 / 3.0))
        data_range  = data.max() - data.min()
        
        # Return the number of bins, ensuring at least 1 bin
        return max(1, int(np.ceil(data_range / h)))

    #  Perform downsampling with optional category separation -------------------------------------------------------------#
    def perform_downsampling(self, x_var: List, y_var: List, metadata: List,
                             filtering_criteria    : Optional[List[Tuple[int, str]]] = None,
                             criteria_var_position : Optional[int]                   = None,
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use 2D histogram-based downsampling. If criteria given, separate and sample by categories."""
        # Convert to numpy arrays
        x_array     = np.array(x_var)
        y_array     = np.array(y_var)
        mdata_array = np.array(metadata)

        # Determine iteration chunks: split by criteria or treat the full dataset as one chunk
        if filtering_criteria is not None and criteria_var_position is not None:
            chunks = [
                (code, name, mdata_array[:, criteria_var_position] == code)
                for code, name in filtering_criteria
            ]
        else:
            chunks = [(None, 'ALL', np.ones(len(x_array), dtype=bool))]

        # Initialize storage for downsampled data
        x_parts, y_parts, meta_parts = [], [], []

        for _, chunk_name, mask in chunks:

            # Skip if no data points for this chunk
            if not np.any(mask): continue

            # Extract chunk data
            x_chunk    = x_array[mask]
            y_chunk    = y_array[mask]
            meta_chunk = mdata_array[mask]

            # Remove inf/nan values
            valid_mask = np.isfinite(x_chunk) & np.isfinite(y_chunk)
            if not np.all(valid_mask):
                x_chunk    = x_chunk[valid_mask]
                y_chunk    = y_chunk[valid_mask]
                meta_chunk = meta_chunk[valid_mask]

            if len(x_chunk) == 0:
                continue

            # Perform downsampling: use automatic Freedman-Diaconis bins or fixed bins from config
            auto_bins = getattr(self.config, 'downsample_auto_bins', True)
            if auto_bins:
                bins_x = self._freedman_diaconis_bins(x_chunk)
                bins_y = self._freedman_diaconis_bins(y_chunk)
            else:
                fixed_bins = getattr(self.config, 'histogram_bins', 200)
                bins_x     = fixed_bins
                bins_y     = fixed_bins

            H, xedges, yedges = np.histogram2d(x_chunk, y_chunk, bins=[bins_x, bins_y])

            idxs = filter_and_downsample_hist2d(x_chunk, y_chunk, H, xedges, yedges,
                                                max_count = self.config.downsample_max_count,
                                                seed      = 42)

            if len(idxs) > 0:
                x_parts.append(x_chunk[idxs])
                y_parts.append(y_chunk[idxs])
                meta_parts.append(meta_chunk[idxs])

        # Concatenate results
        if x_parts:
            return np.concatenate(x_parts), np.concatenate(y_parts), np.concatenate(meta_parts)
        else:
            return np.array([]), np.array([]), np.array([])

#--------------------------------------------------------------------------------------------------------------------------#
