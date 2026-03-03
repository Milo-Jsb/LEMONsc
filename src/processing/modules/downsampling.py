# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Tuple
from loguru import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.filters import filter_and_downsample_hist2d

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/downsampling_output.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Perfom Downsampling in a full dataset if needed -------------------------------------------------------------------------#
class DownsamplingProcessor:
    """
    ________________________________________________________________________________________________________________________
    Optimized downsampling with channel separation for LEMONsc experiments.
    ________________________________________________________________________________________________________________________
    Config options recognised:
    -> downsample_max_count  (int)  : Maximum points per bin (excess is randomly dropped).
    -> downsample_auto_bins  (bool) : If True (default), bin counts are chosen automatically using the Freedman-Diaconis 
                                      rule. If False, the fixed value in `histogram_bins` is used for both axes
                                      (useful when reproducibility or visual consistency matters).
    -> histogram_bins        (int)  : Fixed bin count used when downsample_auto_bins=False.
    ________________________________________________________________________________________________________________________
    """
    # Initialize with config ----------------------------------------------------------------------------------------------#
    def __init__(self, config):
        self.config = config
    
    # Freedman-Diaconis binning method ------------------------------------------------------------------------------------#
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

    #  Perform downsampling with formation channel separation -------------------------------------------------------------#
    def perform_downsampling(self, t_augm: List, m_augm: List, phy_augm: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use 2D histogram-based downsampling with separate processing for FAST and SLOW formation channels."""
        # Convert to numpy arrays 
        t_array   = np.array(t_augm)
        m_array   = np.array(m_augm)
        phy_array = np.array(phy_augm)
        
        # Initialize storage for downsampled data
        t_parts, m_parts, phy_parts = [], [], []
        
        # Process each formation channel. Note: Using -1 index for the type_sim feature (last column in phy_array)
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW')]:

            # Set mask for current channel
            mask = phy_array[:, -1] == channel_code

            # Skip if no data points for this channel
            if not np.any(mask): continue
            
            # Extract channel data
            t_channel   = t_array[mask]
            m_channel   = m_array[mask]
            phy_channel = phy_array[mask]
            
            # Remove inf/nan values
            valid_mask = np.isfinite(t_channel) & np.isfinite(m_channel)
            if not np.all(valid_mask):
                t_channel   = t_channel[valid_mask]
                m_channel   = m_channel[valid_mask]
                phy_channel = phy_channel[valid_mask]
            
            if len(t_channel) == 0:
                continue
            
            # Perform downsampling: use automatic Freedman-Diaconis bins or fixed bins from config
            auto_bins = getattr(self.config, 'downsample_auto_bins', True)
            if auto_bins:
                bins_x = self._freedman_diaconis_bins(t_channel)
                bins_y = self._freedman_diaconis_bins(m_channel)
            else:
                fixed_bins = getattr(self.config, 'histogram_bins', 200)
                bins_x     = fixed_bins
                bins_y     = fixed_bins
            
            H1, xedges, yedges = np.histogram2d(t_channel, m_channel, 
                                                bins=[bins_x, bins_y])
            
            idxs = filter_and_downsample_hist2d(t_channel, m_channel, H1, xedges, yedges,
                                                max_count = self.config.downsample_max_count,
                                                seed      = 42)
            
            if len(idxs) > 0:
                t_parts.append(t_channel[idxs])
                m_parts.append(m_channel[idxs])
                phy_parts.append(phy_channel[idxs])
        
        # Concatenate results
        if t_parts:
            return np.concatenate(t_parts), np.concatenate(m_parts), np.concatenate(phy_parts)
        else:
            return np.array([]), np.array([]), np.array([])

#--------------------------------------------------------------------------------------------------------------------------#
