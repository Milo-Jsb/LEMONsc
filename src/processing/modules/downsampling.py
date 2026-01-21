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
    """Optimized downsampling with channel separation."""
    
    def __init__(self, config):
        self.config = config
    
    def perform_downsampling(self, t_augm: List, m_augm: List, phy_augm: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform downsampling with formation channel separation."""
        # Convert to numpy arrays once
        t_array   = np.array(t_augm)
        m_array   = np.array(m_augm)
        phy_array = np.array(phy_augm)
        
        # Initialize results
        t_parts, m_parts, phy_parts = [], [], []
        
        # Process each formation channel
        # Note: Using -1 index for the type_sim feature (last column in phy_array)
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW')]:

            mask = phy_array[:, -1] == channel_code

            if not np.any(mask):
                continue
                
            # Extract channel data
            t_channel   = t_array[mask]
            m_channel   = m_array[mask]
            phy_channel = phy_array[mask]
            
            # Preprocess
            time_feat = np.log10(t_channel + 1)
            mass_feat = m_channel
            
            # Remove inf/nan values
            valid_mask = np.isfinite(time_feat) & np.isfinite(mass_feat)
            if not np.all(valid_mask):
                time_feat   = time_feat[valid_mask]
                mass_feat   = mass_feat[valid_mask]
                t_channel   = t_channel[valid_mask]
                m_channel   = m_channel[valid_mask]
                phy_channel = phy_channel[valid_mask]
            
            if len(time_feat) == 0:
                continue
            
            # Perform downsampling
            H1, xedges, yedges = np.histogram2d(time_feat, mass_feat, 
                                                bins=[self.config.histogram_bins, self.config.histogram_bins])
            
            idxs = filter_and_downsample_hist2d(time_feat, mass_feat, H1, xedges, yedges, 
                                                min_count = self.config.downsample_min_count, 
                                                max_count = self.config.downsample_max_count,
                                                seed      = 42
                                                )
            
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
