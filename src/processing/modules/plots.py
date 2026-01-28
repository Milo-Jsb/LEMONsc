# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Optional

# Custom plotting functions ------------------------------------------------------------------------------------------------#
from src.utils.visualize import plot_simulation_example
from src.utils.visualize import truncate_colormap, dataset_2Dhist_comparison
from src.utils.visualize import boxplot_features_with_points, classic_correlogram
from src.utils.visualize import plot_efficiency_mass_ratio_dataset

# Set matplotlib configuration ---------------------------------------------------------------------------------------------#
plt.rcParams.update({
    # LaTeX rendering
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    
    # Font configuration
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans Serif"],
    "font.monospace": ["Computer Modern Typewriter"],
    "font.size": 12,})

# Unique class for plot relevant features of the dataset or simulations ----------------------------------------------------#
class PlotGenerator:
    """Optimized plotting of files under the same configuration."""
    
    # Initialize the class with configuration and colormap
    def __init__(self, config, cmap:Optional[str] = None):
        
        self.config     = config
        self.cmap_trunc = truncate_colormap("gist_stern_r" if cmap is None else cmap)
    
    # Create an example plot of one simulation ----------------------------------------------------------------------------#
    def simulation_example(self, imbh_df: np.ndarray, ifeats: dict, out_figs : str):
        """Create a plot showing the evolution of one simulation with key features indicated."""
        plot_simulation_example(imbh_df, save_path = out_figs,
                                t_cc      = ifeats["tcc"][0], 
                                t_coll    = ifeats["tcoll"][0], 
                                t_relax   = ifeats["trelax"][0], 
                                M_crit    = ifeats["mcrit"][0],
                                rho_half  = ifeats["rho_half"][0])
        
    # Create a comparison plot between different preprocessing datasets ---------------------------------------------------#
    def create_comparison_preprocessing(self, t_base: List, m_base: List, phy_base: List, t_augm: List, m_augm: List, 
                                        phy_augm: List, t_down: List, m_down: List, phy_down: List, out_figs : str):
        
        """Compare time vs mass for different datasets. Check also environment-specific plots."""
        # Convert to numpy arrays once
        t_base_arr, m_base_arr, phy_base_arr = np.array(t_base), np.array(m_base), np.array(phy_base)
        t_augm_arr, m_augm_arr, phy_augm_arr = np.array(t_augm), np.array(m_augm), np.array(phy_augm)
        t_down_arr, m_down_arr, phy_down_arr = np.array(t_down), np.array(m_down), np.array(phy_down)
            
        # Normalized time scale (log10_1p scale)
        tb_scaled, ta_scaled, td_scaled = np.log10(t_base_arr +1), np.log10(t_augm_arr +1), np.log10(t_down_arr +1)

        # Labels
        xlabel = r"$\log_{10}$($t$ [Myr])"
        ylabel = r"M$_{\rm{MMO}}$ [M$_\odot$]"

        # Plot of the full dataset comparison
        self._create_single_plot(tb_scaled, m_base_arr, ta_scaled, m_augm_arr, td_scaled, m_down_arr, 
                                 xaxis_label = xlabel,
                                 yaxis_label = ylabel,
                                 name        = "full", 
                                 out_figs    = out_figs)
        
        
        # Create environment-specific plots
        for channel_code, env_name in [(0., 'fast'), (1., 'slow')]:
            self._create_environment_plot(tb_scaled, m_base_arr, phy_base_arr,
                                          ta_scaled, m_augm_arr, phy_augm_arr,
                                          td_scaled, m_down_arr, phy_down_arr,
                                          xlabel, ylabel,
                                          channel_code, 
                                          env_name, 
                                          out_figs)
    
    # [Helper] Create a single comparison plot ----------------------------------------------------------------------------#
    def _create_single_plot(self, t_base: np.ndarray, m_base: np.ndarray, t_augm: np.ndarray, m_augm: np.ndarray,
                            t_down: np.ndarray, m_down: np.ndarray,
                            xaxis_label : str,
                            yaxis_label : str,
                            name        : str, 
                            out_figs    : str):
        """Create a single comparison plot using a 2D histogram for time vs mass."""        
        dataset_2Dhist_comparison(x_base = t_base, y_base = m_base, x_aug  = t_augm, y_aug  = m_augm,
                                  x_filt = t_down, y_filt = m_down,
                                  name       = name,
                                  titles     = ("Original Dataset", "Augmented Dataset", "Downsampled Dataset"),
                                  axislabels = (xaxis_label, yaxis_label),
                                  bins       = self.config.histogram_bins, 
                                  cmap       = self.cmap_trunc, 
                                  savepath   = out_figs)
    
    # [Helper] Create environment-specific comparison plot ----------------------------------------------------------------#
    def _create_environment_plot(self, t_base: np.ndarray, m_base: np.ndarray, phy_base: np.ndarray,
                                t_augm: np.ndarray, m_augm: np.ndarray, phy_augm: np.ndarray,
                                t_down: np.ndarray, m_down: np.ndarray, phy_down: np.ndarray,
                                xlabel:str, ylabel:str,
                                channel_code: int, 
                                env_name: str, 
                                out_figs: str):
        
        """Create environment-specific comparison plot. Assume the position of the channel code in the parameters array."""
        mask = phy_base[:, -1] == channel_code
        
        if not np.any(mask):
            return
        
        # Extract environment-specific data
        t_base_env = t_base[mask]
        m_base_env = m_base[mask]
        
        mask_augm  = phy_augm[:, -1] == channel_code
        t_augm_env = t_augm[mask_augm]
        m_augm_env = m_augm[mask_augm]
        
        mask_down  = phy_down[:, -1] == channel_code
        t_down_env = t_down[mask_down]
        m_down_env = m_down[mask_down]
        
        self._create_single_plot(t_base_env, m_base_env, t_augm_env, m_augm_env,
                                 t_down_env, m_down_env, xlabel, ylabel,
                                 env_name, out_figs)
    
    # Create features analysis plots --------------------------------------------------------------------------------------#
    @staticmethod
    def create_features_analysis(feats : np.ndarray, names: List[str], dataset : str, experiment: str, out_figs: str):

        # Boxplot of the features selected        
        boxplot_features_with_points(features= feats, feature_names= names,
                                     path_save     = out_figs,
                                     figsize       = (18, 9),
                                     name_file     = experiment,
                                     dataset_name  = dataset,
                                     nrows         = 2,
                                     ncols         = 5,
                                     point_color   = "wheat",
                                     ifsave        = True,
                                     ifshow        = False)
        
        # Correlation plot of the features selected (Spearman coefficient)        
        classic_correlogram(df= feats, method= "spearman", path_save=out_figs,
                            name_file    = experiment,
                            dataset_name = None,
                            labels       = names,
                            cmap         = "RdGy",
                            figsize      = (10, 10),
                            show         = False)
     
    # Create efficiency plot ----------------------------------------------------------------------------------------------#
    @staticmethod
    def create_efficiency_plot(data: dict, config: dict, out_figs: str):
        
        plot_efficiency_mass_ratio_dataset(data_dict         = data,
                                           cmap_label        = config.get("cmap_label"),
                                           cmap              = config.get("cmap"),
                                           cmap_name         = config.get("cmap_name"),
                                           figsize           = (7, 5),
                                           norm_mode         = config.get("norm_mode"),
                                           include_fit_curve = config.get("include_fit_curve", True),
                                           show              = True,
                                           savepath          = out_figs)

#--------------------------------------------------------------------------------------------------------------------------#