# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing      import List, Optional

# Custom plotting functions ------------------------------------------------------------------------------------------------#
from src.utils.visualize import truncate_colormap, dataset_2Dhist_comparison
from src.utils.visualize import boxplot_features_with_points, classic_correlogram
from src.utils.visualize import plot_efficiency_mass_ratio_dataset

# Unique class for plot relevant features of the dataset or simulations ----------------------------------------------------#
class PlotGenerator:
    """Optimized plotting of files."""
    
    def __init__(self, config, cmap:Optional[str] = None):
        self.config     = config
        self.cmap_trunc = truncate_colormap("gist_stern_r" if cmap is None else cmap)
    
    def create_comparison_plots(self, t_base: List, m_base: List, phy_base: List,
                               t_augm: List, m_augm: List, phy_augm: List,
                               t_down: List, m_down: List, phy_down: List,
                               out_figs : str,
                               scaled   : bool = False):
        """Create all comparison plots efficiently."""
        # Convert to numpy arrays once
        t_base_arr   = np.array(t_base)
        m_base_arr   = np.array(m_base)
        phy_base_arr = np.array(phy_base)
        t_augm_arr   = np.array(t_augm)
        m_augm_arr   = np.array(m_augm)
        phy_augm_arr = np.array(phy_augm)
        t_down_arr   = np.array(t_down)
        m_down_arr   = np.array(m_down)
        phy_down_arr = np.array(phy_down)
        
        # Create full dataset plot for scaled adimentional quantities
        if scaled:
            
            # Normalized time
            tb_scaled = np.log10(t_base_arr / phy_base_arr[:,2]+1)
            ta_scaled = np.log10(t_augm_arr / phy_augm_arr[:,2]+1)
            td_scaled = np.log10(t_down_arr / phy_down_arr[:,2]+1)
            
            # Normalized mass
            mb_scaled = m_base_arr / (phy_base_arr[:,4])
            ma_scaled = m_augm_arr / (phy_augm_arr[:,4])
            md_scaled = m_down_arr / (phy_down_arr[:,4])

            # Labels
            xlabel = r"Normalized Time ($\log_{10}(t/t_{cc}+\epsilon)$)"
            ylabel = r"$f_{\rm{mass}}=$M$_{\rm{MMO}}$/M$_{\rm{tot}}$"
            
        # Else use normal dimentional quantities but in logscale for time
        else:
            # Normalized time
            tb_scaled = np.log10(t_base_arr +1)
            ta_scaled = np.log10(t_augm_arr +1)
            td_scaled = np.log10(t_down_arr +1)

            # Keep mass the same
            mb_scaled = m_base_arr 
            ma_scaled = m_augm_arr 
            md_scaled = m_down_arr 

            # Labels
            xlabel = r"log($t$ [Myr])"
            ylabel = r"M$_{\rm{MMO}}$ [M$_\odot$]"

        # Plot
        self._create_single_plot(tb_scaled, mb_scaled, 
                                 ta_scaled, ma_scaled, 
                                 td_scaled, md_scaled, 
                                 xaxis_label = xlabel,
                                 yaxis_label = ylabel,
                                 name        = "full", 
                                 out_figs    = out_figs)
        
        
        # Create environment-specific plots
        for channel_code, env_name in [(0., 'fast'), (1., 'slow')]:
            self._create_environment_plot(tb_scaled, mb_scaled, phy_base_arr,
                                        ta_scaled, ma_scaled, phy_augm_arr,
                                        td_scaled, md_scaled, phy_down_arr,
                                        xlabel,
                                        ylabel,
                                        channel_code, 
                                        env_name, 
                                        out_figs)
    
    def _create_single_plot(self, t_base: np.ndarray, m_base: np.ndarray,
                           t_augm: np.ndarray, m_augm: np.ndarray,
                           t_down: np.ndarray, m_down: np.ndarray,
                           xaxis_label : str,
                           yaxis_label : str,
                           name: str, out_figs: str):
        """Create a single comparison plot."""        
        dataset_2Dhist_comparison(x_base = t_base, y_base = m_base, 
                                 x_aug   = t_augm, y_aug  = m_augm,
                                 x_filt  = t_down, y_filt = m_down,
                                 name        = name,
                                 titles      = ("Moccasurvey Dataset", "Augmented Dataset", "Downsampled Dataset"),
                                 axislabels  = (xaxis_label, yaxis_label),
                                 bins        = 200, 
                                 cmap        = self.cmap_trunc, 
                                 savepath    = out_figs)
    
    def _create_environment_plot(self, t_base: np.ndarray, m_base: np.ndarray, phy_base: np.ndarray,
                                t_augm: np.ndarray, m_augm: np.ndarray, phy_augm: np.ndarray,
                                t_down: np.ndarray, m_down: np.ndarray, phy_down: np.ndarray,
                                xlabel:str, ylabel:str,
                                channel_code: int, 
                                env_name: str, 
                                out_figs: str):
        
        """Create environment-specific comparison plot."""
        mask = phy_base[:, -1] == channel_code
        
        if not np.any(mask):
            return
        
        # Extract environment-specific data
        t_base_env = t_base[mask]
        m_base_env = m_base[mask]
        
        mask_augm = phy_augm[:, -1] == channel_code
        t_augm_env = t_augm[mask_augm]
        m_augm_env = m_augm[mask_augm]
        
        mask_down = phy_down[:, -1] == channel_code
        t_down_env = t_down[mask_down]
        m_down_env = m_down[mask_down]
        
        self._create_single_plot(t_base_env, m_base_env, t_augm_env, m_augm_env,
                                 t_down_env, m_down_env, xlabel, ylabel,
                                 env_name, out_figs)
    
    @staticmethod
    def _create_features_analysis(feats, names, dataset, experiment, out_figs):

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
                            cmap         = "PuOr",
                            figsize      = (10, 10),
                            show         = False)
     
    @staticmethod
    def _create_efficiency_plot(data, out_figs):
        
        plot_efficiency_mass_ratio_dataset(data_dict = data,
                                           cmap       = "plasma",
                                           figsize    = (7, 5),
                                           title      = None,
                                           cmap_label = "Density",
                                           s_valid    = 10,
                                           s_outlier  = 10,
                                           savepath   = out_figs,
                                           show       = False)

#--------------------------------------------------------------------------------------------------------------------------#