# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Dict, List, Optional, Tuple, Union

# Custom plotting functions ------------------------------------------------------------------------------------------------#
from src.utils.visualize import truncate_colormap
from src.utils.visualize import violinplot_features
from src.utils.visualize import dataset_2Dhist_comparison, plot_feature_distributions
from src.utils.visualize import classic_correlogram
from src.utils.visualize import plot_simulation_example
from src.utils.visualize import plot_efficiency_mass_ratio_dataset, plot_stellar_mass_half_mass_radius_dataset
from src.utils.visualize import correlation_plot, residual_plot
from src.utils.visualize import feature_importance_plot, plot_simulation_grid
# Set matplotlib configuration ---------------------------------------------------------------------------------------------#
plt.rcParams.update({
    # LaTeX rendering
    "text.usetex"         : True,
    "text.latex.preamble" : r"\usepackage{amsmath} \usepackage{amssymb}",
    
    # Font configuration
    "font.family"         : "serif",
    "font.serif"          : ["Computer Modern Roman"],
    "font.sans-serif"     : ["Computer Modern Sans Serif"],
    "font.monospace"      : ["Computer Modern Typewriter"]})

# Unique class for plot relevant features of the dataset or simulations ----------------------------------------------------#
class PlotGenerator:
    """Optimized plotting of files under the same configuration."""
    
    # Initialize the class with configuration and colormap
    def __init__(self, config, cmap:Optional[str] = None):
        
        self.config     = config
        self.cmap_trunc = truncate_colormap("magma_r" if cmap is None else cmap)
    
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
    def create_comparison_preprocessing(self, t_base: List, m_base: List, phy_base: List, 
                                              t_augm: List, m_augm: List, phy_augm: List, 
                                              t_down: List, m_down: List, phy_down: List, 
                                              out_figs              : str,
                                              filtering_criteria    : Optional[List[Tuple[int, str]]] = None,
                                              criteria_var_position : Optional[int]                   = None,
                                              ):
        
        """Compare time vs mass for different datasets. Check also environment-specific plots."""
        # Convert to numpy arrays once
        t_base_arr, m_base_arr, phy_base_arr = np.array(t_base), np.array(m_base), np.array(phy_base)
        t_augm_arr, m_augm_arr, phy_augm_arr = np.array(t_augm), np.array(m_augm), np.array(phy_augm)
        t_down_arr, m_down_arr, phy_down_arr = np.array(t_down), np.array(m_down), np.array(phy_down)
            
        # Normalized time arrays (if needed, here we keep them as they are for better interpretability)
        tb_scaled, ta_scaled, td_scaled = t_base_arr, t_augm_arr, t_down_arr
        
        # Normalized mass arrays (here we normalize by initial total mass in the cluster assuming is in the 6th column)
        mtot_base = phy_base_arr[:,5] 
        mtot_augm = phy_augm_arr[:,5]
        mtot_down = phy_down_arr[:,5]
        
        m_base_arr, m_augm_arr, m_down_arr = m_base_arr/mtot_base, m_augm_arr/mtot_augm, m_down_arr/mtot_down

        # Labels
        xlabel = r"$t$ [Myr]"
        ylabel = r"M$_{\rm{MMO}}$/M$_{\rm{tot}}$"

        # Plot of the full dataset comparison
        self._create_single_plot(tb_scaled, m_base_arr, ta_scaled, m_augm_arr, td_scaled, m_down_arr, 
                                 xaxis_label = xlabel,
                                 yaxis_label = ylabel,
                                 name        = "full", 
                                 out_figs    = out_figs)
        
        
        # Create category-specific plots
        if filtering_criteria is not None and criteria_var_position is not None:
            for channel_code, env_name in filtering_criteria:
                self._create_environment_plot(tb_scaled, m_base_arr, phy_base_arr,
                                            ta_scaled, m_augm_arr, phy_augm_arr,
                                            td_scaled, m_down_arr, phy_down_arr,
                                            xlabel, ylabel,
                                            channel_code, 
                                            env_name,
                                            criteria_var_position,
                                            out_figs)
    
    # [Helper] Create a single comparison plot ----------------------------------------------------------------------------#
    def _create_single_plot(self, t_base: np.ndarray, m_base: np.ndarray, 
                                  t_augm: np.ndarray, m_augm: np.ndarray,
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
                                       xlabel                : str, 
                                       ylabel                : str,
                                       channel_code          : int, 
                                       env_name              : str,
                                       criteria_var_position : int,
                                       out_figs              : str):
        
        """Create environment-specific comparison plot using criteria_var_position to select the category column."""
        mask = phy_base[:, criteria_var_position] == channel_code
        
        if not np.any(mask):
            return
        
        # Extract environment-specific data
        t_base_env = t_base[mask]
        m_base_env = m_base[mask]
        
        mask_augm  = phy_augm[:, criteria_var_position] == channel_code
        t_augm_env = t_augm[mask_augm]
        m_augm_env = m_augm[mask_augm]
        
        mask_down  = phy_down[:, criteria_var_position] == channel_code
        t_down_env = t_down[mask_down]
        m_down_env = m_down[mask_down]
        
        self._create_single_plot(t_base_env, m_base_env, t_augm_env, m_augm_env,
                                 t_down_env, m_down_env, xlabel, ylabel,
                                 env_name, out_figs)
    
    # Create simple histograms of the features selected -------------------------------------------------------------------#
    @staticmethod
    def create_features_histograms(feats_or : np.ndarray, feats_pr : np.ndarray, labels:List[str], labels_names: List[str], 
                                   out_figs: str):
        
        # Plot the distribution of the features selected
        plot_feature_distributions(feats_raw= feats_or, feats_processed= feats_pr, labels=labels_names, 
                                   cont_features = labels,
                                   sample_size   = 1e6,
                                   bins          = 50,
                                   save_dir      = out_figs)
    
    # Create features analysis plots --------------------------------------------------------------------------------------#
    @staticmethod
    def create_features_analysis(feats : np.ndarray, names: List[str], map_dict: Dict[str, dict], dataset : str, 
                                 experiment: str, 
                                 out_figs:  str):
        
        # Correlation plot of the features selected (Spearman coefficient)        
        classic_correlogram(df= feats, method= "spearman", path_save=out_figs,
                            name_file    = experiment,
                            dataset_name = dataset,
                            mapping_dict = map_dict,
                            cmap         = "RdGy",
                            figsize      = (10, 10),
                            show         = False)   
        
        # Violin plot of the features selected
        violinplot_features(features = feats, mapping_dict=map_dict, path_save= out_figs, 
                            name_file    = experiment, 
                            dataset_name = dataset,
                            figsize      = (18, 8),
                            violin_color = 'goldenrod',
                            nrows        = 2,
                            ncols        = 6,
                            num_points   = 800,
                            ifsave       = True,
                            ifshow       = False)
                  
    # Create efficiency plot ----------------------------------------------------------------------------------------------#
    @staticmethod
    def create_efficiency_plot(data: dict, config: dict, out_figs: str):
        
        plot_efficiency_mass_ratio_dataset(data_dict               = data,
                                           cmap_label              = config.get("cmap_label"),
                                           cmap                    = config.get("cmap"),
                                           cmap_name               = config.get("cmap_name"),
                                           figsize                 = (7, 5),
                                           norm_mode               = config.get("norm_mode"),
                                           include_fit_curve       = config.get("include_fit_curve", True),
                                           include_fit_uncertainty = config.get("include_fit_uncertainty", True),
                                           show                    = True,
                                           savepath                = out_figs)

    # Create mass - half mass radius plot ---------------------------------------------------------------------------------#
    @staticmethod
    def create_mass_radius_plot(data: dict, config: dict, out_figs: str):
                
        plot_stellar_mass_half_mass_radius_dataset(data_dict  = data,
                                                   cmap_label = config.get("cmap_label"),
                                                   cmap       = config.get("cmap"),
                                                   cmap_name  = config.get("cmap_name"),
                                                   figsize    = (7, 5),
                                                   norm_mode  = config.get("norm_mode"),
                                                   show       = True,
                                                   savepath   = out_figs)
        
    # Create ML training results plots ------------------------------------------------------------------------------------#
    @staticmethod
    def create_ml_results_plots(predictions_df_mean : Union[np.ndarray, pd.DataFrame], 
                                true_values_df      : Union[np.ndarray, pd.DataFrame],
                                out_path    : str,
                                model_name  : str,
                                model_title : dict,):
        
        # Generate correlation plot between predictions and true values (using the mean of the predictions across folds)
        correlation_plot(predictions = predictions_df_mean,
                     true_values = true_values_df,
                     path_save   = str(out_path),
                     name_file   = f"{model_name}_mean_preds",
                     model_name  = f"{model_title[model_name]}",
                     cmap        = "magma_r",
                     scale       = None,
                     show        = False)

        # Generate residual plot between predictions and true values (using the mean of the predictions across folds)
        residual_plot(predictions = predictions_df_mean,
                    true_values = true_values_df,
                    path_save   = str(out_path),
                    name_file   = f"{model_name}_mean_preds",
                    model_name  = f"{model_title[model_name]}",
                    cmap        = "magma_r",
                    scale       = None,
                    show        = False)
        
    # Display model predictions from grid selection -----------------------------------------------------------------------#
    @staticmethod
    def plot_simulation_predictions_agaist_gt(subplot_data : list,
                                              n_rows       : int,
                                              n_cols       : int,
                                              save_path    : Optional[str] = None,
                                              figsize      : tuple         = (8, 4),
                                              show         : bool          = False):
        
        plot_simulation_grid(subplot_data, n_rows, n_cols, figsize=figsize,
                             save_path = save_path,
                             show      = show)

    
    # Show feature importance ---------------------------------------------------------------------------------------------#
    @staticmethod
    def plot_feature_importance_bars(importances_dict : dict,
                                     path_save        : str,
                                     name_file        : str,
                                     model_name       : str,
                                     features_names   : Optional[List[str]]        = None,
                                     importance_name  : Optional[str]              = None,
                                     direction_dict   : Optional[Dict[str, float]] = None,
                                     bar_color        : str                        = 'silver',
                                     bar_edgecolor    : str                        = 'black',
                                     bar_width        : float                      = 0.6,
                                     figsize          : tuple                      = (8, 6),
                                     rotation         : int                        = 45,
                                     top_n            : Optional[int]              = None,
                                     ifsave           : bool                       = True,
                                     ifshow           : bool                       = False):
        
        feature_importance_plot(importances_dict, path_save, name_file, model_name,
                                features_names  = features_names,
                                importance_name = importance_name,
                                direction_dict  = direction_dict,
                                bar_color       = bar_color,
                                bar_edgecolor   = bar_edgecolor,
                                bar_width       = bar_width,
                                figsize         = figsize,
                                rotation        = rotation,
                                top_n           = top_n,
                                ifsave          = ifsave,
                                ifshow          = ifshow)
    
#--------------------------------------------------------------------------------------------------------------------------#