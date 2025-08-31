# Modules -----------------------------------------------------------------------------------------------------------------#
import math
import os

import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing                                import Optional, Union, Tuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib                            import cm
from matplotlib.colors                     import Normalize, LogNorm, Colormap
from matplotlib.cm                         import ScalarMappable    
from sklearn.metrics                       import r2_score
from scipy.stats                           import gaussian_kde

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.format import time_preparation, target_preparation

# Matplotlib latex and font configuration ---------------------------------------------------------------------------------#

### NOTE: ADD FROM GRACE PROGRAMS WHEN YOU CAN

# Helpers -----------------------------------------------------------------------------------------------------------------#
def truncate_colormap(cmap, minval=0.05, maxval=1.0, n=256):
    """Helper to truncate a color mat from a min to a max val"""
    cmap     = plt.get_cmap(cmap)
    new_cmap = cm.colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                                                            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Custom Correlation plot with density color map --------------------------------------------------------------------------#
def correlation_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name:str,
                    cmap  : Union[str, Colormap]="magma",
                    scale : Optional[str] = None,
                    show  : bool = True):
    """
    _______________________________________________________________________________________________________________________
    Generate a correlation plot between predictions and true values, with density coloring and R²-Score annotation.
    _______________________________________________________________________________________________________________________
    Parameters:
        predictions (array-like) : Predicted values from the model. Mandatory.
        true_values (array-like) : Ground truth values. Mandatory.
        path_save   (str)        : Directory path to save the plot. Mandatory.
        name_file   (str)        : Name for the saved plot file (without extension). Mandatory.
        model_name  (str)        : Name of the model for annotation. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Calculates R²-Score between predictions and true values.
        - Colors points by density using gaussian_kde.
        - Saves the plot to the specified path with the given file name.
        - Handles input validation and file saving errors.
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if predictions is None or true_values is None:
        raise ValueError("predictions and true_values must not be None.")
    try:
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    if predictions.shape != true_values.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, true_values {true_values.shape}")
    if predictions.ndim != 1:
        raise ValueError("predictions and true_values must be 1D arrays.")
    if not isinstance(path_save, str) or not isinstance(name_file, str) or not isinstance(model_name, str):
        raise TypeError("path_save, name_file, and model_name must be strings.")

    # Compute RMSE -------------------------------------------------------------------------------------------------------#
    try:
        r2 = r2_score(y_true=true_values, y_pred=predictions))
    except Exception as e:
        raise ValueError(f"Error computing RMSE: {e}")

    # Correlation Plot ---------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=(8, 5))

    # Identity line
    p1 = max(np.max(predictions), np.max(true_values))
    p2 = min(np.min(predictions), np.min(true_values))
    ax.plot([p1, p2], [p1, p2], color="black", linewidth=0.7, linestyle="dashed")

    # Density coloring 
    try:
        xy      = np.vstack([true_values, predictions])
        kde     = gaussian_kde(xy)
        z       = kde(xy)
        # Normalize density values to [0,1]
        z       = (z - z.min()) / (z.max() - z.min())
        idx     = z.argsort()
        x, y, z = true_values[idx], predictions[idx], z[idx]
    except Exception as e:
        raise ValueError(f"Error computing density for scatter plot: {e}")

    # Points
    sc = ax.scatter(x, y, marker=".", s=2.5, c=z, cmap=cmap, label=r"$f_{*}(t)$", vmin=0, vmax=1)

    # Colormap 
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Density", size=14)
    cbar.ax.tick_params(labelsize=12)

    # Labels and annotation
    ax.set_ylabel(r"Model's predictions [$M_{\odot}$]", size=14)
    ax.set_xlabel(r"True simulated values [$M_{\odot}$]", size=14)
    ax.tick_params(labelsize=12)
    ax.text(0.05, 0.95, f"{model_name}\n$R^2$-Score: {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    
    if scale is not None:
        ax.set_xscale(scale)
        ax.set_yscale(scale)

    # Save and show plot -------------------------------------------------------------------------------------------------#
    file_path = os.path.join(path_save, f"corr_plot_{name_file}.jpg")
    try:
        os.makedirs(path_save, exist_ok=True)  
        plt.savefig(file_path, bbox_inches="tight", dpi=900)
    except Exception as e:
        raise OSError(f"Could not save plot to {file_path}: {e}")
    
    if show: plt.show()
    plt.close(fig)

# Custom Residual plot with density color map ---------------------------------------------------------------------------#
def residual_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name: str,
                  cmap  : Union[str, Colormap]="magma",
                  scale : Optional[str] = None,
                  show  : bool = True):
    """
    _______________________________________________________________________________________________________________________
    Generate a residual plot (residuals vs predicted values) with density coloring and RMSE annotation.
    _______________________________________________________________________________________________________________________
    Parameters:
        predictions (array-like) : Predicted values from the model. Mandatory.
        true_values (array-like) : Ground truth values. Mandatory.
        path_save   (str)        : Directory path to save the plot. Mandatory.
        name_file   (str)        : Name for the saved plot file (without extension). Mandatory.
        model_name  (str)        : Name of the model for annotation. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Calculates residuals as (true - predicted) values
        - Colors points by density using gaussian_kde
        - Shows RMSE and mean absolute error
        - Includes a horizontal line at y=0 for reference
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if predictions is None or true_values is None:
        raise ValueError("predictions and true_values must not be None.")
    try:
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")

    if predictions.shape != true_values.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, true_values {true_values.shape}")
    if predictions.ndim != 1:
        raise ValueError("predictions and true_values must be 1D arrays.")
    if not isinstance(path_save, str) or not isinstance(name_file, str) or not isinstance(model_name, str):
        raise TypeError("path_save, name_file, and model_name must be strings.")

    # Compute metrics ---------------------------------------------------------------------------------------------------#
    try:
        residuals = true_values - predictions
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
    except Exception as e:
        raise ValueError(f"Error computing metrics: {e}")

    # Residual Plot ----------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=(8, 5))

    # Reference line at y=0
    ax.axhline(y=0, color="black", linewidth=0.7, linestyle="dashed")

    # Density coloring 
    try:
        xy = np.vstack([predictions, residuals])
        kde = gaussian_kde(xy)
        z = kde(xy)
        # Normalize density values to [0,1]
        z = (z - z.min()) / (z.max() - z.min())
        idx = z.argsort()
        x, y, z = predictions[idx], residuals[idx], z[idx]
    except Exception as e:
        raise ValueError(f"Error computing density for scatter plot: {e}")

    # Points
    sc = ax.scatter(x, y, marker=".", s=2.5, c=z, cmap=cmap, label=r"$f_{*}(t)$", vmin=0, vmax=1)

    # Colormap 
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Density", size=14)
    cbar.ax.tick_params(labelsize=12)

    # Labels and annotation
    ax.set_xlabel(r"Predicted values [$M_{\odot}$]", size=14)
    ax.set_ylabel(r"Residuals [$M_{\odot}$]", size=14)
    ax.tick_params(labelsize=12)
    ax.text(0.05, 0.95, f"{model_name}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))
    
    if scale is not None:
        ax.set_xscale(scale)

    # Save and show plot -------------------------------------------------------------------------------------------------#
    file_path = os.path.join(path_save, f"resid_plot_{name_file}.jpg")
    try:
        os.makedirs(path_save, exist_ok=True)  
        plt.savefig(file_path, bbox_inches="tight", dpi=900)
    except Exception as e:
        raise OSError(f"Could not save plot to {file_path}: {e}")
    
    if show: plt.show()
    plt.close(fig)

# Custom Boxplot Analysis with mean values display -----------------------------------------------------------------------#
def boxplot_features_with_points(features: np.ndarray, feature_names: list, path_save: str, name_file: str, 
                                 dataset_name : str,
                                 figsize      : tuple = (24, 6),
                                 point_alpha  : float = 0.2,
                                 point_color  : str   = 'gray',
                                 point_size   : float = 5,
                                 point_jitter : float = 0.02):
    """
    _______________________________________________________________________________________________________________________
    Generate boxplots for all features with mean values displayed, providing distribution analysis.
    _______________________________________________________________________________________________________________________
    Parameters:
        features      (array-like) : Feature array with shape (n_samples, n_features). Mandatory.
        feature_names (list)       : List of feature names corresponding to columns. Mandatory.
        path_save     (str)        : Directory path to save the plot. Mandatory.
        name_file     (str)        : Name for the saved plot file (without extension). Mandatory.
        dataset_name  (str)        : Name of the dataset for annotation. Mandatory.
        figsize       (tuple)      : Figure size (width, height). Default is (24, 6).
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Creates boxplots for each feature showing distribution statistics.
        - Displays mean values as text annotations on each boxplot.
        - Saves the plot to the specified path with the given file name.
        - Handles input validation and file saving errors.
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    if features is None or feature_names is None:
        raise ValueError("features and feature_names must not be None.")
    
    try:
        features = np.asarray(features)
    except Exception as e:
        raise TypeError(f"Could not convert features to numpy array: {e}")
    
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    
    if len(feature_names) != features.shape[1]:
        raise ValueError(f"Mismatch between feature_names and feature columns.")
    
    if not all(isinstance(arg, str) for arg in [path_save, name_file, dataset_name]):
        raise TypeError("path_save, name_file, and dataset_name must be strings.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")
    
    df_features = pd.DataFrame(features, columns=feature_names)
    fig, axes = plt.subplots(figsize=figsize, ncols=features.shape[1])
    if features.shape[1] == 1:
        axes = [axes]

    for i, col in enumerate(feature_names):
        try:
            # Boxplot
            bp = axes[i].boxplot(df_features[col], patch_artist=True)
            bp['boxes'][0].set_color('rebeccapurple')
            bp['medians'][0].set_color('white')
            bp['medians'][0].set_linewidth(2)
            for key in ['whiskers', 'caps']:
                for line in bp[key]:
                    line.set_color('rebeccapurple')
                    line.set_linewidth(1.5)
            if 'fliers' in bp and bp['fliers']:
                bp['fliers'][0].set(marker='.', color='black')

            # Coordenadas X para los puntos, centradas en x=1 (posición del boxplot)
            jitter_strength = 0.05
            x_jitter = np.random.normal(loc=0.0, scale=jitter_strength, size=len(df_features[col]))
            x_coords = 1 + x_jitter
            y_vals   = df_features[col].values
            
            axes[i].scatter(x_coords, y_vals, alpha=point_alpha, color=point_color, s=point_size, marker=".")

            # Stats
            mean_val = df_features[col].mean()
            q1_val = df_features[col].quantile(0.25)
            q3_val = df_features[col].quantile(0.75)
            std_val = df_features[col].std()
            axes[i].text(1.1, mean_val, rf'$\mu=${mean_val:.2f}',
                         transform=axes[i].transData,
                         va='bottom', ha='left', fontsize=12)
            axes[i].axhline(y=mean_val, color='rebeccapurple', linestyle='--', linewidth=1.5, alpha=0.8)
            if i==0: axes[i].set_ylabel('Value', fontsize=12)
            axes[i].set_xticks([])

            axes[i].text(0.02, 0.98, col,
                         transform=axes[i].transAxes,
                         fontsize=20, fontweight='bold',
                         verticalalignment='top')

            stats_text = f'Q1: {q1_val:.3f}\nQ3: {q3_val:.3f}\n$\sigma$ : {std_val:.3f}'
            axes[i].text(0.02, 0.90, stats_text,
                         transform=axes[i].transAxes,
                         fontsize=12,
                         verticalalignment='top',
                         horizontalalignment='left')

        except Exception as e:
            raise ValueError(f"Error with feature {col}: {e}")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    file_path = os.path.join(path_save, f"boxplot_analysis_{name_file}.jpg")
    try:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
    except Exception as e:
        raise OSError(f"Could not save plot to {file_path}: {e}")

    plt.show()
    plt.close(fig)


# Custom Full Correlogram Plot (Upper Triangle) ---------------------------------------------------------------------------#
def classic_correlogram(df: pd.DataFrame, method: str = "pearson", cmap: str = "PuOr",
                         path_save: str = None, name_file: str = None, dataset_name: str = None,
                         show: bool = True, figsize: tuple = None, labels: list = None):
    """
    _______________________________________________________________________________________________________________________
    Generate a full NxN correlogram (upper triangle only) between all numeric variables in the DataFrame.
    _______________________________________________________________________________________________________________________
    Parameters:
        df            (pd.DataFrame) : DataFrame containing numeric variables. Mandatory.
        method        (str)          : Correlation method ("pearson",spearman", "kendall). Default is pearson".
        cmap          (str)          : Matplotlib colormap name. Default is "PuOr".
        path_save     (str)          : Directory path to save the plot. Optional.
        name_file     (str)          : Name for the saved plot file (without extension). Optional.
        dataset_name  (str)          : Name of the dataset for annotation. Optional.
        show          (bool)         : Whether to display the plot. Default is True.
        figsize       (tuple)        : Figure size (width, height). Auto-calculated if None.
        labels        (list)         : Custom labels for axis ticks. If None, uses column names. Optional.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function optionally saves the plot as a .jpg file and displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Computes correlation matrix using the specified method.
        - Displays only the upper triangle to avoid redundancy.
        - Uses circular markers with color intensity representing correlation strength.
        - Saves the plot to the specified path if path_save and name_file are provided.
        - Handles input validation and file saving errors.
        - Tick labels default to column names if not specified.
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if df is None:
        raise ValueError("df must not be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame.")
    
    if df.empty:
        raise ValueError("DataFrame is empty.")
    
    # Check for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")
    
    if method not in ["pearson", "spearman", "kendall"]:
        raise ValueError(f"Invalid method: {method}. Choose from 'pearson', 'spearman', or 'kendall'.")
    
    if not isinstance(cmap, str):
        raise TypeError("cmap must be a string.")
    
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean.")
    
    if path_save is not None and not isinstance(path_save, str):
        raise TypeError("path_save must be a string or None.")
    
    if name_file is not None and not isinstance(name_file, str):
        raise TypeError("name_file must be a string or None.")
    
    if dataset_name is not None and not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string or None.")   
    if figsize is not None and (not isinstance(figsize, tuple) or len(figsize) != 2):
        raise TypeError("figsize must be a tuple of length 2 or None.")
    
    if labels is not None and not isinstance(labels, list):
        raise TypeError("labels must be a list or None.")
    
    # Validate label length if provided
    if labels is not None and len(labels) != len(df.columns):
        raise ValueError(f"Length of labels ({len(labels)}) must match number of columns ({len(df.columns)})")
    
    # Compute correlation matrix -------------------------------------------------------------------------------------------#
    try:
        matrix = df.corr(method=method).to_numpy()
        norm_range = (-1, 1)
    except Exception as e:
        raise ValueError(f"Error computing correlation matrix: {e}")  
    # Keep only upper triangle
    matrix_triu = np.triu(matrix, k=0)
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        n = len(df.columns)
        figsize = (1.4 * n, 1.4 * n)
    
    # Full Correlogram Plot ------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        norm = Normalize(*norm_range)
        cmap_obj = plt.get_cmap(cmap)
        
        # Create circular markers for each correlation value
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                val = matrix_triu[i, j]
                if not np.isnan(val) and val != 0:
                    color = cmap_obj(norm(val))
                    radius = 0.4
                    circle = plt.Circle((j, i), radius=radius, color=color, fill=True)
                    ax.add_patch(circle)
                    
                    # Text color based on background intensity
                    text_color = 'white' if norm(val) > 0.6 else 'black'
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                           color=text_color, fontsize=11)
        
        # Customize plot appearance
        ax.set_xlim(-0.5, len(df.columns) - 0.5)
        ax.set_ylim(-0.5, len(df.columns) - 0.5)
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.columns)))
        
        # Set tick labels - use custom labels if provided, otherwise use column names
        tick_labels = labels if labels is not None else df.columns
        
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=13)
        ax.set_yticklabels(tick_labels, fontsize=13)
        
        # Add background grid
        ax.imshow(matrix_triu, cmap='Greys', alpha=0.1, vmin=-1, vmax=1)
        
        # Set title
        title_map = {
            "pearson" : "Pearson Correlation Matrix",
            "spearman": "Spearman Correlation Matrix",
            "kendall" : "Kendall Correlation Matrix"
        }
        
        title = title_map[method]
        if dataset_name:
            title = f"{title} - {dataset_name}"
        ax.set_title(title, fontsize=16, loc="left", pad=10)
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.70, pad=0.04)
        cbar.ax.tick_params(labelsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
    except Exception as e:
        raise ValueError(f"Error creating correlogram plot: {e}")   
    
    # Save and show plot -------------------------------------------------------------------------------------------------#
    if path_save is not None and name_file is not None:
        file_path = os.path.join(path_save, f"correlogram_{method}_{name_file}.jpg")
        try:
            os.makedirs(path_save, exist_ok=True)
            plt.savefig(file_path, bbox_inches="tight", dpi=300)
        except Exception as e:
            raise OSError(f"Could not save plot to {file_path}: {e}")
    
    if show: plt.show()
    plt.close(fig)

# Single simulation plot --------------------------------------------------------------------------------------------------#
def plot_simulation_example(df: pd.DataFrame, target_types:list, mass_column:str= "massNew[Msun](10)",
                            time_column : str= "time[Myr]",
                            log10_scale : bool = False,
                            norm_factor : Optional[float] = None,
                            save_path   : str ="./figures/",
                            t_cc        : Optional[float] = None, 
                            t_coll      : Optional[float] = None, 
                            t_relax     : Optional[float] = None, 
                            M_crit      : Optional[float] = None,
                            rho_half    : Optional[float] = None,
                            show        : bool= False):
    """
    Plots selected mass evolution targets for a simulation example.
    """
    # Axis and some definitions -------------------------------------------------------------------------------------------#
    TARGETS = {
        "point_mass": {
            "label": r'$M[\rm M_\odot]$',
            "yscale": "linear"
        },
        "delta_mass": {
            "label": r'$\Delta M[\rm M_\odot]$',
            "yscale": "linear"
        },
        "mass_rate": {
            "label": r'$\Delta M /\Delta t[\rm M_\odot/Myr]$',
            "yscale": "linear"
        },
    }

    # Input validation ----------------------------------------------------------------------------------------------------#
    for col in [mass_column, time_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    for t in target_types:
        if t not in TARGETS:
            raise ValueError(f"Unknown target_type '{t}'. Valid types: {list(TARGETS.keys())}")

    # Retrieve temporal values --------------------------------------------------------------------------------------------#
    try:
        time_evol = time_preparation(df[time_column], norm_factor=norm_factor)
    except Exception as e:
        raise RuntimeError(f"Error preparing time: {e}")

    # Plot elements -------------------------------------------------------------------------------------------------------#
    fig, axes = plt.subplots(1, len(target_types), figsize=(6 * len(target_types), 5), squeeze=False)
    axes = axes[0]

    for i, target_type in enumerate(target_types):
        props = TARGETS[target_type]

        try:
            target_vals = target_preparation(mass_evolution= df[mass_column], time_evolution= df[time_column],
                                            norm_factor = norm_factor,
                                            target_type = target_type,
                                            log10_scale = log10_scale)

        except Exception as e:
            raise RuntimeError(f"Error computing target '{target_type}': {e}")

        ax = axes[i]
        ax.plot(time_evol, target_vals, lw=0.75, label='MMO', color="darkblue", marker=".")
        ax.set_ylabel(props["label"], size=14)
        ax.set_xlabel('Time $[Myr]$', size=14)
        ax.set_yscale(props.get("yscale", "linear"))
        ax.legend()

    # Title if metadata ----------------------------------------------------------------------------------------------------#
    if all(x is not None for x in [t_cc, t_coll, t_relax, M_crit, rho_half]):
        title = (
            f"Simulation example: "
            f"$t_{{\\rm{{cc}}}}={t_cc:.3f}$[Myr]; "
            f"$t_{{\\rm{{coll}}}}={t_coll:.3f}$[Myr]; "
            f"$t_{{\\rm{{relax}}}}={t_relax:.3f}$[Myr]; "
            f"$M_{{\\rm{{crit}}}}={M_crit:.2e}$[M$_\\odot$]; "
            f"$\\rho(R_h)={rho_half:.2e}$[pc$^{{-3}}$]"
        )
        fig.suptitle(title, size=16)

    # Save file -----------------------------------------------------------------------------------------------------------#
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    try:
        plt.tight_layout()
        plt.savefig(f"{save_path}evol_sim_example.jpg", dpi=600)
        if show: plt.show()
        plt.close(fig)

    except Exception as e:
        raise RuntimeError(f"Failed to save or display plot: {e}")

# Get a comparison between datasets ---------------------------------------------------------------------------------------#
def dataset_2Dhist_comparison(times_base: np.ndarray, masses_base: np.ndarray,
                              times_aug : np.ndarray, masses_aug : np.ndarray,
                              times_filt: np.ndarray, masses_filt: np.ndarray,
                              name       : Optional[str] = "full",
                              bins       : int = 200,
                              cmap       : Union[str, Colormap ] = "viridis",
                              savepath   : Optional[str] = None,
                              figsize    : tuple = (18, 5),
                              titles     : tuple = ("RawDataset", "AugSampledDataset", "DownSampledDataset"),
                              axislabels : Tuple[str,str]= (r"log($t$ [Myr])", r"M$_{\rm{MMO}}$ [M$_\odot$]"),
                              cmap_label : str = "Count",
                              show       : bool = False):
    """
    _______________________________________________________________________________________________________________________
    Generate 2D histogram comparison plots for three datasets (base, augmented, filtered) showing time-mass distributions.
    _______________________________________________________________________________________________________________________
    Parameters:
        times_base   (array-like) : Time values for base dataset. Mandatory.
        masses_base  (array-like) : Mass values for base dataset. Mandatory.
        times_aug    (array-like) : Time values for augmented dataset. Mandatory.
        masses_aug   (array-like) : Mass values for augmented dataset. Mandatory.
        times_filt   (array-like) : Time values for filtered dataset. Mandatory.
        masses_filt  (array-like) : Mass values for filtered dataset. Mandatory.
        bins         (int)        : Number of bins for histograms. Default is 200.
        cmap         (str)        : Colormap for histograms. Default is "viridis".
        savepath     (str)        : Path to save the plot. Default is None (no saving).
        figsize      (tuple)      : Figure size (width, height). Default is (18, 5).
        titles       (tuple)      : Titles for the three subplots. Default is ("RawDataset", "AugSampledDataset", "DownSampledDataset").
        cmap_label   (str)        : Label for colorbar. Default is "Count".
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function displays the plot and optionally saves it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Converts time values to log10 scale for better visualization.
        - Uses shared bin edges across all histograms for consistent comparison.
        - Applies logarithmic normalization to color scale.
        - Displays point counts for each dataset.
        - Handles input validation and file saving errors.
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if any(x is None for x in [times_base, masses_base, times_aug, masses_aug, times_filt, masses_filt]):
        raise ValueError("All input arrays must not be None.")
    
    try:
        times_base  = np.asarray(times_base)
        masses_base = np.asarray(masses_base)
        times_aug   = np.asarray(times_aug)
        masses_aug  = np.asarray(masses_aug)
        times_filt  = np.asarray(times_filt)
        masses_filt = np.asarray(masses_filt)
    
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")
    
    if not all(len(x) == len(y) for x, y in [(times_base, masses_base), (times_aug, masses_aug), (times_filt, masses_filt)]):
        raise ValueError("Time and mass arrays must have the same length for each dataset.")
    
    if not isinstance(bins, int) or bins <= 0:
        raise TypeError("bins must be a positive integer.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")
    
    if not isinstance(titles, tuple) or len(titles) != 3:
        raise TypeError("titles must be a tuple of length 3.")

    # Data preparation ----------------------------------------------------------------------------------------------------#
    t_base = np.log10(times_base + 1)
    m_base = masses_base
    t_aug  = np.log10(times_aug.flatten() + 1)
    m_aug  = masses_aug.flatten()
    t_filt = np.log10(times_filt.flatten() + 1)
    m_filt = masses_filt.flatten()

    # Calculate histograms manually to share vmax
    H1, xedges, yedges = np.histogram2d(t_base, m_base, bins=bins)
    H2, _, _ = np.histogram2d(t_aug, m_aug, bins=[xedges, yedges])
    H3, _, _ = np.histogram2d(t_filt, m_filt, bins=[xedges, yedges])

    vmax = max(H1.max(), H2.max(), H3.max())
    norm = LogNorm(vmin=1, vmax=vmax)

    # Create figure and grid ----------------------------------------------------------------------------------------------#
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)

    datasets = [
        (t_base, m_base, H1, titles[0]),
        (t_aug, m_aug, H2, titles[1]),
        (t_filt, m_filt, H3, titles[2]),
    ]

    # Create subplots ----------------------------------------------------------------------------------------------------#
    for i, (t, m, H, title) in enumerate(datasets):
        ax = fig.add_subplot(gs[0, i])
        h  = ax.hist2d(t, m, bins=[xedges, yedges], cmap=cmap, norm=norm, cmin=1)
        
        ax.set_title(title, loc="left", size=14)
        ax.set_xlabel(axislabels[0], size=12)
        if i == 0:
            ax.set_ylabel(axislabels[1], size=12)
        
        if i != 0:
            ax.set_yticklabels([])

        # Colorbar (inset)
        cax = inset_axes(ax, width="50%", height="4%", loc="upper left", borderpad=1)
        cb = fig.colorbar(h[3], cax=cax, orientation='horizontal')
        cb.set_label(cmap_label, size=10)
        cb.ax.tick_params(labelsize=8)

        # Add count text
        n_points = len(t)
        ax.text(0.03, 0.75, f"$N_{{\\rm points}}$ = {n_points:,}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Save and display ----------------------------------------------------------------------------------------------------#
    if savepath:
        try:
            plt.savefig(f"{savepath}2dhist_comparison_{name}.jpg", dpi=600, bbox_inches="tight")
        except Exception as e:
            raise OSError(f"Error saving plot to {savepath}: {e}")
    
    if show: plt.show()
    plt.close(fig)
    
#--------------------------------------------------------------------------------------------------------------------------#