# Modules -----------------------------------------------------------------------------------------------------------------#
import math
import os

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing              import Optional
from matplotlib.colors   import Normalize
from matplotlib.cm       import ScalarMappable    
from sklearn.metrics     import mean_squared_error
from scipy.stats         import gaussian_kde

# Custom Correlation plot with density color map --------------------------------------------------------------------------#
def corr_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name:str,
              scale:Optional[str] = None):
    """
    _______________________________________________________________________________________________________________________
    Generate a correlation plot between predictions and true values, with density coloring and RMSE annotation.
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
        - Calculates RMSE between predictions and true values.
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
        rmse = math.sqrt(mean_squared_error(y_true=true_values, y_pred=predictions))
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
        xy = np.vstack([true_values, predictions])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = true_values[idx], predictions[idx], z[idx]
    except Exception as e:
        raise ValueError(f"Error computing density for scatter plot: {e}")

    # Points
    sc = ax.scatter(x, y, marker=".", s=2.5, c=z, cmap="magma", label=r"$f_{*}(t)$")

    # Colormap 
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Density", size=14)
    cbar.ax.tick_params(labelsize=12)

    # Labels and annotation
    ax.set_ylabel("Model's predictions", size=14)
    ax.set_xlabel("True simulated values", size=14)
    ax.tick_params(labelsize=12)
    ax.text(0.05, 0.95, f"{model_name}\nRMSE: {rmse:.4f}",
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
    
    plt.show()
    plt.close(fig)

# Custom Boxplot Analysis with mean values display -----------------------------------------------------------------------#
def boxplot_analysis(features: np.ndarray, feature_names: list, path_save: str, name_file: str, 
                    dataset_name: str, figsize: tuple = (24,6)):
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
    # Input validation ----------------------------------------------------------------------------------------------------#
    if features is None or feature_names is None:
        raise ValueError("features and feature_names must not be None.")
    
    try:
        features = np.asarray(features)
    except Exception as e:
        raise TypeError(f"Could not convert features to numpy array: {e}")
    
    if features.ndim != 2:
        raise ValueError("features must be a 2D array.")
    
    if len(feature_names) != features.shape[1]:
        raise ValueError(f"Number of feature_names ({len(feature_names)}) must match number of features ({features.shape[1]})")
    
    if not isinstance(path_save, str) or not isinstance(name_file, str) or not isinstance(dataset_name, str):
        raise TypeError("path_save, name_file, and dataset_name must be strings.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")
    
    # Create DataFrame for easier manipulation
    df_features = pd.DataFrame(features, columns=feature_names)
    
    # Boxplot Analysis ----------------------------------------------------------------------------------------------------#
    fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=features.shape[1])
    
    # Ensure axes is always an array for consistent iteration
    if features.shape[1] == 1:
        axes = [axes]
    
    # Create boxplots for each feature
    for i, col in enumerate(feature_names):
        try:
            # Create boxplot for the current feature
            bp = axes[i].boxplot(df_features[col], patch_artist=True)
            
            # Customize boxplot appearance
            bp['boxes'][0].set_facecolor('olive')
            bp['medians'][0].set_color('maroon')
            bp['medians'][0].set_linewidth(2)
            bp['whiskers'][0].set_color('black')
            bp['whiskers'][1].set_color('black')
            bp['caps'][0].set_color('black')
            bp['caps'][1].set_color('black')
            
            # Mark outliers with goldenrod dots
            if 'fliers' in bp and len(bp['fliers']) > 0:
                bp['fliers'][0].set_marker('.')
                bp['fliers'][0].set_markerfacecolor('goldenrod')
                bp['fliers'][0].set_markeredgecolor('goldenrod')
            
            # Calculate statistics
            mean_val = df_features[col].mean()
            q1_val = df_features[col].quantile(0.25)
            q3_val = df_features[col].quantile(0.75)
            std_val = df_features[col].std()
            
            # Add mean value as text (center right)
            axes[i].text(1.1, mean_val, rf'$\mu=${mean_val:.2f}',
                transform=axes[i].transData,
                va='bottom', ha='left', fontsize=12)

            # Add mean line
            axes[i].axhline(y=mean_val, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
            # Customize plot appearance
            axes[i].set_ylabel('Value', fontsize=12)
            axes[i].set_xticks([])  # Remove x-axis ticks
            
            # Add feature name in upper left corner
            axes[i].text(0.02, 0.98, col,
                transform=axes[i].transAxes,
                fontsize=20, fontweight='bold',
                verticalalignment='top')
            
            # Add statistics text in upper 
            stats_text = f'Q1: {q1_val:.3f}\nQ3: {q3_val:.3f}\n$\sigma$ : {std_val:.3f}'
            axes[i].text(0.02, 0.90, stats_text,
                transform=axes[i].transAxes,
                fontsize=12,
                verticalalignment='top',
                horizontalalignment='left',)
            
        except Exception as e:
            raise ValueError(f"Error creating boxplot for feature {col}: {e}")
    
    # Adjust layout to prevent overlap
    fig.tight_layout()
    
    # Save and show plot -------------------------------------------------------------------------------------------------#
    file_path = os.path.join(path_save, f"boxplot_analysis_{name_file}.jpg")
    try:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
    except Exception as e:
        raise OSError(f"Could not save plot to {file_path}: {e}")
    
    plt.show()
    plt.close(fig)

# Custom Full Correlogram Plot (Upper Triangle) ---------------------------------------------------------------------------#
def plot_full_correlogram(df: pd.DataFrame, method: str = "pearson", cmap: str = "PuOr",
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
            "pearson": "Pearson Correlation Matrix",
            "spearman": "Spearman Correlation Matrix",
            "kendall": "Kendall Correlation Matrix"
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
    
    if show:
        plt.show()
    
    plt.close(fig)

#--------------------------------------------------------------------------------------------------------------------------#