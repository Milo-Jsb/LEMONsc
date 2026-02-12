# Modules -----------------------------------------------------------------------------------------------------------------#
import os

import numpy               as np
import pandas              as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing                                import Optional, Union, Tuple, Dict, Literal, List
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches                    import Patch
from matplotlib                            import cm
from matplotlib.colors                     import Normalize, LogNorm, Colormap, Normalize, BoundaryNorm, LogNorm
from matplotlib.cm                         import ScalarMappable    
from sklearn.metrics                       import r2_score
from scipy.stats                           import gaussian_kde, pearsonr, spearmanr
from scipy.optimize                        import curve_fit
from sklearn.linear_model                  import LinearRegression

# Helpers -----------------------------------------------------------------------------------------------------------------#
def truncate_colormap(cmap, minval=0.05, maxval=1.0, n=256):
    """Helper to truncate a color mat from a min to a max val"""
    cmap     = plt.get_cmap(cmap)
    new_cmap = cm.colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                                                            cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Plot the partial correlation coefficients with bootstrap uncertainties --------------------------------------------------#
def plot_partial_correlation_bars(df: pd.DataFrame, features: list, target: str, path_save: str, name_file: str,
                                  corr_metric    : Literal['pearson', 'spearman'] = 'pearson',
                                  features_names : Optional[List[str]] = None,
                                  target_name    : Optional[str] = None,
                                  n_bootstrap    : int   = 1000,
                                  bar_color      : str   = 'steelblue',
                                  bar_edgecolor  : str   = 'black',
                                  bar_width      : float = 0.6,
                                  figsize        : tuple = (10, 6),
                                  rotation       : int   = 45,
                                  ifsave         : bool  = True,
                                  ifshow         : bool  = False):
    """
    _______________________________________________________________________________________________________________________
    Plot Partial Correlation Coefficients with bootstrap-estimated uncertainties for feature importance analysis.
    _______________________________________________________________________________________________________________________
    Parameters:
    -> df             (pd.DataFrame) : Input dataframe containing features and target. Mandatory.
    -> features       (list)         : List of feature column names. Mandatory.
    -> target         (str)          : Target column name. Mandatory.
    -> path_save      (str)          : Directory path to save the plot. Mandatory.
    -> name_file      (str)          : Name for the saved plot file (without extension). Mandatory.
    -> corr_metric    (str)          : Correlation metric to use ('pearson' or 'spearman'). Default is 'pearson'.
    -> features_names (list)         : Optional list of feature display names. Default is None.
    -> target_name    (str)          : Optional display name for the target variable. Default is None.
    -> n_bootstrap    (int)          : Number of bootstrap resamples for uncertainty estimation. Default is 1000.
    -> bar_color      (str)          : Color for the bars. Default is 'steelblue'.
    -> bar_edgecolor  (str)          : Edge color for the bars. Default is 'black'.
    -> bar_width      (float)        : Width of the bars. Default is 0.6.
    -> figsize        (tuple)        : Figure size (width, height). Default is (10, 6).
    -> rotation       (int)          : Rotation angle for x-axis labels. Default is 45.
    -> ifsave         (bool)         : Whether to save the plot. Default is True.
    -> ifshow         (bool)         : Whether to show the plot. Default is False.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and optionally displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Computes partial correlation by regressing out other features from both Xi and y.
        - Uses bootstrap resampling to estimate standard errors of partial correlations.
        - Reference line at y=0 helps identify positive vs negative partial correlations.
        - Handles input validation and ensures all required columns are present.
        - For single feature case, computes regular correlation instead of partial correlation.
    _______________________________________________________________________________________________________________________
    Methodology:
        For each feature Xi:
        1. Fit X_{-i} -> Xi to get residuals res_xi (Xi | X_{-i})
        2. Fit X_{-i} -> y to get residuals res_y (y | X_{-i})
        3. Compute Pearson/Spearman correlation between res_xi and res_y
        4. Bootstrap: resample indices with replacement, recompute correlation
        5. Use std of bootstrap samples as error bars
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, KeyError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if df is None or features is None or target is None:
        raise ValueError("df, features, and target must not be None.")
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    if not isinstance(features, list) or len(features) == 0:
        raise TypeError("features must be a non-empty list of column names.")
    
    if not isinstance(target, str):
        raise TypeError("target must be a string.")
    
    if not all(isinstance(arg, str) for arg in [path_save, name_file]):
        raise TypeError("path_save and name_file must be strings.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")
    
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise TypeError("n_bootstrap must be a positive integer.")
    
    if corr_metric not in ['pearson', 'spearman']:
        raise ValueError("corr_metric must be 'pearson' or 'spearman'.")
    
    # Check if columns exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features not found in dataframe: {missing_features}")
    
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe.")
    
    # Validate feature_names length if provided
    if features_names is not None:
        if not isinstance(features_names, list):
            raise TypeError("features_names must be a list or None.")
        if len(features_names) != len(features):
            raise ValueError(f"features_names length ({len(features_names)}) must match features length ({len(features)}).")
    
    # Prepare data --------------------------------------------------------------------------------------------------------#
    try:
        X = df[features].values.astype(float)
        y = df[target].values.astype(float)
    except Exception as e:
        raise TypeError(f"Could not convert features/target to numeric arrays: {e}")
    
    # Check for missing values
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Data contains NaN values. Please handle missing data before plotting.")
    
    n_samples = len(y)
    if n_samples < 10:
        raise ValueError(f"Insufficient samples ({n_samples}). Need at least 10 samples for reliable estimates.")
    
    # Helper function for computing correlation ---------------------------------------------------------------------------#
    def compute_correlation(x, y, method='pearson'):
        """Compute correlation between two arrays."""
        if method == 'pearson':
            return pearsonr(x, y)[0]
        else:  # spearman
            return spearmanr(x, y)[0]
    
    # Compute Partial Correlation Coefficients ----------------------------------------------------------------------------#
    pcc_values = np.zeros(len(features))
    pcc_errors = np.zeros(len(features))
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for i, feat_name in enumerate(features):
        # Indices of other features
        idx_other = [j for j in range(len(features)) if j != i]
        
        if len(idx_other) == 0:
            # Only one feature, partial correlation = regular correlation
            pcc_values[i] = compute_correlation(X[:, i], y, corr_metric)
            pcc_errors[i] = 0.0
            continue
        
        # Regress Xi on other features to get residuals
        reg_x = LinearRegression(fit_intercept=True).fit(X[:, idx_other], X[:, i])
        res_x = X[:, i] - reg_x.predict(X[:, idx_other])
        
        # Regress y on other features to get residuals
        reg_y = LinearRegression(fit_intercept=True).fit(X[:, idx_other], y)
        res_y = y - reg_y.predict(X[:, idx_other])
        
        # Partial correlation is correlation between residuals
        pcc_values[i] = compute_correlation(res_x, res_y, corr_metric)
        
        # Bootstrap to estimate uncertainty -------------------------------------------------------------------------------#
        bootstrap_pccs = np.zeros(n_bootstrap)
        
        for b in range(n_bootstrap):
            # Resample indices with replacement
            boot_idx = np.random.randint(0, n_samples, size=n_samples)
            
            # Compute correlation on bootstrap sample
            try:
                r_boot = compute_correlation(res_x[boot_idx], res_y[boot_idx], corr_metric)
                bootstrap_pccs[b] = r_boot
            except:
                # In rare cases correlation might fail (e.g., constant values)
                bootstrap_pccs[b] = np.nan
        
        # Remove any NaN values from bootstrap
        bootstrap_pccs = bootstrap_pccs[~np.isnan(bootstrap_pccs)]
        
        # Use standard deviation as uncertainty estimate
        if len(bootstrap_pccs) > 0:
            pcc_errors[i] = np.std(bootstrap_pccs)
        else:
            pcc_errors[i] = 0.0
    
    # Create Plot ---------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(features))
    
    # Create bars with error bars
    bars = ax.bar(x_pos, pcc_values, yerr=pcc_errors, width=bar_width, color=bar_color, edgecolor=bar_edgecolor,
                  capsize=4, linewidth=1.2, alpha=0.8, error_kw={'linewidth': 1.5})
    
    # Add reference line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    
    # Customize axes
    feature_labels = features_names if features_names is not None else features
    target_label   = target_name if target_name is not None else target
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_labels, rotation=rotation, ha='right', fontsize=11)
    
    ax.set_ylabel(fr"PCC ($y$ = {target_label})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Features", fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)
    
    # Add value labels on top of bars
    for i, (val, err) in enumerate(zip(pcc_values, pcc_errors)):
        y_pos = val + err + 0.02 if val >= 0 else val - err - 0.02
        va = 'bottom' if val >= 0 else 'top'
        ax.text(i, y_pos, f'{val:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Set y-axis limits with some padding
    y_max = np.max(np.abs(pcc_values) + pcc_errors)
    if y_max > 0:
        ax.set_ylim(-y_max * 1.15, y_max * 1.15)
    else:
        # Fallback if all values are zero
        ax.set_ylim(-0.1, 0.1)
    
    fig.tight_layout()
    
    # Save and show plot --------------------------------------------------------------------------------------------------#
    file_path = os.path.join(path_save, f"partial_corr_{name_file}.jpg")
    
    if ifsave:
        try:
            os.makedirs(path_save, exist_ok=True)
            plt.savefig(file_path, bbox_inches="tight", dpi=600)
        except Exception as e:
            raise OSError(f"Could not save plot to {file_path}: {e}")
    
    if ifshow: plt.show()
    
    plt.close(fig)

# Custom Correlation plot with density color map --------------------------------------------------------------------------#
def correlation_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name:str,
                    cmap  : Union[str, Colormap]="inferno",
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
        r2 = r2_score(y_true=true_values, y_pred=predictions)
    except Exception as e:
        raise ValueError(f"Error computing R2-Score: {e}")

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
    sc = ax.scatter(x, y, marker=".", s=2.5, c=z, cmap=cmap, alpha=0.6, label=r"$f_{*}(t)$", vmin=0, vmax=1)

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

# Custom Residual plot with density color map -----------------------------------------------------------------------------#
def residual_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name: str,
                  cmap  : Union[str, Colormap]="inferno",
                  scale : Optional[str] = None,
                  show  : bool = True):
    """
    ________________________________________________________________________________________________________________________
    Generate a residual plot (residuals vs predicted values) with density coloring and RMSE annotation.
    ________________________________________________________________________________________________________________________
    Parameters:
        predictions (array-like) : Predicted values from the model. Mandatory.
        true_values (array-like) : Ground truth values. Mandatory.
        path_save   (str)        : Directory path to save the plot. Mandatory.
        name_file   (str)        : Name for the saved plot file (without extension). Mandatory.
        model_name  (str)        : Name of the model for annotation. Mandatory.
    ________________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and displays it.
    ________________________________________________________________________________________________________________________
    Notes:
        - Calculates residuals as (true - predicted) values
        - Colors points by density using gaussian_kde
        - Shows RMSE and mean absolute error
        - Includes a horizontal line at y=0 for reference
    ________________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    ________________________________________________________________________________________________________________________
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
        r2  = r2_score(true_values, predictions)

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
    sc = ax.scatter(x, y, marker=".", s=2.5, c=z, cmap=cmap, alpha=0.6, label=r"$f_{*}(t)$", vmin=0, vmax=1)

    # Colormap 
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Density", size=14)
    cbar.ax.tick_params(labelsize=12)

    # Labels and annotation
    ax.set_xlabel(r"Predicted values [$M_{\odot}$]", size=14)
    ax.set_ylabel(r"Residuals [$M_{\odot}$]", size=14)
    ax.tick_params(labelsize=12)
    ax.text(0.05, 0.95, f"{model_name}\nR$^2$-Score: {r2:.4f}",
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

# Custom Boxplot Analysis with mean values display ------------------------------------------------------------------------#
def boxplot_features_with_points(features: np.ndarray, feature_names: list, path_save: str, name_file: str, 
                                 dataset_name : str,
                                 figsize      : tuple = (20, 10),
                                 point_alpha  : float = 0.2,
                                 point_color  : str   = 'gray',
                                 point_size   : float = 5,
                                 point_jitter : float = 0.02,
                                 nrows        : int   = 2,
                                 ncols        : int   = 4,
                                 ifsave       : bool  = True,
                                 ifshow       : bool  = False):
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
        figsize       (tuple)      : Figure size (width, height). Default is (20, 10).
        nrows         (int)        : Number of rows in the subplot grid. Default is 2.
        ncols         (int)        : Number of columns in the subplot grid. Default is 4.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Creates boxplots for each feature showing distribution statistics.
        - Arranges subplots in a nrows x ncols grid.
        - Displays mean values as text annotations on each boxplot.
        - Saves the plot to the specified path with the given file name.
        - Handles input validation and file saving errors.
        - If fewer features than subplots, remaining subplots are hidden.
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
    
    if not isinstance(nrows, int) or not isinstance(ncols, int) or nrows <= 0 or ncols <= 0:
        raise TypeError("nrows and ncols must be positive integers.")
    
    # Check if we have enough subplots for all features
    total_subplots = nrows * ncols
    n_features = features.shape[1]
    
    if n_features > total_subplots:
        raise ValueError(f"Not enough subplots ({total_subplots}) for all features ({n_features}). "
                        f"Increase nrows or ncols, or reduce number of features.")
    
    df_features = pd.DataFrame(features, columns=feature_names)
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(feature_names):
        try:
            ax = axes_flat[i]
            
            # Boxplot
            bp = ax.boxplot(df_features[col], patch_artist=True)
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
            
            ax.scatter(x_coords, y_vals, alpha=point_alpha, color=point_color, s=point_size, marker=".")

            # Stats
            mean_val = df_features[col].mean()
            q1_val = df_features[col].quantile(0.25)
            q3_val = df_features[col].quantile(0.75)
            std_val = df_features[col].std()
            ax.text(1.1, mean_val, rf'$\mu=${mean_val:.2f}',
                    transform=ax.transData,
                    va='bottom', ha='left', fontsize=10)
            ax.axhline(y=mean_val, color='rebeccapurple', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Add ylabel only to leftmost plots
            if i % ncols == 0:
                ax.set_ylabel('Value', fontsize=12)
            
            ax.set_xticks([])

            ax.text(0.02, 0.98, col,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    verticalalignment='top')

            stats_text = f'Q1: {q1_val:.3f}\nQ3: {q3_val:.3f}\n$\sigma$ : {std_val:.3f}'
            ax.text(0.02, 0.85, stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left')

        except Exception as e:
            raise ValueError(f"Error with feature {col}: {e}")
    
    # Hide unused subplots
    for i in range(n_features, total_subplots):
        axes_flat[i].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    file_path = os.path.join(path_save, f"boxplot_analysis_{name_file}.jpg")
    
    if ifsave:
        try:
            os.makedirs(path_save, exist_ok=True)
            plt.savefig(file_path, bbox_inches="tight", dpi=600)
        except Exception as e:
            raise OSError(f"Could not save plot to {file_path}: {e}")

    if ifshow: plt.show()
    
    plt.close(fig)

# Custom Violin Plot Analysis with statistics display ---------------------------------------------------------------------#
def violinplot_features(features: np.ndarray, feature_names: list, path_save: str, name_file: str, 
                        dataset_name : str,
                        figsize      : tuple = (20, 10),
                        violin_color : str   = 'rebeccapurple',
                        nrows        : int   = 2,
                        ncols        : int   = 4,
                        num_points   : int   = 1000,
                        ifsave       : bool  = True,
                        ifshow       : bool  = False):
    """
    _______________________________________________________________________________________________________________________
    Generate violin plots for all features with statistical information displayed, providing distribution analysis.
    _______________________________________________________________________________________________________________________
    Parameters:
        features      (array-like) : Feature array with shape (n_samples, n_features). Mandatory.
        feature_names (list)       : List of feature names corresponding to columns. Mandatory.
        path_save     (str)        : Directory path to save the plot. Mandatory.
        name_file     (str)        : Name for the saved plot file (without extension). Mandatory.
        dataset_name  (str)        : Name of the dataset for annotation. Mandatory.
        figsize       (tuple)      : Figure size (width, height). Default is (20, 10).
        violin_color  (str)        : Color for the violin plots. Default is 'rebeccapurple'.
        nrows         (int)        : Number of rows in the subplot grid. Default is 2.
        ncols         (int)        : Number of columns in the subplot grid. Default is 4.
        num_points    (int)        : Number of points to use for the violin plot. Default is 1000.
        ifsave        (bool)       : Whether to save the plot. Default is True.
        ifshow        (bool)       : Whether to show the plot. Default is False.
    _______________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plot as a .jpg file and optionally displays it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Creates violin plots for each feature showing distribution shape.
        - Arranges subplots in a nrows x ncols grid.
        - Displays mean, Q1, Q3, and standard deviation as text annotations.
        - Saves the plot to the specified path with the given file name.
        - Handles input validation and file saving errors.
        - If fewer features than subplots, remaining subplots are hidden.
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
    
    if not isinstance(nrows, int) or not isinstance(ncols, int) or nrows <= 0 or ncols <= 0:
        raise TypeError("nrows and ncols must be positive integers.")
    
    # Check if we have enough subplots for all features
    total_subplots = nrows * ncols
    n_features     = features.shape[1]
    
    if n_features > total_subplots:
        raise ValueError(f"Not enough subplots ({total_subplots}) for all features ({n_features}). "
                        f"Increase nrows or ncols, or reduce number of features.")
    
    df_features = pd.DataFrame(features, columns=feature_names)
    fig, axes   = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, col in enumerate(feature_names):
        try:
            ax = axes_flat[i]
            
            # Violin plot
            parts = ax.violinplot([df_features[col].values], positions=[1], widths=0.5, 
                                  points      = num_points,
                                  showmeans   = False, 
                                  showmedians = False, 
                                  showextrema = False)
            
            # Style the violin plot
            for pc in parts['bodies']:
                pc.set_facecolor(violin_color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('silver')
                pc.set_linewidth(0.8)

            # Calculate statistics
            mean_val   = df_features[col].mean()
            median_val = df_features[col].median()
            q1_val     = df_features[col].quantile(0.25)
            q3_val     = df_features[col].quantile(0.75)
            std_val    = df_features[col].std()
            
            # Add markers for quartiles and median
            ax.scatter([1], [median_val], color='white', s=80, zorder=3, edgecolors='silver', linewidths=1.0)
            ax.plot([1, 1], [q1_val, q3_val], color='black', linewidth=2.5, zorder=2)
            
            # Add mean line
            ax.axhline(y=mean_val, color=violin_color, linestyle='--', linewidth=1.5, alpha=0.8)
            ax.text(1.1, mean_val, rf'$\mu=${mean_val:.2f}',
                    transform=ax.transData,
                    va='bottom', ha='left', fontsize=10)
            
            # Add ylabel only to leftmost plots
            if i % ncols == 0:
                ax.set_ylabel('Value', fontsize=12)
            
            ax.set_xticks([])
            ax.set_xlim(0.5, 1.5)

            # Feature name
            ax.text(0.02, 0.98, col,
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold',
                    verticalalignment='top')

            # Statistics text
            stats_text = f'Q1: {q1_val:.3f}\nQ3: {q3_val:.3f}\n$\sigma$ : {std_val:.3f}'
            ax.text(0.02, 0.85, stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left')

        except Exception as e:
            raise ValueError(f"Error with feature {col}: {e}")
    
    # Hide unused subplots
    for i in range(n_features, total_subplots):
        axes_flat[i].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    file_path = os.path.join(path_save, f"violinplot_analysis_{name_file}.jpg")
    
    if ifsave:
        try:
            os.makedirs(path_save, exist_ok=True)
            plt.savefig(file_path, bbox_inches="tight", dpi=600)
        except Exception as e:
            raise OSError(f"Could not save plot to {file_path}: {e}")

    if ifshow: plt.show()
    
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
            "pearson" : "Pearson Correlogram",
            "spearman": "Spearman Correlogram",
            "kendall" : "Kendall Correlogram"
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
            plt.savefig(file_path, bbox_inches="tight", dpi=600)
        except Exception as e:
            raise OSError(f"Could not save plot to {file_path}: {e}")
    
    if show: plt.show()
    plt.close(fig)

# Single simulation plot --------------------------------------------------------------------------------------------------#
def plot_simulation_example(df: pd.DataFrame, y_var: str = "massNew[Msun](10)",
                            x_var       : str = "time[Myr]",
                            y_label     : str = r'$M[\rm M_\odot]$',
                            x_label     : str = r'Time $[Myr]$',
                            norm_factor : Optional[float] = None,
                            save_path   : str = "./figures/",
                            t_cc        : Optional[float] = None, 
                            t_coll      : Optional[float] = None, 
                            t_relax     : Optional[float] = None, 
                            M_crit      : Optional[float] = None,
                            rho_half    : Optional[float] = None,
                            show        : bool = False):
    """
    _______________________________________________________________________________________________________________________
    Plots a single simulation example with optional metadata displayed in a text box.
    _______________________________________________________________________________________________________________________
    Parameters:
        df          (pd.DataFrame)    : DataFrame containing simulation data. Mandatory.
        y_var       (str)             : Column name for y-axis variable. Default is "massNew[Msun](10)".
        x_var       (str)             : Column name for x-axis variable. Default is "time[Myr]".
        y_label     (str)             : Label for y-axis. Default is mass in solar masses.
        x_label     (str)             : Label for x-axis. Default is time in Myr.
        norm_factor (Optional[float]) : Normalization factor for x-axis values. Optional.
        save_path   (str)             : Path to save the plot. Default is "./figures/".
        t_cc        (Optional[float]) : Core collapse time for metadata display. Optional.
        t_coll      (Optional[float]) : Collision time for metadata display. Optional.
        t_relax     (Optional[float]) : Relaxation time for metadata display. Optional.
        M_crit      (Optional[float]) : Critical mass for metadata display. Optional.
        rho_half    (Optional[float]) : Half-mass density for metadata display. Optional.
        show        (bool)            : Whether to display the plot. Default is False.
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    for col in [y_var, x_var]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Retrieve x and y values ---------------------------------------------------------------------------------------------#
    try:
        if norm_factor is not None:
            x_vals = df[x_var].values / norm_factor
        else:
            x_vals = df[x_var].values
        y_vals = df[y_var].values
    except Exception as e:
        raise RuntimeError(f"Error preparing data: {e}")

    # Create plot ---------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(x_vals, y_vals, lw=0.75, color="darkblue", marker=".", label='Most Massive Object (MMO)')
    ax.set_ylabel(y_label, size=14)
    ax.set_xlabel(x_label, size=14)
    ax.set_title("Simulation example", size=16, loc="left", pad=10)
    ax.legend(loc="best")

    # Add metadata text box if parameters are provided -------------------------------------------------------------------#
    if any(x is not None for x in [t_cc, t_coll, t_relax, M_crit, rho_half]):
        text_lines = []
        if t_cc is not None:
            text_lines.append(f"$t_{{\\rm{{cc}}}}={t_cc:.3f}$ Myr")
        if t_coll is not None:
            text_lines.append(f"$t_{{\\rm{{coll}}}}={t_coll:.3f}$ Myr")
        if t_relax is not None:
            text_lines.append(f"$t_{{\\rm{{relax}}}}={t_relax:.3f}$ Myr")
        if M_crit is not None:
            text_lines.append(f"$M_{{\\rm{{crit}}}}={M_crit:.2e}$ M$_\\odot$")
        if rho_half is not None:
            text_lines.append(f"$\\rho(R_h)={rho_half:.2e}$ pc$^{{-3}}$")
        
        text_content = "\n".join(text_lines)
        ax.text(0.01, 0.98, text_content,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

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
def dataset_2Dhist_comparison(x_base: np.ndarray, y_base: np.ndarray, x_aug : np.ndarray, y_aug : np.ndarray,
                              x_filt: np.ndarray, y_filt: np.ndarray,
                              name       : Optional[str] = "full",
                              bins       : int = 200,
                              cmap       : Union[str, Colormap ] = "viridis",
                              savepath   : Optional[str] = None,
                              figsize    : tuple = (18, 5),
                              titles     : tuple = ("Original Dataset", "Augmented Dataset", "Downsampled Dataset"),
                              axislabels : Tuple[str,str]= (r"log($t$ [Myr])", r"M$_{\rm{MMO}}$ [M$_\odot$]"),
                              cmap_label : str = "Count per bin",
                              show       : bool = False):
    """
    _______________________________________________________________________________________________________________________
    Generate 2D histogram comparison plots for three datasets (base, augmented, filtered) showing time-mass distributions.
    _______________________________________________________________________________________________________________________
    Parameters:
    -> x_base     (array-like) : Time values for base dataset. Mandatory.
    -> y_base     (array-like) : Mass values for base dataset. Mandatory.
    -> x_aug      (array-like) : Time values for augmented dataset. Mandatory.
    -> y_aug      (array-like) : Mass values for augmented dataset. Mandatory.
    -> x_filt     (array-like) : Time values for filtered dataset. Mandatory.
    -> y_filt     (array-like) : Mass values for filtered dataset. Mandatory.
    -> bins       (int)        : Number of bins for histograms. Default is 200.
    -> cmap       (str)        : Colormap for histograms. Default is "viridis".
    -> savepath   (str)        : Path to save the plot. Default is None (no saving).
    -> figsize    (tuple)      : Figure size (width, height). Default is (18, 5).
    -> titles     (tuple)      : Titles for the three subplots. Default is 
                                    ("Original Dataset", "Augmented Dataset", "Downsampled Dataset").
    -> cmap_label (str)        : Label for colorbar. Default is "Count per bin".
    _______________________________________________________________________________________________________________________
    Returns:
    -> None. The function displays the plot and optionally saves it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Converts time values to log10 1p scale for better visualization.
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
    if any(x is None for x in [x_base, y_base, x_aug, y_aug, x_filt, y_filt]):
        raise ValueError("All input arrays must not be None.")
    
    try:
        x_base, y_base  = np.asarray(x_base), np.asarray(y_base)
        x_aug, y_aug    = np.asarray(x_aug), np.asarray(y_aug)
        x_filt, y_filt  = np.asarray(x_filt), np.asarray(y_filt)
    
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")
    
    if not all(len(x) == len(y) for x, y in [(x_base, y_base), (x_aug, y_aug), (x_filt, y_filt)]):
        raise ValueError("Time and mass arrays must have the same length for each dataset.")
    
    if not isinstance(bins, int) or bins <= 0:
        raise TypeError("bins must be a positive integer.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")
    
    if not isinstance(titles, tuple) or len(titles) != 3:
        raise TypeError("titles must be a tuple of length 3.")

    # Data preparation ----------------------------------------------------------------------------------------------------#
    x_base, y_base = x_base.flatten(), y_base.flatten()
    x_aug, y_aug   = x_aug.flatten(), y_aug.flatten()
    x_filt, y_filt = x_filt.flatten(), y_filt.flatten()

    # Calculate histograms manually to share vmax
    H1, xedges, yedges = np.histogram2d(x_base, y_base, bins=bins)
    H2, _, _ = np.histogram2d(x_aug, y_aug, bins=[xedges, yedges])
    H3, _, _ = np.histogram2d(x_filt, y_filt, bins=[xedges, yedges])

    vmax = max(H1.max(), H2.max(), H3.max())
    norm = LogNorm(vmin=1, vmax=vmax)

    # Create figure and grid ----------------------------------------------------------------------------------------------#
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)

    datasets = [
        (x_base, y_base, H1, titles[0]),
        (x_aug, y_aug, H2, titles[1]),
        (x_filt, y_filt, H3, titles[2]),
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
        
        # Set x limits
        ax.set_xlim(left=0)
        
        # Set y limits to encompass all data
        y_max = max(y_base.max(), y_aug.max(), y_filt.max())
        ax.set_ylim(0, y_max+10000)
        
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
    
# Single 2D histogram plot ------------------------------------------------------------------------------------------------#
def dataset_2Dhist(x_values: np.ndarray, y_values: np.ndarray, name : Optional[str] = "dataset", bins : int = 200,
                   cmap       : Union[str, Colormap] = "viridis",
                   savepath   : Optional[str] = None,
                   figsize    : tuple = (8, 6),
                   title      : str = "Dataset",
                   axislabels : Tuple[str,str] = (r"log($t$ [Myr])", r"M$_{\rm{MMO}}$ [M$_\odot$]"),
                   cmap_label : str = "Count",
                   show       : bool = False):
    """
    _______________________________________________________________________________________________________________________
    Generate a 2D histogram plot for a single dataset showing time-mass distribution.
    _______________________________________________________________________________________________________________________
    Parameters:
    -> x_values     (array-like) : Time values for dataset. Mandatory.
    -> y_values     (array-like) : Mass values for dataset. Mandatory.
    -> name         (str)        : Name identifier for saving. Default is "dataset".
    -> bins         (int)        : Number of bins for histogram. Default is 200.
    -> cmap         (str)        : Colormap for histogram. Default is "viridis".
    -> savepath     (str)        : Path to save the plot. Default is None (no saving).
    -> figsize      (tuple)      : Figure size (width, height). Default is (8, 6).
    -> title        (str)        : Title for the plot. Default is "Dataset".
    -> axislabels   (tuple)      : Labels for x and y axes. Default is time and mass labels.
    -> cmap_label   (str)        : Label for colorbar. Default is "Count".
    -> show         (bool)       : Whether to display the plot. Default is False.
    _______________________________________________________________________________________________________________________
    Returns:
    -> None. The function displays the plot and optionally saves it.
    _______________________________________________________________________________________________________________________
    Notes:
        - Applies logarithmic normalization to color scale.
        - Displays point count for the dataset.
        - Handles input validation and file saving errors.
    _______________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if x_values is None or y_values is None:
        raise ValueError("Time and mass arrays must not be None.")
    
    try:
        x_values  = np.asarray(x_values)
        y_values = np.asarray(y_values)
    except Exception as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {e}")
    
    if len(x_values) != len(y_values):
        raise ValueError("Time and mass arrays must have the same length.")
    
    if not isinstance(bins, int) or bins <= 0:
        raise TypeError("bins must be a positive integer.")
    
    if not isinstance(figsize, tuple) or len(figsize) != 2:
        raise TypeError("figsize must be a tuple of length 2.")

    # Data preparation ----------------------------------------------------------------------------------------------------#
    x = x_values.flatten()
    y = y_values.flatten()

    # Calculate histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    vmax = H.max()
    norm = LogNorm(vmin=1, vmax=vmax)

    # Create figure -------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=figsize)
    
    h = ax.hist2d(x, y, bins=[xedges, yedges], cmap=cmap, norm=norm, cmin=1)
    
    ax.set_title(title, loc="left", size=14)
    ax.set_xlabel(axislabels[0], size=12)
    ax.set_ylabel(axislabels[1], size=12)

    # Colorbar (inset)
    cax = inset_axes(ax, width="50%", height="4%", loc="upper left", borderpad=1)
    cb = fig.colorbar(h[3], cax=cax, orientation='horizontal')
    cb.set_label(cmap_label, size=10)
    cb.ax.tick_params(labelsize=8)

    # Add count text
    n_points = len(x)
    ax.text(0.03, 0.75, f"$N_{{\\rm points}}$ = {n_points:,}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Save and display ----------------------------------------------------------------------------------------------------#
    if savepath:
        try:
            plt.savefig(f"{savepath}2dhist_{name}.jpg", dpi=600, bbox_inches="tight")
        except Exception as e:
            raise OSError(f"Error saving plot to {savepath}: {e}")
    
    if show: plt.show()
    plt.close(fig)
    
#Histograms and KDE for different features --------------------------------------------------------------------------------#
def plot_feature_distributions(feats_raw: pd.DataFrame, feats_processed: pd.DataFrame, labels: dict, cont_features: list, 
                               sample_size : int = 1e6,
                               bins        : Union[int, str] = 50,
                               save_dir    : str = "dist_aug_study"):
    """
    ________________________________________________________________________________________________________________________
    Plot histogram and KDE for each continuous feature (original vs augmented) using pure matplotlib.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> feats_raw      (pd.DataFrame) : Original features dataframe. Mandatory.
    -> feats_processed(pd.DataFrame) : Augmented/processed features dataframe. Mandatory.
    -> cont_features  (list)         : List of continuous feature names to plot. Mandatory.
    -> sample_size    (int)          : Number of points to sample from each dataframe for plotting. Default is 100000.
    -> save_dir       (str)          : Directory to save the figures. Default is "dist_aug_study".
    ________________________________________________________________________________________________________________________
    Returns:
        None. The function saves the plots as .png files in the specified directory.
    ________________________________________________________________________________________________________________________
    Notes:
        - Plots both histogram and Gaussian KDE for each feature, comparing original and augmented distributions.
        - Samples up to sample_size points from each dataframe for efficiency.
        - Handles input validation and file saving errors.
    ________________________________________________________________________________________________________________________
    Raises:
        ValueError, TypeError, OSError
    ________________________________________________________________________________________________________________________
    """
    
    # Set saving directory
    path = save_dir + "comparison_features/"
    os.makedirs(path, exist_ok=True)

    # Plot each feature ---------------------------------------------------------------------------------------------------#
    for idx, feature in enumerate(cont_features):
        
        raw_sample  = feats_raw[feature].dropna()
        proc_sample = feats_processed[feature].dropna()
        label       = labels.get(feature, feature)

        if len(raw_sample) > sample_size:
            raw_sample  = raw_sample.sample(int(sample_size), random_state=42)
        
        if len(proc_sample) > sample_size:
            proc_sample = proc_sample.sample(int(sample_size), random_state=42)

        # Histogram
        fig, ax = plt.subplots(figsize=(6, 4))

        # Original dataset
        counts_raw, bin_edges_raw, _ = ax.hist(raw_sample, bins=bins, color='darkcyan', alpha=0.4, density=True, 
                                               label='Original dataset')

        # Augmented dataset
        counts_proc, bin_edges_proc, _ = ax.hist(proc_sample, bins=bins, color='goldenrod', alpha=0.4, density=True, 
                                                 label='Training sample')

        # Gaussian KDE
        kde_raw  = gaussian_kde(raw_sample)
        kde_proc = gaussian_kde(proc_sample)

        x_min  = min(raw_sample.min(), proc_sample.min())
        x_max  = max(raw_sample.max(), proc_sample.max())
        x_vals = np.linspace(x_min, x_max, 500)

        ax.plot(x_vals, kde_raw(x_vals), color='darkslategrey', lw=2)
        ax.plot(x_vals, kde_proc(x_vals), color='darkorange', lw=2)

        ax.set_title(f'Distribution of {label}')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend()

        # Save figure
        save_path = os.path.join(path, f"feature{idx}.jpg")
        fig.tight_layout()
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

# Plot efficiency v mass ratio scatter plot -------------------------------------------------------------------------------#
def plot_efficiency_mass_ratio_dataset(
    data_dict                : Dict[str, Dict[str, Union[list, np.ndarray]]],
    figsize                  : tuple                                                         = (7, 5),
    title                    : Optional[str]                                                 = None,
    cmap_label               : Optional[str]                                                 = None,
    cmap                     : Optional[Union[str, Colormap]]                                = None,
    cmap_name                : Optional[str]                                                 = None,
    norm_mode                : Optional[Literal["linear", "log", "discrete", "categorical"]] = "linear",
    n_bins                   : Optional[int]                                                 = 6,
    log_vmin                 : Optional[float]                                               = None,
    include_fit_curve        : bool                                                          = True,
    include_fit_uncertainty  : bool                                                          = False,
    savepath                 : Optional[str]                                                 = None,
    show                     : bool                                                          = False
    ):
    # Fit from Vergara et al. (2025) --------------------------------------------------------------------------------------#
    def V2025_epsilon_BH(m_ratio: np.ndarray, k: float = 4.63, x0: float = 4.0, a: float = -0.1):
        X = np.log(m_ratio)
        return (1 + np.exp(-k * (X - x0)))**a

    # Prepare data --------------------------------------------------------------------------------------------------------#
    tags = list(data_dict.keys())

    # Collect all cmap values globally if a colormap is requested
    if cmap is not None:
        c_all = []
        for tag in tags:
            if cmap_name not in data_dict[tag]:
                raise ValueError(f"Group '{tag}' is missing but a cmap {cmap_name}.")
            c_all.append(np.asarray(data_dict[tag][cmap_name]))
        c_all = np.concatenate(c_all)

        cmap_obj      = cm.get_cmap(cmap)
        cb_ticks      = None
        cb_ticklabels = None
        cb_spacing    = "uniform"

        # Normalization set up for the colorbar --------------------------------------------------------------------------# 
        if norm_mode == "linear":
            norm = Normalize(vmin=c_all.min(), vmax=c_all.max())

        elif norm_mode == "log":
            if np.any(c_all <= 0):
                raise ValueError("LogNorm requires cmap_values > 0.")
            vmin = log_vmin if log_vmin is not None else np.percentile(c_all, 1)
            norm = LogNorm(vmin=vmin, vmax=c_all.max())

        elif norm_mode == "discrete":
            bounds     = np.linspace(c_all.min(), c_all.max(), n_bins + 1)
            norm       = BoundaryNorm(bounds, n_bins)
            cb_spacing = "proportional"

        elif norm_mode == "categorical":
            categories = np.unique(c_all)
            n_cat      = len(categories)
            cmap_obj   = cm.get_cmap(cmap, n_cat)

            cat_to_idx = {cat: i for i, cat in enumerate(categories)}
            bounds     = np.arange(-0.5, n_cat + 0.5, 1)
            norm       = BoundaryNorm(bounds, n_cat)

            cb_ticks      = np.arange(n_cat)
            cb_ticklabels = categories

            for tag in tags:
                vals = np.asarray(data_dict[tag]["cmap_values"])
                data_dict[tag]["_cmap_mapped"] = np.array([cat_to_idx[v] for v in vals])

        else:
            raise ValueError(f"Unknown norm_mode: {norm_mode}")

    # Figure --------------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(figsize=figsize)

    if title:
        ax.set_title(title, loc="left", fontsize=14)

    # Plot uncertainties in the model if requested ------------------------------------------------------------------------#
    if include_fit_uncertainty:
        
        # Define fit function
        def fit_func(m, k, x0, a):
            X = np.log(m)
            return (1 + np.exp(-k * (X - x0)))**a
                    
        try:
            
            # Load historical data from V25
            historical_path = os.path.join(os.path.dirname(__file__), "../../rawdata/historical/V25.txt")
            historical_data = pd.read_csv(historical_path)
            
            # Get mass_ratio and epsilon from historical data
            m_hist = historical_data['mass_ratio'].values
            eps_hist = historical_data['epsilon'].values
            
            # Bootstrap parameters
            n_bootstrap = 1000
            n_data = len(m_hist)
            
            # Create smooth x-axis for predictions
            xx_unc = np.logspace(-5, 4, 500)
            bootstrap_predictions = np.zeros((n_bootstrap, len(xx_unc)))
            
            # Bootstrap resampling and fitting
            np.random.seed(42)  # For reproducibility
            for i in range(n_bootstrap):
                # Resample with replacement
                indices  = np.random.randint(0, n_data, size=n_data)
                m_boot   = m_hist[indices]
                eps_boot = eps_hist[indices]
                
                # Fit the model to bootstrap sample
                try:
                    
                    # Fit with initial guess from original parameters
                    # Bounds explanation:
                    # k  > 0.1  : Steepness must be positive (avoid flat sigmoid)
                    # x0 unbounded : Inflection point can be anywhere in log-space
                    # -0.5 < a < 0 : Power must be negative (for decreasing sigmoid), but not too extreme
                    popt, _ = curve_fit(fit_func, m_boot, eps_boot, 
                                       p0     = [4.63, 4.0, -0.1],
                                       maxfev = 5000,
                                       bounds = ([0.1, -np.inf, -0.5], [20.0, np.inf, -0.01]))
                    
                    # Predict with fitted parameters
                    bootstrap_predictions[i, :] = fit_func(xx_unc, *popt)
                    
                except:
                    # If fit fails, flag as nan to exclude from percentile calculation
                    bootstrap_predictions[i, :]  = np.nan
            
            # Remove any bootstrap samples where the fit failed (nan values)            
            bootstrap_predictions = bootstrap_predictions[~np.isnan(bootstrap_predictions).any(axis=1)]

            # Calculate percentiles for confidence bands: 1-sigma: 68% CI -> 16th to 84th percentile
            lower_1sigma = np.percentile(bootstrap_predictions, 16, axis=0)
            upper_1sigma = np.percentile(bootstrap_predictions, 84, axis=0)
            
            # Calculate percentiles for confidence bands: 2-sigma: 95% CI -> 2.5th to 97.5th percentile
            lower_2sigma = np.percentile(bootstrap_predictions, 2.5, axis=0)
            upper_2sigma = np.percentile(bootstrap_predictions, 97.5, axis=0)
            
            # Calculate percentiles for confidence bands: 3-sigma: 99.7% CI -> 0.15th to 99.85th percentile
            lower_3sigma = np.percentile(bootstrap_predictions, 0.15, axis=0)
            upper_3sigma = np.percentile(bootstrap_predictions, 99.85, axis=0)
            
            # Plot uncertainty bands (from outermost to innermost)
            ax.fill_between(xx_unc, lower_3sigma, upper_3sigma, color='olive', alpha=0.15, 
                          label=r'1$\sigma$ / 2$\sigma$ / 3$\sigma$ Confidence')
            ax.fill_between(xx_unc, lower_2sigma, upper_2sigma, color='olive', alpha=0.25)
            ax.fill_between(xx_unc, lower_1sigma, upper_1sigma, color='olive', alpha=0.35)
            
        except Exception as e:
            print(f"Warning: Could not compute fit uncertainty: {e}")
    
    # Scatter plots (one per tag) inside the dictionary -------------------------------------------------------------------#
    last_scatter = None

    for tag in tags:

        # Retrieve group
        group = data_dict[tag]

        # Set variables
        x = np.asarray(group["mass_ratio"])
        y = np.asarray(group["epsilon"])

        # Retrieve configuration
        marker    = group.get("marker", "o")
        color     = group.get("color", None)
        edgecolor = group.get("edgecolor", "none")
        s         = group.get("s", 30)
        label     = group.get("label", tag)

        if cmap is None:
            sc = ax.scatter(x, y, s=s, marker=marker, c=color, edgecolor=edgecolor,label=label+f" ({len(x)})")
        else:
            if norm_mode == "categorical":
                cvals = group["_cmap_mapped"]
            else:
                cvals = np.asarray(group[cmap_name])

            sc = ax.scatter(x, y, s=s, marker=marker, c=cvals, cmap=cmap_obj, norm=norm, edgecolor=edgecolor, 
                            label=label+f" ({len(x)})")

        last_scatter = sc  # for colorbar handle

    # Plot fit curve if requested -----------------------------------------------------------------------------------------#
    if include_fit_curve:
        xx = np.logspace(-5, 4)
        ax.plot(xx, V2025_epsilon_BH(xx), "--", color="black", lw=1.0, label="Fit from V+25b")

    # Axes Configuration --------------------------------------------------------------------------------------------------#
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e4)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlabel(r"$M_{\rm tot}/M_{\rm crit}$", fontsize=12)
    ax.set_ylabel(r"$\epsilon_{\rm BH}$", fontsize=12)

    # Colorbar (only if cmap is provided) ---------------------------------------------------------------------------------#
    if cmap is not None:
        cax = inset_axes(ax, width="50%", height="4%", loc="upper left", borderpad=1)

        cb = fig.colorbar(last_scatter, cax=cax, orientation="horizontal", spacing=cb_spacing, ticks=cb_ticks)

        if cb_ticklabels is not None:
            cb.set_ticklabels(cb_ticklabels)

        cb.set_label(cmap_label, fontsize=10)
        cb.ax.tick_params(labelsize=8)

    # Legend --------------------------------------------------------------------------------------------------------------#
    ax.legend(loc="lower right", fontsize=8)

    # Save / Show ---------------------------------------------------------------------------------------------------------#
    if savepath:
        namefile = f"efficiency_vs_mass_ratio_{cmap_name}.jpg" if cmap_name else "efficiency_vs_mass_ratio.jpg"
        plt.savefig(savepath + namefile, dpi=700, bbox_inches="tight")

    if show: plt.show()

    plt.close(fig)
    
#--------------------------------------------------------------------------------------------------------------------------#