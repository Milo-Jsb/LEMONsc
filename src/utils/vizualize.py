# Modules -----------------------------------------------------------------------------------------------------------------#
import math
import os

import numpy             as np
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from sklearn.metrics     import mean_squared_error
from scipy.stats         import gaussian_kde

# Custom Correlation plot with density color map --------------------------------------------------------------------------#
def corr_plot(predictions: np.ndarray, true_values: np.ndarray, path_save: str, name_file: str, model_name:str):
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

    # Save and show plot -------------------------------------------------------------------------------------------------#
    file_path = os.path.join(path_save, f"corr_plot_{name_file}.jpg")
    try:
        os.makedirs(path_save, exist_ok=True)  
        plt.savefig(file_path, bbox_inches="tight", dpi=900)
    except Exception as e:
        raise OSError(f"Could not save plot to {file_path}: {e}")
    
    plt.show()
    plt.close(fig)
#-------------------------------------------------------------------------------------------------------------------------#