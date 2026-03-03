# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np 
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing          import Union, Optional, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

# Constants ---------------------------------------------------------------------------------------------------------------#
DEFAULT_REGRESSION_METRICS = ["r2", "mae", "mse", "rmse"]
        
# Custom metrics ----------------------------------------------------------------------------------------------------------#
def huber_loss(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], delta: float = 1.0) -> float:
    """
    ________________________________________________________________________________________________________________________
    Calculate Huber loss between true and predicted values.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> y_true (Union[np.ndarray, pd.Series]) : True target values.
    -> y_pred (Union[np.ndarray, pd.Series]) : Predicted target values.  
    -> delta  (float)                        : Threshold parameter for Huber loss (default=1.0)
    ________________________________________________________________________________________________________________________
    Returns:
    -> float: Huber loss value (lower is better, use direction="minimize")
    ________________________________________________________________________________________________________________________
    Notes:
    -> Huber loss is less sensitive to outliers than squared error
    -> For |y_true - y_pred| <= delta: uses squared error (0.5 * residual^2)
    -> For |y_true - y_pred| > delta: uses linear error (delta * (|residual| - 0.5 * delta))
    -> Returns positive loss value for minimization (standard behavior)
    ________________________________________________________________________________________________________________________
    """
    # Convert to numpy arrays for computation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate residuals
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    
    # Apply Huber loss formula
    huber = np.where(abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta))
    
    # Return positive mean loss for minimization (standard behavior)
    result = np.mean(huber)
    
    # Guard against inf/nan caused by extreme predictions during hyperparameter search
    return float(result) if np.isfinite(result) else float('inf')


# Regression metrics computation ------------------------------------------------------------------------------------------#
def compute_metrics(y_true  : Union[np.ndarray, pd.Series],
                    y_pred  : Union[np.ndarray, pd.Series],
                    metrics : Optional[List[str]] = None,
                    verbose : bool = False
                    ) -> Dict[str, float]:
    """
    ________________________________________________________________________________________________________________________
    Compute regression metrics between true and predicted values.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> y_true   (Union[np.ndarray, pd.Series]) : True target values.
    -> y_pred   (Union[np.ndarray, pd.Series]) : Predicted target values.
    -> metrics  (Optional[List[str]])          : List of metric names to compute.
                                                 Supported: "r2", "mae", "mse", "rmse".
                                                 Defaults to DEFAULT_REGRESSION_METRICS = ["r2", "mae", "mse", "rmse"].
    -> verbose  (bool)                         : If True, print a warning for unknown metric names (default=False).
    ________________________________________________________________________________________________________________________
    Returns:
    -> Dict[str, float]: Dictionary mapping metric names to their computed float values.
    ________________________________________________________________________________________________________________________
    Notes:
    -> All values are cast to float for consistent serialisation (e.g. yaml, json).
    -> Unknown metric names are silently skipped unless verbose=True.
    ________________________________________________________________________________________________________________________
    """
    if metrics is None:
        metrics = DEFAULT_REGRESSION_METRICS

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results: Dict[str, float] = {}

    for metric in metrics:
        m = metric.lower()
        if   m == "mse"  : results["mse"]  = float(mean_squared_error(y_true, y_pred))
        elif m == "rmse" : results["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        elif m == "mae"  : results["mae"]  = float(mean_absolute_error(y_true, y_pred))
        elif m == "r2"   : results["r2"]   = float(r2_score(y_true, y_pred))
        else:
            if verbose:
                print(f"Warning: Unknown metric '{metric}' ignored")

    return results

#--------------------------------------------------------------------------------------------------------------------------#