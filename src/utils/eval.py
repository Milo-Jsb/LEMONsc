# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing   import Dict, List, Optional, Union
from loguru   import logger

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
    -> XGBoost equivalent: objective='reg:pseudohubererror', huber_slope=delta
    -> LightGBM equivalent: objective='fair', fair_c=delta
    ________________________________________________________________________________________________________________________
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residual     = y_true - y_pred
    abs_residual = np.abs(residual)

    huber = np.where(abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta))

    return float(np.mean(huber))

# Regression metrics computation -----------------------------------------------------------------------------------------#
def compute_metrics(y_true  : Union[np.ndarray, pd.Series], 
                    y_pred  : Union[np.ndarray, pd.Series], 
                    metrics : Optional[List[str]] = None,
                    verbose : bool                = False) -> Dict[str, float]:
    """
    ________________________________________________________________________________________________________________________
    Compute regression evaluation metrics between true and predicted values.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> y_true  (Union[np.ndarray, pd.Series]) : True target values.
    -> y_pred  (Union[np.ndarray, pd.Series]) : Predicted target values.
    -> metrics (Optional[List[str]])           : List of metric names to compute. 
                                                 If None, computes all available metrics.
                                                 Supported: "mse", "rmse", "mae", "r2", "huber", "mape".
    -> verbose (bool)                          : Whether to log the computed metrics (default=False).
    ________________________________________________________________________________________________________________________
    Returns:
    -> Dict[str, float]: Dictionary mapping metric names to their computed values.
    ________________________________________________________________________________________________________________________
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

    # Default metrics
    if metrics is None:
        metrics = ["mse", "rmse", "mae", "r2", "huber"]

    # Available metric computations
    residuals  = y_true - y_pred
    ss_res     = np.sum(residuals ** 2)
    ss_tot     = np.sum((y_true - np.mean(y_true)) ** 2)

    available = {
        "mse"   : lambda: float(np.mean(residuals ** 2)),
        "rmse"  : lambda: float(np.sqrt(np.mean(residuals ** 2))),
        "mae"   : lambda: float(np.mean(np.abs(residuals))),
        "r2"    : lambda: float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "huber" : lambda: huber_loss(y_true, y_pred),
        "mape"  : lambda: float(np.mean(np.abs(residuals / np.where(np.abs(y_true) > 1e-10, y_true, 1e-10))) * 100),
    }

    results = {}
    for m in metrics:
        m_lower = m.lower()
        if m_lower in available:
            results[m_lower] = available[m_lower]()
        else:
            logger.warning(f"Unknown metric '{m}' — skipping.")

    if verbose:
        logger.info("Computed regression metrics:")
        for name, value in results.items():
            logger.info(f"  - {name}: {value:.6f}")

    return results
#--------------------------------------------------------------------------------------------------------------------------#