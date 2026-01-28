# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np 
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Union

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
    # Convert to numpy arrays for computation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate residuals
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    
    # Apply Huber loss formula
    huber = np.where(abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta))
    
    # Return positive mean loss for minimization (standard behavior)
    return np.mean(huber)

#--------------------------------------------------------------------------------------------------------------------------#