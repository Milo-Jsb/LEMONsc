# Modules -----------------------------------------------------------------------------------------------------------------#
import warnings

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing       import List, Union
from scipy.signal import savgol_filter

# Internal parameters -----------------------------------------------------------------------------------------------------#
window_length  = 15   # window_length    (Savgol filter)
polyorder      = 2    # polynomial order (Savgol filter)
sigma_error    = 0.01 # Relative gaussian noise for data augmentation

# Prepare mass evolution --------------------------------------------------------------------------------------------------#
def target_preparation(mass_evolution: pd.Series) -> np.ndarray:
    """
    _______________________________________________________________________________________________________________________
    Create the target vector for a regression problem. (Preprocess supported through scipy and numpy)
    _______________________________________________________________________________________________________________________
    Parameters:
    -> mass_evolution (pd.Series) : Raw target. Assuming data in format to preprocess. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
    -> tgt (np.ndarray) : Target evolution ready to implement in a regression problem.
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(mass_evolution, pd.Series):
        raise TypeError("mass_evolution must be a pandas Series.")

    if len(mass_evolution) < window_length:
        raise ValueError(f"mass_evolution must contain at least {window_length} points for smoothing.")

    # Direct mass estimation (smoothed) -----------------------------------------------------------------------------------#
    try:
        tgt = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder, mode='nearest')
        tgt = np.clip(tgt, 0, None)

    # Fallback ------------------------------------------------------------------------------------------------------------#
    except Exception as e: raise RuntimeError(f"Unexpected error in target preparation. {e}")

    return tgt

# Prepare time evolution --------------------------------------------------------------------------------------------------#
def time_preparation(time_evolution: pd.Series) -> np.ndarray:
    """
    _______________________________________________________________________________________________________________________
    Prepare time evolution vector for regression problems. 
    _______________________________________________________________________________________________________________________
    Parameters:
    -> time_evolution (pd.Series): Raw time series. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
    -> time (np.ndarray) : Time evolution ready to implement in a regression problem.
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(time_evolution, pd.Series):
        raise TypeError("time_evolution must be a pandas Series.")

    # Convert to numpy array ----------------------------------------------------------------------------------------------#
    try:
        time = time_evolution.to_numpy()
        time = np.clip(time, 0, None)

    except Exception as e:
        raise RuntimeError(f"Unexpected error in time preparation. {e}")

    return time

# Apply gaussian noise to a selected variable -----------------------------------------------------------------------------#
def apply_noise(value: Union[float, np.ndarray, List[float]], sigma: float = sigma_error) -> Union[float, np.ndarray]:
    """
    ________________________________________________________________________________________________________________________
    Helper function that creates a gaussian noise for a single tabular value or array of values (Default to 1sigma)
    ________________________________________________________________________________________________________________________
    Parameters:
    -> value (Union[float, np.ndarray, List[float]]) : Input value(s) to which noise will be applied. Mandatory.
    -> sigma (float)                                 : Relative standard deviation of the gaussian noise. 
                                                       Default is 0.01 (1%).
    ________________________________________________________________________________________________________________________
    Returns:
    -> noised_value (Union[float, np.ndarray]) : Value(s) after applying gaussian noise.
    ________________________________________________________________________________________________________________________
    Notes:
    -> The noise is generated as a normal distribution with mean 0 and standard deviation equal to the input value 
       multiplied by sigma.
    -> The function ensures that the noised value does not become negative (all our parameters are non-negative, for 
       different datasets should be handled accordingly). If the noised value is negative or artificially set to 0, it 
       returns the original value instead.
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    value_array  = np.array(value)

    if np.any(value_array < 0):
        warnings.warn("apply_noise() received negative values. The function is designed for non-negative inputs; "
                      "results for negative elements may not behave as intended.", UserWarning, stacklevel=2)

    # Apply gaussian noise ------------------------------------------------------------------------------------------------#
    noised_array = value_array + np.random.normal(loc=0, scale=np.abs(value_array) * sigma)

    return np.where(noised_array > 0, noised_array, value_array)

#--------------------------------------------------------------------------------------------------------------------------#