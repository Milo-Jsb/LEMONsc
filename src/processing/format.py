# Modules -----------------------------------------------------------------------------------------------------------------#
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
        - mass_evolution (pd.Series) : Raw target. Assuming data in format to preprocess. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
        - tgt (np.ndarray) : Target evolution ready to implement in a regression problem.
    _______________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(mass_evolution, pd.Series):
        raise TypeError("mass_evolution must be a pandas Series.")

    if len(mass_evolution) < window_length:
        raise ValueError(f"mass_evolution must contain at least {window_length} points for smoothing.")

    # Direct mass estimation (smoothed) -----------------------------------------------------------------------------------#
    try:
        tgt = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder)
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
        time_evolution (pd.Series): Raw time series. Mandatory.
    _______________________________________________________________________________________________________________________
    Returns:
        time (np.ndarray) : Time evolution ready to implement in a regression problem.
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
    _______________________________________________________________________________________________________________________
    Helper function that creates a gaussian noise for a single tabular value or array of values (Default to 1sigma)
    _______________________________________________________________________________________________________________________
    Parameters:
        - value (Union[float, np.ndarray, List[float]]) : Input value(s) to which noise will be applied. Mandatory.
        - sigma (float)                                 : Relative standard deviation of the gaussian noise. 
                                                          Default is 0.01 (1%).
    _______________________________________________________________________________________________________________________
    Returns:
        - noised_value (Union[float, np.ndarray]) : Value(s) after applying gaussian noise.
    _______________________________________________________________________________________________________________________
    """
    value_array  = np.array(value)
    noised_array = value_array + np.random.normal(loc=0, scale=value_array * sigma)
    
    return np.maximum(noised_array, 0)

#--------------------------------------------------------------------------------------------------------------------------#