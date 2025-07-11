# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing       import Optional
from scipy.signal import savgol_filter, medfilt

# Prepare mass evolution --------------------------------------------------------------------------------------------------#
def target_preparation(mass_evolution: pd.Series, time_evolution: Optional[pd.Series], norm_factor: Optional[float], 
                       target_type: str, log10_scale:bool = False) -> np.ndarray:
    
    """
    _______________________________________________________________________________________________________________________
    Create the target vector for a regression problem. (Preprocess supported through scipy and numpy)
    _______________________________________________________________________________________________________________________
    Parameters:
        mass_evolution (pd.Series)           : Raw target. Assuming data in format to preprocess. Mandatory.
        time_evolution (Optional[pd.Series]) : If "dM/dt", clean time evolution must be provided with same dimensions as 
                                               the mass evolution (assuming an irregular timestep). Optional.
        norm_factor    (Optional[float])     : If provided, the output is normalized by dividing by this value. Optional.
        target_type    (str)                 : Type of target to prepare. Options are "M", "dM" and "dM/dt". Mandatory.
        log10_scale    (bool)                : Scale the target using np.log1p().
    _______________________________________________________________________________________________________________________
    Returns:
        tgt (np.ndarray) : Target evolution ready to implement in a regression problem.
    _______________________________________________________________________________________________________________________
    Processing steps:
        - For "M"      : Applies Savitzky-Golay smoothing (window_length=15, polyorder=2) to the mass series.
        - For "dM"     : Smooths mass, computes first difference, applies median filter (kernel_size=5), 
                         clips to non-negative, then applies log1p transformation.
        - For "dM/dt"  : Smooths mass, computes time and mass differences, replaces zero time steps with a small value to 
                         avoid division by zero, computes rate, clips to non-negative.
        - For all      : Optionally normalizes the result by dividing by norm_factor if provided. Optionally scale the 
                         target using log10 scale.
    _______________________________________________________________________________________________________________________
    Notes:
        - Input series must have at least 15 points (for smoothing).
        - For "dM/dt", time_evolution must be provided and have the same length as mass_evolution.
        - For "dM/dt", time steps of zero are replaced with a small value (1e-10) to avoid division by zero.
        - NaNs or Infs in the output will raise errors.
        - Raises TypeError, ValueError for invalid input types, values, or unsupported target_type.
        - Raises RuntimeError for unexpected internal errors.
    _______________________________________________________________________________________________________________________
    """
    
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(mass_evolution, pd.Series):
        raise TypeError("mass_evolution must be a pandas Series.")

    if time_evolution is not None:
        if not isinstance(time_evolution, pd.Series):
            raise TypeError("time_evolution must be a pandas Series or None.")
        if len(time_evolution) != len(mass_evolution):
            raise ValueError("time_evolution and mass_evolution must have the same length.")

    if norm_factor is not None:
        if not isinstance(norm_factor, (float, int)):
            raise TypeError("norm_factor must be a float, int, or None.")
        if norm_factor <= 0:
            raise ValueError("norm_factor must be positive.")

    if target_type not in {"M", "dM", "dM/dt"}:
        raise ValueError(f"Invalid target_type: '{target_type}'. Choose from 'M', 'dM', 'dM/dt'.")

    if len(mass_evolution) < 15:
        raise ValueError("mass_evolution must contain at least 15 points for smoothing.")

    # Direct mass estimation (smoothed) -----------------------------------------------------------------------------------#
    if target_type == "M":
        tgt = savgol_filter(mass_evolution.to_numpy(), window_length=15, polyorder=2)
        tgt = np.clip(tgt, 0, None)

        if norm_factor is not None:
            tgt *= (1 / norm_factor)
        
        if log10_scale:
            tgt = np.log10(tgt + 1)

        return tgt

    # Change in mass estimation -------------------------------------------------------------------------------------------#
    elif target_type == "dM":
        mass  = savgol_filter(mass_evolution.to_numpy(), window_length=15, polyorder=2)
        dmass = np.diff(mass, prepend=mass[0])

        dmass = medfilt(dmass, kernel_size=5)
        tgt   = np.clip(dmass, 0, None)

        if np.any(np.isnan(tgt)):
            raise ValueError("NaNs encountered in log1p(dM) — check input values.")

        if norm_factor is not None:
            tgt *= (1 / norm_factor)
        
        if log10_scale:
            tgt = np.log10(tgt + 1)

        return tgt

    # Growth rate estimation ----------------------------------------------------------------------------------------------#
    elif target_type == "dM/dt":
        if time_evolution is None:
            raise ValueError("time_evolution must be provided for 'dM/dt' target.")

        mass = savgol_filter(mass_evolution.to_numpy(), window_length=15, polyorder=2)
        time = time_evolution.to_numpy()

        dt = np.diff(time, prepend=time[0])
        dt[dt == 0] = 1e-10  

        dmass = np.diff(mass, prepend=mass[0])
        rate  = dmass / dt
        tgt   = np.clip(rate, 0, None)

        if np.any(np.isnan(tgt)) or np.any(np.isinf(tgt)):
            raise ValueError("NaNs or Infs encountered in dM/dt computation — check time steps.")

        if norm_factor is not None:
            tgt *= (1 / norm_factor)
        
        if log10_scale:
            tgt = np.log10(tgt + 1)

        return tgt

    # Fallback 
    raise RuntimeError("Unexpected error in target preparation.")

# Prepare time evolution --------------------------------------------------------------------------------------------------#
def time_preparation(time_evolution: pd.Series, norm_factor: Optional[float] = None, return_diff: bool = False) -> np.ndarray:
    
    """
    _______________________________________________________________________________________________________________________
    Prepare time evolution vector for regression problems. (Simple normalization supported)
    _______________________________________________________________________________________________________________________
    Parameters:
        time_evolution (pd.Series)       : Raw time series. Mandatory.
        norm_factor    (Optional[float]) : If provided, the output is normalized by dividing by this value. Optional.
        return_diff    (bool)            : If True, the difference between the time steps is returned. Optional.
    _______________________________________________________________________________________________________________________
    Returns:
        time (np.ndarray) : Time evolution ready to implement in a regression problem.
        diff (np.ndarray) : Difference between the time steps if return_diff is True.
    _______________________________________________________________________________________________________________________
    Processing steps:
        - Converts pandas Series to numpy array.
        - Optionally normalizes the result by dividing by norm_factor if provided.
    _______________________________________________________________________________________________________________________
    Notes:
        - Input validation ensures time_evolution is a pandas Series.
        - If norm_factor is provided, it must be positive.
        - Raises TypeError, ValueError for invalid input types or values.
    _______________________________________________________________________________________________________________________
    """
    
    # Input validation ----------------------------------------------------------------------------------------------------#
    if not isinstance(time_evolution, pd.Series):
        raise TypeError("time_evolution must be a pandas Series.")

    if norm_factor is not None:
        if not isinstance(norm_factor, (float, int)):
            raise TypeError("norm_factor must be a float, int, or None.")
        if norm_factor <= 0:
            raise ValueError("norm_factor must be positive.")

    # Convert to numpy array ----------------------------------------------------------------------------------------------#
    time = time_evolution.to_numpy()

    # Normalize if requested ----------------------------------------------------------------------------------------------#
    if norm_factor is not None:
        time *= (1 / norm_factor)

    # Return difference between the time steps if requested ----------------------------------------------------------------#
    if return_diff:
        diff = np.diff(time, prepend=time[0])
        return time, diff
    
    else:
        return time

#--------------------------------------------------------------------------------------------------------------------------#