# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing       import Optional
from scipy.signal import savgol_filter, medfilt

# Helper function for normalization and log scaling -----------------------------------------------------------------------#
def _normalize_and_logscale(arr: np.ndarray, norm_factor: Optional[float] = None, log10_scale: bool = False) -> np.ndarray:
    """
    ________________________________________________________________________________________________________________________
    Normalize and/or log-scale an array.
    ________________________________________________________________________________________________________________________
    """
    result = arr.copy()
    if norm_factor is not None:
        result = result / norm_factor
    if log10_scale:
        result = np.log10(result + 1)
    return result

# Prepare mass evolution --------------------------------------------------------------------------------------------------#
def target_preparation(mass_evolution: pd.Series, time_evolution: Optional[pd.Series], norm_factor: Optional[float], 
    target_type    : str,
    log10_scale    : bool = False,
    window_length  : int = 15,
    polyorder      : int = 2,
    medfilt_kernel : int = 5
    ) -> np.ndarray:
    """
    _______________________________________________________________________________________________________________________
    Create the target vector for a regression problem. (Preprocess supported through scipy and numpy)
    _______________________________________________________________________________________________________________________
    Parameters:
        mass_evolution (pd.Series)           : Raw target. Assuming data in format to preprocess. Mandatory.
        time_evolution (Optional[pd.Series]) : If "dM/dt", clean time evolution must be provided with same dimensions as 
                                               the mass evolution (assuming an irregular timestep). Optional.
        norm_factor    (Optional[float])     : If provided, the output is normalized by dividing by this value. Optional.
        target_type    (str)                 : Type of target to prepare. Options are "point_mass", "delta_mass" and 
                                               "mass_rate". Mandatory.
        log10_scale    (bool)                : Scale the target using np.log1p().
        window_length  (int)                 : Window length for Savitzky-Golay smoothing. Default 15.
        polyorder      (int)                 : Polynomial order for Savitzky-Golay smoothing. Default 2.
        medfilt_kernel (int)                 : Kernel size for median filter. Default 5.
    _______________________________________________________________________________________________________________________
    Returns:
        tgt (np.ndarray) : Target evolution ready to implement in a regression problem.
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

    if target_type not in {"point_mass", "delta_mass", "mass_rate"}:
        raise ValueError(f"Invalid target_type: '{target_type}'. Choose from 'point_mass', 'mass_rate', 'delta_mass'.")

    if len(mass_evolution) < window_length:
        raise ValueError(f"mass_evolution must contain at least {window_length} points for smoothing.")

    # Direct mass estimation (smoothed) -----------------------------------------------------------------------------------#
    if target_type == "point_mass":
        tgt = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder)
        tgt = np.clip(tgt, 0, None)
        tgt = _normalize_and_logscale(tgt, norm_factor, log10_scale)
        return tgt

    # Change in mass estimation -------------------------------------------------------------------------------------------#
    elif target_type == "delta_mass":
        mass  = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder)
        dmass = np.diff(mass, prepend=mass[0])
        dmass = medfilt(dmass, kernel_size=medfilt_kernel)
        tgt   = np.clip(dmass, 0, None)
        if np.any(np.isnan(tgt)):
            raise ValueError("NaNs encountered in log1p(dM) — check input values.")
        tgt = _normalize_and_logscale(tgt, norm_factor, log10_scale)
        return tgt

    # Growth rate estimation ----------------------------------------------------------------------------------------------#
    elif target_type == "mass_rate":
        if time_evolution is None:
            raise ValueError("time_evolution must be provided for 'dM/dt' target.")
        mass = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder)
        time = time_evolution.to_numpy()
        dt = np.diff(time, prepend=time[0])
        dt[dt == 0] = 1e-10  
        dmass = np.diff(mass, prepend=mass[0])
        rate  = dmass / dt
        tgt   = np.clip(rate, 0, None)
        if np.any(np.isnan(tgt)) or np.any(np.isinf(tgt)):
            raise ValueError("NaNs or Infs encountered in dM/dt computation — check time steps.")
        tgt = _normalize_and_logscale(tgt, norm_factor, log10_scale)
        return tgt

    # Fallback 
    raise RuntimeError("Unexpected error in target preparation.")

# Prepare time evolution --------------------------------------------------------------------------------------------------#
def time_preparation(time_evolution: pd.Series, norm_factor: Optional[float] = None, 
    return_diff: bool = False
    ) -> np.ndarray:
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
        time = time / norm_factor
    # Return difference between the time steps if requested ----------------------------------------------------------------#
    if return_diff:
        diff = np.diff(time, prepend=time[0])
        return time, diff
    else:
        return time
#--------------------------------------------------------------------------------------------------------------------------#