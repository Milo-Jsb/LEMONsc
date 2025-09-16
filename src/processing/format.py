# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from collections  import defaultdict
from typing       import Optional, List
from scipy.signal import savgol_filter, medfilt

# Internal parameters -----------------------------------------------------------------------------------------------------#
window_length  = 15 # window_length    (Savgol filter)
polyorder      = 2  # polynomial order (Savgol filter)
medfilt_kernel = 5  # Kernel size      (median filter)
        
# Fix the name tag of the target based on preparation ---------------------------------------------------------------------#
def __get_target_name(base: str, norm_target: bool, log10_target: bool) -> List[str]:
    """Helper function to generate target names based on normalization and scaling options."""
    if norm_target:
        return [f"{base}/M_tot"]
    elif log10_target:
        return [f"log({base})"]
    return [base]

# Helper function for normalization and log scaling -----------------------------------------------------------------------#
def __normalize_and_logscale(arr: np.ndarray, norm_factor: Optional[float] = None, log10_scale: bool = False) -> np.ndarray:
    """Normalize and/or log-scale an array"""
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
    ) -> np.ndarray:
    """
    _______________________________________________________________________________________________________________________
    Create the target vector for a regression problem. (Preprocess supported through scipy and numpy)
    _______________________________________________________________________________________________________________________
    Parameters:
        - mass_evolution (pd.Series)           : Raw target. Assuming data in format to preprocess. Mandatory.
        - time_evolution (Optional[pd.Series]) : If "dM/dt", clean time evolution must be provided with same dimensions as 
                                                 the mass evolution (assuming an irregular timestep). Optional.
        - norm_factor    (Optional[float])     : If provided, the output is normalized by dividing by this value. Optional.
        - target_type    (str)                 : Type of target to prepare. Options are "point_mass", "delta_mass" and 
                                                 "mass_rate". Mandatory.
        - log10_scale    (bool)                : Scale the target using np.log1p().
    _______________________________________________________________________________________________________________________
    Returns:
        - tgt (np.ndarray) : Target evolution ready to implement in a regression problem.
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
        tgt = __normalize_and_logscale(tgt, norm_factor, log10_scale)
        return tgt

    # Change in mass estimation -------------------------------------------------------------------------------------------#
    elif target_type == "delta_mass":
        mass  = savgol_filter(mass_evolution.to_numpy(), window_length=window_length, polyorder=polyorder)
        dmass = np.diff(mass, prepend=mass[0])
        dmass = medfilt(dmass, kernel_size=medfilt_kernel)
        tgt   = np.clip(dmass, 0, None)
        if np.any(np.isnan(tgt)):
            raise ValueError("NaNs encountered in log1p(dM) — check input values.")
        tgt = __normalize_and_logscale(tgt, norm_factor, log10_scale)
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
        tgt = __normalize_and_logscale(tgt, norm_factor, log10_scale)
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

# Filter and downsize a dataset based in a 2D histogram -------------------------------------------------------------------#
def filter_and_downsample_hist2d(x: np.ndarray, y: np.ndarray, H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, 
                                 min_count: int = 10, 
                                 max_count: int = 100, 
                                 seed     : Optional[int] = None
                                ) -> np.ndarray:
    """
    ________________________________________________________________________________________________________________________
    Filter and downsample a dataset based on 2D histogram binning.
    ________________________________________________________________________________________________________________________
    Parameters:
        - x          (np.ndarray)        : X-coordinates of the points. Mandatory.
        - y          (np.ndarray)        : Y-coordinates of the points. Mandatory.
        - H          (np.ndarray)        : 2D histogram array. Mandatory.
        - xedges     (np.ndarray)        : X-axis bin edges. Mandatory.
        - yedges     (np.ndarray)        : Y-axis bin edges. Mandatory.
        - min_count  (int)               : Minimum count threshold. Points in bins with fewer points are discarded. 
        - max_count  (int)               : Maximum count threshold. Points in bins with more points are downsampled.
        - seed       (Optional[int])     : Random seed for reproducible downsampling. 
    ________________________________________________________________________________________________________________________
    Returns:
        - selected_indices (np.ndarray) : Array of indices corresponding to the selected points.
    ________________________________________________________________________________________________________________________
    Notes:
        Points in bins with accumulation < min_count are discarded. For bins with accumulation > max_count, random 
        downsampling is performed. Between min_count and max_count, all points are kept.
    ________________________________________________________________________________________________________________________
    """
    # Input validation ----------------------------------------------------------------------------------------------------#
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if min_count < 0:
        raise ValueError("min_count must be non-negative.")
    if max_count <= 0:
        raise ValueError("max_count must be positive.")
    if min_count >= max_count:
        raise ValueError("min_count must be less than max_count.")
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Asignar cada punto a su bin -----------------------------------------------------------------------------------------#
    x_bin = np.digitize(x, xedges) - 1
    y_bin = np.digitize(y, yedges) - 1

    valid = (x_bin >= 0) & (x_bin < H.shape[0]) & (y_bin >= 0) & (y_bin < H.shape[1])

    # Agrupar puntos por bin
    bin_points = defaultdict(list)
    for i in np.where(valid)[0]:
        bin_id = (x_bin[i], y_bin[i])
        bin_points[bin_id].append(i)

    selected_indices = []

    for bin_id, indices in bin_points.items():
        count = len(indices)
        if count < min_count:
            continue 
        elif count > max_count:
            sampled = np.random.choice(indices, size=max_count, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)

    selected_indices = np.array(selected_indices)
    return selected_indices
       
#--------------------------------------------------------------------------------------------------------------------------#
def tabular_features(process_df: pd.DataFrame, names:list, return_names:bool=True, onehot:bool=True) -> pd.DataFrame:

    # Set possible features and possible names with nested operations -----------------------------------------------------#
    default_feats = {
        "M_MMO/M_tot" :{
            "label"     : r"$M_{\rm{MMO}}/M_{\rm{tot}}$",
            "operation" : lambda df: df['M_MMO'] / df['M_tot']
             },
        "log(M_MMO)" :{
            "label"     : r"$\log(M_{\rm{MMO}})$",
            "operation" : lambda df: np.log10(df['M_MMO'] + 1)
        },
        "t/t_coll" :{
            "label"     : r"$t/t_{\rm{coll}}$",
            "operation" : lambda df: df['t'] / df['t_coll']
        },
        "log(t_coll/t_cc)" :{
            "label"     : r"$\log(t_{\rm{coll}}/t_{\rm cc})$",
            "operation" : lambda df: np.log10((df['t_coll']/df['t_cc']) + 1)
        },
        "log(t)" :{
            "label"     : r"$\log(t)$",
            "operation" : lambda df: np.log10(df['t']+1)
        },
        "log(rho(R_h))" :{
            "label"     : r"$\log(\rho(R_{h}))$",
            "operation" : lambda df: np.log10(df['rho(R_h)'] + 1)
        },
        "M_tot/M_crit" :{
            "label"     : r"$M_{\rm tot}/M_{\rm crit}$",
            "operation" : lambda df: df['M_tot'] / df['M_crit']
        },
        "log(R_h/R_core)" :{
            "label"     : r"$\log(R_{h}/R_{\rm{core}})$",
            "operation" : lambda df: np.log10((df['R_h'] / df['R_core']) + 1)
        },
        "type_sim" :{
            "label"     : r"environment",
            "operation" : lambda df: pd.get_dummies(df['type_sim'], prefix="type_sim") if onehot else df['type_sim'].astype('category')
        }}

    # Apply operations and create new columns -----------------------------------------------------------------------------#
    result_df = process_df.copy()
    feats_labels = {}  # dict: column_name -> label

    for feature_name, feature_info in default_feats.items():
        try:
            new_feature = feature_info['operation'](process_df)

            if isinstance(new_feature, pd.DataFrame):
                # Expand dummy columns
                for col in new_feature.columns:
                    result_df[col] = new_feature[col]
                    feats_labels[col] = f"{feature_info['label']} ({col})"
            else:
                result_df[feature_name] = new_feature
                feats_labels[feature_name] = feature_info['label']

        except KeyError as e:
            print(f"Warning: Column {e} not found for feature {feature_name}")
        except Exception as e:
            print(f"Error processing feature {feature_name}: {e}")
        

    # Apply operations to create new features -----------------------------------------------------------------------------#
    if names is not None:
        filtered_columns = []
        for name in names:
            # check for exact match or expanded dummy columns
            matching = [col for col in result_df.columns if col == name or col.startswith(name)]
            filtered_columns.extend(matching)
        result_df = result_df[filtered_columns]

    # Return DataFrame (compatible) + labels for logging
    if return_names:
        return result_df, feats_labels
    else:
        return result_df

#--------------------------------------------------------------------------------------------------------------------------#