# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing            import Optional, Union
from pandas.api.types  import CategoricalDtype

# Custom utilities and functions ------------------------------------------------------------------------------------------#
from src.utils.phyfactors  import relaxation_time, core_collapse_time, collision_time, crossing_time
from src.utils.phyfactors  import rho_at_rh, critical_mass, safronov_num, mean_stellar_radius

# Define relevant operations for desired tabular features -----------------------------------------------------------------#
def tabular_features(process_df: pd.DataFrame, names:list, return_names:bool=True, onehot:bool=True,
                     eps_logscale_all_range     : float = 0,
                     eps_logscale_limited_range : float = 1e-6
                     ) -> pd.DataFrame:
    """
    ________________________________________________________________________________________________________________________
    Generation of a dataframe with relevant features of interest for ML exploration
    ________________________________________________________________________________________________________________________
    Parameters:
    -> process_df                 (pd.DataFrame) : Dataframe with raw data to process. Mandatory.
    -> names                      (list)         : List of feature names to include. If None, all features are included. 
                                                   Optional.
    -> return_names               (bool)         : Whether to return feature labels along with the DataFrame. Default is 
                                                   True.
    -> onehot                     (bool)         : Whether to apply one-hot encoding to categorical features. Default is 
                                                   True.
    -> eps_logscale_all_range     (float)        : Small constant to add to all values before log transformation to avoid 
                                                   log(0). Default is 0.
    -> eps_logscale_limited_range (float)        : Small constant to add to all values before log transformation to avoid 
                                                   log(0). Default is 1e-6.
    ________________________________________________________________________________________________________________________
    Returns: 
    -> result_df (pd.DataFrame)  : DataFrame with the selected and processed features.
    -> feats_labels (dict)       : Dictionary mapping feature names to their labels (only if return_names is True).
    ________________________________________________________________________________________________________________________
    Notes:
    - The function applies predefined operations to create new features based on the input DataFrame.
    - If 'names' is provided, only the specified features (or their expanded one-hot encoded columns) will be included in 
      the final DataFrame.
    - All features come with a logarithmic transformation in base 10. This should be adecuaded based on the dataset
      objective, as logscale can contribute errors for non strictly positive values.
    - eps_logscale_all_range is set to 0 by default, assuming the input DataFrame has been filtered to remove t=0
      rows and near-zero denominators (via filter_simulation_artifacts or equivalent preprocessing at the source).
      If unfiltered data is used, set this to a small positive value to avoid log(0) errors.
    - eps_logscale_limited_range is set to 1e-6 as a small constant added to all values before log transformation to avoid 
      log(0) errors, but where the variable has a limited range of possible values (e.g., between 0 and 1).
      Should be set based on the dataset and the expected range of values for the features.
    ________________________________________________________________________________________________________________________
    """
    # Set defaul types for categorical features ---------------------------------------------------------------------------#
    type_sim_dtype = CategoricalDtype(categories=[0., 1.], ordered=False)

    # Set possible features and possible names with nested operations -----------------------------------------------------#
    default_feats = {
        "log(t/t_cc)" :{
            "label"     : r"$\log(t/t_{\rm{cc}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_cc'] + eps_logscale_all_range)
        },
        "log(t/t_cross)" :{
            "label"     : r"$\log(t/t_{\rm{cross}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_cross'] + eps_logscale_all_range)
        },
        "log(t/t_relax)" :{
            "label"     : r"$\log(t/t_{\rm{relax}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_relax'] + eps_logscale_all_range)
        },
        "log(t)" :{
            "label"     : r"$\log(t)$",
            "operation" : lambda df: np.log10(df['t'] + eps_logscale_all_range)
        },
        "log(t_coll)" :{
            "label"     : r"$\log(t_{\rm{coll}})$",
            "operation" : lambda df: np.log10(df['t_coll'] + eps_logscale_all_range)
        },
        "log(rho(R_h))" :{
            "label"     : r"$\log(\rho(R_{h}))$",
            "operation" : lambda df: np.log10(df['rho(R_h)'] + eps_logscale_all_range)
        },
        "log(M_tot/M_crit)" :{
            "label"     : r"$\log(M_{\rm tot}/M_{\rm crit})$",
            "operation" : lambda df:np.log10(df['M_tot']/df['M_crit'] + eps_logscale_all_range)
        },
        "log(M_MMO/M_tot)" :{
            "label"     : r"$\log(M_{\rm MMO}/M_{\rm tot})$",
            "operation" : lambda df: np.log10(df['M_MMO']/df['M_tot'] + eps_logscale_limited_range)
        },
        "log(R_h/R_core)" :{
            "label"     : r"$\log(R_{h}/R_{\rm{core}})$",
            "operation" : lambda df: np.log10((df['R_h'] / df['R_core']) + eps_logscale_all_range)
        },
        "log(R_tid/R_core)" :{
            "label"     : r"$\log(R_{\rm{tidal}}/R_{\rm{core}})$",
            "operation" : lambda df: np.log10((df['R_tid'] / df['R_core']) + eps_logscale_all_range)
        },
        "log(Z)":{
            "label"     : r"$\log(Z)$",
            "operation" : lambda df: np.log10(df['z'] + eps_logscale_all_range)
        },
        "fracbin":{
            "label"     : r"$f_{\rm{bin}}$",
            "operation" : lambda df: df['fracbin']
        },
        "type_sim" :{
            "label"     : r"environment",
            "operation" : lambda df: (pd.get_dummies(df["type_sim"].astype(type_sim_dtype),
                                                     prefix = "type_sim"))}
        }
    
    # Apply operations and create new columns -----------------------------------------------------------------------------#
    result_df    = process_df.copy()
    feats_labels = {}  

    for feature_name, feature_info in default_feats.items():
        try:
            new_feature = feature_info['operation'](process_df)

            if isinstance(new_feature, pd.DataFrame):
                # Expand dummy columns
                for col in new_feature.columns:
                    result_df[col]    = new_feature[col]
                    feats_labels[col] = f"{feature_info['label']} ({col})"
            else:
                result_df[feature_name]    = new_feature
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

# Filter numerical artifacts and undefined states from simulation data ----------------------------------------------------#
def filter_simulation_artifacts(raw_df                    : pd.DataFrame,
                                columns_to_check          : Optional[list] = None,
                                min_denominator_threshold : float          = 1e-10,
                                filter_null_mass          : bool           = True,
                                filter_initial_state      : bool           = True,
                                verbose                   : bool           = True
                                ) -> pd.DataFrame:
    """
    ________________________________________________________________________________________________________________________
    Remove numerical artifacts and physically undefined states from a simulation DataFrame before feature engineering.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> raw_df                    (pd.DataFrame) : Raw simulation DataFrame, same format as input to tabular_features.
    -> columns_to_check          (list)         : List of column names to check for near-zero values as denominators in feature ratios. 
                                                  If None, defaults to ['t_cc', 't_cross', 't_relax
    -> min_denominator_threshold (float)        : Minimum absolute value for columns used as denominators in feature 
                                                  ratios. Rows where any denominator falls below this threshold are 
                                                  removed as numerical artifacts. Default is 1e-10.
    -> filter_null_mass          (bool)         : Whether to remove rows where the mass is null. Default is True.
    -> filter_initial_state      (bool)         : Whether to remove rows where t=0 (initial state). At t=0 all 
                                                  temporal ratios are undefined (log(-inf)). Default is True.
    -> verbose                   (bool)         : Whether to print a summary of removed rows. Default is True.
    ________________________________________________________________________________________________________________________
    Returns:
    -> clean_df (pd.DataFrame) : DataFrame with numerical artifacts and undefined states removed.
    ________________________________________________________________________________________________________________________
    Notes:
    - Columns checked as denominators are: ['t_cc', 't_cross', 't_relax', 'M_crit', 'R_core', 'M_tot'].
      These correspond exactly to the denominators used in tabular_features ratio computations.
    - Rows where t=0 are removed because all log(t/t_scale) features are undefined at the initial state.
      This is a conceptual singularity, not a numerical artifact.
    - A column is only checked if it exists in the input DataFrame. Missing columns are skipped silently.
    - The returned index is reset to ensure contiguous integer indexing.
    ________________________________________________________________________________________________________________________
    """
    n_initial   = len(raw_df)
    removal_log = {}

    # Make a working copy to accumulate masks without modifying input
    valid_mask = pd.Series(True, index=raw_df.index)

    # Filter 1: t=0 (physically undefined initial state) ------------------------------------------------------------------#
    if filter_initial_state and ('t' in raw_df.columns):
        
        # Identify rows where t=0 and count them
        t_zero_mask  = raw_df['t'] == 0
        n_t_zero    = t_zero_mask.sum()
        
        # Log the number of rows removed due to t=0
        removal_log['t=0 (initial state)'] = int(n_t_zero)
        
        # Create mask to exclude t=0 rows
        valid_mask = valid_mask & ~t_zero_mask
    
    # Filter 2: null mass (simulation with no massive object) -------------------------------------------------------------#
    if filter_null_mass and ('M_MMO' in raw_df.columns):
        
        # Identify rows where M_MMO=0 and count them
        null_mass_mask = raw_df['M_MMO'] == 0
        n_null_mass    = null_mass_mask.sum()
        
        # log the number of rows removed due to null mass
        removal_log['null mass (M_MMO=0)'] = int(n_null_mass)
        
        # Create mask to exclude rows with null mass
        valid_mask = valid_mask & ~null_mass_mask

    # Filter 2: near-zero denominators (numerical artifacts) --------------------------------------------------------------#
    default_columns     = ['t_cc', 't_cross', 't_relax', 'M_crit', 'R_core', 'M_tot']
    denominator_columns = columns_to_check if columns_to_check is not None else default_columns

    for col in denominator_columns:
        if col not in raw_df.columns:
            continue
        # Identify rows where the absolute value of the denominator is below the threshold and only count rows
        artifact_mask  = raw_df[col].abs() < min_denominator_threshold
        n_artifacts    = (artifact_mask & valid_mask).sum()  
        
        # Log the number of rows removed due to near-zero denominators for this column
        removal_log[f'{col} ~ 0 (denominator artifact)'] = int(n_artifacts)
        
        # Create mask to exclude rows with near-zero denominators
        valid_mask = valid_mask & ~artifact_mask

    # --- Apply mask and report ------------------------------------------------------------------------------------------#
    clean_df  = raw_df[valid_mask].reset_index(drop=True)
    n_removed = n_initial - len(clean_df)

    if verbose:
        print(f"{'─'*64}")
        print(f"  filter_simulation_artifacts: {n_initial} rows in → {len(clean_df)} rows out")
        print(f"  Total removed : {n_removed} ({100 * n_removed / n_initial:.2f}%)")
        if n_removed > 0:
            print(f"  Breakdown:")
            for reason, count in removal_log.items():
                if count > 0:
                    print(f"    [{count:>6}]  {reason}")
        print(f"{'─'*64}")

    return clean_df

# Categorize the simulations given their IMBH formation channel ----------------------------------------------------------#
def determine_formation_channel(imbh_df: pd.DataFrame, mass_column_name: str, time_column_name: str,
                                minimum_bh_mass    : float = 100,
                                minimum_time_thres : float = 20
                                ) -> str:
    """
    ________________________________________________________________________________________________________________________
    Classify a simulation based on IMBH formation channel using formation time criteria
    ________________________________________________________________________________________________________________________
    Parameters:
    -> imbh_df    (pd.DataFrame)  : DataFrame containing IMBH formation data with mass and time information. Mandatory.
    -> mass_column_name (str)     : Name of the column in imbh_df representing IMBH mass. Mandatory.
    -> time_column_name (str)     : Name of the column in imbh_df representing IMBH formation time. Mandatory.
    -> minimun_bh_mass (float)    : Minimum mass threshold to consider IMBH formation [Msun]. Default is 100 Msun.
    -> minimun_time_thres (float) : Time threshold to classify formation channel [Myr]. Default is 20 Myr.
    ________________________________________________________________________________________________________________________
    Returns:
    -> chform  (str) : Classification of the formation channel ("FAST", "SLOW").
    ________________________________________________________________________________________________________________________
    Notes:
        Classification criteria based on IMBH formation time:
        - "FAST"  : Early formation of a massive central object (T_bh(>100Msun) ≤ 20 Myr)
                    
                    Gravitationally bound BH subsystem form early on in the cluster evolution. IMBH can rapidly form via 
                    dynamical interactions between single and binary BHs. In this FAST scenario for IMBH formation, 
                    the large cluster density (and escape velocity) serves to drive the growth of the most massive
                    BH in the system. The FAST scenario might occur more commonly in the NSCs of low-mass galaxies than in 
                    GCs.
                    The FAST scenario is initiated practically from the very beginning of our simulations, has a very high 
                    accretion rate and requires extreme densities of around 10^8 Msun/pc^3
        
        - "SLOW"  : Late formation of a massive central object  (T_bh(>100Msun) > 20 Myr)
                    
                    BH mass growth due solely to dynamical interactions and mass transfer from binary companions. 
                    In this SLOW scenario, IMBH formation happens at late times in the cluster evolution, typically during
                    the post-core-collapse phase of evolution. The SLOW scenario is initiated at later times in the cluster 
                    evolution, has a small accretion rate and requires modest cluster densities of around 10^5 Msun/pc^3.
    
    References:
    -> https://ui.adsabs.harvard.edu/link_gateway/2015MNRAS.454.3150G/doi:10.1093/mnras/stv2162
    ________________________________________________________________________________________________________________________
    """   
    try:
        
        mass_time = imbh_df[imbh_df[mass_column_name] > minimum_bh_mass].iloc[0][time_column_name]

    except Exception as e:
        raise ValueError("Could not determine formation channel. Check if IMBH formed in the simulation.")
    
    if (mass_time <= minimum_time_thres):
        chform = "FAST" 
    
    else:
        chform = "SLOW" 
    
    return chform

# Compute physical parameters derived of quantities from the simulations --------------------------------------------------#
def compute_physical_parameters(time_values : np.ndarray, rh_values: Union[float, np.ndarray], 
                                v_disp_values    : Union[float, np.ndarray],
                                n_values         : Union[int, np.ndarray],
                                m_mean_values    : Union[float, np.ndarray],
                                m_max_values     : Union[float, np.ndarray],
                                comp_stellar_val : bool = False,
                                ) -> dict:
    """
    ________________________________________________________________________________________________________________________
    Compute physical parameters for star cluster evolution.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> time_values      (np.ndarray)         [Myr]  : Array of time values
    -> rh_values        (float | np.ndarray) [pc]   : Array of half-mass radius values
    -> v_disp_values    (float | np.ndarray) [km/s] : Array of velocity dispersion values
    -> n_values         (int | np.ndarray)   [#]    : Array of number of stars values
    -> m_mean_values    (float | np.ndarray) [Msun] : Array of mean stellar mass values
    -> m_max_values     (float | np.ndarray) [Msun] : Array of maximum stellar mass values
    -> comp_stellar_val (bool)                      : Whether to compute stellar radius (default False)
    ________________________________________________________________________________________________________________________
    Returns:
        Dictionary with all computed features as arrays:
            - 'stellar_radius' : Mean stellar radius values [Rsun] if comp_stellar_val is True, else default to 1.0 [Rsun]
            - 'mcrit'          : Critical mass values       [Msun]
            - 'safnum'         : Safronov number values     [#]
            - 'trelax'         : Relaxation time values     [Myr]
            - 'tcc'            : Core collapse time values  [Myr]
            - 'tcoll'          : Collision time values      [Myr]
            - 'tcross'         : Crossing time values       [Myr]
            - 'rho_half'       : Number density values      [pc^-3]
    ________________________________________________________________________________________________________________________
    """
    # Ensure all inputs are arrays with positive time values
    time_values = np.maximum(np.asarray(time_values), 1e-6)
    
    # Retrieve or set default stellar radius
    if comp_stellar_val:
        stellar_radius = mean_stellar_radius(m_mean_values)[0]
    else:
        stellar_radius = 1.0  
    
    # Compute all features using vectorized functions
    results = {
        'stellar_radius' : stellar_radius,
        'mcrit'          : critical_mass(rh_values, m_mean_values, time_values, v_disp_values, stellar_radius)[0],
        'safnum'         : safronov_num(m_mean_values, v_disp_values, stellar_radius)[0],
        'trelax'         : relaxation_time(n_values, rh_values, v_disp_values)[0],
        'tcc'            : core_collapse_time(m_mean_values, m_max_values, n_values, rh_values, v_disp_values)[0],
        'tcoll'          : collision_time(rh_values, n_values, m_mean_values, v_disp_values, stellar_radius)[0],
        'tcross'         : crossing_time(rh_values, v_disp_values)[0],
        'rho_half'       : rho_at_rh(n_values, rh_values)[0]
    }
    
    return results

# Store the transformation of the target to retrieve the physical value ---------------------------------------------------#
class TargetLogScaler:
    """
    ________________________________________________________________________________________________________________________
    TargetLogScaler: transformation and inverse transformation of the target variable in a safe logspace.
    ________________________________________________________________________________________________________________________
    -> Current transformation: 
        y = log10(target/norm_factor + epsilon)
    -> Inverse transformation: 
        target = (10^y - epsilon) * norm_factor
    ________________________________________________________________________________________________________________________
    Notes:
    This Scaler works if the target variable its strictly positive and has was intended to be used on a variable restricted
    from 0 to 1. The epsilon parameter is a small constant added to avoid log(0) errors, and should be set based on the 
    expected range of values for the target variable.
    ________________________________________________________________________________________________________________________
    """
    # Initialization ------------------------------------------------------------------------------------------------------#
    def __init__(self, norm_factor: Optional[np.ndarray], epsilon: float = 1e-6):
        """
        ____________________________________________________________________________________________________________________
        Parameters:
        -> norm_factor : Array of normalization factors for scaling
        -> epsilon     : Small constant to avoid log(0)
        ____________________________________________________________________________________________________________________
        """
        self.norm_factor = norm_factor
        self.epsilon     = epsilon
    
    # Inverse transformation ----------------------------------------------------------------------------------------------#
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Transform from log-space predictions back to target variable.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> y: Predictions in log-space (log10(target/norm_factor + epsilon))
        ____________________________________________________________________________________________________________________    
        Returns:
            target: Target variable in original scale, ensuring non-negative values by clipping at 0.
        ____________________________________________________________________________________________________________________
        """
        # Clip y to a safe range before exponentiation to avoid float64 overflow (max ~10^308)
        y_safe = np.clip(np.asarray(y, dtype=np.float64), -300.0, 300.0)

        # If no normalization factor, just apply transformation
        if self.norm_factor is None:
            target = np.clip(np.power(10, y_safe) - self.epsilon, 0, None)
        
        else:
            target = (np.power(10, y_safe) - self.epsilon) * self.norm_factor
        
        return np.clip(target, 0, None)  
    
    # Forward transformation ----------------------------------------------------------------------------------------------#
    def transform(self, target: np.ndarray) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Transform from target variable to log-space.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> target: Target variable in original scale
        ____________________________________________________________________________________________________________________    
        Returns:
            y: Transformed values in log-space
        ____________________________________________________________________________________________________________________
        """
        if self.norm_factor is None:
            y = np.log10(target + self.epsilon)
        else:
            y = np.log10(target / self.norm_factor + self.epsilon) 
        
        return y
    
    # Representation for logging and debugging ----------------------------------------------------------------------------#
    def __repr__(self):
        return f"TargetLogScaler(norm_factor={len(self.norm_factor)}, epsilon={self.epsilon})"
    
#--------------------------------------------------------------------------------------------------------------------------#