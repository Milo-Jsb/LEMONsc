# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing                import Optional, Union, List, Dict, Any
from src.utils.phyfactors  import relaxation_time, core_collapse_time, collision_time, crossing_time
from src.utils.phyfactors  import rho_at_rh, critical_mass
from src.processing.format import apply_noise

# Define relevant operations for desired tabular features -----------------------------------------------------------------#
def tabular_features(process_df: pd.DataFrame, names:list, return_names:bool=True, onehot:bool=True) -> pd.DataFrame:
    """
    _______________________________________________________________________________________________________________________
    Generation of a Dataframe with Relevant Elements of interest for ML exploration
    _______________________________________________________________________________________________________________________
    Parameters:
        - process_df (pd.DataFrame) : Dataframe with raw data to process. Mandatory.
        - names      (list)         : List of feature names to include. If None, all features are included. Optional.
        - return_names (bool)       : Whether to return feature labels along with the DataFrame. Default is True.
        - onehot     (bool)         : Whether to apply one-hot encoding to categorical features. Default is True.
    _______________________________________________________________________________________________________________________
    Returns: 
        - result_df (pd.DataFrame)  : DataFrame with the selected and processed features.
    _______________________________________________________________________________________________________________________
    """
    # Set possible features and possible names with nested operations -----------------------------------------------------#
    default_feats = {
        "M_MMO/M_tot" :{
            "label"     : r"$M_{\rm{MMO}}/M_{\rm{tot}}^{10\%}$",
            "operation" : lambda df: df['M_MMO'] / (0.1* df['M_tot'])
             },
        "t/t_cc" :{
            "label"     : r"$\log(t/t_{\rm{cc}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_cc']+1)
        },
        "log(t_coll/t_relax)" :{
            "label"     : r"$\log(t_{\rm coll}/t_{\rm relax})$",
            "operation" : lambda df: np.log10((df['t_coll']/df['t_relax']) + 1)
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
        "Z":{
            "label"     : r"$Z$",
            "operation" : lambda df: np.log10(df['z'])
        },
        "fracbin":{
            "label"     : r"$f_{\rm{bin}}$",
            "operation" : lambda df: df['fracbin']
        },
        "type_sim" :{
            "label"     : r"environment",
            "operation" : lambda df: pd.get_dummies(df['type_sim'], prefix="type_sim") if onehot else df['type_sim'].astype('category')
        }}

    # Apply operations and create new columns -----------------------------------------------------------------------------#
    result_df    = process_df.copy()
    feats_labels = {}  # dict: column_name -> label

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
    
# Retrieve elements from the simulation and compute relevant features ------------------------------------------------------#
def compute_cluster_features(system_df: pd.DataFrame, imbh_df: pd.DataFrame, iconds_dict : Optional[Dict[str, Any]]=None, 
                             noise     : bool          = False, 
                             sim_env   : Optional[str] = None,
                             temp_evol : bool          = False) -> Dict[str, Union[float, str]]:
    """
    ________________________________________________________________________________________________________________________
    Compute relevant cluster features from simulation data 
    ________________________________________________________________________________________________________________________
    Parameters:
        - system_df   (pd.DataFrame)             : DataFrame with system evolution data. Mandatory.
        - imbh_df     (pd.DataFrame)             : DataFrame with IMBH formation
        - iconds_dict (Optional[Dict[str, Any]]) : Dictionary with initial conditions. Optional.
        - noise       (bool)                     : Whether to apply noise to the features. Default is False.
        - sim_env     (Optional[str])            : Predefined simulation environment type. If None, it will be determined.
                                                   Default is None.
        - temp_evol   (bool)                     : Whether to compute features as temporal evolution (arrays) or single
                                                   values. Default is False
    ________________________________________________________________________________________________________________________
    Returns:
        - features_dict (Dict[str, Union[float, str]]) : Dictionary with computed cluster features.
    ________________________________________________________________________________________________________________________
    """
    # Extract base values from the simulation data ------------------------------------------------------------------------#
    try: 
        if temp_evol:
            base_values = {
                'tau'    : system_df["tphys"].values,
                'rh'     : system_df["r_h"].values,
                'v_disp' : system_df["vc"].values,
                'm_tot'  : system_df["smt"].values,
                'n'      : system_df["nt"].values,
                'm_mean' : system_df["atot"].values,
                'm_max'  : system_df["smsm"].values,
                'cr'     : system_df["rc"].values
            }
            
        else:
            base_values = {
                'tau'    : system_df["tphys"].iloc[-1],
                'rh'     : system_df["r_h"].iloc[0],
                'v_disp' : system_df["vc"].iloc[0],
                'm_tot'  : system_df["smt"].iloc[0],
                'n'      : system_df["nt"].iloc[0],
                'm_mean' : system_df["atot"].iloc[0],
                'm_max'  : system_df["smsm"].iloc[0],
                'cr'     : system_df["rc"].iloc[0]
            }

        if iconds_dict:
            z_val       = float(iconds_dict["zini"])
            fracbin_val = float(iconds_dict.get("fracb", 0.0))

            if temp_evol:
                # Repeat scalar values to match temporal dimension
                n_timesteps            = len(system_df)
                base_values["z"]       = np.full(n_timesteps, z_val)
                base_values["fracbin"] = np.full(n_timesteps, fracbin_val)
            else:
                base_values["z"]       = z_val
                base_values["fracbin"] = fracbin_val

    except Exception as e:
        raise ValueError(f"Error extracting base cluster features: {e}")
    
    # Apply noise if required, no consideration for "n" As it represents a count of elements in the cluster.
    if noise:
        for key in base_values:
            if key != "n":
                base_values[key] = apply_noise(base_values[key], sigma=0.01)

    # Compute derived features using physical formulas -------------------------------------------------------------------#
    derived_values = compute_physical_parameters(
                        time_values    = base_values["tau"],
                        rh_values      = base_values['rh'],
                        v_disp_values  = base_values['v_disp'],
                        m_tot_values   = base_values['m_tot'],
                        n_values       = base_values['n'],
                        m_mean_values  = base_values['m_mean'],
                        m_max_values   = base_values['m_max'],
                        stellar_radius = 1.0)
            
    # Type of the environment of the simulation (repeat if the simulation its already classified)
    if (sim_env is not None):  env = sim_env
    
    # Determine the formation channel based on initial core density and time of IMBH formation if first seen
    else: env = determine_formation_channel(system_df, imbh_df, None)

   
    return {**base_values, **derived_values, 'type_sim': env}

# Categorize the simulations given their IMBH formation channel ----------------------------------------------------------#
def determine_formation_channel(system_df: pd.DataFrame, imbh_df: pd.DataFrame, n_virtual: Optional[int]=None):
    """
    ________________________________________________________________________________________________________________________
    Classify a simulation based on IMBH formation channel using initial core density and formation time criteria
    ________________________________________________________________________________________________________________________
    Parameters:
        - system_df  (pd.DataFrame)  : DataFrame containing system evolution data with core density information. Mandatory.
        - imbh_df    (pd.DataFrame)  : DataFrame containing IMBH formation data with mass and time information. Mandatory.
        - n_virtual  (Optional[int]) : Number of virtual simulations for tracking purposes. Default is None.
    ________________________________________________________________________________________________________________________
    Returns:
        - chform  (str) : Classification of the formation channel ("FAST", "SLOW", or "STEADY").
    ________________________________________________________________________________________________________________________
    Notes:
        Classification criteria based on initial core density and IMBH formation time:
        - "FAST"  : High initial density (≥ 1e7) and early formation (≤ 50 Myr)
        - "SLOW"  : Low initial density (≤ 1e6) and late formation (≥ 500 Myr)  
        - "STEADY": All other cases (intermediate density/time combinations)
        
        Requires system_df to have "roc" column (core density) and imbh_df to have "massNew[Msun](10)" and "time[Myr]" 
        columns for IMBH mass and formation time respectively.
    ________________________________________________________________________________________________________________________
    """
    
    try:
        i_density = system_df["roc"].iloc[0]
        mass_time = imbh_df[imbh_df['massNew[Msun](10)'] > 100].iloc[0]['time[Myr]']
    
    except Exception as e:
        raise ValueError("Could not determine formation channel. Check if IMBH formed in the simulation.")
    
    if (i_density >= 1e7) and (mass_time <= 50):
        chform  = "FAST" 

    elif (i_density <= 1e6) and (mass_time >= 500):
        chform  = "SLOW" 
    
    else:
        chform  = "STEADY"
    
    return chform

# Compute physical parameters derived of quantities from the simulations --------------------------------------------------#
def compute_physical_parameters(time_values : np.ndarray, rh_values: Union[float, np.ndarray], 
                                v_disp_values  : Union[float, np.ndarray],
                                n_values       : Union[int, np.ndarray],
                                m_mean_values  : Union[float, np.ndarray],
                                m_max_values   : Union[float, np.ndarray],
                                stellar_radius : float = 1.0,
                                ) -> dict:
    """
    ________________________________________________________________________________________________________________________
    Compute physical parameters for star cluster evolution.
    ________________________________________________________________________________________________________________________
    Parameters:
        time_values    (np.ndarray)         [Myr]  : Array of time values
        rh_values      (float | np.ndarray) [pc]   : Array of half-mass radius values
        v_disp_values  (float | np.ndarray) [km/s] : Array of velocity dispersion values
        m_tot_values   (float | np.ndarray) [Msun] : Array of total mass values (not used directly)
        n_values       (int | np.ndarray)   [#]    : Array of number of stars values
        m_mean_values  (float | np.ndarray) [Msun] : Array of mean stellar mass values
        m_max_values   (float | np.ndarray) [Msun] : Array of maximum stellar mass values
        stellar_radius (float)              [Rsun] : Stellar radius (default 1.0)
        return_units   (bool)                      : Whether to return results as astropy Quantities with units.
                                                     Default is False.
    ________________________________________________________________________________________________________________________
    Returns:
        Dictionary with all computed features as arrays:
            - 'mcrit'    : Critical mass values      [Msun]
            - 'trelax'   : Relaxation time values    [Myr]
            - 'tcc'      : Core collapse time values [Myr]
            - 'tcoll'    : Collision time values     [Myr]
            - 'tcross'   : Crossing time values      [Myr]
            - 'rho_half' : Number density values     [pc^-3]
    ________________________________________________________________________________________________________________________
    Notes:
    ________________________________________________________________________________________________________________________
    """
    # Ensure all inputs are arrays with positive time values
    time_values = np.maximum(np.asarray(time_values), 1e-6)
    
    # Compute all features using vectorized functions
    results = {
        'mcrit'    : critical_mass(rh_values, m_mean_values, time_values, v_disp_values, stellar_radius)[0],
        'trelax'   : relaxation_time(n_values, rh_values, v_disp_values)[0],
        'tcc'      : core_collapse_time(m_mean_values, m_max_values, n_values, rh_values, v_disp_values)[0],
        'tcoll'    : collision_time(rh_values, n_values, m_mean_values, v_disp_values, stellar_radius)[0],
        'tcross'   : crossing_time(rh_values, v_disp_values)[0],
        'rho_half' : rho_at_rh(n_values, rh_values)[0]
    }
    
    return results

#--------------------------------------------------------------------------------------------------------------------------#