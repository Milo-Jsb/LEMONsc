# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing                import Optional, Union, Dict, Any
from pandas.api.types      import CategoricalDtype

# Custom utilities and functions ------------------------------------------------------------------------------------------#
from src.utils.phyfactors  import relaxation_time, core_collapse_time, collision_time, crossing_time
from src.utils.phyfactors  import rho_at_rh, critical_mass, safronov_num, mean_stellar_radius

# Define relevant operations for desired tabular features -----------------------------------------------------------------#
def tabular_features(process_df: pd.DataFrame, names:list, return_names:bool=True, onehot:bool=True) -> pd.DataFrame:
    """
    ________________________________________________________________________________________________________________________
    Generation of a dataframe with relevant features of interest for ML exploration
    ________________________________________________________________________________________________________________________
    Parameters:
        - process_df (pd.DataFrame) : Dataframe with raw data to process. Mandatory.
        - names      (list)         : List of feature names to include. If None, all features are included. Optional.
        - return_names (bool)       : Whether to return feature labels along with the DataFrame. Default is True.
        - onehot     (bool)         : Whether to apply one-hot encoding to categorical features. Default is True.
    ________________________________________________________________________________________________________________________
    Returns: 
        - result_df (pd.DataFrame)  : DataFrame with the selected and processed features.
    ________________________________________________________________________________________________________________________
    """
    # Set defaul types for categorical features ---------------------------------------------------------------------------#
    type_sim_dtype = CategoricalDtype(categories=[0., 1., 2.], ordered=False)

    # Set possible features and possible names with nested operations -----------------------------------------------------#
    default_feats = {
        "log(t/t_cc)" :{
            "label"     : r"$\log(t/t_{\rm{cc}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_cc']+1)
        },
        "log(t/t_cross)" :{
            "label"     : r"$\log(t/t_{\rm{cross}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_cross']+1)
        },
        "log(t/t_relax)" :{
            "label"     : r"$\log(t/t_{\rm{relax}})$",
            "operation" : lambda df: np.log10(df['t'] / df['t_relax']+1)
        },
        "log(t)" :{
            "label"     : r"$\log(t)$",
            "operation" : lambda df: np.log10(df['t']+1)
        },
        "log(t_coll)" :{
            "label"     : r"$\log(t_{\rm{coll}})$",
            "operation" : lambda df: np.log10(df['t_coll']+1)
        },
        "log(rho(R_h))" :{
            "label"     : r"$\log(\rho(R_{h}))$",
            "operation" : lambda df: np.log10(df['rho(R_h)'] + 1)
        },
        "log(M_tot/M_crit)" :{
            "label"     : r"$\log(M_{\rm tot}/M_{\rm crit})$",
            "operation" : lambda df:np.log10(df['M_tot']/df['M_crit']+1)
        },
        "M_MMO/M_tot" :{
            "label"     : r"$M_{\rm MMO}/M_{\rm tot}$",
            "operation" : lambda df: df['M']/df['M_tot']
        },
        "log(R_h/R_core)" :{
            "label"     : r"$\log(R_{h}/R_{\rm{core}})$",
            "operation" : lambda df: np.log10((df['R_h'] / df['R_core']) + 1)
        },
        "log(R_tid/R_core)" :{
            "label"     : r"$\log(R_{\rm{tidal}}/R_{\rm{core}})$",
            "operation" : lambda df: np.log10((df['R_tid'] / df['R_core']) + 1)
        },
        "Z":{
            "label"     : r"$Z$",
            "operation" : lambda df: df['z']
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
        time_values    (np.ndarray)         [Myr]  : Array of time values
        rh_values      (float | np.ndarray) [pc]   : Array of half-mass radius values
        v_disp_values  (float | np.ndarray) [km/s] : Array of velocity dispersion values
        n_values       (int | np.ndarray)   [#]    : Array of number of stars values
        m_mean_values  (float | np.ndarray) [Msun] : Array of mean stellar mass values
        m_max_values   (float | np.ndarray) [Msun] : Array of maximum stellar mass values
        comp_stellar_val (bool)                    : Whether to compute stellar radius (default False)
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

#--------------------------------------------------------------------------------------------------------------------------#