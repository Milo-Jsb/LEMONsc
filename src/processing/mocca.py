
# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Tuple, Optional

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory       import load_mocca_imbh_history
from src.utils.phyfactors      import critical_mass, relaxation_time, core_collapse_time,  collision_time
from src.processing.format     import time_preparation, target_preparation

# Retrieve a partition of input / target values for a ML-Experiment -------------------------------------------------------#
def mocca_survey_dataset(simulations_path: list, experiment_type: str, augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment, supporting different experiment types.
    ________________________________________________________________________________________________________________________
    Parameters:
        simulations_path (list) : List of simulation paths.
        experiment_type  (str)  : Type of experiment ("mass", "delta_mass", "mass_rate").
        augmentation     (bool) : Whether to augment data (not implemented here).
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple [features (np.ndarray), targets (np.ndarray)]
    ________________________________________________________________________________________________________________________
    Notes:
        - The function will ignore simulations with less than 1000 points.
        - Initial conditions are retrieved and sampled following documentation according the MOCCA Survey Dataset.
        - The function will return a tuple with the features and targets.
    ________________________________________________________________________________________________________________________
    """

    # Initialize lists for features and targets ---------------------------------------------------------------------------#
    features   = []
    targets    = []
    ignored_count = 0

    # Loop over the simulations -------------------------------------------------------------------------------------------#
    for path in simulations_path:

        # Retrieve imbh simulation and the system initial conditions ------------------------------------------------------#
        imbh_history, system = load_mocca_imbh_history( file_path   = f"{path}/",
                                                        init_conds_sim  = False,
                                                        col_description = False,
                                                        stellar_map     = False,
                                                        init_conds_evo  = True,
                                                        verbose         = False)
            
        # Drop duplicate times in the imbh_dataframe
        imbh_df = imbh_history[0].drop_duplicates(subset="time[Myr]")

        # Ignore simulation if the dataframe contains less that 1000 points
        if len(imbh_df) <= 1000:
            ignored_count += 1
            continue
        
        # Retrieve the initial conditions of the simulation ---------------------------------------------------------------#
        rh      = system[0]["r_h"][0]
        v_disp  = system[0]["vc"][0]
        m_tot   = system[0]["smt"][0]
        n       = system[0]["nt"][0]
        m_mean  = system[0]["atot"][0]
        m_max   = system[0]["smsm"][0]
        tau     = imbh_df["time[Myr]"].max() - imbh_df["time[Myr]"].min()

        # Compute relevant physical quantities ---------------------------------------------------------------------------#
        mcrit  = critical_mass(hm_radius=rh, mass_per_star=m_mean, cluster_age=tau, v_disp=v_disp)
        tcc    = core_collapse_time(m_mean=m_mean, m_max=m_max, n_stars=n, hm_radius=rh, v_disp=v_disp)
        trelax = relaxation_time(n_stars=n, hm_radius=rh, v_disp=v_disp)
        tcoll  = collision_time(hm_radius=rh, n_stars=n, mass_per_star=m_mean, v_disp=v_disp)

        # Get time evolution ready (keep natural units to this list, normalization its performed latter)
        time_evol = time_preparation(time_evolution=imbh_df["time[Myr]"], norm_factor=None)

        # Select 500 random (if the simulation doesn't have 500 steps its ignored earlier)
        selected_idx = np.random.choice(len(time_evol), size=500, replace=False)

        # Choose features and targets based on experiment_type ------------------------------------------------------------#
        if (experiment_type == "mass"):
            
            x1 = m_tot / mcrit.value
            x2 = tcc.value / trelax.value
            x3 = m_mean / m_max
            x4 = (3 * m_tot) / (8 * np.pi * rh**3)

            # Retrieve target for the experiment
            m_evol = target_preparation(mass_evolution=imbh_df["massNew[Msun](10)"], time_evolution=None,
                                        norm_factor = m_tot,
                                        target_type = "M",
                                        log10_scale = False)
            
            # Store information into the lists
            features.extend([[time_evol[i] * 1/tcc.value , time_evol[i] * 1/trelax.value ,time_evol[i] * 1/tcoll.value , x1, x2, x3, x4] for i in selected_idx])
            targets.extend([m_evol[i] for i in selected_idx])

        # Choose features and targets based on experiment_type ------------------------------------------------------------#
        elif (experiment_type == "delta_mass"):
            
            raise ValueError("This functionallity is not yet implemented")

        # Choose features and targets based on experiment_type ------------------------------------------------------------#
        elif (experiment_type == "mass_rate"):
            
            raise ValueError("This functionallity is not yet implemented")

        # Raise error if the experiment_type is not supported -------------------------------------------------------------#
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

    # Transform to nunmpy -------------------------------------------------------------------------------------------------#
    features = np.array(features)
    targets  = np.array(targets)

    return [features, targets]
#--------------------------------------------------------------------------------------------------------------------------#