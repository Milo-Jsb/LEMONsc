
# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Tuple, Optional
from loguru._logger import Logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory       import load_mocca_survey_imbh_history
from src.utils.phyfactors      import critical_mass, relaxation_time, core_collapse_time,  collision_time, rho_at_rh
from src.processing.format     import time_preparation, target_preparation

# Helper functions --------------------------------------------------------------------------------------------------------#
def __get_target_name(base: str, norm_target: bool, log10_target: bool) -> list:
    """Helper function to generate target names based on normalization and scaling options."""
    if norm_target:
        return [f"{base}/M_tot"]
    elif log10_target:
        return [f"log({base})"]
    else:
        return [base]

def __process_simulation_data(imbh_df, system, experiment_type, norm_target, log10_target, config,
                             window_length    : int = 15,
                             polyorder        : int = 2,
                             medfilt_kernel   : int = 5,
                             time_norm_factor : Optional[float] = None,
                             time_return_diff : bool = False,
                             points_per_sim   : int = 500,
                             augment          : bool = False,
                             n_virtual        : int = 1):
    """
    Helper function to process a single simulation with parameterization for smoothing/filtering and optional augmentation.
    If augment is True, samples n_virtual virtual simulations from the real simulation, each as a window of points_per_sim consecutive points.
    """
    features_list         = []
    targets_list          = []
    selected_indices_list = []
    imbh_df               = imbh_df.reset_index(drop=True)
    system_df             = system.reset_index(drop=True)
    n_points              = len(imbh_df)
    
    if augment:
        max_start = n_points - points_per_sim
        if max_start < 0:
            raise ValueError(f"Not enough points in simulation (required {points_per_sim}, got {n_points})")
        
        for _ in range(n_virtual):
            # Select the ramdon points from the simulation
            start_idx  = np.random.randint(0, max_start + 1)
            idx_window = np.arange(start_idx, start_idx + points_per_sim)
            idx_window = np.sort(idx_window)

            # Filter dataframes
            imbh_df_sampled = imbh_df.iloc[idx_window].copy()
            system_sampled  = system_df.iloc[idx_window].copy()
            
            # Reset time to zero
            t0 = imbh_df_sampled["time[Myr]"].iloc[0]
            imbh_df_sampled["time[Myr]"] -= t0
            system_sampled["time[Myr]"]  -= t0

            # Use initial conditions from first row
            rh     = system_sampled["r_h"].iloc[0]
            v_disp = system_sampled["vc"].iloc[0]
            m_tot  = system_sampled["smt"].iloc[0]
            n      = system_sampled["nt"].iloc[0]
            m_mean = system_sampled["atot"].iloc[0]
            m_max  = system_sampled["smsm"].iloc[0]
            cr     = system_sampled["rc"].iloc[0]
            tau    = imbh_df_sampled["time[Myr]"].max() - imbh_df_sampled["time[Myr]"].min()
            
            # Compute relevant physical quantities
            mcrit     = critical_mass(hm_radius=rh, mass_per_star=m_mean, cluster_age=tau, v_disp=v_disp)
            trelax    = relaxation_time(n_stars=n, hm_radius=rh, v_disp=v_disp)
            tcc       = core_collapse_time(m_mean=m_mean, m_max=m_max, n_stars=n, hm_radius=rh, v_disp=v_disp )
            tcoll     = collision_time(hm_radius=rh, n_stars=n, mass_per_star=m_mean, v_disp=v_disp)
            rho_half  = rho_at_rh(n_stars=n, hm_radius=rh)
            
            # Prepare tme and target
            time_evol = time_preparation(time_evolution = imbh_df_sampled["time[Myr]"], norm_factor = time_norm_factor, return_diff = False)
            m_evol    = target_preparation(mass_evolution = imbh_df_sampled["massNew[Msun](10)"], time_evolution = imbh_df_sampled["time[Myr]"],
                                            norm_factor    = m_tot if norm_target else None,
                                            target_type    = experiment_type,
                                            log10_scale    = log10_target,
                                            window_length  = window_length,
                                            polyorder      = polyorder,
                                            medfilt_kernel = medfilt_kernel)
            if config["requires_time_diff"]:
                _, t_diff = time_preparation(time_evolution=imbh_df_sampled["time[Myr]"], norm_factor=time_norm_factor, return_diff=True)
                sim_features = np.column_stack((
                    time_evol,
                    t_diff,
                    np.full(points_per_sim, tcoll.value),
                    np.full(points_per_sim, trelax.value),
                    np.full(points_per_sim, tcc.value),
                    np.full(points_per_sim, m_tot),
                    np.full(points_per_sim, m_mean),
                    np.full(points_per_sim, m_max),
                    np.full(points_per_sim, mcrit.value),
                    np.full(points_per_sim, rho_half.value),
                    np.full(points_per_sim, rh),
                    np.full(points_per_sim, cr)
                ))
            else:
                sim_features = np.column_stack((
                    time_evol,
                    np.full(points_per_sim, tcoll.value),
                    np.full(points_per_sim, trelax.value),
                    np.full(points_per_sim, tcc.value),
                    np.full(points_per_sim, m_tot),
                    np.full(points_per_sim, m_mean),
                    np.full(points_per_sim, m_max),
                    np.full(points_per_sim, mcrit.value),
                    np.full(points_per_sim, rho_half.value),
                    np.full(points_per_sim, rh),
                    np.full(points_per_sim, cr)
                ))
            features_list.append(sim_features)
            targets_list.append(m_evol)
            selected_indices_list.append(idx_window)
        # Concatenate all virtual simulations
        features     = np.vstack(features_list)
        targets      = np.concatenate(targets_list)
        selected_idx = np.concatenate(selected_indices_list)
        return features, targets, selected_idx
    
    else:
        # Retrieve initial conditions of the simulation
        rh     = system_df["r_h"].iloc[0]
        v_disp = system_df["vc"].iloc[0]
        m_tot  = system_df["smt"].iloc[0]
        n      = system_df["nt"].iloc[0]
        m_mean = system_df["atot"].iloc[0]
        m_max  = system_df["smsm"].iloc[0]
        cr     = system_df["rc"].iloc[0]
        tau    = imbh_df["time[Myr]"].max() - imbh_df["time[Myr]"].min()
       
        # Compute relevant physical quantities
        mcrit    = critical_mass(hm_radius=rh, mass_per_star=m_mean, cluster_age=tau, v_disp=v_disp)
        trelax   = relaxation_time(n_stars=n, hm_radius=rh, v_disp=v_disp)
        tcc      = core_collapse_time(m_mean=m_mean, m_max=m_max, n_stars=n, hm_radius=rh, v_disp=v_disp )
        tcoll    = collision_time(hm_radius=rh, n_stars=n, mass_per_star=m_mean, v_disp=v_disp)
        rho_half = rho_at_rh(n_stars=n, hm_radius=rh)

        # Prepare time input
        time_evol = time_preparation(time_evolution = imbh_df["time[Myr]"], 
                                    norm_factor = time_norm_factor, 
                                    return_diff = False)
        n_points  = len(time_evol)

        if n_points < points_per_sim:
            raise ValueError(f"Not enough points in simulation (required {points_per_sim}, got {n_points})")
        
        selected_idx = np.random.choice(n_points, size=points_per_sim, replace=False)
        m_evol = target_preparation(mass_evolution = imbh_df["massNew[Msun](10)"], time_evolution = imbh_df["time[Myr]"],
                                    norm_factor    = m_tot if norm_target else None,
                                    target_type    = experiment_type,
                                    log10_scale    = log10_target,
                                    window_length  = window_length,
                                    polyorder      = polyorder,
                                    medfilt_kernel = medfilt_kernel)

        if config["requires_time_diff"]:
            _, t_diff = time_preparation(time_evolution=imbh_df["time[Myr]"], norm_factor=time_norm_factor, return_diff=True)
            sim_features = np.column_stack((
                time_evol[selected_idx],
                t_diff[selected_idx],
                np.full(points_per_sim, tcoll.value),
                np.full(points_per_sim, trelax.value),
                np.full(points_per_sim, tcc.value),
                np.full(points_per_sim, m_tot),
                np.full(points_per_sim, m_mean),
                np.full(points_per_sim, m_max),
                np.full(points_per_sim, mcrit.value),
                np.full(points_per_sim, rho_half.value),
                np.full(points_per_sim, rh),
                np.full(points_per_sim, cr)
            ))
        else:
            sim_features = np.column_stack((
                time_evol[selected_idx],
                np.full(points_per_sim, tcoll.value),
                np.full(points_per_sim, trelax.value),
                np.full(points_per_sim, tcc.value),
                np.full(points_per_sim, m_tot),
                np.full(points_per_sim, m_mean),
                np.full(points_per_sim, m_max),
                np.full(points_per_sim, mcrit.value),
                np.full(points_per_sim, rho_half.value),
                np.full(points_per_sim, rh),
                np.full(points_per_sim, cr)
            ))
        sim_targets = m_evol[selected_idx]
        return sim_features, sim_targets, selected_idx


# Retrieve a partition of input / target values for a ML-Experiment -------------------------------------------------------#
def mocca_survey_dataset(simulations_path: list, experiment_type: str, augmentation: bool = False, 
    norm_target      : bool = False,
    log10_target     : bool = False,
    logger           : Optional[Logger] = None,
    test_partition   : bool = False,
    window_length    : int = 15,
    polyorder        : int = 2,
    medfilt_kernel   : int = 5,
    time_norm_factor : Optional[float] = None,
    time_return_diff : bool = False,
    points_per_sim   : int = 500,
    n_virtual        : int = 1
    ) -> tuple:
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment, supporting different experiment types.
    ________________________________________________________________________________________________________________________
    Parameters:
        simulations_path (list)             : List of simulation paths.
        experiment_type  (str)              : Type of experiment ("point_mass", "delta_mass", "mass_rate").
        augmentation     (bool)             : Whether to augment data based on the time evolution of the initial conditions.
        norm_target      (bool)             : Whether to normalize the target by some initial condition.
        log10_target     (bool)             : Whether to use a logarithmic scale.
        logger           (Optional[Logger]) : If given, print relevant comments in the console.
        test_partition   (bool)             : If True, store simulation paths for test data tracking.
        window_length    (int)              : Window length for Savitzky-Golay smoothing (target_preparation).
        polyorder        (int)              : Polynomial order for Savitzky-Golay smoothing (target_preparation).
        medfilt_kernel   (int)              : Kernel size for median filter (target_preparation).
        time_norm_factor (Optional[float])  : Normalization factor for time_preparation.
        time_return_diff (bool)             : Whether to return time differences in time_preparation.
        points_per_sim   (int)              : Number of points to sample per simulation.
        n_virtual (int): Number of virtual simulations to sample per real simulation if augmentation is enabled.
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple: ([features, feature_names], [targets, target_names]) or 
               ([features, feature_names], [targets, target_names], simulation_paths) if test_partition=True
    ________________________________________________________________________________________________________________________
    """

    # Experiment configuration --------------------------------------------------------------------------------------------#
    EXPERIMENT_CONFIG = {
        "point_mass": {
            "feature_names": ["t", "t_coll", "t_relax", "t_cc",
                             "M_tot", "M_mean", "M_max", "M_crit", 
                             "rho(R_h)", "R_h", "R_core"],
            "base_target"  : "M_MMO",
            "requires_time_diff": False
        },
        "delta_mass": {
            "feature_names": ["t/t_coll", "delta t", "M_tot/M_crit", "rho(R_h)", "R_h/R_core"],
            "base_target"  : "delta M_MMO",
            "requires_time_diff": True
        },
        "mass_rate": {
            "feature_names": [],
            "base_target"  : "delta M_MMO/delta t",
            "requires_time_diff": True
        }
    }

    # Input validation ----------------------------------------------------------------------------------------------------#
    if experiment_type not in EXPERIMENT_CONFIG:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    if experiment_type == "mass_rate":
        raise ValueError("This functionality is not yet implemented")

    config = EXPERIMENT_CONFIG[experiment_type]

    # Log start of processing
    if logger is not None:
        logger.info(f"Starting mocca_survey_dataset with {len(simulations_path)} simulations'")
        logger.info(f"experiment_type='{experiment_type}'")

    # Pre-calculate expected size for better memory allocation ------------------------------------------------------------#
    max_simulations  = len(simulations_path)
    features         = []
    targets          = []
    simulation_paths = [] if test_partition else None
    ignored_count    = 0
    processed_count  = 0

    # Loop over the simulations -------------------------------------------------------------------------------------------#
    for path in simulations_path:
        try:
            # Retrieve imbh simulation and the system initial conditions
            imbh_history, system = load_mocca_survey_imbh_history(file_path=f"{path}/",
                                    init_conds_sim  = False,
                                    col_description = False,
                                    stellar_map     = False,
                                    init_conds_evo  = True,
                                    verbose         = False)
            
            # Drop duplicate times in the imbh_dataframe
            imbh_df   = imbh_history[0].drop_duplicates(subset="time[Myr]")
            
            imbh_df_sorted   = imbh_df.sort_values("time[Myr]").reset_index(drop=True)
            system_df_sorted = system[0].sort_values("tphys").reset_index(drop=True)

            # Use merge_asof to align system_df to imbh_df times
            matched_system_df = pd.merge_asof(imbh_df_sorted[["time[Myr]"]], system_df_sorted,
                                            left_on   = "time[Myr]",
                                            right_on  = "tphys",
                                            direction = "nearest")

            # Ignore simulation if the dataframe contains less than 1000 points
            if len(imbh_df) <= points_per_sim * 2:
                ignored_count += 1
                if logger is not None:
                    logger.warning(f"Simulation at '{path}' ignored (only {len(imbh_df)} points")
                continue

            processed_count += 1
            
            # Process simulation data
            sim_features, sim_targets, selected_idx = __process_simulation_data(imbh_df, matched_system_df, experiment_type, 
                norm_target,
                log10_target,
                config,
                window_length    = window_length,
                polyorder        = polyorder,
                medfilt_kernel   = medfilt_kernel,
                time_norm_factor = time_norm_factor,
                time_return_diff = time_return_diff,
                points_per_sim   = points_per_sim,
                augment          = augmentation and not test_partition,
                n_virtual        = n_virtual
            )
                                                        
            # Extend main lists
            features.append(sim_features)
            targets.append(sim_targets)
            
            # Store simulation paths for test partition if requested
            if test_partition:
                simulation_paths.extend([path] * len(selected_idx))

        except Exception as e:
            ignored_count += 1
            if logger is not None:
                logger.warning(f"Error processing simulation at '{path}': {str(e)}")
            continue

    # Transform to numpy arrays
    if features:
        features = np.vstack(features)
        targets  = np.concatenate(targets)
    else:
        features = np.empty((0, len(config['feature_names'])))
        targets  = np.empty((0,))
    
    if test_partition: simulation_paths = np.array(simulation_paths)

    # Generate target names
    target_names = __get_target_name(config["base_target"], norm_target, log10_target)

    # Log completion
    if logger is not None:
        logger.info(f"Finished mocca_survey_dataset:")
        logger.info(f"Processed {processed_count} simulations, Ignored {ignored_count} simulations.")
        logger.info(f"Features names: {config['feature_names']}")
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Target names: {target_names}")
        logger.info(f"Targets shape: {targets.shape}")
        if test_partition:
            logger.info(f"Simulation paths shape: {simulation_paths.shape}")

    # Return results
    if test_partition:
        return [features, config['feature_names']], [targets, target_names], simulation_paths
    else:
        return [features, config['feature_names']], [targets, target_names]

#--------------------------------------------------------------------------------------------------------------------------#