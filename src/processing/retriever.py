# Modules -----------------------------------------------------------------------------------------------------------------#
import traceback

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Optional, List, Tuple, Union
from loguru._logger import Logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.modules.downsampling     import DownsamplingProcessor
from src.processing.scalers                  import TargetTransform
from src.processing.constructors.moccasurvey import MoccaSurveyExperimentConfig, def_config
from src.processing.constructors.moccasurvey import load_moccasurvey_imbh_history, process_single_moccasurvey_simulation

# Retrieve a partition of input / target values for a ML-Experiment -------------------------------------------------------#
def moccasurvey_dataset(simulations_path: List[str], experiment_config: MoccaSurveyExperimentConfig = def_config,
                        simulations_type     : Optional[List[str]]         = None,
                        augmentation         : bool                        = False, 
                        logger               : Optional[Logger]            = None, 
                        test_partition       : bool                        = False,
                        noise                : bool                        = True,
                        points_per_sim       : Optional[Union[int, float]] = None, 
                        n_virtual            : int                         = 1,
                        downsampled          : bool                        = False,
                        max_resolution_ratio : float                       = 20.0
                        )-> Union[Tuple[List[np.ndarray], List[np.ndarray]],
                                  Tuple[List[np.ndarray], List[np.ndarray], List[str]]]:
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment using MOCCA Survey simulation data.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> simulations_path     (list)                        : List of simulation paths.
    -> experiment_config    (MoccaSurveyExperimentConfig) : Configuration object defining features and targets to retrieve.
    -> simulations_type     (Optional[list])              : List of simulation environment types (e.g., 'FAST', 'SLOW')
    -> augmentation         (bool)                        : Whether to augment data based on the time evolution of the 
                                                            initial conditions.
    -> logger               (Optional[Logger])            : If given, print relevant comments in the console.
    -> test_partition       (bool)                        : If True, store simulation paths for test data tracking.
    -> noise                (bool)                        : Implement gaussian noise to data (not included for test 
                                                            partition)
    -> points_per_sim       (Union[int, float])           : Number of points to sample per simulation. Only used when 
                                                            augmentation = True.
                                                            If int, uses fixed number of points. If float (between 0 and 1), 
                                                            uses proportional sampling 
                                                            (size = len(imbh_df) * points_per_sim).
    -> n_virtual            (int)                         : Number of virtual simulations to sample per real simulation if 
                                                            augmentation is enabled.
    -> downsampled          (bool)                        : If perform a downsampled of the dataset by accumulation points
                                                            in a 2D histogram
    -> max_resolution_ratio (float)                       : Maximum allowed ratio of (median match error) / (median IMBH
                                                            timestep) after merge_asof alignment. Simulations where this
                                                            ratio exceeds the threshold are skipped with a warning.
                                                            Default=20.0 (match error <= 20 x dt_imbh).
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple: ([features, feature_names], [targets, target_names]) or 
               ([features, feature_names], [targets, target_names], simulation_paths) if test_partition=True
    ________________________________________________________________________________________________________________________
    Notes:
        - When checking the temporal resolution compatibility, we compute the median IMBH timestep and the median match 
          error produced by the nearest-neighbour merge. If the resolutions are too disparate, every virtual window sample 
          will pair an IMBH state with a system state that is physically far away in time, silently biasing the features.
        - If the merger is done correctly, we overwrite time_column_system with the exact IMBH timestamps so that when 
          process_single_mocca_simulation re-sorts both DataFrames by their respective time columns, both produce the same 
          row order (no duplicates from nearest-neighbor matching can cause misalignment during window sampling).
    ________________________________________________________________________________________________________________________
    """
    # Retrieve configuration and prelocated values
    features, targets, simulation_paths = [], [], [] if test_partition else None
    ignored, processed = 0, 0

    # Run elements per simulation
    for idx, path in enumerate(simulations_path):

        try:
            imbh, system     = load_moccasurvey_imbh_history(f"{path}/", init_conds_sim=True, init_conds_evo=True)
            imbh_df          = imbh[0].drop_duplicates(experiment_config.time_column_imbh
                                                       ).sort_values(experiment_config.time_column_imbh
                                                                     ).reset_index(drop=True)
            iconds_imbh_dict = imbh[1]
            system_df        = system[0].sort_values(experiment_config.time_column_system).reset_index(drop=True)
            env_type         = simulations_type[idx] if simulations_type else None

            # Ignore short simulations first (cheap check before alignment)
            if len(imbh_df) <= experiment_config.min_points_threshold:
                ignored += 1
                if logger: logger.warning(f"Ignored {path} (too few points: {len(imbh_df)})")
                continue

            # Use merge_asof to align system_df to imbh_df times
            matched_system_df = pd.merge_asof(imbh_df[[experiment_config.time_column_imbh]], system_df,
                                              left_on   = experiment_config.time_column_imbh,
                                              right_on  = experiment_config.time_column_system, 
                                              direction = "nearest"
                                              ).reset_index(drop=True)
            
            # Check temporal resolution compatibility before accepting the alignment.
            imbh_dt  = imbh_df[experiment_config.time_column_imbh].diff().median()
            res_time = matched_system_df[experiment_config.time_column_system]- imbh_df[experiment_config.time_column_imbh]
            
            # Compute the median match error after alignment to assess temporal resolution compatibility.
            match_error = res_time.abs().median()

            # If the median match error is larger than the allowed threshold times the median IMBH timestep, skip.
            if imbh_dt > 0 and match_error > max_resolution_ratio * imbh_dt:
                ignored += 1
                if logger: logger.warning(
                    f"Ignored '{path}': temporal resolutions too disparate "
                    f"(median match error={match_error:.2f} Myr, "
                    f"imbh_dt={imbh_dt:.2f} Myr, "
                    f"ratio={match_error/imbh_dt:.1f}x > threshold={max_resolution_ratio}x)"
                )
                continue

            # Overwrite time_column_system with the exact IMBH timestamp.
            matched_system_df[experiment_config.time_column_system] = imbh_df[experiment_config.time_column_imbh].values

            processed += 1

            # Retrieve single simulation data
            if_aument = augmentation and not test_partition
            if_noise  = noise        and not test_partition
            
            feats, targs, idxs = process_single_moccasurvey_simulation(imbh_df=imbh_df, system_df=matched_system_df,
                                                                       meta_dict      = iconds_imbh_dict,
                                                                       config         = experiment_config, 
                                                                       points_per_sim = points_per_sim, 
                                                                       environment    = env_type,
                                                                       augment        = if_aument,
                                                                       noise          = if_noise,
                                                                       n_virtual      = n_virtual
                                                                      )

            features.append(feats)
            targets.append(targs)
            
            if test_partition: simulation_paths.extend([path] * len(idxs))

        except Exception as e:
            ignored += 1
            if logger: logger.warning(f"Error in '{path}': {e}\n{traceback.format_exc()}")
            continue
    
    # Stack the features and the target given the selected features
    X = np.vstack(features) if features else np.empty((0, len(experiment_config.feature_names)))
    y = np.concatenate(targets) if targets else np.empty((0,))
    
    # Perform downsampling if desired (target axis scale is controlled by config.downsample_target_scale)
    if downsampled:
        
        # Retrieve config values before using them
        norm_col_name = getattr(experiment_config, 'downsample_norm_column', 'M_tot')
        norm_col_idx  = experiment_config.feature_names.index(norm_col_name)
        target_scale  = getattr(experiment_config, 'downsample_target_scale', 'ratio')
        
        # Check for numerical errors
        invalid_y = (~np.isfinite(y)) | (y == 0)

        if target_scale in ("ratio", "log_ratio"):
            norm_factor_fwd = X[:, norm_col_idx]
            invalid_norm    = (~np.isfinite(norm_factor_fwd)) | (norm_factor_fwd == 0)
        else:
            norm_factor_fwd = None
            invalid_norm    = np.zeros_like(y, dtype=bool)

        # Combine masks
        invalid_mask = invalid_y | invalid_norm

        # Apply filtering
        if np.any(invalid_mask):
            if logger: logger.warning(f"Removing {np.sum(invalid_mask)} invalid samples before downsampling")
            X = X[~invalid_mask]
            y = y[~invalid_mask]
            
        # Forward transform: scale target using pre-downsampling M_tot (row-aligned with y)
        norm_factor_fwd = X[:, norm_col_idx] if target_scale in ("ratio", "log_ratio") else None
        scaler_fwd      = TargetTransform(transformation = target_scale, 
                                          norm_factor    = norm_factor_fwd, 
                                          epsilon        = experiment_config.eps_target)
        y_for_hist      = scaler_fwd.transform(y)
        
        if logger: logger.info(f"Downsampling in target space: {target_scale}")

        # Build channel-separation criteria from config (if enabled)
        down_category = getattr(experiment_config, 'downsample_category', 'none')
        if down_category != "none" and "type_sim" in experiment_config.feature_names:
            class_labels          = getattr(experiment_config, 'class_labels', ["FAST", "SLOW"])
            filtering_criteria    = [(float(i), label) for i, label in enumerate(class_labels)]
            criteria_var_position = experiment_config.feature_names.index("type_sim") - 1  # offset by time column
        else:
            filtering_criteria    = None
            criteria_var_position = None

        # Downsample
        downsampler = DownsamplingProcessor(experiment_config)
        t_down, y_down, phy_down = downsampler.perform_downsampling(x_var                 = X[:, 0], 
                                                                    y_var                 = y_for_hist, 
                                                                    metadata              = X[:, 1:],
                                                                    filtering_criteria    = filtering_criteria,
                                                                    criteria_var_position = criteria_var_position)
        
        if len(t_down) > 0:
            X = np.column_stack([t_down, phy_down])
            # Inverse transform: use post-downsampling M_tot (row-aligned with y_down)
            norm_factor_inv = phy_down[:, norm_col_idx - 1] if target_scale in ("ratio", "log_ratio") else None
            scaler_inv      = TargetTransform(transformation = target_scale, 
                                              norm_factor    = norm_factor_inv, 
                                              epsilon        = experiment_config.eps_target)
            y               = scaler_inv.inverse_transform(y_down)
        else:
            if logger: logger.error("No data remaining after downsampling all channels")

    if logger:
        logger.info(f"Processed {processed}, Ignored {ignored}, Shape: {X.shape}, Targets: {experiment_config.target_name}")

    if test_partition:
        
        if downsampled and logger:
            logger.warning("Downsampling with test_partition=True shouldn't be used. Review the configuration")

        return [X, experiment_config.feature_names], [y, experiment_config.target_name], simulation_paths

    return [X, experiment_config.feature_names], [y, experiment_config.target_name]

#--------------------------------------------------------------------------------------------------------------------------#