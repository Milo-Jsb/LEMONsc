# Modules -----------------------------------------------------------------------------------------------------------------#
import traceback

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing         import Optional, List, Tuple, Union
from loguru._logger import Logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.filters              import safe_downsampling_of_points
from src.processing.datasets.moccasurvey import MoccaSurveyExperimentConfig, def_config
from src.processing.datasets.moccasurvey import load_moccasurvey_imbh_history, process_single_mocca_simulation

# Retrieve a partition of input / target values for a ML-Experiment -------------------------------------------------------#
def moccasurvey_dataset(simulations_path: List[str], experiment_config: MoccaSurveyExperimentConfig = def_config,
                        simulations_type   : Optional[List[str]]         = None,
                        augmentation       : bool                        = False, 
                        logger             : Optional[Logger]            = None, 
                        test_partition     : bool                        = False,
                        noise              : bool                        = True,
                        points_per_sim     : Optional[Union[int, float]] = None, 
                        n_virtual          : int                         = 1,
                        downsampled        : bool                        = False
                        )-> Union[Tuple[List[np.ndarray], List[np.ndarray]],
                                  Tuple[List[np.ndarray], List[np.ndarray], List[str]]]:
    """
    ________________________________________________________________________________________________________________________
    Retrieve a partition of input/target values for a ML-Experiment using MOCCA Survey simulation data.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> simulations_path  (list)                        : List of simulation paths.
    -> experiment_config (MoccaSurveyExperimentConfig) : Configuration object defining features and targets to retrieve.
    -> simulations_type  (Optional[list])              : List of simulation environment types (e.g., 'FAST', 'SLOW')
    -> augmentation      (bool)                        : Whether to augment data based on the time evolution of the initial 
                                                         conditions.
    -> logger            (Optional[Logger])            : If given, print relevant comments in the console.
    -> test_partition    (bool)                        : If True, store simulation paths for test data tracking.
    -> noise             (bool)                        : Implement gaussian noise to data (not included for test partition)
    -> points_per_sim    (Union[int, float])           : Number of points to sample per simulation. Only used when 
                                                         augmentation = True.
                                                         If int, uses fixed number of points. If float (between 0 and 1), 
                                                         uses proportional sampling (size = len(imbh_df) * points_per_sim).
    -> n_virtual        (int)                          : Number of virtual simulations to sample per real simulation if 
                                                         augmentation is enabled.
    -> downsampled      (bool)                         : If perform a downsampled of the dataset by accumulation points in 
                                                         a 2D histogram
    ________________________________________________________________________________________________________________________
    Returns:
        Tuple: ([features, feature_names], [targets, target_names]) or 
               ([features, feature_names], [targets, target_names], simulation_paths) if test_partition=True
    ________________________________________________________________________________________________________________________
    """
    # Retrieve configuration and prelocated values
    features, targets, simulation_paths = [], [], [] if test_partition else None
    ignored, processed = 0, 0

    # Run elements per simulation
    for idx, path in enumerate(simulations_path):

        try:
            imbh, system     = load_moccasurvey_imbh_history(f"{path}/", init_conds_sim=True, init_conds_evo=True)
            imbh_df          = imbh[0].drop_duplicates(def_config.time_column_imbh).sort_values(def_config.time_column_imbh)
            iconds_imbh_dict = imbh[1]
            system_df        = system[0].sort_values(def_config.time_column_system).reset_index(drop=True)
            env_type         = simulations_type[idx] if simulations_type else None

            # Use merge_asof to align system_df to imbh_df times
            matched_system_df = pd.merge_asof(imbh_df[[def_config.time_column_imbh]], system_df,
                                              left_on   = def_config.time_column_imbh,
                                              right_on  = def_config.time_column_system
                                              ).reset_index(drop=True, direction="nearest")
            
            # Ignore short simulations and raise warning
            if len(imbh_df) <= def_config.min_points_threshold:
                ignored += 1
                if logger: logger.warning(f"Ignored {path} (too few points: {len(imbh_df)})")
                continue

            processed += 1

            # Retrieve single simulation data
            feats, targs, idxs = process_single_mocca_simulation(imbh_df=imbh_df, system_df=matched_system_df,
                                                                 meta_dict      = iconds_imbh_dict,
                                                                 config         = experiment_config, 
                                                                 points_per_sim = points_per_sim, 
                                                                 environment    = env_type,
                                                                 augment        = augmentation and not test_partition,
                                                                 noise          = noise and not test_partition,
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

    # Perform downsampled if desired:
    if downsampled: 
        # Initialize empty lists to collect results
        X_parts, y_parts = [], []
        
        # Process each category with safety checks
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW')]:
            mask = X[:, -1] == channel_code
            if np.any(mask):
                X_channel, y_channel = safe_downsampling_of_points(X[mask], y[mask], logger)
                if X_channel is not None and len(X_channel) > 0:
                    X_parts.append(X_channel)
                    y_parts.append(y_channel)
                else:
                    if logger: logger.warning(f"No data after downsampling for {channel_name} channel")
            else:
                if logger: logger.warning(f"No data found for {channel_name} channel")
        
        # Only concatenate if we have results
        if X_parts:
            X = np.concatenate(X_parts)
            y = np.concatenate(y_parts)
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