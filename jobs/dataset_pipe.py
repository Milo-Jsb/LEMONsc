# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse

import yaml
import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from scipy.stats import ks_2samp, wasserstein_distance
from typing      import Any

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory                     import PathManagerDatasetPipeline, save_dataset_to_csv
from src.processing.features                 import filter_simulation_artifacts, tabular_features
from src.processing.scalers                  import TargetTransform
from src.processing.retriever                import moccasurvey_dataset
from src.processing.constructors.moccasurvey import compute_moccasurvey_cluster_features
from src.processing.modules.simulations      import LoadSimulationFiles
from src.processing.modules.processor        import DataProcessor
from src.processing.modules.partitions       import DataPartitioner
from src.processing.modules.plots            import PlotGenerator
from src.processing.modules.downsampling     import DownsamplingProcessor
from jobs.config._dataset                    import JobConfig, MAP_FEATS_DICT

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/dataset_pipeline.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Global Arguments --------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Preparation of the simulated dataset")
    
    # Main mode of the script
    parser.add_argument("--mode", type=str, default="study",
                        choices=["study", "comp", "feats", "plot", "dist"],
                        help="Pipeline stage to implement.")

    # Directories
    parser.add_argument("--root_dir", type=str, default="./rawdata/", 
                        help="Root directory of the data without preprocess.")
    parser.add_argument("--out_dir", type=str, default="./datasets/", 
                        help="Directory to store the processed dataset.")
    parser.add_argument("--fig_dir", type=str, default="./figures/", 
                        help="Directory to store output figures of the analysis.")

    # Dataset specifics
    parser.add_argument("--dataset", type=str, default="moccasurvey", 
                        choices = ["moccasurvey", "dragon2", "ymcs", "petar"],
                        help    = "Specific dataset to implement.")
    parser.add_argument("--exp_name", type=str, default="pof",
                        help = "Tag to name the dataset and output related elements.")
    parser.add_argument("--aug", action="store_true",
                        help="If perform augmentation with virtual simulations.")
    parser.add_argument("--down", action="store_true",
                        help="If perform downsampling by 2D histogram selection.")                    
    parser.add_argument("--down_scale", type=str, default="log_ratio",
                        choices=["ratio", "identity", "log_ratio", "log_raw"],
                        help="Target axis scale for downsampling.")
    parser.add_argument("--down_category", type=str, default="mass",
                        choices=["none", "environment", "mass"],
                        help="Category criterion for channel-separated downsampling.")
    parser.add_argument("--class_type", type=str, default="mass",
                        choices=["environment", "mass"],
                        help="Classification criterion for simulations.")
    parser.add_argument("--class_labels", type=str, nargs="+", default=["Q1", "Q2", "Q3", "Q4"],
                        help="Category labels for classification. Defaults to FAST/SLOW for env, Q1/Q2/Q3/Q4 for mass.")
    parser.add_argument("--partition", type=str, default="mass",
                        choices=["random", "env", "mass"],
                        help="Partition strategy.")
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of cross-validation folds.")
    parser.add_argument("--no_snake_draft", action="store_true",
                        help="Disable snake draft for fold assignment. Use random chunking instead.")

    return parser.parse_args()

# Pipeline Modes [Study original dataset] ---------------------------------------------------------------------------------#
def run_study_mode(data_path: str, out_figs: str, config: Any, root_dir: str, dataset: str):
    """Optimized study mode pipeline."""
    
    # Validate inputs
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    os.makedirs(out_figs, exist_ok=True)
    
    # Logging into console
    logger.info(110*"_")
    logger.info(f"Study of simulations available for {config.dataset_name} dataset")
    logger.info(110*"_")

    # Initialize processors
    sim_path_files = LoadSimulationFiles(config)
    data_processor = DataProcessor(config)
    plot_generator = PlotGenerator(config, cmap="cubehelix_r")
    
    # Load simulations
    simulations = sim_path_files.get_simulation_paths(data_path     = data_path, 
                                                      root_dir      = f"{root_dir}{dataset}/", 
                                                      load_all_sims = True,
                                                      verbose       = config.verbose)
    
    # Single simulation example
    sim_path                        = simulations[np.random.randint(0, len(simulations))]
    imbh_df, system_df, iconds_dict = sim_path_files.load_single_simulation(sim_path)
    
    # Retrieve initial conditions for one simulation
    if config.dataset_name == "moccasurvey":
        ifeats = compute_moccasurvey_cluster_features(system_df, imbh_df=imbh_df, iconds_dict=iconds_dict, 
                                                      noise=False, temp_evol=False)
    else:
        raise NotImplementedError(f"Initial conditions retrieval not implemented for dataset: {config.dataset_name}")
    
    # Plot single simulation example
    plot_generator.simulation_example(imbh_df, ifeats, out_figs)
    
    # Classify and save simulations by environment type
    if config.classify and config.class_type == "environment":
        simulations_by_type = sim_path_files.classify_simulations_by_environment(
                                    simulations   = simulations, 
                                    cache_dir     = f"{root_dir}{dataset}/",
                                    category_list = config.class_labels)
        sim_path_files.save_simulation_paths_by_type(simulations_by_type, f"{root_dir}{dataset}/")
    

    elif config.classify and config.class_type == "mass":
        simulations_by_type = sim_path_files.classify_simulations_by_mass(
                                    simulations   = simulations, 
                                    cache_dir     = f"{root_dir}{dataset}/",
                                    category_list = config.class_labels)

        sim_path_files.save_simulation_paths_by_type(simulations_by_type, f"{root_dir}{dataset}/")
    
    else:
        simulations_by_type = None
        logger.info("No classification of the simulations was performed")
        
    # Filter usable simulations based on configuration
    filtered_info = data_processor.select_suitable_sims(simulations, simulations_by_type, 
                                                        out_path = f"{root_dir}{dataset}/")
    
    # Compute and cache simulation lengths for stratified k-fold snake draft
    sim_path_files.compute_simulation_lengths(filtered_info["valid_sims"]["paths"],
                                              cache_dir=f"{root_dir}{dataset}/")
    
    # Plot efficiency vs mass ratio filter results
    filtered_info["valid_sims"]['marker']    = 'o'
    filtered_info["valid_sims"]['color']     = 'maroon'
    filtered_info["valid_sims"]['edgecolor'] = 'white'
    filtered_info["valid_sims"]['s']         = 20
    
    plot_generator.create_efficiency_plot(data     = {config.dataset_name: filtered_info["valid_sims"]}, 
                                          config   = {'cmap'              : None, 
                                                      'cmap_label'        : None, 
                                                      'cmap_name'         : None, 
                                                      'norm_mode'         : None,
                                                      'include_fit_curve' : True},            
                                          out_figs = out_figs)
    
    plot_generator.create_mass_radius_plot(data     = {config.dataset_name: filtered_info["valid_sims"]}, 
                                           config   = {'cmap'              : None, 
                                                      'cmap_label'        : None, 
                                                      'cmap_name'         : None, 
                                                      'norm_mode'         : None},            
                                           out_figs = out_figs)
    
    # Final logging
    logger.success("Study mode completed")
    logger.info(110*"_")

# Pipeline Modes [Compare Datasets Processings] ---------------------------------------------------------------------------#
def run_comparison_mode(root_dir : str, data_path: str, out_figs: str, config: Any, 
                        path_manager : PathManagerDatasetPipeline, 
                        partition    : str):
    """Run the comparison mode pipeline."""
    
    # Initialize processors
    partitioner            = DataPartitioner(config)
    sim_loader             = LoadSimulationFiles(config)
    data_processor         = DataProcessor(config)
    downsampling_processor = DownsamplingProcessor(config)
    plot_generator         = PlotGenerator(config, cmap="afmhot_r")
    
    # Load simulations
    simulations = sim_loader.get_simulation_paths(data_path, root_dir=root_dir, load_all_sims=False)
    
    # Partition simulations based on selected strategy
    if partition == "random":
        train_simulations, _, _ = partitioner.create_random_partitions(simulations)
        train_labels            = None
        logger.info("Using random partitioning")
    else:
        # Classify simulations to create/update partition files
        if partition == "mass":
            simulations_by_type = sim_loader.classify_simulations_by_mass(simulations, 
                                                                          cache_dir     = root_dir, 
                                                                          category_list = config.class_labels)
        else:
            simulations_by_type = sim_loader.classify_simulations_by_environment(simulations, 
                                                                                 cache_dir     = root_dir, 
                                                                                 category_list = config.class_labels)
        
        sim_loader.save_simulation_paths_by_type(simulations_by_type, root_dir)
        
        stratified_partitions = partitioner.create_stratified_partitions(path_manager, simulations, 
                                                                         str_crit=config.class_labels)
        
        if stratified_partitions is not None:
            train_data, _, _  = stratified_partitions
            train_simulations, train_labels = train_data
            logger.info(f"Using stratified partitioning by {partition}")
        else:
            raise FileNotFoundError(
                f"Stratified partition files for labels {config.class_labels} not found. "
                f"Run '--mode study' first with matching --class_type and --class_labels.")
    
    # Process original simulations only with window sampling, no noise, no downsampling 
    t_base, m_base, phy_base, _ = data_processor.process_simulations(train_simulations, train_labels,
                                                                     augmentation = False, 
                                                                     apply_noise  = False, 
                                                                     n_virtual    = None, 
                                                                     verbose      = True,
                                                                     study_mode   = False)
    
    # Augment training simulations, applying noise and window sampling, but no downsampling
    t_augm, m_augm, phy_augm, _ = data_processor.process_simulations(train_simulations, train_labels,
                                                                     augmentation = True, 
                                                                     apply_noise  = True, 
                                                                     n_virtual    = config.n_virtual, 
                                                                     verbose      = True,
                                                                     study_mode   = False)
    
    # Downsample the augmented dataset by 2D histogram selection
    target_scale  = getattr(config, 'downsample_target_scale', 'ratio')
    norm_col_name = getattr(config, 'downsample_norm_column', 'M_tot')
    norm_col_idx  = config.feature_names.index(norm_col_name)
    phy_augm_arr  = np.array(phy_augm)
    m_augm_arr    = np.array(m_augm)
    t_augm_arr    = np.array(t_augm)
        
    # Check for numerical errors
    invalid_y = (~np.isfinite(m_augm_arr)) | (m_augm_arr == 0)

    if target_scale in ("ratio", "log_ratio"):
        norm_factor_fwd = phy_augm_arr[:, norm_col_idx- 1]
        invalid_norm    = (~np.isfinite(norm_factor_fwd)) | (norm_factor_fwd == 0)
    else:
        norm_factor_fwd = None
        invalid_norm    = np.zeros_like(m_augm_arr, dtype=bool)

    # Combine masks
    invalid_mask = invalid_y | invalid_norm

    # Apply filtering
    if np.any(invalid_mask):
        logger.warning(f"Removing {np.sum(invalid_mask)} invalid samples before downsampling")
        phy_augm_arr = phy_augm_arr[~invalid_mask]
        m_augm_arr   = m_augm_arr[~invalid_mask]
        t_augm_arr   = t_augm_arr[~invalid_mask]
        
    # Forward transform: scale target before histogramming
    norm_factor_fwd = phy_augm_arr[:, norm_col_idx - 1] if target_scale in ("ratio", "log_ratio") else None
    scaler_fwd      = TargetTransform(transformation=target_scale, norm_factor=norm_factor_fwd,
                                      epsilon=getattr(config, 'eps_target', 0))
    m_scaled        = scaler_fwd.transform(m_augm_arr)
    
    # Select filtering technique (disabled for random partition: labels are None so type_sim=NaN, no chunk would match)
    if config.downsample_category != "none" and "type_sim" in config.feature_names and partition != "random":
        type_sim_idx       = config.feature_names.index("type_sim") - 1  
        filtering_criteria = [(float(i), lbl) for i, lbl in enumerate(config.class_labels)]
    else:
        type_sim_idx       = None
        filtering_criteria = None
    
    # Downsampling
    t_down, m_down_scaled, phy_down = downsampling_processor.perform_downsampling(t_augm_arr, m_scaled, phy_augm_arr,
                                                                                  filtering_criteria    = filtering_criteria,
                                                                                  criteria_var_position = type_sim_idx)
    
    # Inverse transform: restore raw M_MMO after downsampling
    if len(t_down) > 0:
        norm_factor_inv = phy_down[:, norm_col_idx - 1] if target_scale in ("ratio", "log_ratio") else None
        scaler_inv      = TargetTransform(transformation=target_scale, norm_factor=norm_factor_inv,
                                          epsilon=getattr(config, 'eps_target', 0))
        m_down          = scaler_inv.inverse_transform(m_down_scaled)
    else:
        m_down = m_down_scaled
    
    # Create all comparison plots (per-category breakdowns only when labels are available)
    if partition != "random" and config.class_labels is not None and "type_sim" in config.feature_names:
        plot_filtering_criteria    = [(float(i), lbl) for i, lbl in enumerate(config.class_labels)]
        plot_criteria_var_position = config.feature_names.index("type_sim") - 1
    else:
        plot_filtering_criteria    = None
        plot_criteria_var_position = None

    plot_generator.create_comparison_preprocessing(t_base    , m_base    , phy_base,
                                                   t_augm_arr, m_augm_arr, phy_augm_arr,
                                                   t_down    , m_down    , phy_down, 
                                                   out_figs,
                                                   filtering_criteria    = plot_filtering_criteria,
                                                   criteria_var_position = plot_criteria_var_position)
    
    # Verbose
    logger.success("Processing comparison completed")
    logger.info(110*"_")
    
# Helper: save simulation partition paths per fold ------------------------------------------------------------------------#
def _save_partition_paths(fold_path: str, train_sims: list, val_sims: list):
    """Write simulation paths for train and val partitions into the fold directory."""
    for name, sims in [("train", train_sims), ("val", val_sims)]:
        filepath = os.path.join(fold_path, f"{name}_simulations.txt")
        with open(filepath, 'w') as f:
            f.write('\n'.join(sims))

# Pipeline Modes [Feature generation] -------------------------------------------------------------------------------------#
def run_feats_mode(root_dir : str, data_path: str, out_path: str, folds: int, augment: bool, downsampled: bool, config: Any, 
                   path_manager: PathManagerDatasetPipeline, partition: str = "mass", snake_draft: bool = True):
    """Generate tabular features for the configured dataset using k-fold cross-validation."""
    
    # Validate data path to ensure it exists before processing
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Logging into console
    logger.info(110*"_")
    logger.info(f"Generating tabular features from {config.dataset_name} simulations")
    logger.info(110*"_")

    # Initialize processors
    sim_loader  = LoadSimulationFiles(config)
    partitioner = DataPartitioner(config)
    
    # Load simulations
    simulations = sim_loader.get_simulation_paths(data_path, root_dir=root_dir, load_all_sims=False)
    
    # Create k-fold partitions with fixed test set
    if partition == "random":
        kfold_result = partitioner.create_kfold_partitions(n_folds=folds, simulations=simulations)
        logger.info("Using random k-fold partitioning")
    else:
        kfold_result = partitioner.create_kfold_partitions(n_folds=folds, simulations=simulations,
                                                           path_manager = path_manager, 
                                                           str_crit     = config.class_labels,
                                                           snake_draft  = snake_draft)
        if kfold_result is None:
            raise FileNotFoundError(
                f"Stratified partition files for labels {config.class_labels} not found. "
                f"Run '--mode study' first with matching --class_type and --class_labels.")
        logger.info(f"Using stratified {folds}-fold partitioning by {partition}")
    
    test_data, fold_list = kfold_result
    test_simulations, test_labels = test_data
    
    # Save test partition paths once (shared across all folds)
    test_sims_path = os.path.join(out_path, "test_simulations.txt")
    with open(test_sims_path, 'w') as f:
        f.write('\n'.join(test_simulations))
    
    # Generate folds
    for fold, (train_data, val_data) in enumerate(fold_list):
        fold_path = os.path.join(out_path, f"{fold}_fold/")
        os.makedirs(fold_path, exist_ok=True)
        
        train_simulations_fold, train_labels_fold = train_data
        val_simulations_fold, val_labels_fold     = val_data
        
        # Save partition simulation paths for reproducibility
        _save_partition_paths(fold_path, train_simulations_fold, val_simulations_fold)
        
        # Training data 
        if config.dataset_name == "moccasurvey":
            xtrain_info, ytrain_info = moccasurvey_dataset(simulations_path     = train_simulations_fold, 
                                                           experiment_config    = config,
                                                           simulations_type     = train_labels_fold,
                                                           augmentation         = augment, 
                                                           logger               = logger, 
                                                           points_per_sim       = config.points_per_sim,
                                                           n_virtual            = config.n_virtual, 
                                                           downsampled          = downsampled,
                                                           max_resolution_ratio = config.max_resolution_ratio)
        else:
            raise NotImplementedError(f"Feature generation not implemented for dataset: {config.dataset_name}")
        
        save_dataset_to_csv(data=xtrain_info[0], columns=xtrain_info[1], target_data=ytrain_info[0], 
                            target_name = ytrain_info[1], 
                            filepath    = f"{fold_path}train.csv", 
                            logger      = logger)
        
        logger.info(f"Fold {fold} Train - {len(train_simulations_fold)} simulations, stored at {fold_path}")
        
        # Validation data
        if config.dataset_name == "moccasurvey":
            xval_info, yval_info, _ = moccasurvey_dataset(simulations_path     = val_simulations_fold, 
                                                          experiment_config    = config,
                                                          simulations_type     = val_labels_fold,
                                                          augmentation         = False, 
                                                          logger               = logger,
                                                          test_partition       = True, 
                                                          noise                = False,
                                                          points_per_sim       = config.points_per_sim,
                                                          downsampled          = False,
                                                          max_resolution_ratio = config.max_resolution_ratio)
        else:
            raise NotImplementedError(f"Feature generation not implemented for dataset: {config.dataset_name}")
        
        save_dataset_to_csv(data=xval_info[0], columns=xval_info[1], target_data=yval_info[0], 
                            target_name = yval_info[1], 
                            filepath    = f"{fold_path}val.csv", 
                            logger      = logger)
        
        logger.info(f"Fold {fold} Val - {len(val_simulations_fold)} simulations, stored at {fold_path}")
    
    # Testing data (fixed across all folds)
    if config.dataset_name == "moccasurvey":
        xtest_info, ytest_info, sim_paths = moccasurvey_dataset(simulations_path     = test_simulations, 
                                                                experiment_config    = config,
                                                                simulations_type     = test_labels,
                                                                augmentation         = False,
                                                                logger               = logger,
                                                                test_partition       = True, 
                                                                noise                = False,
                                                                points_per_sim       = config.points_per_sim,
                                                                downsampled          = False,
                                                                max_resolution_ratio = config.max_resolution_ratio)
    else:
        raise NotImplementedError(f"Feature generation not implemented for dataset: {config.dataset_name}")
            
    additional_columns = {"or_sim_path": sim_paths, "tag": f"{config.dataset_name}"}
    
    save_dataset_to_csv(data=xtest_info[0], columns=xtest_info[1], target_data=ytest_info[0], 
                        target_name        = ytest_info[1], 
                        filepath           = f"{out_path}test.csv", 
                        additional_columns = additional_columns,
                        logger             = logger)
    
    logger.success("Features generation completed")
    logger.info(110*"_")

# Pipeline Modes [Plot generation] ----------------------------------------------------------------------------------------#
def run_plot_mode(datafile: str, contfeats: list, catfeats: list, target: list, out_figs: str, 
                  config: Any, dataset: str):
    """Run the plotting mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Plotting information for tabular training dataset")
    logger.info(110*"_")

    # Initialize plot generator
    plot_generator = PlotGenerator(config)

    # Load files
    try:
        tab_data_df = pd.read_csv(datafile, index_col=False)
    except FileNotFoundError:
        logger.error(f"Data file not found: {datafile}")
        raise
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        raise
    
    # Filter possible numerical effects and times equal to zero
    filter_tab_df = filter_simulation_artifacts(raw_df = tab_data_df,
                                                min_denominator_threshold = config.min_dem_threshold,
                                                filter_null_mass          = True,
                                                filter_initial_state      = False,
                                                verbose                   = True)
    
    # Retrieve input features to compute 
    n_cats = len(config.class_labels)
    tab_feats_df, labels = tabular_features(process_df                 = filter_tab_df, 
                                            names                      = contfeats + target + catfeats, 
                                            return_names               = True,
                                            onehot                     = False,
                                            eps_logscale_all_range     = config.eps_feats,
                                            eps_logscale_limited_range = config.eps_target,
                                            n_sim_categories           = n_cats) 
    
    labels_names = [labels[name] for name in contfeats + target]

    logger.info(f"Features retrieved")
    logger.info(f"  - Continuos features   : {contfeats}")
    logger.info(f"  - Categorical features : {catfeats}")
    logger.info(f"  - Target               : {target}")
        

    # Plot full tabular feats:
    plot_generator.create_features_analysis(feats      = tab_feats_df[contfeats+target],
                                            names      = labels_names, 
                                            map_dict   = MAP_FEATS_DICT,
                                            dataset    = dataset, 
                                            experiment = "full", 
                                            out_figs   = out_figs)

    # Plot tabular feats by category
    for channel_code, env_name in enumerate(config.class_labels):
        
        # Check if the channel code exists in the dataset to avoid empty plots
        mask = tab_feats_df[catfeats[0]] == float(channel_code)
        
        if not np.any(mask):
            continue
        
        # Plot elements by category type
        plot_generator.create_features_analysis(feats      = tab_feats_df[mask][contfeats+target],
                                                names      = labels_names, 
                                                map_dict   = MAP_FEATS_DICT,
                                                dataset    = dataset, 
                                                experiment = env_name.lower(), 
                                                out_figs   = out_figs + f"{env_name.lower()}/")
    logger.success("Plotting completed")
    logger.info(110*"_")

# Pipeline Modes [Distribution analysis] ----------------------------------------------------------------------------------#
def run_dist_mode(root_dir: str, dataset: str, data_path: str, aug_path: str, contfeats: list, catfeats: list, 
                  target    : list,
                  figs_path : str, 
                  config    : Any,
                  partition : str = "mass",
                  max_samples: int = 100_000):
    """Run the distribution analysis mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Checking conservation of distributions for tabular training augmented dataset against raw dataset")
    logger.info(110*"_")

    # Initialize processors
    sim_loader     = LoadSimulationFiles(config)
    data_processor = DataProcessor(config)
    plot_generator = PlotGenerator(config)

    # Determine whether to split analysis by category
    use_categories = partition != "random" and catfeats is not None and len(catfeats) > 0

    # Define features (include catfeats only when needed)
    feature_names  = contfeats + target + (catfeats if use_categories else [])
    
    # Load raw simulations
    simulations = sim_loader.get_simulation_paths(data_path, 
                                                  root_dir      = f"{root_dir}{dataset}/",
                                                  load_all_sims = False)
    
    # Classify simulations based on configured class_type (only when using categories)
    if use_categories:
        if config.class_type == "mass":
            simulations_by_type = sim_loader.classify_simulations_by_mass(simulations,
                                                                          cache_dir     = f"{root_dir}{dataset}/",
                                                                          category_list = config.class_labels)
        else:
            simulations_by_type = sim_loader.classify_simulations_by_environment(simulations,
                                                                                 cache_dir     = f"{root_dir}{dataset}/",
                                                                                 category_list = config.class_labels)
        path_to_label = {path: env_type
                         for env_type, paths in simulations_by_type.items()
                         for path in paths}
        labels = [path_to_label.get(path, np.nan) for path in simulations]
    else:
        labels = [np.nan] * len(simulations)

    logger.info(f"Loading raw features...")

    # Load raw data
    t_base, m_base, phy_base, path_list = data_processor.process_simulations(simulations, labels,
                                                                  augmentation = False, 
                                                                  apply_noise  = False, 
                                                                  n_virtual    = None, 
                                                                  verbose      = True,
                                                                  study_mode   = False)
    
    # Create dataframe in the same format as the processed one
    columns = config.feature_names + [config.target_name]
    raw_df  = pd.DataFrame(data= np.column_stack((t_base, phy_base, m_base)), columns= columns)
    
    # Filter numerical artifacts and t=0 rows before log-scale feature engineering
    raw_df = filter_simulation_artifacts(raw_df, 
                                         min_denominator_threshold = config.min_dem_threshold,
                                         filter_null_mass          = True,
                                         filter_initial_state      = False,
                                         verbose                   = True)
    
    # Retrieve input features to compute statistical test
    n_cats = len(config.class_labels) if use_categories else 1
    feats_raw, _ = tabular_features(raw_df, names=feature_names, return_names=True, onehot=False,
                                    eps_logscale_all_range     = config.eps_feats,
                                    eps_logscale_limited_range = config.eps_target,
                                    n_sim_categories           = n_cats)

    # Load processed augmented features
    logger.info(f"Loading trained augmented features...")

    # Load augmented training data 
    try:
        processed_df = pd.read_csv(aug_path, index_col=False)
    except FileNotFoundError:
        logger.error(f"Augmented data file not found: {aug_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading augmented data file: {e}")
        raise
    
    # Filter numerical artifacts before log-scale feature engineering
    processed_df = filter_simulation_artifacts(processed_df,
                                               min_denominator_threshold = config.min_dem_threshold,
                                               filter_initial_state      = False,
                                               filter_null_mass          = True,
                                               verbose                   = True)
        
    feats_processed, feats_names = tabular_features(processed_df, names=feature_names, return_names=True, onehot=False,
                                                     eps_logscale_all_range     = config.eps_feats,
                                                     eps_logscale_limited_range = config.eps_target,
                                                     n_sim_categories           = n_cats)
    
    # Select continuous features inside the dataframes for comparison
    cont_features = contfeats + target  

    logger.info(110*"_")
    logger.info(f"Distribution comparison results (KS test + Wasserstein distance):")
    logger.info(f"  Max samples per distribution: {max_samples}")
    logger.info(110*"_")

    # Build list of (subset_label, raw_subset, proc_subset) groups
    comparison_groups = []
    if use_categories:
        for sim_type in feats_processed[catfeats[0]].unique():
            raw_sub  = feats_raw[feats_raw[catfeats[0]] == sim_type]
            proc_sub = feats_processed[feats_processed[catfeats[0]] == sim_type]
            comparison_groups.append((f"category_{int(sim_type)}", raw_sub, proc_sub))
    else:
        comparison_groups.append(("all", feats_raw, feats_processed))
    
    # Run statistical tests per group
    results_summary = {}
    
    for group_label, raw_group, proc_group in comparison_groups:
        logger.info(f"Group: {group_label} (raw={len(raw_group)}, processed={len(proc_group)})")
        results_summary[group_label] = {}
        
        for feature in cont_features:
            x = raw_group[feature].dropna().to_numpy()
            y = proc_group[feature].dropna().to_numpy()
            
            # Random subsample for test accuracy
            rng = np.random.default_rng()
            if len(x) > max_samples:
                x = rng.choice(x, size=max_samples, replace=False)
            if len(y) > max_samples:
                y = rng.choice(y, size=max_samples, replace=False)

            # KS test
            ks_stat, ks_pval = ks_2samp(x, y)

            # Wasserstein distance
            w_dist = wasserstein_distance(x, y)

            logger.info(f"  {feature}")
            logger.info(f"    - KS statistic={ks_stat:.3f}, p-value={ks_pval:.2e}")
            logger.info(f"    - Wasserstein distance={w_dist:.3f}")
            
            results_summary[group_label][feature] = {
                "ks_statistic"       : float(f"{ks_stat:.6f}"),
                "ks_pvalue"          : float(f"{ks_pval:.6e}"),
                "wasserstein_distance": float(f"{w_dist:.6f}"),
                "n_raw"              : int(len(x)),
                "n_processed"        : int(len(y)),
            }
    
        logger.info(110*"-")
    
    # Save results summary to YAML
    summary_path = os.path.join(figs_path, "distcomp/")
    os.makedirs(summary_path, exist_ok=True)
    yaml_path = os.path.join(summary_path, "dist_comparison_results.yaml")
    
    yaml_output = {
        "dataset"          : dataset,
        "partition"         : partition,
        "max_samples"       : max_samples,
        "use_categories"    : use_categories,
        "continuous_features": cont_features,
        "results"           : results_summary,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Results summary saved to {yaml_path}")
    
    # Create distribution comparison plots for continuous features
    plot_generator.create_features_histograms(feats_or      = feats_raw, feats_pr = feats_processed,
                                              labels        = cont_features,
                                              labels_names  = feats_names,
                                              out_figs      = summary_path)
    
    logger.success("Distribution analysis completed")
    logger.info(110*"_")

# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    
    # Set directory path manager
    path_manager = PathManagerDatasetPipeline(args.root_dir, args.dataset, args.exp_name, args.out_dir, args.fig_dir)
    
    # Setup configuration
    dataconfig              = JobConfig()
    dataconfig.dataset_name = args.dataset
    
    # Override downsample settings from CLI
    dataconfig.downsample_target_scale = args.down_scale
    dataconfig.downsample_category     = args.down_category
    
    # Override classification settings from CLI
    dataconfig.class_type = args.class_type
    if args.class_labels is not None:
        dataconfig.class_labels = args.class_labels
    elif args.class_type == "mass" or args.partition == "mass":
        dataconfig.class_labels = ["Q1", "Q2", "Q3", "Q4"]
        
    # Run appropriate mode
    if args.mode == "study":
        run_study_mode(data_path = path_manager.data_path, 
                       out_figs  = path_manager.out_figs,
                       config    = dataconfig,
                       root_dir  = args.root_dir,
                       dataset   = args.dataset)
    
    elif args.mode == "comp":
        run_comparison_mode(root_dir     = args.root_dir + args.dataset + "/",
                            data_path    = path_manager.data_path, 
                            out_figs     = path_manager.out_figs, 
                            config       = dataconfig, 
                            path_manager = path_manager,
                            partition    = args.partition)
        
    elif args.mode == "feats":
        run_feats_mode(root_dir     = args.root_dir + args.dataset + "/",
                       data_path    = path_manager.data_path, 
                       out_path     = path_manager.out_path, 
                       folds        = args.folds, 
                       augment      = args.aug, 
                       downsampled  = args.down,
                       config       = dataconfig,
                       path_manager = path_manager,
                       partition    = args.partition,
                       snake_draft  = not args.no_snake_draft)  
    
    elif args.mode == "plot":
        run_plot_mode(datafile  = f"{path_manager.out_path}0_fold/train.csv",
                      contfeats = dataconfig.cont_feats,
                      catfeats  = dataconfig.cat_feats, 
                      target    = dataconfig.target_feat,
                      out_figs  = path_manager.out_figs,
                      config    = dataconfig,
                      dataset   = args.dataset)

    elif args.mode == "dist":
        run_dist_mode(root_dir  = args.root_dir,
                      dataset   = args.dataset,
                      data_path = path_manager.data_path,
                      aug_path  = f"{path_manager.out_path}0_fold/train.csv",
                      contfeats = dataconfig.cont_feats,
                      catfeats  = dataconfig.cat_feats, 
                      target    = dataconfig.target_feat,
                      figs_path = path_manager.out_figs,
                      config    = dataconfig,
                      partition = args.partition)

# Run the full Job --------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#