# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from scipy.stats import ks_2samp, wasserstein_distance
from typing      import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Custom functions --------------------------------------------------------------------------------------------------------#

# Directory
from src.utils.directory import PathManagerDatasetPipeline, save_dataset_to_csv

# Data processing
from src.processing.features import tabular_features

# Specifics to dataset
from src.processing.retriever                import moccasurvey_dataset
from src.processing.constructors.moccasurvey import MoccaSurveyExperimentConfig, compute_mocca_cluster_features

# Import helper classes from processing modules
from src.processing.modules.simulations  import SimulationProcessor
from src.processing.modules.processor    import DataProcessor
from src.processing.modules.partitions   import DataPartitioner
from src.processing.modules.plots        import PlotGenerator
from src.processing.modules.downsampling import DownsamplingProcessor

# Visualization
from src.utils.visualize import plot_feature_distributions

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/moccaset_pipe.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Configuration -----------------------------------------------------------------------------------------------------------#
def get_config_class(dataset: str):
    """Factory function to get the appropriate config class based on the dataset type."""
    config_map = {
        'moccasurvey': MoccaSurveyExperimentConfig}

    return config_map.get(dataset, MoccaSurveyExperimentConfig)

def create_processing_config(dataset: str):
    """Create a configuration dataclass for processing features based on the dataset type."""
    # Define base config class
    BaseConfig = get_config_class(dataset)
    
    # Set default parameters for preprocessing
    @dataclass
    class ProcessingFeaturesConfig(BaseConfig):
        """Configuration class for the processing of get_features() parameters."""
        dataset_name         : str               = dataset
        points_per_sim       : Union[int, float] = 0.8 
        n_virtual            : int               = 10 
        train_split          : float             = 0.7
        val_split            : float             = 0.2
        test_split           : float             = 0.1
        min_points_threshold : int               = 1000
        histogram_bins       : int               = 200
        downsample_min_count : int               = 10
        downsample_max_count : int               = 150
        requires_temp_evol   : bool              = False
        sample_window        : bool              = False
    
    return ProcessingFeaturesConfig()

# Global Arguments --------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Preparation of the simulated dataset")
    
    # Main mode of the script
    parser.add_argument("--mode", type=str, default="study",
                        choices=["study", "feats", "plot", "dist"],
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
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of folds to retrieve for kfold cross-validation.")

    return parser.parse_args()

# List of tabular features to compute for the dataset (can be extended for other datasets if needed) ----------------------#
TabFeats = {
    "cont_feats"  : ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)", "log(t_coll)", 
                     "log(M_tot/M_crit)", 
                     "log(R_h/R_core)", "log(R_tid/R_core)",
                     "log(rho(R_h))"],
    "cat_feats"   : ["type_sim"],
    
    "target_feat" : ["M_MMO/M_tot"],
            }

# Pipeline Modes (Study original dataset) ---------------------------------------------------------------------------------#
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
    processor              = SimulationProcessor(config)
    data_processor         = DataProcessor(config)
    downsampling_processor = DownsamplingProcessor(config)
    plot_generator         = PlotGenerator(config)
    
    # Load simulations
    simulations = processor.load_simulation_data(data_path)
    
    # Single simulation example
    imbh_df, system_df, sim_path, iconds_dict = processor.load_single_simulation_example(simulations)
    
    # Retrieve initial conditions for one simulation
    if config.dataset_name == "moccasurvey":
        ifeats = compute_mocca_cluster_features(system_df, imbh_df, iconds_dict, noise=False)
    else:
        raise NotImplementedError(f"Initial conditions retrieval not implemented for dataset: {config.dataset_name}")
    
    # Plot single simulation example
    plot_generator.simulation_example(imbh_df, ifeats, out_figs)
    
    # Classify and save simulations by environment type
    simulations_by_type = processor.classify_simulations_by_environment(simulations, 
                                                                        cache_dir=f"{root_dir}{dataset}/")

    
    path_to_label = {path: env_type
                    for env_type, paths in simulations_by_type.items()
                    for path in paths
                    }

    labels = [path_to_label.get(path, np.nan) for path in simulations]

    processor.save_simulation_paths_by_type(simulations_by_type, f"{root_dir}{dataset}/")
    
    # Filter usable simulations based on configuration
    filtered_info = data_processor.select_suitable_sims(simulations, simulations_by_type, 
                                                        out_path = f"{root_dir}{dataset}/")
    
    # Plot efficiency vs mass ratio filter results
    filtered_info["valid_sims"]['marker']    = 'o'
    filtered_info["valid_sims"]['color']     = 'navy'
    filtered_info["valid_sims"]['edgecolor'] = 'white'
    filtered_info["valid_sims"]['s']         = 20
    
    plot_generator.create_efficiency_plot(data     = {config.dataset_name: filtered_info["valid_sims"]}, 
                                          config   = {'cmap': None, 'cmap_label': None, 'cmap_name': None, 'norm_mode': None,
                                                      'include_fit_curve' : True},            
                                          out_figs = out_figs)
    
    # Update the simulations and labels to only valid ones
    simulations = filtered_info["valid_sims"]["paths"]
    labels      = filtered_info["valid_sims"]["labels"]
    
    # Process simulations
    t_base, m_base, phy_base, _ = data_processor.process_simulations(simulations, labels,
                                                                     augmentation = False, 
                                                                     apply_noise  = False, 
                                                                     n_virtual    = None, 
                                                                     verbose      = True,
                                                                     study_mode   = False)
    
    # Augment simulations
    t_augm, m_augm, phy_augm, _ = data_processor.process_simulations(simulations, labels,
                                                                     augmentation = True, 
                                                                     apply_noise  = True, 
                                                                     n_virtual    = config.n_virtual, 
                                                                     verbose      = True,
                                                                     study_mode   = False)
    
    # Downsampling analysis
    t_down, m_down, phy_down = downsampling_processor.perform_downsampling(t_augm, m_augm, phy_augm)
    
    # Create all comparison plots
    plot_generator.create_comparison_preprocessing(t_base, m_base, phy_base,
                                                   t_augm, m_augm, phy_augm,
                                                   t_down, m_down, phy_down, 
                                                   out_figs)
    
    # Final logging
    logger.success("Study mode completed")
    logger.info(110*"_")

# Pipeline Modes (Feature generation) ------------------------------------------------------------------------------------#
def run_feats_mode(data_path: str, out_path: str, folds: int, augment: bool, downsampled: bool, config: Any, 
                   path_manager: PathManagerDatasetPipeline):
    """Generate tabular features for the configured dataset."""
    # Validate data path to ensure it exists before processing
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Logging into console
    logger.info(110*"_")
    logger.info(f"Generating tabular features from {config.dataset_name} simulations")
    logger.info(110*"_")

    # Initialize processors
    processor   = SimulationProcessor(config)
    partitioner = DataPartitioner(config)
    
    # Load simulations
    simulations = processor.load_simulation_data(data_path)
    
    # Try stratified partitioning first
    stratified_partitions = partitioner.create_stratified_partitions(path_manager, simulations)
    
    if stratified_partitions is not None:
        train_data, val_data, test_data = stratified_partitions  
        train_simulations , train_labels = train_data
        val_simulations   , val_labels   = val_data
        test_simulations  , test_labels  = test_data

        logger.info("Using stratified partitioning based on environment type")
    else:
        train_simulations, val_simulations, test_simulations = partitioner.create_random_partitions(simulations)
        train_labels = val_labels = test_labels = None

        logger.info("Using random partitioning (stratified files not found)")
    
    # Generate folds
    for fold in range(folds):
        fold_path = os.path.join(out_path, f"{fold}_fold/")
        os.makedirs(fold_path, exist_ok=True)
        
        # Training data 
        if config.dataset_name == "moccasurvey":
            xtrain_info, ytrain_info = moccasurvey_dataset(simulations_path = train_simulations, 
                                                           simulations_type = train_labels,
                                                           augmentation     = augment, 
                                                           logger           = logger, 
                                                           points_per_sim   = config.points_per_sim,
                                                           n_virtual        = config.n_virtual, 
                                                           downsampled      = downsampled)
        else:
            raise NotImplementedError(f"Feature generation not implemented for dataset: {config.dataset_name}")
        
        save_dataset_to_csv(data=xtrain_info[0], columns=xtrain_info[1], target_data=ytrain_info[0], 
                            target_name = ytrain_info[1], 
                            filepath    = f"{fold_path}train.csv", 
                            logger      = logger)
        
        logger.info(f"Fold {fold} Train - Stored at {fold_path}")
        
        # Validation data
        if config.dataset_name == "moccasurvey":
            xval_info, yval_info, _ = moccasurvey_dataset(simulations_path = val_simulations, 
                                                          simulations_type = val_labels,
                                                          augmentation     = False, 
                                                          logger           = logger,
                                                          test_partition   = True, 
                                                          noise            = False,
                                                          points_per_sim   = config.points_per_sim,
                                                          downsampled      = False)
        else:
            raise NotImplementedError(f"Feature generation not implemented for dataset: {config.dataset_name}")
        
        save_dataset_to_csv(data=xval_info[0], columns=xval_info[1], target_data=yval_info[0], 
                            target_name = yval_info[1], 
                            filepath    = f"{fold_path}val.csv", 
                            logger      = logger)
        
        logger.info(f"Fold {fold} Val - Stored at {fold_path}")
    
    # Testing data
    if config.dataset_name == "moccasurvey":
        xtest_info, ytest_info, sim_paths = moccasurvey_dataset(simulations_path = test_simulations, 
                                                                simulations_type = test_labels,
                                                                augmentation     = False,
                                                                logger           = logger,
                                                                test_partition   = True, 
                                                                noise            = False,
                                                                points_per_sim   = config.points_per_sim,
                                                                downsampled      = False)
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

# Pipeline Modes (Plot generation) ----------------------------------------------------------------------------------------#
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

    # Retrieve input features to compute 
    tab_feats_df, labels = tabular_features(process_df   = tab_data_df, 
                                            names        = contfeats + target + catfeats, 
                                            return_names = True,
                                            onehot       = False) 
    
    labels_names = [labels[name] for name in contfeats + target]

    logger.info(f"Features retrieved")
    logger.info(f"  - Continuos features   : {contfeats}")
    logger.info(f"  - Categorical features : {catfeats}")
    logger.info(f"  - Target               : {target}")

    # Plot full tabular feats:
    plot_generator.create_features_analysis(feats       = tab_feats_df[contfeats+target],
                                             names      = labels_names, 
                                             dataset    = dataset, 
                                             experiment = "full", 
                                             out_figs   = out_figs)

    # Plot tabular feats by envirioment
    for channel_code, env_name in [(0., 'fast'), (1., 'slow')]:
        
        mask = tab_feats_df[catfeats[0]] == channel_code
        
        if not np.any(mask):
            return
        
        plot_generator.create_features_analysis(feats      = tab_feats_df[mask][contfeats+target],
                                                names      = labels_names, 
                                                dataset    = dataset, 
                                                experiment = env_name, 
                                                out_figs   = out_figs + f"{env_name}/")
    logger.success("Plotting completed")
    logger.info(110*"_")

# Pipeline Modes (Distribution analysis) ---------------------------------------------------------------------------------#
def run_dist_mode(root_dir: str, dataset: str, data_path: str, aug_path: str, contfeats: list, catfeats: list, target: list,
                  figs_path: str, config: Any):
    """Run the distribution analysis mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Checking conservation of distributions for tabular training augmented dataset agains raw dataset")
    logger.info(110*"_")

    # Initialize processors
    processor      = SimulationProcessor(config)
    data_processor = DataProcessor(config)
    plot_generator = PlotGenerator(config)

    # Define features
    feature_names  = contfeats + catfeats + target
    
    # Load raw simulations
    simulations = processor.load_simulation_data(data_path, 
                                                 root_dir      = f"{root_dir}{dataset}/",
                                                 load_all_sims = False)
    
    # Classify and save simulations by environment type
    simulations_by_type = processor.classify_simulations_by_environment(simulations,
                                                                        cache_dir=f"{root_dir}{dataset}/")
    path_to_label       = {path: env_type
                           for env_type, paths in simulations_by_type.items()
                           for path in paths
                           }
    labels              = [path_to_label.get(path, np.nan) for path in simulations]

    logger.info(f"Loading raw features...")

    # Load raw data
    t_base, m_base, phy_base, path_list = data_processor.process_simulations(simulations, labels,
                                                                  augmentation = False, 
                                                                  apply_noise  = False, 
                                                                  n_virtual    = None, 
                                                                  verbose      = False,
                                                                  study_mode   = False)
    
    # Create dataframe in the same format as the processed one
    columns = config.feature_names + [config.target_name]
    raw_df  = pd.DataFrame(data= np.column_stack((t_base, phy_base, m_base)), columns= columns)
    
    # Retrieve input features to compute statistical test
    feats_raw, raw_names = tabular_features(raw_df, names=feature_names, return_names=True, onehot=False)

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
        
    feats_processed, feats_names = tabular_features(processed_df, names=feature_names, return_names=True, onehot=False)
    
    # Select continuous features inside the dataframes for comparison
    cont_features = contfeats + target  

    logger.info(110*"_")
    logger.info(f"Distribution comparison results (KS test + Wasserstein distance):")
    logger.info(110*"_")

    for sim_type in feats_processed[catfeats[0]].unique():
        
        logger.info(f"Envirioment type: {sim_type}")
        
        raw  = feats_raw[feats_raw[catfeats[0]] == sim_type]
        proc = feats_processed[feats_processed[catfeats[0]] == sim_type]

        for feature in cont_features:
            x = raw[feature].dropna().to_numpy()
            y = proc[feature].dropna().to_numpy()

            # KS test
            ks_stat, ks_pval = ks_2samp(x, y)

            # Wasserstein distance
            w_dist = wasserstein_distance(x, y)

            logger.info(f"{feature}")
            logger.info(f"  - KS statistic={ks_stat:.3f}, p-value={ks_pval:.2e}")
            logger.info(f"  - Wasserstein distance={w_dist:.3f}")
    
        logger.info(110*"-")
    
    # Create distribution comparison plots for continuous features
    plot_generator.create_features_histograms(feats_or      = feats_raw, feats_pr = feats_processed,
                                              labels        = cont_features,
                                              labels_names  = feats_names,
                                              out_figs      = figs_path + f"distcomp/")
    
    logger.success("Distribution analysis completed")
    logger.info(110*"_")

# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    
    # Set directory path manager
    path_manager = PathManagerDatasetPipeline(args.root_dir, args.dataset, args.exp_name, args.out_dir, args.fig_dir)
    
    # Setup configuration by dataset
    if args.dataset == "moccasurvey":
        dataconfig = create_processing_config(args.dataset)
        dataconfig.dataset_name = args.dataset
    else:
        # Fallback to default config for extensibility
        logger.warning(f"Dataset '{args.dataset}' not in predefined config map. Using default configuration.")
        dataconfig = create_processing_config(args.dataset)
        dataconfig.dataset_name = args.dataset
        
    # Run appropriate mode
    if args.mode == "study":
        run_study_mode(data_path = path_manager.data_path, 
                       out_figs  = path_manager.out_figs,
                       config    = dataconfig,
                       root_dir  = args.root_dir,
                       dataset   = args.dataset)
    
    elif args.mode == "feats":
        run_feats_mode(data_path    = path_manager.data_path, 
                       out_path     = path_manager.out_path, 
                       folds        = args.folds, 
                       augment      = args.aug, 
                       downsampled  = args.down,
                       config       = dataconfig,
                       path_manager = path_manager)  
    
    elif args.mode == "plot":
        run_plot_mode(datafile  = f"{path_manager.out_path}0_fold/train.csv",
                      contfeats = TabFeats["cont_feats"],
                      catfeats  = TabFeats["cat_feats"], 
                      target    = TabFeats["target_feat"],
                      out_figs  = path_manager.out_figs,
                      config    = dataconfig,
                      dataset   = args.dataset)

    elif args.mode == "dist":
        run_dist_mode(root_dir  = args.root_dir,
                      dataset   = args.dataset,
                      data_path = path_manager.data_path,
                      aug_path  = f"{path_manager.out_path}0_fold/train.csv",
                      contfeats = TabFeats["cont_feats"],
                      catfeats  = TabFeats["cat_feats"], 
                      target    = TabFeats["target_feat"],
                      figs_path = path_manager.out_figs,
                      config    = dataconfig)
    else:
        logger.error(f"Unknown mode: {args.mode}. Please choose from 'study', 'feats', 'plot', or 'dist'.")
        raise ValueError(f"Unknown mode: {args.mode}")
    
# Run the full Job --------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#