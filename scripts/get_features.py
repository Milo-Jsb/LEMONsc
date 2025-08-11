# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from tqdm        import tqdm
from typing      import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Custom functions --------------------------------------------------------------------------------------------------------#

# Directory
from src.utils.directory        import list_all_directories

# Data processing
from src.processing.format      import filter_and_downsample_hist2d

# Specifics to moccasurvey
from src.processing.moccasurvey import DEFAULT_CONFIG, load_moccasurvey_imbh_history, moccasurvey_dataset
from src.processing.moccasurvey import process_single_simulation, compute_cluster_features, determine_formation_channel

# Vizualization
from src.utils.vizualize        import plot_simulation_example, dataset_2Dhist_comparison, truncate_colormap
from src.utils.vizualize        import boxplot_features_with_points, classic_correlogram

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/experimental_dataset_preparation.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Configuration -----------------------------------------------------------------------------------------------------------#
@dataclass
class ProcessingFeaturesConfig:
    """Configuration class for the processing of get_features() parameters."""
    points_per_sim: Union[int, float] = 0.85
    n_virtual: int = 20
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    min_points_threshold: int = 1000
    histogram_bins: int = 200
    downsample_min_count: int = 50
    downsample_max_count: int = 150

DEFAULT_PARAMS = ProcessingFeaturesConfig()

# Arguments ---------------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Preparation of the simulated dataset")
    
    # Main mode of the script
    parser.add_argument("--mode", type=str, default="train",
                        choices=["study", "feats", "plot"],
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
                        choices = ["moccasurvey"],
                        help    = "Specific dataset to implement.")
    parser.add_argument("--exp_name", type=str, default="pof",
                        help = "Tag to name the dataset and output related elements.")
    parser.add_argument("--aug", action="store_true",
                        help="If perform augmentation with virtual simulations.")
    parser.add_argument("--down", action="store_true",
                        help="If perform downsampling by 2D histogram selection.")                    
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of folds to retrieve for kfold cross-validation.")
    
    # Target specifics
    parser.add_argument("--exp_type", type=str, default="point_mass", 
                        choices = ["point_mass", "delta_mass", "mass_rate"], 
                        help    = "Specific target expected of the dataset, also affect the possible features selected.")
        
    return parser.parse_args()

# Path Management ---------------------------------------------------------------------------------------------------------#
class PathManager:
    """Centralized path management for the pipeline."""
    
    def __init__(self, root_dir: str, dataset: str, exp_name: str, out_dir: str, fig_dir: str):
        self.data_path       = f"{root_dir}{dataset}/simulations/"
        self.out_path        = f"{out_dir}{exp_name}/{dataset}/"
        self.out_figs        = f"{fig_dir}{exp_name}/{dataset}/"
        self.stratified_path = f"{root_dir}{dataset}/"
        
        # Create directories
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.out_figs, exist_ok=True)
    
    def get_stratified_file_path(self, env_type: str) -> str:
        """Get path for stratified simulation files."""
        return os.path.join(self.stratified_path, f"{env_type}_simulations.txt")

# Data Processing Classes -------------------------------------------------------------------------------------------------#
class SimulationProcessor:
    """Efficient simulation data processor with caching."""
    
    def __init__(self, config: ProcessingFeaturesConfig):
        self.config = config
        self._cache = {}
    
    def load_simulation_data(self, data_path: str, verbose: bool = True) -> List[str]:
        """Load all simulation paths."""
        simulations = list_all_directories(data_path)
        if verbose:
            logger.info(f"Total simulation models available for analysis: {len(simulations)}")
        return simulations
    
    def load_single_simulation(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess a single simulation with caching."""
        cache_key = f"sim_{hash(path)}"
        
        if cache_key not in self._cache:
            imbh_history, system = load_moccasurvey_imbh_history(file_path=f"{path}/",
                                                                 init_conds_sim = False,
                                                                 init_conds_evo = True,
                                                                 verbose        = False
                                                                )
            
            imbh_df   = imbh_history[0].drop_duplicates(subset="time[Myr]").sort_values("time[Myr]").reset_index(drop=True)
            system_df = pd.merge_asof(imbh_df["time[Myr]"], system[0].sort_values("tphys").reset_index(drop=True),
                                      left_on="time[Myr]",
                                      right_on="tphys",
                                      direction="nearest"
                                     )
            
            self._cache[cache_key] = (imbh_df, system_df)
        
        return self._cache[cache_key]
    
    def load_single_simulation_example(self, simulations: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Load a single simulation for analysis and plotting."""
        sim_num = np.random.randint(0, len(simulations))
        imbh_df, system_df = self.load_single_simulation(simulations[sim_num])
        return imbh_df, system_df, simulations[sim_num]
    
    def classify_simulations_by_environment(self, simulations: List[str], exp_type: str) -> Dict[str, List[str]]:
        """Classify simulations by environment type efficiently."""
        simulations_by_type = {"FAST": [], "SLOW": [], "HYBRID": []}
        
        logger.info("Classifying simulations by environment type...")
        
        for path in tqdm(simulations, desc="Classifying simulations", unit="sim"):
            try:
                imbh_df, system_df = self.load_single_simulation(path)
                
                # Only process if sufficient points
                if len(imbh_df) <= self.config.min_points_threshold:
                    continue
                
                # Determine formation channel
                chform, _ = determine_formation_channel(system_df, imbh_df, None)
                simulations_by_type[chform].append(path)
                
            except Exception as e:
                logger.warning(f"Error classifying simulation {path}: {e}")
                continue
        
        # Log statistics
        for env_type, paths in simulations_by_type.items():
            logger.info(f"{env_type} simulations: {len(paths)}")
        
        return simulations_by_type
    
    def save_simulation_paths_by_type(self, simulations_by_type: Dict[str, List[str]], output_dir: str):
        """Save simulation paths to text files."""
        for env_type, paths in simulations_by_type.items():
            if paths:
                filepath = os.path.join(output_dir, f"{env_type.lower()}_simulations.txt")
                with open(filepath, 'w') as f:
                    f.write('\n'.join(paths) + '\n')
                logger.info(f"Saved {len(paths)} {env_type} simulation paths to {filepath}")

class DataPartitioner:
    """Efficient data partitioning with stratification support."""
    
    def __init__(self, config: ProcessingFeaturesConfig):
        self.config = config
    
    def create_random_partitions(self, simulations: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create random train/validation/test partitions."""
        simulations  = np.array(simulations)
        n_total      = len(simulations)
        shuffled_idx = np.random.permutation(n_total)

        n_train = int(n_total * self.config.train_split)
        n_val   = int(n_total * self.config.val_split)
        n_test  = n_total - n_train - n_val

        train_idx = shuffled_idx[:n_train]
        val_idx   = shuffled_idx[n_train:n_train + n_val]
        test_idx  = shuffled_idx[n_train + n_val:]

        train_simulations = simulations[train_idx].tolist()
        val_simulations   = simulations[val_idx].tolist()
        test_simulations  = simulations[test_idx].tolist()

        logger.info("Random partitioning:")
        logger.info(f"Training   : {len(train_simulations)}")
        logger.info(f"Validation : {len(val_simulations)}")
        logger.info(f"Testing    : {len(test_simulations)}")

        return train_simulations, val_simulations, test_simulations
    
    def create_stratified_partitions(self, path_manager: PathManager) -> Optional[Tuple[List[str], List[str], List[str]]]:
        """Create stratified partitions based on environment type."""
        simulations_by_type = {}
        
        # Load simulation paths by environment type
        for env_type in ["fast", "slow", "hybrid"]:
            filepath = path_manager.get_stratified_file_path(env_type)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    simulations_by_type[env_type.upper()] = [line.strip() for line in f.readlines()]
            else:
                logger.warning(f"File {filepath} not found. Using random partitioning instead.")
                return None
        
        # Create stratified partitions
        train_simulations, val_simulations, test_simulations = [], [], []
        
        for env_type, paths in simulations_by_type.items():
            if not paths:
                continue
                
            n_total      = len(paths)
            shuffled_idx = np.random.permutation(n_total)
            
            n_train = int(n_total * self.config.train_split)
            n_val   = int(n_total * self.config.val_split)
            n_test  = n_total - n_train - n_val
            
            train_idx = shuffled_idx[:n_train]
            val_idx   = shuffled_idx[n_train:n_train + n_val]
            test_idx  = shuffled_idx[n_train + n_val:]
            
            train_simulations.extend([paths[i] for i in train_idx])
            val_simulations.extend([paths[i] for i in val_idx])
            test_simulations.extend([paths[i] for i in test_idx])
            
            logger.info(f"{env_type} environment:")
            logger.info(f"  Training   : {len(train_idx)}")
            logger.info(f"  Validation : {len(val_idx)}")
            logger.info(f"  Testing    : {len(test_idx)}")
        
        logger.info("Total stratified partitions:")
        logger.info(f"Training   : {len(train_simulations)}")
        logger.info(f"Validation : {len(val_simulations)}")
        logger.info(f"Testing    : {len(test_simulations)}")
        
        return train_simulations, val_simulations, test_simulations

class DataProcessor:
    """Optimized data processing"""
    
    def __init__(self, config: ProcessingFeaturesConfig):
        self.config = config
    
    def process_simulations(self, simulations: List[str], exp_type: str, 
                           augmentation: bool = False, apply_noise: bool = False, 
                           n_virtual: Optional[int] = None, verbose: bool = True) -> Tuple[List, List, List]:
        """Process simulations efficiently with reduced memory allocations."""
        time_list, mass_list, phy_list = [], [], []
        config = DEFAULT_CONFIG[exp_type]
        
        # Statistics tracking
        stats = {
            'used_sims': 0,
            'ignored_sims': 0,
            'environment': [],
            'points_per_sim': []
        }

        processor = SimulationProcessor(self.config)
        
        for path in tqdm(simulations, desc="Processing simulations", unit="sim"):
            try:
                imbh_df, system_df = processor.load_single_simulation(path)
                
                # Skip if insufficient points
                if len(imbh_df) <= self.config.min_points_threshold:
                    stats['ignored_sims'] += 1
                    continue
                
                # Determine formation channel
                chform, virsims = determine_formation_channel(system_df, imbh_df, n_virtual)
                
                # Process simulation
                feats, masses, idxs = process_single_simulation(imbh_df=imbh_df, system_df=system_df, config=config,
                                                                experiment_type  = exp_type, 
                                                                points_per_sim   = self.config.points_per_sim, 
                                                                augment          = augmentation, 
                                                                apply_noise      = apply_noise,
                                                                n_virtual        = virsims,
                                                                time_norm_factor = None,
                                                                norm_target      = False,
                                                                log10_target     = False
                                                            )
                
                # Track statistics
                stats['used_sims'] += 1
                stats['points_per_sim'].append(len(feats))
                stats['environment'].append(chform)
                
                # Extend lists efficiently
                time_list.extend(feats[:, 0])
                mass_list.extend(masses)
                phy_list.extend(feats[:, 1:])
                
            except Exception as e:
                stats['ignored_sims'] += 1
                logger.warning(f"Error processing simulation {path}: {e}")
                continue
        
        # Report statistics
        if verbose:
            self._report_statistics(stats, len(simulations))
        
        return time_list, mass_list, phy_list
    
    def _report_statistics(self, stats: Dict, total_sims: int):
        """Report processing statistics."""
        if stats['points_per_sim']:
            avg_points = np.mean(stats['points_per_sim'])
            std_points = np.std(stats['points_per_sim'])
        else:
            avg_points = std_points = 0
        
        fast_count = stats['environment'].count("FAST")
        hybrid_count = stats['environment'].count("HYBRID")
        slow_count = stats['environment'].count("SLOW")
        
        logger.info(f"Simulation processing completed:")
        logger.info(f"  - Total simulations processed   : {total_sims}")
        logger.info(f"  - Simulations used              : {stats['used_sims']}")
        logger.info(f"  - Simulations ignored           : {stats['ignored_sims']}")
        logger.info(f"  - FAST formation channel sims   : {fast_count}")
        logger.info(f"  - HYBRID formation channel sims : {hybrid_count}")
        logger.info(f"  - SLOW formation channel sims   : {slow_count}")
        logger.info(f"  - Average points per simulation : {avg_points:.1f} ± {std_points:.1f}")

class DownsamplingProcessor:
    """Optimized downsampling with channel separation."""
    
    def __init__(self, config: ProcessingFeaturesConfig):
        self.config = config
    
    def perform_downsampling(self, t_augm: List, m_augm: List, phy_augm: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform downsampling with formation channel separation."""
        # Convert to numpy arrays once
        t_array   = np.array(t_augm)
        m_array   = np.array(m_augm)
        phy_array = np.array(phy_augm)
        
        # Initialize results
        t_parts, m_parts, phy_parts = [], [], []
        
        # Process each formation channel
        for channel_code, channel_name in [(0, 'FAST'), (1, 'SLOW'), (2, 'HYBRID')]:
            mask = phy_array[:, 10] == channel_code
            if not np.any(mask):
                continue
                
            # Extract channel data
            t_channel   = t_array[mask]
            m_channel   = m_array[mask]
            phy_channel = phy_array[mask]
            
            # Preprocess
            time_feat = np.log10(t_channel + 1)
            mass_feat = m_channel
            
            # Remove inf/nan values
            valid_mask = np.isfinite(time_feat) & np.isfinite(mass_feat)
            if not np.all(valid_mask):
                time_feat   = time_feat[valid_mask]
                mass_feat   = mass_feat[valid_mask]
                t_channel   = t_channel[valid_mask]
                m_channel   = m_channel[valid_mask]
                phy_channel = phy_channel[valid_mask]
            
            if len(time_feat) == 0:
                continue
            
            # Perform downsampling
            H1, xedges, yedges = np.histogram2d(time_feat, mass_feat, bins=[self.config.histogram_bins, self.config.histogram_bins])
            
            idxs = filter_and_downsample_hist2d(time_feat, mass_feat, H1, xedges, yedges, 
                                                min_count = self.config.downsample_min_count, 
                                                max_count = self.config.downsample_max_count,
                                                seed      = 42
                                                )
            
            if len(idxs) > 0:
                t_parts.append(t_channel[idxs])
                m_parts.append(m_channel[idxs])
                phy_parts.append(phy_channel[idxs])
        
        # Concatenate results
        if t_parts:
            return np.concatenate(t_parts), np.concatenate(m_parts), np.concatenate(phy_parts)
        else:
            return np.array([]), np.array([]), np.array([])

class PlotGenerator:
    """Optimized plotting of files."""
    
    def __init__(self, config: ProcessingFeaturesConfig, cmap:Optional[str] = None):
        self.config     = config
        self.cmap_trunc = truncate_colormap("CMRmap_r" if cmap is None else cmap)
    
    def create_comparison_plots(self, t_base: List, m_base: List, phy_base: List,
                               t_augm: List, m_augm: List, phy_augm: List,
                               t_down: List, m_down: List, phy_down: List,
                               out_figs: str):
        """Create all comparison plots efficiently."""
        # Convert to numpy arrays once
        t_base_arr   = np.array(t_base)
        m_base_arr   = np.array(m_base)
        phy_base_arr = np.array(phy_base)
        t_augm_arr   = np.array(t_augm)
        m_augm_arr   = np.array(m_augm)
        phy_augm_arr = np.array(phy_augm)
        t_down_arr   = np.array(t_down)
        m_down_arr   = np.array(m_down)
        phy_down_arr = np.array(phy_down)
        
        # Create full dataset plot
        self._create_single_plot(t_base_arr, m_base_arr, t_augm_arr, m_augm_arr, 
                                t_down_arr, m_down_arr, "full", out_figs)
        
        # Create environment-specific plots
        for channel_code, env_name in [(0, 'fast'), (1, 'slow'), (2, 'hybrid')]:
            self._create_environment_plot(t_base_arr, m_base_arr, phy_base_arr,
                                        t_augm_arr, m_augm_arr, phy_augm_arr,
                                        t_down_arr, m_down_arr, phy_down_arr,
                                        channel_code, env_name, out_figs)
    
    def _create_single_plot(self, t_base: np.ndarray, m_base: np.ndarray,
                           t_augm: np.ndarray, m_augm: np.ndarray,
                           t_down: np.ndarray, m_down: np.ndarray,
                           name: str, out_figs: str):
        """Create a single comparison plot."""
        dataset_2Dhist_comparison(times_base = t_base, masses_base = m_base, 
                                 times_aug   = t_augm, masses_aug  = m_augm,
                                 times_filt  = t_down, masses_filt = m_down,
                                 name     = name, 
                                 bins     = 200, 
                                 cmap     = self.cmap_trunc, 
                                 savepath = out_figs)
    
    def _create_environment_plot(self, t_base: np.ndarray, m_base: np.ndarray, phy_base: np.ndarray,
                                t_augm: np.ndarray, m_augm: np.ndarray, phy_augm: np.ndarray,
                                t_down: np.ndarray, m_down: np.ndarray, phy_down: np.ndarray,
                                channel_code: int, env_name: str, out_figs: str):
        """Create environment-specific comparison plot."""
        mask = phy_base[:, 10] == channel_code
        
        if not np.any(mask):
            return
        
        # Extract environment-specific data
        t_base_env = t_base[mask]
        m_base_env = m_base[mask]
        
        mask_augm = phy_augm[:, 10] == channel_code
        t_augm_env = t_augm[mask_augm]
        m_augm_env = m_augm[mask_augm]
        
        mask_down = phy_down[:, 10] == channel_code
        t_down_env = t_down[mask_down]
        m_down_env = m_down[mask_down]
        
        self._create_single_plot(t_base_env, m_base_env, t_augm_env, m_augm_env,
                                t_down_env, m_down_env, env_name, out_figs)
    
    def _create_features_analysis(feats, names, dataset, experiment, out_figs):

        # Boxplot of the features selected        
        boxplot_features_with_points(features= feats, feature_names= names,
                                     path_save     = out_figs,
                                     name_file     = experiment,
                                     dataset_name  = dataset,
                                     point_color   = "wheat",
                                     figsize       = (len(feats)*4,6))
        
        # Correlation plot of the features selected (Spearman coefficient)        
        classic_correlogram(df= feats, method= "spearman", path_save=out_figs,
                            name_file    = experiment,
                            dataset_name = dataset,
                            labels       = names,
                            cmap         = "RdGy")

# Dataset Generation ------------------------------------------------------------------------------------------------------#
def save_dataset_to_csv(data: np.ndarray, columns: List[str], target_data: np.ndarray, 
                       target_name: str, filepath: str, additional_columns: Optional[Dict] = None):
    """Save dataset to CSV file efficiently."""
    df = pd.DataFrame(data=data, columns=columns, index=None)
    df[target_name] = target_data
    
    if additional_columns:
        for col_name, col_data in additional_columns.items():
            df[col_name] = col_data
    
    df.to_csv(filepath, index=False)
    logger.info(f"Dataset saved to: {filepath}")

# Pipeline Modes ---------------------------------------------------------------------------------------------------------#
def run_study_mode(data_path: str, out_figs: str, exp_type: str, config: ProcessingFeaturesConfig = DEFAULT_PARAMS):
    """Optimized study mode pipeline."""
    logger.info(f"Study of simulations available for moccasurvey dataset")
    
    # Initialize processors
    processor              = SimulationProcessor(config)
    data_processor         = DataProcessor(config)
    downsampling_processor = DownsamplingProcessor(config)
    plot_generator         = PlotGenerator(config)
    
    # Load simulations
    simulations = processor.load_simulation_data(data_path)
    
    # Single simulation example
    imbh_df, system_df, sim_path = processor.load_single_simulation_example(simulations)
    ifeats = compute_cluster_features(system_df, imbh_df, 
                                       imbh_df["time[Myr]"].max() - imbh_df["time[Myr]"].min(), 
                                       apply_noise=False)
    
    # Plot single simulation example
    plot_simulation_example(imbh_df, target_types=["point_mass", "delta_mass", "mass_rate"], 
                           save_path=out_figs,
                           t_cc=ifeats["tcc"], t_coll=ifeats["tcoll"], 
                           t_relax=ifeats["trelax"], M_crit=ifeats["mcrit"],
                           rho_half=ifeats["rho_half"])
    
    # Classify and save simulations by environment type
    simulations_by_type = processor.classify_simulations_by_environment(simulations, exp_type)
    processor.save_simulation_paths_by_type(simulations_by_type, f"{args.root_dir}{args.dataset}/")
    
    # Process simulations
    t_base, m_base, phy_base = data_processor.process_simulations(
        simulations, exp_type, augmentation=False, apply_noise=False, n_virtual=None, verbose=True
    )
    
    t_augm, m_augm, phy_augm = data_processor.process_simulations(
        simulations, exp_type, augmentation=True, apply_noise=True, 
        n_virtual=config.n_virtual, verbose=False
    )
    
    # Downsampling analysis
    t_down, m_down, phy_down = downsampling_processor.perform_downsampling(t_augm, m_augm, phy_augm)
    
    # Create all comparison plots
    plot_generator.create_comparison_plots(t_base, m_base, phy_base,
                                          t_augm, m_augm, phy_augm,
                                          t_down, m_down, phy_down, 
                                          out_figs)
    
    logger.success("Study mode completed")

def run_feats_mode(data_path: str, out_path: str, exp_type: str, folds: int, 
                  augment: bool, norm_target: bool, log_target: bool, 
                  downsampled: bool = False, config: ProcessingFeaturesConfig = DEFAULT_PARAMS):
    """Optimized features generation mode pipeline."""
    logger.info(f"Generating tabular features from moccasurvey simulations")
    
    # Initialize processors
    processor    = SimulationProcessor(config)
    partitioner  = DataPartitioner(config)
    path_manager = PathManager(args.root_dir, args.dataset, args.exp_name, args.out_dir, args.fig_dir)
    
    # Load simulations
    simulations = processor.load_simulation_data(data_path)
    
    # Try stratified partitioning first
    stratified_partitions = partitioner.create_stratified_partitions(path_manager)
    
    if stratified_partitions is not None:
        train_simulations, val_simulations, test_simulations = stratified_partitions
        logger.info("Using stratified partitioning based on environment type")
    else:
        train_simulations, val_simulations, test_simulations = partitioner.create_random_partitions(simulations)
        logger.info("Using random partitioning (stratified files not found)")
    
    # Generate folds
    for fold in range(folds):
        fold_path = os.path.join(out_path, f"{fold}_fold/")
        os.makedirs(fold_path, exist_ok=True)
        
        # Training data
        xtrain_info, ytrain_info = moccasurvey_dataset(simulations_path = train_simulations, 
                                                       experiment_type  = exp_type,
                                                       augmentation     = augment, 
                                                       norm_target      = norm_target, 
                                                       log10_target     = log_target,
                                                       logger           = logger, 
                                                       points_per_sim   = config.points_per_sim,
                                                       n_virtual        = config.n_virtual, 
                                                       downsampled      = downsampled
                                                    )
        
        save_dataset_to_csv(xtrain_info[0], xtrain_info[1], ytrain_info[0], ytrain_info[1][0], f"{fold_path}train.csv")
        
        logger.info(f"Fold {fold} Train - Stored at {fold_path}")
        
        # Validation data
        xval_info, yval_info = moccasurvey_dataset(simulations_path = val_simulations, 
                                                   experiment_type  = exp_type,
                                                   augmentation     = augment, 
                                                   norm_target      = norm_target, 
                                                   log10_target     = log_target,
                                                   logger           = logger, 
                                                   points_per_sim   = config.points_per_sim,
                                                   n_virtual        = config.n_virtual, 
                                                   downsampled      = downsampled)
        
        save_dataset_to_csv(xval_info[0], xval_info[1], yval_info[0], yval_info[1][0], f"{fold_path}val.csv")
        
        logger.info(f"Fold {fold} Val - Stored at {fold_path}")
    
    # Testing data
    xtest_info, ytest_info, sim_paths = moccasurvey_dataset(simulations_path = test_simulations, 
                                                            experiment_type  = exp_type,
                                                            norm_target      = norm_target, 
                                                            log10_target     = log_target, 
                                                            logger           = logger,
                                                            test_partition   = True, 
                                                            downsampled      = False)
    
    additional_columns = {"or_sim_path": sim_paths, "tag": "moccasurvey"}
    save_dataset_to_csv(xtest_info[0], xtest_info[1], ytest_info[0], ytest_info[1][0], f"{out_path}test.csv", 
                        additional_columns)
    
    logger.success("Features generation completed")

def run_plot_mode(datafile, out_figs: str):
    """Run the plotting mode pipeline."""
    logger.info(f"Plotting information for tabular training moccasurvey dataset")
    
    # Load files
    tab_features_df = pd.read_csv(datafile, index_col=False)

    # Retrieve input features to compute 
    tab_feats_df    = tabular_features() 



    logger.success("Plotting completed")

# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    # Setup path manager
    path_manager = PathManager(args.root_dir, args.dataset, args.exp_name, args.out_dir, args.fig_dir)
    
    # Run appropriate mode
    if args.mode == "study":
        run_study_mode(path_manager.data_path, path_manager.out_figs, args.exp_type)
    elif args.mode == "feats":
        run_feats_mode(data_path=path_manager.data_path, out_path=path_manager.out_path, 
                      exp_type=args.exp_type, folds=args.folds, 
                      augment=args.aug, norm_target=False, log_target=False, 
                      downsampled=args.down)  
    elif args.mode == "plot":
        run_plot_mode(path_manager.out_figs)

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#