# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import os

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing      import Any, Dict, List, Optional, Tuple, Union
from tqdm        import tqdm
from loguru      import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.modules.simulations import SimulationProcessor
from src.processing.features            import determine_formation_channel
from src.processing.filters             import efficiency_mass_ratio_relation

# Import dataset-specific processing functions
from src.processing.constructors.moccasurvey import process_single_mocca_simulation

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/processor_outputs.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Configuration for processing features -----------------------------------------------------------------------------------#
@dataclass
class ProcessingStats:
    """Statistics tracking for simulation processing."""
    used_sims      : int = 0
    ignored_sims   : int = 0
    environment    : List[str] = field(default_factory=list)
    points_per_sim : List[int] = field(default_factory=list)
    
    # Augmentation-specific tracking
    virtual_sims_generated : int = 0
    points_per_original    : List[int] = field(default_factory=list)
    augmentation_enabled   : bool = False
    
    def increment_used(self, chform: str, num_points: int, is_augmented: bool = False, n_virtual: Optional[int] = None):
        """Increment used simulation counter and record stats."""
        self.used_sims += 1
        self.environment.append(chform)
        self.points_per_sim.append(num_points)
        
        # Track augmentation statistics
        if is_augmented and n_virtual is not None:
            self.augmentation_enabled = True
            self.virtual_sims_generated += n_virtual
            self.points_per_original.append(num_points)
    
    def increment_ignored(self):
        """Increment ignored simulation counter."""
        self.ignored_sims += 1
    
    def get_summary(self) -> Dict[str, Union[int, float]]:
        """Get statistical summary of processing."""
        if self.points_per_sim:
            avg_points = float(np.mean(self.points_per_sim))
            std_points = float(np.std(self.points_per_sim))
        else:
            avg_points = std_points = 0.0
        
        summary = {
            'used_sims'    : self.used_sims,
            'ignored_sims' : self.ignored_sims,
            'avg_points'   : avg_points,
            'std_points'   : std_points,
            'fast_count'   : self.environment.count("FAST"),
            'slow_count'   : self.environment.count("SLOW")
                }
        
        # Add augmentation statistics if enabled
        if self.augmentation_enabled:
            summary['virtual_sims_generated'] = self.virtual_sims_generated
            summary['total_effective_sims']   = self.used_sims + self.virtual_sims_generated
            
            if self.points_per_original:
                summary['avg_points_per_original']  = float(np.mean(self.points_per_original))
                summary['std_points_per_original']  = float(np.std(self.points_per_original))
                summary['avg_virtual_per_original'] = float(self.virtual_sims_generated / len(self.points_per_original))
        
        return summary

# Main Data Processor Class -----------------------------------------------------------------------------------------------#
class DataProcessor:
    """Optimized data processing for simulations."""
    
    def __init__(self, config):
        self.config     = config
        self._processor = SimulationProcessor(config)
    
    def process_simulations(self, simulations : List[str], labels: Optional[List[int]] = None,
                            augmentation: bool = False, 
                            apply_noise : bool = False, 
                            n_virtual   : Optional[int] = None, 
                            study_mode  : bool = False,
                            verbose     : bool = True
                            ) -> Tuple[List, List, List, List]:
        """
        ____________________________________________________________________________________________________________________
        Process simulations efficiently with reduced memory allocations.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> simulations  (list)         : List of simulation paths
        -> labels       (list or None) : Optional list of environment labels for each simulation
        -> augmentation (bool)         : Whether to apply data augmentation
        -> apply_noise  (bool)         : Whether to apply noise to features
        -> n_virtual    (int or None)  : Number of virtual simulations per real simulation
        -> study_mode   (bool)         : Whether to keep simulations separated for study mode
        -> verbose      (bool)         : Whether to print statistics
        ____________________________________________________________________________________________________________________    
        Returns:
            Tuple of (time_list, mass_list, phy_list, path_list)
        ____________________________________________________________________________________________________________________
        Notes:
        - Uses ProcessingStats to track statistics during processing.
        - Handles exceptions per simulation to ensure robust processing.
        - Reports detailed statistics if verbose=True.
        - Check processor to look for specific details of processing.
        ____________________________________________________________________________________________________________________
        """
        time_list, mass_list, phy_list, path_list = [], [], [], []
        stats = ProcessingStats()
        
        for idx, path in enumerate(tqdm(simulations, desc="Processing simulations", unit="sim")):
            try:
                # Load simulation data
                imbh_df, system_df, iconds_dict = self._processor.load_single_simulation(path)
                label = labels[idx] if (labels is not None and idx < len(labels)) else None
                
                # Validate simulation has sufficient points
                if not self._validate_simulation(imbh_df):
                    stats.increment_ignored()
                    continue
                
                # Determine formation channel and process
                chform = determine_formation_channel(imbh_df          = imbh_df, 
                                                     mass_column_name = self.config.mass_column_imbh,
                                                     time_column_name = self.config.time_column_imbh)
                
                feats, masses, idxs = self._process_single(imbh_df, system_df, iconds_dict, self.config, label, 
                                                           augmentation, 
                                                           apply_noise)
                
                # Track statistics and accumulate results
                stats.increment_used(chform, len(feats), is_augmented=augmentation, n_virtual=n_virtual)
                self._accumulate_results(feats, masses, time_list, mass_list, phy_list, path_list, path, study_mode)
                
            except Exception as e:
                stats.increment_ignored()
                logger.warning(f"Error processing simulation {path}: {e}")
                continue
        
        # Report statistics if requested
        if verbose:
            self._report_statistics(stats, len(simulations))
        
        return time_list, mass_list, phy_list, path_list
    
    def select_suitable_sims(self, simulations: List[str], simulations_by_type: Dict[str, List[str]], 
                             out_path: str) -> Dict:
        """Retrieve suitable simulations based on efficiency-mass ratio filtering"""
        
        # Check if division already exist
        valid_sims_path = os.path.join(out_path, 'valid_simulations.txt')
        outliers_path   = os.path.join(out_path, 'outliers.txt')
        
        # If both files exist, load them and skip filtering
        if os.path.exists(valid_sims_path) and os.path.exists(outliers_path):
            logger.info("Found existing filter files. Loading from disk...")
        
            # Load valid simulations
            with open(valid_sims_path, 'r') as f:
                valid_sim_paths = [line.strip() for line in f if line.strip()]
        
            # Load outliers
            with open(outliers_path, 'r') as f:
                outlier_paths = [line.strip() for line in f if line.strip()]
                    
            logger.info(f"Loaded {len(valid_sim_paths)} valid simulations from: {valid_sims_path}")
            logger.info(f"Loaded {len(outlier_paths)} outliers from: {outliers_path}")
            
            simulations = valid_sim_paths

        # If not, proceed with filtering
        else:
            logger.info("No existing filter files found. Proceeding with filtering...")
            
        # Retrieve labels
        path_to_label = {path: env_type
                        for env_type, paths in simulations_by_type.items()
                        for path in paths
                        }

        labels = [path_to_label.get(path, np.nan) for path in simulations]
        
        # First check which simulations can be retrieved with the actual configuration
        t_base, m_base, phy_base, path_list = self.process_simulations(simulations, labels, 
                                                                       augmentation = False, 
                                                                       apply_noise  = False,
                                                                       n_virtual    = None, 
                                                                       study_mode   = True, 
                                                                       verbose      = False)
        
        # Update label list
        filt_labels = [path_to_label.get(path, np.nan) for path in path_list]

        # Now filter simulations based on their values of efficiency-mass_ratio
        filter_output = efficiency_mass_ratio_relation(mmo_mass        = m_base, 
                                                       physical_params = phy_base, 
                                                       path_list       = path_list,
                                                       labels_list     = filt_labels,
                                                       logger          = logger)
        
        # Ensure output directory exists
        os.makedirs(out_path, exist_ok=True)
        
        if not os.path.exists(valid_sims_path) and not os.path.exists(outliers_path):
            
            # Save filtered simulation paths of usable simulations
            with open(valid_sims_path, 'w') as f:
                for sim_path in filter_output['valid_sims']['paths']:
                    f.write(f"{sim_path}\n")
            
            logger.info(f"Filtered simulation paths saved to: {valid_sims_path}")
            
            # Save outliers paths (overwrite mode)
            with open(outliers_path, 'w') as f:
                for sim_path in filter_output['outliers']['paths']:
                    f.write(f"{sim_path}\n")
            
            logger.info(f"Outliers paths saved to: {out_path+'outliers.txt'}")
        
        return filter_output
    
    def _validate_simulation(self, imbh_df: pd.DataFrame) -> bool:
        """Check if simulation has sufficient points for processing."""
        
        return len(imbh_df) > self.config.min_points_threshold
    
    def _process_single(self, imbh_df: pd.DataFrame, system_df: pd.DataFrame, iconds_dict: Dict,
                       config      : Any,
                       label       : Optional[str],
                       augmentation: bool,
                       apply_noise : bool,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single simulation and return features, masses, and indices."""
        
        # Process based on dataset type
        if (self.config.dataset_name == "moccasurvey"):
        
            return process_single_mocca_simulation(imbh_df= imbh_df, system_df = system_df, meta_dict = iconds_dict,
                                                    config         = config,
                                                    environment    = label,
                                                    points_per_sim = self.config.points_per_sim, 
                                                    augment        = augmentation, 
                                                    noise          = apply_noise,
                                                    n_virtual      = self.config.n_virtual,
                                                )
        
        else:
            raise NotImplementedError(f"Dataset '{self.config.dataset_name}' not supported in processor.")
        
    def _accumulate_results(self, feats: np.ndarray, masses: np.ndarray, time_list : List, mass_list: List, 
                            phy_list   : List, 
                            path_list  : List, 
                            sim_path   : str, 
                            study_mode : bool = False) -> None:
        """Accumulate features into output lists."""
        if study_mode:
            # Keep simulations separated - append as sublists and track corresponding simulation path
            time_list.append(list(np.clip(feats[:, 0], a_min=0, a_max=None)))
            mass_list.append(list(np.clip(masses, a_min=0, a_max=None)))
            phy_list.append(feats[:, 1:].tolist())
            path_list.append(sim_path) 
        
        else:
            # Flatten - extend the lists directly (no path tracking)
            time_list.extend(np.clip(feats[:, 0], a_min=0, a_max=None))
            mass_list.extend(np.clip(masses, a_min=0, a_max=None))
            phy_list.extend(feats[:, 1:])
    
    def _report_statistics(self, stats: ProcessingStats, total_sims: int) -> None:
        """Report processing statistics to logger."""
        
        summary = stats.get_summary()
        
        logger.info(110*"_")
        logger.info("Simulation processing completed:")
        logger.info(110*"_")
        logger.info(f"  - Total simulations processed   : {total_sims}")
        logger.info(f"  - Simulations used              : {summary['used_sims']}")
        logger.info(f"  - Simulations ignored           : {summary['ignored_sims']}")
        
        # Add augmentation-specific statistics if enabled
        if 'virtual_sims_generated' in summary:
            logger.info(f"  - Virtual simulations generated : {summary['virtual_sims_generated']}")
            logger.info(f"  - Total effective simulations   : {summary['total_effective_sims']}")
            logger.info(f"  - Avg virtual per original sim  : {summary['avg_virtual_per_original']:.1f}")
        
        logger.info(f"  - FAST formation channel sims   : {summary['fast_count']}")
        logger.info(f"  - SLOW formation channel sims   : {summary['slow_count']}")
        
        # Report points statistics based on augmentation state
        if 'avg_points_per_original' in summary:
            logger.info(f"  - Average points per original   : {summary['avg_points_per_original']:.1f} ± \
                        {summary['std_points_per_original']:.1f}")
            logger.info(f"  - Average points per output     : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        else:
            logger.info(f"  - Average points per simulation : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        
        logger.info(110*"_")
        
#--------------------------------------------------------------------------------------------------------------------------#