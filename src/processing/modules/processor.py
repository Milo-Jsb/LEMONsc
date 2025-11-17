# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
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
from src.processing.dataset             import process_single_simulation

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
            'steady_count' : self.environment.count("STEADY"),
            'slow_count'   : self.environment.count("SLOW")
        }
        
        # Add augmentation statistics if enabled
        if self.augmentation_enabled:
            summary['virtual_sims_generated'] = self.virtual_sims_generated
            summary['total_effective_sims'] = self.used_sims + self.virtual_sims_generated
            
            if self.points_per_original:
                summary['avg_points_per_original'] = float(np.mean(self.points_per_original))
                summary['std_points_per_original'] = float(np.std(self.points_per_original))
                summary['avg_virtual_per_original'] = float(self.virtual_sims_generated / len(self.points_per_original))
        
        return summary

# Main Data Processor Class --------------------------------------------------------------------------------------------#
class DataProcessor:
    """Optimized data processing for MOCCA simulations."""
    
    def __init__(self, config):
        self.config     = config
        self._processor = SimulationProcessor(config)
    
    def process_simulations(self, simulations : List[str], labels: Optional[List[int]] = None,
                            augmentation: bool = False, 
                            apply_noise : bool = False, 
                            n_virtual   : Optional[int] = None, 
                            verbose     : bool = True
                            ) -> Tuple[List, List, List]:
        """
        Process simulations efficiently with reduced memory allocations.
        
        Parameters:
            simulations  : List of simulation paths
            labels       : Optional list of environment labels for each simulation
            augmentation : Whether to apply data augmentation
            apply_noise  : Whether to apply noise to features
            n_virtual    : Number of virtual simulations per real simulation
            verbose      : Whether to print statistics
            
        Returns:
            Tuple of (time_list, mass_list, phy_list)
        """
        time_list, mass_list, phy_list = [], [], []
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
                chform = determine_formation_channel(system_df, imbh_df, n_virtual)
                
                feats, masses, idxs = self._process_single(imbh_df, system_df, iconds_dict, self.config, label, 
                                                           augmentation, 
                                                           apply_noise)
                
                # Track statistics and accumulate results
                # If augmentation is enabled, pass n_virtual to track virtual simulations
                stats.increment_used(chform, len(feats), is_augmented=augmentation, n_virtual=n_virtual)
                self._accumulate_results(feats, masses, time_list, mass_list, phy_list)
                
            except Exception as e:
                stats.increment_ignored()
                logger.warning(f"Error processing simulation {path}: {e}")
                continue
        
        # Report statistics if requested
        if verbose:
            self._report_statistics(stats, len(simulations))
        
        return time_list, mass_list, phy_list
    
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
        return process_single_simulation(imbh_df= imbh_df, system_df = system_df, meta_dict = iconds_dict,
                                         config         = config,
                                         environment    = label,
                                         points_per_sim = self.config.points_per_sim, 
                                         augment        = augmentation, 
                                         noise          = apply_noise,
                                         n_virtual      = self.config.n_virtual,
                                    )
    
    def _accumulate_results(self, feats: np.ndarray, masses: np.ndarray, time_list : List, mass_list: List, 
                            phy_list  : List) -> None:
        """Accumulate features into output lists."""
        time_list.extend(feats[:, 0])
        mass_list.extend(masses)
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
        logger.info(f"  - STEADY formation channel sims : {summary['steady_count']}")
        logger.info(f"  - SLOW formation channel sims   : {summary['slow_count']}")
        
        # Report points statistics based on augmentation state
        if 'avg_points_per_original' in summary:
            logger.info(f"  - Average points per original   : {summary['avg_points_per_original']:.1f} ± {summary['std_points_per_original']:.1f}")
            logger.info(f"  - Average points per output     : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        else:
            logger.info(f"  - Average points per simulation : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        
        logger.info(110*"_")
        
#--------------------------------------------------------------------------------------------------------------------------#