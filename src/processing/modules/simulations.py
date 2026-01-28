# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import os

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Dict, List, Tuple
from tqdm   import tqdm
from loguru import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory                     import list_all_directories
from src.processing.features                 import determine_formation_channel
from src.processing.constructors.moccasurvey import load_moccasurvey_imbh_history

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/simulations_outputs.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Process individual simulations ------------------------------------------------------------------------------------------#
class SimulationProcessor:
    """Efficient simulation data processor with caching."""
    
    def __init__(self, config):
        
        # Store configuration and initialize cache
        self.config = config
        self._cache = {}
    
    def load_simulation_data(self, data_path: str, root_dir: str ="./rawdata/", load_all_sims : bool = True,
                             verbose : bool = True) -> List[str]:
        """Load simulation paths from valid_simulations.txt if available, otherwise scan directory."""
        
        # Check for valid_simulations.txt file
        valid_sims_file = os.path.join(root_dir, "valid_simulations.txt")
        
        # File exists and user does not want to load all sims, read from file
        if os.path.exists(valid_sims_file) and not load_all_sims:
            if verbose:
                logger.info(f"Loading simulation paths from: {valid_sims_file}")
            
            with open(valid_sims_file, 'r') as f:
                simulations = [line.strip() for line in f if line.strip()]
            
            if verbose:
                logger.info(f"Loaded {len(simulations)} simulation paths from {root_dir}valid_simulations.txt")
        
        # Either file does not exist or user wants to load all sims, scan directory
        else:
            if verbose:
                logger.info(f"Scanning directory: {data_path}")
            
            simulations = list_all_directories(data_path)
            
            if verbose:
                logger.info(f"Total simulation models available for analysis: {len(simulations)}")
        
        return simulations
    
    def load_single_simulation(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess a single simulation with caching."""
        
        # Create a unique cache key based on the simulation path
        cache_key = f"sim_{hash(path)}"
        
        # Check if simulation data is already cached
        if cache_key not in self._cache:
            
            # Load data based on dataset type
            if self.config.dataset_name == "moccasurvey":
                imbh_history, system = load_moccasurvey_imbh_history(file_path=f"{path}/",
                                                                    init_conds_sim = True,
                                                                    init_conds_evo = True,
                                                                    verbose        = False
                                                                    )
                
                imbh_df     = imbh_history[0].drop_duplicates(subset=self.config.time_column_imbh
                                                              ).sort_values(self.config.time_column_imbh
                                                                            ).reset_index(drop=True)
                iconds_dict = imbh_history[1]  
                system_df   = pd.merge_asof(imbh_df[self.config.time_column_imbh], 
                                            system[0].sort_values(self.config.time_column_system).reset_index(drop=True),
                                            left_on   = self.config.time_column_imbh,
                                            right_on  = self.config.time_column_system,
                                            direction = "nearest"
                                        )
            
            self._cache[cache_key] = (imbh_df, system_df, iconds_dict)
        
        return self._cache[cache_key]
    
    def load_single_simulation_example(self, simulations: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
        """Load a single simulation for analysis and plotting."""
        
        # Chose a random simulation
        sim_num = np.random.randint(0, len(simulations))
        
        # Load the simulation data
        imbh_df, system_df, iconds_dict = self.load_single_simulation(simulations[sim_num])
        
        return imbh_df, system_df, simulations[sim_num], iconds_dict
    
    def classify_simulations_by_environment(self, simulations: List[str], cache_dir: str = "./rawdata/"
                                            ) -> Dict[str, List[str]]:
        """Classify simulations by environment type efficiently."""
        
        # Set up storage for classified simulations
        simulations_by_type = {"FAST": [], "SLOW": []}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
        # Try to load from cache first
        cache_files = {
            "FAST"   : os.path.join(cache_dir, "fast_simulations.txt"),
            "SLOW"   : os.path.join(cache_dir, "slow_simulations.txt")
                      }
        # Check if all cache files exist
        all_cache_exists = all(os.path.exists(f) for f in cache_files.values())

        # If cache exists, load from cache
        if all_cache_exists:
            logger.info("Loading classification from cache...")
            for env_type, filepath in cache_files.items():
                with open(filepath, 'r') as f:
                    paths = [line.strip() for line in f if line.strip()]
                    simulations_by_type[env_type] = paths
                logger.info(f"Loaded {len(paths)} {env_type} simulations from cache")
            
            return simulations_by_type
    
        # If cache doesn't exist, perform classification
        logger.info("No cache found. Classifying simulations by environment type...")
        
        for path in tqdm(simulations, desc="Classifying simulations", unit="sim"):
            try:
                imbh_df, system_df, iconds_dict = self.load_single_simulation(path)
                
                # Only process if sufficient points
                if len(imbh_df) <= self.config.min_points_threshold:
                    continue
                
                # Determine formation channel
                chform = determine_formation_channel(imbh_df         = imbh_df, 
                                                     mass_column_name = self.config.mass_column_imbh,
                                                     time_column_name = self.config.time_column_imbh)
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
        
        # Loop through each environment type and save paths
        for env_type, paths in simulations_by_type.items():
            if paths:
                filepath = os.path.join(output_dir, f"{env_type.lower()}_simulations.txt")
                with open(filepath, 'w') as f:
                    f.write('\n'.join(paths) + '\n')
                logger.info(f"Saved {len(paths)} {env_type} simulation paths to {filepath}")

#--------------------------------------------------------------------------------------------------------------------------#