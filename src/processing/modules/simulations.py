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
from src.utils.directory     import list_all_directories
from src.processing.dataset  import load_moccasurvey_imbh_history
from src.processing.features import determine_formation_channel

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
                                                                 init_conds_sim = True,
                                                                 init_conds_evo = True,
                                                                 verbose        = False
                                                                )
            
            imbh_df     = imbh_history[0].drop_duplicates(subset="time[Myr]").sort_values("time[Myr]").reset_index(drop=True)
            iconds_dict = imbh_history[1]  
            system_df   = pd.merge_asof(imbh_df["time[Myr]"], system[0].sort_values("tphys").reset_index(drop=True),
                                        left_on   = "time[Myr]",
                                        right_on  = "tphys",
                                        direction = "nearest"
                                       )
            
            self._cache[cache_key] = (imbh_df, system_df, iconds_dict)
        
        return self._cache[cache_key]
    
    def load_single_simulation_example(self, simulations: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict]:
        """Load a single simulation for analysis and plotting."""
        sim_num = np.random.randint(0, len(simulations))
        imbh_df, system_df, iconds_dict = self.load_single_simulation(simulations[sim_num])
        return imbh_df, system_df, simulations[sim_num], iconds_dict
    
    def classify_simulations_by_environment(self, simulations: List[str]) -> Dict[str, List[str]]:
        """Classify simulations by environment type efficiently."""
        simulations_by_type = {"FAST": [], "SLOW": [], "STEADY": []}
        
        logger.info("Classifying simulations by environment type...")
        
        for path in tqdm(simulations, desc="Classifying simulations", unit="sim"):
            try:
                imbh_df, system_df, iconds_dict = self.load_single_simulation(path)
                
                # Only process if sufficient points
                if len(imbh_df) <= self.config.min_points_threshold:
                    continue
                
                # Determine formation channel
                chform = determine_formation_channel(system_df, imbh_df, None)
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

#--------------------------------------------------------------------------------------------------------------------------#