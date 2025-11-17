# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import os
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Tuple, Optional
from loguru import logger

# Custome functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory import PathManagerMOCCAExperiment

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/partitions_output.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# How to partition the simulations in the datafile ------------------------------------------------------------------------#
class DataPartitioner:
    """Efficient data partitioning with stratification support."""
    
    def __init__(self, config):
        self.config = config
    
    def create_random_partitions(self, simulations: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create random train/validation/test partitions."""
        simulations  = np.array(simulations)
        n_total      = len(simulations)
        shuffled_idx = np.random.permutation(n_total)

        n_train = int(n_total * self.config.train_split)
        n_val   = int(n_total * self.config.val_split)

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
    
    def create_stratified_partitions(self, path_manager: PathManagerMOCCAExperiment
                                     )-> Optional[Tuple[List[str], List[str], List[str]]]:
        """Create stratified partitions based on environment type."""
        simulations_by_type = {}
        
        # Mapping of env name to numeric label
        env_to_label = {"FAST": 0, "SLOW": 1, "STEADY": 2}
        
        # Load simulation paths by environment type
        for env_type in ["fast", "slow", "steady"]:
            filepath = path_manager.get_stratified_file_path(env_type)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    simulations_by_type[env_type.upper()] = [line.strip() for line in f.readlines()]
            else:
                logger.warning(f"File {filepath} not found. Using random partitioning instead.")
                return None
        
        # Create stratified partitions
        train_simulations, val_simulations, test_simulations = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for env_type, paths in simulations_by_type.items():
            if not paths:
                continue
                
            n_total      = len(paths)
            shuffled_idx = np.random.permutation(n_total)
            
            n_train = int(n_total * self.config.train_split)
            n_val   = int(n_total * self.config.val_split)
            
            train_idx = shuffled_idx[:n_train]
            val_idx   = shuffled_idx[n_train:n_train + n_val]
            test_idx  = shuffled_idx[n_train + n_val:]
            
            train_simulations.extend([paths[i] for i in train_idx])
            val_simulations.extend([paths[i] for i in val_idx])
            test_simulations.extend([paths[i] for i in test_idx])

            train_labels.extend([env_type] * len(train_idx))
            val_labels.extend([env_type] * len(val_idx))
            test_labels.extend([env_type] * len(test_idx))
            
            logger.info(f"{env_type} environment:")
            logger.info(f"  Training   : {len(train_idx)}")
            logger.info(f"  Validation : {len(val_idx)}")
            logger.info(f"  Testing    : {len(test_idx)}")
        
        logger.info("Total stratified partitions:")
        logger.info(f"Training   : {len(train_simulations)}")
        logger.info(f"Validation : {len(val_simulations)}")
        logger.info(f"Testing    : {len(test_simulations)}")
        
        return [train_simulations, train_labels], [val_simulations, val_labels], [test_simulations, test_labels]

#--------------------------------------------------------------------------------------------------------------------------#