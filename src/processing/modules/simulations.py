# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import Dict, List, Tuple, Optional
from tqdm   import tqdm
from loguru import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory                     import list_all_directories
from src.processing.features                 import determine_formation_channel
from src.processing.constructors.moccasurvey import load_moccasurvey_imbh_history

# Class for load and classify simulations from paths ----------------------------------------------------------------------#
class LoadSimulationFiles:
    """Efficient load and preparation of simulation files from string paths."""
    
    # Initialize configuration --------------------------------------------------------------------------------------------#
    def __init__(self, config):
        
        # Store configuration and initialize cache
        self.config = config
        self._cache = {}
    
    # Given a folder load all simulations in that folder or from an especific .txt file -----------------------------------#
    def get_simulation_paths(self, data_path: str, root_dir: str ="./rawdata/", load_all_sims : bool = True,
                              verbose : bool = True
                              ) -> List[str]:
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
    
    # From a given path, load a single simulation -------------------------------------------------------------------------#
    def load_single_simulation(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load and preprocess a single simulation with caching."""
        
        # Create a unique cache key based on the simulation path
        cache_key = f"sim_{hash(path)}"
        
        # Check if simulation data is already cached
        if cache_key not in self._cache:
            
            # Load data based on dataset type
            if self.config.dataset_name == "moccasurvey":
                imbh_history, system = load_moccasurvey_imbh_history(file_path     = f"{path}/",
                                                                    init_conds_sim = True,
                                                                    init_conds_evo = True,
                                                                    verbose        = False,
                                                                    config         = self.config
                                                                    )
                
                imbh_df     = imbh_history[0].drop_duplicates(subset=self.config.time_column_imbh
                                                              ).sort_values(self.config.time_column_imbh
                                                                            ).reset_index(drop=True)
                iconds_dict = imbh_history[1]
                system_df   = system[0].sort_values(self.config.time_column_system).reset_index(drop=True)
            
            self._cache[cache_key] = (imbh_df, system_df, iconds_dict)
        
        return self._cache[cache_key]
    
    def classify_simulations_by_environment(self, simulations: List[str], cache_dir: str = "./rawdata/",
                                            category_list    : Optional[List[str]] = None,
                                            ) -> Dict[str, List[str]]:
        """Classify simulations by environment type efficiently according to Giersz, et al, 2015."""
        
        # Default categories
        if category_list is None:
            category_list = ["FAST", "SLOW"]
        
        # Set up storage for classified simulations
        simulations_by_type = {cat: [] for cat in category_list}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
        # Try to load from cache first
        cache_files = {cat: os.path.join(cache_dir, f"{cat.lower()}_simulations.txt") for cat in category_list}
        
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
        
        # Perform the classification of simulations using Mirek criteria after a cleaning to ensure resolution
        for path in tqdm(simulations, desc="Classifying simulations", unit="sim"):
            try:
                # Load simulation data
                imbh_df, system_df, _ = self.load_single_simulation(path)
                
                # Flag the total number of points
                num_points_flag = len(imbh_df) <= self.config.min_points_threshold

                # Safe merge between initial conditions and imbh history
                aligned     = pd.merge_asof(imbh_df[[self.config.time_column_imbh]], system_df,
                                            left_on   = self.config.time_column_imbh,
                                            right_on  = self.config.time_column_system,
                                            direction = "nearest")
                
                # Compute error based on differences between time columns
                imbh_dt     = imbh_df[self.config.time_column_imbh].diff().median()
                match_error = (aligned[self.config.time_column_system]
                               - imbh_df[self.config.time_column_imbh]).abs().median()
                
                # Create resolution flag
                resolution_flag = (imbh_dt > 0) and (match_error > self.config.max_resolution_ratio * imbh_dt)

                # If skipped simulations, raise warning for transparency
                if num_points_flag or resolution_flag:
                    logger.warning(f"Skipping classification [{('insufficient points' if num_points_flag else '')} "
                                   f"{'resolution mismatch' if resolution_flag else ''}]: {path}")
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
    
    # Classification by maximun mass of the most massive object in the simulation -----------------------------------------#
    def classify_simulations_by_mass(self, simulations: List[str], cache_dir: str = "./rawdata/",
                                     category_list    : Optional[List[str]] = None,
                                    ) -> Dict[str, List[str]]:
        """Classify simulations by quantile rank of their final MMO mass."""
        
        # Validate category list
        if category_list is None:
            raise ValueError("category_list must be provided as proxy to create the bins of the classification.")

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
        # Try to load from cache first
        cache_files = {cat: os.path.join(cache_dir, f"{cat.lower()}_simulations.txt") for cat in category_list}
        
        # Check if all cache files exist
        all_cache_exists = all(os.path.exists(f) for f in cache_files.values())

        # If cache exists, load from cache
        if all_cache_exists:
            logger.info("Loading mass classification from cache...")
            simulations_by_type = {}
            for cat, filepath in cache_files.items():
                with open(filepath, 'r') as f:
                    paths = [line.strip() for line in f if line.strip()]
                    simulations_by_type[cat] = paths
                logger.info(f"Loaded {len(paths)} {cat} simulations from cache")
            return simulations_by_type
    
        # If cache doesn't exist, collect final masses across all valid simulations
        logger.info("No cache found. Collecting final masses for quantile classification...")
        
        valid_paths     = []
        m_mmo_last_sims = []

        for path in tqdm(simulations, desc="Collecting masses", unit="sim"):
            try:
                imbh_df, system_df, _ = self.load_single_simulation(path)
                
                # Validation: minimum points
                num_points_flag = len(imbh_df) <= self.config.min_points_threshold

                # Validation: temporal resolution
                aligned         = pd.merge_asof(imbh_df[[self.config.time_column_imbh]], system_df,
                                                left_on   = self.config.time_column_imbh,
                                                right_on  = self.config.time_column_system,
                                                direction = "nearest")
                imbh_dt         = imbh_df[self.config.time_column_imbh].diff().median()
                match_error     = (aligned[self.config.time_column_system]
                                  - imbh_df[self.config.time_column_imbh]).abs().median()
                resolution_flag = (imbh_dt > 0) and (match_error > self.config.max_resolution_ratio * imbh_dt)

                if num_points_flag or resolution_flag:
                    logger.warning(f"Skipping classification [{('insufficient points' if num_points_flag else '')} "
                                   f"{'resolution mismatch' if resolution_flag else ''}]: {path}")
                    continue
                
                # Extract final mass value of the MMO
                m_mmo_last = imbh_df[self.config.mass_column_imbh].values[-1]
                
                valid_paths.append(path)
                m_mmo_last_sims.append(m_mmo_last)
                
            except Exception as e:
                logger.warning(f"Error classifying simulation {path}: {e}")
                continue
        
        # Compute quantile boundaries from the collected masses
        # N categories → N+1 boundaries at equal-frequency percentiles (0, 100/N, ..., 100)
        n_cats      = len(category_list)
        percentiles = np.linspace(0, 100, n_cats + 1)
        boundaries  = np.percentile(m_mmo_last_sims, percentiles)
        logger.info(f"Mass quantile boundaries ({n_cats} bins): {np.round(boundaries, 2).tolist()}")

        # Assign each simulation to a category based on its quantile bin
        m_array     = np.array(m_mmo_last_sims)
        bin_indices = np.clip(np.digitize(m_array, boundaries[1:-1]), 0, n_cats - 1)

        # Build results dictionary
        simulations_by_type = {cat: [] for cat in category_list}
        for path, bin_idx in zip(valid_paths, bin_indices):
            simulations_by_type[category_list[bin_idx]].append(path)

        # Log statistics
        for cat, paths in simulations_by_type.items():
            logger.info(f"{cat} simulations: {len(paths)}")
        
        return simulations_by_type
    
    # Save a dictionary of classified simulations -------------------------------------------------------------------------#
    def save_simulation_paths_by_type(self, simulations_by_type: Dict[str, List[str]], output_dir: str):
        """Save simulation paths to text files separated by a classification criteria."""
        
        # Loop through each environment type and save paths
        for clf, paths in simulations_by_type.items():
            if paths:
                filepath = os.path.join(output_dir, f"{clf.lower()}_simulations.txt")
                with open(filepath, 'w') as f:
                    f.write('\n'.join(paths) + '\n')
                logger.info(f"Saved {len(paths)} {clf} simulation paths to {filepath}")

    # Compute and cache number of timesteps per simulation ----------------------------------------------------------------#
    def compute_simulation_lengths(self, simulations: List[str], cache_dir: str = "./rawdata/") -> Dict[str, int]:
        """Compute and cache the number of IMBH-history timesteps for each simulation path."""

        lengths_file = os.path.join(cache_dir, "simulation_lengths.json")

        # Load from cache if it already exists
        if os.path.exists(lengths_file):
            logger.info(f"Loading simulation lengths from cache: {lengths_file}")
            with open(lengths_file, 'r') as f:
                return json.load(f)

        # Compute lengths by loading each simulation
        logger.info("Computing simulation lengths (number of timesteps per simulation)...")
        sim_lengths: Dict[str, int] = {}

        for path in tqdm(simulations, desc="Computing lengths", unit="sim"):
            try:
                imbh_df, _, _ = self.load_single_simulation(path)
                sim_lengths[path] = len(imbh_df)
            except Exception as e:
                logger.warning(f"Could not compute length for {path}: {e}")
                sim_lengths[path] = 0

        # Persist to disk
        with open(lengths_file, 'w') as f:
            json.dump(sim_lengths, f, indent=2)

        logger.info(f"Saved simulation lengths for {len(sim_lengths)} simulations to {lengths_file}")
        return sim_lengths

#--------------------------------------------------------------------------------------------------------------------------#