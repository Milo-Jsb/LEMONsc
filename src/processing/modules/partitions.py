# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import json
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Tuple, Optional
from loguru import logger

# Custome functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory import PathManagerDatasetPipeline

# How to partition the simulations in the datafile ------------------------------------------------------------------------#
class DataPartitioner:
    """
    ________________________________________________________________________________________________________________________
    Simulation-level data partitioning with random and stratified strategies for LEMONsc experiments.
    ________________________________________________________________________________________________________________________
    Methods:
    -> create_random_partitions(simulations): Receives a flat list of simulation paths and splits it into
       train/val/test using the ratios defined in config (train_split, val_split). Returns three lists of paths.

    -> create_stratified_partitions(path_manager, valid_simulations, str_crit): Splits simulations preserving
       category proportions. Each category in str_crit must have a corresponding <category>_simulations.txt file
       readable via path_manager. Each file is split independently and the results are concatenated. If
       valid_simulations is provided, only paths present in that set are kept. Returns three [paths, labels] pairs
       for train, validation, and test.
    ________________________________________________________________________________________________________________________
    """
    # Intialize configuration ---------------------------------------------------------------------------------------------#
    def __init__(self, config):
        self.config = config
    
    # Random parition of simulations from a list of str paths -------------------------------------------------------------#
    def create_random_partitions(self, simulations: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Create random train/validation/test partitions."""
        
        # Simulations, number of files and shuffled indices
        simulations  = np.array(simulations)
        n_total      = len(simulations)
        shuffled_idx = np.random.permutation(n_total)

        # Calculate split sizes
        n_train = int(n_total * self.config.train_split)
        n_val   = int(n_total * self.config.val_split)

        # Get indices for each partition
        train_idx = shuffled_idx[:n_train]
        val_idx   = shuffled_idx[n_train:n_train + n_val]
        test_idx  = shuffled_idx[n_train + n_val:]

        # Create partitions
        train_simulations = simulations[train_idx].tolist()
        val_simulations   = simulations[val_idx].tolist()
        test_simulations  = simulations[test_idx].tolist()

        # Logging in console
        logger.info("Random partitioning:")
        logger.info(f"Training   : {len(train_simulations)}")
        logger.info(f"Validation : {len(val_simulations)}")
        logger.info(f"Testing    : {len(test_simulations)}")

        return train_simulations, val_simulations, test_simulations
    
    # Stratified partitions at a simulation path level, given a categorization critieria ----------------------------------#
    def create_stratified_partitions(self, path_manager: PathManagerDatasetPipeline, 
                                     valid_simulations : Optional[List[str]] = None,
                                     str_crit          : Optional[List[str]] = None,
                                     ) -> Optional[Tuple[List[str], List[str], List[str]]]:
        """Create stratified partitions based on a categorical criteria."""
        # Create dictionary
        simulations_by_type = {}
        
        # Convert valid_simulations to set for faster lookup
        valid_set = set(valid_simulations) if valid_simulations is not None else None
        
        # Validate stratification criteria
        if str_crit is None:
            raise ValueError("str_crit must be provided: list of category names to stratify.")

        # Load simulation paths by environment type
        for category in str_crit:
            filepath = path_manager.get_stratified_file_path(category)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_paths = [line.strip() for line in f.readlines()]
                    
                    # Filter by valid_simulations if provided
                    if valid_set is not None:
                        filtered_paths = [p for p in all_paths if p in valid_set]
                        if len(filtered_paths) < len(all_paths):
                            logger.info(110*"_")
                            logger.info(f"{category.upper()}:")
                            logger.info(f"Elements found    : {len(all_paths)}.")
                            logger.info(f"Elements filtered : {len(filtered_paths)-len(all_paths)}")
                        simulations_by_type[category.upper()] = filtered_paths
                    else:
                        simulations_by_type[category.upper()] = all_paths
            else:
                logger.warning(f"File {filepath} not found. Using random partitioning instead.")
                return None
        
        # Create stratified partitions
        train_simulations, val_simulations, test_simulations = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        for category, paths in simulations_by_type.items():
            
            if not paths: continue
                
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

            train_labels.extend([category] * len(train_idx))
            val_labels.extend([category] * len(val_idx))
            test_labels.extend([category] * len(test_idx))
            
            logger.info(f"{category} environment:")
            logger.info(f"  Training   : {len(train_idx)}")
            logger.info(f"  Validation : {len(val_idx)}")
            logger.info(f"  Testing    : {len(test_idx)}")
        
        logger.info("Total stratified partitions:")
        logger.info(f"Training   : {len(train_simulations)}")
        logger.info(f"Validation : {len(val_simulations)}")
        logger.info(f"Testing    : {len(test_simulations)}")
        
        return [train_simulations, train_labels], [val_simulations, val_labels], [test_simulations, test_labels]

    # K-fold cross-validation partitions with a fixed test set ------------------------------------------------------------#
    def create_kfold_partitions(self, n_folds: int,
                                simulations  : Optional[List[str]] = None,
                                path_manager : Optional[PathManagerDatasetPipeline] = None,
                                str_crit     : Optional[List[str]] = None,
                                snake_draft  : bool = True
                                ) -> Optional[Tuple]:
        """
        ____________________________________________________________________________________________________________________
        Create k-fold cross-validation partitions with a fixed held-out test set.

        A test set is split off first (using config.test_split). The remaining simulations are divided into
        n_folds chunks. For each fold i, chunk i becomes the validation set and the other k-1 chunks form
        the training set. When n_folds=1, a single train/val split is created using config ratios.

        For stratified mode, provide path_manager and str_crit. For random mode, provide simulations only.
        ____________________________________________________________________________________________________________________
        Returns:
            ([test_simulations, test_labels], fold_list)
            where fold_list is a list of ([train_sims, train_labels], [val_sims, val_labels]) per fold.
            Returns None if stratified files are missing.
        ____________________________________________________________________________________________________________________
        """
        stratified = str_crit is not None and path_manager is not None

        # K-fold stratification -------------------------------------------------------------------------------------------#
        if stratified:
            valid_set = set(simulations) if simulations is not None else None

            # Load simulation paths per category
            simulations_by_type = {}
            for category in str_crit:
                filepath = path_manager.get_stratified_file_path(category)
                # Check that filepath exist and load valid elements
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        all_paths = [line.strip() for line in f.readlines()]
                    if valid_set is not None:
                        all_paths = [p for p in all_paths if p in valid_set]
                    simulations_by_type[category.upper()] = all_paths
                
                # Raise warning
                else:
                    logger.warning(f"File {filepath} not found.")
                    return None

            # Load simulation lengths for snake draft (generated during --mode study)
            sim_lengths  = {}
            
            if snake_draft:
                lengths_path = path_manager.get_simulation_lengths_path()
                
                if os.path.exists(lengths_path):
                    with open(lengths_path, 'r') as f:
                        sim_lengths = json.load(f)
                    logger.info("Snake draft enabled: simulation lengths loaded from cache")
                
                else:
                    logger.warning("simulation_lengths.json not found — falling back to random chunking. "
                                   "Run '--mode study' first to enable snake draft.")
            else:
                logger.info("Snake draft disabled: using random chunking for fold assignment")

            # Seeded RNG for reproducible shuffles
            rng = np.random.default_rng(getattr(self.config, 'partition_seed', 42))

            # Per-category: split off test, then divide pool into k chunks
            test_simulations, test_labels = [], []
            fold_chunks_sims   = [[] for _ in range(n_folds)]
            fold_chunks_labels = [[] for _ in range(n_folds)]

            for category, paths in simulations_by_type.items():
                
                if not paths: continue

                n_total      = len(paths)
                shuffled_idx = rng.permutation(n_total)
                n_test       = max(1, int(n_total * self.config.test_split))

                test_idx = shuffled_idx[:n_test]
                pool_idx = shuffled_idx[n_test:]

                test_simulations.extend([paths[i] for i in test_idx])
                test_labels.extend([category] * len(test_idx))

                # Split pool into folds (or single train/val for n_folds=1)
                pool_paths = [paths[i] for i in pool_idx]

                if n_folds == 1:
                    val_ratio = self.config.val_split / (self.config.train_split + self.config.val_split)
                    n_val     = max(1, int(len(pool_paths) * val_ratio))
                    fold_chunks_sims[0].extend(pool_paths[:n_val])
                    fold_chunks_labels[0].extend([category] * n_val)
  
                else:
                    # Snake draft: sort by n_timesteps descending, then assign in serpentine order
                    if sim_lengths:
                        pool_paths_sorted = sorted(pool_paths,
                                                   key     = lambda p: sim_lengths.get(p, 0),
                                                   reverse = True)
                    # Normal selection
                    else:
                        pool_paths_sorted = pool_paths

                    cycle = 2 * n_folds
                    for i, path in enumerate(pool_paths_sorted):
                        pos    = i % cycle
                        fold_i = pos if pos < n_folds else cycle - 1 - pos
                        fold_chunks_sims[fold_i].append(path)
                        fold_chunks_labels[fold_i].append(category)

                logger.info(f"{category}: {n_total} total, {n_test} test, {n_total - n_test} in pool")

            # Build fold list
            fold_list = []
            if n_folds == 1:
                val_sims   , val_labels_fold   = fold_chunks_sims[0], fold_chunks_labels[0]
                train_sims , train_labels_fold = []                 , []
                
                # Everything not in val chunk is train
                for category, paths in simulations_by_type.items():
                    pool_paths = [p for p in paths if p not in set(test_simulations) and p not in set(val_sims)]
                    
                    train_sims.extend(pool_paths)
                    train_labels_fold.extend([category] * len(pool_paths))
                
                fold_list.append(([train_sims, train_labels_fold], [val_sims, val_labels_fold]))
                logger.info(f"Fold 0: train={len(train_sims)}, val={len(val_sims)}")
            
            else:
                for fold_i in range(n_folds):
                    val_sims         = fold_chunks_sims[fold_i]
                    val_labels_fold  = fold_chunks_labels[fold_i]
                    train_sims, train_labels_fold = [], []
                    
                    for fold_j in range(n_folds):
                        if fold_j != fold_i:
                            train_sims.extend(fold_chunks_sims[fold_j])
                            train_labels_fold.extend(fold_chunks_labels[fold_j])
                    
                    fold_list.append(([train_sims, train_labels_fold], [val_sims, val_labels_fold]))
                    logger.info(f"Fold {fold_i}: train={len(train_sims)}, val={len(val_sims)}")

            logger.info(f"Test: {len(test_simulations)}")
            return [test_simulations, test_labels], fold_list
        
        # Random Fold selection -------------------------------------------------------------------------------------------#
        else:
            
            if simulations is None:
                raise ValueError("simulations must be provided for random partitioning")

            sims         = np.array(simulations)
            n_total      = len(sims)
            rng          = np.random.default_rng(getattr(self.config, 'partition_seed', 42))
            shuffled_idx = rng.permutation(n_total)
            n_test       = max(1, int(n_total * self.config.test_split))

            test_sims = sims[shuffled_idx[:n_test]].tolist()
            pool_sims = sims[shuffled_idx[n_test:]].tolist()

            fold_list = []
            if n_folds == 1:
                val_ratio = self.config.val_split / (self.config.train_split + self.config.val_split)
                n_val     = max(1, int(len(pool_sims) * val_ratio))
                val_sims   = pool_sims[:n_val]
                train_sims = pool_sims[n_val:]
                fold_list.append(([train_sims, None], [val_sims, None]))
                logger.info(f"Fold 0: train={len(train_sims)}, val={len(val_sims)}")
            else:
                chunk_indices = np.array_split(np.arange(len(pool_sims)), n_folds)
                fold_chunks   = [[pool_sims[i] for i in indices] for indices in chunk_indices]
                for fold_i in range(n_folds):
                    val_sims   = fold_chunks[fold_i]
                    train_sims = [s for j, chunk in enumerate(fold_chunks) if j != fold_i for s in chunk]
                    fold_list.append(([train_sims, None], [val_sims, None]))
                    logger.info(f"Fold {fold_i}: train={len(train_sims)}, val={len(val_sims)}")

            logger.info(f"Test: {len(test_sims)}")
            return [test_sims, None], fold_list

#--------------------------------------------------------------------------------------------------------------------------#