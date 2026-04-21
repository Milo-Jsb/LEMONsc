# Modules -----------------------------------------------------------------------------------------------------------------#
import os

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing      import Any, Dict, List, Optional, Tuple, Union
from tqdm        import tqdm
from loguru      import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.modules.simulations      import LoadSimulationFiles
from src.processing.filters                  import efficiency_mass_ratio_relation, def_config as filter_def_config
from src.processing.constructors.moccasurvey import process_single_moccasurvey_simulation

# Configuration for processing features -----------------------------------------------------------------------------------#
@dataclass
class ProcessingStats:
    """Statistics tracking for simulation processing."""
    # Counters
    used_sims      : int = 0
    ignored_sims   : int = 0
    
    # Relevant atributes of the simulations
    simulation_type : List[str] = field(default_factory=list)
    points_per_sim  : List[int] = field(default_factory=list)
    
    # Augmentation-specific tracking
    virtual_sims_generated : int       = 0
    points_per_original    : List[int] = field(default_factory=list)
    augmentation_enabled   : bool      = False
    
    def increment_used(self, clf_type: str, num_points: int, is_augmented: bool = False, n_virtual: Optional[int] = None):
        """Increment used simulation counter and record stats."""
        
        self.used_sims += 1
        self.simulation_type.append(clf_type)
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
            'used_sims'       : self.used_sims,
            'ignored_sims'    : self.ignored_sims,
            'avg_points'      : avg_points,
            'std_points'      : std_points,
            'category_counts' : {cat: self.simulation_type.count(cat) for cat in set(self.simulation_type)},
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
    
    # Initialize configuration and single simulation processing -----------------------------------------------------------#
    def __init__(self, config):
        self.config             = config
        self._loader_from_paths = LoadSimulationFiles(config)
    
    def process_simulations(self, simulations : List[str], labels: Optional[List[str]] = None,
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
                imbh_df, system_df, iconds_dict = self._loader_from_paths.load_single_simulation(path)
                label = labels[idx] if (labels is not None and idx < len(labels)) else None
                
                # Validate simulation and obtain the time-aligned system DataFrame.
                is_valid, matched_system_df, reject_reason = self._validate_simulation(imbh_df, system_df)
                if not is_valid:
                    stats.increment_ignored()
                    logger.warning(f"Ignored [{reject_reason}]: {path}")
                    continue
                
                feats, masses, idxs = self._process_single(imbh_df, matched_system_df, iconds_dict, self.config, label, 
                                                           augmentation, 
                                                           apply_noise)
                
                # Track statistics and accumulate results
                stats.increment_used(label, len(feats), is_augmented=augmentation, n_virtual=n_virtual)
                self._accumulate_results(feats, masses, time_list, mass_list, phy_list, path_list, path, study_mode)
                
            except Exception as e:
                stats.increment_ignored()
                logger.warning(f"Error processing simulation {path}: {e}")
                continue
        
        # Report statistics if requested
        if verbose:
            self._report_statistics(stats, len(simulations))
        
        return time_list, mass_list, phy_list, path_list
    
    # From the raw file of simulations review if filtering or outliers are present ----------------------------------------#
    def select_suitable_sims(self, simulations: List[str], simulations_by_type: Optional[Dict[str, List[str]]], 
                             out_path : str, 
                             verbose  : bool = True
                             ) -> Dict:
        """Retrieve suitable simulations based on efficiency-mass ratio filtering"""
        # Treat None as empty dict to allow label-free operation
        if simulations_by_type is None:
            simulations_by_type = {}
        
        # Check if division already exist
        valid_sims_path = os.path.join(out_path, 'valid_simulations.txt')
        outliers_path   = os.path.join(out_path, 'outliers.txt')
        cache_exists    = os.path.exists(valid_sims_path) and os.path.exists(outliers_path)

        # Build label lookup (always needed)
        path_to_label = {path: env_type for env_type, paths in simulations_by_type.items() for path in paths}

        # If filtering files already exist, load them to avoid redundant processing.
        if cache_exists:

            logger.info("Found existing filter files. Loading from disk...")
            with open(valid_sims_path, 'r') as f:
                valid_sim_paths = [line.strip() for line in f if line.strip()]
            with open(outliers_path, 'r') as f:
                outlier_paths = [line.strip() for line in f if line.strip()]

            logger.info(f"Loaded {len(valid_sim_paths)} valid simulations from: {valid_sims_path}")
            logger.info(f"Loaded {len(outlier_paths)} outliers from: {outliers_path}")

            # Process only valid sims to obtain mass_ratio/epsilon for plotting.
            valid_labels = [path_to_label.get(p, np.nan) for p in valid_sim_paths]
            
            # Process the valid simulation
            _, m_valid, phy_valid, path_list = self.process_simulations(valid_sim_paths, valid_labels,
                                                                        augmentation = False,
                                                                        apply_noise  = False,
                                                                        n_virtual    = None,
                                                                        study_mode   = True,
                                                                        verbose      = verbose)

            # Extract necessary parameters for plotting the efficiency-mass ratio relation.
            filt_labels = [path_to_label.get(p, np.nan) for p in path_list]
            
            # Select initial total mass, final total mass and critical mass from the physical array
            init_totmass  = np.array([e[filter_def_config.sim_pos][filter_def_config.Mtot_pos]   for e in phy_valid])
            final_totmass = np.array([e[filter_def_config.final_pos][filter_def_config.Mtot_pos] for e in phy_valid])
            init_mcrit    = np.array([e[filter_def_config.sim_pos][filter_def_config.Mcrit_pos]  for e in phy_valid])
            
            # Select the final BH mass from the mass array
            final_bhmass = np.array([mmo[filter_def_config.Mmmo_pos] for mmo in m_valid])
            
            # Compute the mass ratio and the epsilon parameters for the valid simulations.
            mass_ratio = init_totmass / init_mcrit
            epsilon    = final_bhmass / (final_totmass - final_bhmass)

            # Retrieve the half mass radius for further plotting
            init_rh  = np.array([e[filter_def_config.sim_pos][filter_def_config.r_h_pos] for e in phy_valid])

            return {
                "valid_sims": {"paths"      : path_list,
                               "labels"     : filt_labels,
                               "mass_ratio" : mass_ratio,
                               "epsilon"    : epsilon,
                               "mtot"       : init_totmass,
                               "rh"         : init_rh
                               },
                "outliers"  : {"paths"      : outlier_paths,
                               "labels"     : [path_to_label.get(p, np.nan) for p in outlier_paths],
                               "mass_ratio" : np.array([]),
                               "epsilon"    : np.array([]),
                               "mtot"       : np.array([]),
                               "rh"         : np.array([])
                               }
            }

        # No cache — proceed with full filtering
        logger.info("No existing filter files found. Proceeding with filtering...")

        labels = [path_to_label.get(path, np.nan) for path in simulations]
        
        # Process simulations to retrieve per-simulation time series in study_mode
        _, m_base, phy_base, path_list = self.process_simulations(simulations, labels, 
                                                                  augmentation = False, 
                                                                  apply_noise  = False,
                                                                  n_virtual    = None, 
                                                                  study_mode   = True, 
                                                                  verbose      = verbose)
        
        # Update label list
        filt_labels = [path_to_label.get(path, np.nan) for path in path_list]

        # Filter simulations based on their values of efficiency-mass_ratio
        filter_output = efficiency_mass_ratio_relation(mmo_mass        = m_base, 
                                                       physical_params = phy_base, 
                                                       path_list       = path_list,
                                                       labels_list     = filt_labels,
                                                       logger          = logger)
        
        # Ensure output directory exists and save results
        os.makedirs(out_path, exist_ok=True)

        with open(valid_sims_path, 'w') as f:
            for sim_path in filter_output['valid_sims']['paths']:
                f.write(f"{sim_path}\n")
        logger.info(f"Filtered simulation paths saved to: {valid_sims_path}")

        with open(outliers_path, 'w') as f:
            for sim_path in filter_output['outliers']['paths']:
                f.write(f"{sim_path}\n")
        logger.info(f"Outliers paths saved to: {outliers_path}")
        
        return filter_output
    
    def _validate_simulation(self, imbh_df: pd.DataFrame, system_df: pd.DataFrame
                               ) -> Tuple[bool, Optional[pd.DataFrame], Optional[str]]:
        """Check temporal resolution compatibility and point count for a simulation."""
        # Align system_df to imbh_df times via nearest-neighbour merge
        matched_system_df = pd.merge_asof(imbh_df[[self.config.time_column_imbh]], system_df,
                                          left_on   = self.config.time_column_imbh,
                                          right_on  = self.config.time_column_system,
                                          direction = "nearest"
                                          ).reset_index(drop=True)

        # Check temporal resolution compatibility before accepting the alignment.
        imbh_dt  = imbh_df[self.config.time_column_imbh].diff().median()
        res_time = matched_system_df[self.config.time_column_system] - imbh_df[self.config.time_column_imbh]
        
        # Use absolute value of the time difference for resolution check, and compute median error for reporting.
        match_error = res_time.abs().median()

        # Create flags for resolution and point count
        resolution_flag = (imbh_dt > 0) and (match_error > self.config.max_resolution_ratio * imbh_dt)
        num_points_flag = len(imbh_df) <= self.config.min_points_threshold

        if resolution_flag and num_points_flag:
            reason = (f"resolution mismatch (match_error={match_error:.3f} > "
                      f"{self.config.max_resolution_ratio}*dt={self.config.max_resolution_ratio*imbh_dt:.3f}) "
                      f"AND insufficient points ({len(imbh_df)} <= {self.config.min_points_threshold})")
            return False, None, reason
        if resolution_flag:
            reason = (f"resolution mismatch (match_error={match_error:.3f} > "
                      f"{self.config.max_resolution_ratio}*dt={self.config.max_resolution_ratio*imbh_dt:.3f})")
            return False, None, reason
        if num_points_flag:
            reason = f"insufficient points ({len(imbh_df)} <= {self.config.min_points_threshold})"
            return False, None, reason

        # Overwrite system time column with exact IMBH timestamps
        matched_system_df[self.config.time_column_system] = imbh_df[self.config.time_column_imbh].values

        return True, matched_system_df, None
    
    def _process_single(self, imbh_df: pd.DataFrame, system_df: pd.DataFrame, iconds_dict: Dict,
                       config      : Any,
                       label       : Optional[str],
                       augmentation: bool,
                       apply_noise : bool,
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single simulation and return features, masses, and indices."""
        
        # Process based on dataset type
        if (self.config.dataset_name == "moccasurvey"):
        
            return process_single_moccasurvey_simulation(imbh_df= imbh_df, system_df = system_df, meta_dict = iconds_dict,
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
        
        for cat, count in summary['category_counts'].items():
            if (cat is None) or (count is None):
                continue
            else:
                logger.info(f"  - {cat:<30} : {count}")
        
        # Report points statistics based on augmentation state
        if 'avg_points_per_original' in summary:
            logger.info(f"  - Average points per original   : {summary['avg_points_per_original']:.1f} ± \
                        {summary['std_points_per_original']:.1f}")
            logger.info(f"  - Average points per output     : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        else:
            logger.info(f"  - Average points per simulation : {summary['avg_points']:.1f} ± {summary['std_points']:.1f}")
        
        logger.info(110*"_")
        
#--------------------------------------------------------------------------------------------------------------------------#