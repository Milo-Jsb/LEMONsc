# Modules -----------------------------------------------------------------------------------------------------------------#
import yaml
import optuna
import joblib
import warnings
import logging

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from abc                     import ABC, abstractmethod
from dataclasses             import dataclass
from pathlib                 import Path
from typing                  import Dict, List, Optional, Union, Callable, Any, Sequence, TypeVar, Literal
from sklearn.model_selection import KFold
from sklearn.metrics         import get_scorer

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.mltrees.regressor  import MLTreeRegressor
from src.optim.grid                import RandomForestGrid, LightGBMGrid, XGBoostGrid, DARTBoostGrid
from src.utils.resources           import ResourceConfig, ResourceManager

# Type definitions --------------------------------------------------------------------------------------------------------#
T         = TypeVar('T')
ModelType = Literal["random_forest", "lightgbm", "xgboost", "dartboost"]

# Configuration and result classes ----------------------------------------------------------------------------------------#
@dataclass
class SpaceSearchConfig:
    """Configuration settings for SpaceSearch optimization"""
    model_type : ModelType
    n_jobs         : int = 10                                       # Core numbers for parallel processing
    n_trials       : int = 100                                      # Number of trials for optimization
    device         : str = "cpu"                                    # Device to use ('cpu' or 'cuda')
    verbose        : bool = True                                    # Verbosity flag
    seed           : int = 42                                       # Fixed seed for reproducibility
    sampler        : Optional[optuna.samplers.BaseSampler] = None   # Custom sampler, defaults to TPESampler
    storage        : Optional[str] = None                           # Storage URL for study persistence
    load_if_exists : bool = False                                   # Checkpoint loading flag

@dataclass
class SpaceSearchResult:
    """Results from a SpaceSearch optimization study"""
    best_params          : Dict[str, Any]                  
    best_score           : float
    study                : optuna.study.Study
    n_trials             : int
    output_dir           : str
    optimization_summary : Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format"""
        output_dict = {
                    'best_params': self.best_params, 
                    'best_score' : self.best_score, 
                    'n_trials'   : self.n_trials, 
                    'summary'    : self.optimization_summary
                      }

        return output_dict

# Base components --------------------------------------------------------------------------------------------------------#
class BaseModelBuilder(ABC):
    """Abstract base class for model building strategies"""
    @abstractmethod
    def create_model(self, trial: optuna.trial.Trial) -> MLTreeRegressor:
        """Create a model instance with parameters from trial"""
        pass

class BaseMetricResolver(ABC):
    """Abstract base class for metric resolution strategies"""
    @abstractmethod
    def resolve(self, metric: Union[str, Callable, List[Union[str, Callable]]]) -> List[Callable]:
        """Resolve metrics to callable functions"""
        pass

# Implementation components ----------------------------------------------------------------------------------------------#
class ModelBuilder(BaseModelBuilder):
    """Concrete implementation of model building strategy"""
    def __init__(self, model_type: ModelType, device: str, n_jobs: int):
        
        self.model_type = model_type
        self.device     = device
        self.n_jobs     = n_jobs
        
        self.param_grid_map = {
                        "random_forest" : RandomForestGrid,
                        "lightgbm"      : LightGBMGrid,
                        "xgboost"       : XGBoostGrid,
                        "dartboost"     : DARTBoostGrid
                              }
    
    def create_model(self, trial: optuna.trial.Trial, device: str = "cpu", n_jobs: int = 1) -> MLTreeRegressor:
        """Create a model instance with parameters from trial and resource constraints"""
        if self.model_type not in self.param_grid_map:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        param_grid = self.param_grid_map[self.model_type](trial)
        model      = MLTreeRegressor(model_type   = self.model_type, 
                                     model_params = param_grid, 
                                     device       = device, 
                                     n_jobs       = n_jobs)
        
        return model

class MetricResolver(BaseMetricResolver):
    """Concrete implementation of metric resolution strategy"""
    def resolve(self, metric: Union[str, Callable, List[Union[str, Callable]]]) -> List[Callable]:
        """Resolve metrics to callable functions"""
        if not isinstance(metric, list):
            metric = [metric]
            
        resolved = []
        for m in metric:
            if isinstance(m, str):
                scorer = get_scorer(m)
                resolved.append(scorer)
            elif callable(m):
                resolved.append(m)
            else:
                raise ValueError(f"Invalid metric type: {type(m)}")
        return resolved

class StudyPersistence:
    """Handles saving and loading of optimization studies"""
    @staticmethod
    def save_study(study: optuna.study.Study, output_path: Path, compress: bool = True) -> None:
        """Save study state to disk"""
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"study_state{'_compressed' if compress else ''}.joblib"
        joblib.dump(study, save_path, compress=compress)
    
    @staticmethod
    def load_study(study_path: Path) -> Optional[optuna.study.Study]:
        """Load study state from disk"""
        try:
            return joblib.load(study_path)
        except Exception as e:
            warnings.warn(f"Failed to load study: {e}")
            return None

class Visualization:
    """Handles visualization of optimization results"""
    @staticmethod
    def create_plots(study: optuna.study.Study, output_path: Path) -> None:
        """Generate and save visualization plots"""
        try:
            from optuna.visualization.matplotlib import (
                plot_param_importances,
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_contour)
            
            plots = {
                "optimization_history" : plot_optimization_history(study),
                "param_importances"    : plot_param_importances(study),
                "parallel_coordinate"  : plot_parallel_coordinate(study),
                "contour"              : plot_contour(study)
                    }
            
            for name, fig in plots.items():
                fig.figure.savefig(output_path / f"{name}.png", bbox_inches="tight", dpi=500)
                plt.close(fig.figure)
                
        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")

# Hyperparameter search ---------------------------------------------------------------------------------------------------#
class SpaceSearch:
    """
    ________________________________________________________________________________________________________________________
    SpaceSearch: A comprehensive hyperparameter optimization class using Optuna 
    ________________________________________________________________________________________________________________________
    Features:
        - Support for multiple model types (LightGBM, XGBoost, Random Forest, DARTBoost)
        - Customizable search spaces and objective functions
        - Advanced visualization and analysis tools
        - Study persistence and loading
        - Early stopping and pruning support
        - Multi-objective optimization capabilities
    ________________________________________________________________________________________________________________________
    Note:
        - SpaceSearch is made to work upon MLTreeRegressor.
    ________________________________________________________________________________________________________________________
    """
    
    def __init__(self, config: SpaceSearchConfig):
        """
        ____________________________________________________________________________________________________________________
        Initialize SpaceSearch with configuration
        ____________________________________________________________________________________________________________________
        Parameters
            - config (SpaceSearchConfig): Configuration object containing all settings
        ____________________________________________________________________________________________________________________
        """
        self.config = config
        self.logger = self.__setup_logging()
        
        # Initialize components (Model, Metrics, Sampler)
        self.model_builder   = ModelBuilder(model_type=config.model_type, device=config.device, n_jobs=config.n_jobs)
        self.metric_resolver = MetricResolver()
        self.sampler         = config.sampler or optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
        
        # Internal state
        self.study          : Optional[optuna.study.Study]   = None
        self.best_params    : Optional[Dict[str, Any]]       = None
        self.best_score     : Optional[float]                = None
        self.trials_history : List[optuna.trial.FrozenTrial] = []
        
        # Define early stopping state
        self._early_stopping = {'patience'         : None,
                                'best_value'       : None,
                                'no_improve_count' : 0
                               }
        
        # Log initialization
        if self.config.verbose:
            self.logger.info(
                f"SpaceSearch initialized for {self.config.model_type} "
                f"model (device={self.config.device})"
                )

        # Add resource management
        self.resource_config  = ResourceConfig(max_parallel_trials = min(config.n_jobs, 4),  
                                               prefer_gpu          = (config.device == "cuda"),
                                               n_jobs_per_trial    = 4 if config.device == "cpu" else 1)
        self.resource_manager = ResourceManager(self.resource_config)

    
    def __setup_logging(self) -> logging.Logger:
        """Configure logging for the optimization process"""
        logger = logging.getLogger("SpaceSearch")
        if not logger.handlers:
            
            handler   = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)

        return logger
    
    def __validate_data(self, X_train, y_train, X_val, y_val) -> None:
        """Validate input data shapes and types"""
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Features and target sizes don't match")
        
        if isinstance(X_train, pd.DataFrame) != isinstance(X_val, pd.DataFrame):
            raise ValueError("X_train and X_val must be the same type")
            
        if isinstance(y_train, pd.Series) != isinstance(y_val, pd.Series):
            raise ValueError("y_train and y_val must be the same type")
    
    def __create_objective(self, scorer: Callable, X_train: Any, y_train: Any, X_val: Any, y_val: Any,
                            direction : str,
                            scaler    : Optional[pd.Series] = None
                        ) -> Callable[[optuna.trial.Trial], float]:
        """Create the objective function for optimization, if scaler is provided, it will be applied to predictions"""
        
        def objective(trial: optuna.trial.Trial) -> float:
            try:
                # Acquire resources for this trial
                resources = self.resource_manager.acquire_resources(trial.number)

                # Create and train model
                model = self.model_builder.create_model(trial, device=resources['device'], n_jobs=resources['n_jobs'])
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Apply scaling if provided
                if scaler is not None:
                    y_pred       = y_pred * scaler
                    y_val_scaled = y_val * scaler
                    score = float(scorer(y_val_scaled, y_pred))
                else:
                    score = float(scorer(y_val, y_pred))
                
                return score
                
            except optuna.TrialPruned:
                raise
            
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return -float('inf') if direction == "maximize" else float('inf')
            
            finally:
                self.resource_manager.release_resources(trial.number)
        
        return objective
    
    def __early_stopping_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Handle early stopping logic"""
        current = study.best_value
        
        # Update best value and no improvement count
        if self._early_stopping['best_value'] is None or (
            (study.direction == optuna.study.StudyDirection.MAXIMIZE and
             current > self._early_stopping['best_value']) or
            (study.direction == optuna.study.StudyDirection.MINIMIZE and
             current < self._early_stopping['best_value'])):
            
            self._early_stopping['best_value'] = current
            self._early_stopping['no_improve_count'] = 0
        
        # No improvement, add to count
        else:
            self._early_stopping['no_improve_count'] += 1

        # Stop if patience exceeded 
        if (self._early_stopping['patience'] is not None and
            self._early_stopping['no_improve_count'] >= self._early_stopping['patience']):
            
            self.logger.info(
                f"Early stopping: no improvement in "
                f"{self._early_stopping['patience']} trials.")
            study.stop()
    
    def run_study(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series],
                        X_val: Union[np.ndarray, pd.DataFrame]  ,y_val: Union[np.ndarray, pd.Series],
                  study_name : str = "optuna_study",
                  direction  : str = "maximize",
                  metric     : Union[str, Callable] = "r2",
                  output_dir : str = "./optuna_output",
                  save_study : bool = True,
                  patience   : Optional[int] = None,
                  pruner     : Optional[optuna.pruners.BasePruner] = None,
                  timeout    : Optional[int] = None,
                  catch      : Union[tuple, Sequence[Exception]] = (Exception,),
                  callbacks  : Optional[List[Callable]] = None,
                  scaler     : Optional[pd.Series] = None
                ) -> SpaceSearchResult:
        """
        ____________________________________________________________________________________________________________________
        Run the hyperparameter optimization study
        ____________________________________________________________________________________________________________________
        Parameters:
            - X_train    (Union[np.ndarray, pd.DataFrame])     : Training features.
            - y_train    (Union[np.ndarray, pd.Series])        : Training targets.
            - X_val      (Union[np.ndarray, pd.DataFrame])     : Validation features.
            - y_val      (Union[np.ndarray, pd.Series])        : Validation targets.
            - study_name (str)                                 : Name for the study.
            - direction  (str)                                 : Optimization direction ('maximize' or 'minimize').
            - metric     (Union[str, Callable])                : Metric to optimize
            - output_dir (str)                                 : Directory to save results.
            - save_study (bool)                                : Whether to save the study.
            - patience   (Optional[int])                       : Early stopping patience.
            - pruner     (Optional[optuna.pruners.BasePruner]) : Optuna pruner.
            - timeout    (Optional[int])                       : Timeout for the study in seconds.
            - catch      (Union[tuple, Sequence[Exception]])   : Exceptions to catch during trials.
            - callbacks  (Optional[List[Callable]])            : Additional callbacks for the study.
            - scaler     (Optional[pd.Series])                 : Optional scaler to apply to predictions.
        ____________________________________________________________________________________________________________________
        Returns:
            - SpaceSearchResult: Results from the optimization study
        ____________________________________________________________________________________________________________________
        Notes:
            - run_study() works over a single partition of training/validation data.
            - If scaler is provided, predictions and true values will be scaled before metric evaluation.
            - If patience is set, early stopping will be applied based on validation performance.
            - If save_study is True, visualizations and study state will be saved to output.
        ____________________________________________________________________________________________________________________
        """
        # Validate inputs
        self.__validate_data(X_train, y_train, X_val, y_val)
        
        # Set up early stopping
        self._early_stopping['patience'] = patience
        
        # Create output directory
        output_path = Path(output_dir) / study_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metric
        scorer = self.metric_resolver.resolve(metric)[0]
        
        # Set up suggested number of jobs to use
        n_jobs = self.resource_manager.get_suggested_parallel_trials()

        # Create or load study
        self.study = optuna.create_study(direction      = direction,
                                         study_name     = study_name,
                                         pruner         = pruner,
                                         sampler        = self.sampler,
                                         storage        = self.config.storage,
                                         load_if_exists = self.config.load_if_exists)
        
        # Create objective
        objective = self.__create_objective(scorer    = scorer,
                                            X_train   = X_train,
                                            y_train   = y_train,
                                            X_val     = X_val,
                                            y_val     = y_val,
                                            direction = direction,
                                            scaler    = scaler)
        
        # Set up callbacks
        study_callbacks = callbacks or []
        if patience: study_callbacks.append(self.__early_stopping_callback)
        
        # Optimize
        self.study.optimize(objective, n_trials=self.config.n_trials, timeout=timeout, catch=catch,
                            callbacks = study_callbacks if study_callbacks else None,
                            n_jobs    = n_jobs)  

        # Store results
        self.best_params    = self.study.best_params
        self.best_score     = self.study.best_value
        self.trials_history = self.study.trials
        
        # Generate visualization and save study
        if save_study:
            Visualization.create_plots(self.study, output_path)
            StudyPersistence.save_study(self.study, output_path)
        
        return SpaceSearchResult(best_params          = self.best_params, 
                                 best_score           = self.best_score, 
                                 study                = self.study, 
                                 n_trials             = len(self.study.trials),
                                 output_dir           = str(output_path),
                                 optimization_summary = self.get_optimization_summary())

    def run_cv_study(self, partitions: List[Dict[str, Union[np.ndarray, pd.DataFrame]]], 
                     study_name     : str = "optuna_cv_study",
                     direction      : str = "maximize",
                     metric         : Union[str, Callable] = "r2",
                     output_dir     : str = "./optuna_output",
                     save_study     : bool = True,
                     patience       : Optional[int] = None,
                     pruner         : Optional[optuna.pruners.BasePruner] = None,
                     timeout        : Optional[int] = None,
                     catch          : Union[tuple, Sequence[Exception]] = (Exception,),
                     callbacks      : Optional[List[Callable]] = None,
                     lambda_penalty : float = 0.0
                    ) -> SpaceSearchResult:
        """
        ____________________________________________________________________________________________________________________
        Run hyperparameter optimization using multiple train/validation partitions
        ____________________________________________________________________________________________________________________
        Parameters:        
            - partitions (List[Dict])                             : List of dictionaries, each containing:
                                                                    - 'X_train' : Training features.
                                                                    - 'y_train' : Training targets.
                                                                    - 'X_val'   : Validation features.
                                                                    - 'y_val'   : Validation targets.
                                                                    - 'scaler'  : Optional scaler to apply to predictions 
                                                                                  else None.
            - study_name     (str)                                : Name for the study
            - direction      (str)                                : Optimization direction ('maximize' or 'minimize')
            - metric         (Union[str, Callable])               : Metric to optimize
            - output_dir     (str)                                : Directory to save results
            - save_study     (bool)                               : Whether to save the study
            - patience       (Optional[int])                      : Early stopping patience
            - pruner         (Optional[optuna.pruners.BasePruner] : Optuna pruner
            - lambda_penalty (float)                              : Weight for penalizing standard deviation across 
                                                                    partitions
        ____________________________________________________________________________________________________________________
        Returns:
            - SpaceSearchResult : Results from the optimization study
        ____________________________________________________________________________________________________________________
        Notes:
            - Each partition should contain its own training and validation data.
            - If lambda_penalty > 0, the objective will be mean_score - lambda_penalty * std_score.
            - If patience is set, early stopping will be applied based on mean validation performance across partitions.
            - If save_study is True, visualizations and study state will be saved to output.
        ____________________________________________________________________________________________________________________
        
        """
        # Validate partitions
        self.__validate_partitions(partitions)
        
        # Set up early stopping
        self._early_stopping['patience'] = patience
        
        # Create output directory
        output_path = Path(output_dir) / study_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metric
        scorer = self.metric_resolver.resolve(metric)[0]
        
        # Create or load study
        self.study = optuna.create_study(direction=direction, study_name=study_name, pruner=pruner, sampler=self.sampler,
                                         storage        = self.config.storage,
                                         load_if_exists = self.config.load_if_exists)
        
        def objective(trial: optuna.trial.Trial) -> float:
            try:
                # Get initial resources for this trial
                resources        = self.resource_manager.acquire_resources(trial.number)
                partition_scores = []
            
                # Calculate resources per partition
                n_partitions = len(partitions)
                resources_per_partition = {'n_jobs': max(1, resources['n_jobs'] // n_partitions),
                                           'device': resources['device']}
            
                # Train and evaluate on each partition with allocated resources
                for idx, partition in enumerate(partitions):
                    try:
                        # Create model with partition-specific resources
                        model = self.model_builder.create_model(trial=trial,
                                                                device=resources_per_partition['device'],
                                                                n_jobs=resources_per_partition['n_jobs'])
                        
                        # Train and evaluate
                        score = self.__evaluate_partition(model=model, partition=partition, 
                                                          scorer    = scorer, 
                                                          direction = direction)
                        partition_scores.append(score)
                        
                        # Report intermediate values
                        trial.set_user_attr(f'partition_{idx}_score', score)
                        
                        # Optional: Release GPU memory between partitions
                        if 'cuda' in resources_per_partition['device']:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        self.logger.warning(f"Trial {trial.number}, Partition {idx} failed: {e}")
                        partition_scores.append(-float('inf') if direction == "maximize" else float('inf'))
            
                # Calculate aggregate score
                mean_score = np.mean(partition_scores)
                std_score = np.std(partition_scores)
                
                # Store partition-specific information
                trial.set_user_attr('partition_scores', partition_scores)
                trial.set_user_attr('score_std', std_score)
                trial.set_user_attr('score_mean', mean_score)
                
                # Apply penalty if requested
                final_score = mean_score - lambda_penalty * std_score
                
                return final_score
            
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed completely: {e}")
                return -float('inf') if direction == "maximize" else float('inf')
                
            finally:
                # Always release resources
                self.resource_manager.release_resources(trial.number)
    
        # Set up callbacks
        study_callbacks = callbacks or []
        if patience:
            study_callbacks.append(self.__early_stopping_callback)
        
        # Get optimal number of parallel trials based on resources and partitions
        n_jobs = self.resource_manager.get_suggested_parallel_trials()
        
        # Adjust n_jobs based on number of partitions
        n_jobs = max(1, n_jobs // len(partitions))
        
        # Optimize
        self.study.optimize(objective, n_trials=self.config.n_trials, timeout=timeout, catch=catch,
                            callbacks = study_callbacks if study_callbacks else None,
                            n_jobs    = n_jobs)
        
        # Store results
        self.best_params    = self.study.best_params
        self.best_score     = self.study.best_value
        self.trials_history = self.study.trials
        
        # Generate visualization and save results
        if save_study:
            self.__save_cv_results(output_path)
            StudyPersistence.save_study(self.study, output_path)
            self.__plot_cv_distributions(pd.DataFrame([
                    {
                        'trial'           : t.number,
                        'mean_score'      : t.user_attrs.get('score_mean', None),
                        'std_score'       : t.user_attrs.get('score_std', None),
                        'partition_scores': t.user_attrs.get('partition_scores', [])
                    }
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]),
                output_path
            )
        
        return SpaceSearchResult(best_params          = self.best_params,
                                best_score           = self.best_score,
                                study                = self.study,
                                n_trials             = len(self.study.trials),
                                output_dir           = str(output_path),
                                optimization_summary = self.get_optimization_summary())

    def __validate_partitions(self, partitions: List[Dict]) -> None:
        """Validate format and consistency of partitions"""
        if not partitions:
            raise ValueError("No partitions provided")
        
        required_keys = {'X_train', 'y_train', 'X_val', 'y_val'}
        for i, partition in enumerate(partitions):
            if not all(key in partition for key in required_keys):
                raise ValueError(f"Partition {i} missing required keys: {required_keys}")
            
            self.__validate_data(partition['X_train'], partition['y_train'], partition['X_val'], partition['y_val'])

    def __evaluate_partition(self, model: MLTreeRegressor, partition: Dict, scorer: Callable, direction: str) -> float:
        """Train and evaluate model on a single partition, applying scaler if provided else return unscaled results"""
        X_train, y_train = partition['X_train'], partition['y_train']
        X_val, y_val     = partition['X_val'], partition['y_val']
        scaler           = partition.get('scaler', None)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Apply scaling if provided
        if scaler is not None:
            y_pred       = y_pred * scaler
            y_val_scaled = y_val * scaler
            score = float(scorer(y_val_scaled, y_pred))
        else:
            score = float(scorer(y_val, y_pred))
        
        return score

    def __save_cv_results(self, output_path: Path) -> None:
        """Save CV-specific results and visualizations (experimental)"""
        # Extract partition-specific information
        cv_results = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                cv_results.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'mean_score': trial.value,
                    'std_score': trial.user_attrs.get('score_std', None),
                    'partition_scores': trial.user_attrs.get('partition_scores', [])
                })
        
        # Save detailed CV results
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.to_csv(output_path / 'cv_results.csv', index=False)
        
        # Create CV-specific visualizations
        self.__plot_cv_distributions(cv_results_df, output_path)

    def __plot_cv_distributions(self, cv_results: pd.DataFrame, output_path: Path) -> None:
        """Create CV-specific visualization plots (experimental)"""
        try:
            # Plot score distributions across partitions
            plt.figure(figsize=(10, 6))
            plt.boxplot(np.vstack(cv_results['partition_scores'].values).T)
            plt.title('Score Distribution Across Partitions')
            plt.ylabel('Score')
            plt.xlabel('Partition')
            plt.savefig(output_path / 'cv_score_distributions.png')
            plt.close()
            
            # Plot mean vs std
            plt.figure(figsize=(10, 6))
            plt.scatter(cv_results['mean_score'], cv_results['std_score'])
            plt.xlabel('Mean Score')
            plt.ylabel('Score Standard Deviation')
            plt.title('Mean-Std Score Trade-off')
            plt.savefig(output_path / 'cv_mean_std_tradeoff.png')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create CV visualizations: {e}")

#---------------------------------------------------------------------------------------------------------------------------#