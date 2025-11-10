# Modules -----------------------------------------------------------------------------------------------------------------#
import optuna
import joblib
import warnings
import logging
import torch

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from dataclasses             import dataclass
from pathlib                 import Path
from typing                  import Dict, List, Optional, Union, Callable, Any, Sequence, TypeVar, Literal
from sklearn.metrics         import get_scorer

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.mltrees.regressor  import MLTreeRegressor
from src.optim.grid                import RandomForestGrid, LightGBMGrid, XGBoostGrid, DARTBoostGrid
from src.utils.callbacks           import huber_loss

# Type definitions --------------------------------------------------------------------------------------------------------#
T         = TypeVar('T')
ModelType = Literal["random_forest", "lightgbm", "xgboost", "dartboost"]

# Configuration and result classes ----------------------------------------------------------------------------------------#
@dataclass
class SpaceSearchConfig:
    """Configuration settings for SpaceSearch optimization"""
    model_type : ModelType
    n_jobs         : int = 10                                               # Core numbers for parallel processing
    n_trials       : int = 100                                              # Number of trials for optimization
    device         : str = "cuda" if torch.cuda.is_available() else "cpu"   # Device to use ('cpu' or 'cuda')
    verbose        : bool = True                                            # Verbosity flag
    seed           : int = 42                                               # Fixed seed for reproducibility
    sampler        : Optional[optuna.samplers.BaseSampler] = None           # Custom sampler, defaults to TPESampler
    storage        : Optional[str] = None                                   # Storage URL for study persistence
    load_if_exists : bool = False                                           # Checkpoint loading flag
    huber_delta    : float = 1.0                                            # Delta parameter for Huber loss (typical: 1.0-2.0)

@dataclass
class SpaceSearchResult:
    """Results from a SpaceSearch optimization study"""
    best_params          : Dict[str, Any]                  
    best_score           : float
    study                : optuna.study.Study
    n_trials             : int
    output_dir           : str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format"""
        output_dict = {
                    'best_params': self.best_params, 
                    'best_score' : self.best_score, 
                    'n_trials'   : self.n_trials,
                    'output_dir'  : self.output_dir}

        return output_dict



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
        
        # Initialize sampler
        self.sampler = config.sampler or optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
        
        # Parameter grid mapping
        self.param_grid_map = {
            "random_forest" : RandomForestGrid,
            "lightgbm"      : LightGBMGrid,
            "xgboost"       : XGBoostGrid,
            "dartboost"     : DARTBoostGrid
                              }
        
        # Custom metrics registry
        self.custom_metrics = {
            "huber": lambda y_true, y_pred: huber_loss(y_true, y_pred, delta=self.config.huber_delta)
        }
        
        # Internal state
        self.study          : Optional[optuna.study.Study]   = None
        self.best_params    : Optional[Dict[str, Any]]       = None
        self.best_score     : Optional[float]                = None
        self.trials_history : List[optuna.trial.FrozenTrial] = []
        
        # Define early stopping state
        self._early_stopping = {
            'patience'         : None,
            'best_value'       : None,
            'no_improve_count' : 0
                               }
        
        # Log initialization
        if self.config.verbose:
            self.logger.info(
                f"SpaceSearch initialized for {self.config.model_type} "
                f"model (device={self.config.device})"
                )
 
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
    
    def __create_model(self, trial: optuna.trial.Trial, features_names: List[str]) -> MLTreeRegressor:
        """Create a model instance with parameters from trial"""
        if self.config.model_type not in self.param_grid_map:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
        
        param_grid = self.param_grid_map[self.config.model_type](trial)
        model      = MLTreeRegressor(
                        model_type  = self.config.model_type,
                        model_params = param_grid,
                        feat_names   = features_names,
                        device       = self.config.device,
                        n_jobs       = self.config.n_jobs
                        )
        return model
    
    def __resolve_metric(self, metric: Union[str, Callable]) -> Callable:
        """Resolve metric to callable function"""
        if isinstance(metric, str):
            # Check if it's a custom metric first
            if metric.lower() in self.custom_metrics:
                return self.custom_metrics[metric.lower()]
            else:
                # Use sklearn's get_scorer for standard metrics
                scorer = get_scorer(metric)
                return scorer._score_func
        elif callable(metric):
            return metric
        else:
            raise ValueError(f"Invalid metric type: {type(metric)}")
    
    def __save_study(self, output_path: Path, compress: bool = True) -> None:
        """Save study state to disk"""
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"study_state{'_compressed' if compress else ''}.joblib"
        joblib.dump(self.study, save_path, compress=compress)
    
    def __create_visualizations(self, output_path: Path) -> None:
        """Generate and save visualization plots"""
        try:
            from optuna.visualization.matplotlib import (
                plot_param_importances,
                plot_optimization_history,
                plot_parallel_coordinate,
                plot_contour
            )
            
            plots = {
                "optimization_history": plot_optimization_history(self.study),
                "param_importances": plot_param_importances(self.study),
                "parallel_coordinate": plot_parallel_coordinate(self.study),
                "contour": plot_contour(self.study)
            }
            
            for name, fig in plots.items():
                fig.figure.savefig(output_path / f"{name}.jpg", bbox_inches="tight", dpi=500)
                plt.close(fig.figure)
                
        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")
    
    def __validate_data(self, X_train, y_train, X_val, y_val) -> None:
        """Validate input data shapes and types"""
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("Features and target sizes don't match")
        
        if isinstance(X_train, pd.DataFrame) != isinstance(X_val, pd.DataFrame):
            raise ValueError("X_train and X_val must be the same type")
            
        if isinstance(y_train, pd.Series) != isinstance(y_val, pd.Series):
            raise ValueError("y_train and y_val must be the same type")
    
    def __normalize_partition(self, X_train, y_train, X_val, y_val, features_names: Optional[List[str]] = None,
                             weights : Optional[pd.Series] = None,
                             scaler  : Optional[pd.Series] = None) -> Dict[str, Any]:
        """Normalize inputs into standard partition format"""
        # Extract feature names if not provided
        if features_names is None:
            if isinstance(X_train, pd.DataFrame):
                features_names = X_train.columns.tolist()
            else:
                features_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'features_names': features_names,
            'weights': weights,
            'scaler': scaler
        }
    
    def __create_objective(self, partitions: Union[Dict, List[Dict]], scorer: Callable, direction: str, 
                           lambda_penalty: float = 0.0) -> Callable[[optuna.trial.Trial], float]:
        """
        Create unified objective function for both single partition and cross-validation.
        
        Args:
            partitions: Single partition dict or list of partition dicts
            scorer: Metric scoring function
            direction: 'maximize' or 'minimize'
            lambda_penalty: Penalty for std deviation in CV mode (0.0 for single partition)
        """
        is_cv          = isinstance(partitions, list)
        partition_list = partitions if is_cv else [partitions]
        
        def objective(trial: optuna.trial.Trial) -> float:
            try:
                scores = []
                
                for idx, partition in enumerate(partition_list):
                    try:
                        # Create model
                        model = self.__create_model(trial, features_names=partition['features_names'])
                        
                        # Evaluate partition
                        score = self.__evaluate_partition(model, partition, scorer)
                        scores.append(score)
                        
                        # Report intermediate values for CV
                        if is_cv:
                            trial.set_user_attr(f'partition_{idx}_score', score)
                    
                    except Exception as e:
                        self.logger.warning(f"Trial {trial.number}, Partition {idx} failed: {e}")
                        scores.append(-float('inf') if direction == "maximize" else float('inf'))
                
                # Calculate final score
                if is_cv:
                    mean_score = np.mean(scores)
                    std_score  = np.std(scores)
                    trial.set_user_attr('partition_scores', scores)
                    trial.set_user_attr('score_std', std_score)
                    trial.set_user_attr('score_mean', mean_score)
                    return mean_score - lambda_penalty * std_score
                else:
                    return scores[0]
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                return -float('inf') if direction == "maximize" else float('inf')
        
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
    
    def optimize(self, 
                 X_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 y_train: Optional[Union[np.ndarray, pd.Series]] = None,
                 X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 y_val: Optional[Union[np.ndarray, pd.Series]] = None,
                 partitions: Optional[List[Dict[str, Any]]] = None,
                 features_names: Optional[List[str]] = None,
                 study_name: str = "optuna_study",
                 direction: str = "maximize",
                 metric: Union[str, Callable] = "r2",
                 output_dir: str = "./optuna_output",
                 save_study: bool = True,
                 patience: Optional[int] = None,
                 pruner: Optional[optuna.pruners.BasePruner] = None,
                 timeout: Optional[int] = None,
                 catch: Union[tuple, Sequence[Exception]] = (Exception,),
                 callbacks: Optional[List[Callable]] = None,
                 weights: Optional[pd.Series] = None,
                 scaler: Optional[pd.Series] = None,
                 lambda_penalty: float = 0.0
                ) -> SpaceSearchResult:
        """
        ____________________________________________________________________________________________________________________
        Unified hyperparameter optimization method supporting both single partition and cross-validation modes
        ____________________________________________________________________________________________________________________
        Parameters:
            Single Partition Mode (provide these):
                - X_train (Union[np.ndarray, pd.DataFrame]) : Training features
                - y_train (Union[np.ndarray, pd.Series])    : Training targets
                - X_val (Union[np.ndarray, pd.DataFrame])   : Validation features
                - y_val (Union[np.ndarray, pd.Series])      : Validation targets
            
            Cross-Validation Mode (provide this instead):
                - partitions (List[Dict])                   : List of partition dicts, each containing:
                                                              'X_train', 'y_train', 'X_val', 'y_val'
                                                              Optional: 'features_names', 'weights', 'scaler'
            
            Common Parameters:
                - features_names (Optional[List[str]])          : Feature names (auto-detected if not provided)
                - study_name (str)                              : Name for the study
                - direction (str)                               : 'maximize' or 'minimize'
                - metric (Union[str, Callable])                 : Metric to optimize
                - output_dir (str)                              : Directory to save results
                - save_study (bool)                             : Whether to save study and visualizations
                - patience (Optional[int])                      : Early stopping patience
                - pruner (Optional[optuna.pruners.BasePruner])  : Optuna pruner
                - timeout (Optional[int])                       : Timeout in seconds
                - catch (Union[tuple, Sequence[Exception]])     : Exceptions to catch
                - callbacks (Optional[List[Callable]])          : Additional callbacks
                - weights (Optional[pd.Series])                 : Sample weights (single partition mode)
                - scaler (Optional[pd.Series])                  : Scaler for predictions (single partition mode)
                - lambda_penalty (float)                        : Penalty for std in CV mode (default: 0.0)
        ____________________________________________________________________________________________________________________
        Returns:
            - SpaceSearchResult: Results from the optimization study
        ____________________________________________________________________________________________________________________
        Notes:
            - Mode is auto-detected: if partitions is provided, uses CV mode; otherwise uses single partition mode
            - For single partition: provide X_train, y_train, X_val, y_val
            - For CV: provide partitions list
            - For Huber loss: use metric="huber" with direction="minimize"
        ____________________________________________________________________________________________________________________
        """
        # Auto-detect mode
        if partitions is not None:
            # Cross-validation mode
            if any([X_train is not None, y_train is not None, X_val is not None, y_val is not None]):
                warnings.warn("Both partitions and single partition data provided. Using CV mode with partitions.")
            
            # Validate and normalize partitions
            self.__validate_partitions(partitions)
            partitions_normalized = self.__normalize_partitions(partitions)
            is_cv = True
            
        else:
            # Single partition mode
            if any([x is None for x in [X_train, y_train, X_val, y_val]]):
                raise ValueError("For single partition mode, must provide X_train, y_train, X_val, y_val")
            
            # Validate data
            self.__validate_data(X_train, y_train, X_val, y_val)
            
            # Normalize to partition format
            partitions_normalized = self.__normalize_partition(X_train, y_train, X_val, y_val, features_names, 
                                                               weights, scaler)
            is_cv = False
        
        # Set up early stopping
        self._early_stopping['patience'] = patience
        
        # Create output directory
        output_path = Path(output_dir) / study_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metric
        scorer = self.__resolve_metric(metric)
        
        # Create or load study
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=pruner,
            sampler=self.sampler,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists
        )
        
        # Create unified objective
        objective = self.__create_objective(partitions     = partitions_normalized,
                                            scorer         = scorer,
                                            direction      = direction,
                                            lambda_penalty = lambda_penalty if is_cv else 0.0)
        # Set up callbacks
        study_callbacks = callbacks or []
        if patience:
            study_callbacks.append(self.__early_stopping_callback)
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=timeout,
            catch=catch,
            callbacks=study_callbacks if study_callbacks else None,
            n_jobs=1
        )
        
        # Store results
        self.best_params    = self.study.best_params
        self.best_score     = self.study.best_value
        self.trials_history = self.study.trials
        
        # Generate visualization and save study
        if save_study:
            if is_cv:
                self.__save_cv_results(output_path)
                self.__plot_cv_distributions(pd.DataFrame([
                    {
                        'trial': t.number,
                        'mean_score': t.user_attrs.get('score_mean', None),
                        'std_score': t.user_attrs.get('score_std', None),
                        'partition_scores': t.user_attrs.get('partition_scores', [])
                    }
                    for t in self.study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]), output_path)
            
            self.__create_visualizations(output_path)
            self.__save_study(output_path)

        return SpaceSearchResult(best_params=self.best_params, best_score=self.best_score, 
                                 study      = self.study,
                                 n_trials   = len(self.study.trials),
                                 output_dir = str(output_path))
    
    def __validate_partitions(self, partitions: List[Dict]) -> None:
        """Validate format and consistency of partitions"""
        if not partitions:
            raise ValueError("No partitions provided")
        
        required_keys = {'X_train', 'y_train', 'X_val', 'y_val'}
        for i, partition in enumerate(partitions):
            if not all(key in partition for key in required_keys):
                raise ValueError(f"Partition {i} missing required keys: {required_keys}")
            
            self.__validate_data(partition['X_train'], partition['y_train'], partition['X_val'], partition['y_val'])
    
    def __normalize_partitions(self, partitions: List[Dict]) -> List[Dict]:
        """Normalize partitions to ensure they have all required keys"""
        normalized = []
        for i, partition in enumerate(partitions):
            # Get existing or infer features_names
            if 'features_names' not in partition and 'feats' not in partition:
                X_train = partition['X_train']
                if isinstance(X_train, pd.DataFrame):
                    features_names = X_train.columns.tolist()
                else:
                    features_names = [f"feature_{j}" for j in range(X_train.shape[1])]
            else:
                features_names = partition.get('features_names', partition.get('feats', []))
            
            normalized.append({
                'X_train': partition['X_train'],
                'y_train': partition['y_train'],
                'X_val': partition['X_val'],
                'y_val': partition['y_val'],
                'features_names': features_names,
                'weights': partition.get('weights', None),
                'scaler': partition.get('scaler', None)
            })
        return normalized

    def __evaluate_partition(self, model: MLTreeRegressor, partition: Dict, scorer: Callable) -> float:
        """Train and evaluate model on a single partition, applying scaler if provided else return unscaled results"""
        X_train = partition['X_train']
        y_train = partition['y_train']
        X_val = partition['X_val']
        y_val = partition['y_val']
        weights = partition.get('weights', None)
        scaler = partition.get('scaler', None)
        
        # Train model
        model.fit(X_train, y_train, sample_weight=weights)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Apply scaling if provided
        if scaler is not None:
            y_pred       = y_pred * scaler
            y_val_scaled = y_val * scaler
            score = float(scorer(y_true=y_val_scaled, y_pred=y_pred))
        else:
            score = float(scorer(y_true=y_val, y_pred=y_pred))
        
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