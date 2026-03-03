# Modules -----------------------------------------------------------------------------------------------------------------#
import optuna
import joblib
import warnings
import logging
import torch
import gc

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib         import Path
from typing          import Dict, List, Optional, Union, Callable, Any, Sequence
from sklearn.metrics import get_scorer
            
# Custom functions --------------------------------------------------------------------------------------------------------#

# Main regressor constructors
from src.models.mlbasics.regressor import MLBasicRegressor
from src.models.mltrees.regressor  import MLTreeRegressor
from src.models.dltab.regressor    import DLTabularRegressor

# Grid of hyperparams to optimize
from src.optim.grid import ElasticNetGrid, LinearSVRGrid, RandomForestGrid, LightGBMGrid, XGBoostGrid, MLPGrid, NODEGrid

# Import helper functions for each type of regressor framework
from src.optim.utils._ml    import validate_data_ml, normalize_partitions_ml, evaluate_partition_ml
from src.optim.utils._dltab import validate_data_dl, normalize_partitions_dl, evaluate_partition_dl

# Custom function for Huber loss
from src.utils.eval import huber_loss

# Visualization plots
from src.optim.utils._visuals import create_visualizations_per_study, plot_cv_evol_distributions

# Configuration and result format dictionaries
from src.optim.utils._formats import SpaceSearchConfig, SpaceSearchResult

# Hyperparameter search ---------------------------------------------------------------------------------------------------#
class SpaceSearch:
    """
    ________________________________________________________________________________________________________________________
    SpaceSearch: A comprehensive hyperparameter optimization class using Optuna for Supervised Regression
    ________________________________________________________________________________________________________________________
    Features:
    -> Support for multiple model types (ElasticNet, SVR, LightGBM, XGBoost, Random Forest, MultiLayer Perceptron)
    -> Customizable search spaces and objective functions
    -> Advanced visualization and analysis tools
    -> Study persistence and loading
    -> Early stopping and pruning support
    -> Multi-objective optimization capabilities
    ________________________________________________________________________________________________________________________
    Note:
    -> SpaceSearch is made to work upon MLBasicRegressor, MLTreeRegressor, and DLTabularRegressor.
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, config: SpaceSearchConfig):
        """Initialize SpaceSearch with configuration"""
        
        # Store configuration and set up logging
        self.config = config
        self.logger = self.__setup_logging()
        
        # Initialize sampler
        self.sampler = config.sampler or optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
        
        # Tag of regressor type and mapping to model types
        self.model_grid_map = {
            "mlbasic": ["elasticnet", "linearsvr"],
            "mltrees": ["rf", "lightgbm", "xgboost"],
            "dltab"  : ["mlp", "node"]}
        
        # Config already normalizes model_type in __post_init__, so we can use it directly
        self.model_type = config.model_type
        self.regressor  = next((key for key, models in self.model_grid_map.items() if self.model_type in models), None)
        
        if self.regressor is None:
            valid_models = [model for models in self.model_grid_map.values() for model in models]
            raise ValueError(f"Unsupported model_type: '{config.model_type}'. "
                             f"Valid options are: {valid_models}")
        
        # Parameter grid mapping
        self.param_grid_map = {
            "elasticnet" : ElasticNetGrid,
            "linearsvr"  : LinearSVRGrid,
            "rf"         : RandomForestGrid,
            "lightgbm"   : LightGBMGrid,
            "xgboost"    : XGBoostGrid,
            "mlp"        : MLPGrid,
            "node"       : NODEGrid
                              }
        
        # Dispatch maps for validation and normalization (avoids repetitive if/elif chains)
        self._validate_fn  = {"mlbasic" : validate_data_ml, 
                              "mltrees" : validate_data_ml, 
                              "dltab"   : validate_data_dl}
        self._normalize_fn = {"mlbasic" : normalize_partitions_ml, 
                              "mltrees" : normalize_partitions_ml, 
                              "dltab"   : normalize_partitions_dl}
        
        # Custom metrics registry
        self.custom_metrics = {"huber": lambda y_true, y_pred: huber_loss(y_true, y_pred, delta=self.config.huber_delta)}
        
        # Internal state
        self.study          : Optional[optuna.study.Study]   = None
        self.best_params    : Optional[Dict[str, Any]]       = None
        self.best_score     : Optional[float]                = None
        self.trials_history : List[optuna.trial.FrozenTrial] = []
        
        # Define early stopping state
        self._early_stopping = {'patience' : None, 'best_value' : None, 'no_improve_count' : 0}
        
        # Log initialization
        if self.config.verbose:
            self.logger.info(f"SpaceSearch initialized for {self.model_type} model (device={self.config.device})")
    
    # [Helper] Logging ----------------------------------------------------------------------------------------------------#
    def __setup_logging(self) -> logging.Logger:
        """Configure logging for the optimization process"""
        logger = logging.getLogger("SpaceSearch")
        if not logger.handlers:
            
            handler   = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.config.verbose else logging.WARNING)

        return logger
    
    # [Helper] Model creation ---------------------------------------------------------------------------------------------#
    def __create_model(self, trial: optuna.trial.Trial, features_names: List[str]
                       ) -> Union[MLBasicRegressor, MLTreeRegressor, DLTabularRegressor]:
        """Create a model instance with parameters from trial"""
        if self.model_type not in self.param_grid_map:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # Get trial parameters from grid
        trial_params = self.param_grid_map[self.model_type](trial)
        
        # Route based on model type
        if self.regressor == "dltab":
            
            # Deep Learning path: specify fixed and optimizable parameters
            model_params = self.config.dl_architecture["model_params"]     if self.config.dl_architecture else {}
            opt_params   = self.config.dl_architecture["optimizer_params"] if self.config.dl_architecture else {}
            loss_params  = self.config.dl_architecture["loss_params"]      if self.config.dl_architecture else {}
            
            # Merge fixed and trial parameters (overwrite the fixed with trial where applicable)
            merge_model = {**model_params, **trial_params["model_params"]}
            merge_opt   = {**opt_params, **trial_params["optimizer_params"]}
            merge_loss  = {**loss_params, **trial_params["loss_params"]}
            
            # Store loss params for later use
            trial.set_user_attr("delta", merge_loss.get("delta", 1.0))
            
            # Extract optimizer name from config and remove from merge_opt to avoid passing as kwarg
            optimizer_name = merge_opt.pop("optimizer_name", "adam")
            
            # Extract use_amp from model params (training flag, not architecture param)
            use_amp = merge_model.pop("use_amp", False)
            
            # Infer in_features from feature names
            in_features = len(features_names)
            
            # Construct the Regressor with the given params
            model = DLTabularRegressor(model_type       = self.model_type,
                                       in_features      = in_features,
                                       model_params     = merge_model,
                                       optimizer_name   = optimizer_name,
                                       optimizer_params = merge_opt,
                                       feat_names       = features_names,
                                       device           = self.config.device,
                                       use_amp          = use_amp,
                                       verbose          = self.config.verbose)
        
        # Route for tree-based models
        elif self.regressor == "mltrees":
            
            # Construct the Regressor with the given params
            model = MLTreeRegressor(model_type = self.model_type, model_params = trial_params, 
                                    feat_names = features_names,
                                    device     = self.config.device,
                                    n_jobs     = self.config.n_jobs)
        
        # Route for basic linear and kernel-based models
        elif self.regressor == "mlbasic":
            
            # Construct the Regressor with the given params
            model = MLBasicRegressor(model_type = self.model_type, model_params = trial_params, 
                                     feat_names = features_names,
                                     use_scaler = self.config.use_scaler,
                                     device     = self.config.device,
                                     n_jobs     = self.config.n_jobs)
        
        else:
            raise ValueError(f"Unknown regressor type: {self.regressor}")
        
        return model
    
    # [Helper] GPU memory cleanup -----------------------------------------------------------------------------------------#
    def __cleanup_gpu_memory(self, model: Optional[Union[MLTreeRegressor, MLBasicRegressor, DLTabularRegressor]] = None
                             ) -> None:
        """Clean up GPU memory after trial to prevent memory leaks"""
        # Delete model if provided
        if model is not None:
            del model
        
        # Force garbage collection before clearing CUDA cache
        gc.collect()
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available() and self.config.device != 'cpu':
            torch.cuda.empty_cache()
            if self.config.verbose:
                # Get current memory usage (in MB)
                allocated = torch.cuda.memory_allocated() / 1024**2 
                reserved  = torch.cuda.memory_reserved() / 1024**2 
                self.logger.debug(f"GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
    
    # [Helper] Metric resolution ------------------------------------------------------------------------------------------#
    def __resolve_metric(self, metric: Union[str, Callable]) -> Callable:
        """Resolve metric to callable function for scoring after the fit"""
        
        # If a str given, check custom metrics first and then sklearn
        if isinstance(metric, str):
            
            # If a self implemented metric, return the corresponding function
            if metric.lower() in self.custom_metrics:
                return self.custom_metrics[metric.lower()]
            # Elif use sklearn's get_scorer. This returns a function that accepts (y_true, y_pred, **kwargs) 
            else:
                scorer = get_scorer(metric)
                return scorer._score_func
            
        # Elif a callable given, return as is
        elif callable(metric):
            return metric
        
        # Else, invalid metric type
        else:
            raise ValueError(f"Invalid metric type: {type(metric)}")
    
    # [Helper] Save study state -------------------------------------------------------------------------------------------#
    def __save_study(self, output_path: Path, compress: bool = True) -> None:
        """Save study state to disk"""
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / f"study_state{'_compressed' if compress else ''}.joblib"
        joblib.dump(self.study, save_path, compress=compress)
            
    # [Helper] Define Optuna objective function ---------------------------------------------------------------------------#
    def __create_objective(self, partitions: List[Dict], scorer: Callable, direction: str, is_cv : bool = True, 
                           lambda_penalty : float = 0.0
                           ) -> Callable[[optuna.trial.Trial], float]:
        """Create the Optuna objective function for optimization, main check for CV or single partition mode"""
        
        # Set the list of partitions
        partition_list = partitions
        
        def objective(trial: optuna.trial.Trial) -> float:
            # Initialize model reference for cleanup
            model = None  
            
            try: 
                # Initialize list to store scores for each partition
                scores = []
                
                # Loop through each partition and evaluate the model
                for idx, partition in enumerate(partition_list):
                    # Create model
                    model = self.__create_model(trial, features_names=partition['features_names'])
                    
                    # Evaluate partition
                    score = self.__evaluate_partition(model, partition, scorer, trial=trial)
                    scores.append(score)
                    
                    # Report intermediate values for CV
                    if is_cv:
                        trial.set_user_attr(f'partition_{idx}_score', score)
                    
                    # Clean up GPU memory after each partition evaluation
                    self.__cleanup_gpu_memory(model)
                    model = None  
                
                # Calculate final score
                if is_cv:
                    mean_score = float(np.mean(scores))
                    std_score  = float(np.std(scores)) if np.isfinite(np.std(scores)) else 0.0
                    trial.set_user_attr('partition_scores', scores)
                    trial.set_user_attr('score_std', std_score)
                    trial.set_user_attr('score_mean', mean_score)
                    final_score = mean_score - lambda_penalty * std_score
                    
                    # Guard: return worst-case finite value if score is nan/inf (bad trial)
                    if not np.isfinite(final_score):
                        return -float('inf') if direction == "maximize" else float('inf')
                    return final_score
                else:
                    score = scores[0]
                    if not np.isfinite(score):
                        return -float('inf') if direction == "maximize" else float('inf')
                    return score
            
            # If trial is pruned, re-raise the exception to let Optuna handle it    
            except optuna.TrialPruned:
                raise
            
            # Catch other exceptions to prevent study from crashing and log them
            except Exception as e:
                self.logger.warning(f"Trial {trial.number} failed: {e}")
                return -float('inf') if direction == "maximize" else float('inf')
            
            # At the end ensure GPU memory is cleaned up to prevent leaks, even if an exception occurred
            finally:
                if model is not None:
                    self.__cleanup_gpu_memory(model)
        
        return objective
    
    # [Helper] callback for early stopping --------------------------------------------------------------------------------#
    def __early_stopping_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Handle early stopping logic"""
        current = study.best_value
        
        # Update best value and no improvement count
        if self._early_stopping['best_value'] is None or (
            (study.direction == optuna.study.StudyDirection.MAXIMIZE and current > self._early_stopping['best_value']) or
            (study.direction == optuna.study.StudyDirection.MINIMIZE and current < self._early_stopping['best_value'])):
            
            self._early_stopping['best_value']       = current
            self._early_stopping['no_improve_count'] = 0
        
        # No improvement, add to count
        else:
            self._early_stopping['no_improve_count'] += 1

        # Stop if patience exceeded 
        if (self._early_stopping['patience'] is not None and 
            self._early_stopping['no_improve_count'] >= self._early_stopping['patience']):
            
            self.logger.info(f"Early stopping: no improvement in {self._early_stopping['patience']} trials.")
            study.stop()

    # [Helper] Router for input validation of data-------------------------------------------------------------------------#
    def __validate_data(self, partitions: List[Dict]) -> None:
        """Validate input data shapes and types"""
        self._validate_fn[self.regressor](partitions)
        
    # [Helper] Router to normalize partitions and ensure standardization of format ----------------------------------------#
    def __normalize_partitions(self, partitions: List[Dict]) -> List[Dict]:
        """Normalize partitions for both ML and DL models"""
        if self.regressor not in self._normalize_fn:
            raise ValueError(f"Unknown regressor type: {self.regressor}")
        return self._normalize_fn[self.regressor](partitions)

    # [Helper] Router to evaluation methods acording to the type of model selected ----------------------------------------#
    def __evaluate_partition(self, model: Union[MLBasicRegressor, MLTreeRegressor, DLTabularRegressor], 
                             partition : Dict, 
                             scorer    : Callable, 
                             trial     : Optional[optuna.trial.Trial] = None
                             ) -> float:
        """Router to appropriate evaluation method based on model type"""
        
        if isinstance(model, DLTabularRegressor):
            if trial is None:
                raise ValueError("Trial object required for DL model evaluation")
            return evaluate_partition_dl(model, partition, scorer, trial, self.logger, self.config)
        
        elif isinstance(model, MLTreeRegressor) or isinstance(model, MLBasicRegressor):
            return evaluate_partition_ml(model, partition, scorer)
        
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    # [Helper] Extract CV results as DataFrame ----------------------------------------------------------------------------#
    def __extract_cv_dataframe(self) -> pd.DataFrame:
        """Extract CV results from completed trials into a DataFrame"""
        return pd.DataFrame([
            {
                'trial_number'     : t.number,
                'params'           : t.params,
                'mean_score'       : t.user_attrs.get('score_mean', t.value),
                'std_score'        : t.user_attrs.get('score_std', None),
                'partition_scores' : t.user_attrs.get('partition_scores', [])
            }
            for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
    
    # Store CV-specific results and visualizations ------------------------------------------------------------------------#  
    def __save_cv_results(self, output_path: Path) -> None:
        """Save CV-specific results and visualizations (experimental)"""
        # Extract partition-specific information using helper method
        cv_results_df = self.__extract_cv_dataframe()
        
        # Save detailed CV results
        cv_results_df.to_csv(output_path / 'cv_results.csv', index=False)
        
        # Create CV-specific visualizations using imported function
        plot_cv_evol_distributions(cv_results_df, output_path)
    
    # Main Optimization Method --------------------------------------------------------------------------------------------#
    def optimize(self, 
                 partitions     : Optional[List[Dict]]                      = None,
                 study_name     : str                                       = "optuna_study",
                 direction      : str                                       = "maximize",
                 metric         : Union[str, Callable]                      = "r2",
                 output_dir     : str                                       = "./optuna_output",
                 save_study     : bool                                      = True,
                 patience       : Optional[int]                             = None,
                 pruner         : Optional[optuna.pruners.BasePruner]       = None,
                 timeout        : Optional[int]                             = None,
                 catch          : Union[tuple, Sequence[Exception]]         = (ValueError, RuntimeError),
                 callbacks      : Optional[List[Callable]]                  = None,
                 lambda_penalty : float                                     = 0.0
                ) -> SpaceSearchResult:
        """
        ____________________________________________________________________________________________________________________
        Unified hyperparameter optimization method supporting cross-validation modes for ML/DL models
        ____________________________________________________________________________________________________________________
        Parameters:
        ____________________________________________________________________________________________________________________
        -> Expected inpud for each Mode:
            -> Classical ML models:
                - partitions (List[Dict]) : List of partition dicts, each containing (or single dict) 
                                            {'X_train', 'y_train', 'X_val', 'y_val'}.
                                            Optional elements include 'features_names', 'weights', 'scaler'
            -> Deep Learning models:
                - partitions (List[Dict]) : List of partition dicts, each containing (or single dict) 
                                            {'train_loader', 'val_loader'}.
                                            Optional elements include 'features_names', 'weights', 'scaler'
        ____________________________________________________________________________________________________________________  
        Common Parameters:
        -> study_name (str)                              : Name for the study
        -> direction (str)                               : 'maximize' or 'minimize'
        -> metric (Union[str, Callable])                 : Metric to optimize
        -> output_dir (str)                              : Directory to save results
        -> save_study (bool)                             : Whether to save study and visualizations
        -> patience (Optional[int])                      : Early stopping patience
        -> pruner (Optional[optuna.pruners.BasePruner])  : Optuna pruner
        -> timeout (Optional[int])                       : Timeout in seconds
        -> catch (Union[tuple, Sequence[Exception]])     : Exceptions to catch
        -> callbacks (Optional[List[Callable]])          : Additional callbacks
        -> lambda_penalty (float)                        : Penalty for std in CV mode (default: 0.0)
        ____________________________________________________________________________________________________________________
        Returns:
        -> SpaceSearchResult: Results from the optimization study
        ____________________________________________________________________________________________________________________
        Notes:
        -> Mode is auto-detected: if partitions is provided, uses CV mode; otherwise uses single partition mode
        -> For single partition: provide X_train, y_train, X_val, y_val or train_loader, val_loader.
        -> For CV: provide partitions list
        -> For Huber loss: use metric="huber" with direction="minimize"
        -> Ensure the full DL model can be constructed with the given parameters when initializing SpaceSearch with 
           DLTabularRegressor. Else optimization won't work.
        ____________________________________________________________________________________________________________________
        """  
        # Validate that partitions are provided
        if partitions is None:
            raise ValueError("partitions must be provided (List[Dict] for CV or single Dict for single partition mode)")
        
        # Define as list if is_cv is False
        is_cv          = isinstance(partitions, list)
        partition_list = [partitions] if not is_cv else partitions
        
        # Validate and normalize partitions
        self.__validate_data(partition_list)
        partitions_normalized = self.__normalize_partitions(partition_list)
            
            
        # Check if pruner is mandatory based on the modeltype
        if (self.regressor == "dltab") and (pruner is None): 
            warnings.warn("No pruner provided for DL model. Using MedianPruner by default to prevent state leakage.")
            pruner = optuna.pruners.MedianPruner(n_startup_trials = 5,
                                                 n_warmup_steps   = self.config.dl_patience,
                                                 interval_steps   = 1)

        # Set up early stopping
        self._early_stopping['patience'] = patience
        
        # Create output directory
        output_path = Path(output_dir) / study_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metric
        scorer = self.__resolve_metric(metric)
        
        # Create or load study
        self.study = optuna.create_study(direction      = direction,
                                         study_name     = study_name,
                                         pruner         = pruner,
                                         sampler        = self.sampler,
                                         storage        = self.config.storage,
                                         load_if_exists = self.config.load_if_exists)
        
        # Create unified objective
        objective = self.__create_objective(partitions     = partitions_normalized,
                                            scorer         = scorer,
                                            direction      = direction,
                                            is_cv          = is_cv,
                                            lambda_penalty = lambda_penalty if is_cv else 0.0)
        # Set up callbacks
        study_callbacks = callbacks or []
        if patience:
            study_callbacks.append(self.__early_stopping_callback)
        
        # Optimize (n_jobs=1: sequential trials for reproducibility with TPESampler and GPU memory safety)
        self.study.optimize(objective, n_trials = self.config.n_trials, timeout = timeout, catch = catch,
                            callbacks = study_callbacks if study_callbacks else None,
                            n_jobs    = 1)
        
        # Store results
        self.best_params    = self.study.best_params
        self.best_score     = self.study.best_value
        self.trials_history = self.study.trials
        
        # Generate visualization and save study
        if save_study:
            if is_cv:
                self.__save_cv_results(output_path)
            
            # Create standard Optuna visualizations
            create_visualizations_per_study(self.study, output_path)
            self.__save_study(output_path)

        return SpaceSearchResult(best_params=self.best_params, best_score=self.best_score, 
                                 study      = self.study,
                                 n_trials   = len(self.study.trials),
                                 output_dir = str(output_path))
    
#---------------------------------------------------------------------------------------------------------------------------#