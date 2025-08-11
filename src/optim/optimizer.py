# Modules -----------------------------------------------------------------------------------------------------------------#
import yaml
import optuna
import joblib
import warnings

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from dataclasses             import dataclass
from pathlib                 import Path
from typing                  import Dict, List, Optional, Union, Callable, Any, Sequence
from sklearn.model_selection import KFold
from sklearn.metrics         import get_scorer

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.mltrees .regressor import MLTreeRegressor
from src.optim.grid                import RandomForestGrid, LightGBMGrid, XGBoostGrid, DARTBoostGrid

# Object tipe to return ---------------------------------------------------------------------------------------------------#
@dataclass
class SpaceSearchResult:
    best_params     : Dict[str, Any]
    best_score      : float
    study           : optuna.study.Study
    n_trials        : int
    output_dir      : str

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
    
    # Initialize the object -----------------------------------------------------------------------------------------------#
    def __init__(self, model_type: str, n_jobs: int = 10, n_trials: int = 100, device:str="cpu", verbose: bool = True,
                seed           : int = 42,
                sampler        : Optional[optuna.samplers.BaseSampler] = None,
                storage        : Optional[str] = None,
                load_if_exists : bool = False
                ):
        """
        ___________________________________________________________________________________________________________________
        Initialize the SpaceSearch optimizer.
        ____________________________________________________________________________________________________________________
        Parameters:
            - model_type     (str)                         : Initialized model to optimize.
            - n_trials       (int)                         : Number of optimization trials
            - verbose        (bool)                        : Enable verbose output
            - seed           (int)                         : Global random seed for reproducibility.
            - sampler        (optuna.samplers.BaseSampler) : Optional. Custom sampler. Defaults to TPESampler(seed=seed).
            - storage        (str)                         : Optional. Optuna storage URL (e.g., "sqlite:///optuna.db") 
                                                             to persist and resume studies.
            - load_if_exists (bool)                        : If True and a study with the same name exists in storage, 
                                                             it will be loaded.
        ____________________________________________________________________________________________________________________
        """
        self.model_type     = model_type
        self.device         = device
        self.n_jobs         = n_jobs
        self.n_trials       = n_trials
        self.verbose        = verbose
        self.seed           = seed
        self.storage        = storage
        self.load_if_exists = load_if_exists
        self.sampler        = sampler or optuna.samplers.TPESampler(seed=seed, multivariate=True)

        # Internals
        self.study             : Optional[optuna.study.Study] = None
        self.best_params       : Optional[Dict[str, Any]] = None
        self.best_score        : Optional[float] = None
        self.trials_history    : List[optuna.trial.FrozenTrial] = []
        self._patience         : Optional[int] = None
        self._no_improve_count : int = 0
        self._best_value_seen  : Optional[float] = None
        self._direction        : Union[str, List[str]] = "maximize"  
        self.best_model_       : Optional[MLTreeRegressor] = None

        if self.verbose:
            print(f"SpaceSearch initialized for {self.model_type} model (device={self.device})")
    
    # Run the space search given data -------------------------------------------------------------------------------------#
    def run_study(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], 
                        X_val: Union[np.ndarray, pd.DataFrame], y_val: Union[np.ndarray, pd.Series],
                    study_name      : str = "optuna_study",
                    direction       : str = "maximize",
                    metric          : Union[str, Callable[[np.ndarray, np.ndarray], float]] = "r2",
                    output_dir      : str = "./optuna_output",
                    save_study      : bool = True,
                    patience        : Optional[int] = None,
                    pruner          : Optional[optuna.pruners.BasePruner] = None,
                    timeout         : Optional[int] = None,
                    catch           : Union[tuple, Sequence[Exception]] = (Exception,),
                    callbacks       : Optional[List[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]]] = None,
                    cv              : Optional[int] = None,
                    scaler          : Optional[pd.Series]=None,
                    ) -> Dict[str, Any]:

        """
        ____________________________________________________________________________________________________________________
        Run the hyperparameter optimization study.
        ____________________________________________________________________________________________________________________
        Parameters:
            - X_train, y_train: Training data
            - X_val  , y_val: Validation data
            - study_name (str): Name for the Optuna study
            - direction (str): Optimization direction ('maximize' or 'minimize')
            - output_dir (str): Directory to save results
            - save_study (bool): Whether to save the study
            - patience (int): Early stopping patience
            - pruner: Optuna pruner for early stopping
        ____________________________________________________________________________________________________________________
        Returns:
            - Dict containing study results and best parameters
        ____________________________________________________________________________________________________________________
        """

        self._direction = direction
        self._patience  = patience

        if self.verbose:
            print(f"Starting study: {study_name} | Trials: {self.n_trials} | Direction: {direction}")

        # Create output directory
        output_path = Path(f"{output_dir}{study_name}/")
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare metric
        scorer = self.__resolve_metric(metric)
        
        # Create or load study
        study = optuna.create_study(direction=direction, study_name=study_name, pruner=pruner,
                                    sampler        = self.sampler,
                                    storage        = self.storage,
                                    load_if_exists = self.load_if_exists)
        # Build objective
        objective_func = self.__build_objective(scorer=scorer, X_train=X_train, y_train=y_train, X_val=X_val,
                                                y_val     = y_val,
                                                cv        = cv,
                                                direction = direction,
                                                scaler    = scaler)
        # Callbacks
        cb = callbacks or []
        if patience: cb.append(self.__early_stopping_callback)

        # Optimize
        study.optimize(objective_func, n_trials=self.n_trials, timeout=timeout, catch=catch, 
                        callbacks = cb if cb else None,
                        n_jobs    = 1)
        # Store results
        self.study          = study
        self.best_params    = study.best_params
        self.best_score     = study.best_value
        self.trials_history = study.trials
        
        # Save results
        if save_study:
            self.__save_results(output_path)
            self.__save_study_object(output_path, study_name)
        
        # Generate visualizations
        self.__visualize_search_space(output_path)
        
        if self.verbose:
            print(110*"_")
            print(f"Study completed!")
            print(110*"_")
            print(f"Best Score: {self.best_score:.5f}")
            print(f"Best parameters: {self.best_params}")
            print(110*"_")

        return SpaceSearchResult(
            best_params=self.best_params,
            best_score=self.best_score,
            study=study,
            n_trials=len(study.trials),
            output_dir=str(output_path))
    
    # Display summary -----------------------------------------------------------------------------------------------------#
    def get_trial_summary(self) -> pd.DataFrame:
        if self.study is None:
            raise ValueError("No study available. Run run_study() first.")
        rows = []
        for t in self.study.trials:
            if t.value is not None:
                r = t.params.copy()
                r.update(
                    {
                        "score": t.value,
                        "trial_number": t.number,
                        "state": t.state.name,
                        "duration_s": t.duration.total_seconds(),
                    }
                )
                rows.append(r)
        return pd.DataFrame(rows)

    # Load a saved study --------------------------------------------------------------------------------------------------$
    def load_study(self, study_path: str) -> bool:
        
        try:
            self.study          = joblib.load(study_path)
            self.best_params    = self.study.best_params
            self.best_score     = self.study.best_value
            self.trials_history = self.study.trials
            
            if self.verbose:
                print(f"Study loaded successfully from: {study_path}")
                print(f"Best score: {self.best_score:.5f}")
            
            return True

        except Exception as e:
            if self.verbose:
                print(f"Failed to load study: {e}")
            return False

    # Retrieve the desired metric -----------------------------------------------------------------------------------------#
    def __resolve_metric(self, metric):
        if isinstance(metric, str):
            scorer = get_scorer(metric)
            def _sc(y_true, y_pred):
                return scorer._score_func(y_true, y_pred, **scorer._kwargs)
            return _sc
        elif callable(metric):
            return metric
        else:
            raise ValueError("metric must be a string scorer name or a callable")

    # Built the objecyive to optimize -------------------------------------------------------------------------------------#
    def __build_objective(self, scorer: Callable, X_train, y_train, X_val, y_val, 
                                cv        : Optional[int], 
                                direction : str,
                                scaler    : Optional[pd.Series],
                                ) -> Callable[[optuna.trial.Trial], float]:

        maximize = direction == "maximize"

        def objective(trial: optuna.trial.Trial) -> float:
            
            # Define search space based on model type ---------------------------------------------------------------------#
            if   (self.model_type == "random_forest"): param_grid = RandomForestGrid(trial)
            elif (self.model_type == "lightgbm"):      param_grid = LightGBMGrid(trial)
            elif (self.model_type == "xgboost"):       param_grid = XGBoostGrid(trial)
            elif (self.model_type == "dartboost"):     param_grid = DARTBoostGrid(trial)
            
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            # Train model -------------------------------------------------------------------------------------------------#
            try:
                
                model = MLTreeRegressor(model_type   = self.model_type, 
                                        model_params = param_grid, 
                                        device       = self.device,
                                        n_jobs       = self.n_jobs)

                # Cross-Validation study ----------------------------------------------------------------------------------#
                if cv:

                    # Define the Kfold separation
                    kf     = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
                    scores = []
                    
                    for tr_idx, va_idx in kf.split(X_train):
                        
                        model.fit(X_train[tr_idx], y_train[tr_idx])
                        
                        y_pred = model.predict(X_train[va_idx])
                        
                        # Apply scaling if provided, otherwise use direct values
                        if scaler is not None:
                            y_pred_scaled = y_pred * scaler
                            y_true_scaled = y_train[va_idx] * scaler
                            score_val = scorer(y_true_scaled, y_pred_scaled)
                        else:
                            score_val = scorer(y_train[va_idx], y_pred)
                        
                        scores.append(score_val)
                        trial.report(np.mean(scores), step=len(scores))
                        
                        # Optional pruner
                        if trial.should_prune(): raise optuna.TrialPruned()
                    
                    score = float(np.mean(scores))
                
                # Full data study -----------------------------------------------------------------------------------------#
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Apply scaling if provided, otherwise use direct values
                    if scaler is not None:
                        y_pred_scaled = y_pred * scaler
                        y_val_scaled  = y_val * scaler
                        score = float(scorer(y_val_scaled, y_pred_scaled))
                    else:
                        score = float(scorer(y_val, y_pred))

                return score                        
            
            except optuna.TrialPruned:
                raise
            
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Trial failed: {e}")
                # Penalize, but finite
                return -1e15 if maximize else 1e15

        return objective
    
    # Early stopping for optimization trials ------------------------------------------------------------------------------#
    def __early_stopping_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        # Simple patience on best_value improvement
        current = study.best_value
        if self._best_value_seen is None or (
            (self._direction == "maximize" and current > self._best_value_seen) or
            (self._direction == "minimize" and current < self._best_value_seen)):

            self._best_value_seen  = current
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._patience is not None and self._no_improve_count >= self._patience:
            if self.verbose:
                print(f"Early stopping: no improvement in {self._patience} trials.")
            study.stop()
    
    # Store results of the study-------------------------------------------------------------------------------------------#
    def __save_results(self, output_path: Path):
        with open(output_path / "best_params.yaml", "w") as f:
            yaml.dump(self.best_params, f, default_flow_style=False)

        with open(output_path  / "best_score.txt", "w") as f:
            f.write(f"Best Score       : {self.best_score:.6f}\n")
            f.write(f"Model Type       : {self.model_type}\n")
            f.write(f"Number of Trials : {len(self.study.trials)}\n")
    
    # Save study object as checkpoint if needed ---------------------------------------------------------------------------#
    def __save_study_object(self, output_path: Path, study_name: str):
        joblib.dump(self.study, output_path /f"{study_name}.joblib")
    
    def __visualize_search_space(self, output_path: Path):
        # Optional visualizations if plotly/matplotlib backends are available
        try:
            from optuna.visualization.matplotlib import (
                plot_param_importances,
                plot_optimization_history,
            )
            figs = {
                "optimization_history": plot_optimization_history(self.study),
                "param_importances": plot_param_importances(self.study),
            }
            for name, fig in figs.items():
                fig.figure.savefig(output_path/ f"{name}.png", bbox_inches="tight", dpi=500)
                plt.close(fig.figure)
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Visualization failed: {e}")

#--------------------------------------------------------------------------------------------------------------------------#


