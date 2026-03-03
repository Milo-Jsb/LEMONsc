# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import joblib
import warnings

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib             import Path
from typing              import Optional, Dict, List, Union, Any
from lightgbm            import LGBMRegressor
from xgboost             import XGBRegressor
from cuml.ensemble       import RandomForestRegressor
from sklearn.exceptions  import NotFittedError

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory  import load_yaml_dict
from src.utils.eval       import compute_metrics
from src.utils.resources  import check_gpu_available

# Constants ---------------------------------------------------------------------------------------------------------------#
SUPPORTED_MODELS        = ["lightgbm", "xgboost", "rf"]
DEFAULT_METRICS         = ["mse", "rmse", "mae", "r2"]
DEFAULT_IMPORTANCE_TYPE = "gain"

# Custom Tree Regressor ---------------------------------------------------------------------------------------------------#
class MLTreeRegressor:
    """
    ________________________________________________________________________________________________________________________
    MLTreeRegressor: A comprehensive machine learning tree-based regressor wrapper for basic model comparison
    ________________________________________________________________________________________________________________________
    Models supported:
    -> Random Forest  (cuml.ensemble.RandomForestRegressor)
    -> LightGBM       (lightgbm.LGBMRegressor)
    -> XGBoost        (xgboost.XGBRegressor)
    ________________________________________________________________________________________________________________________
    """
    
    # Initialization of the Regressor--------------------------------------------------------------------------------------#
    def __init__(self, model_type: str = "lightgbm", model_params: Optional[Dict] = None, feat_names: Optional[List] = None, 
                n_jobs  : Optional[int] = None,
                device  : str           = "cpu", 
                verbose : bool          = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the MLTreeRegressor with specified model type and parameters.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model_type   (str)  : Mandatory. Type of model to use ('lightgbm', 'xgboost', 'rf').
        -> model_params (dict) : Optional. Dictionary containing model-specific hyperparameters.
        -> n_jobs       (int)  : Optional. Number of cores to use during computation.
        -> verbose      (bool) : Optional. Enable verbose logging for debugging purposes.
        -> feat_names   (List) : Optional. List of feature names.
        -> device       (str)  : Optional. 'cpu' (default) or 'cuda' for GPU acceleration.
                                Note: RandomForest (cuML) only supports 'cuda' and will override 'cpu' if specified.
        ____________________________________________________________________________________________________________________
        Raises:
            - ValueError, TypeError
        ____________________________________________________________________________________________________________________
        """
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")
        if model_params is not None and not isinstance(model_params, dict):
            raise TypeError("model_params must be a dictionary or None")
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        
        # Main parameters -------------------------------------------------------------------------------------------------#
        self.model_type    = model_type.lower()
        self.model_params  = model_params or {}
        self.n_jobs        = n_jobs
        self.verbose       = verbose
        self.feature_names = feat_names
        self.device        = device
        
        # Internal state variables ----------------------------------------------------------------------------------------#
        self.is_fitted                = False
        self.importance_type          = DEFAULT_IMPORTANCE_TYPE
        self.feature_importance_      = None
        self.feature_importance_cache = {}  # Cache for multiple importance types    
        
        # Validate model type ---------------------------------------------------------------------------------------------#
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {SUPPORTED_MODELS}")
        
        # Validate device for RandomForest (cuML RF only supports GPU)
        if self.model_type == "rf" and device == "cpu":
            # Check if GPU is actually available
            if not check_gpu_available():
                raise RuntimeError(
                    "cuML RandomForest requires GPU (CUDA), but no GPU detected. "
                    "Please ensure CUDA is properly installed and a GPU is available."
                )
            if verbose:
                warnings.warn("cuML RandomForest only supports GPU (cuda). Device will be set to 'cuda'.")
            self.device = "cuda"
        
        # Create the class element ----------------------------------------------------------------------------------------#
        if self.verbose: print(f"Initializing {self.model_type} model (device={self.device})...")
        
        try:
            self.model = self._init_model()
            if self.verbose: print(f"Successfully initialized {self.model_type} model")
        
        except (TypeError, KeyError, ImportError, AttributeError) as e:
            raise ValueError(f"Error initializing model: {e}") from e

    # Model selection -----------------------------------------------------------------------------------------------------#
    def _init_model(self):
        """
        ____________________________________________________________________________________________________________________
        Initialize the underlying model with appropriate default parameters.
        ____________________________________________________________________________________________________________________
        Returns:
            - model : Initialized model instance (RandomForestRegressor, LGBMRegressor, or XGBRegressor)
        ____________________________________________________________________________________________________________________
        """
        # Get the default model params path relative to this file
        default_model_params_path = Path(__file__).parent / "model_params"
        
        params = self.model_params.copy()
        
        # Original RandomForest Regressor (cuML library for single GPU training) ------------------------------------------#
        if self.model_type == "rf":
            
            # Set cuML dictionary of valid objectives (fixed for cuml-cu12==25.2.1) 
            cuml_objectives = {
                'squared_error'    : 2,
                'poisson'          : 4,
                'gamma'            : 5,
                'inverse_gaussian' : 6
                               }
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(default_model_params_path / "random_forest.yaml"))
            default_params.update(params)
                 
            # Remove device parameter if present (not supported by cuML RandomForest)
            clean_params = default_params.copy()
            clean_params.pop("device", None)
            clean_params.pop("n_jobs", None)

            # Changing objective to criterion if needed
            if "objective" in clean_params:
                if clean_params["objective"] not in cuml_objectives.keys():
                    if self.verbose:
                        warnings.warn(f"RandomForestRegressor doesn't support {clean_params['objective']}. "
                                     f"Overriding to 'squared_error'.")
                    
                    # Pop objective for correct name in cuML and set criterion to squared_error (default for regression)
                    clean_params.pop("objective", None)
                    clean_params["criterion"] = cuml_objectives["squared_error"]
                
                else:
                    # Pop objective for correct name in cuML and set criterion to corresponding value
                    objective = clean_params.pop("objective")
                    clean_params["criterion"] = cuml_objectives[objective]

            return RandomForestRegressor(**clean_params)
        
        # LightGBM --------------------------------------------------------------------------------------------------------#
        elif self.model_type == "lightgbm":
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(default_model_params_path / "lgbm.yaml"))
            default_params.update(params)
            
            # Set type of computation
            default_params["importance_type"] = DEFAULT_IMPORTANCE_TYPE
            default_params["n_jobs"]          = self.n_jobs
            default_params["device"]          = self.device  

            return LGBMRegressor(**default_params)
        
        # Extreme Gradient Boosting algorithm -----------------------------------------------------------------------------#
        elif self.model_type == "xgboost":

            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(default_model_params_path / "xgboost.yaml"))
            default_params.update(params)

            # Set type of computation
            default_params["importance_type"] = DEFAULT_IMPORTANCE_TYPE
            default_params["n_jobs"]          = self.n_jobs
            default_params["tree_method"]     = "hist"
            default_params["booster"]         = "gbtree"
            default_params["device"]          = self.device

            return XGBRegressor(**default_params)

    # Fit a given model using data points ---------------------------------------------------------------------------------#
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], 
            dict_params: Optional[dict]=None
            ) -> "MLTreeRegressor":
        """
        ____________________________________________________________________________________________________________________
        Fit the model to the training data with optional validation set for early stopping.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X_train     (np.ndarray/pd.DataFrame) : Mandatory. Training features.
        -> y_train     (np.ndarray/pd.Series)    : Mandatory. Training targets.
        -> dict_params (dict)                    : Optional. Additional parameters to set for each regressor.           
        ____________________________________________________________________________________________________________________
        Returns:
        -> self : Fitted model instance
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError, TypeError, NotFittedError
        ____________________________________________________________________________________________________________________
        """
        # Input validation ------------------------------------------------------------------------------------------------#
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")
        
        # Get number of features and store features names -----------------------------------------------------------------#
        n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        
        # Store or validate
        if hasattr(X_train, 'columns'):
            if self.feature_names is None:
                self.feature_names = list(X_train.columns)
            elif len(self.feature_names) != n_features:
                raise ValueError(f"Provided {len(self.feature_names)} feature names but X_train has {n_features} features")
        
        elif self.feature_names is not None and len(self.feature_names) != n_features:
            raise ValueError(f"Provided {len(self.feature_names)} feature names but X_train has {n_features} features")
        
        # Validate feature names are unique
        if self.feature_names and len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Feature names must be unique")
        
        if self.verbose: print(f"Fitting {self.model_type} model with {X_train.shape} samples...")

        # Fit the model and handle exceptions -----------------------------------------------------------------------------#
        try:
            # Prepare fit parameters
            fit_params = dict_params if dict_params is not None else {}
            
            # Fit the model and flag as fitted
            self.model.fit(X_train, y_train, **fit_params)
            self.is_fitted = True
            
            # Automatically compute and store feature importance after fitting
            try:
                self.feature_importance_ = self.get_feature_importance()
                if self.verbose: print(f"Feature importance computed and stored")
            
            except (AttributeError, ValueError) as e:
                if self.verbose: print(f"Warning: Could not compute feature importance: {e}")
                self.feature_importance_ = None
            
            if self.verbose: print(f"Successfully fitted {self.model_type} model")
            
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Error fitting model: {e}") from e
        
        # Return the fitted model instance for chaining -------------------------------------------------------------------#
        return self

    # Perfom inference with new data points -------------------------------------------------------------------------------# 
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________   
        Make predictions using the fitted model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X (np.ndarray/pd.DataFrame): Mandatory. Features to predict on.
        ____________________________________________________________________________________________________________________
        Returns:
        -> preds (np.ndarray) : Predicted values
        ____________________________________________________________________________________________________________________
        Raises:
        ->  NotFittedError, ValueError
        ____________________________________________________________________________________________________________________
        """
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before making predictions")
        
        if X is None:
            raise ValueError("X cannot be None")
        
        try:           
            preds = self.model.predict(X)
            
            return preds
        
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Error making predictions: {e}") from e

    # Return feature importance for evaluation of the training ------------------------------------------------------------#
    def get_feature_importance(self, importance_type: Optional[str] = None, recompute: bool = False) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Get feature importance scores from the fitted model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> importance_type (str)  : Optional. Type of importance names (check individual model documentation).
                                    If None, uses self.importance_type. Only applicable for LightGBM/XGBoost.
        -> recompute (bool)       : Optional. If True, recompute importance even if cached. Default False.
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary mapping feature names to importance scores
        ____________________________________________________________________________________________________________________
        Raises:
        -> NotFittedError, AttributeError, ValueError
        ____________________________________________________________________________________________________________________
        Notes:
        -> Default importance type is "gain" for fair comparison across models.
        -> RandomForest (cuML) does not support different importance types.
        -> Results are cached in self.feature_importance_ after first computation.
        -> If feature names were not provided, generic names (f0, f1, ...) will be generated.
        ____________________________________________________________________________________________________________________
        """
        # Input and model validation
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before getting feature importance")
        
        # Determine which importance type to use
        imp_type = importance_type or self.importance_type or DEFAULT_IMPORTANCE_TYPE
        
        # Return cached version if available and not recomputing
        if not recompute:
            if importance_type is None and self.feature_importance_ is not None:
                return self.feature_importance_
            elif imp_type in self.feature_importance_cache:
                return self.feature_importance_cache[imp_type]
        
        # Warning for RandomForest with custom importance_type
        if self.model_type == "rf" and importance_type is not None:
            if self.verbose:
                print(f"Warning: RandomForest does not support custom importance_type. Ignoring '{importance_type}'.")
            importance_type = None
        
        # Set importance type
        if importance_type is not None:
            self.importance_type = importance_type
        
        # Retrieve the feature importance of a trained model
        try:
            # For LightGBM
            if self.model_type =="lightgbm":
                booster     = self.model.booster_
                imp_type    = importance_type or self.importance_type or "gain"
                importances = booster.feature_importance(importance_type=imp_type)
            
            # For XGBoost
            elif self.model_type == "xgboost":
                booster     = self.model.get_booster()
                imp_type    = importance_type or self.importance_type or "gain"
                score       = booster.get_score(importance_type=imp_type)
                n_features  = len(self.feature_names) if self.feature_names else booster.num_features()
                importances = np.zeros(n_features)

                for k, v in score.items():
                    # k is f0, f1, ...
                    idx = int(k[1:])
                    if idx < n_features:
                        importances[idx] = v
            
            # RandomForest (cuML) - only supports default importance (similar to Gini importance)
            elif self.model_type == "rf":
                importances = self.model.feature_importances_
            
            else:
                raise ValueError(f"Unsupported model type for feature importance: {self.model_type}")   
            
            # Ensure we have feature names (generate generic ones if needed)
            if self.feature_names is None or len(self.feature_names) == 0:
                n_features = len(importances)
                self.feature_names = [f"f{i}" for i in range(n_features)]
                if self.verbose:
                    print(f"Warning: No names provided. Using generic names.")
            
            # Validate dimensions match
            if len(self.feature_names) != len(importances):
                raise ValueError(f"Mismatch: {len(self.feature_names)} names but {len(importances)} importance scores")
                
            # Create dictionary mapping feature names to importance scores and sort by importance (descending)
            importance_dict = dict(sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True))

            # Cache the result
            self.feature_importance_cache[imp_type] = importance_dict
            
            # Also set as default if this is the default importance type
            if importance_type is None:
                self.feature_importance_ = importance_dict

            return importance_dict
            
        except (AttributeError, KeyError, IndexError) as e:
            raise ValueError(f"Error getting feature importance: {e}") from e

    # Metric evaluation of a given model ----------------------------------------------------------------------------------#
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
                 metrics   : Optional[List[str]] = None
                 ) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Evaluate the model performance using multiple metrics.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X       (np.ndarray/pd.DataFrame) : Mandatory. Test features.
        -> y       (np.ndarray/pd.Series)    : Mandatory. Test targets.
        -> metrics (list)                    : Optional. List of metrics to compute.
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary containing evaluation metrics
        ____________________________________________________________________________________________________________________
        Raises:
        -> NotFittedError, ValueError
        ____________________________________________________________________________________________________________________
        """
        # Model and input validation
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before evaluation")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        if metrics is None:
            metrics = DEFAULT_METRICS
        
        try:
            y_pred  = self.model.predict(X)
            results = compute_metrics(y, y_pred, metrics, verbose=self.verbose)

            return results
            
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error evaluating model: {e}") from e

    # Save model weights and hyperparameters into output directory --------------------------------------------------------#
    def save_model(self, path: str) -> None:
        """
        ____________________________________________________________________________________________________________________
        Save the model to disk using joblib.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path (str) : Mandatory. Path where to save the model.
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError, OSError, NotFittedError
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before saving")
        
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            save_data = {
                "model_type"               : self.model_type,
                "model_params"             : self.model_params,
                "model"                    : self.model,
                "is_fitted"                : self.is_fitted,
                "feature_names"            : self.feature_names,
                "importance_type"          : self.importance_type,
                "feature_importance_"      : self.feature_importance_,
                "feature_importance_cache" : self.feature_importance_cache
            }
            
            joblib.dump(save_data, path)
            
            if self.verbose: print(f"Model saved successfully to: {path}")
            
        except (OSError, IOError, TypeError) as e:
            raise ValueError(f"Error saving model: {e}") from e

    # Load model weights and attributes using classmethod decorator -------------------------------------------------------#
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None, verbose: bool = False) -> "MLTreeRegressor":
        """
        ____________________________________________________________________________________________________________________
        Load a saved model from disk using joblib.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path    (str)  : Mandatory. Path to the saved model file.
        -> device  (str)  : Optional. Device to use ('cpu' or 'cuda'). If None, uses saved device.
        -> verbose (bool) : Optional. Enable verbose logging.
        ____________________________________________________________________________________________________________________
        Returns:
        -> MLTreeRegressor : Loaded model instance
        ____________________________________________________________________________________________________________________
        Raises:
        -> FileNotFoundError, ValueError, OSError
        ____________________________________________________________________________________________________________________
        """
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if device is not None and device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu', 'cuda', or None")
        
        # Load weights and information using binary files -----------------------------------------------------------------#
        try:
            # Load information stored at the path folder
            data = joblib.load(path)
            
            # Create instance (model will be overwritten with saved one, so device here is temporary)
            instance = cls(model_type=data["model_type"],  model_params=data["model_params"], 
                           verbose = verbose, 
                           device  = device if device is not None else "cpu")
            
            # Restore model state
            instance.model                     = data["model"]
            instance.is_fitted                 = data.get("is_fitted", False)
            instance.feature_names             = data.get("feature_names", None)
            instance.importance_type           = data.get("importance_type", DEFAULT_IMPORTANCE_TYPE)
            instance.feature_importance_       = data.get("feature_importance_", None)
            instance.feature_importance_cache  = data.get("feature_importance_cache", {})
            
            # Set device: use provided device or fall back to saved device
            if device is not None:
                instance.device = device
                # Update model device settings if applicable (not for RF as it only supports GPU)
                if instance.model_type != "rf" and hasattr(instance.model, 'set_params'):
                    try:
                        instance.model.set_params(device=device)
                        if verbose: print(f"Model device updated to: {device}")
                    except Exception as e:
                        if verbose: print(f"Warning: Could not update device parameter: {e}")
                elif instance.model_type == "rf":
                    if verbose: print(f"RandomForest model uses GPU by default (cuML)")
            
            if verbose: print(f"Model loaded successfully from: {path}")
            
            return instance
            
        except (OSError, IOError, KeyError, EOFError) as e:
            raise ValueError(f"Error loading model: {e}") from e

    # Retrieve model information ------------------------------------------------------------------------------------------#
    def get_model_info(self) -> Dict[str, Any]:
        """
        ____________________________________________________________________________________________________________________
        Get comprehensive information about the model.
        ____________________________________________________________________________________________________________________
        Returns:
            - dict : Dictionary containing model information
        ____________________________________________________________________________________________________________________
        """
        info = {
            "model_type"             : self.model_type,
            "is_fitted"              : self.is_fitted,
            "model_params"           : self.model_params,
            "feature_names"          : self.feature_names,
            "n_features"             : len(self.feature_names) if self.feature_names else None,
            "importance_type"        : self.importance_type,
            "has_feature_importance" : self.feature_importance_ is not None,
            "device"                 : self.device,
        }
        
        if self.is_fitted and hasattr(self.model, "n_estimators"):
            info["n_estimators"] = self.model.n_estimators
        
        return info
    
    # String representation for debugging ---------------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """
        ____________________________________________________________________________________________________________________
        Return a string representation of the MLTreeRegressor instance.
        ____________________________________________________________________________________________________________________
        Returns:
            - str : String representation
        ____________________________________________________________________________________________________________________
        """
        fitted_str     = "fitted" if self.is_fitted else "not fitted"
        n_features_str = f", n_features={len(self.feature_names)}" if self.feature_names else ""
        
        return f"MLTreeRegressor(model_type='{self.model_type}', {fitted_str}, device='{self.device}'{n_features_str})"

#--------------------------------------------------------------------------------------------------------------------------#