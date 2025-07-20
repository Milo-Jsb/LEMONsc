# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import joblib

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib                 import Path
from typing                  import Optional, Dict, List, Tuple, Union, Any
from lightgbm                import LGBMRegressor
from xgboost                 import XGBRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions      import NotFittedError

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory import load_yaml_dict

# Custom Tree Regressor ---------------------------------------------------------------------------------------------------#
class MLTreeRegressor:
    """
    ________________________________________________________________________________________________________________________
    MLTreeRegressor: A comprehensive machine learning tree-based regressor wrapper supporting: 
    
        - LightGBM       (lightgbm.LGBMRegressor in gtb mode)
        - XGBoost        (xgboost.XGBRegressor)
        - Random Forest  (lightgbm.LGBMRegressor in rf mode)
    ________________________________________________________________________________________________________________________
    """
    
    # Initialization of the Regressor--------------------------------------------------------------------------------------#
    def __init__(self, model_type: str = "lightgbm", model_params: Optional[Dict] = None, 
                feat_names: Optional[list] = None, device: str = "cpu", verbose: bool = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the MLTreeRegressor with specified model type and parameters.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model_type   (str)  : Mandatory. Type of model to use ('lightgbm', 'xgboost', 'random_forest').
        -> model_params (dict) : Optional. Dictionary containing model-specific hyperparameters.
        -> feat_names   (list) : Optional. List of feature names.
        -> device       (str)  : Optional. 'cpu' (default) or 'gpu' for CUDA acceleration.
        -> verbose      (bool) : Optional. Enable verbose logging for debugging purposes.
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError, TypeError
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")
        if model_params is not None and not isinstance(model_params, dict):
            raise TypeError("model_params must be a dictionary or None")
        if device not in ["cpu", "gpu"]:
            raise ValueError("device must be either 'cpu' or 'gpu'")
        self.model_type       = model_type.lower()
        self.model_params     = model_params or {}
        self.verbose          = verbose
        self.is_fitted        = False
        self.feature_names    = feat_names
        self.device           = device
        # Validate model type 
        supported_models = ["lightgbm", "xgboost", "random_forest"]
        if self.model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {supported_models}")
        # Create the class element
        if self.verbose: print(f"Initializing {self.model_type} model (device={self.device})...")
        try:
            self.model = self._init_model()
            if self.verbose: print(f"Successfully initialized {self.model_type} model")
        except Exception as e:
            raise ValueError(f"Error initializing model: {e}")

    # Model selection -----------------------------------------------------------------------------------------------------#
    def _init_model(self):
        """
        ____________________________________________________________________________________________________________________
        Initialize the underlying model with appropriate default parameters.
        ____________________________________________________________________________________________________________________
        Returns:
        -> model : Initialized model instance (LGBMRegressor or XGBRegressor)
        ____________________________________________________________________________________________________________________
        """
        params = self.model_params.copy()
        if self.model_type == "random_forest":
            default_params = load_yaml_dict(path= "./src/models/mltrees/model_params/rf.yaml")
            default_params.update(params)
            if self.device == "gpu":
                default_params["device"] = "gpu"
            else:
                default_params["device"] = "cpu"
            return LGBMRegressor(**default_params)
        elif self.model_type == "xgboost":
            default_params = load_yaml_dict(path= "./src/models/mltrees/model_params/xgb.yaml")
            default_params.update(params)
            if self.device == "gpu":
                default_params["tree_method"] = "gpu_hist"
                default_params["predictor"] = "gpu_predictor"
            else:
                default_params["tree_method"] = "hist"
                default_params["predictor"] = "auto"
            return XGBRegressor(**default_params)
        elif self.model_type == "lightgbm":
            default_params = load_yaml_dict(path= "./src/models/mltrees/model_params/lgbm.yaml")
            default_params.update(params)
            if self.device == "gpu":
                default_params["device"] = "gpu"
            else:
                default_params["device"] = "cpu"
            return LGBMRegressor(**default_params)

    # Fit a given model using data points ---------------------------------------------------------------------------------#
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], 
                X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                y_val: Optional[Union[np.ndarray, pd.Series]]    = None,
                early_stopping_rounds: Optional[int] = 50) -> "MLTreeRegressor":
        """
        ____________________________________________________________________________________________________________________
        Fit the model to the training data with optional validation set for early stopping.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X_train               (np.ndarray/pd.DataFrame) : Mandatory. Training features.
        -> y_train               (np.ndarray/pd.Series)    : Mandatory. Training targets.
        -> X_val                 (np.ndarray/pd.DataFrame) : Optional. Validation features for early stopping.
        -> y_val                 (np.ndarray/pd.Series)    : Optional. Validation targets for early stopping.
        -> early_stopping_rounds (int)                     : Optional. Number of rounds for early stopping.
        ____________________________________________________________________________________________________________________
        Returns:
        -> self : Fitted model instance
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError, TypeError, NotFittedError
        ________________________________________________________________________________________________________________________
        """
        # Input validation
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")
        
        if X_val is not None and y_val is not None and len(X_val) != len(y_val):
            raise ValueError("X_val and y_val must have the same number of samples")
        
        # Store feature names if available
        if hasattr(X_train, 'columns') and (self.feature_names is None):
            self.feature_names = list(X_train.columns)
        
        if self.verbose: print(f"Fitting {self.model_type} model with {len(X_train)} samples...")
        
        try:
            # Prepare fit parameters
            fit_params = {}
            
            # Add evaluation set for early stopping
            if X_val is not None and y_val is not None:
                fit_params['eval_set'] = [(X_val, y_val)]
                if early_stopping_rounds is not None and self.model_type in ["xgboost", "lightgbm"]:
                    fit_params['early_stopping_rounds'] = early_stopping_rounds
                    fit_params['verbose']               = False
            
            # Fit the model
            self.model.fit(X_train, y_train, **fit_params)
            self.is_fitted = True
            
            if self.verbose: print(f"Successfully fitted {self.model_type} model")
            
        except Exception as e:
            raise ValueError(f"Error fitting model: {e}")
        
        return self

    # Perfom inference with new data points -------------------------------------------------------------------------------# 
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________   
        Make predictions using the fitted model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X (np.ndarray/pd.DataFrame) : Mandatory. Features to predict on.
        ____________________________________________________________________________________________________________________
        Returns:
        -> np.ndarray : Predicted values
        ____________________________________________________________________________________________________________________
        Raises:
        -> NotFittedError, ValueError
        ____________________________________________________________________________________________________________________
        """
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before making predictions")
        
        if X is None:
            raise ValueError("X cannot be None")
        
        try:
            return self.model.predict(X)
        except Exception as e:
            raise ValueError(f"Error making predictions: {e}")

    # Return feature importance for evaluation of the training ------------------------------------------------------------#
    def get_feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Get feature importance scores from the fitted model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> importance_type (str) : Optional. Type of importance ('gain', 'split', 'cover' for LightGBM).
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary mapping feature names to importance scores
        ____________________________________________________________________________________________________________________
        Raises:
        -> NotFittedError, AttributeError, ValueError
        ____________________________________________________________________________________________________________________
        """
        # Input and model validation
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before getting feature importance")
        
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("This model does not support feature importance")
        
        # Retrieve the feature importance of a trained model
        try:
            importances = self.model.feature_importances_
            
            # Create feature names if not available
            if self.feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                feature_names = self.feature_names
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            raise ValueError(f"Error getting feature importance: {e}")

    # Metric evaluation of a given model ----------------------------------------------------------------------------------#
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
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
            metrics = ["mse", "rmse", "mae", "r2"]
        
        try:
            # Get model predictions
            y_pred  = self.predict(X)
            results = {}
            
            # Compute metrics and store in dictionary
            for metric in metrics:
                if   (metric.lower() == "mse")  : results["mse"]  = mean_squared_error(y, y_pred)
                elif (metric.lower() == "rmse") : results["rmse"] = mean_squared_error(y, y_pred, squared=False)
                elif (metric.lower() == "mae")  : results["mae"]  = mean_absolute_error(y, y_pred)
                elif (metric.lower() == "r2")   : results["r2"]   = r2_score(y, y_pred)
                
                else:
                    if self.verbose: print(f"Warning: Unknown metric '{metric}' ignored")
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error evaluating model: {e}")

    # Save model weights and hyperparameters into output directory --------------------------------------------------------#
    def save_model(self, path: str, include_data: bool = False) -> None:
        """
        ____________________________________________________________________________________________________________________
        Save the model to disk using joblib.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path         (str)  : Mandatory. Path where to save the model.
        -> include_data (bool) : Optional. Whether to include training data in the saved model.
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError, OSError
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
                "model_type"       : self.model_type,
                "model_params"     : self.model_params,
                "model"            : self.model,
                "is_fitted"        : self.is_fitted,
                "feature_names"    : self.feature_names,
                "training_history" : self.training_history
                        }
            
            joblib.dump(save_data, path)
            
            if self.verbose: print(f"Model saved successfully to: {path}")
            
        except Exception as e:
            raise ValueError(f"Error saving model: {e}")

    # Load model weights and attributes using classmethod decorator -------------------------------------------------------#
    @classmethod
    def load_model(cls, path: str, verbose: bool = False) -> "MLTreeRegressor":
        """
        ____________________________________________________________________________________________________________________
        Load a saved model from disk using joblib.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> path    (str)  : Mandatory. Path to the saved model file.
        -> verbose (bool) : Optional. Enable verbose logging.
        ____________________________________________________________________________________________________________________
        Returns:
        -> MLTreeRegressor : Loaded model instance
        ____________________________________________________________________________________________________________________
        Raises:
        -> FileNotFoundError, ValueError, OSError
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # Load informaion stored at the path folder
            data = joblib.load(path)
            
            # Create instance
            instance = cls(model_type=data["model_type"],  model_params=data["model_params"], verbose=verbose)
            
            # Restore model state
            instance.model            = data["model"]
            instance.is_fitted        = data.get("is_fitted", False)
            instance.feature_names    = data.get("feature_names", None)
            instance.training_history = data.get("training_history", {})
            
            if verbose: print(f"Model loaded successfully from: {path}")
            
            return instance
            
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    # Retrive model information -------------------------------------------------------------------------------------------#
    def get_model_info(self) -> Dict[str, Any]:
        """
        ____________________________________________________________________________________________________________________
        Get comprehensive information about the model.
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary containing model information
        ____________________________________________________________________________________________________________________
        """
        info = {
            "model_type"      : self.model_type,
            "is_fitted"       : self.is_fitted,
            "model_params"    : self.model_params,
            "feature_names"   : self.feature_names,
                }
        
        if self.is_fitted and hasattr(self.model, "n_estimators"):
            info["n_estimators"] = self.model.n_estimators
        
        return info

#---------------------------------------------------------------------------------------------------------------------------