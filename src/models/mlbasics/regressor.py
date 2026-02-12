# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import joblib
import warnings

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from pathlib                 import Path
from typing                  import Optional, Dict, List, Union, Any
from sklearn.linear_model    import ElasticNet
from cuml.svm                import SVR as cuSVR
from sklearn.metrics         import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.exceptions      import NotFittedError

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory import load_yaml_dict
from src.utils.callbacks  import check_gpu_available

# Constants ---------------------------------------------------------------------------------------------------------------#
SUPPORTED_MODELS = ["elasticnet", "svr"]
DEFAULT_METRICS = ["mse", "rmse", "mae", "r2"]
DEFAULT_IMPORTANCE_TYPE = "coefficients"

# Custom Basic ML Regressor -----------------------------------------------------------------------------------------------#
class MLBasicRegressor:
    """
    ________________________________________________________________________________________________________________________
    MLBasicRegressor: A comprehensive machine learning basic regressor wrapper for model comparison
    ________________________________________________________________________________________________________________________
    Models supported:
        - ElasticNet : sklearn.linear_model.ElasticNet (CPU only)
        - SVR        : cuml.svm.SVR (GPU CUDA only)
    ________________________________________________________________________________________________________________________
    """
    
    # Initialization of the Regressor--------------------------------------------------------------------------------------#
    def __init__(self, model_type: str = "elasticnet", model_params: Optional[Dict] = None, 
                 feat_names: Optional[List] = None, 
                 n_jobs    : Optional[int] = None,
                 device    : str           = "cpu", 
                 verbose   : bool          = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the MLBasicRegressor with specified model type and parameters.
        ____________________________________________________________________________________________________________________
        Parameters:
            - model_type   (str)  : Mandatory. Type of model to use ('elasticnet', 'svr').
            - model_params (dict) : Optional. Dictionary containing model-specific hyperparameters.
            - feat_names   (List) : Optional. List of feature names.
            - n_jobs       (int)  : Optional. Number of cores to use during computation (ElasticNet only).
            - device       (str)  : Optional. 'cpu' (default) or 'cuda' for GPU acceleration.
                                    Note: SVR (cuML) only supports 'cuda' and will override 'cpu' if specified.
                                    ElasticNet only supports 'cpu' and will override 'cuda' if specified.
            - verbose      (bool) : Optional. Enable verbose logging for debugging purposes.
        
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
        self.model_type          = model_type.lower()
        self.model_params        = model_params or {}
        self.n_jobs              = n_jobs
        self.verbose             = verbose
        self.feature_names       = feat_names
        self.device              = device
        
        # Internal attributes ---------------------------------------------------------------------------------------------#
        self.is_fitted           = False
        self.importance_type     = DEFAULT_IMPORTANCE_TYPE
        self.feature_importance_ = None            
        
        # Validate model type ---------------------------------------------------------------------------------------------#
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {SUPPORTED_MODELS}")
        
        # Warning for GPU usage with ElasticNet ---------------------------------------------------------------------------#
        if self.model_type == "elasticnet" and self.device == "cuda":
            if self.verbose:
                warnings.warn("ElasticNet does not benefit from GPU acceleration. Using CPU implementation.")
            self.device = "cpu"
        
        # Validate device for SVR (GPU only) ------------------------------------------------------------------------------#
        if self.model_type == "svr" and self.device == "cpu":
            # Check if GPU is actually available
            if not check_gpu_available():
                raise RuntimeError(
                    "cuML SVR requires GPU (CUDA), but no GPU detected. "
                    "Please ensure CUDA is properly installed and a GPU is available."
                )
            if self.verbose:
                warnings.warn("cuML SVR only supports GPU (cuda). Device will be set to 'cuda'.")
            self.device = "cuda"
        
        # Create the class element ----------------------------------------------------------------------------------------#
        if self.verbose: 
            print(f"Initializing {self.model_type} model (device={self.device})...")
        
        try:
            self.model = self._init_model()
            if self.verbose: 
                print(f"Successfully initialized {self.model_type} model")
        
        except (TypeError, KeyError, ImportError, AttributeError) as e:
            raise ValueError(f"Error initializing model: {e}") from e

    # Model selection -----------------------------------------------------------------------------------------------------#
    def _init_model(self):
        """
        ____________________________________________________________________________________________________________________
        Initialize the underlying model with appropriate default parameters.
        ____________________________________________________________________________________________________________________
        Returns:
            - model : Initialized model instance (ElasticNet or SVR)
        ____________________________________________________________________________________________________________________
        """
        # Get the default model params path relative to this file
        default_model_params_path = Path(__file__).parent / "model_params"
        
        params = self.model_params.copy()
        
        # ElasticNet Regressor (sklearn, CPU only) ------------------------------------------------------------------------#
        if self.model_type == "elasticnet":
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(default_model_params_path / "elasticnet.yaml"))
            default_params.update(params)
            
            # Remove unsupported parameters
            clean_params = default_params.copy()
            clean_params.pop("device", None)
            
            # Set n_jobs if provided (ElasticNet supports n_jobs for coordinate descent)
            if self.n_jobs is not None:
                clean_params["n_jobs"] = self.n_jobs
            
            return ElasticNet(**clean_params)
        
        # Support Vector Regressor ----------------------------------------------------------------------------------------#
        elif self.model_type == "svr":
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(default_model_params_path / "svr.yaml"))
            default_params.update(params)
            
            # Remove unsupported parameters
            clean_params = default_params.copy()
            clean_params.pop("device", None)
            clean_params.pop("n_jobs", None)  

            if self.verbose:
                print("Using cuML SVR for GPU acceleration")
        
            return cuSVR(**clean_params)
 
    # Fit a given model using data points ---------------------------------------------------------------------------------#
    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series], 
            dict_params: Optional[dict] = None
            ) -> "MLBasicRegressor":
        """
        ____________________________________________________________________________________________________________________
        Fit the model to the training data.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X_train     (np.ndarray/pd.DataFrame) : Mandatory. Training features.
        -> y_train     (np.ndarray/pd.Series)    : Mandatory. Training targets.
        -> dict_params (dict)                    : Optional. Additional parameters to pass to fit method.           
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
        
        # Validate y_train dimensionality (should be 1D) and convert if necessary
        if hasattr(y_train, 'shape') and len(y_train.shape) > 1:
            if y_train.shape[1] == 1:
                # Auto-convert (n, 1) to (n,)
                y_train = y_train.ravel()
                if self.verbose:
                    print(f"Info: y_train was reshaped from 2D to 1D")
            elif y_train.shape[1] > 1:
                raise ValueError(f"y_train must be 1D array, got shape {y_train.shape}")
        
        # Get number of features
        n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
        
        # Store feature names if available, or validate existing ones
        if hasattr(X_train, 'columns'):
            if self.feature_names is None:
                self.feature_names = list(X_train.columns)
        
            elif len(self.feature_names) != n_features:
                raise ValueError(f"Provided {len(self.feature_names)} names but X_train has {n_features} features")
        
        elif self.feature_names is not None and len(self.feature_names) != n_features:
            raise ValueError(f"Provided {len(self.feature_names)} names but X_train has {n_features} features")
        
        # Validate feature names are unique
        if self.feature_names and len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("Feature names must be unique")
        
        # Start fitting the model -----------------------------------------------------------------------------------------#
        if self.verbose: 
            print(f"Fitting {self.model_type} model with {X_train.shape} samples...")

        try:
            # Prepare parameters
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
        
        return self

    # Perform inference with new data points ------------------------------------------------------------------------------# 
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
        -> NotFittedError, ValueError
        ____________________________________________________________________________________________________________________
        """
        # Input validation ------------------------------------------------------------------------------------------------#
        if not self.is_fitted : raise NotFittedError("Model must be fitted before making predictions")
        if X is None          : raise ValueError("X cannot be None")
        
        # Make predictions ------------------------------------------------------------------------------------------------#
        try:           
            preds = self.model.predict(X)
            
            return preds
        
        except (TypeError, ValueError, AttributeError) as e:
            raise ValueError(f"Error making predictions: {e}") from e

    # Helper method to compute importance from coefficients --------------------------------------------------------------#
    def _compute_importance_from_coefficients(self, coefficients: np.ndarray) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Helper method to compute feature importance from model coefficients.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> coefficients (np.ndarray) : Model coefficients
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary mapping feature names to absolute coefficient values
        ____________________________________________________________________________________________________________________
        """
        # Ensure we have feature names (generate generic ones if needed)
        if self.feature_names is None or len(self.feature_names) == 0:
            self.feature_names = [f"f{i}" for i in range(len(coefficients))]
            if self.verbose:
                print(f"Warning: No feature names provided. Using generic names.")
        
        # Validate dimensions match
        if len(self.feature_names) != len(coefficients):
            raise ValueError(
                f"Mismatch: {len(self.feature_names)} feature names but {len(coefficients)} coefficients"
            )
        
        # Create importance dictionary using absolute values and sort by importance
        importance_dict = dict(zip(self.feature_names, np.abs(coefficients)))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    # Return feature importance/coefficients for evaluation ---------------------------------------------------------------#
    def get_feature_importance(self, recompute: bool = False) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Get feature importance or coefficients from the fitted model.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> recompute (bool) : Optional. If True, recompute importance even if cached. Default False.
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Dictionary mapping feature names to importance scores/coefficients
        ____________________________________________________________________________________________________________________
        Raises:
        -> NotFittedError, AttributeError, ValueError
        ____________________________________________________________________________________________________________________
        Notes:
            - ElasticNet returns absolute coefficients (coef_)
            - SVR returns absolute coefficients for linear kernel only
            - Results are cached in self.feature_importance_ after first computation
            - importance_type is set to "coefficients" for linear models
        ____________________________________________________________________________________________________________________
        """
        # Input and model validation --------------------------------------------------------------------------------------#
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before getting feature importance")
        
        # Return cached version if available and not recomputing ----------------------------------------------------------#
        if not recompute and self.feature_importance_ is not None:
            return self.feature_importance_
        
        # Compute feature importance based on model type ------------------------------------------------------------------#
        try:
            # ElasticNet has coefficients
            if self.model_type == "elasticnet":
                # Raise error if the model does not have coefficients (should not happen if fitted correctly)
                if not hasattr(self.model, "coef_"):
                    raise AttributeError("ElasticNet model does not have coefficients")
                
                coefficients = self.model.coef_
                importance_dict = self._compute_importance_from_coefficients(coefficients)
                
            # SVR with linear kernel has coefficients
            elif self.model_type == "svr":
                # Check for linear kernel and coefficients in the model
                if not hasattr(self.model, "coef_"):
                    kernel = getattr(self.model, "kernel", "unknown")
                    raise AttributeError(
                        f"SVR with kernel '{kernel}' does not have coefficients. "
                        "Only linear kernel provides feature importance."
                    )
                
                if hasattr(self.model, "kernel") and self.model.kernel != "linear":
                    raise AttributeError(
                        f"SVR with kernel '{self.model.kernel}' does not support feature importance. "
                        "Only linear kernel provides coefficients."
                    )
                
                # Extract coefficients (handle different shapes)
                coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                importance_dict = self._compute_importance_from_coefficients(coefficients)
                
            else:
                raise AttributeError(f"Model type '{self.model_type}' does not support feature importance")
            
            # Cache the result
            self.feature_importance_ = importance_dict
            
            return importance_dict
            
        except (AttributeError, KeyError, IndexError) as e:
            raise ValueError(f"Error getting feature importance: {e}") from e

    # Metric evaluation of a given model ----------------------------------------------------------------------------------#
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], 
                 metrics: Optional[List[str]] = None
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
        # Model and input validation --------------------------------------------------------------------------------------#
        if not self.is_fitted:
            raise NotFittedError("Model must be fitted before evaluation")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Metrics to compute (default to common regression metrics if not provided) ---------------------------------------#
        if metrics is None:
            metrics = DEFAULT_METRICS
        
        try:
            y_pred = self.model.predict(X)
            results = {}
            
            # Compute metrics and store in dictionary
            for metric in metrics:
                if   (metric.lower() == "mse")  : results["mse"]  = mean_squared_error(y, y_pred)
                elif (metric.lower() == "rmse") : results["rmse"] = root_mean_squared_error(y, y_pred)
                elif (metric.lower() == "mae")  : results["mae"]  = mean_absolute_error(y, y_pred)
                elif (metric.lower() == "r2")   : results["r2"]   = r2_score(y, y_pred)
                
                else:
                    if self.verbose: 
                        print(f"Warning: Unknown metric '{metric}' ignored")
            
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
                "model_type"          : self.model_type,
                "model_params"        : self.model_params,
                "model"               : self.model,
                "is_fitted"           : self.is_fitted,
                "feature_names"       : self.feature_names,
                "device"              : self.device,
                "importance_type"     : self.importance_type,
                "feature_importance_" : self.feature_importance_
            }
            
            joblib.dump(save_data, path)
            
            if self.verbose: 
                print(f"Model saved successfully to: {path}")
            
        except (OSError, IOError, TypeError) as e:
            raise ValueError(f"Error saving model: {e}") from e

    # Load model weights and attributes using classmethod decorator -------------------------------------------------------#
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None, verbose: bool = False) -> "MLBasicRegressor":
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
        -> MLBasicRegressor : Loaded model instance
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
            
            # Use provided device or fall back to saved device
            load_device = device if device is not None else data.get("device", "cpu")
            
            # Create instance (model will be overwritten with saved one, so device here is temporary)
            instance = cls(
                model_type   = data["model_type"],  
                model_params = data["model_params"], 
                verbose      = verbose, 
                device       = load_device
            )
            
            # Restore model state
            instance.model               = data["model"]
            instance.is_fitted           = data.get("is_fitted", False)
            instance.feature_names       = data.get("feature_names", None)
            instance.device              = load_device
            instance.importance_type     = data.get("importance_type", DEFAULT_IMPORTANCE_TYPE)
            instance.feature_importance_ = data.get("feature_importance_", None)
            
            # Device validation and warnings
            if instance.model_type == "elasticnet" and load_device == "cuda":
                if verbose:
                    warnings.warn("ElasticNet loaded with 'cuda' device, but will use CPU.")
                instance.device = "cpu"
            
            elif instance.model_type == "svr" and load_device == "cpu":
                if verbose:
                    warnings.warn("cuML SVR loaded with 'cpu' device, but requires GPU. Device will be set to 'cuda'.")
                instance.device = "cuda"
            
            if verbose: 
                print(f"Model loaded successfully from: {path}")
            
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
            "device"                 : self.device,
            "importance_type"        : self.importance_type,
            "has_feature_importance" : self.feature_importance_ is not None,
        }
        
        # Add model-specific information
        if self.is_fitted:
            if self.model_type == "elasticnet":
                info["alpha"]    = getattr(self.model, "alpha", None)
                info["l1_ratio"] = getattr(self.model, "l1_ratio", None)
                info["n_iter"]   = getattr(self.model, "n_iter_", None)
                
            elif self.model_type == "svr":
                info["kernel"]  = getattr(self.model, "kernel", None)
                info["C"]       = getattr(self.model, "C", None)
                info["epsilon"] = getattr(self.model, "epsilon", None)
                if hasattr(self.model, "support_"):
                    info["n_support"] = len(self.model.support_)
            
            # Add convergence information for ElasticNet
            if self.model_type == "elasticnet" and hasattr(self.model, "n_iter_"):
                if hasattr(self.model, "max_iter"):
                    info["converged"] = self.model.n_iter_ < self.model.max_iter
        
        return info
    
    # String representation for debugging ---------------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """
        ____________________________________________________________________________________________________________________
        Return a string representation of the MLBasicRegressor instance.
        ____________________________________________________________________________________________________________________
        Returns:
            - str : String representation
        ____________________________________________________________________________________________________________________
        """
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        n_features_str = f", n_features={len(self.feature_names)}" if self.feature_names else ""
        return f"MLBasicRegressor(model_type='{self.model_type}', {fitted_str}, device='{self.device}'{n_features_str})"

#--------------------------------------------------------------------------------------------------------------------------#
