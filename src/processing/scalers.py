# Modules -----------------------------------------------------------------------------------------------------------------#
import joblib

import numpy                 as np
import sklearn.preprocessing as sklprep

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing  import Union, Literal, Optional, Dict, Any
from pathlib import Path

# Scaler Registry ---------------------------------------------------------------------------------------------------------#
SCALER_REGISTRY = {
    'standard' : sklprep.StandardScaler ,
    'robust'   : sklprep.RobustScaler ,
    'quantile' : sklprep.QuantileTransformer ,
    'power'    : sklprep.PowerTransformer
    }

# Store the transformation of the target to retrieve the physical value ---------------------------------------------------#
class TargetTransform:
    """
    ________________________________________________________________________________________________________________________
    TargetScaler: transformation and inverse transformation of the target variable in a safe space.
    ________________________________________________________________________________________________________________________
    -> Current transformations: 
        -> "identity" : No transformation, target is used as is.
        -> "ratio"    : y = target / norm_factor
        -> "log_ratio": y = log10(target / norm_factor + epsilon)
        -> "log_raw"  : y = log10(target + epsilon)
        
    -> Inverse transformation: 
        -> "identity" : target = y
        -> "ratio"    : target = y * norm_factor
        -> "log_ratio": target = (10^y - epsilon) * norm_factor
        -> "log_raw"  : target = 10^y - epsilon
    ________________________________________________________________________________________________________________________
    Notes:
    This Scaler works if the target variable its strictly positive and has was intended to be used on a variable restricted
    from 0 to 1. The epsilon parameter is a small constant added to avoid log(0) errors, and should be set based on the 
    expected range of values for the target variable.
    ________________________________________________________________________________________________________________________
    """
    # Initialization ------------------------------------------------------------------------------------------------------#
    def __init__(self, transformation:Literal["identity", "log_raw", "ratio", "log_ratio"], 
                 norm_factor : Optional[np.ndarray] = None,
                 epsilon     : Optional[float]      = 1e-6):
        """
        ____________________________________________________________________________________________________________________
        Parameters:
        -> transformation : Type of transformation to apply ("identity", "log_raw", "ratio", "log_ratio")
        -> norm_factor    : Array of normalization factors for scaling. Required for "ratio" and "log_ratio".
        -> epsilon        : Small constant to avoid log(0)
        ____________________________________________________________________________________________________________________
        """
        if transformation in ("ratio", "log_ratio") and norm_factor is None:
            raise ValueError(f"norm_factor is required for transformation '{transformation}'")
        
        self.norm_factor    = norm_factor
        self.epsilon        = epsilon
        self.transformation = transformation
        
    # Forward transformation ----------------------------------------------------------------------------------------------#
    def transform(self, target: np.ndarray) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Transform from target variable to the transformed space.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> target: Target variable in original scale
        ____________________________________________________________________________________________________________________    
        Returns:
            y: Transformed values
        ____________________________________________________________________________________________________________________
        """
        if   self.transformation == "identity":  y = target
        elif self.transformation == "log_raw":   y = np.log10(target + self.epsilon)
        elif self.transformation == "ratio":     y = target / self.norm_factor
        elif self.transformation == "log_ratio": y = np.log10(target / self.norm_factor + self.epsilon)
        else: raise ValueError(f"Unknown transformation type: {self.transformation}")
        
        return y

    # Inverse transformation ----------------------------------------------------------------------------------------------#
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Transform from transformed space back to the original target variable.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> y: Predictions in the transformed space
        ____________________________________________________________________________________________________________________    
        Returns:
            target: Target variable in original scale, ensuring non-negative values by clipping at 0.
        ____________________________________________________________________________________________________________________
        """
        if self.transformation == "identity":
            return y
        
        if self.transformation == "ratio":
            return np.clip(y * self.norm_factor, 0, None)
        
        # For log-based transformations, clip to safe range to avoid float64 overflow (max ~10^308)
        y_safe = np.clip(np.asarray(y, dtype=np.float64), -300.0, 300.0)
        
        if   self.transformation == "log_raw"  :   target = np.power(10, y_safe) - self.epsilon
        elif self.transformation == "log_ratio": target = (np.power(10, y_safe) - self.epsilon) * self.norm_factor
        
        else: raise ValueError(f"Unknown transformation type: {self.transformation}")
        
        return np.clip(target, 0, None)
    
    # Representation for logging and debugging ----------------------------------------------------------------------------#
    def __repr__(self):
        nf_repr = len(self.norm_factor) if self.norm_factor is not None else None
        return f"TargetScaler(transformation={self.transformation}, norm_factor={nf_repr}, epsilon={self.epsilon})"
    
#--------------------------------------------------------------------------------------------------------------------------#

# Feature Scaler ----------------------------------------------------------------------------------------------------------#
class FeatureScaler:
    """
    ________________________________________________________________________________________________________________________
    FeatureScaler: Scaling utility for tabular features with safe persistence
    ________________________________________________________________________________________________________________________

    -> Supported scalers:
        - "standard" : StandardScaler
        - "robust"   : RobustScaler
        - "quantile" : QuantileTransformer
        - "power"    : PowerTransformer

    -> Behavior:
        - Fit and transform feature matrices
        - Safe inverse transformation
        - Save/load fitted scaler for reproducibility

    -> Notes:
        - Input expected shape: (n_samples, n_features)
        - No transformation of physical meaning (only scaling)
    ________________________________________________________________________________________________________________________
    """

    # Initialization ------------------------------------------------------------------------------------------------------#
    def __init__(self, scaler_name : Literal["standard", "robust", "quantile", "power"],
                 scaler_kwargs     : Optional[Dict[str, Any]] = None,
                 load_scaler_from  : Optional[str]            = None):

        # Validation ------------------------------------------------------------------------------------------------------#
        if scaler_name not in SCALER_REGISTRY:
            raise ValueError(f"scaler_name must be one of {list(SCALER_REGISTRY.keys())}, got '{scaler_name}'")

        # Initialize variables
        self.scaler_name   = scaler_name
        self.scaler_kwargs = scaler_kwargs or {}
        self.scaler        = None
        self.fitted        = False

        # Initialize / Load ----------------------------------------------------------------------------------------------#
        if load_scaler_from is not None:
            self.load_scaler(load_scaler_from)
        else:
            self._init_scaler()

    # Fit ----------------------------------------------------------------------------------------------------------------#
    def fit(self, X: np.ndarray):
        X = self._validate_input(X)

        self.scaler.fit(X)
        self.fitted = True
        return self

    # Transform -----------------------------------------------------------------------------------------------------------#
    def transform(self, X: np.ndarray) -> np.ndarray:
        is_1d  = np.ndim(X) == 1
        X      = self._validate_input(X)

        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Use fit or fit_transform first.")
        
        X_scaled = self.scaler.transform(X)

        if is_1d:
            X_scaled = X_scaled.reshape(-1)

        return X_scaled

    # Fit + Transform ----------------------------------------------------------------------------------------------------#
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        is_1d       = np.ndim(X) == 1
        
        X           = self._validate_input(X)
        X_scaled    = self.scaler.fit_transform(X)
        
        self.fitted = True

        if is_1d:
            X_scaled = X_scaled.reshape(-1)
        
        return X_scaled

    # Inverse Transform --------------------------------------------------------------------------------------------------#
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        is_1d    = np.ndim(X_scaled) == 1
        X_scaled = self._validate_input(X_scaled)

        if not self.fitted:
            raise RuntimeError("Scaler not fitted.")
        
        X =self.scaler.inverse_transform(X_scaled)
        
        if is_1d:
            X = X.reshape(-1)

        return X 

    # [Helper] Initialize scaler ------------------------------------------------------------------------------------------#
    def _init_scaler(self):
        self.scaler = SCALER_REGISTRY[self.scaler_name](**self.scaler_kwargs)
        self.fitted = False

    # [Helper] Input validation -------------------------------------------------------------------------------------------#
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            # Convert (n_samples,) → (n_samples, 1)
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("Input must be 2D array (n_samples, n_features)")

        return X

    # Save scaler ---------------------------------------------------------------------------------------------------------#
    def save_scaler(self, path: Union[str, Path]):
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path object")
        joblib.dump({
            "scaler_name"   : self.scaler_name,
            "scaler_kwargs" : self.scaler_kwargs,
            "scaler"        : self.scaler,
            "fitted"        : self.fitted
        }, path)

    # Load scaler ---------------------------------------------------------------------------------------------------------#
    def load_scaler(self, path: Union[str, Path]):
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path object")
        
        obj = joblib.load(path)

        self.scaler_name   = obj["scaler_name"]
        self.scaler_kwargs = obj["scaler_kwargs"]
        self.scaler        = obj["scaler"]
        self.fitted        = obj["fitted"]

    # Representation ------------------------------------------------------------------------------------------------------#
    def __repr__(self):
        return (
            f"FeatureScaler("
            f"scaler_name = {self.scaler_name}, "
            f"fitted      = {self.fitted}"
            f")"
        )
#--------------------------------------------------------------------------------------------------------------------------#