# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from typing           import Union, Optional, List, Dict
from src.utils.eval   import compute_metrics

# Constants ---------------------------------------------------------------------------------------------------------------#
DEFAULT_METRICS = ["mse", "rmse", "mae", "r2"]

# Evaluator Class ---------------------------------------------------------------------------------------------------------#
class Evaluator:
    """
    ________________________________________________________________________________________________________________________
    Evaluator: Handles model evaluation and metric computation
    ________________________________________________________________________________________________________________________
    Responsibilities:
    -> Compute regression metrics (MSE, RMSE, MAE, R²)
    -> Handle different input formats (arrays, tensors, DataLoaders)
    -> Provide comprehensive evaluation reports
    ________________________________________________________________________________________________________________________
    """
    
    def __init__(self, predictor, verbose: bool = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the Evaluator.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> predictor : Predictor instance to use for making predictions
        -> verbose   : Whether to log warnings for unknown metrics
        ____________________________________________________________________________________________________________________
        """
        self.predictor = predictor
        self.verbose   = verbose
    
    def evaluate(self, X: Union[np.ndarray, torch.Tensor, DataLoader], 
                 y: Union[np.ndarray, torch.Tensor],
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Evaluate the model performance using multiple metrics.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> X       (array-like or DataLoader) : Test features
        -> y       (array-like)               : Test targets
        -> metrics (list)                     : Optional. List of metrics to compute. Default: ["mse", "rmse", "mae", "r2"]
        ____________________________________________________________________________________________________________________
        Returns:
        -> results (dict) : Dictionary containing evaluation metrics
        ____________________________________________________________________________________________________________________
        Raises:
        -> ValueError : If X or y are None, or if dimensions don't match
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if metrics is None:
            metrics = DEFAULT_METRICS
        
        try:
            # Get predictions
            y_pred = self.predictor.predict(X)
            
            # Convert y to numpy if needed
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            elif not isinstance(y, np.ndarray):
                y = np.array(y)
            
            # Flatten predictions and targets if needed
            y_pred = y_pred.flatten()
            y      = y.flatten()
            
            # Check dimensions match
            if len(y_pred) != len(y):
                raise ValueError(f"Prediction length ({len(y_pred)}) doesn't match target length ({len(y)})")
            
            # Compute metrics
            results = compute_metrics(y, y_pred, metrics, verbose=self.verbose)
            
            return results
            
        except (TypeError, ValueError, RuntimeError) as e:
            raise ValueError(f"Error evaluating model: {e}") from e

#--------------------------------------------------------------------------------------------------------------------------#
