# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy as np

# Custom implementation of early stopping for regularization --------------------------------------------------------------#
class EarlyStopping:
    """
    ________________________________________________________________________________________________________________________
    Early stopping as the convergence criterion.
    ________________________________________________________________________________________________________________________
    Regularization of a DL models through Monitors a metric and stops training when it stops improving.
    By default, assumes lower metric values are better (minimization).
    ________________________________________________________________________________________________________________________
    Args:
        -patience (int) : Number of epochs to wait for improvement before stopping.
        -mode     (str) : One of 'min' or 'max'. 
    ________________________________________________________________________________________________________________________
    Attributes:
        best_metric (float) : Best metric value observed so far.
        counter     (int)   : Number of epochs without improvement.
        patience    (int)   : Maximum epochs to wait for improvement.
        mode        (str)   : Whether to minimize or maximize the metric.
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, patience: int = 10, mode: str = 'min'):
        """Initialize the EarlyStopping instance."""
        # Validate inputs -------------------------------------------------------------------------------------------------#
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

        # Store configuration ---------------------------------------------------------------------------------------------#
        self.patience    = patience
        self.mode        = mode
        self.best_metric = np.inf if mode == 'min' else -np.inf
        self.counter     = 0

    def step(self, metric: float) -> tuple[bool, bool]:
        """Evaluate the current metric and determine if training should stop."""
        
        if self.mode == 'min':
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric
            self.counter    = 0
        else:
            self.counter += 1

        stop = self.counter >= self.patience

        return stop, is_best

#--------------------------------------------------------------------------------------------------------------------------#