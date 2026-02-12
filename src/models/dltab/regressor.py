# Modules -----------------------------------------------------------------------------------------------------------------#
import os

import numpy as np
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from pathlib          import Path
from typing           import Optional, Dict, List, Union, Any
from loguru           import logger

# Custom functions --------------------------------------------------------------------------------------------------------#

# Safe load of yaml dictionaries
from src.utils.directory import load_yaml_dict

# Check GPU availability
from src.utils.callbacks import check_gpu_available

# DLTabular specific imports: Encoder architecture
from src.models.dltab.encoders.mlp import MLPRegressor

# DLTabular specific imports: Trainer, Predictor, Evaluator, CheckpointManager
from src.models.dltab.core.trainer     import Trainer
from src.models.dltab.core.predictor   import Predictor
from src.models.dltab.core.evaluator   import Evaluator
from src.models.dltab.utils.checkpoint import CheckpointManager

# Optimizer selection
from src.models.dltab.utils.optimizers import select_optimizer

# Helpers
from src.models.dltab.utils.handling import _setup_logger

# Constants ---------------------------------------------------------------------------------------------------------------#
SUPPORTED_MODELS      = ["mlp"]
SUPPORTED_OPTIMIZERS  = ["adam", "sgd"]
DEFAULT_METRICS       = ["mse", "rmse", "mae", "r2"]
DEFAULT_VERBOSE_EPOCH = 10

# Custom Deep Learning Tabular Regressor ----------------------------------------------------------------------------------#
class DLTabularRegressor:
    """
    ________________________________________________________________________________________________________________________
    DLTabularRegressor: A comprehensive deep learning supervised regressor wrapper for model comparison
    ________________________________________________________________________________________________________________________
    Models supported:
    -> MultiLayer Perceptron Network (MLPRegressor) [custom implementation - pytorch]
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, model_type : str='mlp', in_features: int = 5, model_params: Optional[dict]=None, 
                 optimizer_name   : str            = 'adam',
                 optimizer_params : Optional[dict] = None,
                 scheduler_name   : Optional[str]  = None,
                 scheduler_params : Optional[dict] = None,
                 feat_names       : Optional[list] = None, 
                 n_jobs           : Optional[int]  = None, 
                 device           : Union[str, torch.device] = 'cuda',
                 use_amp          : bool           = False,
                 log_file         : Optional[str]  = None,
                 verbose          : bool           = False):
        """    
        ____________________________________________________________________________________________________________________
        Initialize the DLTabularRegressor with specified model type and parameters.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model_type       (str)  : Mandatory. Type of model to use ('mlp').
        -> in_features      (int)  : Mandatory. Number of input features for the model.
        -> model_params     (dict) : Optional. Dictionary containing model-specific hyperparameters.
        -> optimizer_name   (str)  : Mandatory. Name of the optimizer to use ('adam', 'sgd'). Default: 'adam'.
        -> optimizer_params (dict) : Optional. Dictionary containing optimizer-specific hyperparameters.
        -> scheduler_name   (str)  : Optional. Name of the learning rate scheduler to use.
        -> scheduler_params (dict) : Optional. Dictionary containing scheduler-specific hyperparameters.
        -> feat_names       (list) : Optional. List of feature names.
        -> device           (str)  : Optional. 'cpu' or 'cuda' for GPU CUDA driven acceleration. Default: 'cuda'.
        -> n_jobs           (int)  : Optional. Number of cores to use during computation.
        -> log_file         (str)  : Optional. Path to log file. If None, only console logging.
        -> use_amp          (bool) : Optional. Enable Automatic Mixed Precision training. Default: False.
        -> verbose          (bool) : Optional. Enable verbose logging for debugging purposes.
        ____________________________________________________________________________________________________________________
        Raises:
        -> TypeError, KeyError, ImportError, AttributeError, RuntimeError for invalid inputs or issues during initialization
        ____________________________________________________________________________________________________________________
        """
        # Setup logger first -----------------------------------------------------------------------------------------------#
        _setup_logger(log_file)
        
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")
        if model_params is not None and not isinstance(model_params, dict):
            raise TypeError("model_params must be a dictionary or None")
        if not isinstance(optimizer_name, str):
            raise TypeError("optimizer_name must be a string")
        
        # Validate device before creating torch.device
        if isinstance(device, str):
            if device not in ["cpu", "cuda"]:
                raise ValueError(f"device must be 'cpu' or 'cuda', got '{device}'")
            if device == "cuda" and not check_gpu_available():
                raise RuntimeError(
                    "CUDA device requested but no GPU detected. "
                    "Please ensure CUDA is properly installed and a GPU is available, or use device='cpu'."
                )
        
        # Main parameters -------------------------------------------------------------------------------------------------#
        self.model_type       = model_type.lower()
        self.in_features      = in_features
        self.model_params     = model_params if model_params is not None else {}
        self.optimizer_name   = optimizer_name.lower()
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler_name   = scheduler_name.lower() if scheduler_name is not None else None
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.n_jobs           = n_jobs
        self.verbose          = verbose
        self.is_fitted        = False
        self.feature_names    = feat_names
        self.device           = torch.device(device) if isinstance(device, str) else device
        self.use_amp          = use_amp
        self.history          = {}
        self.best_model_state = None  
        
        # Validate model type ---------------------------------------------------------------------------------------------#
        if self.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {SUPPORTED_MODELS}")
        
        # Validate optimizer ----------------------------------------------------------------------------------------------#
        if self.optimizer_name not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}. Supported optimizers: {SUPPORTED_OPTIMIZERS}")
        
        # Create the class element ----------------------------------------------------------------------------------------#
        if self.verbose: logger.info(f"Initializing {self.model_type} model (device={self.device})...")
        
        try:
            self._init_model()
            
            if self.verbose: logger.success(f"Successfully initialized {self.model_type} architecture")
            
            # Optimizer is now mandatory
            self._init_optimizer()
            
            if self.verbose: logger.success(f"Successfully initialized {self.optimizer_name} optimizer")
            
            if self.use_amp:
                self.scaler = torch.amp.GradScaler(self.device.type)
                if self.verbose: logger.success("Enabled Automatic Mixed Precision (AMP) for training")
            
            # Initialize Predictor, Evaluator, and CheckpointManager
            self.predictor          = Predictor(model=self.model, device=self.device, use_amp=self.use_amp)
            self.evaluator          = Evaluator(predictor=self.predictor, verbose=self.verbose)
            self.checkpoint_manager = CheckpointManager(verbose=self.verbose)
        
        except (TypeError, KeyError, ImportError, AttributeError, RuntimeError) as e:
            raise ValueError(f"Error initializing model: {e}") from e
         
    # Main fitting function -----------------------------------------------------------------------------------------------#        
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader]=None, epochs: int=100, loss_fn: str='mse', 
            loss_params             : Optional[Dict[str, Any]] = None,
            early_stopping_patience : Optional[int] = None, 
            verbose_epoch           : int           = DEFAULT_VERBOSE_EPOCH,
            checkpoint_path         : Optional[str] = None, 
            save_best_only          : bool          = True):
        """
        ____________________________________________________________________________________________________________________
        Train the model on the provided dataset.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> train_loader             (DataLoader) : DataLoader containing training batches
        -> val_loader               (DataLoader) : Optional DataLoader containing validation batches
        -> epochs                   (int)        : Number of training epochs
        -> loss_fn                  (str)        : Loss function name ('mse', 'l1', 'smooth_l1', 'huber')
        -> loss_params              (dict)       : Optional dictionary of additional loss function parameters
        -> early_stopping_patience  (int)        : Epochs to wait before stopping if no improvement
        -> verbose_epoch            (int)        : Print training info every N epochs
        -> checkpoint_path          (str)        : Optional path to save model checkpoints during training
        -> save_best_only           (bool)       : If True, only save checkpoint when validation loss improves
        ____________________________________________________________________________________________________________________
        """
        # Create checkpoint callback if path provided
        checkpoint_callback = None
        if checkpoint_path is not None:
            def _checkpoint_fn(epoch, is_best, val_loss):
                # Save if best_only and is_best, or if not best_only
                should_save = (save_best_only and is_best) or (not save_best_only)
                if should_save:
                    self.save_model(checkpoint_path)
                    if self.verbose:
                        if is_best:
                            logger.info(f"Model saved: Best val loss = {val_loss:.6f}")
                        else:
                            logger.info(f"Checkpoint saved at epoch {epoch}")
            checkpoint_callback = _checkpoint_fn
        
        # Create trainer instance
        trainer = Trainer(model=self.model, optimizer=self.opt, device=self.device, use_amp=self.use_amp, 
                          scheduler = self.sched if hasattr(self, 'sched') else None, 
                          scaler    = self.scaler if self.use_amp else None,
                          verbose   = self.verbose)
        
        # Run training
        if self.verbose:
            logger.info(f"Model: {self.model_type}, Optimizer: {self.optimizer_name}")
        
        result = trainer.train(train_loader=train_loader, val_loader=val_loader, epochs=epochs, loss_fn=loss_fn,
                               loss_params             = loss_params,
                               early_stopping_patience = early_stopping_patience,
                               verbose_epoch           = verbose_epoch,
                               checkpoint_callback     = checkpoint_callback
                               )
        
        # Update instance state
        self.history          = result['history']
        self.best_model_state = result['best_model_state']
        self.is_fitted        = True
        
        return self
    
    # Main prediction function --------------------------------------------------------------------------------------------#
    def predict(self, X: Union[np.ndarray, torch.Tensor, DataLoader]) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Make predictions on new data.
        ____________________________________________________________________________________________________________________
        Parameters:
            - X (array-like or DataLoader) : Input features as array/tensor or DataLoader for batch predictions
        ____________________________________________________________________________________________________________________
        Returns:
            - predictions (np.ndarray) : Model predictions
        ____________________________________________________________________________________________________________________
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Delegate to predictor
        return self.predictor.predict(X)
    
    # Saving of models ----------------------------------------------------------------------------------------------------#
    def save_model(self, path: str) -> None:
        """
        ____________________________________________________________________________________________________________________
        Save the model to disk using PyTorch's state dict.
        ____________________________________________________________________________________________________________________
        Parameters:
            - path (str) : Mandatory. Path where to save the model.
        ____________________________________________________________________________________________________________________
        Raises:
            - ValueError, OSError
        ____________________________________________________________________________________________________________________
        Notes:
            All tensors are moved to CPU before saving to ensure compatibility.
        ____________________________________________________________________________________________________________________
        """
        # Delegate to checkpoint manager
        optimizer = self.opt if hasattr(self, 'opt') else None
        scheduler = self.sched if hasattr(self, 'sched') else None
        
        self.checkpoint_manager.save(path=path, model=self.model, model_type=self.model_type, 
                                     model_params     = self.model_params,
                                     optimizer_name   = self.optimizer_name,
                                     optimizer_params = self.optimizer_params,
                                     is_fitted        = self.is_fitted,
                                     feature_names    = self.feature_names,
                                     history          = self.history,
                                     optimizer        = optimizer,
                                     scheduler        = scheduler,
                                     scheduler_name   = self.scheduler_name,
                                     scheduler_params = self.scheduler_params)
    
    # Load of training weights --------------------------------------------------------------------------------------------#
    @classmethod
    def load_model(cls, path: str, device: Optional[Union[str, torch.device]] = None, 
                   verbose: bool = False) -> "DLTabularRegressor":
        """
        ____________________________________________________________________________________________________________________
        Load a saved model from disk.
        ____________________________________________________________________________________________________________________
        Parameters:
            - path    (str)  : Mandatory. Path to the saved model file.
            - device  (str)  : Optional. Device to use ('cpu' or 'cuda'). If None, uses saved device.
            - verbose (bool) : Optional. Enable verbose logging.
        ____________________________________________________________________________________________________________________
        Returns:
            - DLTabularRegressor : Loaded model instance
        ____________________________________________________________________________________________________________________
        Raises:
            - FileNotFoundError, ValueError, OSError
        ____________________________________________________________________________________________________________________
        """
        # Use CheckpointManager to load
        checkpoint_manager = CheckpointManager(verbose=verbose)
        checkpoint         = checkpoint_manager.load(path)
        
        try:
            # Determine device
            target_device = device if device is not None else checkpoint.get("device", "cpu")
            
            # Create instance with saved parameters
            instance = cls(
                model_type       = checkpoint["model_type"],
                model_params     = checkpoint["model_params"],
                optimizer_name   = checkpoint.get("optimizer_name"),
                optimizer_params = checkpoint.get("optimizer_params"),
                scheduler_name   = checkpoint.get("scheduler_name"),
                scheduler_params = checkpoint.get("scheduler_params"),
                feat_names       = checkpoint.get("feature_names"),
                device           = target_device,
                verbose          = verbose
            )
            
            # Load model state
            instance.model.load_state_dict(checkpoint["model_state_dict"])
            instance.model.to(target_device)
            
            # Load optimizer state if available
            if "optimizer_state_dict" in checkpoint and hasattr(instance, 'opt'):
                instance.opt.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state if available
            if "scheduler_state_dict" in checkpoint and hasattr(instance, 'sched'):
                instance.sched.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Restore other attributes
            instance.is_fitted     = checkpoint.get("is_fitted", False)
            instance.history       = checkpoint.get("history", {})
            instance.feature_names = checkpoint.get("feature_names")
            
            if verbose:
                logger.info(f"Model loaded successfully from: {path}")
                logger.info(f"Model type : {instance.model_type}")
                logger.info(f"Device     : {target_device}")
                logger.info(f"Is fitted  : {instance.is_fitted}")
            
            return instance
            
        except (KeyError, RuntimeError) as e:
            raise ValueError(f"Error loading model: {e}") from e
    
    # Evaluation of model -------------------------------------------------------------------------------------------------#
    def evaluate(self, X: Union[np.ndarray, torch.Tensor, DataLoader], y: Union[np.ndarray, torch.Tensor],
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        ____________________________________________________________________________________________________________________
        Evaluate the model performance using multiple metrics.
        ____________________________________________________________________________________________________________________
        Parameters:
            - X       (array-like or DataLoader) : Test features
            - y       (array-like)               : Test targets
            - metrics (list)                     : Optional. List of metrics to compute.
        ____________________________________________________________________________________________________________________
        Returns:
            - dict : Dictionary containing evaluation metrics
        ____________________________________________________________________________________________________________________
        Raises:
            - ValueError
        ____________________________________________________________________________________________________________________
        """
        # Model validation
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Delegate to evaluator
        return self.evaluator.evaluate(X, y, metrics)
    
    # [Helper] Architecture Selection -------------------------------------------------------------------------------------#
    def _init_model(self) -> None:
        """
        ____________________________________________________________________________________________________________________
        Initialize the underlying model with appropriate default parameters.
        ____________________________________________________________________________________________________________________
        """
        # Get config path relative to this file
        config_path = Path(__file__).parent / "config" / "arch" / "mlp.yaml"
        
        params = self.model_params.copy()
        
        # Custom Multilayer Perceptron Regressor --------------------------------------------------------------------------#
        if self.model_type == "mlp":
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path=str(config_path))
            default_params.update(params)
            
            # Set input features - first layer
            default_params['in_features'] = self.in_features

            self.model = MLPRegressor(**default_params).to(self.device)
    
    # [Helper] Optimizer selection ----------------------------------------------------------------------------------------#
    def _init_optimizer(self) -> None:
        """
        ____________________________________________________________________________________________________________________
        Initialize the optimizer for training the model.
        ____________________________________________________________________________________________________________________
        Returns:
            - optimizer              : Initialized optimizer instance.
            - (optimizer, scheduler) : Tuple of optimizer and learning rate scheduler if applicable.
        ____________________________________________________________________________________________________________________
        """
        opt_params = self.optimizer_params.copy()
        sch_params = self.scheduler_params.copy() if self.scheduler_params is not None else None
        
        if self.scheduler_name is not None:
            
            self.opt, self.sched = select_optimizer(name             = self.optimizer_name,
                                                    model            = self.model,
                                                    optimizer_params = opt_params,
                                                    scheduler_name   = self.scheduler_name,
                                                    scheduler_params = sch_params)
        
        else:
            self.opt = select_optimizer(name             = self.optimizer_name,
                                        model            = self.model,
                                        optimizer_params = opt_params,
                                        scheduler_name   = None,
                                        scheduler_params = None)
    
    # Get current learning rate -------------------------------------------------------------------------------------------#
    def get_current_lr(self) -> List[float]:
        """
        ____________________________________________________________________________________________________________________
        Get the current learning rate(s) from the optimizer.
        ____________________________________________________________________________________________________________________
        Returns:
            - list : List of current learning rates for each parameter group
        ____________________________________________________________________________________________________________________
        """
        if not hasattr(self, 'opt'):
            raise RuntimeError("Optimizer not initialized")
        return [group['lr'] for group in self.opt.param_groups]
    
    # Load best model weights ---------------------------------------------------------------------------------------------#
    def load_best_weights(self) -> None:
        """
        ____________________________________________________________________________________________________________________
        Load the best model weights stored during training.
        ____________________________________________________________________________________________________________________
        Raises:
            - RuntimeError if best weights were not saved
        ____________________________________________________________________________________________________________________
        """
        if self.best_model_state is None:
            raise RuntimeError("No best model weights available. Train with early stopping first.")
        self.model.load_state_dict(self.best_model_state)
        if self.verbose:
            logger.info("Loaded best model weights")
    
    # [Helper] Prediction with DataLoader ---------------------------------------------------------------------------------#

    
    # Get model information (public method) -------------------------------------------------------------------------------#
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
            "model_type"       : self.model_type,
            "is_fitted"        : self.is_fitted,
            "model_params"     : self.model_params,
            "optimizer_name"   : self.optimizer_name,
            "optimizer_params" : self.optimizer_params,
            "scheduler_name"   : self.scheduler_name,
            "scheduler_params" : self.scheduler_params,
            "feature_names"    : self.feature_names,
            "device"           : self.device,
            "n_jobs"           : self.n_jobs,
        }
        
        if self.is_fitted and self.history:
            info["training_epochs"] = len(self.history.get("train_loss", []))
            if self.history.get("train_loss"):
                info["final_train_loss"] = self.history["train_loss"][-1]
            if self.history.get("val_loss"):
                info["final_val_loss"] = self.history["val_loss"][-1]
                info["best_val_loss"]  = min(self.history["val_loss"])
        
        # Add current learning rate if optimizer exists
        if hasattr(self, 'opt'):
            try:
                info["current_lr"] = self.get_current_lr()
            except RuntimeError:
                pass
        
        return info
    
    # String representation for debugging ---------------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """
        ____________________________________________________________________________________________________________________
        Return a string representation of the DLTabularRegressor instance.
        ____________________________________________________________________________________________________________________
        Returns:
            - str : String representation
        ____________________________________________________________________________________________________________________
        """
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        n_features_str = f", in_features={self.in_features}"
        return f"DLTabularRegressor(model_type='{self.model_type}', {fitted_str}, device='{self.device}'{n_features_str})"

#--------------------------------------------------------------------------------------------------------------------------#