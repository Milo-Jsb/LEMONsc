# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import copy

import numpy as np
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from pathlib          import Path
from typing           import Optional, Dict, List, Union, Any
from sklearn.metrics  import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from loguru           import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory               import load_yaml_dict
from src.models.dltab.encoders.mlp     import MLPRegressor
from src.models.dltab.utils.optimizers import select_optimizer
from src.models.dltab.utils.losses     import get_loss_function
from src.models.dltab.utils.callbacks  import EarlyStopping

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/dltab_output.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")


# Helpers -----------------------------------------------------------------------------------------------------------------#
def _to_cpu_state_dict(state_dict: dict) -> dict:
    """Convert a state_dict with tensors to CPU."""
    cpu_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        else:
            cpu_state[k] = v
    return cpu_state

# Custom Deep Learning Tabular Regressor ----------------------------------------------------------------------------------#
class DLTabularRegressor:
    """
    ________________________________________________________________________________________________________________________
    DLTabularRegressor: A comprehensive deep learning supervised regressor wrapper for model comparison
    ________________________________________________________________________________________________________________________
    Models supported:
        - MultiLayer Perceptron Network (MLPRegressor) [custom implementation - pytorch]
    ________________________________________________________________________________________________________________________
    """
    def __init__(self, model_type : str='mlp', in_features: int = 5, model_params: Optional[dict]=None, 
                 optimizer_name   : Optional[str]  = None,
                 optimizer_params : Optional[dict] = None,
                 scheduler_name   : Optional[str]  = None,
                 scheduler_params : Optional[dict] = None,
                 feat_names       : Optional[list] = None, 
                 n_jobs           : Optional[int]  = None, 
                 device           : Union[str, torch.device] = 'cuda',
                 use_amp          : bool           = False,
                 verbose          : bool           = False):
        """    
        ____________________________________________________________________________________________________________________
        Initialize the DLTabularRegressor with specified model type and parameters.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model_type       (str)  : Mandatory. Type of model to use ('mlp').
        -> in_features      (int)  : Mandatory. Number of input features for the model.
        -> model_params     (dict) : Optional. Dictionary containing model-specific hyperparameters.
        -> optimizer_name   (str)  : Optional. Name of the optimizer to use ('adam', 'sgd').
        -> optimizer_params (dict) : Optional. Dictionary containing optimizer-specific hyperparameters.
        -> scheduler_name   (str)  : Optional. Name of the learning rate scheduler to use.
        -> scheduler_params (dict) : Optional. Dictionary containing scheduler-specific hyperparameters.
        -> feat_names       (list) : Optional. List of feature names.
        -> device           (str)  : Optional. 'cpu' (default) or 'cuda' for GPU CUDA driven acceleration.
        -> n_jobs           (int)  : Optional. Number of cores to use during computation.
        -> verbose          (bool) : Optional. Enable verbose logging for debugging purposes.
        ____________________________________________________________________________________________________________________
        Raises:
            - ValueError, TypeError
        ____________________________________________________________________________________________________________________
        """
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(model_type , str):
            raise TypeError("model_type  must be a string")
        if model_params is not None and not isinstance(model_params, dict):
            raise TypeError("model_params must be a dictionary or None")
        
        # Main parameters -------------------------------------------------------------------------------------------------#
        self.model_type       = model_type.lower()
        self.in_features      = in_features
        self.model_params     = model_params if model_params is not None else {}
        self.optimizer_name   = optimizer_name.lower() if optimizer_name is not None else None
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler_name   = scheduler_name.lower() if scheduler_name is not None else None
        self.scheduler_params = scheduler_params if scheduler_params is not None else None
        self.n_jobs           = n_jobs
        self.verbose          = verbose
        self.is_fitted        = False
        self.feature_names    = feat_names
        self.device           = torch.device(device)
        self.use_amp          = use_amp
        self.history          = {}
        
        # Validate model type ---------------------------------------------------------------------------------------------#
        supported_models = ["mlp"]
        if self.model_type  not in supported_models:
            raise ValueError(f"Unsupported model type: {self.model_type }. Supported types: {supported_models}")
        
        # Validate optimizer ----------------------------------------------------------------------------------------------#
        supported_models = ["adam", "sgd"]
        if self.optimizer_name is not None and self.optimizer_name.lower() not in supported_models:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}. Supported optimizers: {supported_models}")
        
        # Create the class element ----------------------------------------------------------------------------------------#
        if self.verbose: logger.info(f"Initializing {self.model_type} model (device={self.device})...")
        
        try:
            self._init_model()
            
            if self.verbose: logger.success(f"Successfully initialized {self.model_type} architecture")
            
            if self.optimizer_name is not None:
                
                self._init_optimizer()
            
                if self.verbose: logger.success(f"Successfully initialized {self.optimizer_name} optimizer")
            
            if self.use_amp:
                self.scaler = torch.amp.GradScaler()
                if self.verbose: logger.success("Enabled Automatic Mixed Precision (AMP) for training")
        
        except Exception as e:
            raise ValueError(f"Error initializing model: {e}")
         
    # Main fitting function -----------------------------------------------------------------------------------------------#        
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader]=None, epochs: int=100, loss_fn: str='mse', 
            loss_params             : Optional[Dict[str, Any]] = None,
            early_stopping_patience : Optional[int] = None, 
            verbose_epoch           : int           = 10,
            checkpoint_path         : Optional[str] = None, 
            save_best_only          : bool          = True):
        """
        ____________________________________________________________________________________________________________________
        Train the model on the provided dataset.
        ____________________________________________________________________________________________________________________
        Parameters:
            - train_loader             (DataLoader) : DataLoader containing training batches
            - val_loader               (DataLoader) : Optional DataLoader containing validation batches
            - epochs                   (int)        : Number of training epochs
            - loss_fn                  (str)        : Loss function name ('mse', 'l1', 'smooth_l1', 'huber')
            - loss_params              (dict)       : Optional dictionary of additional loss function parameters
            - early_stopping_patience  (int)        : Epochs to wait before stopping if no improvement
            - verbose_epoch            (int)        : Print training info every N epochs
            - checkpoint_path          (str)        : Optional path to save model checkpoints during training
            - save_best_only           (bool)       : If True, only save checkpoint when validation loss improves
        ____________________________________________________________________________________________________________________
        """
        # Setup loss function and move model to device --------------------------------------------------------------------#
        criterion = get_loss_function(loss_fn, **(loss_params or {})).to(self.device)
        
        # Validate early stopping configuration ---------------------------------------------------------------------------#
        if early_stopping_patience is not None and val_loader is None:
            raise ValueError("val_loader must be provided when early_stopping_patience is set")
        
        # Initialize early stopping callback ------------------------------------------------------------------------------#
        early_stopper = None
        
        if early_stopping_patience is not None:
            early_stopper = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        # Training history ------------------------------------------------------------------------------------------------#
        history       = {'train_loss': [], 'val_loss': [], 'epochs': []}
        best_val_loss = float('inf')
        should_stop   = False

        # Training loop ---------------------------------------------------------------------------------------------------#
        if self.verbose:
            logger.info(f"Starting training for {epochs} epochs (model={self.model_type}, opt={self.optimizer_name})")
            logger.info(f"Loss function : {loss_fn}")
            logger.info(f"Device        : {self.device}")
        
        # Start loop
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.train_one_epoch(train_loader, criterion)
            history['train_loss'].append(train_loss)
            
            # Validate one epoch
            val_loss = self.val_one_epoch(val_loader, criterion)
            history['val_loss'].append(val_loss)
            history['epochs'].append(epoch + 1)
            
            # Update history before any checkpoint saving
            self.history = history
            
            # Handle checkpointing and early stopping
            if early_stopper is not None:
                stop, is_best = early_stopper.step(val_loss)
                
                if is_best:
                    self.is_fitted = True
                    best_val_loss  = val_loss
                    
                    # Save checkpoint if best model and checkpoint path provided
                    if checkpoint_path is not None and save_best_only:
                        self.save_model(checkpoint_path)
                        if self.verbose:
                            logger.info(f"Model saved: Best val loss = {best_val_loss:.6f}")
                
                if stop:
                    should_stop    = True
                    self.is_fitted = True

                    if self.verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        logger.info(f"No improvement for {early_stopping_patience} epochs")
            
            # Save checkpoint every epoch if not save_best_only (regardless of early stopping)
            if checkpoint_path is not None and not save_best_only:
                self.save_model(checkpoint_path)
                if self.verbose:
                    logger.info(f"Checkpoint saved at epoch {epoch + 1}")
            
            # Break if early stopping triggered
            if should_stop:
                break
            
            # Update learning rate scheduler if available
            if hasattr(self, 'sched') and self.sched is not None:
                if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.sched.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.sched.step()
            
            # Verbose logging
            if self.verbose and (epoch + 1) % verbose_epoch == 0:
                msg = f"Epoch [{epoch + 1:4d}/{epochs}] - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        # Mark model as fitted after successful training
        self.is_fitted = True
        self.history = history
        
        if self.verbose:
            logger.success("Training completed!")
            logger.success(f"Final train loss: {history['train_loss'][-1]:.6f}")
            if history['val_loss']:
                logger.success(f"Final val loss : {history['val_loss'][-1]:.6f}")
                logger.success(f"Best val loss  : {best_val_loss:.6f}")
        
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
        
        if X is None:
            raise ValueError("X cannot be None")
        
        self.model.eval()
        
        # Check if input is a DataLoader for batch prediction
        if isinstance(X, DataLoader):
            return self._predict_dataloader(X)
        
        # Standard prediction for arrays/tensors Prepare input data
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        X = X.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X)
        
        return predictions.detach().cpu().numpy()
    
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
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not self.is_fitted:
            logger.warning("Saving model although `is_fitted` is False")
        
        # Save model to checkpoint ----------------------------------------------------------------------------------------#
        try:
            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare save data
            save_data = {
                "model_type"       : self.model_type,
                "model_params"     : self.model_params.copy(),
                "model_state_dict" : _to_cpu_state_dict(self.model.state_dict()),
                "optimizer_name"   : self.optimizer_name,
                "optimizer_params" : self.optimizer_params.copy(),
                "is_fitted"        : self.is_fitted,
                "feature_names"    : self.feature_names,
                "device"           : 'cpu',
                "history"          : self.history.copy(),
            }
            
            # Save optimizer state if available
            optimizer = self.opt if hasattr(self, 'opt') else None
            
            if optimizer:
                # Create a deep copy to avoid modifying the original optimizer state
                opt_sd = copy.deepcopy(optimizer.state_dict())
                if "state" in opt_sd and isinstance(opt_sd["state"], dict):
                    for st_key, st_val in opt_sd["state"].items():
                        for inner_k, inner_v in list(st_val.items()):
                            if isinstance(inner_v, torch.Tensor):
                                st_val[inner_k] = inner_v.cpu()
                save_data["optimizer_state_dict"] = opt_sd
            
            # Save scheduler state and params if available
            scheduler = self.sched if hasattr(self, 'sched') else None
            
            if scheduler:
                # Create a deep copy to avoid modifying the original scheduler state
                sched_sd = copy.deepcopy(scheduler.state_dict())
                if isinstance(sched_sd, dict):
                    for k, v in list(sched_sd.items()):
                        if isinstance(v, torch.Tensor):
                            sched_sd[k] = v.cpu()
                save_data["scheduler_name"]       = self.scheduler_name
                save_data["scheduler_params"]     = self.scheduler_params
                save_data["scheduler_state_dict"] = sched_sd        
            
            # Save using torch
            torch.save(save_data, path)
            
            if self.verbose:
                logger.success(f"Model saved successfully to: {path}")
            
        except Exception as e:
            raise ValueError(f"Error saving model: {e}")
    
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
        # Input validation ------------------------------------------------------------------------------------------------#
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model from checkpoint --------------------------------------------------------------------------------------#
        try:
            # Load data from checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
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
            
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
    
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
        # Model and input validation
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if metrics is None:
            metrics = ["mse", "rmse", "mae", "r2"]
        
        try:
            # Get predictions
            y_pred = self.predict(X)
            
            # Convert y to numpy if needed
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            elif not isinstance(y, np.ndarray):
                y = np.array(y)
            
            # Flatten predictions and targets if needed
            y_pred = y_pred.flatten()
            y = y.flatten()
            
            # Check dimensions match
            if len(y_pred) != len(y):
                raise ValueError(f"Prediction length ({len(y_pred)}) doesn't match target length ({len(y)})")
            
            results = {}
            
            # Compute metrics and store in dictionary
            for metric in metrics:
                if   (metric.lower() == "mse")  : results["mse"]  = mean_squared_error(y, y_pred)
                elif (metric.lower() == "rmse") : results["rmse"] = root_mean_squared_error(y, y_pred)
                elif (metric.lower() == "mae")  : results["mae"]  = mean_absolute_error(y, y_pred)
                elif (metric.lower() == "r2")   : results["r2"]   = r2_score(y, y_pred)
                else:
                    if self.verbose: print(f"Warning: Unknown metric '{metric}' ignored")
            
            return results
            
        except Exception as e:
            raise ValueError(f"Error evaluating model: {e}")
    
    # [Helper] Architecture Selection -------------------------------------------------------------------------------------#
    def _init_model(self) -> torch.nn.Module:
        """
        ____________________________________________________________________________________________________________________
        Initialize the underlying model with appropriate default parameters.
        ____________________________________________________________________________________________________________________
        Returns:
            - model : Initialized model instance (MLPRegressor).
        ____________________________________________________________________________________________________________________
        """
        params = self.model_params.copy()
        
        # Custom Multilayer Perceptron Regressor --------------------------------------------------------------------------#
        if self.model_type == "mlp":
            
            # Retrieve default parameters and update the dictionary
            default_params = load_yaml_dict(path= "./src/models/dltab/config/arch/mlp.yaml")
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

    
    
    
    # [Helper] Prediction with DataLoader ---------------------------------------------------------------------------------#
    def _predict_dataloader(self, dataloader: DataLoader) -> np.ndarray:
        """
        ____________________________________________________________________________________________________________________
        Make predictions using a DataLoader for memory-efficient batch processing.
        ____________________________________________________________________________________________________________________
        Parameters:
            - dataloader (DataLoader) : DataLoader containing batches to predict on
        ____________________________________________________________________________________________________________________
        Returns:
            - predictions (np.ndarray) : Concatenated predictions from all batches
        ____________________________________________________________________________________________________________________
        """
        all_predictions = []
        
        with torch.no_grad():
            
            for batch in dataloader:
                batch_X = batch[0] if isinstance(batch, (list,tuple)) else batch
                batch_X = batch_X.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds = self.model(batch_X)
                    all_predictions.append(preds.detach().cpu())
                    
        return torch.cat(all_predictions, dim=0).numpy()
    
    # [Helper] Training loop for one epoch --------------------------------------------------------------------------------#
    def train_one_epoch(self, train_loader: DataLoader, criterion: torch.nn.Module) -> float:
        """
        ____________________________________________________________________________________________________________________
        Train the model for one epoch.
        ____________________________________________________________________________________________________________________
        Parameters:
            - train_loader (DataLoader) : DataLoader containing training batches
            - criterion    (nn.Module)  : Loss function
        ____________________________________________________________________________________________________________________
        Returns:
            - avg_loss (float) : Average training loss for the epoch
        ____________________________________________________________________________________________________________________
        """
        
        # Set up training mode and counters  
        self.model.train()
        epoch_loss = 0.0
        n_batches  = 0
        
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.opt.zero_grad(set_to_none=True)
            
            # Forward pass (enabled AMP if specified)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(batch_X)
                loss  = criterion(preds, batch_y)

            # Backward pass and optimization
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                
            else:
                loss.backward()
                self.opt.step()
                    
            # Accumulate loss
            epoch_loss += loss.item()
            n_batches  += 1
        
        # Return average loss for the epoch
        avg_loss = epoch_loss / n_batches 
        
        return avg_loss
    
    # [Helper] Validation loop for one epoch ------------------------------------------------------------------------------#
    def val_one_epoch(self, val_loader: DataLoader, criterion: torch.nn.Module) -> float:
        """
        ____________________________________________________________________________________________________________________
        Validate the model for one epoch.
        ____________________________________________________________________________________________________________________
        Parameters:
            - val_loader (DataLoader) : DataLoader containing validation batches
            - criterion  (nn.Module)  : Loss function
        ____________________________________________________________________________________________________________________
        Returns:
            - avg_loss (float) : Average validation loss for the epoch
        ____________________________________________________________________________________________________________________
        """
        self.model.eval()
        epoch_loss = 0.0
        n_batches  = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move data to device (non_blocking for async transfer)
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Accumulate loss
                epoch_loss += loss.item()
                n_batches += 1
        
        # Return average loss for the epoch
        if n_batches == 0:
            raise ValueError("Validation set is empty or DataLoader returned no batches")
        
        avg_loss = epoch_loss / n_batches
        return avg_loss
    
    # [Helper] Get model information --------------------------------------------------------------------------------------#
    def _get_model_info(self) -> Dict[str, Any]:
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
        
        return info

#--------------------------------------------------------------------------------------------------------------------------#