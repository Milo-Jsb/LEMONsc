# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy as np
import copy
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from torch.utils.data import DataLoader
from typing           import Optional, Dict, Any
from loguru           import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.dltab.utils.losses    import get_loss_function
from src.models.dltab.utils.callbacks import EarlyStopping

# Constants ---------------------------------------------------------------------------------------------------------------#
DEFAULT_VERBOSE_EPOCH = 10

# Trainer Class -----------------------------------------------------------------------------------------------------------#
class Trainer:
    """
    ________________________________________________________________________________________________________________________
    Trainer: Handles the training and validation loop for deep learning models
    ________________________________________________________________________________________________________________________
    The training supports:
    -> Standard training loop with separate train and validation phases
    -> Early stopping based on validation loss
    -> Learning rate scheduling
    -> Gradient clipping for stable training
    -> Automatic Mixed Precision (AMP) for faster training on compatible hardware
    -> Checkpointing via callback function
    ________________________________________________________________________________________________________________________
    """
    
    # Initialize the object -----------------------------------------------------------------------------------------------#
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
                 use_amp   : bool                                            = False,
                 scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 scaler    : Optional[torch.cuda.amp.GradScaler]             = None,
                 grad_clip : Optional[float]                                 = None,
                 verbose   : bool                                            = False):
        """
        ____________________________________________________________________________________________________________________
        Initialize the Trainer.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> model     (torch.nn.Module)      : Model to train
        -> optimizer (torch.optim.Optimizer): Optimizer instance
        -> device    (torch.device)         : Device to use for training
        -> use_amp   (bool)                 : Whether to use Automatic Mixed Precision
        -> scheduler (LRScheduler)          : Optional learning rate scheduler
        -> scaler    (GradScaler)           : Optional gradient scaler for AMP
        -> grad_clip (float)                : Optional max norm for gradient clipping. None disables clipping.
                                              Recommended: 1.0 for NODE, None for simpler models (MLP, etc.)
        -> verbose   (bool)                 : Enable verbose logging
        ____________________________________________________________________________________________________________________
        """
        # Set attributes
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.use_amp   = use_amp
        self.scheduler = scheduler
        self.scaler    = scaler
        self.grad_clip = grad_clip
        self.verbose   = verbose
        
        # Training state
        self.history          = {'train_loss': [], 'val_loss': [], 'epochs': []}
        self.best_model_state = None
        self.best_val_loss    = float('inf')
        
    # [Helper] One epoch of training operations ---------------------------------------------------------------------------#
    def train_one_epoch(self, train_loader: DataLoader, criterion: torch.nn.Module) -> float:
        """
        ____________________________________________________________________________________________________________________
        Train the model for one epoch.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> train_loader (DataLoader) : DataLoader containing training batches
        -> criterion    (nn.Module)  : Loss function
        ____________________________________________________________________________________________________________________
        Returns:
        -> avg_loss (float) : Average training loss for the epoch
        ____________________________________________________________________________________________________________________
        """
        
        # Set model mode, epoch loss variable, and batch counter for averaging
        self.model.train()
        epoch_loss = 0.0
        n_batches  = 0
        
        # Iterate in the batches of the training DataLoader
        for batch in train_loader:
            
            # Consistent unpacking
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Expected batch to contain (X, y)")
                batch_X, batch_y = batch[0], batch[1]
            else:
                raise ValueError("Expected batch to be a tuple/list of (X, y)")
            
            # Move data to device
            batch_X = batch_X.to(self.device, dtype=torch.float32, non_blocking=True)
            batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional AMP (Get predictions and compute loss)
            loss = self._forward_pass(batch_X, batch_y, criterion)
            
            # Backward pass (backpropagation with optional gradient scaling and clipping, and optimizer step)
            self._backward_pass(loss)
            
            # Accumulate loss
            epoch_loss += loss.item()
            n_batches  += 1
        
        # Return average loss
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        
        return avg_loss
    
    # [Helper] One epoch of validation operations -------------------------------------------------------------------------#
    def validate_one_epoch(self, val_loader: DataLoader, criterion: torch.nn.Module) -> float:
        """
        ____________________________________________________________________________________________________________________
        Validate the model for one epoch.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> val_loader (DataLoader) : DataLoader containing validation batches
        -> criterion  (nn.Module)  : Loss function
        ____________________________________________________________________________________________________________________
        Returns:
        -> avg_loss (float) : Average validation loss for the epoch
        ____________________________________________________________________________________________________________________
        """
        # Input validation
        if val_loader is None:
            raise ValueError("val_loader cannot be None")
        
        # Set model to evaluation mode, epoch loss variable, and batch counter for averaging
        self.model.eval()
        epoch_loss = 0.0
        n_batches  = 0
        
        # Deactivate autograd for validation to save memory and computations
        with torch.no_grad():
            
            # Iterate over the batches of the validation DataLoader
            for batch in val_loader:
                
                # Consistent unpacking
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError("Expected batch to contain (X, y)")
                    batch_X, batch_y = batch[0], batch[1]
                else:
                    raise ValueError("Expected batch to be a tuple/list of (X, y)")
                
                # Move data to device
                batch_X = batch_X.to(self.device, dtype=torch.float32, non_blocking=True)
                batch_y = batch_y.to(self.device, dtype=torch.float32, non_blocking=True)
                
                # Forward pass with autocast for consistency
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    predictions = self.model(batch_X)
                    
                    # Squeeze predictions to match target shape for regression
                    if predictions.dim() > 1 and predictions.size(-1) == 1:
                        predictions = predictions.squeeze(-1)
                    
                    loss = criterion(predictions, batch_y)
                
                # Accumulate loss
                epoch_loss += loss.item()
                n_batches  += 1
        
        # Simple check to avoid division by zero and provide feedback if validation set is empty
        if n_batches == 0:
            raise ValueError("Validation set is empty or DataLoader returned no batches")
        
        # Compute the average loss for the epoch
        avg_loss = epoch_loss / n_batches
        
        return avg_loss
    
    # Full training loop with early stopping and checkpointing ------------------------------------------------------------#
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 100, 
              loss_fn                 : str                      = 'mse',
              loss_params             : Optional[Dict[str, Any]] = None,
              early_stopping_patience : Optional[int]            = None,
              verbose_epoch           : int                      = DEFAULT_VERBOSE_EPOCH,
              checkpoint_callback     : Optional[callable]       = None
              ) -> Dict[str, Any]:
        """
        ____________________________________________________________________________________________________________________
        Full training loop with early stopping and checkpointing.
        ____________________________________________________________________________________________________________________
        Parameters:
        -> train_loader             (DataLoader) : DataLoader containing training batches
        -> val_loader               (DataLoader) : Optional validation DataLoader
        -> epochs                   (int)        : Number of training epochs
        -> loss_fn                  (str)        : Loss function name
        -> loss_params              (dict)       : Optional loss function parameters
        -> early_stopping_patience  (int)        : Epochs to wait before stopping
        -> verbose_epoch            (int)        : Print info every N epochs
        -> checkpoint_callback      (callable)   : Callback function for saving checkpoints
                                                   Signature: callback(epoch, is_best, val_loss)
        ____________________________________________________________________________________________________________________
        Returns:
        -> dict : Training history and results
        ____________________________________________________________________________________________________________________
        """
        # Setup loss function
        criterion = get_loss_function(loss_fn, **(loss_params or {})).to(self.device)
        
        # Validate early stopping configuration
        if early_stopping_patience is not None and val_loader is None:
            raise ValueError("val_loader must be provided when early_stopping_patience is set")
        
        # Initialize early stopping callback
        early_stopper = None
        if early_stopping_patience is not None:
            early_stopper = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        # Reset training state
        self.history          = {'train_loss': [], 'val_loss': [], 'epochs': []}
        self.best_val_loss    = float('inf')
        self.best_model_state = None
        should_stop           = False
        
        # Training info
        if self.verbose:
            logger.info(f"Starting training for {epochs} epochs")
            logger.info(f"Loss function : {loss_fn}")
            logger.info(f"Device        : {self.device}")
        
        # Training loop
        for epoch in range(epochs):
            # Train one epoch
            train_loss = self.train_one_epoch(train_loader, criterion)
            self.history['train_loss'].append(train_loss)
            
            # Validate one epoch (if val_loader provided)
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_one_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
            
            self.history['epochs'].append(epoch + 1)
            
            # Track best model (works with or without early stopping)
            is_best = False
            if val_loss is not None and val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss    = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                
                if self.verbose:
                    logger.info(f"New best model: val_loss = {self.best_val_loss:.6f}")
            
            # Handle early stopping (only if configured)
            if early_stopper is not None and val_loss is not None:
                stop, _ = early_stopper.step(val_loss)
                
                if stop:
                    should_stop = True
                    if self.verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        logger.info(f"No improvement for {early_stopping_patience} epochs")
            
            # Verbose logging (before potential break to log last epoch)
            if self.verbose and (epoch + 1) % verbose_epoch == 0:
                msg = f"Epoch [{epoch + 1:4d}/{epochs}] - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)
            
            # Call checkpoint callback if provided
            if checkpoint_callback is not None:
                checkpoint_callback(epoch=epoch + 1, is_best=is_best, val_loss=val_loss)
            
            # Break if early stopping triggered
            if should_stop:
                break
            
            # Update learning rate scheduler
            self._update_scheduler(val_loss if val_loss is not None else train_loss)
        
        # Load best model if available
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                logger.info("Loaded best model weights from training")
        
        # Final logging
        if self.verbose:
            logger.success("Training completed!")
            logger.success(f"Final train loss: {self.history['train_loss'][-1]:.6f}")
            if self.history['val_loss'] and len(self.history['val_loss']) > 0:
                logger.success(f"Final val loss : {self.history['val_loss'][-1]:.6f}")
                logger.success(f"Best val loss  : {self.best_val_loss:.6f}")
        
        # Result dictionary containing training history and best model info
        results = {
            'history'          : self.history,
            'best_val_loss'    : self.best_val_loss,
            'best_model_state' : self.best_model_state
                  }
        
        return results
    
    # [Helper] Forward pass and loss computation with optional AMP --------------------------------------------------------#
    def _forward_pass(self, batch_X: torch.Tensor, batch_y: torch.Tensor, 
                      criterion: torch.nn.Module) -> torch.Tensor:
        """Forward pass with optional AMP."""
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            preds = self.model(batch_X)
            if preds.dim() > 1 and preds.size(-1) == 1:
                preds = preds.squeeze(-1)
            loss  = criterion(preds, batch_y)
        return loss
    
    # [Helper] Backward pass with optional gradient scaling and clipping --------------------------------------------------#
    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Backward pass with optional gradient scaling and clipping."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
    
    # [Helper] Update learning rate scheduler if available ----------------------------------------------------------------#
    def _update_scheduler(self, metric: float) -> None:
        """Update learning rate scheduler if available."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    # Getters for training history and best model state ------------------------------------------------------------------#
    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history.copy()
    
    def get_best_model_state(self) -> Optional[Dict]:
        """Get best model state dict."""
        return self.best_model_state

#--------------------------------------------------------------------------------------------------------------------------#
