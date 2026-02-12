# Modules -----------------------------------------------------------------------------------------------------------------#
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
    Trainer: Handles the training loop for deep learning models
    ________________________________________________________________________________________________________________________
    Responsibilities:
    -> Training one epoch
    -> Validation one epoch
    -> Full training loop with early stopping
    -> Learning rate scheduling
    -> Checkpoint management coordination
    ________________________________________________________________________________________________________________________
    """
    
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
                 use_amp   : bool                                            = False,
                 scheduler : Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 scaler    : Optional[torch.cuda.amp.GradScaler]             = None,
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
        -> verbose   (bool)                 : Enable verbose logging
        ____________________________________________________________________________________________________________________
        """
        self.model     = model
        self.optimizer = optimizer
        self.device    = device
        self.use_amp   = use_amp
        self.scheduler = scheduler
        self.scaler    = scaler
        self.verbose   = verbose
        
        # Training state
        self.history          = {'train_loss': [], 'val_loss': [], 'epochs': []}
        self.best_model_state = None
        self.best_val_loss    = float('inf')
    
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
        self.model.train()
        epoch_loss = 0.0
        n_batches  = 0
        
        for batch in train_loader:
            # Consistent unpacking
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Expected batch to contain (X, y)")
                batch_X, batch_y = batch[0], batch[1]
            else:
                raise ValueError("Expected batch to be a tuple/list of (X, y)")
            
            # Move data to device
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with optional AMP
            loss = self._forward_pass(batch_X, batch_y, criterion)
            
            # Backward pass
            self._backward_pass(loss)
            
            # Accumulate loss
            epoch_loss += loss.item()
            n_batches  += 1
        
        # Return average loss
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
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
        if val_loader is None:
            raise ValueError("val_loader cannot be None")
        
        self.model.eval()
        epoch_loss = 0.0
        n_batches  = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Consistent unpacking
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError("Expected batch to contain (X, y)")
                    batch_X, batch_y = batch[0], batch[1]
                else:
                    raise ValueError("Expected batch to be a tuple/list of (X, y)")
                
                # Move data to device
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass with autocast for consistency
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                
                # Accumulate loss
                epoch_loss += loss.item()
                n_batches  += 1
        
        if n_batches == 0:
            raise ValueError("Validation set is empty or DataLoader returned no batches")
        
        avg_loss = epoch_loss / n_batches
        return avg_loss
    
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
            
            # Handle early stopping and best model tracking
            is_best = False
            if early_stopper is not None and val_loss is not None:
                stop, is_best = early_stopper.step(val_loss)
                
                if is_best:
                    self.best_val_loss = val_loss
                    # Store best model state in memory
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
                    if self.verbose:
                        logger.info(f"New best model: val_loss = {self.best_val_loss:.6f}")
                
                if stop:
                    should_stop = True
                    if self.verbose:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        logger.info(f"No improvement for {early_stopping_patience} epochs")
            
            # Call checkpoint callback if provided
            if checkpoint_callback is not None:
                checkpoint_callback(epoch=epoch + 1, is_best=is_best, val_loss=val_loss)
            
            # Break if early stopping triggered
            if should_stop:
                break
            
            # Update learning rate scheduler
            self._update_scheduler(val_loss if val_loss is not None else train_loss)
            
            # Verbose logging
            if self.verbose and (epoch + 1) % verbose_epoch == 0:
                msg = f"Epoch [{epoch + 1:4d}/{epochs}] - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
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
        
        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }
    
    def _forward_pass(self, batch_X: torch.Tensor, batch_y: torch.Tensor, 
                      criterion: torch.nn.Module) -> torch.Tensor:
        """Forward pass with optional AMP."""
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            preds = self.model(batch_X)
            loss  = criterion(preds, batch_y)
        return loss
    
    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Backward pass with optional gradient scaling."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
    
    def _update_scheduler(self, metric: float) -> None:
        """Update learning rate scheduler if available."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
    
    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history.copy()
    
    def get_best_model_state(self) -> Optional[Dict]:
        """Get best model state dict."""
        return self.best_model_state

#--------------------------------------------------------------------------------------------------------------------------#

    
    
    
    
    
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
            
            # Validate one epoch (only if val_loader provided)
            val_loss = None
            if val_loader is not None:
                val_loss = self.val_one_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
            
            history['epochs'].append(epoch + 1)
            
            # Update history before any checkpoint saving
            self.history = history
            
            # Handle checkpointing and early stopping
            if early_stopper is not None and val_loss is not None:
                stop, is_best = early_stopper.step(val_loss)
                
                if is_best:
                    self.is_fitted = True
                    best_val_loss  = val_loss
                    
                    # Store best model state in memory
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    
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
        
        # Load best model if we have it stored
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                logger.info("Loaded best model weights from training")
        
        if self.verbose:
            logger.success("Training completed!")
            logger.success(f"Final train loss: {history['train_loss'][-1]:.6f}")
            if history['val_loss'] and len(history['val_loss']) > 0:
                logger.success(f"Final val loss : {history['val_loss'][-1]:.6f}")
                logger.success(f"Best val loss  : {best_val_loss:.6f}")
        
        return self
    
    
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
        
        for batch in train_loader:
            # Consistent unpacking
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Expected batch to contain (X, y)")
                batch_X, batch_y = batch[0], batch[1]
            else:
                raise ValueError("Expected batch to be a tuple/list of (X, y)")
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
        if val_loader is None:
            raise ValueError("val_loader cannot be None")
        
        self.model.eval()
        epoch_loss = 0.0
        n_batches  = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Consistent unpacking
                if isinstance(batch, (list, tuple)):
                    if len(batch) < 2:
                        raise ValueError("Expected batch to contain (X, y)")
                    batch_X, batch_y = batch[0], batch[1]
                else:
                    raise ValueError("Expected batch to be a tuple/list of (X, y)")
                # Move data to device (non_blocking for async transfer)
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass with autocast for consistency
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
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
