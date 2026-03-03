# Modules -----------------------------------------------------------------------------------------------------------------#
import copy
import logging
import torch
import optuna

import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing           import Any, Dict, List, Callable
from torch.utils.data import DataLoader

# Custom functions and utilities ------------------------------------------------------------------------------------------#
from src.models.dltab.utils.losses import get_loss_function

# [Helper] Validate data format for DL models in SpaceSearch --------------------------------------------------------------#
def validate_data_dl(partitions:List) -> None:
    """Validate input data shapes and types"""
    required_keys = {'train_loader', 'val_loader'}
    
    for i, partition in enumerate(partitions):
        if not all(key in partition for key in required_keys):
            raise ValueError(f"Partition {i} missing required keys: {required_keys}")
    
        train_loader = partition['train_loader']
        val_loader   = partition['val_loader']
        
        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader):
            raise TypeError("train_loader and val_loader must be DataLoader instances") 

# [Helper] Normalize partitions for DL models based in required features for SpaceSearch ----------------------------------#   
def normalize_partitions_dl(partitions: List[Dict]) -> List[Dict]:
    """Normalize partitions to ensure they have all required keys for traditional DL models"""
    normalized = []
    
    for i, partition in enumerate(partitions):
        # Validate required keys
        if 'train_loader' not in partition or 'val_loader' not in partition:
            raise ValueError(f"DL partition {i} must contain 'train_loader' and 'val_loader'")
        
        train_loader = partition['train_loader']
        val_loader   = partition['val_loader']
        
        # Validate DataLoader types
        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader):
            raise TypeError(f"Partition {i}: train_loader and val_loader must be DataLoader instances")
        
        # Extract feature names from DataLoader's dataset if available
        features_names = partition.get('features_names', None)
        if features_names is None:
            # Try to get from dataset metadata
            dataset = train_loader.dataset
            if hasattr(dataset, 'get_feature_names'):
                features_names = dataset.get_feature_names()
            else:
                # Fallback: get from first batch
                try:
                    first_batch    = next(iter(train_loader))
                    X_sample       = first_batch[0]
                    features_names = [f"feature_{j}" for j in range(X_sample.shape[1])]
                except Exception:
                    raise ValueError(f"Cannot infer feature names for partition {i}")
        
        normalized.append({'train_loader'   : train_loader,
                           'val_loader'     : val_loader,
                           'features_names' : features_names,
                           'scaler'         : partition.get('scaler', None)
                           })
    
    return normalized

# [Helper] Evaluate a DL model on a given partition for SpaceSearch -------------------------------------------------------#
def evaluate_partition_dl(model: Any, partition: Dict, scorer: Callable, trial: optuna.trial.Trial,
                          logger : logging.Logger,
                          config : Any,
                          ) -> float:
    """
    ________________________________________________________________________________________________________________________
    Train and evaluate DL model with epoch-wise pruning and best model checkpointing.
    ________________________________________________________________________________________________________________________
    This function implements:
    
    1. Epoch-wise training loop with Optuna pruning integration
    2. Best model state checkpointing (saves model when validation loss improves)
    3. Automatic restoration of best model before final evaluation
    4. Early stopping based on validation loss plateau
    
    The model is evaluated using the BEST checkpoint, not the final epoch state,
    preventing overfitting issues and ensuring optimal performance.
    ________________________________________________________________________________________________________________________
    """
    
    train_loader = partition['train_loader']
    val_loader   = partition['val_loader']
    scaler       = partition.get('scaler', None)
    
    # Get loss function name from config (default to 'huber')
    loss_fn = getattr(config, 'dl_loss_fn', 'huber')
    
    # Build criterion with appropriate parameters
    if loss_fn == 'huber':
        delta     = trial.user_attrs.get('delta', 1.0)
        criterion = get_loss_function(loss_fn, delta=delta).to(model.device)
    else:
        criterion = get_loss_function(loss_fn).to(model.device)
            
    # Training loop with pruning and best model checkpoint
    best_val_loss     = float('inf')
    best_model_state  = None  # Store the best model state
    epochs_no_improve = 0
    
    for epoch in range(config.max_epochs):
        # Train one epoch
        train_loss = model.train_one_epoch(train_loader, criterion)
        
        # Validate one epoch
        val_loss = model.validate_one_epoch(val_loader, criterion)
        
        # Track best validation loss AND save model state
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            # Save a deep copy of the best model state
            best_model_state = copy.deepcopy(model.model.state_dict())
            if config.verbose:
                logger.debug(f"Trial {trial.number}, Epoch {epoch}: New best val_loss = {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
        
        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping (internal)
        if config.dl_patience is not None and epochs_no_improve >= config.dl_patience:
            if config.verbose:
                logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
            break
    
    # Restore the BEST model state before evaluation
    if best_model_state is not None:
        model.model.load_state_dict(best_model_state)
        if config.verbose:
            logger.debug(f"Trial {trial.number}: Loaded best model state (val_loss={best_val_loss:.6f})")
    else:
        if config.verbose:
            logger.warning(f"Trial {trial.number}: No best model state found, using final epoch state")
    
    # Mark model as fitted so predict() works
    model.is_fitted = True
    
    # Final evaluation with scorer metric
    y_pred = model.predict(val_loader)
    
    # Extract targets from val_loader
    all_targets = []
    for batch_X, batch_y in val_loader:
        all_targets.append(batch_y.cpu())
    y_val = torch.cat(all_targets, dim=0).numpy()
    
    # Ensure arrays are properly flattened to 1D to prevent pairwise distance computation
    y_pred = y_pred.ravel()
    y_val = y_val.ravel()
    
    # Apply scaler if provided (unified logic with _ml.py for consistency)
    if scaler is not None:
        # Check if scaler is an object with inverse_transform method (e.g., TargetLogScaler)
        if hasattr(scaler, 'inverse_transform') and callable(getattr(scaler, 'inverse_transform')):
            y_pred = scaler.inverse_transform(y_pred)
            y_val  = scaler.inverse_transform(y_val)
        # Or if its just a numeric factor (e.g., standard deviation for scaling)
        else:
            y_pred = y_pred * scaler
            y_val  = y_val  * scaler
    
    # Compute final score with user's metric (use positional args for sklearn compatibility)
    score = float(scorer(y_val, y_pred))
    
    return score
#---------------------------------------------------------------------------------------------------------------------------#