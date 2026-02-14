# Modules -----------------------------------------------------------------------------------------------------------------#
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
    """Train and evaluate DL model with epoch-wise pruning to prevent state leakage."""
    
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
            
    # Training loop with pruning
    best_val_loss     = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(config.max_epochs):
        # Train one epoch
        train_loss = model.train_one_epoch(train_loader, criterion)
        
        # Validate one epoch
        val_loss = model.validate_one_epoch(val_loader, criterion)
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
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
                logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch}")
            break
    
    # Mark model as fitted so predict() works
    model.is_fitted = True
    
    # Final evaluation with scorer metric
    y_pred = model.predict(val_loader)
    
    # Extract targets from val_loader
    all_targets = []
    for batch_X, batch_y in val_loader:
        all_targets.append(batch_y.cpu())
    y_val = torch.cat(all_targets, dim=0).squeeze().numpy()
    
    # Apply scaler if provided
    if scaler is not None:
        y_pred = y_pred * scaler
        y_val  = y_val  * scaler
    
    # Compute final score with user's metric
    score = float(scorer(y_true=y_val, y_pred=y_pred))
    
    return score
#---------------------------------------------------------------------------------------------------------------------------#