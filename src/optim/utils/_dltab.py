# Modules -----------------------------------------------------------------------------------------------------------------#
import logging
import torch
import optuna


# External functions and utilities ----------------------------------------------------------------------------------------#
from typing           import Any, Dict, List, Callable
from torch.utils.data import DataLoader

# Custom functions and utilities ------------------------------------------------------------------------------------------#
from src.models.dltab.utils.losses import get_loss_function

# [Helper] Validate data format for DL models in SpaceSearch --------------------------------------------------------------#
def validate_data_dl(partitions:List) -> None:
    """Validate input data shapes and types"""
    
    # Set of required keys for each partition
    required_keys = {'train_loader', 'val_loader'}
    
    # Loop over the list of partitions and validate each one
    for i, partition in enumerate(partitions):
        
        # Raise error if any required key is missing
        if not all(key in partition for key in required_keys):
            raise ValueError(f"Partition {i} missing required keys: {required_keys}")

        # Retrieve keys for validation
        train_loader = partition['train_loader']
        val_loader   = partition['val_loader']
        
        # Check correct types for train_loader and val_loader
        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader):
            raise TypeError("train_loader and val_loader must be DataLoader instances") 

# [Helper] Normalize partitions for DL models based in required features for SpaceSearch ----------------------------------#   
def normalize_partitions_dl(partitions: List[Dict]) -> List[Dict]:
    """Normalize partitions to ensure they have all required keys for traditional DL models"""
    
    # List for storage of results
    normalized = []
    
    # Loop over partitions and ensure they have all required keys, filling in defaults where necessary
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
            
            # Fallback: get from first batch if possible (assumes features are in the first element of the batch)
            else:
                try:
                    first_batch    = next(iter(train_loader))
                    X_sample       = first_batch[0]
                    features_names = [f"feature_{j}" for j in range(X_sample.shape[1])]
                except Exception:
                    raise ValueError(f"Cannot infer feature names for partition {i}")
        
        normalized.append({'train_loader'   : train_loader,
                           'val_loader'     : val_loader,
                           'features_names' : features_names,
                           'trs'            : partition.get('trs', None),
                           'scaler'         : partition.get('scaler', None)
                           })
    
    return normalized

# [Helper] Evaluate a DL model on a given partition for SpaceSearch -------------------------------------------------------#
def evaluate_partition_dl(model: Any, partition: Dict, scorer: Callable, trial: optuna.trial.Trial,
                          logger         : logging.Logger,
                          config         : Any,
                          fold_idx       : int  = 0,
                          enable_pruning : bool = False
                          ) -> float:
    """
    ________________________________________________________________________________________________________________________
    Train and evaluate DL model with best model checkpointing and optional epoch-level Optuna pruning.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> model          : The DL model instance to train and evaluate 
                        (must have train_one_epoch, validate_one_epoch, predict methods)
    -> partition      : A dict containing 'train_loader', 'val_loader', and optionally 'scaler'
    -> scorer         : A callable metric function that takes (y_true, y_pred) and returns a scalar score
    -> trial          : The Optuna trial object for reporting and pruning
    -> logger         : Logger for debug/info messages
    -> config         : Configuration object containing training parameters 
    -> fold_idx       : Index of the partition used for global step offset.
    -> enable_pruning : If True, report val_loss to Optuna each epoch and allow epoch-level pruning.
                        If False, only use internal early stopping (no Optuna pruning within this fold).
    ________________________________________________________________________________________________________________________
    Returns:
    -> score          : The final evaluation score computed by the scorer on the validation set after training
    ________________________________________________________________________________________________________________________
    Notes:
        -> When enable_pruning=True, each epoch reports to the pruner using a global step:
                global_step = fold_idx * max_epochs + epoch
           This ensures the pruner compares equivalent (fold, epoch) positions across trials.
        
        -> When enable_pruning=False, training relies solely on internal early stopping (dl_patience).
           This is used for folds > 0 in hybrid pruning mode, where fold-level pruning is handled
           by the caller (_create_objective) instead.
    ________________________________________________________________________________________________________________________
    """
    
    # Get elements of the partition
    train_loader = partition['train_loader']
    val_loader   = partition['val_loader']
    trs          = partition.get('trs', None)
    scaler       = partition.get('scaler', None)
    
    # Get loss function name from config (default to 'huber')
    loss_fn = getattr(config, 'dl_loss_fn', 'huber')
    
    # Build criterion with appropriate parameters
    if loss_fn == 'huber':
        delta     = trial.user_attrs.get('delta', 1.0)
        criterion = get_loss_function(loss_fn, delta=delta).to(model.device)
    else:
        criterion = get_loss_function(loss_fn).to(model.device)
            
    # Training loop with pruning and best model checkpoint. Store the best model state
    best_val_loss     = float('inf')
    best_model_state  = None  
    epochs_no_improve = 0
    
    # Iterate over epochs
    for epoch in range(config.max_epochs):
        
        # Train and Validate one epoch
        train_loss = model.train_one_epoch(train_loader, criterion, grad_clip_norm=config.dl_grad_clip)
        val_loss   = model.validate_one_epoch(val_loader, criterion)
        
        # Track best validation loss and  save model state
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            
            # Clone tensors from the best model state (avoids heavy deepcopy overhead)
            best_model_state = {k: v.clone() for k, v in model.model.state_dict().items()}
            
            # Verbose logging of new best score
            if config.verbose:
                logger.debug(f"Trial {trial.number}, Epoch {epoch}: New best val_loss = {best_val_loss:.6f}")
        
        # Else, add a counter
        else:
            epochs_no_improve += 1
        
        # Report intermediate values for pruning (always before early-stopping check so the
        if enable_pruning:
            global_step = fold_idx * config.max_epochs + epoch
            trial.report(val_loss, global_step)
        
        # Early stopping checked before should_prune() to avoid pruning a converged trial
        if config.dl_patience is not None and epochs_no_improve >= config.dl_patience:
            if config.verbose:
                logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
            break
        
        if enable_pruning and trial.should_prune():
            raise optuna.TrialPruned()
    
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
    y_val  = y_val.ravel()
    
    # Apply scaler if provided 
    if scaler is not None:
        
        # Check if scaler is an object with inverse_transform method (e.g., FeatureScaler)
        if hasattr(scaler, 'inverse_transform') and callable(getattr(scaler, 'inverse_transform')):
            y_pred = scaler.inverse_transform(y_pred)
            y_val  = scaler.inverse_transform(y_val)
        else:
            raise ValueError("Scaler provided in partition does not have an inverse_transform method")
    
    # Apply trs if provided 
    if trs is not None:

        # Check if trs is an object with inverse_transform method (e.g., TargetTransform)
        if hasattr(trs, 'inverse_transform') and callable(getattr(trs, 'inverse_transform')):
            y_pred = trs.inverse_transform(y_pred)
            y_val  = trs.inverse_transform(y_val)
        else:
            raise ValueError("trs provided in partition does not have an inverse_transform method")        # Compute final score with user's metric (use positional args for sklearn compatibility)
    
    score = float(scorer(y_val, y_pred))
    
    return score

#---------------------------------------------------------------------------------------------------------------------------#