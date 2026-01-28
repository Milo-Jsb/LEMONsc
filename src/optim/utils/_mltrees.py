# Modules -----------------------------------------------------------------------------------------------------------------#
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing  import Dict, List, Callable, Optional

# Custom functions and utilities ------------------------------------------------------------------------------------------#
from src.models.mltrees.regressor import MLTreeRegressor

# [Helper] Validate data format for ML models in SpaceSearch --------------------------------------------------------------#
def validate_data_ml(partitions : List) -> None:
        """Validate input data shapes and types"""
        
        required_keys = {'X_train', 'y_train', 'X_val', 'y_val'}
        
        for i, partition in enumerate(partitions):
            if not all(key in partition for key in required_keys):
                raise ValueError(f"Partition {i} missing required keys: {required_keys}")
        
            X_train = partition['X_train']
            y_train = partition['y_train']
            X_val   = partition['X_val']
            y_val   = partition['y_val']
        
            # Check case scenario for MLTreesRegressor 
            if (len(X_train) != len(y_train)) or (len(X_val) != len(y_val)):
                raise ValueError("Features and target sizes don't match")
            
            if isinstance(X_train, pd.DataFrame) != isinstance(X_val, pd.DataFrame):
                raise ValueError("X_train and X_val must be the same type")
                
            if isinstance(y_train, pd.Series) != isinstance(y_val, pd.Series):
                raise ValueError("y_train and y_val must be the same type")
        
# [Helper] Normalize partitions for ML models based in required features for SpaceSearch ----------------------------------#
def normalize_partitions_ml(partitions: List[Dict]) -> List[Dict]:
    """Normalize partitions to ensure they have all required keys for traditional ML models"""
    normalized = []
    for i, partition in enumerate(partitions):
        # Get existing or infer features_names
        if 'features_names' not in partition and 'feats' not in partition:
            X_train = partition['X_train']
            if isinstance(X_train, pd.DataFrame):
                features_names = X_train.columns.tolist()
            else:
                features_names = [f"feature_{j}" for j in range(X_train.shape[1])]
        else:
            features_names = partition.get('features_names', partition.get('feats', []))
        
        normalized.append({
            'X_train'        : partition['X_train'],
            'y_train'        : partition['y_train'],
            'X_val'          : partition['X_val'],
            'y_val'          : partition['y_val'],
            'features_names' : features_names,
            'scaler'         : partition.get('scaler', None)
        })
    return normalized
 
# [Helper] Evaluate a ML model on a given partition for SpaceSearch -------------------------------------------------------#
def evaluate_partition_ml(model: MLTreeRegressor, partition: Dict, scorer: Callable) -> float:
    """Train and evaluate model on a single partition, applying scaler if provided else return unscaled results"""
    X_train = partition['X_train']
    y_train = partition['y_train']
    X_val   = partition['X_val']
    y_val   = partition['y_val']
    scaler  = partition.get('scaler', None)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Apply scaling if provided
    if scaler is not None:
        y_pred       = y_pred * scaler
        y_val_scaled = y_val  * scaler
        score = float(scorer(y_true=y_val_scaled, y_pred=y_pred))
    else:
        score = float(scorer(y_true=y_val, y_pred=y_pred))
    
    return score

#---------------------------------------------------------------------------------------------------------------------------#