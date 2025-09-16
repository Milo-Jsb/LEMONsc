# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse
import yaml
import torch 

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru          import logger
from pathlib         import Path
from dataclasses     import dataclass
from datetime        import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.format        import tabular_features
from src.optim.optimizer          import SpaceSearch, SpaceSearchConfig
from src.utils.directory          import load_yaml_dict
from src.utils.visualize          import correlation_plot, residual_plot
from src.models.mltrees.regressor import MLTreeRegressor

# Warnings managment ------------------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/mltrees_execution.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Configuration -----------------------------------------------------------------------------------------------------------#
@dataclass
class TrainingConfig:
    """Configuration class for the execution of mltrees_training() script."""
    n_folds        : int   = 3
    n_trials       : int   = 100
    n_jobs         : int   = 20
    device         : str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed           : int   = 42
    direction      : str   = "minimize"
    metric         : str   = "neg_mean_absolute_error"
    patience       : int   = 20
    lambda_penalty : float = 0.0

CONFIG = TrainingConfig()

# Arguments ---------------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Execution of MLTrees models")
    
    # Main mode of the script
    parser.add_argument("--mode", type=str, default="optim",
                        choices=["optim", "train", "predict"],
                        help="Pipeline stage to implement.")

    # Directories
    parser.add_argument("--root_dir", type=str, default="./datasets/", 
                        help="Root directory of the data features for the model.")
    parser.add_argument("--out_dir", type=str, default="./output/", 
                        help="Directory to store the output of the optimization and training.")
    parser.add_argument("--fig_dir", type=str, default="./figures/", 
                        help="Directory to store output figures of the analysis.")

    # Dataset specifics
    parser.add_argument("--dataset", type=str, default="moccasurvey", 
                        choices = ["moccasurvey"],
                        help    = "Specific dataset to implement.")
    parser.add_argument("--exp_name", type=str, default="pof",
                        help = "Tag to name the dataset and output related elements.")
    
    # Model specifics
    parser.add_argument("--model", type=str, default="lightgbm",
                       choices=["lightgbm", "xgboost", "random_forest", "dartboost"],
                       help="Type of model to use")

    return parser.parse_args()

# Path Management ---------------------------------------------------------------------------------------------------------#
class PathManager:
    """Centralized path management for the pipeline."""
    
    def __init__(self, root_dir: str, dataset: str, exp_name: str, model: str, out_dir: str, fig_dir: str):
        # Data paths
        self.data_path = Path(root_dir) / exp_name / dataset
        
        # Output structure
        self.base_out   = Path(out_dir) / exp_name / dataset / model
        self.optim_path = self.base_out / "optim"
        self.model_path = self.base_out
        self.fig_path   = Path(fig_dir) / exp_name / dataset / model
        
        # Create directories
        self.optim_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.fig_path.mkdir(parents=True, exist_ok=True)
    
# Pipeline Modes -----------------------------------------------------------------------------------------------------------#
def run_optimization(feats_path: str, contfeats: list, catfeats: list, target: list,  out_path: str, n_folds: int = 3):
    """Run the the optimization mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Space search of the hyperparameters for the MLTree {args.model} regressor using {n_folds}-fold cross-validation")
    logger.info(110*"_")

    # Load and prepare data partitions
    logger.info("Loading and preparing data partitions...")
    partitions = []

    for fold in range(n_folds):
        fold_path  = feats_path / f"{fold}_fold"
        train_path = fold_path / "train.csv"
        val_path   = fold_path / "val.csv"

        # Load fold data
        train_df = pd.read_csv(train_path, index_col=False)
        val_df   = pd.read_csv(val_path, index_col=False)

        # Extract features and target
        feature_names  = contfeats + catfeats + target
        
        # Training
        feats_train, feats_names = tabular_features(train_df, names=feature_names, return_names=True)

        # Identify continuous and categorical and target columns   
        cont_columns   = [col for col in feats_train.columns if col in contfeats]
        cat_columns    = [col for col in feats_train.columns if any(col.startswith(cf+"_") for cf in catfeats)]
        target_columns = [col for col in feats_train.columns if col in target]
        
        # Extract numpy arrays
        X_train    = feats_train[cont_columns+cat_columns].astype(np.float32).to_numpy()
        y_train    = feats_train[target_columns].astype(np.float32).to_numpy().flatten()  
        
        # Validation
        feats_val, _ = tabular_features(val_df, names=feature_names, return_names=True)

        X_val    = feats_val[cont_columns+cat_columns].astype(np.float32).to_numpy()
        y_val    = feats_val[target_columns].astype(np.float32).to_numpy().flatten()  

        # Create partition dictionary with optional scaler, exclude target from features names
        partition = {
                    'X_train': X_train, 'y_train': y_train,
                    'X_val'  : X_val,   'y_val': y_val,
                    'scaler' : val_df["M_tot"].astype(np.float32).to_numpy().flatten() if "M_tot" in val_df.columns else None,
                    'feats'  : cont_columns + cat_columns,
                    'target' : target_columns 
                    }
        partitions.append(partition)
        
        logger.info(f"Fold {fold + 1}/{n_folds} loaded:")
        logger.info(f"  - Training features (Xtrain type={type(X_train)}): {len(X_train)} [{np.shape(X_train)}]")
        logger.info(f"  - Training targets  (ytrain type={type(y_train)}): {len(y_train)} [{np.shape(y_train)}]")
        logger.info(f"  - Validation features (Xval type={type(X_val)}): {len(X_val)} [{np.shape(X_val)}]")
        logger.info(f"  - Validation targets  (yval type={type(y_val)}): {len(y_val)} [{np.shape(y_val)}]")

    # Log feature information
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {contfeats}")
    logger.info(f"  - Categorical features : {catfeats} -> {cat_columns}")
    logger.info(f"  - Target               : {target}")

    # Initialize optimizer with configuration
    config    = SpaceSearchConfig(model_type = args.model, 
                                  n_jobs     = CONFIG.n_jobs, 
                                  n_trials   = CONFIG.n_trials, 
                                  device     = CONFIG.device, 
                                  seed       = CONFIG.seed)
    optimizer = SpaceSearch(config)
    
    # Create study name with timestamp
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"cv_study_{n_folds}fold_{timestamp}"
    
    # Run optimization with cross-validation
    logger.info(f"Starting optimization study: {study_name}")
    logger.info(f"  - Device           : {CONFIG.device}")
    logger.info(f"  - Number of trials : {CONFIG.n_trials}")
    logger.info(f"  - Direction        : {CONFIG.direction}")
    logger.info(f"  - Metric           : {CONFIG.metric}")
    
    try:
        # Update output path to use optim subdirectory
        study_path = Path(out_path) / "optim" 
        
        results = optimizer.run_cv_study(partitions     = partitions,
                                         study_name     = study_name,
                                         direction      = CONFIG.direction,
                                         metric         = CONFIG.metric,
                                         output_dir     = str(study_path),
                                         save_study     = True,
                                         patience       = CONFIG.patience,
                                         lambda_penalty = CONFIG.lambda_penalty)
        
        # Log results
        logger.info(110*"_")
        logger.info("Optimization completed successfully!")
        logger.info(f"Best score: {results.best_score:.6f}")
        logger.info("Best parameters:")
        for param, value in results.best_params.items():
            logger.info(f"  - {param}: {value}")
        
        # Save optimization summary
        summary_path = study_path / study_name / "optimization_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump({
                'study_name'  : study_name,
                'n_folds'     : n_folds,
                'n_trials'    : results.n_trials,
                'best_score'  : float(results.best_score),
                'best_params' : results.best_params,
                'features': {
                    'continuous'  : cont_columns,
                    'categorical' : cat_columns,
                    'target'      : target_columns[0]},
                'config': {
                    'direction'      : CONFIG.direction,
                    'metric'         : CONFIG.metric,
                    'lambda_penalty' : CONFIG.lambda_penalty,
                    'device'         : CONFIG.device,
                    'n_jobs'         : CONFIG.n_jobs
                }
            }, f, default_flow_style=False)
        
        logger.info(f"Results saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise
        
    return results

def run_training(feats_path: str, contfeats: list, catfeats: list, target: list, out_path: str, n_folds: int = 3):
    """Run the the training mode pipeline."""
    
    try:
        # Find latest optimization study
        optimization_path = Path(out_path)  / "optim"
        study_dirs        = [d for d in optimization_path.glob("cv_study_*fold_*") if d.is_dir()]
        
        if study_dirs:
            latest_study = max(study_dirs, key=lambda x: x.stat().st_mtime)
            opt_summary  = load_yaml_dict(f"{latest_study}/optimization_summary.yaml")
            model_params = opt_summary['best_params']
            logger.info(f"Using optimized parameters from: {latest_study.name}")
        else:
            # Load default parameters
            default_params_path = Path(f"./src/models/mltrees/model_params/{args.model}.yaml")
            model_params        = load_yaml_dict(default_params_path)
            logger.info("Using default model parameters")
            
    except Exception as e:
        logger.warning(f"Error loading parameters: {e}")
        logger.warning("Falling back to default parameters")
        default_params_path = f"./src/models/mltrees/model_params/{args.model}.yaml"
        model_params        = load_yaml_dict(default_params_path)
    
    logger.info(110*"_")
    logger.info(f"MLTree {args.model} regressor traning and evaluation using {n_folds}-fold cross-validation")
    logger.info(110*"_")

    # Load and prepare data partitions
    logger.info("Loading and preparing data partitions...")
    partitions = []

    # Load test set (outside the folds, same for all)
    test_path  = feats_path / "test.csv"
    test_df    = pd.read_csv(test_path, index_col=False)
    
    # Get test features and target
    feature_names = contfeats + catfeats + target
    
    feats_test, feats_names = tabular_features(test_df, names=feature_names, return_names=True)

    # Identify continuous and categorical and target columns   
    cont_columns   = [col for col in feats_test.columns if col in contfeats]
    cat_columns    = [col for col in feats_test.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in feats_test.columns if col in target]

    # Extract numpy arrays
    X_test      = feats_test[cont_columns+cat_columns].astype(np.float32).to_numpy()
    y_test      = feats_test[target_columns].astype(np.float32).to_numpy().flatten()  
    scaler_test = test_df[["M_tot"]].astype(np.float32).to_numpy().flatten() if "M_tot" in test_df.columns else None
    
    # Initialize lists to store results
    results        = []
    predictions    = []
    trained_models = []

    for fold in range(n_folds):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")

        fold_path  = feats_path / f"{fold}_fold"
        train_path = fold_path / "train.csv"
        val_path   = fold_path / "val.csv"

        # Load fold data
        train_df = pd.read_csv(train_path, index_col=False)
        val_df   = pd.read_csv(val_path, index_col=False)

        # Concatenate train and val for this fold (optimization has been done for this point)
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)

        # Extract features and target for train+val
        feats_trainval, _ = tabular_features(trainval_df, names=feature_names, return_names=True)
        
        X_trainval = feats_trainval[cont_columns+cat_columns].astype(np.float32).to_numpy()
        y_trainval = feats_trainval[target].astype(np.float32).to_numpy().flatten()  

        # Initialize and train modelflatten
        model = MLTreeRegressor(model_type   = args.model, 
                                model_params = model_params, 
                                device       = CONFIG.device, 
                                feat_names   = feats_names,
                                n_jobs       = CONFIG.n_jobs)

        logger.info(f"Training model on {len(X_trainval)} samples...")
        model.fit(X_trainval, y_trainval)
        trained_models.append(model)

        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Scale predictions if necessary
        if scaler_test is not None:
            y_pred_scaled = y_pred * scaler_test
            y_test_scaled = y_test * scaler_test
        else:
            y_pred_scaled = y_pred
            y_test_scaled = y_test
        
        # Calculate metrics
        fold_metrics = {
                    'fold' : fold,
                    'r2'   : float(r2_score(y_test_scaled, y_pred_scaled)),
                    'mae'  : float(mean_absolute_error(y_test_scaled, y_pred_scaled)),
                    'mse'  : float(mean_squared_error(y_test_scaled, y_pred_scaled)),
                    'rmse' : float(np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled)))
                       }

        results.append(fold_metrics)
        predictions.append({'fold': fold, 'y_pred': y_pred_scaled, 'y_true': y_test_scaled})
        
        logger.info(f"Fold {fold + 1} metrics:")
        for metric, value in fold_metrics.items():
            if metric != 'fold':
                logger.info(f"  - {metric}: {value:.5f}")
    
    # Calculate aggregate metrics
    metrics_df = pd.DataFrame(results)
    aggregate_metrics = {
        'mean_r2'  : float(metrics_df['r2'].mean()),
        'std_r2'   : float(metrics_df['r2'].std()),
        'mean_mae' : float(metrics_df['mae'].mean()),
        'std_mae'  : float(metrics_df['mae'].std()),
        'mean_rmse': float(metrics_df['rmse'].mean()),
        'std_rmse' : float(metrics_df['rmse'].std())
    }

    # Save results
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(out_path) / f"training_results_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_df = pd.DataFrame({'y_true': y_test_scaled.flatten()})
    
    for fold, pred_dict in enumerate(predictions):
        predictions_df[f'y_pred_fold_{fold}'] = pred_dict['y_pred'].flatten()
    
    predictions_df.to_csv(results_path / "predictions.csv", index=False)

    # Save metrics
    metrics_df.to_csv(results_path / "fold_metrics.csv", index=False)

    # Save training summary
    summary = {
        'timestamp'   : timestamp,
        'n_folds'     : n_folds,
        'model_type'  : args.model,
        'model_params': model_params,
        'features': {
            'continuous' : cont_columns,
            'categorical': cat_columns,
            'target'     : target_columns
            },
        'dataset_info': {
            'train_size'     : len(X_trainval),
            'test_size'      : len(X_test),
            'scaling_applied': True if "M_tot" in test_df.columns else False,
        },
        'metrics': {
            'per_fold' : results,
            'aggregate': aggregate_metrics
        }
    }

    with open(results_path / "training_summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    # Save trained models
    for fold, model in enumerate(trained_models):
        model.save_model(str(results_path / f"model_fold_{fold}.joblib"))

    logger.info("Training completed successfully!")
    logger.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        logger.info(f"  - {metric}: {value:.6f}")
    logger.info(f"Results saved to: {results_path}")

    # Create visualization directory
    viz_path = results_path / "figures"
    viz_path.mkdir(parents=True, exist_ok=True)

    # Generate correlation plot
    predictions_df = predictions_df.sample(frac=0.3, random_state=CONFIG.seed).reset_index(drop=True)  
    
    correlation_plot(predictions = predictions_df[["y_pred_fold_0", "y_pred_fold_1", "y_pred_fold_2"]].mean(axis=1),
                     true_values = predictions_df["y_true"],
                     path_save   = str(viz_path),
                     name_file   = f"{args.model}_mean_preds",
                     model_name  = f"{args.model} (Mean {n_folds}-fold)",
                     cmap        = "magma_r",
                     scale       = None,
                     show        = False)

    # Generate residual plot
    residual_plot(predictions = predictions_df[["y_pred_fold_0", "y_pred_fold_1", "y_pred_fold_2"]].mean(axis=1),
                  true_values = predictions_df["y_true"],
                  path_save   = str(viz_path),
                  name_file   = f"{args.model}_mean_preds",
                  model_name  = f"{args.model} (Mean {n_folds}-fold)",
                  cmap        = "magma_r",
                  scale       = None,
                  show        = False)
                  
    return trained_models, summary
    
def run_prediction():
    print("Prediction mode not yet implemented.")

# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    # Setup path manager
    path_manager = PathManager(root_dir = args.root_dir,
                               dataset  = args.dataset,
                               exp_name = args.exp_name,
                               model    = args.model,
                               out_dir  = args.out_dir,
                               fig_dir  = args.fig_dir)
    
    # Run appropriate mode
    if args.mode == "optim":
        
        run_optimization(feats_path = path_manager.data_path,
                         contfeats  = ["log(t)", "log(t_coll/t_cc)" ,"M_tot/M_crit", "log(rho(R_h))", "log(R_h/R_core)"],
                         catfeats   = ["type_sim"], 
                         target     = ["M_MMO/M_tot"],
                         out_path   = path_manager.base_out,
                         n_folds    = CONFIG.n_folds)
    
    elif args.mode == "train":
        run_training(feats_path = path_manager.data_path,
                     contfeats  = ["log(t)", "log(t_coll/t_cc)", "M_tot/M_crit", "log(rho(R_h))", "log(R_h/R_core)"],
                     catfeats   = ["type_sim"],
                     target     = ["M_MMO/M_tot"],
                     out_path   = path_manager.base_out,  
                     n_folds    = CONFIG.n_folds)
    
    elif args.mode == "predict":
        run_prediction()

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#