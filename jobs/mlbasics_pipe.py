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

# Directory
from src.utils.directory import PathManagerTrainOptPipeline, load_yaml_dict

# Computation of features
from src.processing.features import tabular_features

# Optimization of hyperparameters
from src.optim.optimizer import SpaceSearchConfig, SpaceSearch

# Vizualization
from src.processing.modules.plots import PlotGenerator

# Model Wrapper
from src.models.mlbasics.regressor import MLBasicRegressor

# Warnings managment ------------------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# List of tabular features to compute for the dataset (can be extended for other datasets if needed) ----------------------#
TabFeats = {
    "cont_feats"  : ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)", "log(t_coll)", 
                     "log(M_tot/M_crit)", 
                     "log(R_h/R_core)", "log(R_tid/R_core)",
                     "log(rho(R_h))",
                     "Z"],
    "cat_feats"   : ["type_sim"],
    
    "target_feat" : ["M_MMO/M_tot"],
            }

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/mlbasics_execution.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Configuration -----------------------------------------------------------------------------------------------------------#
@dataclass
class TrainingConfig:
    """Configuration class for the execution of mlbasics_training() script."""
    n_folds        : int   = 3
    n_trials       : int   = 100
    n_jobs         : int   = 15
    device         : str   = "cuda" if torch.cuda.is_available() else "cpu"
    seed           : int   = 42
    direction      : str   = "minimize"
    metric         : str   = "huber"
    patience       : int   = 20
    lambda_penalty : float = 0.0

CONFIG = TrainingConfig()

# Arguments ---------------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Execution of MLBasics models")
    
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
    parser.add_argument("--model", type=str, default="elasticnet",
                       choices=["elasticnet", "svr"],
                       help="Type of model to use")

    return parser.parse_args()
    
# Pipeline Modes [Hyperparameter Optimization] ----------------------------------------------------------------------------#
def run_optimization(feats_path: str, contfeats: list, catfeats: list, target: list, out_path: str, 
                     model_type: str, n_folds: int = 3):
    """Run the optimization mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Space search of the params for the mlbasics {model_type} regressor using {n_folds}-fold cross-validation")
    logger.info(110*"_")

    # Load and prepare data partitions
    logger.info("Loading and preparing data partitions...")
    
    # Define feature configuration (once for all folds)
    feature_names = contfeats + catfeats + target
    
    # Determine column names after transformation (use first fold as reference)
    fold_0_path = feats_path + "/0_fold/train.csv"
    temp_df     = pd.read_csv(fold_0_path, index_col=False)
    temp_feats  = tabular_features(temp_df, names=feature_names, return_names=False)
    
    # Identify transformed column names
    cont_columns   = [col for col in temp_feats.columns if col in contfeats]
    cat_columns    = [col for col in temp_feats.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in temp_feats.columns if col in target]
    feature_cols   = cont_columns + cat_columns
    
    # Log feature configuration
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {contfeats} -> {cont_columns}")
    logger.info(f"  - Categorical features : {catfeats} -> {cat_columns}")
    logger.info(f"  - Target               : {target} -> {target_columns}")
    logger.info(f"  - Total features       : {len(feature_cols)}")
    
    # Process each fold
    partitions = []
    for fold in range(n_folds):
        fold_path  = feats_path + f"/{fold}_fold"
        train_path = fold_path + "/train.csv"
        val_path   = fold_path + "/val.csv"

        # Load and transform fold data
        train_df = pd.read_csv(train_path, index_col=False)
        val_df   = pd.read_csv(val_path, index_col=False)
        
        feats_train = tabular_features(train_df, names=feature_names, return_names=False)
        feats_val   = tabular_features(val_df, names=feature_names, return_names=False)
        
        # Extract arrays
        X_train = feats_train[feature_cols].astype(np.float32).to_numpy()
        y_train = feats_train[target_columns].astype(np.float32).to_numpy().flatten()  
        X_val   = feats_val[feature_cols].astype(np.float32).to_numpy()
        y_val   = feats_val[target_columns].astype(np.float32).to_numpy().flatten()  

        # Scaling factor (if available)
        scaler_val = val_df["M_tot"].astype(np.float32).to_numpy().flatten() if "M_tot" in val_df.columns else None

        # Create partition dictionary
        partition = {
            'X_train'        : X_train,
            'y_train'        : y_train,
            'X_val'          : X_val,
            'y_val'          : y_val,
            'scaler'         : scaler_val,
            'features_names' : feature_cols
        }
        partitions.append(partition)
        
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={X_train.shape}, Val={X_val.shape}")

    # Initialize optimizer
    logger.info("Initializing optimizer...")
    config = SpaceSearchConfig(model_type = model_type, n_jobs = CONFIG.n_jobs, n_trials = CONFIG.n_trials, 
                               device = CONFIG.device, 
                               seed   = CONFIG.seed)
    optimizer = SpaceSearch(config)
    
    # Prepare study
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"cv_study_{n_folds}fold_{timestamp}"
    study_path = Path(out_path) / "optim"
    
    logger.info(f"Starting optimization: {study_name}")
    logger.info(f"  - Model            : {model_type}")
    logger.info(f"  - Device           : {CONFIG.device}")
    logger.info(f"  - Trials           : {CONFIG.n_trials}")
    logger.info(f"  - Direction        : {CONFIG.direction}")
    logger.info(f"  - Metric           : {CONFIG.metric}")
    logger.info(f"  - Lambda penalty   : {CONFIG.lambda_penalty}")
    logger.info(f"  - Patience         : {CONFIG.patience}")
    
    try:
        results = optimizer.optimize(partitions     = partitions,
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
        logger.info(f"  - Best score   : {results.best_score:.6f}")
        logger.info(f"  - Trials run   : {results.n_trials}")
        logger.info("  - Best parameters:")
        for param, value in results.best_params.items():
            logger.info(f"      {param}: {value}")
        
        # Save optimization summary
        summary_path = study_path / study_name / "optimization_summary.yaml"
        summary_data = {
            'study_name'  : study_name,
            'model_type'  : model_type,
            'n_folds'     : n_folds,
            'n_trials'    : results.n_trials,
            'best_score'  : float(results.best_score),
            'best_params' : results.best_params,
            'features': {
                'continuous'  : cont_columns,
                'categorical' : cat_columns,
                'target'      : target_columns[0]
            },
            'config': {
                'direction'      : CONFIG.direction,
                'metric'         : CONFIG.metric,
                'lambda_penalty' : CONFIG.lambda_penalty,
                'patience'       : CONFIG.patience,
                'device'         : CONFIG.device,
                'n_jobs'         : CONFIG.n_jobs,
                'seed'           : CONFIG.seed
            }
        }
        
        with open(summary_path, "w") as f:
            yaml.dump(summary_data, f, default_flow_style=False)
        
        logger.info(f"  - Results saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise
        
    return results

# Pipeline Modes [Training and Evaluation] --------------------------------------------------------------------------------#
def run_training(feats_path: str, contfeats: list, catfeats: list, target: list, out_path: str,
                 model_type: str, fig_path: str, n_folds: int = 3):
    """Run the training mode pipeline."""
    
    # Create the plot generator
    plot_generator = PlotGenerator(config=None, cmap="magma_r")
    
    try:
        # Find latest optimization study
        optimization_path = Path(out_path) / "optim"
        study_dirs        = [d for d in optimization_path.glob("cv_study_*fold_*") if d.is_dir()]
        
        if study_dirs:
            latest_study = max(study_dirs, key=lambda x: x.stat().st_mtime)
            opt_summary  = load_yaml_dict(f"{latest_study}/optimization_summary.yaml")
            model_params = opt_summary['best_params']
            logger.info(f"Using optimized parameters from: {latest_study.name}")
        else:
            # Load default parameters
            default_params_path = Path(f"./src/models/mlbasics/model_params/{model_type}.yaml")
            model_params        = load_yaml_dict(default_params_path)
            logger.info("Using default model parameters")
            
    except Exception as e:
        logger.warning(f"Error loading parameters: {e}")
        logger.warning("Falling back to default parameters")
        default_params_path = f"./src/models/mlbasics/model_params/{model_type}.yaml"
        model_params        = load_yaml_dict(default_params_path)
    
    logger.info(110*"_")
    logger.info(f"MLBasic {model_type} regressor training and evaluation using {n_folds}-fold cross-validation")
    logger.info(110*"_")

    # Load and prepare data partitions
    logger.info("Loading and preparing data partitions...")
    partitions = []

    # Load test set (outside the folds, same for all)
    test_path  = feats_path + "/test.csv"
    test_df    = pd.read_csv(test_path, index_col=False)
    
    # Get test features and target
    feature_names = contfeats + catfeats + target
    
    feats_test = tabular_features(test_df, names=feature_names, return_names=False)

    # Identify continuous, categorical, and target columns   
    cont_columns   = [col for col in feats_test.columns if col in contfeats]
    cat_columns    = [col for col in feats_test.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in feats_test.columns if col in target]
    feature_cols   = cont_columns + cat_columns

    # Extract numpy arrays
    X_test      = feats_test[feature_cols].astype(np.float32).to_numpy()
    y_test      = feats_test[target_columns].astype(np.float32).to_numpy().flatten()

    scaler_test = test_df["M_tot"].astype(np.float32).to_numpy().flatten() if "M_tot" in test_df.columns else None
    
    # Scale test target once (same for all folds)
    y_test_scaled = y_test * scaler_test if scaler_test is not None else y_test
    
    # Initialize lists to store results
    results             = []
    predictions         = []
    trained_models      = []
    feature_importances = []

    for fold in range(n_folds):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")

        fold_path  = feats_path + f"/{fold}_fold"
        train_path = fold_path + "/train.csv"

        # Load fold data
        train_df = pd.read_csv(train_path, index_col=False)

        # Extract features and target for train
        feats_train = tabular_features(train_df, names=feature_names, return_names=False)

        X_train = feats_train[feature_cols].astype(np.float32).to_numpy()
        y_train = feats_train[target_columns].astype(np.float32).to_numpy().flatten()

        # Initialize and train model
        model = MLBasicRegressor(model_type   = model_type,
                                 model_params = model_params,
                                 device       = CONFIG.device,
                                 feat_names   = feature_cols,
                                 n_jobs       = CONFIG.n_jobs)

        logger.info(f"Training model on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        trained_models.append(model)
        
        # Extract feature importance
        try:
            feat_importance = model.get_feature_importance()
            feature_importances.append(feat_importance)
            logger.info(f"Feature importance extracted for fold {fold + 1}")
        except Exception as e:
            logger.warning(f"Could not extract feature importance for fold {fold + 1}: {e}")
            feature_importances.append(None)

        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Scale predictions if necessary
        if scaler_test is not None:
            y_pred_scaled = np.clip(y_pred * scaler_test, 0, None)
        else:
            y_pred_scaled = np.clip(y_pred, 0, None)
        
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
        'model_type'  : model_type,
        'model_params': model_params,
        'features': {
            'continuous' : cont_columns,
            'categorical': cat_columns,
            'target'     : target_columns
            },
        'dataset_info': {
            'train_size'     : len(X_train),
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
    
    # Process and save feature importances
    if any(fi is not None for fi in feature_importances):
        # Filter out None values
        valid_importances = [fi for fi in feature_importances if fi is not None]
        
        if valid_importances:
            # Organize importances by feature (collect values across folds)
            all_features = list(valid_importances[0].keys())
            importances_by_feature = {feat: [] for feat in all_features}
            
            for fold_importance in valid_importances:
                for feat, val in fold_importance.items():
                    importances_by_feature[feat].append(val)
            
            # Convert to numpy arrays
            importances_by_feature = {k: np.array(v) for k, v in importances_by_feature.items()}
            
            # Calculate mean and std for each feature
            importance_stats = {
                feat: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'values': values.tolist()
                }
                for feat, values in importances_by_feature.items()
            }
            
            # Save feature importance summary
            with open(results_path / "feature_importance.yaml", 'w') as f:
                yaml.dump(importance_stats, f, default_flow_style=False)
            
            logger.info("Feature importances saved successfully")
            logger.info(f"Top 5 most important features:")
            sorted_features = sorted(importance_stats.items(), 
                                   key=lambda x: x[1]['mean'], reverse=True)[:5]
            for feat, stats in sorted_features:
                logger.info(f"  - {feat}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    logger.info("Training completed successfully!")
    logger.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        logger.info(f"  - {metric}: {value:.6f}")
    logger.info(f"Results saved to: {results_path}")

    # Create visualization directory using the configured figure path
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    # Generate correlation plot and residual plot for mean predictions across folds
    model_title    = {"elasticnet": "ENet", "svr": "SVR"}
    predictions_df = predictions_df.sample(frac=0.3, random_state=CONFIG.seed).reset_index(drop=True)  
    
    # Take the mean of all folds (dynamically based on n_folds)
    fold_columns = [f"y_pred_fold_{fold}" for fold in range(n_folds)]
    df_results   = predictions_df[fold_columns].mean(axis=1)
    
    # Generate plots using the plot generator
    plot_generator.create_ml_results_plots(predictions_df_mean = df_results,
                                           true_values_df      = predictions_df["y_true"],
                                           feature_importances = feature_importances,
                                           out_path            = viz_path,
                                           model_name          = model_type,
                                           model_title         = model_title)
                  
    return trained_models, summary
    
def run_prediction():
    print("Prediction mode not yet implemented.")

# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    # Setup path manager
    path_manager = PathManagerTrainOptPipeline(root_dir = args.root_dir,
                               dataset  = args.dataset,
                               exp_name = args.exp_name,
                               model    = args.model,
                               out_dir  = args.out_dir,
                               fig_dir  = args.fig_dir)
    
    # Run appropriate mode
    if args.mode == "optim":
        run_optimization(feats_path  = path_manager.data_path,
                         contfeats   = TabFeats["cont_feats"],
                         catfeats    = TabFeats["cat_feats"], 
                         target      = TabFeats["target_feat"],
                         out_path    = path_manager.base_out,
                         model_type  = args.model,
                         n_folds     = CONFIG.n_folds)
    
    elif args.mode == "train":
        run_training(feats_path  = path_manager.data_path,
                     contfeats   = TabFeats["cont_feats"],
                     catfeats    = TabFeats["cat_feats"],
                     target      = TabFeats["target_feat"],
                     out_path    = path_manager.base_out,
                     model_type  = args.model,
                     fig_path    = path_manager.fig_path,
                     n_folds     = CONFIG.n_folds)
    
    elif args.mode == "predict":
        run_prediction()

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#