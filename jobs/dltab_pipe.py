# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import argparse
import yaml
import torch 
import optuna

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from pathlib     import Path
from dataclasses import dataclass, field
from typing      import List, Optional
from datetime    import datetime

# Custom functions --------------------------------------------------------------------------------------------------------#

# Directory
from src.utils.directory import PathManagerTrainOptPipeline, load_yaml_dict

# Computation of features
from src.processing.features import tabular_features, TargetLogScaler

# Optimization of hyperparameters
from src.optim.optimizer import SpaceSearchConfig, SpaceSearch

# Param group mappings for structured best_params reconstruction
from src.optim.grid import MLP_PARAM_GROUPS

# Evaluation metrics
from src.utils.eval import compute_metrics

# Vizualization
from src.processing.modules.plots import PlotGenerator

# Model Wrapper
from src.models.dltab.regressor import DLTabularRegressor 

# DL Data Manager
from src.models.dltab.data.datasets import LEMONscDataManager

# Warnings managment ------------------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/dltabs_execution.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    
           retention = "10 days",  
           encoding  = "utf-8")

# Arguments ---------------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Execution of DLTabular models")
    
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
    parser.add_argument("--model", type=str, default="mlp",
                       choices=["mlp"],
                       help="Type of model to use")
    
    # DL architecture and optimizer config files
    parser.add_argument("--arch_config", type=str, default="mlp.yaml",
                        help="Architecture YAML file name (from src/models/dltab/config/arch/)")
    parser.add_argument("--opt_config", type=str, default="adam.yaml",
                        help="Optimizer YAML file name (from src/models/dltab/config/opt/)")

    return parser.parse_args()

# List of tabular features to compute for the dataset  --------------------------------------------------------------------#
CONT_FEATS  = ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)", "log(t_coll)", 
               "log(M_tot/M_crit)", 
               "log(R_h/R_core)", "log(R_tid/R_core)", 
               "log(rho(R_h))"]

CAT_FEATS   = None

TARGET_FEAT = ["log(M_MMO/M_tot)"]

# Configuration -----------------------------------------------------------------------------------------------------------#
@dataclass
class TrainingConfig:
    """Configuration class for the execution of dltab_training() script."""
    n_folds        : int                 = 3
    n_trials       : int                 = 100
    n_jobs         : int                 = 15
    device         : str                 = "cuda" if torch.cuda.is_available() else "cpu"
    cont_feats     : List[str]           = field(default_factory=lambda:CONT_FEATS.copy())
    cat_feats      : Optional[List[str]] = field(default_factory=lambda:CAT_FEATS.copy() if CAT_FEATS is not None else None)
    target_feat    : List[str]           = field(default_factory=lambda:TARGET_FEAT.copy())
    verbose        : bool                = True
    seed           : int                 = 42
    direction      : str                 = "minimize"
    metric         : str                 = "huber"
    patience       : int                 = 20
    lambda_penalty : float               = 0.0
    scale_target   : bool                = True
    epsilon_target : Optional[float]     = 1e-6
    norm_target    : Optional[str]       = "M_tot"
    # DL-specific parameters
    max_epochs     : int                 = 100
    dl_patience    : int                 = 10
    dl_loss_fn     : str                 = "huber"
    batch_size     : int                 = 2048
    num_workers    : int                 = 16 

# Instantiate the configuration
CONFIG = TrainingConfig()
    
# DL Architecture fixed parameters (non-optimizable, used as base for SpaceSearch) ----------------------------------------#
_DL_CONFIG_BASE = Path(__file__).resolve().parent.parent / "src" / "models" / "dltab" / "config"

def build_dl_architecture(arch_config: str = "mlp.yaml", opt_config: str = "adam.yaml") -> dict:
    """Build DL_ARCHITECTURE dict from YAML config files."""
    arch_params = load_yaml_dict(str(_DL_CONFIG_BASE / "arch" / arch_config))
    opt_params  = load_yaml_dict(str(_DL_CONFIG_BASE / "opt"  / opt_config))
    
    arch_elemnts = {
        "model_params"     : arch_params, 
        "optimizer_name"   : Path(opt_config).stem,
        "optimizer_params" : opt_params, 
        "loss_params"      : {}
                    }
    
    return arch_elemnts

# Pipeline Modes [Hyperparameter Optimization] ----------------------------------------------------------------------------#
def run_optimization(feats_path: str, contfeats: list, catfeats: list, target: list, out_path: str, model_type: str,
                     dl_architecture : dict,
                     scale_target    : bool            = True,
                     target_norm     : Optional[str]   = None,
                     target_eps      : Optional[float] = 1e-6,
                     n_folds         : int             = 3):
    """Run the optimization mode pipeline for DL tabular models."""
    
    logger.info(110*"_")
    logger.info(f"Space search of the params for the dltab {model_type} regressor using {n_folds}-fold cross-validation")
    logger.info(110*"_")
    
    # Load and prepare data partitions using LEMONscDataManager
    logger.info("Loading and preparing data partitions via DataLoaders...")
    
    # Handle None categorical features
    catfeats = catfeats if catfeats is not None else []
    
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
    
    # Define transform function to engineer features from raw CSV columns via tabular_features
    transform_fn = lambda df: tabular_features(df, names=feature_names, return_names=False)
    
    # Process each fold: create DataLoaders for SpaceSearch
    partitions = []
    for fold in range(n_folds):
        
        # Use LEMONscDataManager to load datasets and create DataLoaders for the current fold
        data_manager = LEMONscDataManager(dataset_root     = feats_path,
                                          fold             = fold,
                                          target_column    = target_columns[0],
                                          feature_columns  = feature_cols,
                                          metadata_columns = [target_norm] if target_norm is not None else None,
                                          transform_fn     = transform_fn,
                                          batch_size       = CONFIG.batch_size,
                                          num_workers      = CONFIG.num_workers,
                                          device           = CONFIG.device,
                                          logger           = logger)
        
        # Load train and validation partitions (no test needed for optimization)
        data_manager.setup(load_train=True, load_val=True, load_test=False)
        
        # Build scaler for the validation fold (mirrors mlbasics_pipe logic)
        scaler = None
        
        # Ensure that the validation dataset has the required metadata for denormalization (M_tot) and extract it
        if ( scale_target                                                                     and
             (data_manager.val_dataset and hasattr(data_manager.val_dataset, '_metadata_df')) and
             (data_manager.val_dataset._metadata_df is not None)                              and
             (target_norm is not None)                                                        and
             (target_norm in data_manager.val_dataset._metadata_df.columns)
             ):
            
            # Ensure that M_tot is numeric and extract it as a numpy array flattened to 1D
            val_norm = data_manager.val_dataset._metadata_df[target_norm].astype(np.float32).to_numpy().flatten()
            scaler   = TargetLogScaler(norm_factor=val_norm, epsilon=target_eps)
        
        # Scale without normalization factor if target_norm column not found
        elif scale_target:
            scaler = TargetLogScaler(norm_factor=None, epsilon=target_eps)
            logger.warning(f"Target normalization column '{target_norm}' not found in fold {fold + 1}. "
                            "A default TargetLogScaler with no norm_factor will be used.")

        # Create partition dictionary with DataLoaders (DL format expected by SpaceSearch)
        partition = {
            'train_loader'   : data_manager.train_loader,
            'val_loader'     : data_manager.val_loader,
            'scaler'         : scaler,
            'features_names' : feature_cols
        }
        partitions.append(partition)
        
        n_train = len(data_manager.train_dataset) if data_manager.train_dataset else 0
        n_val   = len(data_manager.val_dataset) if data_manager.val_dataset else 0
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={n_train} samples, Val={n_val} samples")

    # Initialize optimizer with DL-specific configuration
    logger.info("Initializing SpaceSearch optimizer for DL...")
    config = SpaceSearchConfig(model_type = model_type, n_jobs = CONFIG.n_jobs, n_trials = CONFIG.n_trials,
                               device          = CONFIG.device,
                               seed            = CONFIG.seed,
                               max_epochs      = CONFIG.max_epochs,
                               dl_patience     = CONFIG.dl_patience,
                               dl_loss_fn      = CONFIG.dl_loss_fn,
                               dl_architecture = dl_architecture)
    optimizer = SpaceSearch(config)
    
    # Prepare study
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"cv_study_{n_folds}fold_{timestamp}"
    study_path = Path(out_path) / "optim"
    
    # Log configuration before starting optimization
    logger.info(f"Starting optimization : {study_name}")
    logger.info(f"  - Model             : {model_type}")
    logger.info(f"  - Device            : {CONFIG.device}")
    logger.info(f"  - Trials            : {CONFIG.n_trials}")
    logger.info(f"  - Direction         : {CONFIG.direction}")
    logger.info(f"  - Metric            : {CONFIG.metric}")
    logger.info(f"  - Lambda penalty    : {CONFIG.lambda_penalty}")
    logger.info(f"  - Patience          : {CONFIG.patience}")
    logger.info(f"  - Max epochs        : {CONFIG.max_epochs}")
    logger.info(f"  - DL patience       : {CONFIG.dl_patience}")
    logger.info(f"  - DL loss fn        : {CONFIG.dl_loss_fn}")
    logger.info(f"  - Batch size        : {CONFIG.batch_size}")
    logger.info(f"  - Architecture      : {dl_architecture}")
    
    try:
        results = optimizer.optimize(partitions= partitions, study_name= study_name, direction= CONFIG.direction, 
                                     metric         = CONFIG.metric,
                                     output_dir     = str(study_path),
                                     save_study     = True,
                                     patience       = CONFIG.patience,
                                     pruner         = optuna.pruners.HyperbandPruner(min_resource     = CONFIG.patience,
                                                                                     max_resource     = CONFIG.max_epochs, 
                                                                                     reduction_factor = 3),
                                     lambda_penalty = CONFIG.lambda_penalty
                                     )
        
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
        
        # Reconstruct nested best_params from flat Optuna dict using MLP_PARAM_GROUPS
        param_groups_map = {"mlp": MLP_PARAM_GROUPS}
        flat_params      = results.best_params
        if model_type in param_groups_map:
            groups = param_groups_map[model_type]
            structured_best_params = {
                group: {k: flat_params[k] for k in keys if k in flat_params}
                for group, keys in groups.items()
            }
        else:
            structured_best_params = flat_params
        
        summary_data = {
            'study_name'  : study_name,
            'model_type'  : model_type,
            'n_folds'     : n_folds,
            'n_trials'    : results.n_trials,
            'best_score'  : float(results.best_score),
            'best_params' : structured_best_params,
            'features': {
                'continuous'  : cont_columns,
                'categorical' : cat_columns,
                'target'      : target_columns[0]
            },
            'architecture': dl_architecture,
            'config': {
                'direction'      : CONFIG.direction,
                'metric'         : CONFIG.metric,
                'lambda_penalty' : CONFIG.lambda_penalty,
                'patience'       : CONFIG.patience,
                'max_epochs'     : CONFIG.max_epochs,
                'dl_patience'    : CONFIG.dl_patience,
                'dl_loss_fn'     : CONFIG.dl_loss_fn,
                'batch_size'     : CONFIG.batch_size,
                'device'         : CONFIG.device,
                'n_jobs'         : CONFIG.n_jobs,
                'seed'           : CONFIG.seed,
                'scale_target'   : scale_target,
                'target_norm'    : target_norm,
                'epsilon_target' : target_eps
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
                 model_type      : str, 
                 fig_path        : str, 
                 dl_architecture : dict,
                 scale_target    : bool            = True,
                 target_norm     : Optional[str]   = None,
                 target_eps      : Optional[float] = 1e-6,
                 n_folds         : int              = 3):
    """Run the training mode pipeline for DL tabular models."""
    
    # Create the plot generator
    plot_generator = PlotGenerator(config=None, cmap="magma_r")
    
    # Load best hyperparameters from optimization or use defaults ---------------------------------------------------------#
    # Default nested structure built from dl_architecture (used when no opt study is found or on error)
    _default_model_params = {
        "architecture_params" : dl_architecture['model_params'],
        "optimizer_params"    : dl_architecture['optimizer_params'],
        "loss_params"         : dl_architecture['loss_params']
    }
    
    try:
        optimization_path = Path(out_path) / "optim"
        study_dirs        = [d for d in optimization_path.glob("cv_study_*fold_*") if d.is_dir()]
        
        if study_dirs:
            latest_study = max(study_dirs, key=lambda x: x.stat().st_mtime)
            opt_summary  = load_yaml_dict(f"{latest_study}/optimization_summary.yaml")
            model_params = opt_summary['best_params']
            
            logger.info(f"Using optimized parameters from: {latest_study.name}")
        else:
            model_params = _default_model_params
            logger.info("No optimization study found. Using default architecture parameters.")
            
    except Exception as e:
        logger.warning(f"Error loading parameters: {e}")
        logger.warning("Falling back to default architecture parameters.")
        model_params = _default_model_params
    
    logger.info(110*"_")
    logger.info(f"DLTab {model_type} regressor training and evaluation using {n_folds}-fold cross-validation")
    logger.info(110*"_")
    
    # Load and prepare data partitions via LEMONscDataManager -------------------------------------------------------------#
    logger.info("Loading and preparing data partitions...")
    
    # Handle None categorical features
    catfeats = catfeats if catfeats is not None else []
    
    # Define feature configuration
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
    
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {contfeats} -> {cont_columns}")
    logger.info(f"  - Categorical features : {catfeats} -> {cat_columns}")
    logger.info(f"  - Target               : {target} -> {target_columns}")
    logger.info(f"  - Total features       : {len(feature_cols)}")
    
    # Define transform function
    transform_fn = lambda df: tabular_features(df, names=feature_names, return_names=False)
    
    # ---- 3. Load test set and build target scaler -----------------------------------------------------------------------#
    test_path  = feats_path + "/test.csv"
    test_df    = pd.read_csv(test_path, index_col=False)
    feats_test = tabular_features(test_df, names=feature_names, return_names=False)
    y_test     = feats_test[target_columns].astype(np.float32).to_numpy().flatten()
    
    # Create target scaler for the test set
    if scale_target and target_norm is not None and target_norm in test_df.columns:
        test_norm   = test_df[target_norm].astype(np.float32).to_numpy().flatten()
        scaler_test = TargetLogScaler(norm_factor=test_norm, epsilon=target_eps)
        y_test_scaled = scaler_test.inverse_transform(y_test)
    elif scale_target:
        scaler_test = TargetLogScaler(norm_factor=None, epsilon=target_eps)
        logger.warning(f"Target normalization column '{target_norm}' not found in test set. "
                       "A default scaler with no norm_factor will be used.")
        y_test_scaled = scaler_test.inverse_transform(y_test)
    else:
        scaler_test   = None
        y_test_scaled = y_test
        logger.warning("No scaling will be applied.")
    
    # Verbose warning 
    logger.warning(f"The scaler is not integrated in LEMONscDataManager for the test set. "
                   "Ensure that the test set maintains the same order to ensure proper evaluation.")
    
    # Train models per fold -----------------------------------------------------------------------------------------------#
    results        = []
    predictions    = []
    trained_models = []
    
    for fold in range(n_folds):
        logger.info(f"Processing fold {fold + 1}/{n_folds}")
        
        # Load fold data using LEMONscDataManager
        data_manager = LEMONscDataManager(dataset_root     = feats_path,
                                          fold             = fold,
                                          target_column    = target_columns[0],
                                          feature_columns  = feature_cols,
                                          metadata_columns = [target_norm] if target_norm is not None else None,
                                          transform_fn     = transform_fn,
                                          batch_size       = CONFIG.batch_size,
                                          num_workers      = CONFIG.num_workers,
                                          device           = CONFIG.device,
                                          logger           = logger)
        
        # Load train, validation and test partitions
        data_manager.setup(load_train=True, load_val=True, load_test=True)
        
        n_train = len(data_manager.train_dataset) if data_manager.train_dataset else 0
        n_val   = len(data_manager.val_dataset) if data_manager.val_dataset else 0
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={n_train} samples, Val={n_val} samples")
        
        # Merge optimized params into architecture model_params (optimized params override defaults)
        fold_model_params = {**dl_architecture['model_params'], **model_params["architecture_params"]}
        fold_opt_name     = dl_architecture['optimizer_name']
        fold_opt_params   = {**dl_architecture['optimizer_params'], **model_params["optimizer_params"]}
        lossfn_params      = model_params["loss_params"]
        
        # Initialize the DL model
        model = DLTabularRegressor(model_type       = model_type,
                                   in_features      = len(feature_cols),
                                   model_params     = fold_model_params,
                                   optimizer_name   = fold_opt_name,
                                   optimizer_params = fold_opt_params,
                                   feat_names       = feature_cols,
                                   device           = CONFIG.device,
                                   verbose          = CONFIG.verbose)
        
        # Train the model
        logger.info(f"Training DL model on {n_train} samples (max {CONFIG.max_epochs} epochs)...")
        model.fit(train_loader             = data_manager.train_loader,
                  val_loader               = data_manager.val_loader,
                  epochs                   = CONFIG.max_epochs,
                  loss_fn                  = CONFIG.dl_loss_fn,
                  loss_params              = lossfn_params,
                  early_stopping_patience  = CONFIG.dl_patience,
                  verbose_epoch            = 10)
        
        # Load best weights from early stopping
        model.load_best_weights()
        trained_models.append(model)
        
        # Generate predictions on test set
        y_pred = model.predict(data_manager.test_loader)
        
        # Transform predictions to original scale
        if scaler_test is not None:
            y_pred_scaled = scaler_test.inverse_transform(y_pred)
        else:
            y_pred_scaled = np.clip(y_pred, 0, None)
        
        # Calculate metrics
        fold_metrics = {'fold': fold, **compute_metrics(y_test_scaled, y_pred_scaled)}
        
        results.append(fold_metrics)
        predictions.append({'fold': fold, 'y_pred': y_pred_scaled, 'y_true': y_test_scaled})
        
        logger.info(f"Fold {fold + 1} metrics:")
        for metric, value in fold_metrics.items():
            if metric != 'fold':
                logger.info(f"  - {metric}: {value:.5f}")
    
    # Aggregate metrics ---------------------------------------------------------------------------------------------------#
    metrics_df = pd.DataFrame(results)
    aggregate_metrics = {
        'mean_r2'  : float(metrics_df['r2'].mean()),
        'std_r2'   : float(metrics_df['r2'].std()),
        'mean_mae' : float(metrics_df['mae'].mean()),
        'std_mae'  : float(metrics_df['mae'].std()),
        'mean_rmse': float(metrics_df['rmse'].mean()),
        'std_rmse' : float(metrics_df['rmse'].std())
    }
    
    # Save results --------------------------------------------------------------------------------------------------------#
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
        'architecture': dl_architecture,
        'features': {
            'continuous'  : cont_columns,
            'categorical' : cat_columns,
            'target'      : target_columns
        },
        'dataset_info': {
            'test_size'      : len(y_test),
            'scale_target'   : scale_target,
            'target_norm'    : target_norm,
            'epsilon_target' : target_eps
        },
        'config': {
            'max_epochs'   : CONFIG.max_epochs,
            'dl_patience'  : CONFIG.dl_patience,
            'dl_loss_fn'   : CONFIG.dl_loss_fn,
            'batch_size'   : CONFIG.batch_size,
            'device'       : CONFIG.device,
            'seed'         : CONFIG.seed
        },
        'metrics': {
            'per_fold'  : results,
            'aggregate' : aggregate_metrics
        }
    }
    
    with open(results_path / "training_summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Save trained models
    for fold, model in enumerate(trained_models):
        model.save_model(str(results_path / f"model_fold_{fold}.pt"))
    
    logger.info("Training completed successfully!")
    logger.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        logger.info(f"  - {metric}: {value:.6f}")
    logger.info(f"Results saved to: {results_path}")
    
    # Generate visualizations ---------------------------------------------------------------------------------------------#
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)
    
    # Generate correlation plot and residual plot for mean predictions across folds
    model_title    = {"mlp": "MLP"}
    predictions_df = predictions_df.sample(frac=0.3, random_state=CONFIG.seed).reset_index(drop=True)
    
    # Take the mean of all folds
    fold_columns = [f"y_pred_fold_{fold}" for fold in range(n_folds)]
    df_results   = predictions_df[fold_columns].mean(axis=1)
    
    # Generate plots using the plot generator
    plot_generator.create_ml_results_plots(predictions_df_mean = df_results,
                                           true_values_df      = predictions_df["y_true"],
                                           feature_importances = None,
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
    # Build DL architecture from YAML config files
    dl_architecture = build_dl_architecture(arch_config=args.arch_config, opt_config=args.opt_config)
    
    if args.mode == "optim":
        run_optimization(feats_path      = path_manager.data_path,
                         contfeats       = CONFIG.cont_feats,
                         catfeats        = CONFIG.cat_feats, 
                         target          = CONFIG.target_feat,
                         out_path        = path_manager.base_out,
                         model_type      = args.model,
                         dl_architecture = dl_architecture,
                         scale_target    = CONFIG.scale_target,
                         target_norm     = CONFIG.norm_target,
                         target_eps      = CONFIG.epsilon_target,
                         n_folds         = CONFIG.n_folds)
    
    elif args.mode == "train":
        run_training(feats_path      = path_manager.data_path,
                     contfeats       = CONFIG.cont_feats,
                     catfeats        = CONFIG.cat_feats,
                     target          = CONFIG.target_feat,
                     out_path        = path_manager.base_out,
                     model_type      = args.model,
                     fig_path        = path_manager.fig_path,
                     dl_architecture = dl_architecture,
                     scale_target    = CONFIG.scale_target,
                     target_norm     = CONFIG.norm_target,
                     target_eps      = CONFIG.epsilon_target,
                     n_folds         = CONFIG.n_folds)
    
    elif args.mode == "predict":
        run_prediction()

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#