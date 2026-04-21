# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import argparse
import yaml
import warnings

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing          import Optional
from pathlib         import Path
from loguru          import logger
from datetime        import datetime

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory           import PathManagerTrainOptPipeline, load_yaml_dict
from src.processing.features       import tabular_features, filter_simulation_artifacts
from src.processing.scalers        import TargetTransform, FeatureScaler
from src.optim.optimizer           import SpaceSearchConfig, SpaceSearch
from src.processing.modules.plots  import PlotGenerator
from src.models.mlbasics.regressor import MLBasicRegressor
from src.utils.eval                import compute_metrics
from jobs.config._mlbasics         import JobConfig

# Warnings managment ------------------------------------------------------------------------------------------------------#
warnings.filterwarnings("ignore", category=UserWarning)

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

# Global job arguments ----------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Execution of MLBasics models")
    
    # Main mode of the script
    parser.add_argument("--mode", type=str, default="optim",
                        choices=["optim", "train", "inter", "predict"],
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
                       choices = ["elasticnet", "linearsvr"],
                       help    = "Type of model to use")
    parser.add_argument("--n_sims", type=int, default=4,
                        help = "Number of random simulations to plot in interpretation mode.")
    
    # Study persistence (SQLite)
    parser.add_argument("--study_name", type=str, default=None,
                        help="Deterministic name for the Optuna study. Required when using --storage to allow "
                             "resume. If omitted, auto-generated (timestamp-based without storage, "
                             "model+exp_name+n_folds based when storage is set).")
    parser.add_argument("--storage", type=str, default=None,
                        help="SQLite storage URL for Optuna study persistence and resume, e.g. "
                             "'sqlite:///./output/optim/study.db'. "
                             "When set, load_if_exists=True is applied automatically.")

    return parser.parse_args()

# List of tabular features to compute for the dataset  --------------------------------------------------------------------#
CONT_FEATS  = ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)", "log(t/t_coll)", 
               "log(M_tot)", "log(M_MMO_0)",
               "log(R_h/R_core)", "log(R_tid/R_core)", 
               "log(rho(R_h))",
               "log(Z)", 
               "log(fbin)"]
CAT_FEATS   = None

TARGET_FEAT = ["log(M_MMO/M_tot)"]
    
# Instantiate the configuration
CONFIG = JobConfig()
CONFIG.dataconfig.cont_feats  = CONT_FEATS
CONFIG.dataconfig.cat_feats   = CAT_FEATS
CONFIG.dataconfig.target_feat = TARGET_FEAT

# Pipeline Modes [Hyperparameter Optimization] ----------------------------------------------------------------------------#
def run_optimization(feats_path: str, contfeats: list, catfeats: list, target: list, trs_target: bool,
                     target_norm : Optional[str], 
                     out_path    : str,
                     model_type  : str, 
                     n_folds     : int           = 3,
                     study_name  : Optional[str] = None,
                     storage     : Optional[str] = None):
    """Run the optimization mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Space search of the params for the mlbasics {model_type} regressor using {n_folds}-fold cross-validation")
    logger.info(110*"_")
    logger.info("Loading and preparing data partitions...")
    
    # Handle None categorical features
    catfeats = catfeats if catfeats is not None else []
    
    # Define feature configuration (once for all folds)
    feature_names = contfeats + catfeats + target
    
    # Determine column names after transformation (use first fold as reference)
    fold_0_path = feats_path + "/0_fold/train.csv"
    temp_df     = pd.read_csv(fold_0_path, index_col=False)
    
    temp_filt   = filter_simulation_artifacts(temp_df, min_denominator_threshold=CONFIG.dataconfig.min_dem_thr,
                                              filter_null_mass     = True,
                                              filter_initial_state = False, 
                                              verbose              = False)
    
    temp_feats  = tabular_features(temp_filt, names=feature_names, return_names=False,
                                   eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                   eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)
    
    # Identify transformed column names
    cont_columns   = [col for col in temp_feats.columns if col in contfeats]
    cat_columns    = [col for col in temp_feats.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in temp_feats.columns if col in target]
    feature_cols   = cont_columns + cat_columns
    
    # Verbose
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {len(cont_columns)}")
    logger.info(f"  - Categorical features : {len(cat_columns)}")
    logger.info(f"  - Target               : {target_columns}")
    logger.info(f"  - Total features       : {len(feature_cols)}")
    
    # Process each fold
    partitions = []
    
    for fold in range(n_folds):     
        # Prepare study name:
        study_path = Path(out_path) / "optim"
        
        # If not name provided, generate one based on storage presence and model/experiment details
        if study_name is not None:
            pass  
        
        # If a storage is provided but no study name, create a deterministic one based on model type, experiment tag and n_folds
        elif storage is not None:
            _exp_tag   = Path(out_path).name   
            study_name = f"cv_study_{model_type}_{_exp_tag}_{n_folds}fold"
            logger.info(f"SQLite storage active — using deterministic study name: '{study_name}'")
        
        # If no storage and no name, generate a timestamp-based name to avoid conflicts between independent runs
        else:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"cv_study_{n_folds}fold_{timestamp}"
        
        # Create fold directory
        fold_output_path = study_path/ f"{study_name}/fold_{fold}"
        fold_output_path.mkdir(parents=True, exist_ok=True)

        # Define paths to the fold data
        fold_data_path  = feats_path + f"/{fold}_fold"
        train_path      = fold_data_path + "/train.csv"
        val_path        = fold_data_path + "/val.csv"

        # Load and transform fold data
        train_df = pd.read_csv(train_path, index_col=False)
        val_df   = pd.read_csv(val_path, index_col=False)
        
        # Filter possible numerical effects
        filter_tab_train = filter_simulation_artifacts(raw_df                    = train_df, 
                                                       min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                                       filter_null_mass          = True,
                                                       filter_initial_state      = False, 
                                                       verbose                   = False)
        
        filter_tab_val   = filter_simulation_artifacts(raw_df                    = val_df, 
                                                       min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                                       filter_null_mass          = True,
                                                       filter_initial_state      = False, 
                                                       verbose                   = False)
        
        # Extract the tabular features
        feats_train = tabular_features(filter_tab_train, names=feature_names, return_names=False,
                                       eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                       eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)
        
        feats_val   = tabular_features(filter_tab_val, names=feature_names, return_names=False,
                                       eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                       eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)
        
        # Extract arrays
        X_train = feats_train[feature_cols].astype(np.float32).to_numpy()
        y_train = feats_train[target_columns].astype(np.float32).to_numpy().flatten()  
        X_val   = feats_val[feature_cols].astype(np.float32).to_numpy()
        y_val   = feats_val[target_columns].astype(np.float32).to_numpy().flatten()  
            
        # Create target transformation (if target_norm available)
        if trs_target and target_norm in filter_tab_val.columns:
            val_norm = filter_tab_val[target_norm].astype(np.float32).to_numpy().flatten()
            val_trs  = TargetTransform(transformation = CONFIG.scalingconfig.target_transform, 
                                       norm_factor    = val_norm, 
                                       epsilon        = CONFIG.scalingconfig.target_log_eps)
        
        # Fall back with a warning
        else:
            val_trs = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
            logger.warning(f"trs_target=True but norm column '{target_norm}' not found in fold {fold} val set. "
                           "Falling back to identity transformation.")
        
        # Create a scaler for the features
        should_scale  = CONFIG.scalingconfig.scale_features
        scaler_kwargs = {
                "scaler_name"   : CONFIG.scalingconfig.feature_scaler_name,
                "scaler_kwargs" : CONFIG.scalingconfig.feature_scaler_kward
                        }
        
        if should_scale:    
            feat_scaler = FeatureScaler(**scaler_kwargs)
            X_train     = feat_scaler.fit_transform(X_train)
            X_val       = feat_scaler.transform(X_val)

            # Store trained scaler in the fold output directory for later use in training and evaluation
            feat_scaler.save_scaler(fold_output_path / "feats_scaler.joblib")
            
        # Create partition dictionary
        partition = {
            'X_train'         : X_train,
            'y_train'         : y_train,
            'X_val'           : X_val,
            'y_val'           : y_val,
            'trs'             : val_trs,
            'feature_scaler'  : feat_scaler if should_scale else None,
            'features_names'  : feature_cols
        }
        partitions.append(partition)
        
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={X_train.shape}, Val={X_val.shape}")

    # Initialize optimizer
    logger.info("Initializing optimizer...")
    config = SpaceSearchConfig(model_type = model_type, n_jobs = CONFIG.loaders_n_jobs,
                               n_trials             = CONFIG.optconfig.n_trials,
                               device               = CONFIG.device,
                               seed                 = CONFIG.seed,
                               storage              = storage,
                               load_if_exists       = storage is not None)
    
    optimizer = SpaceSearch(config)
    
    logger.info(f"Starting optimization: {study_name}")
    logger.info(f"  - Model            : {model_type}")
    logger.info(f"  - Device           : {CONFIG.device}")
    logger.info(f"  - Trials           : {CONFIG.optconfig.n_trials}")
    logger.info(f"  - Direction        : {CONFIG.optconfig.direction}")
    logger.info(f"  - Metric           : {CONFIG.optconfig.metric}")
    logger.info(f"  - Lambda penalty   : {CONFIG.optconfig.lambda_penalty}")
    logger.info(f"  - Patience         : {CONFIG.optconfig.trial_patience}")
    
    try:
        results = optimizer.optimize(partitions     = partitions,
                                     study_name     = study_name,
                                     direction      = CONFIG.optconfig.direction,
                                     metric         = CONFIG.optconfig.metric,
                                     output_dir     = str(study_path),
                                     save_study     = True,
                                     patience       = CONFIG.optconfig.trial_patience,
                                     lambda_penalty = CONFIG.optconfig.lambda_penalty)
        
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
                'direction'      : CONFIG.optconfig.direction,
                'metric'         : CONFIG.optconfig.metric,
                'lambda_penalty' : CONFIG.optconfig.lambda_penalty,
                'patience'       : CONFIG.optconfig.trial_patience,
                'device'         : CONFIG.device,
                'n_jobs'         : CONFIG.loaders_n_jobs,
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
def run_training(feats_path  : str, contfeats : list, catfeats : list, target : list, trs_target : bool,
                 target_norm : Optional[str],
                 out_path    : str,
                 model_type  : str,
                 fig_path    : str,
                 n_folds     : int = 3):
    """Run the training mode pipeline."""
    
    # Save results
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(out_path) / f"training_results_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)

    # Create the plot generator
    plot_generator = PlotGenerator(config=None, cmap="magma")

    # Verbose
    logger.info(110*"_")
    logger.info(f"MLBasic {model_type} regressor training and evaluation using {n_folds}-fold cross-validation")
    logger.info(110*"_")
    
    # Load best hyperparameters from the latest optimization study --------------------------------------------------------#
    try:
        optimization_path = Path(out_path) / "optim"
        study_dirs        = [d for d in optimization_path.glob("cv_study_*") if d.is_dir()]
        
        if study_dirs:
            latest_study = max(study_dirs, key=lambda x: x.stat().st_mtime)
            opt_summary  = load_yaml_dict(f"{latest_study}/optimization_summary.yaml")
            model_params = opt_summary['best_params']
            logger.info(f"Using optimized parameters from: {latest_study.name}")
        else:
            default_params_path = Path(f"./src/models/mlbasics/model_params/{model_type}.yaml")
            model_params        = load_yaml_dict(default_params_path)
            logger.info("No optimization study found. Using default model parameters.")
    
    # Catch any exception during the loading of optimization results and fallback to default parameters            
    except Exception as e:
        logger.warning(f"Error loading optimized parameters: {e}")
        logger.warning("Falling back to default parameters.")
        default_params_path = f"./src/models/mlbasics/model_params/{model_type}.yaml"
        model_params        = load_yaml_dict(default_params_path)

    # Load and prepare partitions for training ----------------------------------------------------------------------------
    logger.info("Loading and preparing data partitions...")
    
    # Handle None categorical features 
    catfeats = catfeats if catfeats is not None else []

    # Define feature configuration (once for all folds)
    feature_names = contfeats + catfeats + target

    # Determine column names after transformation (use first fold as reference)
    fold_0_path = feats_path + "/0_fold/train.csv"
    temp_df     = pd.read_csv(fold_0_path, index_col=False)

    temp_filt  = filter_simulation_artifacts(temp_df, min_denominator_threshold=CONFIG.dataconfig.min_dem_thr,
                                             filter_null_mass     = True,
                                             filter_initial_state = False,
                                             verbose              = False)

    temp_feats = tabular_features(temp_filt, names=feature_names, return_names=False,
                                  eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                  eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)

    # Identify transformed column names and ensure desired featuress
    cont_columns   = [col for col in temp_feats.columns if col in contfeats]
    cat_columns    = [col for col in temp_feats.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in temp_feats.columns if col in target]
    feature_cols   = cont_columns + cat_columns
    
    # Verbose
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {contfeats} -> {cont_columns}")
    logger.info(f"  - Categorical features : {catfeats} -> {cat_columns}")
    logger.info(f"  - Target               : {target} -> {target_columns}")
    logger.info(f"  - Total features       : {len(feature_cols)}")


    # Load test set and build transfomation for the target
    test_path = feats_path + "/test.csv"
    test_df   = pd.read_csv(test_path, index_col=False)

    filter_test = filter_simulation_artifacts(raw_df                    = test_df,
                                              min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                              filter_null_mass          = True,
                                              filter_initial_state      = False,
                                              verbose                   = False)

    feats_test = tabular_features(filter_test, names=feature_names, return_names=False,
                                  eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                  eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)

    X_test = feats_test[feature_cols].astype(np.float32).to_numpy()
    y_test = feats_test[target_columns].astype(np.float32).to_numpy().flatten()

    # Create target transformation if target_norm is available
    if trs_target and target_norm in filter_test.columns:
        test_norm = filter_test[target_norm].astype(np.float32).to_numpy().flatten()
        test_trs  = TargetTransform(transformation = CONFIG.scalingconfig.target_transform,
                                    norm_factor    = test_norm,
                                    epsilon        = CONFIG.scalingconfig.target_log_eps)
        y_test_phys = test_trs.inverse_transform(y_test)

    # Fall back with a warning
    else:
        # Verbose warning
        logger.warning(f"A default identity transformation will be used as configuration is not well defined.")
        
        test_trs    = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
        y_test_phys = y_test

    # Set configuration of the feature scaler
    should_scale = CONFIG.scalingconfig.scale_features
    scaler_kwargs = {
        "scaler_name"   : CONFIG.scalingconfig.feature_scaler_name,
        "scaler_kwargs" : CONFIG.scalingconfig.feature_scaler_kward
                    }

    # Initialize lists to store results -----------------------------------------------------------------------------------#
    results        = []
    predictions    = []
    trained_models = []

    for fold in range(n_folds):

        # Create fold directory
        fold_output_path = results_path / f"fold_{fold}"
        fold_output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing fold {fold + 1}/{n_folds}")

        fold_data_path  = feats_path + f"/{fold}_fold"
        train_data_path = fold_data_path + "/train.csv"

        # Load fold data
        train_df = pd.read_csv(train_data_path, index_col=False)

        # Filter possible numerical artifacts
        filter_tab_train = filter_simulation_artifacts(raw_df                    = train_df,
                                                       min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                                       filter_null_mass          = True,
                                                       filter_initial_state      = False,
                                                       verbose                   = False)

        # Extract features and target for train
        feats_train = tabular_features(filter_tab_train, names=feature_names, return_names=False,
                                       eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                       eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)

        X_train = feats_train[feature_cols].astype(np.float32).to_numpy()
        y_train = feats_train[target_columns].astype(np.float32).to_numpy().flatten()

        # Apply feature scaling
        if should_scale:    
            feat_scaler = FeatureScaler(**scaler_kwargs)
            X_train     = feat_scaler.fit_transform(X_train)

            # Store trained scaler in the fold output directory for later use in training and evaluation
            feat_scaler.save_scaler(fold_output_path / "feats_scaler.joblib")

        else:
            feat_scaler = None

        # Initialize and train model ----------------------------------------------------------------------------------#
        model = MLBasicRegressor(model_type   = model_type,
                                 model_params = model_params,
                                 device       = CONFIG.device,
                                 feat_names   = feature_cols,
                                 n_jobs       = CONFIG.loaders_n_jobs)

        logger.info(f"Training model on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        trained_models.append(model)

        # Generate predictions on scaled test set
        X_test_scaled = feat_scaler.transform(X_test) if feat_scaler is not None else X_test
        y_pred        = model.predict(X_test_scaled)

        # Transform predictions to original scale (m_mmo)
        if test_trs is not None:
            y_pred_phys = test_trs.inverse_transform(y_pred)

        # If no target transform, clip negative predictions
        else:
            y_pred_phys = np.clip(y_pred, 0, None)
        
        # Calculate metrics
        fold_metrics = {'fold': fold, **compute_metrics(y_test_phys, y_pred_phys)}

        results.append(fold_metrics)
        predictions.append({'fold': fold, 'y_pred': y_pred_phys, 'y_true': y_test_phys})
        
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

    # Save predictions
    predictions_df = pd.DataFrame({'y_true': y_test_phys.flatten()})
    
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
            'train_size'          : len(X_train),
            'test_size'           : len(X_test),
            'features_scaled'     : should_scale,
            'feature_scaler_name' : scaler_kwargs['scaler_name'] if should_scale else 'none',
            'target_transform'    : CONFIG.scalingconfig.target_transform if trs_target else 'identity',
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
        model.save_model(str(results_path / f"fold_{fold}/model.joblib"))

    logger.info("Training completed successfully!")
    logger.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        logger.info(f"  - {metric}: {value:.6f}")
    logger.info(f"Results saved to: {results_path}")

    # Create visualization directory using the configured figure path
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    # Generate correlation plot and residual plot for mean predictions across folds
    model_title    = {"elasticnet": "ENet", "linearsvr": "LinearSVR"}
    predictions_df = predictions_df.sample(frac=0.5, random_state=CONFIG.seed).reset_index(drop=True)  
    
    # Take the mean of all folds (dynamically based on n_folds)
    fold_columns = [f"y_pred_fold_{fold}" for fold in range(n_folds)]
    df_results   = predictions_df[fold_columns].mean(axis=1)
    
    # Generate plots using the plot generator
    plot_generator.create_ml_results_plots(predictions_df_mean = df_results,
                                           true_values_df      = predictions_df["y_true"],
                                           out_path            = viz_path,
                                           model_name          = model_type,
                                           model_title         = model_title)
                  
    return trained_models, summary

# Pipeline Modes [Interpretation] -----------------------------------------------------------------------------------------#
def run_interpretation(feats_path : str, contfeats : list, catfeats : list, target : list, trs_target : bool, 
                       target_norm   : Optional[str],
                       out_path      : str,
                       model_type    : str,
                       fig_path      : str,
                       n_folds       : int = 3,
                       n_simulations : int = 4):
    """Run the interpretation pipeline: feature importance and simulation example plots."""

    plot_generator = PlotGenerator(config=None, cmap="magma")

    logger.info(110*"_")
    logger.info(f"MLBasic {model_type} interpretation mode")
    logger.info(110*"_")

    # Locate the latest training results directory -----------------------------------------------------------------------#
    training_dirs = sorted([d for d in Path(out_path).glob("training_results_*") if d.is_dir()],
                           key=lambda x: x.stat().st_mtime)

    if not training_dirs:
        raise FileNotFoundError(f"No training results found in {out_path}. Run training first.")

    latest_results = training_dirs[-1]
    logger.info(f"Loading trained models from: {latest_results.name}")

    # Load trained models and feature scalers from all folds ----------------------------------------------------------#
    trained_models  = []
    feature_scalers = []
    should_scale    = CONFIG.scalingconfig.scale_features

    for fold in range(n_folds):
        model_path = latest_results / f"fold_{fold}/model.joblib"
        model      = MLBasicRegressor.load_model(str(model_path))
        trained_models.append(model)

        # Load per-fold feature scaler saved during training
        if should_scale:
            scaler_path = latest_results / f"fold_{fold}" / "feats_scaler.joblib"
            feat_scaler = FeatureScaler(scaler_name  = CONFIG.scalingconfig.feature_scaler_name,
                                        scaler_kwargs = CONFIG.scalingconfig.feature_scaler_kward,
                                        load_scaler_from = str(scaler_path))
            feature_scalers.append(feat_scaler)
            logger.info(f"Loaded model and scaler for fold {fold + 1}/{n_folds}")
        else:
            feature_scalers.append(None)
            logger.info(f"Loaded model for fold {fold + 1}/{n_folds}")

    # Handle None categorical features -----------------------------------------------------------------------------------#
    catfeats = catfeats if catfeats is not None else []

    # Load and prepare the test set --------------------------------------------------------------------------------------#
    test_path  = feats_path + "/test.csv"
    test_df    = pd.read_csv(test_path, index_col=False)

    feature_names = contfeats + catfeats + target
    filter_test   = filter_simulation_artifacts(raw_df = test_df,
                                                min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                                filter_null_mass          = True,
                                                filter_initial_state      = False,
                                                verbose                   = False)
    filter_test   = filter_test.reset_index(drop=True)

    feats_test, feats_labels = tabular_features(filter_test, names=feature_names, return_names=True,
                                                eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                                eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)

    cont_columns   = [col for col in feats_test.columns if col in contfeats]
    cat_columns    = [col for col in feats_test.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in feats_test.columns if col in target]
    feature_cols   = cont_columns + cat_columns

    # Create output figure directory -------------------------------------------------------------------------------------#
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    # Feature Importance --------------------------------------------------------------------------------------------------#
    logger.info("Extracting feature importances from loaded models...")

    feature_importances = []
    for fold, model in enumerate(trained_models):
        try:
            feat_importance = model.get_feature_importance()
            feature_importances.append(feat_importance)
            logger.info(f"Feature importance extracted for fold {fold + 1}")
        except Exception as e:
            logger.warning(f"Could not extract feature importance for fold {fold + 1}: {e}")
            feature_importances.append(None)

    valid_importances = [fi for fi in feature_importances if fi is not None]

    if valid_importances:
        # Aggregate importances across folds
        all_features           = list(valid_importances[0].keys())
        importances_by_feature = {feat: [] for feat in all_features}

        for fold_importance in valid_importances:
            for feat, val in fold_importance.items():
                importances_by_feature[feat].append(val)

        importances_by_feature = {k: np.array(v) for k, v in importances_by_feature.items()}

        importance_stats = {
            feat: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'values': v.tolist()}
            for feat, v in importances_by_feature.items()
        }

        # Save feature importance summary
        importance_path = latest_results / "feature_importance.yaml"
        with open(importance_path, 'w') as f:
            yaml.dump(importance_stats, f, default_flow_style=False)
        logger.info(f"Feature importance saved to: {importance_path}")

        sorted_features = sorted(importance_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]
        logger.info("Top 5 most important features:")
        for feat, stats in sorted_features:
            logger.info(f"  - {feat}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Build LaTeX display names aligned with importance dict keys
        latex_names = [feats_labels.get(feat, feat) for feat in importances_by_feature.keys()]
        model_title = {"elasticnet": "ENet", "linearsvr": "LinearSVR"}

        # Plot feature importance bar chart
        plot_generator.plot_feature_importance_bars(importances_dict = importances_by_feature,
                                                    path_save        = str(viz_path),
                                                    name_file        = model_type,
                                                    model_name       = model_title[model_type],
                                                    importance_name  = "Coefficients",
                                                    features_names   = latex_names)
        logger.info("Feature importance plot saved.")

    else:
        logger.warning("No feature importances could be extracted from the loaded models.")

    # Simulation Example Plots --------------------------------------------------------------------------------------------#
    logger.info(f"Selecting {n_simulations} random simulations from the test set...")

    sim_col = "or_sim_path"
    if sim_col not in filter_test.columns:
        logger.warning(f"Column '{sim_col}' not found in the test set. Skipping simulation example plots.")
        return trained_models

    sim_ids  = filter_test[sim_col].unique()
    n_select = min(n_simulations, len(sim_ids))

    if n_select == 0:
        logger.warning("No simulations available for example plots.")
        return trained_models

    n_cols   = n_select
    n_rows   = 1

    if n_select == 0:
        logger.warning("No simulations available for example plots.")
        return trained_models

    rng          = np.random.default_rng(CONFIG.seed)
    selected_ids = rng.choice(sim_ids, size=n_select, replace=False)

    subplot_data = []
    for sim_id in selected_ids:

        sim_mask  = filter_test[sim_col] == sim_id
        sim_rows  = filter_test[sim_mask].sort_values('t')
        sim_feats = feats_test.loc[sim_rows.index]

        x_time  = sim_rows['t'].to_numpy()
        y_raw   = sim_feats[target_columns[0]].astype(np.float32).to_numpy()
        X_sim   = sim_feats[feature_cols].astype(np.float32).to_numpy()

        # Build per-simulation scaler for inverse transform
        if trs_target and target_norm is not None and target_norm in sim_rows.columns:
            sim_norm   = sim_rows[target_norm].astype(np.float32).to_numpy()
            scaler_sim = TargetTransform(transformation = CONFIG.scalingconfig.target_transform,
                                         norm_factor    = sim_norm,
                                         epsilon        = CONFIG.scalingconfig.target_log_eps)
        else:
            scaler_sim = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
        
        # Inverse-transform ground truth
        y_true_phys = scaler_sim.inverse_transform(y_raw) if scaler_sim is not None else np.clip(y_raw, 0, None)

        # Per-fold predictions
        preds_dict = {}
        for fold, model in enumerate(trained_models):
            X_sim_scaled  = feature_scalers[fold].transform(X_sim) if feature_scalers[fold] is not None else X_sim
            y_pred_raw    = model.predict(X_sim_scaled)
            y_pred_phys   = scaler_sim.inverse_transform(y_pred_raw) if scaler_sim is not None else np.clip(y_pred_raw, 0, None)
            preds_dict[f"model_fold_{fold}"] = {"rescaled_pred": y_pred_phys}

        subplot_data.append({
            'xaxis'      : {'values': x_time, 'label': r"$t$ [Myr]"},
            'yaxis'      : {'true_values': y_true_phys, 'label': r"M$_{\rm{MMO}}$ [M$_\odot$]"},
            'iconds'     : {},
            'predictions': preds_dict
        })

    save_path = str(viz_path / f"{model_type}_simulation_examples.png")
    plot_generator.plot_simulation_predictions_agaist_gt(subplot_data = subplot_data,
                                                         n_rows       = n_rows,
                                                         n_cols       = n_cols,
                                                         save_path    = save_path,
                                                         figsize      = (4*n_cols, 4))
    logger.info(f"Simulation example plots saved to: {save_path}")

    return trained_models

# Pipeline Modes [Prediction] ---------------------------------------------------------------------------------------------#
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
        run_optimization(feats_path   = path_manager.data_path,
                         contfeats    = CONFIG.dataconfig.cont_feats,
                         catfeats     = CONFIG.dataconfig.cat_feats, 
                         target       = CONFIG.dataconfig.target_feat,
                         trs_target   = CONFIG.scalingconfig.trs_target,
                         target_norm  = CONFIG.scalingconfig.target_norm_column,
                         out_path     = path_manager.base_out,
                         model_type   = args.model,
                         n_folds      = CONFIG.dataconfig.n_folds,
                         study_name   = args.study_name,
                         storage      = args.storage)
    
    elif args.mode == "train":
        run_training(feats_path   = path_manager.data_path,
                     contfeats    = CONFIG.dataconfig.cont_feats,
                     catfeats     = CONFIG.dataconfig.cat_feats,
                     target       = CONFIG.dataconfig.target_feat,
                     trs_target   = CONFIG.scalingconfig.trs_target,
                     target_norm  = CONFIG.scalingconfig.target_norm_column,
                     out_path     = path_manager.base_out,
                     model_type   = args.model,
                     fig_path     = path_manager.fig_path,
                     n_folds      = CONFIG.dataconfig.n_folds)
    
    elif args.mode == "inter":
        run_interpretation(feats_path    = path_manager.data_path,
                           contfeats     = CONFIG.dataconfig.cont_feats,
                           catfeats      = CONFIG.dataconfig.cat_feats,
                           target        = CONFIG.dataconfig.target_feat,
                           trs_target    = CONFIG.scalingconfig.trs_target,
                           target_norm   = CONFIG.scalingconfig.target_norm_column,
                           out_path      = path_manager.base_out,
                           model_type    = args.model,
                           fig_path      = path_manager.fig_path,
                           n_folds       = CONFIG.dataconfig.n_folds,
                           n_simulations = args.n_sims)
        
    elif args.mode == "predict":
        run_prediction()

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#