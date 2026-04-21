# Modules -----------------------------------------------------------------------------------------------------------------#
import sys
import warnings
import argparse
import yaml
import torch
import optuna

import numpy             as np
import pandas            as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from pathlib     import Path
from typing      import Optional
from datetime    import datetime
from captum.attr import GradientShap

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory            import PathManagerTrainOptPipeline, build_dl_architecture_from_YAML, load_yaml_dict
from src.processing.features        import filter_simulation_artifacts, tabular_features
from src.processing.scalers         import TargetTransform, FeatureScaler
from src.optim.optimizer            import SpaceSearchConfig, SpaceSearch
from src.optim.grid                 import MLP_PARAM_GROUPS, NODE_PARAM_GROUPS
from src.utils.eval                 import compute_metrics
from src.processing.modules.plots   import PlotGenerator
from src.models.dltab.regressor     import DLTabularRegressor 
from src.models.dltab.data.datasets import LEMONscDataManager
from src.utils.resources            import set_numpy_torch_seed
from jobs.config._dltab             import JobConfig

# Warnings managment ------------------------------------------------------------------------------------------------------#
warnings.filterwarnings("ignore", category=UserWarning)

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")
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
    parser.add_argument("--mode", type=str, default="optim", choices=["optim", "train", "inter", "predict"],
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
                       choices=["mlp", "node"],
                       help="Type of model to use")
    parser.add_argument("--n_sims", type=int, default=2,
                        help = "Number of random simulations to plot in interpretation mode.")
    
    # DL architecture and optimizer config files
    parser.add_argument("--arch_config", type=str, default="mlp.yaml",
                        help="Architecture YAML file name (from src/models/dltab/config/arch/)")
    parser.add_argument("--opt_config", type=str, default="adam.yaml",
                        help="Optimizer YAML file name (from src/models/dltab/config/opt/)")
    parser.add_argument("--loss_config", type=str, default="huber.yaml",
                        help="Loss YAML file name (from src/models/dltab/config/loss/).")
    parser.add_argument("--schd_config", type=str, default=None,
                        help="Scheduler YAML file name (from src/models/dltab/config/schd/).")

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
def run_optimization(feats_path: str, contfeats: list, catfeats: list, target: list, out_path: str, model_type: str,
                     dl_architecture : dict,
                     trs_target      : bool            = True,
                     target_norm     : Optional[str]   = None,
                     n_folds         : int             = 3,
                     study_name      : Optional[str]   = None,
                     storage         : Optional[str]   = None):
    """Run the optimization mode pipeline for DL tabular models."""
    
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
            
    # Verbose
    logger.info(110*"_")
    logger.info(f"Space search of the params for the dltab {model_type} regressor using {n_folds}-fold cross-validation")
    logger.info(110*"_")
    logger.info("Loading and preparing data partitions via DataLoaders...")
    
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
    
    # Define transform function: filter artifacts then engineer features
    transform_fn = lambda df: tabular_features(
        filter_simulation_artifacts(df, min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                    filter_null_mass     = True, 
                                    filter_initial_state = False, 
                                    verbose              = False
                                    ),
        names                      = feature_names, 
        return_names               = False,
        eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
        eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range
        )
    
    # Process each fold: create DataLoaders for SpaceSearch
    partitions = []
    
    for fold in range(n_folds):
        
        # Create fold directory
        foldpath = study_path/ f"{study_name}/fold_{fold}"
        foldpath.mkdir(parents=True, exist_ok=True)

        # Set fold seed for reproducibility (if applicable, use different seed per fold to avoid identical runs)
        foldseed = set_numpy_torch_seed(seed_num=CONFIG.seed, idx=fold)
        
        # Configure the scalers for the data
        should_scale  = CONFIG.scalingconfig.scale_features and CONFIG.scalingconfig.scale_target
        scaler_kwargs = {
            "feats"  : {
                "scaler_name"   : CONFIG.scalingconfig.feature_scaler_name,
                "scaler_kwargs" : CONFIG.scalingconfig.feature_scaler_kward},
            "target" : {
                "scaler_name"   : CONFIG.scalingconfig.target_scaler_name,
                "scaler_kwargs" : CONFIG.scalingconfig.target_scaler_kward},
            }

        # Use LEMONscDataManager to load datasets and create DataLoaders for the current fold
        data_manager = LEMONscDataManager(
            dataset_root     = feats_path,
            fold             = fold,
            target_column    = target_columns[0],
            feature_columns  = feature_cols,
            metadata_columns = [target_norm] if target_norm is not None else None,
            transform_fn     = transform_fn,
            batch_size       = CONFIG.dataconfig.batch_size,
            num_workers      = CONFIG.dataconfig.num_workers,
            seed             = foldseed,
            device           = CONFIG.device,
            logger           = logger,
            scale            = should_scale,
            scaler_kwargs    = scaler_kwargs,
            scaler_dir       = foldpath
            )
        
        # Load train and validation partitions (no test needed for optimization)
        data_manager.setup(load_train=True, load_val=True, load_test=False)
        
        # Retrieve the feature and the target scaler
        feat_scaler   = data_manager.train_dataset.feat_scaler
        target_scaler = data_manager.train_dataset.target_scaler
        
        # Ensure that the validation dataset has the required metadata for denormalization (M_tot) and extract it
        if ( trs_target                                                                       and
             (data_manager.val_dataset and hasattr(data_manager.val_dataset, '_metadata_df')) and
             (data_manager.val_dataset._metadata_df is not None)                              and
             (target_norm is not None)                                                        and
             (target_norm in data_manager.val_dataset._metadata_df.columns)
             ):
            
            # Ensure that M_tot is numeric and extract it as a numpy array flattened to 1D
            val_norm = data_manager.val_dataset._metadata_df[target_norm].astype(np.float32).to_numpy().flatten()
            val_trs  = TargetTransform(transformation = CONFIG.scalingconfig.target_transform, 
                                       norm_factor    = val_norm, 
                                       epsilon        = CONFIG.scalingconfig.target_log_eps)
        
        # Scale without normalization factor if target_norm column not found
        else:
            val_trs = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
            logger.warning("A default identity transformation will be used as configuration is not well defined.")

        # Create partition dictionary with DataLoaders (DL format expected by SpaceSearch)
        partition = {
            'train_loader'   : data_manager.train_loader,
            'val_loader'     : data_manager.val_loader,
            'trs'            : val_trs,
            'scaler'         : target_scaler,
            'features_names' : feature_cols
                    }
        partitions.append(partition)
        
        # Verbose fold data loading summary
        n_train = len(data_manager.train_dataset) if data_manager.train_dataset else 0
        n_val   = len(data_manager.val_dataset) if data_manager.val_dataset else 0
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={n_train} samples, Val={n_val} samples")

    # Initialize optimizer with DL-specific configuration
    logger.info("Initializing SpaceSearch optimizer for DL...")
    config = SpaceSearchConfig(model_type           = model_type, n_jobs = CONFIG.loaders_n_jobs, 
                               n_trials             = CONFIG.optconfig.n_trials,
                               device               = CONFIG.device,
                               seed                 = CONFIG.seed,
                               max_epochs           = CONFIG.modelconfig.max_epochs,
                               dl_use_amp           = CONFIG.modelconfig.use_amp,
                               dl_patience          = CONFIG.optconfig.trial_patience,
                               dl_loss_fn           = CONFIG.modelconfig.dl_loss_fn,
                               dl_architecture      = dl_architecture,
                               dl_grad_clip         = CONFIG.modelconfig.grad_clip_norm,
                               dl_scheduler_name    = dl_architecture.get('scheduler_name', None),
                               dl_scheduler_params  = dl_architecture.get('scheduler_params', None),
                               storage              = storage,
                               load_if_exists       = storage is not None)
    
    # Generate the optimizer instance with the provided configuration
    optimizer = SpaceSearch(config)
    
    # Log configuration before starting optimization
    logger.info(f"Starting optimization : {study_name}")
    logger.info(f"  - Storage           : {storage if storage is not None else 'in-memory (no persistence)'}")
    logger.info(f"  - Load if exists    : {storage is not None}")
    logger.info(f"  - Model             : {model_type}")
    logger.info(f"  - Device            : {CONFIG.device}")
    logger.info(f"  - Trials            : {CONFIG.optconfig.n_trials}")
    logger.info(f"  - Direction         : {CONFIG.optconfig.direction}")
    logger.info(f"  - Metric            : {CONFIG.optconfig.metric}")
    logger.info(f"  - Lambda penalty    : {CONFIG.optconfig.lambda_penalty}")
    logger.info(f"  - Patience          : {CONFIG.optconfig.trial_patience}")
    logger.info(f"  - Max epochs        : {CONFIG.modelconfig.max_epochs}")
    logger.info(f"  - DL patience       : {CONFIG.modelconfig.train_es_patience}")
    logger.info(f"  - DL loss fn        : {CONFIG.modelconfig.dl_loss_fn}")
    logger.info(f"  - Batch size        : {CONFIG.dataconfig.batch_size}")
    logger.info(f"  - Architecture      : {dl_architecture}")
    
    try:
        # Hybrid pruning: fold 0 uses epoch-level steps [0, max_epochs-1], folds 1+ use fold-level steps 
        _max_resource = CONFIG.modelconfig.max_epochs + max(0, CONFIG.dataconfig.n_folds - 1)
        
        # Min_resource: earliest step where the pruner may act (within fold 0 epoch range)
        _min_resource = max(1, CONFIG.modelconfig.max_epochs // 4)
        
        # Construct the Hyperband pruner with the specified resource limits and reduction factor
        pruner = optuna.pruners.HyperbandPruner(min_resource     = _min_resource,
                                                max_resource     = _max_resource,
                                                reduction_factor = 3)
        
        # Verbose
        logger.info("_"*110)
        logger.info(f"  - Pruner: HyperbandPruner(min={_min_resource}, "
                    f"max={_max_resource}, r=3) | hybrid: epoch-level fold 0, fold-level folds 1+")
        logger.info("_"*110)
        
        # Run the optimization process with the defined partitions, study name, direction, metric, and pruner
        results = optimizer.optimize(partitions= partitions, study_name= study_name, direction= CONFIG.optconfig.direction, 
                                     metric         = CONFIG.optconfig.metric,
                                     output_dir     = str(study_path),
                                     save_study     = True,
                                     patience       = CONFIG.optconfig.trial_patience,
                                     pruner         = pruner,
                                     lambda_penalty = CONFIG.optconfig.lambda_penalty
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

        # Reconstruct nested best_params from flat Optuna dict using param group mappings
        param_groups_map = {"mlp": MLP_PARAM_GROUPS, "node": NODE_PARAM_GROUPS}
        flat_params      = results.best_params
        
        # Depending of model type, reconstruct the nested structure of best_params
        if model_type in param_groups_map:
            
            # Select the appropriate group mapping for the current model type
            groups = param_groups_map[model_type]
            
            # Generate the structured best_params by grouping flat_params according to the defined groups
            structured_best_params = {
                group: {k: flat_params[k] for k in keys if k in flat_params} for group, keys in groups.items()}
        
        # If the model type is not in the param_groups_map, use the flat_params as is
        else:
            structured_best_params = flat_params

        # Set the summary data to be saved in the optimization summary YAML file
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
                'direction'            : CONFIG.optconfig.direction,
                'metric'               : CONFIG.optconfig.metric,
                'lambda_penalty'       : CONFIG.optconfig.lambda_penalty,
                'trial_patience'       : CONFIG.optconfig.trial_patience,
                'scale features'       : CONFIG.scalingconfig.scale_features,
                'scaler features name' : CONFIG.scalingconfig.feature_scaler_name,
                'scale target'         : CONFIG.scalingconfig.scale_target,
                'scaler target name'   : CONFIG.scalingconfig.target_scaler_name,
                'max_epochs'           : CONFIG.modelconfig.max_epochs,
                'train_patience'       : CONFIG.modelconfig.train_es_patience,
                'dl_loss_fn'           : CONFIG.modelconfig.dl_loss_fn,
                'batch_size'           : CONFIG.dataconfig.batch_size,
                'device'               : CONFIG.device,
                'seed'                 : CONFIG.seed
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
                 trs_target      : bool,
                 target_norm     : Optional[str],
                 n_folds         : int):
    """Run the training mode pipeline for DL tabular models."""
    
    # Create unique training timestamp and create result directory
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(out_path) / f"training_results_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create the plot generator
    plot_generator = PlotGenerator(config=None, cmap="magma_r")
    
    # Verbose
    logger.info(110*"_")
    logger.info(f"DLTab {model_type} regressor training and evaluation using {n_folds}-fold cross-validation")
    logger.info(110*"_")

    # Default nested structure built from dl_architecture
    _default_model_params = {"architecture_params" : dl_architecture['model_params'],
                             "optimizer_params"    : dl_architecture['optimizer_params'],
                             "loss_params"         : dl_architecture['loss_params']}
    
    # Try Load of optimized hyperparameters if possible -------------------------------------------------------------------#
    try:
        optimization_path = Path(out_path) / "optim"
        all_study_dirs    = [d for d in optimization_path.glob("cv_study_*") if d.is_dir()]

        # Keep only dirs that have a valid summary for the current model_type
        study_dirs = []
        for d in all_study_dirs:
            summary_file = d / "optimization_summary.yaml"
            if summary_file.exists():
                try:
                    s = load_yaml_dict(str(summary_file))
                    if s.get("model_type") == model_type:
                        study_dirs.append(d)
                except Exception:
                    pass
        
        # If name don't specified, take last optimization available
        if study_dirs:

            latest_study = max(study_dirs, key=lambda x: (x / "optimization_summary.yaml").stat().st_mtime)
            opt_summary  = load_yaml_dict(str(latest_study / "optimization_summary.yaml"))
            model_params = opt_summary['best_params']

            logger.info(110*"_")
            logger.info(f"Using optimized parameters from: {latest_study.name}")
            logger.info(110*"_")
            
        # Else, use the default parameters defined by the dl_architecture config  
        else:
            model_params = _default_model_params
            
            logger.info(110*"_")
            logger.info(f"No optimization study found for model_type='{model_type}'. "
                        "Using default architecture parameters.")
            logger.info(110*"_")
            
    # Catch any exception during the loading of optimization results and fallback to default parameters
    except Exception as e:
        logger.warning(f"Error loading parameters: {e}")
        logger.warning("Falling back to default architecture parameters.")
        model_params = _default_model_params
    
    # Load and prepare data partitions via LEMONscDataManager -------------------------------------------------------------#
    logger.info("Loading and preparing data partitions...")
    
    # Handle None categorical features
    catfeats = catfeats if catfeats is not None else []
    
    # Define feature names for job
    feature_names = contfeats + catfeats + target
    
    # Determine column dataframe names and filter numerical artifacts (use first fold as reference)
    fold_0_path = feats_path + "/0_fold/train.csv"
    temp_df     = pd.read_csv(fold_0_path, index_col=False)
    
    temp_filt   = filter_simulation_artifacts(temp_df, min_denominator_threshold= CONFIG.dataconfig.min_dem_thr,
                                              filter_null_mass     = True,
                                              filter_initial_state = False, 
                                              verbose              = False)
    
    temp_feats  = tabular_features(temp_filt, names= feature_names, return_names= False,
                                   eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                   eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)
    
    # Identify transformed column names and ensure desired features
    cont_columns   = [col for col in temp_feats.columns if col in contfeats]
    cat_columns    = [col for col in temp_feats.columns if any(col.startswith(cf+"_") for cf in catfeats)]
    target_columns = [col for col in temp_feats.columns if col in target]
    feature_cols   = cont_columns + cat_columns
    
    # Verbose
    logger.info("Features configuration:")
    logger.info(f"  - Continuous features  : {contfeats} -> {cont_columns}")
    logger.info(f"  - Categorical features : {catfeats}  -> {cat_columns}")
    logger.info(f"  - Target               : {target}    -> {target_columns}")
    logger.info(f"  - Total features       : {len(feature_cols)}")
    
    # Define transform function for the dataloader: filter artifacts then engineer features 
    transform_fn = lambda df: tabular_features(
        filter_simulation_artifacts(df, min_denominator_threshold = CONFIG.dataconfig.min_dem_thr,
                                    filter_null_mass     = True, 
                                    filter_initial_state = False, 
                                    verbose              = False
                                    ),
        names                      = feature_names, 
        return_names               = False,
        eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
        eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range
        )
    
    # Load test set and build target scaler -------------------------------------------------------------------------------#
    test_path  = feats_path + "/test.csv"
    test_df    = pd.read_csv(test_path, index_col=False)

    # Filter the artifacts and create the input/target vector
    filt_test  = filter_simulation_artifacts(test_df, min_denominator_threshold= CONFIG.dataconfig.min_dem_thr,
                                             filter_null_mass     = True, 
                                             filter_initial_state = False)
    feats_test = tabular_features(filt_test, names= feature_names, return_names= False,
                                  eps_logscale_all_range     = CONFIG.scalingconfig.feature_log_eps_all_range,
                                  eps_logscale_limited_range = CONFIG.scalingconfig.feature_log_eps_lim_range)
    y_test     = feats_test[target_columns].astype(np.float32).to_numpy().flatten() 
    
    # Create target transformer for the test set (recover physical units and keep track of the ratio)
    if trs_target and target_norm is not None and target_norm in filt_test.columns:
        test_norm   = filt_test[target_norm].astype(np.float32).to_numpy().flatten()
        test_trs    = TargetTransform(transformation = CONFIG.scalingconfig.target_transform, 
                                      norm_factor    = test_norm, 
                                      epsilon        = CONFIG.scalingconfig.target_log_eps)
        y_test_phys = test_trs.inverse_transform(y_test)
    
    # If none information, use identity transformation
    else:        
        # Verbose warning
        logger.warning(f"A default identity transformation will be used as configuration is not well defined.")
        
        test_trs    = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
        y_test_phys = test_trs.inverse_transform(y_test)
    
    # Verbose warning 
    logger.warning(f"The scaler is not integrated in LEMONscDataManager for the test set. "
                   "Ensure that the test set maintains the same order to ensure proper evaluation.")
    
    # Train models per fold -----------------------------------------------------------------------------------------------#
    results        = []
    predictions    = []
    trained_models = []
        
    for fold in range(n_folds):
        
        # Create fold directory
        foldpath = results_path / f"fold_{fold}"
        foldpath.mkdir(parents=True, exist_ok=True)

        # Set fold seed for reproducibility (if applicable, use different seed per fold to avoid identical runs)
        foldseed = set_numpy_torch_seed(seed_num=CONFIG.seed, idx=fold)
        
        # Verbose
        logger.info(f"Processing fold {fold + 1}/{n_folds}")
        
        # Configure the scalers for the data
        should_scale  = CONFIG.scalingconfig.scale_features and CONFIG.scalingconfig.scale_target
        scaler_kwargs = {
                        "feats"  : {
                            "scaler_name"   : CONFIG.scalingconfig.feature_scaler_name,
                            "scaler_kwargs" : CONFIG.scalingconfig.feature_scaler_kward},
                        "target" : {
                            "scaler_name"   : CONFIG.scalingconfig.target_scaler_name,
                            "scaler_kwargs" : CONFIG.scalingconfig.target_scaler_kward},
                        }
    
        # Load fold data using LEMONscDataManager
        data_manager = LEMONscDataManager(
            dataset_root     = feats_path,
            fold             = fold,
            target_column    = target_columns[0],
            feature_columns  = feature_cols,
            metadata_columns = [target_norm] if target_norm is not None else None,
            transform_fn     = transform_fn,
            batch_size       = CONFIG.dataconfig.batch_size,
            num_workers      = CONFIG.dataconfig.num_workers,
            seed             = foldseed,
            device           = CONFIG.device,
            logger           = logger,
            scale            = should_scale,
            scaler_kwargs    = scaler_kwargs,
            scaler_dir       = foldpath
            )
        
        # Load train, validation and test partitions
        data_manager.setup(load_train=True, load_val=True, load_test=True)
        
        # Retrieve the feature and the target scaler
        feat_scaler   = data_manager.train_dataset.feat_scaler
        target_scaler = data_manager.train_dataset.target_scaler

        # Verbose
        n_train = len(data_manager.train_dataset) if data_manager.train_dataset else 0
        n_val   = len(data_manager.val_dataset) if data_manager.val_dataset else 0
        logger.info(f"Fold {fold + 1}/{n_folds} loaded: Train={n_train} samples, Val={n_val} samples")
        
        # Merge optimized params into architecture
        fold_model_params = {**dl_architecture['model_params'], **model_params.get("architecture_params", {})}

        # For NODE: route the fold-specific seed so ODST data-aware initialization is reproducible per fold.
        if model_type == "node":
            fold_model_params['seed'] = foldseed

        fold_opt_name     = model_params.get("optimizer_name", dl_architecture['optimizer_name'])
        fold_opt_params   = {**dl_architecture['optimizer_params'], **model_params.get("optimizer_params", {})}
        lossfn_params     = model_params.get("loss_params", dl_architecture['loss_params'])
        
        # Scheduler is a fixed architectural choice, not an optimized hyperparameter
        fold_schd_name    = dl_architecture.get('scheduler_name', None)
        fold_schd_params  = dl_architecture.get('scheduler_params', None)

        # Initialize the DL model
        model = DLTabularRegressor(model_type       = model_type,
                                   in_features      = len(feature_cols),
                                   model_params     = fold_model_params,
                                   optimizer_name   = fold_opt_name,
                                   optimizer_params = fold_opt_params,
                                   scheduler_name   = fold_schd_name,
                                   scheduler_params = fold_schd_params,
                                   feat_names       = feature_cols,
                                   device           = CONFIG.device,
                                   use_amp          = CONFIG.device =="cuda",
                                   verbose          = CONFIG.verbose)
        
        # Train the model
        logger.info(f"Training DL model on {n_train} samples (max {CONFIG.modelconfig.max_epochs} epochs)...")
        model.fit(train_loader             = data_manager.train_loader,
                  val_loader               = data_manager.val_loader,
                  epochs                   = CONFIG.modelconfig.max_epochs,
                  loss_fn                  = CONFIG.modelconfig.dl_loss_fn,
                  loss_params              = lossfn_params,
                  early_stopping_patience  = CONFIG.modelconfig.train_es_patience,
                  use_grad_clipping        = True if CONFIG.modelconfig.grad_clip_norm is not None else False,
                  grad_clip_max_norm       = CONFIG.modelconfig.grad_clip_norm,
                  verbose_epoch            = 1)
        
        # Load best weights from early stopping
        model.load_best_weights()
        trained_models.append(model)
        
        # Generate predictions on test set
        y_pred = model.predict(data_manager.test_loader)
        
        # Transform predictions to original scale physical scale
        if target_scaler is not None:
            y_pred_scaled = target_scaler.inverse_transform(y_pred)
            y_pred_phys   = test_trs.inverse_transform(y_pred_scaled)
        
        else:
            y_pred_phys = test_trs.inverse_transform(y_pred)
        
        # Calculate metrics
        fold_metrics = {'fold': fold, **compute_metrics(y_test_phys, y_pred_phys)}
        
        # Store fold metrics and predictions for later aggregation and visualization
        results.append(fold_metrics)
        predictions.append({'fold': fold, 'y_pred': y_pred_phys, 'y_true': y_test_phys})
        
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
        'architecture': dl_architecture,
        'features': {
            'continuous'  : cont_columns,
            'categorical' : cat_columns,
            'target'      : target_columns
        },
        'dataset_info': {
            'test_size'        : len(y_test),
            'transform target' : trs_target,
            'transformation'   : CONFIG.scalingconfig.target_transform,
            'target_norm'      : target_norm,
            'epsilon_target'   : CONFIG.scalingconfig.target_log_eps
        },
        'config': {
            'scale features'       : CONFIG.scalingconfig.scale_features,
            'scaler features name' : CONFIG.scalingconfig.feature_scaler_name,
            'scale target'         : CONFIG.scalingconfig.scale_target,
            'scaler target name'   : CONFIG.scalingconfig.target_scaler_name,
            'max_epochs'           : CONFIG.modelconfig.max_epochs,
            'train_patience'       : CONFIG.modelconfig.train_es_patience,
            'dl_loss_fn'           : CONFIG.modelconfig.dl_loss_fn,
            'batch_size'           : CONFIG.dataconfig.batch_size,
            'device'               : CONFIG.device,
            'seed'                 : CONFIG.seed
        },
        'metrics': {
            'per_fold'  : results,
            'aggregate' : aggregate_metrics
        }
    }
    
    # Save the summary as a YAML file for easy readability and later reference
    with open(results_path / "training_summary.yaml", 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    
    # Save trained models
    for fold, model in enumerate(trained_models):
        model.save_model(str(results_path / f"fold_{fold}/model.pt"))
    
    # Verbose final results
    logger.info("Training completed successfully!")
    logger.info("Aggregate metrics:")
    for metric, value in aggregate_metrics.items():
        logger.info(f"  - {metric}: {value:.6f}")
    logger.info(f"Results saved to: {results_path}")
    
    # Generate visualizations ---------------------------------------------------------------------------------------------#
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    # Create a mapping for model titles to be used in plots (if needed)
    model_title_map = {"mlp": "MLP", "node": "NODE"}

    # Subsample predictions for visualization only 
    predictions_df_plot = predictions_df.sample(frac=0.5, random_state=CONFIG.seed).reset_index(drop=True)

    # Take the mean of all folds
    fold_columns = [f"y_pred_fold_{fold}" for fold in range(n_folds)]
    df_results   = predictions_df_plot[fold_columns].mean(axis=1)

    # Generate plots using the plot generator
    plot_generator.create_ml_results_plots(predictions_df_mean = df_results,
                                           true_values_df      = predictions_df_plot["y_true"],
                                           out_path            = viz_path,
                                           model_name          = model_type,
                                           model_title         = model_title_map)  

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

    X_test = feats_test[feature_cols].astype(np.float32).to_numpy()
    y_test = feats_test[target_columns].astype(np.float32).to_numpy().flatten()

    # Locate the latest training results directory -----------------------------------------------------------------------#
    training_dirs = sorted([d for d in Path(out_path).glob("training_results_*") if d.is_dir()],
                           key=lambda x: x.stat().st_mtime)

    if not training_dirs:
        raise FileNotFoundError(f"No training results found in {out_path}. Run training first.")

    latest_results = training_dirs[-1]
    logger.info(f"Loading trained models from: {latest_results.name}")

    # Load trained models and feature scalers from all folds ----------------------------------------------------------#
    trained_models  = []
    train_folds     = []
    feature_scalers = []
    target_scalers  = []
    should_scale    = CONFIG.scalingconfig.scale_features

    for fold in range(n_folds):
        
        # Load traning partitions per fold
        train_data_path = feats_path + f"/{fold}_fold/train.csv"
        train_df        = pd.read_csv(train_data_path, index_col=False)

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

        train_folds.append({'X_train': X_train, 'y_train': y_train})

        # Load trained models from the latest training results directory
        model_path = latest_results / f"fold_{fold}/model.pt"
        model      = DLTabularRegressor.load_model(str(model_path))
        trained_models.append(model)

        # Load per-fold feature scaler saved during training
        if should_scale:
            scaler_path  = latest_results / f"fold_{fold}" 
            
            feat_scaler   = FeatureScaler(scaler_name      = CONFIG.scalingconfig.feature_scaler_name,
                                          scaler_kwargs    = CONFIG.scalingconfig.feature_scaler_kward,
                                          load_scaler_from = str(scaler_path / "feats_scaler.joblib"))
            target_scaler = FeatureScaler(scaler_name      = CONFIG.scalingconfig.target_scaler_name,
                                          scaler_kwargs    = CONFIG.scalingconfig.target_scaler_kward,
                                          load_scaler_from = str(scaler_path / "target_scaler.joblib"))
            
            feature_scalers.append(feat_scaler)
            target_scalers.append(target_scaler)

            logger.info(f"Loaded model and scaler for fold {fold + 1}/{n_folds}")
        else:

            feature_scalers.append(None)
            target_scalers.append(None)
            logger.info(f"Loaded model for fold {fold + 1}/{n_folds}")
    
    # Feature Importance (Expected Gradients via GradientShap) -----------------------------------------------------------#
    logger.info("Computing feature importance (Expected Gradients / GradientShap) from trained models...")
    
    # Select random elements from the test set
    idx_inputs        = np.random.choice(len(X_test), CONFIG.interconfig.n_explain, replace=False)
    subset_inputs_raw = X_test[idx_inputs] 

    # Store magnitude, direciton and consistency of the features
    fold_attr_magnitude   = []  
    fold_attr_direction   = []  
    fold_attr_consistency = []  

    # Iterate through all trained models
    for fold in range(len(trained_models)):
        model       = trained_models[fold]
        feat_scaler = feature_scalers[fold]
        X_train     = train_folds[fold]['X_train']

        # Compute the expected gradients
        try:
            model.model.eval()
            gs = GradientShap(model.model)

            # Sample baseline distribution from the real training data 
            idx_base      = np.random.choice(len(X_train), CONFIG.interconfig.n_baselines, replace=False)
            baseline_raw  = X_train[idx_base]                          

            # Scale inputs and baselines independently per fold 
            if feat_scaler is not None:
                subset_inputs_scaled = feat_scaler.transform(subset_inputs_raw)   
                baseline_scaled      = feat_scaler.transform(baseline_raw)        
            else:
                subset_inputs_scaled = subset_inputs_raw
                baseline_scaled      = baseline_raw

            # Make a torch tensor
            baseline_tensor = torch.tensor(baseline_scaled, dtype=torch.float32)  

            # Batched pass to avoid GPU/memory overflow; accumulate mean |attribution| per feature
            n_features      = subset_inputs_scaled.shape[1]
            fold_abs_attr   = np.zeros(n_features, dtype=np.float64)   
            fold_raw_attr   = np.zeros(n_features, dtype=np.float64)   
            fold_sq_attr    = np.zeros(n_features, dtype=np.float64)   
            n_samples_total = 0

            # Compute through batching
            for i in range(0, len(subset_inputs_scaled), CONFIG.interconfig.batch):
                batch  = subset_inputs_scaled[i:i + CONFIG.interconfig.batch]
                inputs = torch.tensor(batch, dtype=torch.float32)

                # GradientShap stochastically samples one baseline per input from baseline_tensor each call
                attr    = gs.attribute(inputs, baselines=baseline_tensor, n_samples=CONFIG.interconfig.n_samples)
                attr_np = attr.detach().cpu().numpy()                  

                # Update the count on all features from the results
                n_samples_total += attr_np.shape[0]
                fold_abs_attr   += np.abs(attr_np).sum(axis=0)
                fold_raw_attr   += attr_np.sum(axis=0)
                fold_sq_attr    += (attr_np ** 2).sum(axis=0)

            # Normalize the results by the total number of samples used
            fold_abs_attr /= n_samples_total
            fold_raw_attr /= n_samples_total
            fold_sq_attr  /= n_samples_total

            # Compute std(|a|) via computational formula: sqrt(E[a²] - E[|a|]²)
            fold_std_abs_attr = np.sqrt(np.maximum(0, fold_sq_attr - fold_abs_attr ** 2))

            # Compute consistency as 1 - CV(|a|), clamped to [0, 1]
            magnitude_protected = np.where(fold_abs_attr > 1e-8, fold_abs_attr, 1.0)
            fold_consistency    = 1.0 - np.minimum(1.0, fold_std_abs_attr / magnitude_protected)

            # Store results
            fold_attr_magnitude.append(fold_abs_attr)
            fold_attr_direction.append(fold_raw_attr)
            fold_attr_consistency.append(fold_consistency)

            logger.info(f"Feature importance extracted for fold {fold + 1}")

        except Exception as e:
            logger.warning(f"Could not extract feature importance for fold {fold + 1}: {e}")

    # Average across folds that succeeded ---------------------------------------------------------------------------------#
    if not fold_attr_magnitude:
        logger.error("Feature importance could not be computed for any fold.")

    # Create output figure directory -------------------------------------------------------------------------------------#
    viz_path = Path(fig_path)
    viz_path.mkdir(parents=True, exist_ok=True)

    # Save and plot feature importances -----------------------------------------------------------------------------------#
    if fold_attr_magnitude:

        n_valid_folds = len(fold_attr_magnitude)

        # importances_by_feature
        importances_by_feature = {feat: np.array([fold_attr_magnitude[fold][j] for fold in range(n_valid_folds)])
                                  for j, feat in enumerate(feature_cols)}

        # Direction
        direction_by_feature = {feat: float(np.mean([fold_attr_direction[fold][j] for fold in range(n_valid_folds)]))
                                for j, feat in enumerate(feature_cols)}

        # Intra-fold consistency: mean CV(|a|) across folds per feature
        consistency_by_feature = {feat: float(np.mean([fold_attr_consistency[fold][j] for fold in range(n_valid_folds)]))
                                  for j, feat in enumerate(feature_cols)}

        # Summary of the results obtained
        importance_stats = {
            feat: {
                'magnitude_mean'       : float(np.mean(v)),
                'magnitude_std'        : float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                'magnitude_folds'      : v.tolist(),
                'direction_mean'       : direction_by_feature[feat],
                'fold_cv'              : float(np.std(v, ddof=1) / np.mean(v)) if (np.mean(v) > 1e-8 and len(v) > 1) else float('inf'),
                'sample_consistency'   : consistency_by_feature[feat],
                  } for feat, v in importances_by_feature.items()
                            }

        # Save feature importance summary
        importance_path = latest_results / "feature_importance.yaml"
        with open(importance_path, 'w') as f:
            yaml.dump(importance_stats, f, default_flow_style=False)
        logger.info(f"Feature importance saved to: {importance_path}")

        sorted_features = sorted(importance_stats.items(), key=lambda x: x[1]['magnitude_mean'], reverse=True)[:5]
        logger.info("Top 5 most important features (mean |EG attribution|):")
        for feat, stats in sorted_features:
            sign_str = "+" if stats['direction_mean'] > 0 else "-"
            logger.info(f"  [{sign_str}] {feat}: {stats['magnitude_mean']:.2f} ± {stats['magnitude_std']:.2f}")
            logger.info(f"       fold_cv={stats['fold_cv']:.3f}  sample_consistency={stats['sample_consistency']:.3f}")
        
        # latex_names is anchored to feature_cols (same order as importances_by_feature)
        latex_names = [feats_labels.get(feat, feat) for feat in feature_cols]
        model_title = {"mlp": "MLP", "node": "NODE"}

        plot_generator.plot_feature_importance_bars(importances_dict = importances_by_feature,
                                                    path_save        = str(viz_path),
                                                    name_file        = model_type,
                                                    model_name       = model_title.get(model_type, model_type),
                                                    importance_name  = "EG Attribution",
                                                    features_names   = latex_names,
                                                    direction_dict   = direction_by_feature)
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

    n_cols   = min(2, n_select)
    n_rows   = n_select // n_cols
    n_select = n_rows * n_cols  # Ensure exact grid fit

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

        x_time = sim_rows['t'].to_numpy()
        y_raw  = sim_feats[target_columns[0]].astype(np.float32).to_numpy()
        X_sim  = sim_feats[feature_cols].astype(np.float32).to_numpy()

        # Build per-simulation transformation of the target
        if trs_target and target_norm is not None and target_norm in sim_rows.columns:
            sim_norm = sim_rows[target_norm].astype(np.float32).to_numpy()
            trs_sim  = TargetTransform(transformation = CONFIG.scalingconfig.target_transform,
                                       norm_factor    = sim_norm,
                                       epsilon        = CONFIG.scalingconfig.target_log_eps)
        else:
            trs_sim = TargetTransform(transformation="identity", norm_factor=None, epsilon=None)
        
        # Inverse-transform ground truth
        y_true_phys = trs_sim.inverse_transform(y_raw) if trs_sim is not None else np.clip(y_raw, 0, None)

        # Per-fold predictions
        preds_dict = {}
        for fold, model in enumerate(trained_models):
            X_sim_scaled   = feature_scalers[fold].transform(X_sim)
            y_pred_raw     = model.predict(X_sim_scaled)
            y_preds_scaled = target_scalers[fold].inverse_transform(y_pred_raw) 
            y_pred_phys    = trs_sim.inverse_transform(y_preds_scaled)
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
                                                         figsize      = (8, 4))
    logger.info(f"Simulation example plots saved to: {save_path}")
    
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
    
    # Auto-select architecture config if not explicitly overridden and model is not mlp
    arch_config = args.arch_config
    if arch_config == "mlp.yaml" and args.model != "mlp":
        arch_config = f"{args.model}.yaml"
        logger.info(f"Auto-selecting architecture config: {arch_config}")
    
    # Build DL architecture from YAML config files
    dl_model_config = Path(__file__).resolve().parent.parent / "src" / "models" / "dltab" / "config"
    dl_architecture = build_dl_architecture_from_YAML(config_path_base=dl_model_config, arch_config=arch_config, 
                                                      opt_config  = args.opt_config,
                                                      schd_config = args.schd_config,
                                                      loss_config = args.loss_config)
    
    if args.mode == "optim":
        run_optimization(feats_path      = path_manager.data_path,
                         contfeats       = CONFIG.dataconfig.cont_feats,
                         catfeats        = CONFIG.dataconfig.cat_feats,
                         target          = CONFIG.dataconfig.target_feat,
                         out_path        = path_manager.base_out,
                         model_type      = args.model,
                         dl_architecture = dl_architecture,
                         trs_target      = CONFIG.scalingconfig.trs_target,
                         target_norm     = CONFIG.scalingconfig.target_norm_column,
                         n_folds         = CONFIG.dataconfig.n_folds,
                         study_name      = args.study_name,
                         storage         = args.storage)
    
    elif args.mode == "train":
        run_training(feats_path      = path_manager.data_path,
                     contfeats       = CONFIG.dataconfig.cont_feats,
                     catfeats        = CONFIG.dataconfig.cat_feats,
                     target          = CONFIG.dataconfig.target_feat,
                     out_path        = path_manager.base_out,
                     model_type      = args.model,
                     fig_path        = path_manager.fig_path,
                     dl_architecture = dl_architecture,
                     trs_target      = CONFIG.scalingconfig.trs_target,
                     target_norm     = CONFIG.scalingconfig.target_norm_column,
                     n_folds         = CONFIG.dataconfig.n_folds)
    
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