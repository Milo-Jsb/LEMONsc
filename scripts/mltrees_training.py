# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse

from loguru      import logger
from dataclasses import dataclass


# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru      import logger
from typing      import Dict, List, Tuple, Optional, Union

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.format        import tabular_features
from src.optim.optimizer          import SpaceSearch
from src.models.mltrees.regressor import MLTreeRegressor

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
    n_trials  : int = 100
    n_jobs    : int = 1
    device    : str = "cpu"
    seed      : int = 42
    direction : str = "minimize"
    metric    : str = "neg_mean_absolute_error"
    patience  : int = 20

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
    parser.add_argument("--folds", type=int, default=1,
                        help="Number of folds to use for kfold cross-validation with trained params.")
    
    # Target specifics
    parser.add_argument("--exp_type", type=str, default="point_mass", 
                        choices = ["point_mass", "delta_mass", "mass_rate"], 
                        help    = "Specific target expected of the dataset, also affect the possible features selected.")
        
    return parser.parse_args()

# Path Management ---------------------------------------------------------------------------------------------------------#
class PathManager:
    """Centralized path management for the pipeline."""
    
    def __init__(self, root_dir: str, dataset: str, exp_name: str, model:str, out_dir: str, fig_dir: str):
        self.data_path = f"{root_dir}{dataset}
        self.out_path  = f"{out_dir}{exp_name}/{dataset}/{model}/"
        self.out_figs  = f"{fig_dir}{exp_name}/{dataset}/{model}/"
        
        # Create directories
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(self.out_figs, exist_ok=True)
    

# Pipeline Modes -----------------------------------------------------------------------------------------------------------#
def run_optimization(train_feats_path:str, val_feats_path:str, contfeats:list, catfeats:list, target:list, out_path:str):
    
    """Run the the optimization mode pipeline."""
    
    logger.info(110*"_")
    logger.info(f"Space search of the hyperparameters for the MLTree {args.model} regressor")
    logger.info(110*"_")

    # Load files
    tab_data_df = pd.read_csv(datafile, index_col=False)

    # Retrieve input features to compute 
    tab_feats_df, labels = tabular_features(process_df   = tab_data_df, 
                                            names        = contfeats  + target + catfeats, 
                                            return_names = True) 

    logger.info(f"Features retrieved")
    logger.info(f"  - Continuos features   : {contfeats}")
    logger.info(f"  - Categorical features : {catfeats}")
    logger.info(f"  - Target               : {target}")

    results = optimizer.run_study(X_train=X_train, y_train=y_train, X_val=X_val,y_val=y_val, 
                                  study_name = "optuna_trial",
                                  output_dir = "./outputs/massive_set/FULL/lightgbm/",
                                  direction  = "minimize", 
                                  metric     = "neg_mean_absolute_error",
                                  save_study = True,
                                  patience   = 20,
                                  scaler     = val_df["M_tot"])
def run_training():

def run_prediction():


# Main Pipeline -----------------------------------------------------------------------------------------------------------#
def run_pipeline(args):
    """Main pipeline orchestrator."""
    # Setup path manager
    path_manager = PathManager(args.root_dir, args.dataset, args.exp_name, args.out_dir, args.fig_dir)
    
    # Run appropriate mode
    if args.mode == "optim":

        run_optimization(train_feats  = f"{path_manager.data_path}0_fold/train.csv",
                         contfeats = ["log(t)", "log(t_coll/t_cc)" ,"M_tot/M_crit", "log(rho(R_h))", "log(R_h/R_core)"],
                         catfeats  = ["type_sim"], 
                         target    = ["M_MMO/M_tot"],
                         outfigs   = path_manager.out_figs)
    
    elif args.mode == "train":
        run_training()  
    
    elif args.mode == "predict":
        run_prediction()

# Run ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
#--------------------------------------------------------------------------------------------------------------------------#