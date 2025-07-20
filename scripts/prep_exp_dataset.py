# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import sys
import argparse

import numpy  as np
import pandas as pd

# External functions and utilities ----------------------------------------------------------------------------------------#
from loguru import logger

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.utils.directory        import list_all_directories
from src.processing.moccasurvey import mocca_survey_dataset

# Logger configuration  ---------------------------------------------------------------------------------------------------#
logger.remove()

# Add outputs to the console
logger.add(sink=sys.stdout, level="INFO", format="<level>{level}: {message}</level>")

# Add outputs to the file
logger.add("./logs/experimental_dataset_preparation.log",
           level     = "INFO",
           format    = "{time:YYYY-MM-DD HH:mm:ss} - {level}: {message}",
           rotation  = "10 MB",    # Rota el archivo al alcanzar 10 MB
           retention = "10 days",  # Elimina logs más viejos de 10 días
           encoding  = "utf-8")

# Arguments ---------------------------------------------------------------------------------------------------------------#
def get_args():
    parser = argparse.ArgumentParser(description="Preparation of the simulated dataset")
    
    # Directories
    parser.add_argument("--root_dir", type=str, default="./rawdata/", 
                        help="Root directory of the data without preprocess.")
    parser.add_argument("--out_dir", type=str, default="./datasets/", 
                        help="Directory to store the processed dataset.")
    parser.add_argument("--fig_dir", type=str, default="./figures/", 
                        help="Directory to store output figures of the analysis.")

    # Taks flags
    parser.add_argument("--create_files", action="store_true",
                        help="If true, then dataset files are computed and replaced")
    parser.add_argument("--statistics", action="store_true",
                        help="If true, a statistic summary is given on the 0_fold train file of the selected dataset")
    parser.add_argument("--plot_study", action="store_true",
                        help="If true, plot figures about the dataset are delivered on the 0_fold train file")

    # Dataset specifics
    parser.add_argument("--dataset", type=str, default="moccasurvey", 
                        choices = ["moccasurvey"],
                        help    = "Specific dataset to implement.")
    parser.add_argument("--exp_name", type=str, default="pof",
                        help="Tag to name the dataset and output related elements.")
    parser.add_argument("--folds", type=int, default=1,
                        help="number of folds to retrieve for kfold cross-validation.")
    
    # Target specifics
    parser.add_argument("--exp_type", type=str, default="point_mass", 
                        choices = ["point_mass", "delta_mass", "mass_rate"], 
                        help    = "Specific target expected of the dataset, also affect the possible features selected.")
    parser.add_argument("--norm_target", type=bool, default=False, 
                        help="If normalize the target by the initial total stellar.")
    parser.add_argument("--log_target", type=bool, default=False, 
                        help="If use a logarithmic scale in the target")
    
    args = parser.parse_args() 

    return args

# Main script -------------------------------------------------------------------------------------------------------------#
def main():
    
    # Retrieve arguments --------------------------------------------------------------------------------------------------#
    args = get_args()

    logger.info(f"Initiating experimental dataset preparation: {args.dataset}")

    # Use rawfiles for dataset creation -----------------------------------------------------------------------------------#
    if args.create_files:
        
        #------------------------------------------------------------------------------------------------------------------#
        try:
    
            # Retrieve moccasurvey dataset --------------------------------------------------------------------------------#
            if args.dataset == "moccasurvey":
                
                # Read all respective directory files
                args.sim_dir  = f"{args.root_dir}{args.dataset}/simulations/"
                simulations    = list_all_directories(args.sim_dir)

                logger.info(f"Total number of simulations available: {len(simulations)}")

                # Create partitions, simulation wise level. Convert to numpy array for easy indexing
                simulations  = np.array(simulations)
                n_total      = len(simulations)
                shuffled_idx = np.random.permutation(n_total)

                # Compute split sizes (70 % training, 20% validation, 10% testing)
                n_train, n_val = int(n_total * 0.7), int(n_total * 0.2)
                n_test         = n_total - n_train - n_val  

                # Partition indices
                train_idx = shuffled_idx[:n_train]
                val_idx   = shuffled_idx[n_train:n_train + n_val]
                test_idx  = shuffled_idx[n_train + n_val:]

                # Partition lists
                train_simulations = simulations[train_idx].tolist()
                val_simulations   = simulations[val_idx].tolist()
                test_simulations  = simulations[test_idx].tolist()

                logger.info("Number of simulations used for each partition")
                logger.info(f"Training   : {len(train_simulations)}")
                logger.info(f"Validation : {len(val_simulations)}")
                logger.info(f"Testing    : {len(test_simulations)}")

                # Create directory path to save information
                exp_path = os.path.join(args.out_dir, f"{args.dataset}/{args.exp_name}/")                       
                os.makedirs(exp_path, exist_ok=True)

                # Create training / validation folds
                for fold in range(0, args.folds, 1):
                    
                    fold_path = os.path.join(exp_path, f"{fold}_fold/")
                    os.makedirs(fold_path, exist_ok=True)

                    xtrain_info, ytrain_info = mocca_survey_dataset(simulations_path = train_simulations,
                                                experiment_type  = args.exp_type,
                                                norm_target      = args.norm_target,
                                                log10_target     = args.log_target,
                                                logger           = logger)

                    # Store training dataset in a dataframe (features and target)
                    df = pd.DataFrame(data=xtrain_info[0], columns=xtrain_info[1], index=None,)
                    df[ytrain_info[1][0]] = ytrain_info[0]

                    # Save into .csv file
                    df.to_csv(f"{fold_path}train.csv", index=False)
                    logger.info(f"Fold {fold} Train - Stored at {fold_path}")

                    xval_info, yval_info = mocca_survey_dataset(simulations_path = val_simulations,
                                            experiment_type  = args.exp_type,
                                            norm_target      = args.norm_target,
                                            log10_target     = args.log_target,
                                            logger           = logger)

                    # Store validation dataset in a dataframe (features and target)
                    df = pd.DataFrame(data=xval_info[0], columns=xval_info[1], index=None,)
                    df[yval_info[1][0]] = yval_info[0]

                    # Save into .csv file
                    df.to_csv(f"{fold_path}val.csv", index=False)
                    logger.info(f"Fold {fold} Val - Stored at {fold_path}")
                
                # Load testing partition
                xtest_info, ytest_info, sim_paths = mocca_survey_dataset(simulations_path = test_simulations,
                                                    experiment_type  = args.exp_type,
                                                    norm_target      = args.norm_target,
                                                    log10_target     = args.log_target,
                                                    logger           = logger,
                                                    test_partition   = True)
                
                # Store testing dataset in a dataframe (features and target)
                df = pd.DataFrame(data=xtest_info[0], columns=xtest_info[1], index=None,)
                df[ytest_info[1][0]] = ytest_info[0]
                df["or_sim_path"]    = sim_paths
                df["tag"]            = args.dataset 
                
                 # Save into .csv file
                df.to_csv(f"{exp_path}test.csv", index=False)
                logger.info(f"Test - Stored at {exp_path}")
            
            logger.success("Creation of the dataset completed")

        except Exception as e:
            logger.error(f"Error while genereting the dataset: {e}")
    
    # Create a statistic display of the features and the target -----------------------------------------------------------#
    if args.statistics:
        
        try:
            # Load the 0_fold training data
            exp_path   = os.path.join(args.out_dir, f"{args.dataset}/{args.exp_name}/")
            fold_path  = os.path.join(exp_path, "0_fold/")
            train_file = os.path.join(fold_path, "train.csv")
            
            if not os.path.exists(train_file):
                logger.error(f"Training file not found: {train_file}")
                logger.error("Please run with --create_files first to generate the dataset")
                return
            
            # Load the training data
            df = pd.read_csv(train_file, index_col=None)
            logger.info(f"Loaded training data from: {train_file}")
            logger.info(f"Dataset shape: {df.shape}")
            
            # Separate features and target
            target_col   = df.columns[-1]  # Last column is the target
            feature_cols = df.columns[:-1]  # All columns except the last are features
            
            logger.info(f"Target column: {target_col}")
            logger.info(f"Feature columns: {list(feature_cols)}")
            
            # Display descriptive statistics for features
            logger.info("_" * 100)
            logger.info("DESCRIPTIVE STATISTICS - FEATURES")
            logger.info("_" * 100)
            
            for col in feature_cols:
                logger.info(f"\nFeature: {col}")
                logger.info(f"  Count : {df[col].count()}")
                logger.info(f"  Mean  : {df[col].mean():.6f}")
                logger.info(f"  Std   : {df[col].std():.6f}")
                logger.info(f"  Min   : {df[col].min():.6f}")
                logger.info(f"  25%   : {df[col].quantile(0.25):.6f}")
                logger.info(f"  50%   : {df[col].quantile(0.50):.6f}")
                logger.info(f"  75%   : {df[col].quantile(0.75):.6f}")
                logger.info(f"  Max   : {df[col].max():.6f}")
                logger.info("_"*100)

            # Display descriptive statistics for target
            logger.info("_" * 100)
            logger.info("DESCRIPTIVE STATISTICS - TARGET")
            logger.info("_" * 100)
            
            logger.info(f"\nTarget: {target_col}")
            logger.info(f"  Count : {df[target_col].count()}")
            logger.info(f"  Mean  : {df[target_col].mean():.6f}")
            logger.info(f"  Std   : {df[target_col].std():.6f}")
            logger.info(f"  Min   : {df[target_col].min():.6f}")
            logger.info(f"  25%   : {df[target_col].quantile(0.25):.6f}")
            logger.info(f"  50%   : {df[target_col].quantile(0.50):.6f}")
            logger.info(f"  75%   : {df[target_col].quantile(0.75):.6f}")
            logger.info(f"  Max   : {df[target_col].max():.6f}")
            logger.info("_" * 100)

            # Additional summary statistics
            logger.info("_" * 100)
            logger.info("SUMMARY STATISTICS")
            logger.info("_" * 100)
            
            logger.info(f"Total number of data points : {len(df)}")
            logger.info(f"Number of features          : {len(feature_cols)}")
            logger.info(f"Number of targets           : 1")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            logger.info("_" * 100)
            
            # Check for missing values
            missing_features = df[feature_cols].isnull().sum()
            missing_target   = df[target_col].isnull().sum()
            
            if missing_features.sum() > 0:
                logger.warning("Missing values found in features:")
                for col, missing in missing_features.items():
                    if missing > 0:
                        logger.warning(f"  {col}: {missing} missing values")
            else:
                logger.info("No missing values found in features")
                
            if missing_target > 0:
                logger.warning(f"Missing values found in target: {missing_target}")
            else:
                logger.info("No missing values found in target")
            
            logger.success("Statistics display completed")
            
        except Exception as e:
            logger.error(f"Error while displaying statistics: {e}")
            logger.error("Please ensure the dataset has been created with --create_files first")

    # Plot some figures from the dataset ----------------------------------------------------------------------------------#
    if args.plot_study:
        print("not yet implemented")
    
    
#--------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------------------#