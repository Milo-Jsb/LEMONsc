# ElasticNet grid to optimize search --------------------------------------------------------------------------------------#
def ElasticNetGrid(trial):
    param_grid = {
        "alpha"     : trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        "l1_ratio"  : trial.suggest_float("l1_ratio", 0.0, 1.0),
                 }
    
    return param_grid

# Support Vector Regressor grid to optimize search -------------------------------------------------------------------------#
def SVRGrid(trial):
    param_grid = {
        "kernel"    : trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        "C"         : trial.suggest_float("C", 1e-3, 100.0, log=True),
        "gamma"     : trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "epsilon"   : trial.suggest_float("epsilon", 0.01, 1.0, log=True),
                 }
    
    return param_grid

# Random forest grid to optimize search -----------------------------------------------------------------------------------#
def RandomForestGrid(trial):
    
    param_grid = {
        "n_estimators"           : trial.suggest_int("n_estimators", 50, 1000, step=50),
        "max_depth"              : trial.suggest_int("max_depth", 3, 30),
        "min_samples_split"      : trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf"       : trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features"           : trial.suggest_float("max_features", 0.1, 1.0),
        "max_leaves"             : trial.suggest_int("max_leaves", 1, 50, step=10),                 
                 }
    
    return param_grid

# XGBoost parameter grid to optimize search -------------------------------------------------------------------------------#
def XGBoostGrid(trial):

    param_grid = {
        "objective"              : "reg:pseudohubererror",
        "huber_slope"            : trial.suggest_float("huber_slope", 0.2, 3.0, step=0.1),
        "learning_rate"          : trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "min_child_samples"      : trial.suggest_int("min_child_samples", 5, 100),
        "bagging_fraction"       : trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "n_estimators"           : trial.suggest_int("n_estimators", 100, 2000, step=100),
        "lambda_l1"              : trial.suggest_float("lambda_l1", 0.0, 5.0),
                 }

    return param_grid

# LightGBM parameter grid to optimze search -------------------------------------------------------------------------------#
def LightGBMGrid(trial):
    
    param_grid = {
        "objective"              : "huber",
        "alpha"                  : trial.suggest_float("alpha", 0.2, 0.3, step=0.1),
        "learning_rate"          : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves"             : trial.suggest_int("num_leaves", 16, 256, step=16),
        "min_child_samples"      : trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction"       : trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction"       : trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq"           : trial.suggest_int("bagging_freq", 1, 10),
        "n_estimators"           : trial.suggest_int("n_estimators", 100, 2000, step=100),
        "max_depth"              : trial.suggest_int("max_depth", 3, 15),
        "lambda_l1"              : trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2"              : trial.suggest_float("lambda_l2", 0.0, 5.0),
                 }
    
    return param_grid

# MultiLayer Perceptron grid to optimize search ---------------------------------------------------------------------------#
def MLPGrid(trial):
    param_grid = { 
        "model_params": { 
            "dropout"       : trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
            "activation"    : trial.suggest_categorical("activation", ["relu", "tanh", "elu"]),
            "normalization" : trial.suggest_categorical("normalization", ["batch", "layer", None])},
        
        "optimizer_params": {
            "lr"            : trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay"  : trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)},
        
        "loss_params": {
            "reduction" : trial.suggest_categorical("reduction", ["mean", "sum"]),
            "delta"     : trial.suggest_float("delta", 0.2, 3.0, step=0.5)}
                 }
    
    return param_grid
        
#--------------------------------------------------------------------------------------------------------------------------#