# Random forest grid to optimize search -----------------------------------------------------------------------------------#
def RandomForestGrid(trial):
    
    param_grid = {
        "objective"              : "huber",
        "alpha"                  : trial.suggest_float("alpha", 0.2, 0.3, step=0.1),
        "n_estimators"           : trial.suggest_int("n_estimators", 50, 5000, step=50),
        "max_depth"              : trial.suggest_int("max_depth", 3, 20),
        "min_samples_split"      : trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf"       : trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features"           : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap"              : trial.suggest_categorical("bootstrap", [True, False]),
                 }
    
    return param_grid

# XGBoost parameter grid to optimize search -------------------------------------------------------------------------------#
def XGBoostGrid(trial):

    param_grid = {
        "objective"              : "reg:pseudohubererror",
        "huber_slope"            : trial.suggest_float("huber_slope", 0.2, 3.0, step=0.1),
        "learning_rate"          : trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
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

# DARTBoost parameter grid to optimze search ------------------------------------------------------------------------------#
def DARTBoostGrid(trial):
    param_grid = {
        "objective"              : "reg:pseudohubererror",
        "learning_rate"          : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth"              : trial.suggest_int("max_depth", 3, 15),
        "n_estimators"           : trial.suggest_int("n_estimators", 100, 1000, step=100),
        "subsample"              : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"       : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma"                  : trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha"              : trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda"             : trial.suggest_float("reg_lambda", 0.0, 5.0),
        "booster"                : "dart",
        "sample_type"            : trial.suggest_categorical("sample_type", ["uniform", "weighted"]),
        "normalize_type"         : trial.suggest_categorical("normalize_type", ["tree", "forest"]),
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

#--------------------------------------------------------------------------------------------------------------------------#