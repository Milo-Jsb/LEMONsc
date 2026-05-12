# Modules -----------------------------------------------------------------------------------------------------------------#
import torch

# External functions and utilities ----------------------------------------------------------------------------------------#
from dataclasses import dataclass, field
from typing      import List, Optional, Literal, Dict, Any

# Default feature parameters ----------------------------------------------------------------------------------------------# 
CONT_FEATS  = ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)",
               "log(M_tot/M_crit)", 
               "log(R_h/R_core)", "log(R_tid/R_core)", 
               "log(rho(R_h))",
               "log(Z)", 
               "log(fbin)"]
CAT_FEATS   = None

TARGET_FEAT = ["log(M_MMO/M_tot)"]

# Default dataset config --------------------------------------------------------------------------------------------------#
@dataclass
class DataConfig:
    n_folds     : int                 = 3
    cont_feats  : List[str]           = field(default_factory=lambda: CONT_FEATS.copy())
    cat_feats   : Optional[List[str]] = field(default_factory=lambda: CAT_FEATS.copy() if CAT_FEATS is not None else None)
    target_feat : List[str]           = field(default_factory=lambda: TARGET_FEAT.copy())
    min_dem_thr : Optional[float]     = 1e-10
    batch_size  : int                 = 4096
    num_workers : int                 = 16 

# Defaul feature transform and feature scaling ----------------------------------------------------------------------------#
@dataclass
class ScalersConfig:
    # Target
    scale_target        : bool                                                 = True
    trs_target          : bool                                                 = True
    target_transform    : Literal["identity", "log_raw", "ratio", "log_ratio"] = "log_ratio"
    target_norm_column  : Optional[str]                                        = "M_tot"
    target_log_eps      : Optional[float]                                      = 0
    target_scaler_name  : Literal["standard", "robust", "power"]               = "robust"
    target_scaler_kward : Optional[Dict[str, Any]]                             = field(default_factory=lambda: {}) 
    
    # Features
    scale_features            : bool                                   = True
    feature_scaler_name       : Literal["standard", "robust", "power"] = "robust"
    feature_scaler_kward      : Optional[Dict[str, Any]]               = field(default_factory=lambda: {}) 
    feature_log_eps_all_range : Optional[float]                        = 1
    feature_log_eps_lim_range : Optional[float]                        = 0

# Default optuna configuration for optimization ---------------------------------------------------------------------------# 
@dataclass
class OptunaConfig:
    n_trials       : int           = 100
    direction      : str           = "maximize"
    metric         : str           = "r2"
    trial_patience : int           = 30
    lambda_penalty : float         = 0.01
    storage        : Optional[str] = None  

# Default deep learning configuration -------------------------------------------------------------------------------------#
@dataclass
class DeepModelConfig:
    max_epochs        : int             = 100
    train_es_patience : int             = 15
    dl_loss_fn        : str             = "huber"
    grad_clip_norm    : Optional[float] = 1.0
    use_amp           : Optional[bool]  = True

# defaul parameters for Expected Gradients --------------------------------------------------------------------------------#
@dataclass
class EGConfig:
    n_explain   : int = 300
    n_baselines : int = 256
    batch       : int = 32
    n_samples   : int = 20
    
# Unify Job configuration for pipeline ------------------------------------------------------------------------------------#
@dataclass
class JobConfig:
    device         : str             = "cuda" if torch.cuda.is_available() else "cpu"
    seed           : int             = 9654
    loaders_n_jobs : int             = 15
    verbose        : bool            = True
    dataconfig     : DataConfig      = field(default_factory=DataConfig)
    scalingconfig  : ScalersConfig   = field(default_factory=ScalersConfig)
    optconfig      : OptunaConfig    = field(default_factory=OptunaConfig)
    modelconfig    : DeepModelConfig = field(default_factory=DeepModelConfig)
    interconfig    : EGConfig        = field(default_factory=EGConfig)
#--------------------------------------------------------------------------------------------------------------------------#