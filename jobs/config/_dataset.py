# External functions and utilities ----------------------------------------------------------------------------------------#
from typing      import List, Optional, Union
from dataclasses import dataclass, field

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.processing.constructors.moccasurvey import MoccaSurveyExperimentConfig

# Default feature parameters ----------------------------------------------------------------------------------------------#
CONT_FEATS  = ["log(t/t_cc)", "log(t/t_relax)", "log(t/t_cross)",  "log(t/t_coll)", 
               "log(M_tot)", "log(M_MMO_0)",
               "log(R_h/R_core)", "log(R_tid/R_core)",
               "log(rho(R_h))",
               "log(Z)",
               "log(fbin)"]

CAT_FEATS   = ["type_sim"]

TARGET_FEAT = ["log(M_MMO/M_tot)"]

# Mapping dict for plot display labels ------------------------------------------------------------------------------------#
MAP_FEATS_DICT = {
    "log(t/t_cc)"       :{
        "xtickname" : r"$\log_{10}(t/t_{\rm cc}+1)$" ,
        "featalias" : r"$x_1$",
    },
    "log(t/t_relax)"    :{
        "xtickname" : r"$\log_{10}(t/t_{\rm relax}+1)$" ,
        "featalias" : r"$x_2$",
    },
    "log(t/t_cross)"    :{
        "xtickname" : r"$\log_{10}(t/t_{\rm cross}+1)$" ,
        "featalias" : r"$x_3$",
    }, 
    "log(t/t_coll)" :{
        "xtickname" : r"$\log_{10}(t/t_{\rm coll}+1)$",
        "featalias" : r"$x_4$",
    }, 
    "log(M_MMO_0)"  :{
        "xtickname" : r"$\log_{10}(M_{{\rm MMO},\, 0})$" ,
        "featalias" : r"$x_{5}$",
    },
    "log(M_tot)"  :{
        "xtickname" : r"$\log_{10}(M_{{\rm tot},\, 0})$" ,
        "featalias" : r"$x_{6}$",
    },
    "log(R_h/R_core)"   :{
        "xtickname" : r"$\log_{10}(R_{h}/R_{{\rm core}}+1)$" ,
        "featalias" : r"$x_7$",
    }, 
    "log(R_tid/R_core)" :{
        "xtickname" : r"$\log_{10}(R_{{\rm tid}}/R_{{\rm core}}+1)$" ,
        "featalias" : r"$x_8$",
    },
    "log(rho(R_h))" :{
        "xtickname" : r"$\log_{10}(\rho(R_h)+1)$" ,
        "featalias" : r"$x_9$",
    },
    "log(Z)"            :{
        "xtickname" : r"$\log_{10}(Z)$" ,
        "featalias" : r"$x_{10}$",
    },
    "log(fbin)"         :{
        "xtickname" : r"$\log_{10}(f_{\rm bin})$" ,
        "featalias" : r"$x_{11}$",
    },
    "log(M_MMO/M_tot)"  :{
        "xtickname" : r"$\log_{10}(M_{{\rm MMO}}/M_{{\rm tot}})$" ,
        "featalias" : r"$y$",
    }
}

# Dataset pipeline configuration -----------------------------------------------------------------------------------------#
@dataclass
class JobConfig(MoccaSurveyExperimentConfig):
    """
    ________________________________________________________________________________________________________________________
    Configuration for the dataset preparation pipeline (dataset_pipe.py).
    
    Inherits base simulation I/O fields from MoccaSurveyExperimentConfig (feature_names, expected_order, target_name,
    time/mass column names, and base downsampling/processing parameters).
    ________________________________________________________________________________________________________________________
    """
    # General -------------------------------------------------------------------------------------------------------------#
    dataset_name : str  = "moccasurvey"
    verbose      : bool = True

    # Sampling and augmentation -------------------------------------------------------------------------------------------#
    points_per_sim       : Union[int, float] = 0.97
    n_virtual            : int               = 9
    max_resolution_ratio : float             = 20

    # Partitioning --------------------------------------------------------------------------------------------------------#
    train_split    : float = 0.7
    val_split      : float = 0.2
    test_split     : float = 0.1
    partition_seed : int   = 42

    # Downsampling (extends base) -----------------------------------------------------------------------------------------#
    downsample_category     : str = "mass"
    downsample_target_scale : str = "log_ratio"
    downsample_norm_column  : str = "M_tot"

    # Classification ------------------------------------------------------------------------------------------------------#
    classify     : bool      = True
    class_type   : str       = "mass"
    class_labels : List[str] = field(default_factory = lambda: ["Q1", "Q2", "Q3", "Q4"])

    # Feature engineering -------------------------------------------------------------------------------------------------#
    cont_feats         : List[str]           = field(default_factory = lambda: CONT_FEATS.copy())
    cat_feats          : Optional[List[str]] = field(default_factory = lambda: CAT_FEATS.copy())
    target_feat        : List[str]           = field(default_factory = lambda: TARGET_FEAT.copy())
    min_dem_threshold  : float               = 1e-10

#--------------------------------------------------------------------------------------------------------------------------#
