# Modules -----------------------------------------------------------------------------------------------------------------#
import warnings

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# External functions and utilities ----------------------------------------------------------------------------------------#
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_contour
from optuna.importance               import get_param_importances
from optuna.study                    import Study
from pathlib                         import Path

# [Helper] Create visualizations ------------------------------------------------------------------------------------------#
def create_visualizations_per_study(study: Study, output_path: Path) -> None:
    """Generate and save visualization plots from a single Optuna study"""
    try:
        # Get top 3 most important parameters
        param_importances = get_param_importances(study)
        top_3_params      = list(param_importances.keys())[:3]
        
        plots = {
            "optimization_history" : plot_optimization_history(study),
            "param_importances"    : plot_param_importances(study),
            "contour"              : plot_contour(study, params=top_3_params)
        }
        
        for name, fig in plots.items():
            fig.figure.savefig(output_path / f"{name}.jpg", bbox_inches="tight", dpi=600)
            plt.close(fig.figure)
            
    except Exception as e:
        warnings.warn(f"Visualization failed: {e}")

# [Helper] Create CV-specific visualizations -----------------------------------------------------------------------------#
def plot_cv_evol_distributions(cv_results: pd.DataFrame, output_path: Path, metric_name: str="Huber Loss") -> None:
    """Create CV-specific visualization plots (experimental)"""
    try:
        # Extract partition scores and calculate statistics
        scores_matrix = np.vstack(cv_results['partition_scores'].values)
        iterations    = np.arange(1, len(scores_matrix) + 1)
        
        means = np.mean(scores_matrix, axis=1)
        stds  = np.std(scores_matrix, axis=1)
        
        # Create scatter plot with error bars
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(iterations, means, yerr=stds, fmt='o', capsize=5, capthick=2, ecolor='gray', markersize=6, alpha=0.7)
        ax.scatter(iterations, means, color='navy', s=20, label= metric_name)
        
        ax.set_title('Score Distribution Across Partitions', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Score', fontsize=12)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "cv_score_distributions.jpg", bbox_inches="tight", dpi=600)
            
        plt.close(fig)
        
    except Exception as e:
        warnings.warn(f"Failed to create CV visualizations: {e}")

#-------------------------------------------------------------------------------------------------------------------------#