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
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if study has completed trials
    completed_trials = [t for t in study.trials if t.state == study.trials[0].state.__class__.COMPLETE]
    if len(completed_trials) < 2:
        warnings.warn(f"Study has only {len(completed_trials)} completed trial(s). Skipping visualizations.")
        return
    
    # Get top 3 most important parameters for contour plot
    try:
        param_importances = get_param_importances(study)
        top_3_params      = list(param_importances.keys())[:min(3, len(param_importances))]
    except Exception as e:
        warnings.warn(f"Could not compute parameter importances: {e}")
        top_3_params = []
    
    # Generate each plot individually with proper error handling
    plot_configs = [
        ("optimization_history", lambda: plot_optimization_history(study)),
        ("param_importances", lambda: plot_param_importances(study)),
    ]
    
    # Add contour plot only if we have parameters to plot
    if len(top_3_params) >= 2:
        plot_configs.append(("contour", lambda: plot_contour(study, params=top_3_params[:2])))
    
    for plot_name, plot_func in plot_configs:
        try:
            # Generate plot (returns matplotlib Axes object)
            ax = plot_func()
            
            # Get the figure from the axes
            fig = ax.get_figure() if hasattr(ax, 'get_figure') else ax.figure
            
            # Save the figure
            save_path = output_path / f"{plot_name}.jpg"
            fig.savefig(save_path, bbox_inches="tight", dpi=600)
            print(f"✓ Saved {plot_name} to {save_path}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            warnings.warn(f"Failed to create {plot_name}: {str(e)}")
            continue

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