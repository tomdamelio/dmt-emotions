"""
Plot time series for Independent Components (IC1 and IC2).

This script generates time series plots showing IC scores over time,
separated by state (RS/DMT) and dose (Low/High).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def plot_ic_timeseries(ic_scores, output_path, components=['IC1', 'IC2']):
    """
    Plot time series for specified IC components.
    
    Args:
        ic_scores: DataFrame with IC scores and metadata
        output_path: Path to save figure
        components: List of IC components to plot
    """
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10
    
    # Create figure with subplots
    n_components = len(components)
    fig, axes = plt.subplots(n_components, 2, figsize=(14, 4 * n_components))
    
    if n_components == 1:
        axes = axes.reshape(1, -1)
    
    # Color scheme
    colors = {
        'Baja': '#3498db',  # Blue for Low dose
        'Alta': '#e74c3c'   # Red for High dose
    }
    
    for idx, component in enumerate(components):
        # Left panel: RS
        ax_rs = axes[idx, 0]
        
        # Filter RS data
        rs_data = ic_scores[ic_scores['state'] == 'RS'].copy()
        
        # Compute mean and SEM by time and dose
        rs_summary = rs_data.groupby(['t_sec', 'dose'])[component].agg(['mean', 'sem']).reset_index()
        
        for dose in ['Baja', 'Alta']:
            dose_data = rs_summary[rs_summary['dose'] == dose]
            
            # Convert t_sec to minutes
            time_min = dose_data['t_sec'] / 60
            
            # Plot mean line
            ax_rs.plot(time_min, dose_data['mean'], 
                      color=colors[dose], linewidth=2, 
                      label=f'{dose} (20mg)' if dose == 'Baja' else f'{dose} (40mg)')
            
            # Plot SEM shading
            ax_rs.fill_between(
                time_min,
                dose_data['mean'] - dose_data['sem'],
                dose_data['mean'] + dose_data['sem'],
                color=colors[dose], alpha=0.2
            )
        
        ax_rs.set_xlabel('Time (minutes)', fontsize=11)
        ax_rs.set_ylabel(f'{component} Score', fontsize=11)
        ax_rs.set_title(f'{component} - Resting State (RS)', fontsize=12, fontweight='bold')
        ax_rs.legend(loc='best', frameon=True)
        ax_rs.grid(True, alpha=0.3)
        ax_rs.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Right panel: DMT
        ax_dmt = axes[idx, 1]
        
        # Filter DMT data
        dmt_data = ic_scores[ic_scores['state'] == 'DMT'].copy()
        
        # Compute mean and SEM by time and dose
        dmt_summary = dmt_data.groupby(['t_sec', 'dose'])[component].agg(['mean', 'sem']).reset_index()
        
        for dose in ['Baja', 'Alta']:
            dose_data = dmt_summary[dmt_summary['dose'] == dose]
            
            # Convert t_sec to minutes
            time_min = dose_data['t_sec'] / 60
            
            # Plot mean line
            ax_dmt.plot(time_min, dose_data['mean'], 
                       color=colors[dose], linewidth=2,
                       label=f'{dose} (20mg)' if dose == 'Baja' else f'{dose} (40mg)')
            
            # Plot SEM shading
            ax_dmt.fill_between(
                time_min,
                dose_data['mean'] - dose_data['sem'],
                dose_data['mean'] + dose_data['sem'],
                color=colors[dose], alpha=0.2
            )
        
        ax_dmt.set_xlabel('Time (minutes)', fontsize=11)
        ax_dmt.set_ylabel(f'{component} Score', fontsize=11)
        ax_dmt.set_title(f'{component} - DMT State', fontsize=12, fontweight='bold')
        ax_dmt.legend(loc='best', frameon=True)
        ax_dmt.grid(True, alpha=0.3)
        ax_dmt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    
    return fig


def plot_ic_combined(ic_scores, output_path, components=['IC1', 'IC2']):
    """
    Plot combined time series (RS + DMT) for IC components.
    
    Args:
        ic_scores: DataFrame with IC scores and metadata
        output_path: Path to save figure
        components: List of IC components to plot
    """
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['font.size'] = 10
    
    # Create figure
    n_components = len(components)
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 4 * n_components))
    
    if n_components == 1:
        axes = [axes]
    
    # Color scheme
    colors = {
        ('RS', 'Baja'): '#95a5a6',   # Gray for RS
        ('RS', 'Alta'): '#95a5a6',   # Gray for RS
        ('DMT', 'Baja'): '#3498db',  # Blue for DMT Low
        ('DMT', 'Alta'): '#e74c3c'   # Red for DMT High
    }
    
    for idx, component in enumerate(components):
        ax = axes[idx]
        
        # Compute mean and SEM by time, state, and dose
        summary = ic_scores.groupby(['t_sec', 'state', 'dose'])[component].agg(['mean', 'sem']).reset_index()
        
        # Plot RS baseline (average across doses)
        rs_data = summary[summary['state'] == 'RS'].groupby('t_sec').agg({'mean': 'mean', 'sem': 'mean'}).reset_index()
        time_min_rs = rs_data['t_sec'] / 60
        ax.plot(time_min_rs, rs_data['mean'], 
               color='#95a5a6', linewidth=2, linestyle='--',
               label='RS Baseline', alpha=0.7)
        ax.fill_between(
            time_min_rs,
            rs_data['mean'] - rs_data['sem'],
            rs_data['mean'] + rs_data['sem'],
            color='#95a5a6', alpha=0.15
        )
        
        # Add vertical line at DMT onset
        rs_duration = rs_data['t_sec'].max() / 60
        ax.axvline(x=rs_duration, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.text(rs_duration + 0.2, ax.get_ylim()[1] * 0.9, 'DMT Onset', 
               rotation=0, fontsize=9, color='gray')
        
        # Plot DMT conditions
        dmt_data = summary[summary['state'] == 'DMT']
        
        for dose in ['Baja', 'Alta']:
            dose_data = dmt_data[dmt_data['dose'] == dose]
            
            # Shift time to continue from RS
            time_min = (dose_data['t_sec'] / 60) + rs_duration
            
            label = f'DMT Low (20mg)' if dose == 'Baja' else f'DMT High (40mg)'
            color = colors[('DMT', dose)]
            
            ax.plot(time_min, dose_data['mean'], 
                   color=color, linewidth=2, label=label)
            
            ax.fill_between(
                time_min,
                dose_data['mean'] - dose_data['sem'],
                dose_data['mean'] + dose_data['sem'],
                color=color, alpha=0.2
            )
        
        ax.set_xlabel('Time (minutes)', fontsize=11)
        ax.set_ylabel(f'{component} Score', fontsize=11)
        ax.set_title(f'{component} Time Course: RS → DMT', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")
    
    return fig


def main():
    """Main plotting workflow."""
    print("=" * 80)
    print("PLOTTING IC TIME SERIES")
    print("=" * 80)
    print()
    
    # Load IC scores
    ica_dir = Path(config.TET_RESULTS_DIR) / 'ica'
    scores_path = ica_dir / 'ica_scores.csv'
    
    if not scores_path.exists():
        print(f"Error: IC scores not found at {scores_path}")
        print("Run ICA analysis first:")
        print("  python scripts/compute_ica_analysis.py")
        return
    
    print(f"Loading IC scores from {scores_path}...")
    ic_scores = pd.read_csv(scores_path)
    print(f"  Loaded {len(ic_scores)} observations")
    print()
    
    # Create output directory
    figures_dir = ica_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Separate RS and DMT panels
    print("Generating separate RS/DMT time series plots...")
    plot_ic_timeseries(
        ic_scores,
        figures_dir / 'ica_timeseries_separate.png',
        components=['IC1', 'IC2']
    )
    print()
    
    # Plot 2: Combined RS → DMT
    print("Generating combined RS→DMT time series plots...")
    plot_ic_combined(
        ic_scores,
        figures_dir / 'ica_timeseries_combined.png',
        components=['IC1', 'IC2']
    )
    print()
    
    print("=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print()
    print(f"Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
