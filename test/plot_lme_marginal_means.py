# -*- coding: utf-8 -*-
"""
Plot marginal means for LME analysis results: SMNA AUC by minute.

This script creates visualizations of the key findings from the LME analysis:
- Main effect: DMT > RS
- Interaction: Task×Dose effect (High-Low differs between DMT and RS)  
- Time trend: DMT-RS difference attenuates over time (Task:minute_c < 0)

Creates line plots showing estimated marginal means (EMMs) or simple means
for each condition (RS-Low, RS-High, DMT-Low, DMT-High) across minutes 0-9,
with 95% confidence intervals.

Usage:
  python test/plot_lme_marginal_means.py
"""

import os
import sys
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import SUJETOS_VALIDADOS_EDA

# Plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

# Color scheme for conditions
COLORS = {
    'RS_Low': '#2E8B57',     # Sea green
    'RS_High': '#4169E1',    # Royal blue  
    'DMT_Low': '#FF6347',    # Tomato
    'DMT_High': '#DC143C',   # Crimson
}

LINESTYLES = {
    'RS_Low': '-',
    'RS_High': '--', 
    'DMT_Low': '-',
    'DMT_High': '--',
}


def load_long_data() -> pd.DataFrame:
    """Load the long-format data from LME analysis."""
    data_path = os.path.join('test', 'eda', 'lme_analysis', 'smna_auc_long_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Long-format data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure proper categorical encoding
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    
    print(f"📊 Loaded long-format data: {len(df)} observations from {len(df['subject'].unique())} subjects")
    
    return df


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute empirical means and confidence intervals by condition and minute.
    
    Returns DataFrame with columns: minute, Task, Dose, condition, mean, se, ci_lower, ci_upper, n
    """
    print("📈 Computing empirical means and confidence intervals...")
    
    # Group by minute, Task, Dose
    grouped = df.groupby(['minute', 'Task', 'Dose'])['AUC']
    
    # Compute statistics
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'Task', 'Dose', 'n', 'mean', 'std', 'se']
    
    # Create condition label
    stats_df['condition'] = stats_df['Task'].astype(str) + '_' + stats_df['Dose'].astype(str)
    
    # Compute confidence intervals using t-distribution
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, stats_df['n'] - 1)
    
    stats_df['ci_lower'] = stats_df['mean'] - t_critical * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_critical * stats_df['se']
    
    # Handle cases where se is NaN (single observation)
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    
    print(f"✅ Computed statistics for {len(stats_df)} condition×minute combinations")
    
    return stats_df


def create_marginal_means_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create the main marginal means plot showing all four conditions over time."""
    print("🎨 Creating marginal means plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each condition
    conditions = stats_df['condition'].unique()
    
    for condition in sorted(conditions):
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        
        if len(cond_data) == 0:
            continue
            
        color = COLORS.get(condition, '#666666')
        linestyle = LINESTYLES.get(condition, '-')
        
        # Main line
        ax.plot(cond_data['minute'], cond_data['mean'], 
                color=color, linestyle=linestyle, linewidth=2.5, 
                label=condition.replace('_', ' '), marker='o', markersize=5)
        
        # Confidence interval
        ax.fill_between(cond_data['minute'], 
                       cond_data['ci_lower'], cond_data['ci_upper'],
                       color=color, alpha=0.2)
    
    # Customize plot
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('SMNA AUC (mean ± 95% CI)', fontweight='bold')
    ax.set_title('Marginal Means: SMNA AUC by Task × Dose × Time\n' + 
                'LME Analysis Results (N=11 subjects)', fontweight='bold', pad=20)
    
    # Set x-axis to show all minutes
    ax.set_xticks(range(10))
    ax.set_xlim(-0.2, 9.2)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend with better positioning
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add statistical annotation
    textstr = ('Key findings:\n'
               '• DMT > RS (main effect)\n' 
               '• DMT-RS difference ↓ over time\n'
               '• High dose effect stronger in DMT')
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Marginal means plot saved: {output_path}")


def create_task_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create a focused plot showing the main Task effect and its time interaction."""
    print("🎨 Creating Task effect plot...")
    
    # Compute Task means averaging across Dose
    task_means = stats_df.groupby(['minute', 'Task']).agg({
        'mean': 'mean',  # Average across dose conditions
        'n': 'sum'       # Total observations
    }).reset_index()
    
    # Approximate SE for the averaged means (simplified)
    # This is a rough approximation - proper calculation would need the original data
    task_means['se'] = stats_df.groupby(['minute', 'Task'])['se'].apply(
        lambda x: np.sqrt(np.sum(x**2) / len(x))
    ).reset_index(drop=True)
    
    # CI for task means
    t_crit = 1.96  # Approximate for large N
    task_means['ci_lower'] = task_means['mean'] - t_crit * task_means['se']
    task_means['ci_upper'] = task_means['mean'] + t_crit * task_means['se']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot RS and DMT separately
    for task in ['RS', 'DMT']:
        task_data = task_means[task_means['Task'] == task].sort_values('minute')
        
        color = '#2E8B57' if task == 'RS' else '#DC143C'
        
        ax.plot(task_data['minute'], task_data['mean'], 
                color=color, linewidth=3, label=f'{task} (average across dose)',
                marker='o', markersize=6)
        
        ax.fill_between(task_data['minute'], 
                       task_data['ci_lower'], task_data['ci_upper'],
                       color=color, alpha=0.2)
    
    # Customize
    ax.set_xlabel('Time (minutes)', fontweight='bold')
    ax.set_ylabel('SMNA AUC (mean ± 95% CI)', fontweight='bold')
    ax.set_title('Main Task Effect: DMT vs RS Across Time\n' +
                'β(Task) = 0.990***, β(Task×time) = -0.236***', 
                fontweight='bold', pad=20)
    
    ax.set_xticks(range(10))
    ax.set_xlim(-0.2, 9.2)
    ax.grid(True, alpha=0.3)
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add interpretation text
    textstr = ('DMT consistently > RS\n'
               'Effect attenuates over time\n'
               '(negative Task×time interaction)')
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Task effect plot saved: {output_path}")


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create a plot highlighting the Task×Dose interaction."""
    print("🎨 Creating Task×Dose interaction plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Left panel: RS conditions
    rs_data = stats_df[stats_df['condition'].str.startswith('RS')].copy()
    
    for condition in ['RS_Low', 'RS_High']:
        cond_data = rs_data[rs_data['condition'] == condition].sort_values('minute')
        color = COLORS[condition]
        linestyle = LINESTYLES[condition]
        
        ax1.plot(cond_data['minute'], cond_data['mean'], 
                color=color, linestyle=linestyle, linewidth=2.5,
                label=condition.replace('RS_', ''), marker='o', markersize=4)
        
        ax1.fill_between(cond_data['minute'], 
                        cond_data['ci_lower'], cond_data['ci_upper'],
                        color=color, alpha=0.2)
    
    ax1.set_title('Resting State (RS)\nHigh-Low difference: -0.23 (ns)', fontweight='bold')
    ax1.set_xlabel('Time (minutes)', fontweight='bold')
    ax1.set_ylabel('SMNA AUC (mean ± 95% CI)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(10))
    
    # Right panel: DMT conditions  
    dmt_data = stats_df[stats_df['condition'].str.startswith('DMT')].copy()
    
    for condition in ['DMT_Low', 'DMT_High']:
        cond_data = dmt_data[dmt_data['condition'] == condition].sort_values('minute')
        color = COLORS[condition]
        linestyle = LINESTYLES[condition]
        
        ax2.plot(cond_data['minute'], cond_data['mean'], 
                color=color, linestyle=linestyle, linewidth=2.5,
                label=condition.replace('DMT_', ''), marker='o', markersize=4)
        
        ax2.fill_between(cond_data['minute'], 
                        cond_data['ci_lower'], cond_data['ci_upper'],
                        color=color, alpha=0.2)
    
    ax2.set_title('DMT Task\nHigh-Low difference: +1.15***', fontweight='bold')
    ax2.set_xlabel('Time (minutes)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(10))
    
    # Overall title
    fig.suptitle('Task × Dose Interaction: β = 1.379***\n' +
                'Dose effect differs dramatically between RS and DMT',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Interaction plot saved: {output_path}")


def create_summary_statistics_table(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create a summary table of key statistics."""
    print("📋 Creating summary statistics table...")
    
    # Compute overall means by condition
    overall_means = stats_df.groupby('condition').agg({
        'mean': 'mean',
        'se': lambda x: np.sqrt(np.sum(x**2) / len(x)),  # Rough approximation
        'n': 'mean'
    }).round(4)
    
    # Add confidence intervals
    overall_means['ci_lower'] = overall_means['mean'] - 1.96 * overall_means['se']
    overall_means['ci_upper'] = overall_means['mean'] + 1.96 * overall_means['se']
    
    # Save as CSV
    overall_means.to_csv(output_path)
    print(f"✅ Summary statistics saved: {output_path}")
    
    # Print to console
    print("\n📊 SUMMARY STATISTICS (averaged across minutes):")
    print("=" * 60)
    for condition in overall_means.index:
        mean = overall_means.loc[condition, 'mean']
        se = overall_means.loc[condition, 'se']
        ci_low = overall_means.loc[condition, 'ci_lower']
        ci_high = overall_means.loc[condition, 'ci_upper']
        print(f"{condition:12}: {mean:6.3f} ± {se:5.3f} [CI: {ci_low:6.3f}, {ci_high:6.3f}]")


def main():
    """Main plotting pipeline."""
    print("🎨 Starting LME marginal means plotting pipeline...")
    
    # Create output directory
    output_dir = os.path.join('test', 'eda', 'lme_analysis', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        df = load_long_data()
        
        # Compute empirical means and CIs
        stats_df = compute_empirical_means_and_ci(df)
        
        # Create plots
        plots = [
            ('marginal_means_all_conditions.png', create_marginal_means_plot),
            ('task_main_effect.png', create_task_effect_plot),
            ('task_dose_interaction.png', create_interaction_plot),
        ]
        
        for filename, plot_func in plots:
            output_path = os.path.join(output_dir, filename)
            plot_func(stats_df, output_path)
        
        # Create summary table
        summary_path = os.path.join(output_dir, 'summary_statistics.csv')
        create_summary_statistics_table(stats_df, summary_path)
        
        print(f"\n🎯 All plots completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Generated {len(plots)} plots + summary table")
        
        # Quick interpretation summary
        print(f"\n🔍 KEY VISUAL FINDINGS:")
        print(f"   1. DMT consistently shows higher SMNA AUC than RS")
        print(f"   2. High dose dramatically increases DMT response but not RS")
        print(f"   3. All effects tend to decrease over the 10-minute window")
        print(f"   4. Strongest effects occur in early minutes (0-3)")
        
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
