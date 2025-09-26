# -*- coding: utf-8 -*-
"""
Cumulative AUC Plot: High vs Low dose across time.

This script creates visualizations showing cumulative AUC (cumAUC) over the 10-minute window:
- For each subject, accumulate AUC minute by minute (0→9)
- Show average curves (High and Low) with 95% CI
- Overlay individual subject trajectories (thin lines, low alpha)
- Separate panels for RS and DMT tasks

Key insight: If High dose accumulates faster only during DMT, 
the separation will grow specifically in the DMT panel.

Usage:
  python test/plot_cumulative_auc.py
"""

import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

# Color scheme
COLORS = {
    'High': '#DC143C',  # Crimson for high dose
    'Low': '#4169E1',   # Royal blue for low dose
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


def compute_cumulative_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative AUC for each subject, task, and dose.
    
    Returns DataFrame with additional 'cumAUC' column.
    """
    print("📈 Computing cumulative AUC for each subject...")
    
    # Sort by subject, task, dose, and minute to ensure proper cumulative calculation
    df_sorted = df.sort_values(['subject', 'Task', 'Dose', 'minute']).copy()
    
    # Compute cumulative sum within each subject/task/dose combination
    df_sorted['cumAUC'] = df_sorted.groupby(['subject', 'Task', 'Dose'])['AUC'].cumsum()
    
    print(f"✅ Computed cumulative AUC for all observations")
    
    return df_sorted


def compute_subject_trajectories(df_cum: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for plotting individual subject trajectories."""
    print("👥 Preparing individual subject trajectories...")
    
    # Ensure we have complete data for each subject
    subject_data = []
    
    for subject in df_cum['subject'].unique():
        subj_df = df_cum[df_cum['subject'] == subject]
        
        # Check if subject has all conditions
        conditions = subj_df.groupby(['Task', 'Dose']).size()
        expected_conditions = [('RS', 'Low'), ('RS', 'High'), ('DMT', 'Low'), ('DMT', 'High')]
        
        if all(cond in conditions.index for cond in expected_conditions):
            subject_data.append(subj_df)
        else:
            print(f"  ⚠️  Skipping {subject}: incomplete data")
    
    if subject_data:
        result = pd.concat(subject_data, ignore_index=True)
        print(f"✅ Prepared trajectories for {len(result['subject'].unique())} subjects")
        return result
    else:
        raise ValueError("No subjects with complete data found!")


def compute_group_statistics(df_cum: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level statistics (mean ± CI) for cumulative AUC."""
    print("📊 Computing group-level statistics...")
    
    # Group by minute, task, and dose
    grouped = df_cum.groupby(['minute', 'Task', 'Dose'])['cumAUC']
    
    # Compute statistics
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'Task', 'Dose', 'n', 'mean', 'std', 'se']
    
    # Compute 95% confidence intervals
    alpha = 0.05
    stats_df['t_crit'] = stats_df.apply(
        lambda row: stats.t.ppf(1 - alpha/2, row['n'] - 1) if row['n'] > 1 else 1.96,
        axis=1
    )
    stats_df['ci_lower'] = stats_df['mean'] - stats_df['t_crit'] * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + stats_df['t_crit'] * stats_df['se']
    
    print(f"✅ Computed group statistics for {len(stats_df)} condition×minute combinations")
    
    return stats_df


def create_cumulative_auc_plot(df_cum: pd.DataFrame, stats_df: pd.DataFrame, 
                              output_path: str) -> None:
    """Create the main cumulative AUC plot with separate panels for RS and DMT."""
    print("🎨 Creating cumulative AUC plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    tasks = ['RS', 'DMT']
    axes = [ax1, ax2]
    
    for task, ax in zip(tasks, axes):
        # Plot individual subject trajectories (thin lines, low alpha)
        task_data = df_cum[df_cum['Task'] == task]
        
        for dose in ['Low', 'High']:
            dose_data = task_data[task_data['Dose'] == dose]
            color = COLORS[dose]
            
            # Individual subject lines
            for subject in dose_data['subject'].unique():
                subj_data = dose_data[dose_data['subject'] == subject].sort_values('minute')
                if len(subj_data) >= 2:  # Need at least 2 points for a line
                    ax.plot(subj_data['minute'], subj_data['cumAUC'], 
                           color=color, alpha=0.15, linewidth=0.8, zorder=1)
        
        # Plot group means with confidence intervals
        task_stats = stats_df[stats_df['Task'] == task]
        
        for dose in ['Low', 'High']:
            dose_stats = task_stats[task_stats['Dose'] == dose].sort_values('minute')
            color = COLORS[dose]
            
            if len(dose_stats) == 0:
                continue
            
            # Main line (group mean)
            ax.plot(dose_stats['minute'], dose_stats['mean'], 
                   color=color, linewidth=3.5, label=f'{dose} dose',
                   marker='o', markersize=6, zorder=3)
            
            # Confidence interval
            ax.fill_between(dose_stats['minute'], 
                           dose_stats['ci_lower'], dose_stats['ci_upper'],
                           color=color, alpha=0.25, zorder=2)
        
        # Customize each panel
        ax.set_xlabel('Time (minutes)', fontweight='bold')
        if task == 'RS':
            ax.set_ylabel('Cumulative SMNA AUC', fontweight='bold')
        
        ax.set_title(f'{task} Task\nCumulative AUC: High vs Low Dose', fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 9.2)
        ax.set_xticks(range(10))
        
        # Add final separation annotation
        if len(task_stats) > 0:
            final_high = task_stats[(task_stats['Dose'] == 'High') & (task_stats['minute'] == 9)]['mean']
            final_low = task_stats[(task_stats['Dose'] == 'Low') & (task_stats['minute'] == 9)]['mean']
            
            if len(final_high) > 0 and len(final_low) > 0:
                separation = final_high.iloc[0] - final_low.iloc[0]
                ax.text(0.98, 0.02, f'Final separation:\n{separation:.2f}', 
                       transform=ax.transAxes, fontsize=9,
                       horizontalalignment='right', verticalalignment='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Overall figure title
    fig.suptitle('Cumulative SMNA AUC: Development of Dose Effects Over Time\n' +
                'Individual trajectories (thin lines) + Group means ± 95% CI (thick lines)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Cumulative AUC plot saved: {output_path}")


def create_difference_trajectory_plot(df_cum: pd.DataFrame, output_path: str) -> None:
    """Create a plot showing the High-Low difference trajectory over time."""
    print("🎨 Creating cumulative difference trajectory plot...")
    
    # Compute High-Low differences for each subject and minute
    diff_data = []
    
    for subject in df_cum['subject'].unique():
        subj_df = df_cum[df_cum['subject'] == subject]
        
        for task in ['RS', 'DMT']:
            task_df = subj_df[subj_df['Task'] == task]
            
            # Pivot to get High and Low in separate columns
            pivot_df = task_df.pivot_table(
                index='minute', columns='Dose', values='cumAUC'
            )
            
            if 'High' in pivot_df.columns and 'Low' in pivot_df.columns:
                pivot_df['diff'] = pivot_df['High'] - pivot_df['Low']
                
                for minute in pivot_df.index:
                    if not pd.isna(pivot_df.loc[minute, 'diff']):
                        diff_data.append({
                            'subject': subject,
                            'minute': minute,
                            'Task': task,
                            'cumulative_diff': pivot_df.loc[minute, 'diff']
                        })
    
    diff_df = pd.DataFrame(diff_data)
    
    if len(diff_df) == 0:
        print("⚠️  No difference data available for trajectory plot")
        return
    
    # Compute group statistics for differences
    diff_stats = diff_df.groupby(['minute', 'Task'])['cumulative_diff'].agg([
        'count', 'mean', 'std', 'sem'
    ]).reset_index()
    
    # Confidence intervals
    alpha = 0.05
    diff_stats['t_crit'] = diff_stats.apply(
        lambda row: stats.t.ppf(1 - alpha/2, row['count'] - 1) if row['count'] > 1 else 1.96,
        axis=1
    )
    diff_stats['ci_lower'] = diff_stats['mean'] - diff_stats['t_crit'] * diff_stats['sem']
    diff_stats['ci_upper'] = diff_stats['mean'] + diff_stats['t_crit'] * diff_stats['sem']
    
    # Create plot with improved dimensions and spacing
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = {'RS': '#2E8B57', 'DMT': '#DC143C'}
    
    for task in ['RS', 'DMT']:
        # Individual subject trajectories
        task_diff = diff_df[diff_df['Task'] == task]
        for subject in task_diff['subject'].unique():
            subj_data = task_diff[task_diff['subject'] == subject].sort_values('minute')
            ax.plot(subj_data['minute'], subj_data['cumulative_diff'], 
                   color=colors[task], alpha=0.15, linewidth=0.8)
        
        # Group mean
        task_stats = diff_stats[diff_stats['Task'] == task].sort_values('minute')
        if len(task_stats) > 0:
            ax.plot(task_stats['minute'], task_stats['mean'], 
                   color=colors[task], linewidth=3, label=f'{task} (High - Low dose)',
                   marker='o', markersize=6)
            
            ax.fill_between(task_stats['minute'], 
                           task_stats['ci_lower'], task_stats['ci_upper'],
                           color=colors[task], alpha=0.25)
    
    # Reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize plot with improved spacing and clearer title
    ax.set_xlabel('Time (minutes)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Cumulative Dose Effect\n(High - Low dose)', fontweight='bold', fontsize=13)
    ax.set_title('How DMT and Resting State respond differently to Dose\nCumulative SMNA AUC is higher in DMT than in RS from higher doses', 
                fontweight='bold', fontsize=15, pad=30)
    
    # Legend in upper left corner
    legend = ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.2, 9.2)
    ax.set_xticks(range(10))
    
    # Improve tick label spacing and size
    ax.tick_params(axis='both', which='major', labelsize=11, pad=8)
    
    # Add numerical results in upper right corner (separate from legend)
    final_dmt = diff_stats[(diff_stats['Task'] == 'DMT') & (diff_stats['minute'] == 9)]['mean']
    final_rs = diff_stats[(diff_stats['Task'] == 'RS') & (diff_stats['minute'] == 9)]['mean']
    
    if len(final_dmt) > 0 and len(final_rs) > 0:
        # Results box in upper right
        results_text = (f'Final Effects (min 9):\n'
                       f'DMT: {final_dmt.iloc[0]:.1f}\n'
                       f'RS: {final_rs.iloc[0]:.1f}\n'
                       f'Difference: {final_dmt.iloc[0] - final_rs.iloc[0]:.1f}')
        
        results_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.95, 
                           edgecolor='gray', linewidth=1.5)
        ax.text(0.98, 0.98, results_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', 
                bbox=results_props, weight='bold')
    
    # Adjust layout with more padding for better label spacing
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"✅ Difference trajectory plot saved: {output_path}")


def create_summary_table(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create a summary table of cumulative AUC statistics."""
    print("📋 Creating cumulative AUC summary table...")
    
    # Pivot to get a more readable format
    summary_pivot = stats_df.pivot_table(
        index=['minute'], 
        columns=['Task', 'Dose'], 
        values=['mean', 'ci_lower', 'ci_upper']
    ).round(4)
    
    # Flatten column names
    summary_pivot.columns = [f'{task}_{dose}_{stat}' for stat, task, dose in summary_pivot.columns]
    
    # Save to CSV
    summary_pivot.to_csv(output_path)
    print(f"✅ Summary table saved: {output_path}")
    
    # Print key findings
    print(f"\n📊 CUMULATIVE AUC SUMMARY:")
    print(f"=" * 50)
    
    # Final values (minute 9)
    final_stats = stats_df[stats_df['minute'] == 9]
    
    for task in ['RS', 'DMT']:
        task_final = final_stats[final_stats['Task'] == task]
        high_final = task_final[task_final['Dose'] == 'High']['mean']
        low_final = task_final[task_final['Dose'] == 'Low']['mean']
        
        if len(high_final) > 0 and len(low_final) > 0:
            diff = high_final.iloc[0] - low_final.iloc[0]
            print(f"{task} final cumAUC - High: {high_final.iloc[0]:.3f}, Low: {low_final.iloc[0]:.3f}, Diff: {diff:.3f}")
    
    # Growth rates (slope from minute 0 to 9)
    print(f"\nGrowth patterns:")
    for task in ['RS', 'DMT']:
        for dose in ['Low', 'High']:
            task_dose = stats_df[(stats_df['Task'] == task) & (stats_df['Dose'] == dose)]
            if len(task_dose) >= 2:
                initial = task_dose[task_dose['minute'] == 0]['mean']
                final = task_dose[task_dose['minute'] == 9]['mean']
                if len(initial) > 0 and len(final) > 0:
                    growth = final.iloc[0] - initial.iloc[0]
                    print(f"  {task}-{dose}: {growth:.3f} total growth")


def main():
    """Main analysis pipeline."""
    print("🚀 Starting Cumulative AUC analysis...")
    
    # Create output directory
    output_dir = os.path.join('test', 'eda', 'lme_analysis', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        df = load_long_data()
        
        # Compute cumulative AUC
        df_cum = compute_cumulative_auc(df)
        
        # Prepare subject trajectories
        df_trajectories = compute_subject_trajectories(df_cum)
        
        # Compute group statistics
        stats_df = compute_group_statistics(df_trajectories)
        
        # Create main cumulative AUC plot
        main_plot_path = os.path.join(output_dir, 'cumulative_auc_high_vs_low.png')
        create_cumulative_auc_plot(df_trajectories, stats_df, main_plot_path)
        
        # Create difference trajectory plot
        diff_plot_path = os.path.join(output_dir, 'cumulative_difference_trajectory.png')
        create_difference_trajectory_plot(df_trajectories, diff_plot_path)
        
        # Create summary table
        table_path = os.path.join(output_dir, 'cumulative_auc_summary.csv')
        create_summary_table(stats_df, table_path)
        
        print(f"\n🎯 Cumulative AUC analysis completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖼️  Main plot: cumulative_auc_high_vs_low.png")
        print(f"🖼️  Difference plot: cumulative_difference_trajectory.png")
        print(f"📊 Summary table: cumulative_auc_summary.csv")
        
        print(f"\n🔍 KEY INTERPRETATIONS:")
        print(f"   • Individual trajectories show subject-level variability")
        print(f"   • Group means reveal systematic dose effects")
        print(f"   • Separation between High/Low develops differently for RS vs DMT")
        print(f"   • Cumulative effects become more pronounced over time")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
