# -*- coding: utf-8 -*-
"""
Difference-of-Differences Plot for LME Analysis: (High-Low)_DMT - (High-Low)_RS by minute.

This script creates a visualization showing:
1. Top panel: High-Low differences within DMT and within RS across time
2. Bottom panel: Difference-of-differences (DoD) with 95% CI
3. Emphasizes time periods where DoD band doesn't cross zero

The DoD represents the Task×Dose interaction effect over time:
- DoD = (DMT_High - DMT_Low) - (RS_High - RS_Low)
- Positive DoD: dose effect stronger in DMT than RS
- Negative DoD: dose effect stronger in RS than DMT

Usage:
  python test/plot_difference_of_differences.py
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


def compute_within_subject_differences(df: pd.DataFrame) -> pd.DataFrame:
    """Compute High-Low differences within each subject, task, and minute.
    
    Returns DataFrame with columns: subject, minute, Task, high_low_diff
    """
    print("🔢 Computing within-subject High-Low differences...")
    
    # Pivot to get High and Low in separate columns
    df_wide = df.pivot_table(
        index=['subject', 'minute', 'Task'], 
        columns='Dose', 
        values='AUC'
    ).reset_index()
    
    # Compute High - Low difference for each subject/minute/task
    df_wide['high_low_diff'] = df_wide['High'] - df_wide['Low']
    
    # Convert back to long format
    result = df_wide[['subject', 'minute', 'Task', 'high_low_diff']].copy()
    
    print(f"✅ Computed {len(result)} within-subject differences")
    
    return result


def compute_difference_of_differences(df_diffs: pd.DataFrame) -> pd.DataFrame:
    """Compute difference-of-differences: (High-Low)_DMT - (High-Low)_RS.
    
    Returns DataFrame with columns: subject, minute, dmt_diff, rs_diff, dod
    """
    print("🔢 Computing difference-of-differences...")
    
    # Pivot to get DMT and RS differences in separate columns
    df_pivot = df_diffs.pivot_table(
        index=['subject', 'minute'], 
        columns='Task', 
        values='high_low_diff'
    ).reset_index()
    
    # Compute DoD = DMT_diff - RS_diff
    df_pivot['dod'] = df_pivot['DMT'] - df_pivot['RS']
    
    # Rename columns for clarity
    df_pivot = df_pivot.rename(columns={'DMT': 'dmt_diff', 'RS': 'rs_diff'})
    
    print(f"✅ Computed {len(df_pivot)} difference-of-differences")
    
    return df_pivot


def compute_summary_statistics(df_diffs: pd.DataFrame, df_dod: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics for plotting.
    
    Returns:
        task_diffs: mean ± CI for High-Low differences by Task and minute
        dod_stats: mean ± CI for DoD by minute
    """
    print("📈 Computing summary statistics...")
    
    # Task differences (High-Low within DMT and within RS)
    task_stats = df_diffs.groupby(['minute', 'Task'])['high_low_diff'].agg([
        'count', 'mean', 'std', 'sem'
    ]).reset_index()
    
    # Compute 95% CI using t-distribution
    alpha = 0.05
    task_stats['t_crit'] = task_stats.apply(
        lambda row: stats.t.ppf(1 - alpha/2, row['count'] - 1) if row['count'] > 1 else 1.96,
        axis=1
    )
    task_stats['ci_lower'] = task_stats['mean'] - task_stats['t_crit'] * task_stats['sem']
    task_stats['ci_upper'] = task_stats['mean'] + task_stats['t_crit'] * task_stats['sem']
    
    # DoD statistics
    dod_stats = df_dod.groupby('minute')['dod'].agg([
        'count', 'mean', 'std', 'sem'
    ]).reset_index()
    
    dod_stats['t_crit'] = dod_stats.apply(
        lambda row: stats.t.ppf(1 - alpha/2, row['count'] - 1) if row['count'] > 1 else 1.96,
        axis=1
    )
    dod_stats['ci_lower'] = dod_stats['mean'] - dod_stats['t_crit'] * dod_stats['sem']
    dod_stats['ci_upper'] = dod_stats['mean'] + dod_stats['t_crit'] * dod_stats['sem']
    
    # Identify significant periods (CI doesn't cross zero)
    dod_stats['significant'] = (
        (dod_stats['ci_lower'] > 0) | (dod_stats['ci_upper'] < 0)
    )
    
    print(f"✅ Significant DoD periods: {dod_stats['significant'].sum()}/{len(dod_stats)} minutes")
    
    return task_stats, dod_stats


def create_difference_of_differences_plot(task_stats: pd.DataFrame, dod_stats: pd.DataFrame, 
                                        output_path: str) -> None:
    """Create the main difference-of-differences plot."""
    print("🎨 Creating difference-of-differences plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                  gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # === TOP PANEL: High-Low differences within each task ===
    
    colors = {'DMT': '#DC143C', 'RS': '#2E8B57'}
    
    for task in ['RS', 'DMT']:
        task_data = task_stats[task_stats['Task'] == task].sort_values('minute')
        
        if len(task_data) == 0:
            continue
            
        color = colors[task]
        
        # Main line
        ax1.plot(task_data['minute'], task_data['mean'], 
                color=color, linewidth=2.5, label=f'(High - Low) within {task}',
                marker='o', markersize=5)
        
        # Confidence interval
        ax1.fill_between(task_data['minute'], 
                        task_data['ci_lower'], task_data['ci_upper'],
                        color=color, alpha=0.25)
    
    # Reference line at zero
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize top panel
    ax1.set_ylabel('High - Low Difference\n(SMNA AUC)', fontweight='bold')
    ax1.set_title('Within-Task Dose Effects (High - Low) Over Time', fontweight='bold', pad=15)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 9.2)
    
    # === BOTTOM PANEL: Difference-of-differences ===
    
    # Sort by minute for proper line plotting
    dod_sorted = dod_stats.sort_values('minute')
    
    # Identify significant and non-significant segments
    sig_mask = dod_sorted['significant'].values
    
    # Plot all points and line first
    ax2.plot(dod_sorted['minute'], dod_sorted['mean'], 
            color='purple', linewidth=2, alpha=0.7, marker='o', markersize=4)
    
    # Confidence interval - use different styling for significant periods
    for i, row in dod_sorted.iterrows():
        minute = row['minute']
        mean = row['mean']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        is_sig = row['significant']
        
        # Choose color and alpha based on significance
        if is_sig:
            color = 'red'
            alpha = 0.4
            linewidth = 3
        else:
            color = 'purple'
            alpha = 0.2
            linewidth = 1
            
        # Plot CI as vertical line
        ax2.plot([minute, minute], [ci_lower, ci_upper], 
                color=color, linewidth=linewidth, alpha=alpha)
    
    # Fill area for confidence band
    ax2.fill_between(dod_sorted['minute'], 
                    dod_sorted['ci_lower'], dod_sorted['ci_upper'],
                    color='purple', alpha=0.2, label='95% CI')
    
    # Emphasize significant periods with thicker points
    sig_data = dod_sorted[dod_sorted['significant']]
    if len(sig_data) > 0:
        ax2.scatter(sig_data['minute'], sig_data['mean'], 
                   color='red', s=50, zorder=5, alpha=0.8,
                   label='Significant periods')
    
    # Reference line at zero
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Customize bottom panel
    ax2.set_xlabel('Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Difference-of-Differences\n(DMT - RS)', fontweight='bold')
    ax2.set_title('Task × Dose Interaction Effect: (High-Low)ᴅᴍᴛ - (High-Low)ᴿˢ', 
                 fontweight='bold', pad=15)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, 9.2)
    ax2.set_xticks(range(10))
    
    # Add interpretation text
    mean_dod = dod_stats['mean'].mean()
    sig_minutes = dod_stats[dod_stats['significant']]['minute'].tolist()
    
    textstr = (f'Mean DoD: {mean_dod:.3f}\n'
               f'Significant minutes: {sig_minutes}\n'
               f'Interaction: β = 1.379***')
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # Overall figure title
    fig.suptitle('Difference-of-Differences Analysis: Task × Dose Interaction Over Time\n' +
                'N=11 subjects, 10 one-minute windows', 
                fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Difference-of-differences plot saved: {output_path}")


def create_dod_summary_table(task_stats: pd.DataFrame, dod_stats: pd.DataFrame, 
                           output_path: str) -> None:
    """Create a summary table of the DoD analysis."""
    print("📋 Creating DoD summary table...")
    
    # Combine task differences and DoD into one table
    summary_rows = []
    
    for minute in range(10):
        # Get task differences
        dmt_row = task_stats[(task_stats['minute'] == minute) & (task_stats['Task'] == 'DMT')]
        rs_row = task_stats[(task_stats['minute'] == minute) & (task_stats['Task'] == 'RS')]
        dod_row = dod_stats[dod_stats['minute'] == minute]
        
        if len(dmt_row) > 0 and len(rs_row) > 0 and len(dod_row) > 0:
            summary_rows.append({
                'minute': minute,
                'dmt_high_low': dmt_row.iloc[0]['mean'],
                'dmt_ci_lower': dmt_row.iloc[0]['ci_lower'],
                'dmt_ci_upper': dmt_row.iloc[0]['ci_upper'],
                'rs_high_low': rs_row.iloc[0]['mean'],
                'rs_ci_lower': rs_row.iloc[0]['ci_lower'],
                'rs_ci_upper': rs_row.iloc[0]['ci_upper'],
                'dod': dod_row.iloc[0]['mean'],
                'dod_ci_lower': dod_row.iloc[0]['ci_lower'],
                'dod_ci_upper': dod_row.iloc[0]['ci_upper'],
                'dod_significant': dod_row.iloc[0]['significant']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Round for readability
    numeric_cols = ['dmt_high_low', 'dmt_ci_lower', 'dmt_ci_upper', 
                   'rs_high_low', 'rs_ci_lower', 'rs_ci_upper',
                   'dod', 'dod_ci_lower', 'dod_ci_upper']
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    
    # Save to CSV
    summary_df.to_csv(output_path, index=False)
    print(f"✅ DoD summary table saved: {output_path}")
    
    # Print key findings
    print(f"\n📊 DoD SUMMARY FINDINGS:")
    print(f"=" * 50)
    
    overall_dod = summary_df['dod'].mean()
    sig_minutes = summary_df[summary_df['dod_significant']]['minute'].tolist()
    
    print(f"Overall mean DoD: {overall_dod:.4f}")
    print(f"Significant minutes (CI doesn't cross 0): {sig_minutes}")
    print(f"Proportion significant: {len(sig_minutes)}/10 = {len(sig_minutes)/10:.1%}")
    
    if sig_minutes:
        print(f"DoD range in significant minutes: "
              f"{summary_df[summary_df['dod_significant']]['dod'].min():.4f} to "
              f"{summary_df[summary_df['dod_significant']]['dod'].max():.4f}")


def main():
    """Main analysis pipeline."""
    print("🚀 Starting Difference-of-Differences analysis...")
    
    # Create output directory
    output_dir = os.path.join('test', 'eda', 'lme_analysis', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        df = load_long_data()
        
        # Compute within-subject differences (High - Low)
        df_diffs = compute_within_subject_differences(df)
        
        # Compute difference-of-differences
        df_dod = compute_difference_of_differences(df_diffs)
        
        # Compute summary statistics
        task_stats, dod_stats = compute_summary_statistics(df_diffs, df_dod)
        
        # Create main plot
        plot_path = os.path.join(output_dir, 'difference_of_differences.png')
        create_difference_of_differences_plot(task_stats, dod_stats, plot_path)
        
        # Create summary table
        table_path = os.path.join(output_dir, 'dod_summary_table.csv')
        create_dod_summary_table(task_stats, dod_stats, table_path)
        
        print(f"\n🎯 Difference-of-Differences analysis completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🖼️  Main plot: difference_of_differences.png")
        print(f"📊 Summary table: dod_summary_table.csv")
        
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
