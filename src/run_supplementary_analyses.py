# -*- coding: utf-8 -*-
"""
Supplementary Analyses: Phase-Based and Feature Extraction

This script performs additional analyses requested by supervisor:
1. Phase-based analysis: Compare doses within temporal phases (onset 0-3 min, recovery 3-9 min)
2. Feature extraction: Extract peak amplitude, time-to-peak, threshold crossings

These analyses complement the main time-to-time FDR analyses and address potential
temporal misalignment issues.

Outputs: results/{modality}/supplementary/

Run:
  python src/run_supplementary_analyses.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.phase_analyzer import (
    define_temporal_phases,
    compute_phase_averages,
    compare_doses_within_phases
)
from scripts.feature_extractor import (
    extract_peak_amplitude,
    extract_time_to_peak,
    extract_threshold_crossings
)
from scripts.baseline_comparator import (
    compare_features_to_baseline,
    visualize_baseline_comparisons,
    format_baseline_comparison_report
)
from scripts.statistical_reporter import (
    format_ttest_result,
    save_results_table
)

# Configuration
WINDOW_SIZE_SEC = 30
TOTAL_DURATION_SEC = 540  # 9 minutes
PHASE_BOUNDARIES = [0, 180, 540]  # Onset: 0-3 min, Recovery: 3-9 min

# Modalities to analyze
MODALITIES = {
    'ecg': {
        'data_path': 'results/ecg/hr/hr_minute_long_data_z.csv',
        'value_column': 'HR',
        'label': 'Heart Rate (Z-scored)',
        'output_dir': 'results/ecg/hr/supplementary'
    },
    'eda': {
        'data_path': 'results/eda/smna/smna_auc_long_data_z.csv',
        'value_column': 'AUC',
        'label': 'SMNA AUC (Z-scored)',
        'output_dir': 'results/eda/smna/supplementary'
    },
    'resp': {
        'data_path': 'results/resp/rvt/resp_rvt_minute_long_data_z.csv',
        'value_column': 'RSP_RVT',
        'label': 'RVT (Z-scored)',
        'output_dir': 'results/resp/rvt/supplementary'
    }
}


def load_modality_data(data_path: str, value_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate modality data, returning both DMT and RS data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required = ['subject', 'window', 'State', 'Dose', value_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {data_path}: {missing}")
    
    # Filter to first 9 minutes (18 windows)
    df = df[df['window'] <= 18].copy()
    
    # Split into DMT and RS
    df_dmt = df[df['State'] == 'DMT'].copy()
    df_rs = df[df['State'] == 'RS'].copy()
    
    return df_dmt, df_rs


def run_phase_analysis(df: pd.DataFrame, value_column: str, output_dir: str, label: str) -> Dict:
    """Run phase-based analysis for a modality."""
    print(f"\n{'='*60}")
    print(f"PHASE ANALYSIS: {label}")
    print(f"{'='*60}")
    
    # Define phases
    phases = define_temporal_phases(TOTAL_DURATION_SEC, PHASE_BOUNDARIES)
    print(f"\nPhases defined:")
    for i, (start, end) in enumerate(phases):
        print(f"  Phase {i}: {start}-{end}s ({start/60:.1f}-{end/60:.1f} min)")
    
    # Compute phase averages
    phase_df = compute_phase_averages(
        df, phases, 
        value_column=value_column,
        window_size_sec=WINDOW_SIZE_SEC
    )
    
    # Compare doses within phases
    comparison_df = compare_doses_within_phases(phase_df, value_column='mean_value')
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    phase_df.to_csv(os.path.join(output_dir, 'phase_averages.csv'), index=False)
    comparison_df.to_csv(os.path.join(output_dir, 'phase_comparisons.csv'), index=False)
    
    # Print results
    print(f"\nPhase Comparison Results:")
    print(f"{'Phase':<30} {'t':<8} {'df':<6} {'p':<10} {'d':<8} {'Sig':<5}")
    print("-" * 70)
    
    for _, row in comparison_df.iterrows():
        sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        print(f"{row['phase_label']:<30} {row['t_stat']:>7.3f} {int(row['df']):>6} {row['p_value']:>9.6f} {row['cohens_d']:>7.3f} {sig_marker:<5}")
    
    # Create visualization
    create_phase_plot(phase_df, comparison_df, output_dir, label)
    
    return {
        'phase_df': phase_df,
        'comparison_df': comparison_df,
        'phases': phases
    }


def create_phase_plot(phase_df: pd.DataFrame, comparison_df: pd.DataFrame, 
                     output_dir: str, label: str):
    """Create visualization of phase-averaged data."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Get unique phases
    phases = sorted(phase_df['phase'].unique())
    
    # Prepare data for plotting
    high_means = []
    high_sems = []
    low_means = []
    low_sems = []
    phase_labels = []
    
    for phase in phases:
        phase_data = phase_df[phase_df['phase'] == phase]
        
        high_data = phase_data[phase_data['Dose'] == 'High']['mean_value']
        low_data = phase_data[phase_data['Dose'] == 'Low']['mean_value']
        
        high_means.append(high_data.mean())
        high_sems.append(high_data.sem())
        low_means.append(low_data.mean())
        low_sems.append(low_data.sem())
        
        phase_label = phase_data['phase_label'].iloc[0]
        # Simplify label
        start_min = phase_data['start_sec'].iloc[0] / 60
        end_min = phase_data['end_sec'].iloc[0] / 60
        phase_labels.append(f"{start_min:.0f}-{end_min:.0f} min")
    
    x = np.arange(len(phases))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, high_means, width, yerr=high_sems, 
           label='High dose (40mg)', color='#d62728', alpha=0.8, capsize=5)
    ax.bar(x + width/2, low_means, width, yerr=low_sems,
           label='Low dose (20mg)', color='#ff7f0e', alpha=0.8, capsize=5)
    
    # Add significance markers
    for i, phase in enumerate(phases):
        comp_row = comparison_df[comparison_df['phase'] == phase]
        if len(comp_row) > 0:
            p_val = comp_row['p_value'].iloc[0]
            if p_val < 0.05:
                y_max = max(high_means[i] + high_sems[i], low_means[i] + low_sems[i])
                y_pos = y_max + 0.1 * abs(y_max)
                
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.text(i, y_pos, sig_marker, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Temporal Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel(label, fontsize=12, fontweight='bold')
    ax.set_title('Phase-Averaged Comparison: High vs Low Dose (DMT)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {os.path.join(output_dir, 'phase_comparison.png')}")


def run_feature_extraction(df_dmt: pd.DataFrame, df_rs: pd.DataFrame, value_column: str, 
                          output_dir: str, label: str) -> Dict:
    """Run feature extraction for a modality, including baseline comparisons."""
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION: {label}")
    print(f"{'='*60}")
    
    features_list = []
    
    # Process DMT data
    subjects_dmt = sorted(df_dmt['subject'].unique())
    
    for subject in subjects_dmt:
        subj_data = df_dmt[df_dmt['subject'] == subject]
        
        for dose in ['High', 'Low']:
            dose_data = subj_data[subj_data['Dose'] == dose]
            
            if len(dose_data) == 0:
                continue
            
            # Reconstruct time series
            dose_data = dose_data.sort_values('window')
            t = (dose_data['window'].values - 1) * WINDOW_SIZE_SEC + WINDOW_SIZE_SEC / 2
            signal = dose_data[value_column].values
            
            # Extract features
            try:
                peak_info = extract_peak_amplitude(t, signal, 0, TOTAL_DURATION_SEC)
                time_to_peak = extract_time_to_peak(t, signal, 0, TOTAL_DURATION_SEC)
                thresholds_info = extract_threshold_crossings(
                    t, signal, 
                    thresholds=[0.33, 0.50],
                    window_start_sec=0,
                    window_end_sec=TOTAL_DURATION_SEC
                )
                
                features_list.append({
                    'subject': subject,
                    'State': 'DMT',
                    'Dose': dose,
                    'peak_amplitude': peak_info['peak_amplitude'],
                    'peak_time_sec': peak_info['peak_time_sec'],
                    'time_to_peak_min': time_to_peak,
                    't_33_min': thresholds_info.get('t_33', np.nan),
                    't_50_min': thresholds_info.get('t_50', np.nan)
                })
            except Exception as e:
                print(f"  Warning: Could not extract features for {subject} DMT {dose}: {e}")
                continue
    
    # Process RS data
    subjects_rs = sorted(df_rs['subject'].unique())
    
    for subject in subjects_rs:
        subj_data = df_rs[df_rs['subject'] == subject]
        
        for dose in ['High', 'Low']:  # RS also has dose labels based on session
            dose_data = subj_data[subj_data['Dose'] == dose]
            
            if len(dose_data) == 0:
                continue
            
            # Reconstruct time series
            dose_data = dose_data.sort_values('window')
            t = (dose_data['window'].values - 1) * WINDOW_SIZE_SEC + WINDOW_SIZE_SEC / 2
            signal = dose_data[value_column].values
            
            # Extract features
            try:
                peak_info = extract_peak_amplitude(t, signal, 0, TOTAL_DURATION_SEC)
                time_to_peak = extract_time_to_peak(t, signal, 0, TOTAL_DURATION_SEC)
                thresholds_info = extract_threshold_crossings(
                    t, signal, 
                    thresholds=[0.33, 0.50],
                    window_start_sec=0,
                    window_end_sec=TOTAL_DURATION_SEC
                )
                
                features_list.append({
                    'subject': subject,
                    'State': 'RS',
                    'Dose': dose,
                    'peak_amplitude': peak_info['peak_amplitude'],
                    'peak_time_sec': peak_info['peak_time_sec'],
                    'time_to_peak_min': time_to_peak,
                    't_33_min': thresholds_info.get('t_33', np.nan),
                    't_50_min': thresholds_info.get('t_50', np.nan)
                })
            except Exception as e:
                print(f"  Warning: Could not extract features for {subject} RS {dose}: {e}")
                continue
    
    if not features_list:
        print("  No features extracted!")
        return {}
    
    features_df = pd.DataFrame(features_list)
    
    # Save features
    os.makedirs(output_dir, exist_ok=True)
    features_df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
    
    # 1. Compare features between doses (DMT only)
    print(f"\n--- DMT Dose Comparison (High vs Low) ---")
    feature_comparisons = []
    feature_cols = ['peak_amplitude', 'time_to_peak_min', 't_33_min', 't_50_min']
    
    print(f"\nFeature Comparison Results (Paired t-tests):")
    print(f"{'Feature':<25} {'High Mean':<12} {'Low Mean':<12} {'t':<8} {'p':<10} {'d':<8} {'Sig':<5}")
    print("-" * 85)
    
    dmt_features = features_df[features_df['State'] == 'DMT'].copy()
    
    for feat in feature_cols:
        high_subj = dmt_features[dmt_features['Dose'] == 'High'][['subject', feat]].dropna()
        low_subj = dmt_features[dmt_features['Dose'] == 'Low'][['subject', feat]].dropna()
        merged = pd.merge(high_subj, low_subj, on='subject', suffixes=('_high', '_low'))
        
        if len(merged) < 3:
            print(f"{feat:<25} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
            continue
        
        high_matched = merged[f'{feat}_high'].values
        low_matched = merged[f'{feat}_low'].values
        
        t_stat, p_val = stats.ttest_rel(high_matched, low_matched)
        diff = high_matched - low_matched
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        
        print(f"{feat:<25} {np.mean(high_matched):>11.3f} {np.mean(low_matched):>11.3f} {t_stat:>7.3f} {p_val:>9.6f} {cohens_d:>7.3f} {sig_marker:<5}")
        
        feature_comparisons.append({
            'feature': feat,
            'comparison': 'DMT_High_vs_Low',
            'high_mean': np.mean(high_matched),
            'high_sem': stats.sem(high_matched),
            'low_mean': np.mean(low_matched),
            'low_sem': stats.sem(low_matched),
            't_stat': t_stat,
            'df': len(high_matched) - 1,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_pairs': len(high_matched)
        })
    
    # 2. Compare DMT (collapsed) vs RS baseline
    print(f"\n--- DMT vs Resting State Baseline Comparison ---")
    print("(DMT doses collapsed - quantifies overall DMT effect)")
    
    baseline_comparison = compare_features_to_baseline(
        features_df,
        feature_columns=feature_cols,
        state_column='State',
        dmt_label='DMT',
        rs_label='RS',
        subject_column='subject'
    )
    
    # Save baseline comparison
    baseline_comparison.to_csv(os.path.join(output_dir, 'baseline_comparison.csv'), index=False)
    
    # Print baseline comparison results
    print(f"\n{'Feature':<25} {'DMT Mean':<12} {'RS Mean':<12} {'t':<8} {'p':<10} {'d':<8} {'Sig':<5}")
    print("-" * 85)
    
    for _, row in baseline_comparison.iterrows():
        if row['n_pairs'] < 2:
            print(f"{row['feature']:<25} {'N/A':<12} {'N/A':<12} {'N/A':<8} {'N/A':<10} {'N/A':<8}")
            continue
        
        sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        
        print(f"{row['feature']:<25} {row['mean_dmt']:>11.3f} {row['mean_rs']:>11.3f} {row['t_stat']:>7.3f} {row['p_value']:>9.6f} {row['cohens_d']:>7.3f} {sig_marker:<5}")
    
    # Save all comparisons
    comparison_df = pd.DataFrame(feature_comparisons)
    comparison_df.to_csv(os.path.join(output_dir, 'feature_comparisons.csv'), index=False)
    
    # Create visualizations
    create_feature_plot(features_df, comparison_df, baseline_comparison, output_dir, label)
    
    # Create baseline comparison visualization
    feature_labels = {
        'peak_amplitude': 'Peak Amplitude\n(z-score)',
        'time_to_peak_min': 'Time to Peak\n(minutes)',
        't_33_min': 'Time to 33% Max\n(minutes)',
        't_50_min': 'Time to 50% Max\n(minutes)'
    }
    
    try:
        visualize_baseline_comparisons(
            baseline_comparison,
            os.path.join(output_dir, 'baseline_comparison.png'),
            feature_labels=feature_labels,
            figsize=(12, 6)
        )
    except Exception as e:
        print(f"  Warning: Could not create baseline comparison plot: {e}")
    
    # Generate text report
    report = format_baseline_comparison_report(baseline_comparison)
    with open(os.path.join(output_dir, 'baseline_comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Saved: {os.path.join(output_dir, 'baseline_comparison_report.txt')}")
    
    return {
        'features_df': features_df,
        'dose_comparison_df': comparison_df,
        'baseline_comparison_df': baseline_comparison
    }


def create_feature_plot(features_df: pd.DataFrame, dose_comparison_df: pd.DataFrame,
                       baseline_comparison_df: pd.DataFrame, output_dir: str, label: str):
    """Create visualization of extracted features (DMT dose comparison only)."""
    feature_cols = ['peak_amplitude', 'time_to_peak_min', 't_33_min', 't_50_min']
    feature_labels = {
        'peak_amplitude': 'Peak Amplitude\n(z-score)',
        'time_to_peak_min': 'Time to Peak\n(minutes)',
        't_33_min': 'Time to 33% Max\n(minutes)',
        't_50_min': 'Time to 50% Max\n(minutes)'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Filter to DMT only for dose comparison
    dmt_features = features_df[features_df['State'] == 'DMT'].copy()
    
    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        
        # Get data
        high_data = dmt_features[dmt_features['Dose'] == 'High'][feat].dropna()
        low_data = dmt_features[dmt_features['Dose'] == 'Low'][feat].dropna()
        
        if len(high_data) == 0 or len(low_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature_labels.get(feat, feat))
            continue
        
        # Bar plot with individual points
        x_pos = [0, 1]
        means = [high_data.mean(), low_data.mean()]
        sems = [high_data.sem(), low_data.sem()]
        
        bars = ax.bar(x_pos, means, yerr=sems, capsize=5,
                     color=['#d62728', '#ff7f0e'], alpha=0.7, width=0.6)
        
        # Add individual points
        np.random.seed(42)
        jitter = 0.05
        ax.scatter(np.random.normal(0, jitter, len(high_data)), high_data, 
                  color='darkred', alpha=0.6, s=30, zorder=3)
        ax.scatter(np.random.normal(1, jitter, len(low_data)), low_data,
                  color='darkorange', alpha=0.6, s=30, zorder=3)
        
        # Add significance marker from dose comparison
        comp_row = dose_comparison_df[dose_comparison_df['feature'] == feat]
        if len(comp_row) > 0:
            p_val = comp_row['p_value'].iloc[0]
            if p_val < 0.05:
                y_max = max(means[0] + sems[0], means[1] + sems[1])
                y_pos = y_max + 0.15 * abs(y_max) if y_max != 0 else 0.5
                
                sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                ax.plot([0, 1], [y_pos, y_pos], 'k-', lw=1.5)
                ax.text(0.5, y_pos, sig_marker, ha='center', va='bottom', 
                       fontsize=14, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['High\n(40mg)', 'Low\n(20mg)'])
        ax.set_ylabel(feature_labels.get(feat, feat), fontsize=10, fontweight='bold')
        ax.set_title(feature_labels.get(feat, feat), fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle(f'Extracted Features (DMT Dose Comparison): {label}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {os.path.join(output_dir, 'feature_comparison.png')}")


def main():
    """Run supplementary analyses for all modalities."""
    print("\n" + "="*70)
    print("SUPPLEMENTARY ANALYSES: Phase-Based and Feature Extraction")
    print("="*70)
    
    all_results = {}
    
    for modality_name, config in MODALITIES.items():
        print(f"\n\n{'#'*70}")
        print(f"# MODALITY: {modality_name.upper()}")
        print(f"{'#'*70}")
        
        try:
            # Load data
            df_dmt, df_rs = load_modality_data(config['data_path'], config['value_column'])
            print(f"\nLoaded data: DMT={len(df_dmt)} rows ({len(df_dmt['subject'].unique())} subjects), RS={len(df_rs)} rows ({len(df_rs['subject'].unique())} subjects)")
            
            # Run phase analysis (DMT only)
            phase_results = run_phase_analysis(
                df_dmt, config['value_column'], 
                config['output_dir'], config['label']
            )
            
            # Run feature extraction (DMT and RS for baseline comparison)
            feature_results = run_feature_extraction(
                df_dmt, df_rs, config['value_column'],
                config['output_dir'], config['label']
            )
            
            all_results[modality_name] = {
                'phase': phase_results,
                'features': feature_results
            }
            
            print(f"\n✓ {modality_name.upper()} analysis complete!")
            print(f"  Results saved to: {config['output_dir']}")
            
        except Exception as e:
            print(f"\n✗ Error analyzing {modality_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("SUPPLEMENTARY ANALYSES COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    for modality_name, config in MODALITIES.items():
        if modality_name in all_results:
            print(f"  - {config['output_dir']}/")
    
    return all_results


if __name__ == '__main__':
    results = main()
