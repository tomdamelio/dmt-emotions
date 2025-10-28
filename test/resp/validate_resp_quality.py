# -*- coding: utf-8 -*-
"""
Respiratory Signal Quality Validation and Artifact Detection

This script validates the quality of preprocessed respiratory signals:
  1) Check physiological ranges (RSP_Rate: 5-40 rpm)
  2) Validate peak/trough detection quality
  3) Detect artifacts and outliers
  4) Generate quality control visualizations
  5) Produce summary statistics per subject and condition

Outputs are written to: test/resp/validation_results/

Run:
  python test/resp/validate_resp_quality.py
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_RESP,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Plot aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Physiological ranges
RATE_MIN = 5    # breaths per minute
RATE_MAX = 40   # breaths per minute
AMPLITUDE_MIN = 0
AMPLITUDE_MAX = 3000  # arbitrary units

# Output directory
OUTPUT_DIR = os.path.join('test', 'resp', 'validation_results')


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_resp_paths(subject: str, high_session: str, low_session: str) -> Dict[str, str]:
    """Build paths to all RESP CSVs for a subject."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_low')
    
    paths = {
        'dmt_high': os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv"),
        'dmt_low': os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv"),
        'rs_high': os.path.join(base_high, f"{subject}_rs_{high_session}_high.csv"),
        'rs_low': os.path.join(base_low, f"{subject}_rs_{low_session}_low.csv"),
    }
    return paths


def load_resp_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load RESP CSV and validate basic structure."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['RSP_Clean', 'RSP_Rate']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Reconstruct time if missing
        if 'time' not in df.columns:
            sr = NEUROKIT_PARAMS.get('sampling_rate', 250)
            df['time'] = np.arange(len(df)) / float(sr)
        
        return df
    except Exception as e:
        warnings.warn(f"Failed to load {csv_path}: {e}")
        return None


def validate_rate_range(df: pd.DataFrame) -> Dict:
    """Validate RSP_Rate is within physiological range."""
    rate = pd.to_numeric(df['RSP_Rate'], errors='coerce')
    rate_valid = rate.dropna()
    
    n_total = len(rate_valid)
    n_below = np.sum(rate_valid < RATE_MIN)
    n_above = np.sum(rate_valid > RATE_MAX)
    n_valid = np.sum((rate_valid >= RATE_MIN) & (rate_valid <= RATE_MAX))
    
    pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0
    
    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_below_min': n_below,
        'n_above_max': n_above,
        'pct_valid': pct_valid,
        'mean_rate': rate_valid.mean(),
        'std_rate': rate_valid.std(),
        'min_rate': rate_valid.min(),
        'max_rate': rate_valid.max(),
    }


def validate_amplitude_range(df: pd.DataFrame) -> Dict:
    """Validate RSP_Amplitude is within reasonable range."""
    if 'RSP_Amplitude' not in df.columns:
        return {'available': False}
    
    amp = pd.to_numeric(df['RSP_Amplitude'], errors='coerce')
    amp_valid = amp.dropna()
    
    n_total = len(amp_valid)
    n_valid = np.sum((amp_valid >= AMPLITUDE_MIN) & (amp_valid <= AMPLITUDE_MAX))
    pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0
    
    return {
        'available': True,
        'n_total': n_total,
        'n_valid': n_valid,
        'pct_valid': pct_valid,
        'mean_amp': amp_valid.mean(),
        'std_amp': amp_valid.std(),
        'min_amp': amp_valid.min(),
        'max_amp': amp_valid.max(),
    }


def validate_peaks_troughs(df: pd.DataFrame) -> Dict:
    """Validate peak and trough detection quality."""
    results = {}
    
    # Check peaks
    if 'RSP_Peaks' in df.columns:
        peaks = pd.to_numeric(df['RSP_Peaks'], errors='coerce')
        n_peaks = np.sum(peaks == 1)
        results['n_peaks'] = n_peaks
        
        # Calculate inter-peak intervals
        peak_indices = np.where(peaks == 1)[0]
        if len(peak_indices) > 1:
            sr = NEUROKIT_PARAMS.get('sampling_rate', 250)
            ipi = np.diff(peak_indices) / sr  # inter-peak intervals in seconds
            results['mean_ipi'] = ipi.mean()
            results['std_ipi'] = ipi.std()
            results['min_ipi'] = ipi.min()
            results['max_ipi'] = ipi.max()
            
            # Check for unreasonably short intervals (< 1.5s = 40 rpm)
            n_short = np.sum(ipi < 1.5)
            results['n_short_intervals'] = n_short
            results['pct_short_intervals'] = (n_short / len(ipi) * 100) if len(ipi) > 0 else 0
    else:
        results['n_peaks'] = 0
    
    # Check troughs
    if 'RSP_Troughs' in df.columns:
        troughs = pd.to_numeric(df['RSP_Troughs'], errors='coerce')
        n_troughs = np.sum(troughs == 1)
        results['n_troughs'] = n_troughs
        
        # Calculate inter-trough intervals
        trough_indices = np.where(troughs == 1)[0]
        if len(trough_indices) > 1:
            sr = NEUROKIT_PARAMS.get('sampling_rate', 250)
            iti = np.diff(trough_indices) / sr
            results['mean_iti'] = iti.mean()
            results['std_iti'] = iti.std()
    else:
        results['n_troughs'] = 0
    
    return results


def detect_artifacts(df: pd.DataFrame) -> Dict:
    """Detect potential artifacts in respiratory signal."""
    clean = pd.to_numeric(df['RSP_Clean'], errors='coerce')
    rate = pd.to_numeric(df['RSP_Rate'], errors='coerce')
    
    artifacts = {
        'n_nan_clean': clean.isna().sum(),
        'n_nan_rate': rate.isna().sum(),
        'pct_nan_clean': (clean.isna().sum() / len(clean) * 100),
        'pct_nan_rate': (rate.isna().sum() / len(rate) * 100),
    }
    
    # Detect sudden jumps in cleaned signal (potential artifacts)
    clean_diff = np.abs(np.diff(clean.dropna()))
    threshold = 3 * np.std(clean_diff)
    n_jumps = np.sum(clean_diff > threshold)
    artifacts['n_sudden_jumps'] = n_jumps
    artifacts['pct_sudden_jumps'] = (n_jumps / len(clean_diff) * 100) if len(clean_diff) > 0 else 0
    
    return artifacts


def validate_subject_condition(subject: str, condition: str, df: pd.DataFrame) -> Dict:
    """Run all validation checks for a subject-condition pair."""
    results = {
        'subject': subject,
        'condition': condition,
        'n_samples': len(df),
        'duration_sec': df['time'].max() if 'time' in df.columns else len(df) / 250,
    }
    
    # Validate ranges
    rate_validation = validate_rate_range(df)
    amp_validation = validate_amplitude_range(df)
    peaks_validation = validate_peaks_troughs(df)
    artifact_detection = detect_artifacts(df)
    
    # Merge all results
    results.update(rate_validation)
    results.update({f'amp_{k}': v for k, v in amp_validation.items()})
    results.update({f'peaks_{k}': v for k, v in peaks_validation.items()})
    results.update({f'artifact_{k}': v for k, v in artifact_detection.items()})
    
    return results


def plot_signal_with_peaks(df: pd.DataFrame, subject: str, condition: str, 
                           start_sec: float = 0, duration_sec: float = 30):
    """Plot respiratory signal with detected peaks and troughs."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Filter time window
    mask = (df['time'] >= start_sec) & (df['time'] < start_sec + duration_sec)
    df_win = df[mask].copy()
    
    if len(df_win) == 0:
        plt.close()
        return None
    
    # Plot 1: Cleaned signal with peaks/troughs
    ax = axes[0]
    ax.plot(df_win['time'], df_win['RSP_Clean'], 'b-', linewidth=1, label='RSP_Clean')
    
    if 'RSP_Peaks' in df_win.columns:
        peaks_mask = df_win['RSP_Peaks'] == 1
        if peaks_mask.any():
            ax.scatter(df_win.loc[peaks_mask, 'time'], 
                      df_win.loc[peaks_mask, 'RSP_Clean'],
                      color='red', s=50, zorder=5, label='Peaks')
    
    if 'RSP_Troughs' in df_win.columns:
        troughs_mask = df_win['RSP_Troughs'] == 1
        if troughs_mask.any():
            ax.scatter(df_win.loc[troughs_mask, 'time'],
                      df_win.loc[troughs_mask, 'RSP_Clean'],
                      color='green', s=50, zorder=5, label='Troughs')
    
    ax.set_ylabel('RSP_Clean (a.u.)')
    ax.set_title(f'{subject} - {condition}: Respiratory Signal with Peaks/Troughs')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Plot 2: Respiratory rate
    ax = axes[1]
    ax.plot(df_win['time'], df_win['RSP_Rate'], 'g-', linewidth=1.5, label='RSP_Rate')
    ax.axhline(RATE_MIN, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Min ({RATE_MIN} rpm)')
    ax.axhline(RATE_MAX, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Max ({RATE_MAX} rpm)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (rpm)')
    ax.set_title('Respiratory Rate')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, f'{subject}_{condition}_signal.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_rate_distribution(all_results: pd.DataFrame):
    """Plot distribution of respiratory rates across all subjects and conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of mean rates
    ax = axes[0, 0]
    ax.hist(all_results['mean_rate'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(RATE_MIN, color='red', linestyle='--', linewidth=2, label=f'Min ({RATE_MIN})')
    ax.axvline(RATE_MAX, color='red', linestyle='--', linewidth=2, label=f'Max ({RATE_MAX})')
    ax.set_xlabel('Mean Respiratory Rate (rpm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Respiratory Rates')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Boxplot by condition
    ax = axes[0, 1]
    conditions = ['dmt_high', 'dmt_low', 'rs_high', 'rs_low']
    data_by_cond = [all_results[all_results['condition'] == c]['mean_rate'].dropna() 
                    for c in conditions]
    ax.boxplot(data_by_cond, labels=conditions)
    ax.axhline(RATE_MIN, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(RATE_MAX, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Mean Respiratory Rate (rpm)')
    ax.set_title('Respiratory Rate by Condition')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Percentage of valid samples
    ax = axes[1, 0]
    ax.hist(all_results['pct_valid'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('% Valid Samples (within range)')
    ax.set_ylabel('Frequency')
    ax.set_title('Data Quality: % Samples in Physiological Range')
    ax.grid(alpha=0.3)
    
    # 4. Artifact detection
    ax = axes[1, 1]
    ax.scatter(all_results['artifact_pct_nan_rate'], 
              all_results['artifact_pct_sudden_jumps'],
              alpha=0.6, s=50)
    ax.set_xlabel('% NaN in Rate')
    ax.set_ylabel('% Sudden Jumps in Signal')
    ax.set_title('Artifact Detection')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'rate_distribution_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution summary plot saved: {plot_path}")


def main():
    """Main validation pipeline."""
    print("="*80)
    print("Respiratory Signal Quality Validation")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    subject_summaries = []  # Track per-subject summaries
    
    for subject in SUJETOS_VALIDADOS_RESP:
        print(f"\nValidating {subject}...")
        
        high_session, low_session = determine_sessions(subject)
        paths = build_resp_paths(subject, high_session, low_session)
        
        subject_results = []  # Track results for this subject
        
        for condition, csv_path in paths.items():
            df = load_resp_csv(csv_path)
            
            if df is None:
                print(f"  {condition}: MISSING or INVALID")
                continue
            
            # Run validation
            results = validate_subject_condition(subject, condition, df)
            all_results.append(results)
            subject_results.append(results)
            
            # Print summary
            print(f"  {condition}:")
            print(f"    Duration: {results['duration_sec']:.1f}s")
            print(f"    Valid rate samples: {results['pct_valid']:.1f}%")
            print(f"    Mean rate: {results['mean_rate']:.1f} ± {results['std_rate']:.1f} rpm")
            print(f"    Peaks detected: {results.get('peaks_n_peaks', 0)}")
            print(f"    Artifacts (NaN): {results['artifact_pct_nan_rate']:.2f}%")
            
            # Generate visualization for first 30 seconds
            plot_signal_with_peaks(df, subject, condition, start_sec=0, duration_sec=30)
        
        # Calculate per-subject summary
        if subject_results:
            pct_valids = [r['pct_valid'] for r in subject_results]
            mean_pct = np.mean(pct_valids)
            min_pct = np.min(pct_valids)
            max_pct = np.max(pct_valids)
            n_conditions = len(subject_results)
            n_below_90 = sum(1 for p in pct_valids if p < 90)
            n_below_70 = sum(1 for p in pct_valids if p < 70)
            
            subject_summaries.append({
                'subject': subject,
                'mean_pct_valid': mean_pct,
                'min_pct_valid': min_pct,
                'max_pct_valid': max_pct,
                'n_conditions': n_conditions,
                'n_below_90': n_below_90,
                'n_below_70': n_below_70,
            })
            
            # Print per-subject summary
            print(f"  {'─'*60}")
            print(f"  Subject {subject} Summary:")
            print(f"    Mean validity: {mean_pct:.1f}% (range: {min_pct:.1f}% - {max_pct:.1f}%)")
            print(f"    Conditions < 90%: {n_below_90}/{n_conditions}")
            print(f"    Conditions < 70%: {n_below_70}/{n_conditions}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    subject_summary_df = pd.DataFrame(subject_summaries)
    
    # Save summaries
    summary_path = os.path.join(OUTPUT_DIR, 'validation_summary.csv')
    results_df.to_csv(summary_path, index=False)
    print(f"\nValidation summary saved: {summary_path}")
    
    subject_summary_path = os.path.join(OUTPUT_DIR, 'subject_summary.csv')
    subject_summary_df.to_csv(subject_summary_path, index=False)
    print(f"Per-subject summary saved: {subject_summary_path}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal recordings validated: {len(results_df)}")
    print(f"Mean % valid rate samples: {results_df['pct_valid'].mean():.1f}% ± {results_df['pct_valid'].std():.1f}%")
    print(f"Mean respiratory rate: {results_df['mean_rate'].mean():.1f} ± {results_df['mean_rate'].std():.1f} rpm")
    print(f"Mean % NaN in rate: {results_df['artifact_pct_nan_rate'].mean():.2f}%")
    print(f"Mean % sudden jumps: {results_df['artifact_pct_sudden_jumps'].mean():.2f}%")
    
    # Per-subject quality ranking
    print("\n" + "="*80)
    print("PER-SUBJECT QUALITY RANKING (sorted by mean validity)")
    print("="*80)
    
    subject_summary_sorted = subject_summary_df.sort_values('mean_pct_valid', ascending=False)
    print(f"\n{'Subject':<10} {'Mean Valid':<12} {'Min Valid':<12} {'Range':<15} {'<90%':<8} {'<70%':<8}")
    print("─" * 75)
    for _, row in subject_summary_sorted.iterrows():
        range_str = f"{row['min_pct_valid']:.1f}-{row['max_pct_valid']:.1f}%"
        quality_flag = "✓" if row['n_below_70'] == 0 and row['n_below_90'] <= 1 else "⚠️"
        print(f"{quality_flag} {row['subject']:<8} {row['mean_pct_valid']:>6.1f}%      "
              f"{row['min_pct_valid']:>6.1f}%      {range_str:<15} "
              f"{row['n_below_90']}/{row['n_conditions']:<6} {row['n_below_70']}/{row['n_conditions']}")
    
    # Identify problematic recordings
    problematic = results_df[
        (results_df['pct_valid'] < 90) | 
        (results_df['artifact_pct_nan_rate'] > 5) |
        (results_df['artifact_pct_sudden_jumps'] > 1)
    ]
    
    print("\n" + "="*80)
    print("QUALITY ISSUES DETECTED")
    print("="*80)
    
    if len(problematic) > 0:
        print(f"\n⚠️  {len(problematic)} recordings with potential quality issues:")
        for _, row in problematic.iterrows():
            print(f"  - {row['subject']} ({row['condition']}): "
                  f"{row['pct_valid']:.1f}% valid, "
                  f"{row['artifact_pct_nan_rate']:.2f}% NaN")
    else:
        print("\n✓ All recordings passed quality checks!")
    
    # Identify subjects to potentially exclude
    problematic_subjects = subject_summary_df[
        (subject_summary_df['mean_pct_valid'] < 80) | 
        (subject_summary_df['n_below_70'] > 0)
    ].sort_values('mean_pct_valid')
    
    if len(problematic_subjects) > 0:
        print(f"\n⚠️  {len(problematic_subjects)} subjects recommended for exclusion:")
        for _, row in problematic_subjects.iterrows():
            print(f"  - {row['subject']}: mean {row['mean_pct_valid']:.1f}% valid, "
                  f"{row['n_below_70']} conditions < 70%, "
                  f"{row['n_below_90']} conditions < 90%")
    
    # Generate distribution plots
    plot_rate_distribution(results_df)
    
    print("\n" + "="*80)
    print("Validation Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
