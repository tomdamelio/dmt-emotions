# -*- coding: utf-8 -*-
"""
Baseline Comparison Module for DMT Physiological Analysis.

This module provides functions for comparing extracted temporal features between
DMT conditions and Resting State (RS) baseline. It implements:
  1. Feature comparison between DMT (collapsed across doses) and RS baseline
  2. Baseline summary statistics computation for static baselines
  3. Visualization of DMT vs RS comparisons

Scientific Rationale:
- Baseline comparisons quantify the magnitude of DMT-induced changes independent of dose
- Collapsing DMT doses increases statistical power for detecting drug effects
- Static baseline statistics provide reference values for temporal features
- These analyses complement dose-specific comparisons but do not address dose-dependent effects

⚠️  IMPORTANT: These analyses do NOT address dose-dependent effects. They only quantify
    the overall magnitude of DMT-induced changes relative to baseline.

Usage:
    from scripts.baseline_comparator import (
        compare_features_to_baseline,
        compute_baseline_summary_stats,
        visualize_baseline_comparisons
    )
    
    # Compare DMT features to baseline
    comparison_df = compare_features_to_baseline(features_df)
    
    # Compute baseline summary statistics
    baseline_stats = compute_baseline_summary_stats(rs_data)
    
    # Visualize comparisons
    visualize_baseline_comparisons(comparison_df, 'output_path.pdf')

Author: DMT Analysis Pipeline
Date: 2026-01-10
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def compare_features_to_baseline(
    features_df: pd.DataFrame,
    feature_columns: List[str] = None,
    state_column: str = 'State',
    dmt_label: str = 'DMT',
    rs_label: str = 'RS',
    subject_column: str = 'subject'
) -> pd.DataFrame:
    """
    Compare DMT features (collapsed across doses) to RS baseline.
    
    This function performs paired t-tests comparing extracted temporal features
    between DMT conditions (High and Low doses combined) and Resting State baseline.
    
    Scientific Context:
    - Quantifies magnitude of DMT-induced changes independent of dose comparisons
    - Collapsing doses increases statistical power for detecting drug effects
    - Uses paired t-tests to account for within-subject variability
    - Includes effect sizes (Cohen's d) for magnitude interpretation
    
    ⚠️  IMPORTANT: This analysis does NOT address dose-dependent effects.
        For dose comparisons, use separate dose-specific analyses.
    
    Args:
        features_df: DataFrame with columns: subject, State, Dose, and feature columns
                    Must contain both 'DMT' and 'RS' states
        feature_columns: List of feature column names to compare.
                        If None, uses: ['peak_amplitude', 'time_to_peak', 't_33', 't_50']
        state_column: Name of column containing state labels (default 'State')
        dmt_label: Label for DMT condition (default 'DMT')
        rs_label: Label for RS baseline condition (default 'RS')
        subject_column: Name of column containing subject IDs (default 'subject')
    
    Returns:
        DataFrame with columns:
            - feature: Feature name
            - mean_dmt: Mean value across DMT conditions (collapsed doses)
            - std_dmt: Standard deviation across DMT conditions
            - mean_rs: Mean value across RS conditions
            - std_rs: Standard deviation across RS conditions
            - t_stat: t-statistic from paired t-test
            - df: Degrees of freedom
            - p_value: Two-tailed p-value
            - cohens_d: Cohen's d effect size
            - n_pairs: Number of paired observations
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
        KeyError: If specified feature columns don't exist
    
    Example:
        >>> features_df = pd.DataFrame({
        ...     'subject': ['sub-01', 'sub-01', 'sub-01', 'sub-02', 'sub-02', 'sub-02'],
        ...     'State': ['DMT', 'DMT', 'RS', 'DMT', 'DMT', 'RS'],
        ...     'Dose': ['High', 'Low', 'Low', 'High', 'Low', 'Low'],
        ...     'peak_amplitude': [1.5, 1.3, 0.8, 1.6, 1.4, 0.9],
        ...     'time_to_peak': [2.5, 2.3, np.nan, 2.6, 2.4, np.nan]
        ... })
        >>> comparison = compare_features_to_baseline(features_df)
        >>> print(comparison[['feature', 't_stat', 'p_value', 'cohens_d']])
    
    References:
        - Requirements 4.1, 4.2, 4.5
    """
    # Input validation
    required_columns = [subject_column, state_column]
    missing_columns = [col for col in required_columns if col not in features_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(features_df.columns)}"
        )
    
    # Set default feature columns if not provided
    if feature_columns is None:
        feature_columns = ['peak_amplitude', 'time_to_peak', 't_33', 't_50']
    
    # Check that feature columns exist
    missing_features = [col for col in feature_columns if col not in features_df.columns]
    if missing_features:
        raise KeyError(
            f"Feature columns not found: {missing_features}. "
            f"Available columns: {list(features_df.columns)}"
        )
    
    # Check that both DMT and RS states exist
    unique_states = features_df[state_column].unique()
    if dmt_label not in unique_states:
        raise ValueError(f"DMT label '{dmt_label}' not found in {state_column} column")
    if rs_label not in unique_states:
        raise ValueError(f"RS label '{rs_label}' not found in {state_column} column")
    
    # Prepare results list
    results = []
    
    # For each feature, compare DMT (collapsed) vs RS
    for feature in feature_columns:
        # Extract DMT data (all doses combined)
        dmt_data = features_df[features_df[state_column] == dmt_label].copy()
        
        # Extract RS data
        rs_data = features_df[features_df[state_column] == rs_label].copy()
        
        # Aggregate by subject for DMT (mean across doses)
        dmt_by_subject = dmt_data.groupby(subject_column)[feature].mean()
        
        # Aggregate by subject for RS (mean across sessions if multiple)
        rs_by_subject = rs_data.groupby(subject_column)[feature].mean()
        
        # Find subjects with both DMT and RS data
        common_subjects = dmt_by_subject.index.intersection(rs_by_subject.index)
        
        if len(common_subjects) < 2:
            # Not enough paired data
            results.append({
                'feature': feature,
                'mean_dmt': np.nan,
                'std_dmt': np.nan,
                'mean_rs': np.nan,
                'std_rs': np.nan,
                't_stat': np.nan,
                'df': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'n_pairs': len(common_subjects)
            })
            continue
        
        # Get paired data
        dmt_values = dmt_by_subject.loc[common_subjects].values
        rs_values = rs_by_subject.loc[common_subjects].values
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(dmt_values) | np.isnan(rs_values))
        dmt_values_clean = dmt_values[valid_mask]
        rs_values_clean = rs_values[valid_mask]
        
        if len(dmt_values_clean) < 2:
            # Not enough valid pairs
            results.append({
                'feature': feature,
                'mean_dmt': np.nan,
                'std_dmt': np.nan,
                'mean_rs': np.nan,
                'std_rs': np.nan,
                't_stat': np.nan,
                'df': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'n_pairs': len(dmt_values_clean)
            })
            continue
        
        # Compute descriptive statistics
        mean_dmt = np.mean(dmt_values_clean)
        std_dmt = np.std(dmt_values_clean, ddof=1)
        mean_rs = np.mean(rs_values_clean)
        std_rs = np.std(rs_values_clean, ddof=1)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(dmt_values_clean, rs_values_clean)
        df = len(dmt_values_clean) - 1
        
        # Compute Cohen's d for paired samples
        # d = mean_diff / std_diff
        diff = dmt_values_clean - rs_values_clean
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        results.append({
            'feature': feature,
            'mean_dmt': mean_dmt,
            'std_dmt': std_dmt,
            'mean_rs': mean_rs,
            'std_rs': std_rs,
            't_stat': t_stat,
            'df': df,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_pairs': len(dmt_values_clean)
        })
    
    return pd.DataFrame(results)


def compute_baseline_summary_stats(
    df: pd.DataFrame,
    value_column: str = 'value',
    subject_column: str = 'subject'
) -> Dict[str, float]:
    """
    Compute baseline (RS) summary statistics for comparison with DMT-derived features.
    
    This function computes descriptive statistics for Resting State baseline data.
    For static baselines without temporal variation, these statistics provide
    reference values for comparison with DMT-derived temporal features.
    
    Scientific Context:
    - Static baselines may lack temporal dynamics (e.g., constant HR during RS)
    - Summary statistics provide reference values for DMT feature comparisons
    - Useful when baseline doesn't have meaningful peak amplitude or time-to-peak
    - Complements temporal feature extraction from DMT conditions
    
    Args:
        df: Long-format dataframe filtered to RS condition
            Must contain value_column and subject_column
        value_column: Column containing signal values (default 'value')
        subject_column: Column containing subject IDs (default 'subject')
    
    Returns:
        Dictionary with keys:
            - 'mean': Overall mean across all observations
            - 'std': Overall standard deviation
            - 'median': Overall median
            - 'q25': 25th percentile
            - 'q75': 75th percentile
            - 'n_samples': Total number of observations
            - 'n_subjects': Number of unique subjects
            - 'mean_per_subject': Mean value per subject (array)
            - 'std_per_subject': Standard deviation per subject (array)
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
    
    Example:
        >>> rs_data = pd.DataFrame({
        ...     'subject': ['sub-01'] * 10 + ['sub-02'] * 10,
        ...     'value': np.random.randn(20) + 70  # HR around 70 bpm
        ... })
        >>> baseline_stats = compute_baseline_summary_stats(rs_data)
        >>> print(f"Baseline mean: {baseline_stats['mean']:.2f}")
        >>> print(f"Baseline std: {baseline_stats['std']:.2f}")
    
    References:
        - Requirements 4.3
    """
    # Input validation
    if value_column not in df.columns:
        raise ValueError(
            f"Value column '{value_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    if subject_column not in df.columns:
        raise ValueError(
            f"Subject column '{subject_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    if len(df) == 0:
        raise ValueError("Input dataframe is empty")
    
    # Extract values and remove NaNs
    values = df[value_column].values
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        raise ValueError("No valid (non-NaN) values found")
    
    # Compute overall statistics
    mean_overall = np.mean(valid_values)
    std_overall = np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0.0
    median_overall = np.median(valid_values)
    q25 = np.percentile(valid_values, 25)
    q75 = np.percentile(valid_values, 75)
    
    # Compute per-subject statistics
    subject_means = []
    subject_stds = []
    
    for subject_id in df[subject_column].unique():
        subject_data = df[df[subject_column] == subject_id][value_column].values
        subject_valid = subject_data[~np.isnan(subject_data)]
        
        if len(subject_valid) > 0:
            subject_means.append(np.mean(subject_valid))
            subject_stds.append(
                np.std(subject_valid, ddof=1) if len(subject_valid) > 1 else 0.0
            )
    
    return {
        'mean': mean_overall,
        'std': std_overall,
        'median': median_overall,
        'q25': q25,
        'q75': q75,
        'n_samples': len(valid_values),
        'n_subjects': len(subject_means),
        'mean_per_subject': np.array(subject_means),
        'std_per_subject': np.array(subject_stds)
    }


def visualize_baseline_comparisons(
    comparison_df: pd.DataFrame,
    output_path: str,
    feature_labels: Optional[Dict[str, str]] = None,
    figsize: tuple = (10, 6),
    palette: str = 'Set2'
) -> None:
    """
    Create bar plots or paired scatter plots for DMT vs RS comparisons.
    
    This function generates publication-ready visualizations of baseline comparisons,
    showing mean values with error bars and significance markers.
    
    Scientific Context:
    - Bar plots show mean ± SEM for DMT and RS conditions
    - Significance markers indicate p-values from paired t-tests
    - Effect sizes (Cohen's d) provide magnitude interpretation
    - Visualizations complement statistical tables
    
    ⚠️  IMPORTANT: These visualizations show overall DMT effects (collapsed doses).
        They do NOT address dose-dependent effects.
    
    Args:
        comparison_df: DataFrame from compare_features_to_baseline()
                      Must contain columns: feature, mean_dmt, std_dmt, mean_rs, 
                      std_rs, p_value, cohens_d, n_pairs
        output_path: Path to save figure (should include extension: .pdf, .png, .svg)
        feature_labels: Optional dictionary mapping feature names to display labels
                       Example: {'peak_amplitude': 'Peak Amplitude (z-score)',
                                'time_to_peak': 'Time to Peak (min)'}
        figsize: Figure size as (width, height) tuple (default (10, 6))
        palette: Seaborn color palette name (default 'Set2')
    
    Returns:
        None (saves figure to output_path)
    
    Raises:
        ValueError: If required columns are missing or output path is invalid
    
    Example:
        >>> comparison_df = compare_features_to_baseline(features_df)
        >>> feature_labels = {
        ...     'peak_amplitude': 'Peak Amplitude (z)',
        ...     'time_to_peak': 'Time to Peak (min)'
        ... }
        >>> visualize_baseline_comparisons(
        ...     comparison_df,
        ...     'results/baseline_comparison.pdf',
        ...     feature_labels=feature_labels
        ... )
    
    References:
        - Requirements 4.4, 4.5
    """
    # Input validation
    required_columns = ['feature', 'mean_dmt', 'std_dmt', 'mean_rs', 'std_rs', 
                       'p_value', 'cohens_d', 'n_pairs']
    missing_columns = [col for col in required_columns if col not in comparison_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(comparison_df.columns)}"
        )
    
    # Filter out features with insufficient data
    valid_df = comparison_df[comparison_df['n_pairs'] >= 2].copy()
    
    if len(valid_df) == 0:
        raise ValueError("No features with sufficient paired data (n_pairs >= 2)")
    
    # Set up feature labels
    if feature_labels is None:
        feature_labels = {feat: feat.replace('_', ' ').title() 
                         for feat in valid_df['feature']}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    n_features = len(valid_df)
    x_positions = np.arange(n_features)
    bar_width = 0.35
    
    # Get colors from palette
    colors = sns.color_palette(palette, 2)
    
    # Compute SEM for error bars
    valid_df['sem_dmt'] = valid_df['std_dmt'] / np.sqrt(valid_df['n_pairs'])
    valid_df['sem_rs'] = valid_df['std_rs'] / np.sqrt(valid_df['n_pairs'])
    
    # Plot bars
    ax.bar(
        x_positions - bar_width/2,
        valid_df['mean_dmt'],
        bar_width,
        yerr=valid_df['sem_dmt'],
        label='DMT (collapsed doses)',
        color=colors[0],
        capsize=5,
        alpha=0.8
    )
    
    ax.bar(
        x_positions + bar_width/2,
        valid_df['mean_rs'],
        bar_width,
        yerr=valid_df['sem_rs'],
        label='Resting State',
        color=colors[1],
        capsize=5,
        alpha=0.8
    )
    
    # Add significance markers
    for idx, row in valid_df.iterrows():
        x_pos = x_positions[idx]
        y_pos = max(row['mean_dmt'] + row['sem_dmt'], 
                   row['mean_rs'] + row['sem_rs']) * 1.1
        
        # Determine significance marker
        if row['p_value'] < 0.001:
            marker = '***'
        elif row['p_value'] < 0.01:
            marker = '**'
        elif row['p_value'] < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        
        # Add marker
        ax.text(
            x_pos, y_pos, marker,
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
        
        # Add effect size annotation
        ax.text(
            x_pos, y_pos * 1.05,
            f"d={row['cohens_d']:.2f}",
            ha='center', va='bottom',
            fontsize=8, style='italic'
        )
    
    # Customize plot
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value (mean ± SEM)', fontsize=12, fontweight='bold')
    ax.set_title(
        'DMT vs Resting State Baseline Comparison\n'
        '⚠️  DMT doses collapsed (does not address dose-dependent effects)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [feature_labels.get(feat, feat) for feat in valid_df['feature']],
        rotation=45, ha='right'
    )
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add significance legend
    sig_text = (
        "Significance: * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant\n"
        "Effect size: Cohen's d (small: 0.2, medium: 0.5, large: 0.8)"
    )
    ax.text(
        0.02, 0.98, sig_text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Baseline comparison figure saved to: {output_path}")


def format_baseline_comparison_report(
    comparison_df: pd.DataFrame,
    baseline_stats: Optional[Dict[str, float]] = None
) -> str:
    """
    Format baseline comparison results for reporting.
    
    Creates a human-readable summary of DMT vs RS baseline comparisons,
    including statistical results and effect sizes.
    
    Args:
        comparison_df: DataFrame from compare_features_to_baseline()
        baseline_stats: Optional dictionary from compute_baseline_summary_stats()
    
    Returns:
        Formatted string summarizing the results
    
    Example:
        >>> comparison_df = compare_features_to_baseline(features_df)
        >>> baseline_stats = compute_baseline_summary_stats(rs_data)
        >>> report = format_baseline_comparison_report(comparison_df, baseline_stats)
        >>> print(report)
    """
    lines = [
        "=" * 80,
        "DMT vs Resting State Baseline Comparison",
        "=" * 80,
        "⚠️  IMPORTANT: DMT doses are collapsed. This analysis does NOT address",
        "   dose-dependent effects. For dose comparisons, see dose-specific analyses.",
        "",
    ]
    
    # Add baseline summary statistics if provided
    if baseline_stats is not None:
        lines.extend([
            "Resting State Baseline Summary:",
            "-" * 80,
            f"  Mean: {baseline_stats['mean']:.3f}",
            f"  SD: {baseline_stats['std']:.3f}",
            f"  Median: {baseline_stats['median']:.3f}",
            f"  IQR: [{baseline_stats['q25']:.3f}, {baseline_stats['q75']:.3f}]",
            f"  N samples: {baseline_stats['n_samples']}",
            f"  N subjects: {baseline_stats['n_subjects']}",
            ""
        ])
    
    # Add feature comparisons
    lines.extend([
        "Feature Comparisons (Paired t-tests):",
        "-" * 80,
    ])
    
    for idx, row in comparison_df.iterrows():
        feature = row['feature']
        
        if row['n_pairs'] < 2:
            lines.append(f"\n{feature}: Insufficient paired data (n={row['n_pairs']})")
            continue
        
        # Format statistics
        lines.extend([
            f"\n{feature}:",
            f"  DMT (collapsed): M = {row['mean_dmt']:.3f}, SD = {row['std_dmt']:.3f}",
            f"  Resting State:   M = {row['mean_rs']:.3f}, SD = {row['std_rs']:.3f}",
            f"  t({row['df']:.0f}) = {row['t_stat']:.3f}, p = {row['p_value']:.4f}",
            f"  Cohen's d = {row['cohens_d']:.3f}",
            f"  N pairs = {row['n_pairs']:.0f}",
        ])
        
        # Add interpretation
        if row['p_value'] < 0.05:
            direction = "higher" if row['mean_dmt'] > row['mean_rs'] else "lower"
            magnitude = (
                "large" if abs(row['cohens_d']) >= 0.8 else
                "medium" if abs(row['cohens_d']) >= 0.5 else
                "small"
            )
            lines.append(
                f"  → DMT significantly {direction} than RS ({magnitude} effect)"
            )
        else:
            lines.append("  → No significant difference")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
