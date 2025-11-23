#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot CCA Canonical Scores to Check for Outliers

This script creates scatter plots of canonical scores (U vs V) to verify
that correlations are not driven by single outlier subjects.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def plot_canonical_scores_by_subject(
    analyzer: TETPhysioCCAAnalyzer,
    state: str,
    output_dir: str
):
    """
    Plot canonical scores colored by subject to identify outliers.
    
    Args:
        analyzer: Fitted TETPhysioCCAAnalyzer
        state: 'RS' or 'DMT'
        output_dir: Directory to save figures
    """
    # Prepare matrices
    X, Y = analyzer.prepare_matrices(state)
    
    # Get subject IDs
    state_data = analyzer.data[analyzer.data['state'] == state].copy()
    X_full = state_data[analyzer.physio_measures].values
    Y_full = state_data[analyzer.tet_affective].values
    valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
    subject_ids = state_data['subject'].values[valid_mask]
    
    # Get CCA model
    cca_model = analyzer.cca_models[state]
    
    # Transform to canonical variates
    U, V = cca_model.transform(X, Y)
    
    # Get canonical correlations
    canonical_corrs = analyzer.canonical_correlations[state]
    n_components = len(canonical_corrs)
    
    # Create figure
    fig, axes = plt.subplots(1, n_components, figsize=(7 * n_components, 6))
    if n_components == 1:
        axes = [axes]
    
    # Get unique subjects and assign colors
    unique_subjects = np.unique(subject_ids)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))
    subject_color_map = dict(zip(unique_subjects, colors))
    
    for i in range(n_components):
        ax = axes[i]
        
        # Plot each subject with different color
        for subject in unique_subjects:
            subject_mask = subject_ids == subject
            ax.scatter(
                U[subject_mask, i],
                V[subject_mask, i],
                c=[subject_color_map[subject]],
                label=subject,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add regression line
        z = np.polyfit(U[:, i], V[:, i], 1)
        p = np.poly1d(z)
        x_line = np.linspace(U[:, i].min(), U[:, i].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Regression')
        
        # Compute correlation
        r = canonical_corrs[i]
        
        # Labels and title
        ax.set_xlabel(f'Physiological Canonical Score (U{i+1})', fontsize=12)
        ax.set_ylabel(f'TET Affective Canonical Score (V{i+1})', fontsize=12)
        ax.set_title(
            f'{state} - Canonical Variate {i+1}\nr = {r:.3f}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', ncol=2)
        
        # Add text with subject-level correlations
        subject_corrs = []
        for subject in unique_subjects:
            subject_mask = subject_ids == subject
            if subject_mask.sum() > 1:
                r_subj = np.corrcoef(U[subject_mask, i], V[subject_mask, i])[0, 1]
                subject_corrs.append(f'{subject}: r={r_subj:.2f}')
        
        # Add text box with subject correlations
        textstr = 'Subject-level r:\n' + '\n'.join(subject_corrs[:5])
        if len(subject_corrs) > 5:
            textstr += f'\n... ({len(subject_corrs)-5} more)'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=props
        )
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / f'cca_canonical_scores_by_subject_{state.lower()}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {fig_path}")
    
    return str(fig_path)


def compute_subject_level_statistics(
    analyzer: TETPhysioCCAAnalyzer,
    state: str
):
    """
    Compute subject-level statistics to identify outliers.
    
    Args:
        analyzer: Fitted TETPhysioCCAAnalyzer
        state: 'RS' or 'DMT'
        
    Returns:
        DataFrame with subject-level statistics
    """
    # Prepare matrices
    X, Y = analyzer.prepare_matrices(state)
    
    # Get subject IDs
    state_data = analyzer.data[analyzer.data['state'] == state].copy()
    X_full = state_data[analyzer.physio_measures].values
    Y_full = state_data[analyzer.tet_affective].values
    valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
    subject_ids = state_data['subject'].values[valid_mask]
    
    # Get CCA model
    cca_model = analyzer.cca_models[state]
    
    # Transform to canonical variates
    U, V = cca_model.transform(X, Y)
    
    # Get canonical correlations
    canonical_corrs = analyzer.canonical_correlations[state]
    n_components = len(canonical_corrs)
    
    # Compute subject-level statistics
    results = []
    
    unique_subjects = np.unique(subject_ids)
    
    for i in range(n_components):
        for subject in unique_subjects:
            subject_mask = subject_ids == subject
            n_obs = subject_mask.sum()
            
            if n_obs > 1:
                # Compute subject-level correlation
                r_subj = np.corrcoef(U[subject_mask, i], V[subject_mask, i])[0, 1]
                
                # Compute leverage (distance from centroid)
                u_mean = U[:, i].mean()
                v_mean = V[:, i].mean()
                u_subj_mean = U[subject_mask, i].mean()
                v_subj_mean = V[subject_mask, i].mean()
                
                leverage = np.sqrt(
                    (u_subj_mean - u_mean)**2 + (v_subj_mean - v_mean)**2
                )
                
                results.append({
                    'state': state,
                    'canonical_variate': i + 1,
                    'subject': subject,
                    'n_observations': n_obs,
                    'subject_correlation': r_subj,
                    'leverage': leverage,
                    'mean_U': u_subj_mean,
                    'mean_V': v_subj_mean
                })
    
    results_df = pd.DataFrame(results)
    
    # Identify potential outliers (high leverage or discordant correlation)
    for i in range(n_components):
        cv_data = results_df[results_df['canonical_variate'] == i + 1]
        
        # Compute z-scores for leverage
        leverage_mean = cv_data['leverage'].mean()
        leverage_std = cv_data['leverage'].std()
        results_df.loc[
            results_df['canonical_variate'] == i + 1,
            'leverage_z'
        ] = (cv_data['leverage'] - leverage_mean) / leverage_std
        
        # Flag outliers (|z| > 2)
        results_df.loc[
            results_df['canonical_variate'] == i + 1,
            'is_outlier'
        ] = np.abs(results_df.loc[
            results_df['canonical_variate'] == i + 1,
            'leverage_z'
        ]) > 2
    
    return results_df


def main():
    """Main execution."""
    print("=" * 80)
    print("CCA Canonical Scores Analysis - Outlier Detection")
    print("=" * 80)
    
    # Load merged data
    print("\nLoading merged data...")
    merged_data = pd.read_csv('results/tet/physio_correlation/merged_physio_tet_data.csv')
    print(f"Loaded {len(merged_data)} observations")
    
    # Initialize analyzer
    print("\nInitializing CCA analyzer...")
    analyzer = TETPhysioCCAAnalyzer(merged_data)
    
    # Fit CCA
    print("\nFitting CCA...")
    analyzer.fit_cca(n_components=2)
    
    # Plot canonical scores for each state
    output_dir = 'results/tet/physio_correlation'
    
    for state in ['RS', 'DMT']:
        if state in analyzer.cca_models:
            print(f"\n{'='*80}")
            print(f"Analyzing {state} State")
            print('='*80)
            
            # Plot scores
            print(f"\nPlotting canonical scores for {state}...")
            plot_canonical_scores_by_subject(analyzer, state, output_dir)
            
            # Compute subject-level statistics
            print(f"\nComputing subject-level statistics for {state}...")
            stats_df = compute_subject_level_statistics(analyzer, state)
            
            # Export statistics
            stats_path = Path(output_dir) / f'cca_subject_statistics_{state.lower()}.csv'
            stats_df.to_csv(stats_path, index=False)
            print(f"Saved: {stats_path}")
            
            # Display statistics
            print(f"\nSubject-Level Statistics for {state}:")
            print(stats_df.to_string(index=False))
            
            # Identify outliers
            outliers = stats_df[stats_df['is_outlier'] == True]
            if len(outliers) > 0:
                print(f"\n⚠️ Potential outliers detected in {state}:")
                print(outliers[['subject', 'canonical_variate', 'leverage_z', 'subject_correlation']])
            else:
                print(f"\n✓ No outliers detected in {state}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
