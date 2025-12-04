#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize CCA Latent Space: Canonical Variates U vs V.

This script creates scatterplots of the canonical variates (U vs V) for RS and DMT,
using Theil BLUS residualized data. Points are colored by subject to reveal:
- Universal coupling (DMT): elongated cloud with all subjects following same slope
- Idiosyncratic coupling (RS): clusters or crossing individual slopes

Author: TET Analysis Pipeline
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cross_decomposition import CCA
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Subject colors - distinct palette
SUBJECT_COLORS = {
    'S04': '#E41A1C',  # Red
    'S06': '#377EB8',  # Blue
    'S07': '#4DAF4A',  # Green
    'S16': '#984EA3',  # Purple
    'S18': '#FF7F00',  # Orange
    'S19': '#A65628',  # Brown
    'S20': '#F781BF',  # Pink
}


def load_data():
    """Load and prepare data."""
    loader = TETPhysioDataLoader(
        composite_physio_path='results/composite/arousal_index_long.csv',
        tet_path='results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec=30
    )
    loader.load_physiological_data()
    loader.load_tet_data()
    merged_df = loader.merge_datasets()
    
    analyzer = TETPhysioCCAAnalyzer(merged_df)
    analyzer.fit_cca(n_components=3)
    
    return merged_df, analyzer


def compute_canonical_variates_theil(analyzer, state):
    """
    Compute canonical variates U and V using Theil BLUS residuals.
    
    Returns:
        U: Physiological canonical variate (n_obs,)
        V: Affective canonical variate (n_obs,)
        subject_ids: Subject IDs for each observation
        r: Canonical correlation
    """
    # Get original data
    X, Y = analyzer.prepare_matrices(state)
    
    # Get subject IDs
    state_data = analyzer.data[analyzer.data['state'] == state].copy()
    subject_ids_full = state_data['subject'].values
    X_full = state_data[analyzer.physio_measures].values
    Y_full = state_data[analyzer.tet_affective].values
    valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
    subject_ids = subject_ids_full[valid_mask]
    
    # Compute Theil BLUS residuals
    Y_blus, X_blus, blus_subject_ids = analyzer._compute_theil_residuals_for_blocks(
        Y, X, subject_ids
    )
    
    # Fit CCA on BLUS residuals
    cca = CCA(n_components=1)
    cca.fit(X_blus, Y_blus)
    
    # Get canonical weights
    A = cca.x_weights_  # Physiological weights
    B = cca.y_weights_  # Affective weights
    
    # Compute canonical variates: U = X @ A, V = Y @ B
    U = X_blus @ A
    V = Y_blus @ B
    
    # Flatten to 1D
    U = U.flatten()
    V = V.flatten()
    
    # Compute canonical correlation
    r = np.corrcoef(U, V)[0, 1]
    
    return U, V, blus_subject_ids, r, cca


def compute_subject_slopes(U, V, subject_ids):
    """Compute regression slope for each subject."""
    slopes = {}
    unique_subjects = np.unique(subject_ids)
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        if mask.sum() > 2:
            slope, intercept, r, p, se = stats.linregress(U[mask], V[mask])
            slopes[subj] = {
                'slope': slope,
                'intercept': intercept,
                'r': r,
                'n': mask.sum()
            }
    
    return slopes


def create_latent_space_figure(analyzer, output_dir):
    """Create the latent space visualization figure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, state in enumerate(['RS', 'DMT']):
        ax = axes[idx]
        
        # Compute canonical variates
        U, V, subject_ids, r, cca = compute_canonical_variates_theil(analyzer, state)
        
        # Compute subject slopes
        slopes = compute_subject_slopes(U, V, subject_ids)
        
        # Plot each subject with different color
        unique_subjects = sorted(np.unique(subject_ids))
        
        for subj in unique_subjects:
            mask = subject_ids == subj
            color = SUBJECT_COLORS.get(subj, 'gray')
            
            ax.scatter(U[mask], V[mask], c=color, alpha=0.6, s=30,
                      edgecolor='white', linewidth=0.3, label=subj)
            
            # Add subject regression line
            if subj in slopes:
                x_range = np.array([U[mask].min(), U[mask].max()])
                y_pred = slopes[subj]['slope'] * x_range + slopes[subj]['intercept']
                ax.plot(x_range, y_pred, color=color, alpha=0.5, linewidth=1.5,
                       linestyle='--')
        
        # Add global regression line
        slope_global, intercept_global, _, _, _ = stats.linregress(U, V)
        x_global = np.array([U.min(), U.max()])
        y_global = slope_global * x_global + intercept_global
        ax.plot(x_global, y_global, 'k-', linewidth=2.5, label='Global fit',
               zorder=10)
        
        # Labels and title
        ax.set_xlabel('Physiological Canonical Variate (U)', fontsize=10)
        ax.set_ylabel('Affective Canonical Variate (V)', fontsize=10)
        ax.set_title(f'{state}: Canonical Correlation r = {r:.3f}',
                    fontsize=11, fontweight='bold')
        
        # Add interpretation annotation
        if state == 'RS':
            interpretation = "Heterogeneous:\nSubject slopes vary\n(some +, some −)"
            bbox_color = 'lightyellow'
        else:
            interpretation = "Homogeneous:\nAll subjects follow\nsimilar positive slope"
            bbox_color = 'lightgreen'
        
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8,
                        edgecolor='gray'))
        
        # Add slope statistics
        slope_values = [s['slope'] for s in slopes.values()]
        n_positive = sum(1 for s in slope_values if s > 0)
        n_total = len(slope_values)
        
        ax.text(0.98, 0.02, f'Positive slopes: {n_positive}/{n_total}',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    
    # Create legend
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=SUBJECT_COLORS[s], markersize=8,
                             label=s) for s in sorted(SUBJECT_COLORS.keys())]
    legend_elements.append(Line2D([0], [0], color='black', linewidth=2,
                                  label='Global fit'))
    legend_elements.append(Line2D([0], [0], color='gray', linewidth=1.5,
                                  linestyle='--', label='Subject fit'))
    
    fig.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(1.12, 0.5), frameon=True, fancybox=False,
              edgecolor='black')
    
    plt.suptitle('CCA Latent Space: Physiological vs Affective Canonical Variates\n'
                '(Theil BLUS Residuals)', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'Figure4_CCA_LatentSpace.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return output_path


def create_detailed_comparison_figure(analyzer, output_dir):
    """Create a more detailed figure with individual subject panels."""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Main panels for RS and DMT
    ax_rs = fig.add_subplot(2, 2, 1)
    ax_dmt = fig.add_subplot(2, 2, 2)
    
    # Slope comparison panel
    ax_slopes = fig.add_subplot(2, 2, 3)
    
    # Summary panel
    ax_summary = fig.add_subplot(2, 2, 4)
    
    all_slopes = {'RS': {}, 'DMT': {}}
    
    for state, ax in [('RS', ax_rs), ('DMT', ax_dmt)]:
        U, V, subject_ids, r, cca = compute_canonical_variates_theil(analyzer, state)
        slopes = compute_subject_slopes(U, V, subject_ids)
        all_slopes[state] = slopes
        
        unique_subjects = sorted(np.unique(subject_ids))
        
        for subj in unique_subjects:
            mask = subject_ids == subj
            color = SUBJECT_COLORS.get(subj, 'gray')
            ax.scatter(U[mask], V[mask], c=color, alpha=0.6, s=25,
                      edgecolor='white', linewidth=0.3)
            
            if subj in slopes:
                x_range = np.array([U[mask].min(), U[mask].max()])
                y_pred = slopes[subj]['slope'] * x_range + slopes[subj]['intercept']
                ax.plot(x_range, y_pred, color=color, alpha=0.7, linewidth=1.5)
        
        # Global fit
        slope_g, intercept_g, _, _, _ = stats.linregress(U, V)
        x_g = np.array([U.min(), U.max()])
        ax.plot(x_g, slope_g * x_g + intercept_g, 'k-', linewidth=2.5, zorder=10)
        
        ax.set_xlabel('Physiological Variate (U)')
        ax.set_ylabel('Affective Variate (V)')
        ax.set_title(f'{state}: r = {r:.3f}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    
    # Slope comparison bar chart
    subjects = sorted(SUBJECT_COLORS.keys())
    x = np.arange(len(subjects))
    width = 0.35
    
    rs_slopes = [all_slopes['RS'].get(s, {}).get('slope', 0) for s in subjects]
    dmt_slopes = [all_slopes['DMT'].get(s, {}).get('slope', 0) for s in subjects]
    
    ax_slopes.bar(x - width/2, rs_slopes, width, label='RS', color='#4A90A4', alpha=0.8)
    ax_slopes.bar(x + width/2, dmt_slopes, width, label='DMT', color='#E07B54', alpha=0.8)
    ax_slopes.axhline(0, color='black', linewidth=0.5)
    ax_slopes.set_xlabel('Subject')
    ax_slopes.set_ylabel('Regression Slope (U → V)')
    ax_slopes.set_title('C. Subject-Level Slopes', fontweight='bold')
    ax_slopes.set_xticks(x)
    ax_slopes.set_xticklabels(subjects)
    ax_slopes.legend()
    ax_slopes.grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    ax_summary.axis('off')
    
    rs_slope_vals = [s['slope'] for s in all_slopes['RS'].values()]
    dmt_slope_vals = [s['slope'] for s in all_slopes['DMT'].values()]
    
    summary_text = f"""
    SUMMARY STATISTICS
    
    RS State:
      • Mean slope: {np.mean(rs_slope_vals):.3f}
      • SD slope: {np.std(rs_slope_vals):.3f}
      • Positive slopes: {sum(1 for s in rs_slope_vals if s > 0)}/7
      • Slope range: [{min(rs_slope_vals):.3f}, {max(rs_slope_vals):.3f}]
    
    DMT State:
      • Mean slope: {np.mean(dmt_slope_vals):.3f}
      • SD slope: {np.std(dmt_slope_vals):.3f}
      • Positive slopes: {sum(1 for s in dmt_slope_vals if s > 0)}/7
      • Slope range: [{min(dmt_slope_vals):.3f}, {max(dmt_slope_vals):.3f}]
    
    INTERPRETATION:
    RS shows heterogeneous slopes (mixed signs),
    indicating subject-specific coupling patterns.
    
    DMT shows homogeneous positive slopes,
    indicating universal coupling across subjects.
    """
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    ax_summary.set_title('D. Summary', fontweight='bold')
    
    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=SUBJECT_COLORS[s], markersize=8,
                             label=s) for s in subjects]
    fig.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, 0.02), ncol=7, frameon=True)
    
    plt.suptitle('CCA Latent Space Analysis: Universal vs Idiosyncratic Coupling',
                fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    output_path = output_dir / 'Figure4_CCA_LatentSpace_Detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    return output_path


def main():
    print("=" * 80)
    print("Generating CCA Latent Space Visualization")
    print("=" * 80)
    
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_df, analyzer = load_data()
    
    print("\nCreating main latent space figure...")
    create_latent_space_figure(analyzer, output_dir)
    
    print("\nCreating detailed comparison figure...")
    create_detailed_comparison_figure(analyzer, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
