#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze subject-level heterogeneity in physiological-affective coupling.

This script examines whether the non-significant DMT result is due to
heterogeneous coupling patterns across subjects.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import CCA
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader


def compute_subject_level_correlations(merged_df: pd.DataFrame):
    """Compute correlations within each subject for each state."""
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    tet_key_var = 'emotional_intensity_z'  # Key TET variable
    
    results = []
    
    for state in ['RS', 'DMT']:
        state_data = merged_df[merged_df['state'] == state]
        
        for subject in state_data['subject'].unique():
            subj_data = state_data[state_data['subject'] == subject]
            
            # Correlations with emotional intensity
            for physio_var in physio_vars:
                if len(subj_data) > 5:  # Need enough data points
                    r, p = stats.pearsonr(
                        subj_data[physio_var].values,
                        subj_data[tet_key_var].values
                    )
                    results.append({
                        'state': state,
                        'subject': subject,
                        'physio_var': physio_var,
                        'tet_var': 'emotional_intensity',
                        'r': r,
                        'p_value': p,
                        'n_obs': len(subj_data)
                    })
    
    return pd.DataFrame(results)


def compute_subject_cca(merged_df: pd.DataFrame):
    """Compute CCA within each subject for each state."""
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    tet_vars = [
        'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
        'interoception_z', 'bliss_z', 'anxiety_z'
    ]
    
    results = []
    
    for state in ['RS', 'DMT']:
        state_data = merged_df[merged_df['state'] == state]
        
        for subject in state_data['subject'].unique():
            subj_data = state_data[state_data['subject'] == subject]
            
            X = subj_data[physio_vars].values
            Y = subj_data[tet_vars].values
            
            # Remove NaN
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
            X = X[valid]
            Y = Y[valid]
            
            if len(X) > 10:  # Need enough observations
                # Standardize within subject
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
                Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)
                
                # Fit CCA with 1 component
                try:
                    cca = CCA(n_components=1)
                    cca.fit(X, Y)
                    U, V = cca.transform(X, Y)
                    r = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
                    
                    results.append({
                        'state': state,
                        'subject': subject,
                        'cca_r': r,
                        'n_obs': len(X)
                    })
                except Exception as e:
                    print(f"CCA failed for {subject} in {state}: {e}")
    
    return pd.DataFrame(results)


def plot_subject_heterogeneity(corr_df: pd.DataFrame, cca_df: pd.DataFrame,
                               output_dir: Path):
    """Visualize subject-level heterogeneity."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel A: Subject-level correlations (HR vs Emotional Intensity)
    ax = axes[0, 0]
    hr_corrs = corr_df[corr_df['physio_var'] == 'HR']
    
    for state, color in [('RS', 'steelblue'), ('DMT', 'coral')]:
        state_data = hr_corrs[hr_corrs['state'] == state]
        subjects = state_data['subject'].values
        correlations = state_data['r'].values
        
        x_pos = np.arange(len(subjects)) + (0.2 if state == 'DMT' else -0.2)
        ax.bar(x_pos, correlations, width=0.35, label=state, color=color, alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('A. HR vs Emotional Intensity\n(Within-Subject Correlations)')
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(subjects))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel B: Subject-level CCA correlations
    ax = axes[0, 1]
    
    for state, color in [('RS', 'steelblue'), ('DMT', 'coral')]:
        state_data = cca_df[cca_df['state'] == state]
        subjects = state_data['subject'].values
        cca_r = state_data['cca_r'].values
        
        x_pos = np.arange(len(subjects)) + (0.2 if state == 'DMT' else -0.2)
        ax.bar(x_pos, cca_r, width=0.35, label=state, color=color, alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Subject')
    ax.set_ylabel('CCA Correlation (r)')
    ax.set_title('B. Within-Subject CCA\n(Physio-TET Coupling per Subject)')
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels([f'S{i+1}' for i in range(len(subjects))])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel C: Distribution of correlations
    ax = axes[1, 0]
    
    rs_cca = cca_df[cca_df['state'] == 'RS']['cca_r'].values
    dmt_cca = cca_df[cca_df['state'] == 'DMT']['cca_r'].values
    
    positions = [1, 2]
    bp = ax.boxplot([rs_cca, dmt_cca], positions=positions, widths=0.6,
                    patch_artist=True)
    
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, (data, pos) in enumerate(zip([rs_cca, dmt_cca], positions)):
        x = np.random.normal(pos, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.8, color=colors[i], edgecolor='black', s=50)
    
    ax.set_xticklabels(['RS', 'DMT'])
    ax.set_ylabel('Within-Subject CCA Correlation')
    ax.set_title('C. Distribution of Subject-Level CCA')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    rs_mean, rs_std = rs_cca.mean(), rs_cca.std()
    dmt_mean, dmt_std = dmt_cca.mean(), dmt_cca.std()
    t_stat, p_val = stats.ttest_ind(rs_cca, dmt_cca)
    
    ax.text(0.05, 0.95, 
            f'RS: {rs_mean:.2f} ± {rs_std:.2f}\n'
            f'DMT: {dmt_mean:.2f} ± {dmt_std:.2f}\n'
            f't = {t_stat:.2f}, p = {p_val:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel D: Heterogeneity comparison
    ax = axes[1, 1]
    
    # Coefficient of variation for CCA correlations
    rs_cv = rs_cca.std() / abs(rs_cca.mean()) if rs_cca.mean() != 0 else 0
    dmt_cv = dmt_cca.std() / abs(dmt_cca.mean()) if dmt_cca.mean() != 0 else 0
    
    # Range
    rs_range = rs_cca.max() - rs_cca.min()
    dmt_range = dmt_cca.max() - dmt_cca.min()
    
    metrics = ['SD', 'Range', 'CV']
    rs_vals = [rs_std, rs_range, rs_cv]
    dmt_vals = [dmt_std, dmt_range, dmt_cv]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, rs_vals, width, label='RS', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, dmt_vals, width, label='DMT', color='coral', alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('D. Heterogeneity Metrics\n(Higher = More Variable Across Subjects)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Subject-Level Heterogeneity in Physiological-Affective Coupling',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / 'subject_coupling_heterogeneity.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def main():
    """Run heterogeneity analysis."""
    
    print("=" * 80)
    print("SUBJECT-LEVEL COUPLING HETEROGENEITY ANALYSIS")
    print("=" * 80)
    
    output_dir = Path('results/tet/physio_correlation/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    loader = TETPhysioDataLoader(
        composite_physio_path='results/composite/arousal_index_long.csv',
        tet_path='results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec=30
    )
    
    loader.load_physiological_data()
    loader.load_tet_data()
    merged_df = loader.merge_datasets()
    
    # Compute subject-level correlations
    print("\nComputing subject-level correlations...")
    corr_df = compute_subject_level_correlations(merged_df)
    corr_df.to_csv(output_dir / 'subject_level_correlations.csv', index=False)
    
    # Compute subject-level CCA
    print("Computing subject-level CCA...")
    cca_df = compute_subject_cca(merged_df)
    cca_df.to_csv(output_dir / 'subject_level_cca.csv', index=False)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\nWithin-Subject CCA Correlations:")
    print("-" * 40)
    
    for state in ['RS', 'DMT']:
        state_cca = cca_df[cca_df['state'] == state]['cca_r']
        print(f"\n{state}:")
        print(f"  Mean: {state_cca.mean():.3f}")
        print(f"  SD:   {state_cca.std():.3f}")
        print(f"  Min:  {state_cca.min():.3f}")
        print(f"  Max:  {state_cca.max():.3f}")
        print(f"  Individual subjects: {state_cca.values}")
    
    # Statistical comparison
    rs_cca = cca_df[cca_df['state'] == 'RS']['cca_r'].values
    dmt_cca = cca_df[cca_df['state'] == 'DMT']['cca_r'].values
    
    t_stat, p_val = stats.ttest_ind(rs_cca, dmt_cca)
    levene_stat, levene_p = stats.levene(rs_cca, dmt_cca)
    
    print("\n" + "-" * 40)
    print("Statistical Comparisons:")
    print(f"  t-test (means): t={t_stat:.3f}, p={p_val:.3f}")
    print(f"  Levene's test (variances): F={levene_stat:.3f}, p={levene_p:.3f}")
    
    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    
    if dmt_cca.std() > rs_cca.std():
        print("\n→ DMT shows HIGHER heterogeneity in coupling across subjects")
        print("  This explains why the group-level permutation test fails:")
        print("  - Some subjects have strong positive coupling")
        print("  - Others have weak or even negative coupling")
        print("  - When permuting subjects, this variability creates a wide null distribution")
    else:
        print("\n→ RS shows higher heterogeneity")
    
    # Generate figure
    print("\nGenerating visualization...")
    fig_path = plot_subject_heterogeneity(corr_df, cca_df, output_dir)
    print(f"Saved: {fig_path}")
    
    return corr_df, cca_df


if __name__ == '__main__':
    main()
