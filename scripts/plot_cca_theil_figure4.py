#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 4: CCA Analysis with Theil BLUS Permutation Testing.

This figure shows:
A) Within-subject correlations (HR vs Emotional Intensity) by state
B) Permutation null distributions with observed values
C) CCA canonical loadings for DMT
D) Subject-level CCA correlations comparison

Author: TET Analysis Pipeline
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cross_decomposition import CCA
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer

# Publication style settings
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

# Color scheme
RS_COLOR = '#4A90A4'  # Muted blue
DMT_COLOR = '#E07B54'  # Coral/orange
SUBJECT_COLORS = plt.cm.Set2(np.linspace(0, 1, 7))


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


def compute_within_subject_correlations(merged_df):
    """Compute within-subject HR-EmotionalIntensity correlations."""
    results = []
    
    for state in ['RS', 'DMT']:
        state_data = merged_df[merged_df['state'] == state]
        
        for subject in sorted(state_data['subject'].unique()):
            subj_data = state_data[state_data['subject'] == subject]
            
            # Demean within subject
            hr_demeaned = subj_data['HR'] - subj_data['HR'].mean()
            ei_demeaned = subj_data['emotional_intensity_z'] - subj_data['emotional_intensity_z'].mean()
            
            r, p = stats.pearsonr(hr_demeaned, ei_demeaned)
            results.append({
                'state': state,
                'subject': subject,
                'r': r,
                'p': p
            })
    
    return pd.DataFrame(results)


def compute_permutation_distributions(analyzer, n_sample=300):
    """Compute permutation null distributions for both states."""
    distributions = {}
    observed = {}
    
    for state in ['RS', 'DMT']:
        X, Y = analyzer.prepare_matrices(state)
        state_data = analyzer.data[analyzer.data['state'] == state].copy()
        subject_ids = state_data['subject'].values[
            ~(np.isnan(state_data[analyzer.physio_measures].values).any(axis=1) |
              np.isnan(state_data[analyzer.tet_affective].values).any(axis=1))
        ]
        
        # Get BLUS residuals
        Y_blus, X_blus, blus_ids = analyzer._compute_theil_residuals_for_blocks(
            Y, X, subject_ids
        )
        
        # Observed
        cca = CCA(n_components=3)
        cca.fit(X_blus, Y_blus)
        U, V = cca.transform(X_blus, Y_blus)
        obs_r = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        observed[state] = obs_r
        
        # Permutation distribution
        unique_subjects = np.unique(blus_ids)
        n_subjects = len(unique_subjects)
        all_derangements = list(analyzer._generate_all_derangements(n_subjects))
        
        # Sample if too many
        if len(all_derangements) > n_sample:
            sample_idx = np.random.choice(len(all_derangements), n_sample, replace=False)
            sample_derangements = [all_derangements[i] for i in sample_idx]
        else:
            sample_derangements = all_derangements
        
        perm_r = []
        for derangement in sample_derangements:
            X_perm, Y_perm = analyzer._apply_derangement(
                X_blus, Y_blus, blus_ids, derangement
            )
            cca_perm = CCA(n_components=1)
            cca_perm.fit(X_perm, Y_perm)
            U_perm, V_perm = cca_perm.transform(X_perm, Y_perm)
            perm_r.append(np.corrcoef(U_perm[:, 0], V_perm[:, 0])[0, 1])
        
        distributions[state] = np.array(perm_r)
    
    return distributions, observed


def get_cca_loadings(analyzer, state='DMT'):
    """Get CCA loadings for a state using BLUS residuals."""
    X, Y = analyzer.prepare_matrices(state)
    state_data = analyzer.data[analyzer.data['state'] == state].copy()
    subject_ids = state_data['subject'].values[
        ~(np.isnan(state_data[analyzer.physio_measures].values).any(axis=1) |
          np.isnan(state_data[analyzer.tet_affective].values).any(axis=1))
    ]
    
    Y_blus, X_blus, _ = analyzer._compute_theil_residuals_for_blocks(Y, X, subject_ids)
    
    cca = CCA(n_components=1)
    cca.fit(X_blus, Y_blus)
    U, V = cca.transform(X_blus, Y_blus)
    
    # Compute loadings (correlations with canonical variates)
    physio_loadings = {}
    for i, var in enumerate(['HR', 'SMNA', 'RVT']):
        physio_loadings[var] = np.corrcoef(X_blus[:, i], U[:, 0])[0, 1]
    
    tet_names = ['Pleasant', 'Unpleasant', 'Emot. Int.', 'Interocep.', 'Bliss', 'Anxiety']
    tet_loadings = {}
    for i, var in enumerate(tet_names):
        tet_loadings[var] = np.corrcoef(Y_blus[:, i], V[:, 0])[0, 1]
    
    return physio_loadings, tet_loadings


def create_figure(merged_df, analyzer, output_dir):
    """Create the main figure."""
    
    print("Computing data for figure...")
    
    # Compute data
    within_subj_corr = compute_within_subject_correlations(merged_df)
    perm_dist, observed = compute_permutation_distributions(analyzer)
    physio_loadings, tet_loadings = get_cca_loadings(analyzer, 'DMT')
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Within-subject correlations by state
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    subjects = sorted(within_subj_corr['subject'].unique())
    x = np.arange(len(subjects))
    width = 0.35
    
    rs_corrs = within_subj_corr[within_subj_corr['state'] == 'RS'].set_index('subject').loc[subjects, 'r']
    dmt_corrs = within_subj_corr[within_subj_corr['state'] == 'DMT'].set_index('subject').loc[subjects, 'r']
    
    bars1 = ax_a.bar(x - width/2, rs_corrs, width, label='RS', color=RS_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax_a.bar(x + width/2, dmt_corrs, width, label='DMT', color=DMT_COLOR, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax_a.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax_a.set_ylabel('Within-Subject Correlation (r)')
    ax_a.set_xlabel('Subject')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f'S{i+1}' for i in range(len(subjects))])
    ax_a.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    ax_a.set_ylim(-1, 1)
    ax_a.set_title('A. Within-Subject HR–Emotional Intensity Coupling', fontweight='bold', loc='left')
    
    # Add annotation about sign consistency
    n_pos_rs = (rs_corrs > 0).sum()
    n_pos_dmt = (dmt_corrs > 0).sum()
    ax_a.text(0.02, 0.02, f'RS: {n_pos_rs}/7 positive\nDMT: {n_pos_dmt}/7 positive',
             transform=ax_a.transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # =========================================================================
    # Panel B: Permutation null distributions
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # RS distribution
    ax_b.hist(perm_dist['RS'], bins=25, alpha=0.6, color=RS_COLOR, 
             label='RS null', edgecolor='black', linewidth=0.3)
    ax_b.axvline(observed['RS'], color=RS_COLOR, linewidth=2, linestyle='--',
                label=f'RS observed (r={observed["RS"]:.2f})')
    
    # DMT distribution
    ax_b.hist(perm_dist['DMT'], bins=25, alpha=0.6, color=DMT_COLOR,
             label='DMT null', edgecolor='black', linewidth=0.3)
    ax_b.axvline(observed['DMT'], color=DMT_COLOR, linewidth=2, linestyle='--',
                label=f'DMT observed (r={observed["DMT"]:.2f})')
    
    ax_b.set_xlabel('Canonical Correlation (r)')
    ax_b.set_ylabel('Frequency')
    ax_b.set_title('B. Permutation Null Distributions (Theil BLUS)', fontweight='bold', loc='left')
    ax_b.legend(loc='upper left', fontsize=7, frameon=True, fancybox=False, edgecolor='black')
    
    # Add p-values
    p_rs = (perm_dist['RS'] >= observed['RS']).mean()
    p_dmt = (perm_dist['DMT'] >= observed['DMT']).mean()
    ax_b.text(0.98, 0.98, f'RS: p = {p_rs:.3f}\nDMT: p = {p_dmt:.3f}',
             transform=ax_b.transAxes, fontsize=8, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # =========================================================================
    # Panel C: CCA Loadings (DMT)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Physio loadings
    physio_vars = list(physio_loadings.keys())
    physio_vals = [physio_loadings[v] for v in physio_vars]
    
    y_pos = np.arange(len(physio_vars))
    colors = [DMT_COLOR if v > 0 else RS_COLOR for v in physio_vals]
    
    bars = ax_c.barh(y_pos, physio_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(physio_vars)
    ax_c.axvline(0, color='black', linewidth=0.5)
    ax_c.axvline(0.3, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_c.axvline(-0.3, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_c.set_xlabel('Canonical Loading (r)')
    ax_c.set_title('C. Physiological Loadings (DMT CV1)', fontweight='bold', loc='left')
    ax_c.set_xlim(-1, 1)
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, physio_vals)):
        ax_c.text(val + 0.05 if val > 0 else val - 0.05, i, f'{val:.2f}',
                 va='center', ha='left' if val > 0 else 'right', fontsize=8)
    
    # =========================================================================
    # Panel D: TET Loadings (DMT)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    
    tet_vars = list(tet_loadings.keys())
    tet_vals = [tet_loadings[v] for v in tet_vars]
    
    y_pos = np.arange(len(tet_vars))
    colors = [DMT_COLOR if v > 0 else RS_COLOR for v in tet_vals]
    
    bars = ax_d.barh(y_pos, tet_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(tet_vars)
    ax_d.axvline(0, color='black', linewidth=0.5)
    ax_d.axvline(0.3, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_d.axvline(-0.3, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax_d.set_xlabel('Canonical Loading (r)')
    ax_d.set_title('D. Affective Loadings (DMT CV1)', fontweight='bold', loc='left')
    ax_d.set_xlim(-1, 1)
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, tet_vals)):
        ax_d.text(val + 0.05 if val > 0 else val - 0.05, i, f'{val:.2f}',
                 va='center', ha='left' if val > 0 else 'right', fontsize=8)
    
    # Save figure
    output_path = output_dir / 'Figure4_CCA_Theil_Analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    
    return output_path


def main():
    """Main function."""
    print("=" * 80)
    print("Generating Figure 4: CCA with Theil BLUS Analysis")
    print("=" * 80)
    
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_df, analyzer = load_data()
    fig_path = create_figure(merged_df, analyzer, output_dir)
    
    print("\nDone!")
    return fig_path


if __name__ == '__main__':
    main()
