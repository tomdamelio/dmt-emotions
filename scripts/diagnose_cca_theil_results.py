#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic analysis to understand the unexpected CCA results with Theil method.

Key questions:
1. Why is RS significant but DMT not with Theil BLUS?
2. What is the variance structure (within vs between subjects)?
3. How does Theil transformation affect the data?

Author: TET Analysis Pipeline
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def compute_icc(data: pd.DataFrame, value_col: str, subject_col: str) -> dict:
    """
    Compute Intraclass Correlation Coefficient (ICC) using one-way random effects.
    
    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    
    Where:
    - MSB = Mean Square Between subjects
    - MSW = Mean Square Within subjects
    - k = average number of observations per subject
    
    Returns:
        dict with ICC, between-subject variance, within-subject variance
    """
    groups = data.groupby(subject_col)[value_col]
    
    # Number of subjects
    n_subjects = groups.ngroups
    
    # Total observations
    n_total = len(data)
    
    # Average observations per subject
    k = n_total / n_subjects
    
    # Grand mean
    grand_mean = data[value_col].mean()
    
    # Between-subjects sum of squares
    ss_between = sum(
        len(group) * (group.mean() - grand_mean)**2 
        for _, group in groups
    )
    
    # Within-subjects sum of squares
    ss_within = sum(
        ((group - group.mean())**2).sum() 
        for _, group in groups
    )
    
    # Degrees of freedom
    df_between = n_subjects - 1
    df_within = n_total - n_subjects
    
    # Mean squares
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    # ICC
    if ms_between + (k - 1) * ms_within > 0:
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    else:
        icc = 0
    
    # Variance components
    var_between = (ms_between - ms_within) / k if ms_between > ms_within else 0
    var_within = ms_within
    var_total = var_between + var_within
    
    return {
        'icc': icc,
        'var_between': var_between,
        'var_within': var_within,
        'var_total': var_total,
        'pct_between': var_between / var_total * 100 if var_total > 0 else 0,
        'pct_within': var_within / var_total * 100 if var_total > 0 else 0,
        'ms_between': ms_between,
        'ms_within': ms_within
    }


def analyze_variance_structure(merged_df: pd.DataFrame, output_dir: Path):
    """Analyze within vs between subject variance for each variable."""
    
    print("\n" + "=" * 80)
    print("VARIANCE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    tet_vars = [
        'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
        'interoception_z', 'bliss_z', 'anxiety_z'
    ]
    
    results = []
    
    for state in ['RS', 'DMT']:
        state_data = merged_df[merged_df['state'] == state].copy()
        
        print(f"\n{state} State:")
        print("-" * 40)
        
        # Physiological variables
        print("\nPhysiological Variables:")
        for var in physio_vars:
            if var in state_data.columns:
                icc_result = compute_icc(state_data, var, 'subject')
                print(f"  {var}: ICC={icc_result['icc']:.3f}, "
                      f"Between={icc_result['pct_between']:.1f}%, "
                      f"Within={icc_result['pct_within']:.1f}%")
                results.append({
                    'state': state,
                    'variable': var,
                    'type': 'physio',
                    **icc_result
                })
        
        # TET variables
        print("\nTET Affective Variables:")
        for var in tet_vars:
            if var in state_data.columns:
                icc_result = compute_icc(state_data, var, 'subject')
                clean_name = var.replace('_z', '')
                print(f"  {clean_name}: ICC={icc_result['icc']:.3f}, "
                      f"Between={icc_result['pct_between']:.1f}%, "
                      f"Within={icc_result['pct_within']:.1f}%")
                results.append({
                    'state': state,
                    'variable': clean_name,
                    'type': 'tet',
                    **icc_result
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'variance_structure_analysis.csv', index=False)
    
    return results_df


def analyze_theil_effect(analyzer: TETPhysioCCAAnalyzer, output_dir: Path):
    """Analyze how Theil transformation affects the data."""
    
    print("\n" + "=" * 80)
    print("THEIL TRANSFORMATION EFFECT")
    print("=" * 80)
    
    results = []
    
    for state in ['RS', 'DMT']:
        print(f"\n{state} State:")
        print("-" * 40)
        
        # Get original data
        X_orig, Y_orig = analyzer.prepare_matrices(state)
        
        # Get subject IDs
        state_data = analyzer.data[analyzer.data['state'] == state].copy()
        subject_ids_full = state_data['subject'].values
        X_full = state_data[analyzer.physio_measures].values
        Y_full = state_data[analyzer.tet_affective].values
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
        subject_ids = subject_ids_full[valid_mask]
        
        # Get Theil BLUS residuals
        Y_blus, X_blus, blus_subject_ids = analyzer._compute_theil_residuals_for_blocks(
            Y_orig, X_orig, subject_ids
        )
        
        print(f"  Original: {X_orig.shape[0]} observations")
        print(f"  After Theil: {X_blus.shape[0]} observations")
        
        # Compare variance
        print(f"\n  Variance comparison (X - Physio):")
        for i, var in enumerate(['HR', 'SMNA_AUC', 'RVT']):
            var_orig = np.var(X_orig[:, i])
            var_blus = np.var(X_blus[:, i])
            print(f"    {var}: Original={var_orig:.3f}, BLUS={var_blus:.3f}, "
                  f"Ratio={var_blus/var_orig:.3f}")
        
        print(f"\n  Variance comparison (Y - TET):")
        tet_names = ['pleasant', 'unpleasant', 'emot_int', 'interocep', 'bliss', 'anxiety']
        for i, var in enumerate(tet_names):
            var_orig = np.var(Y_orig[:, i])
            var_blus = np.var(Y_blus[:, i])
            print(f"    {var}: Original={var_orig:.3f}, BLUS={var_blus:.3f}, "
                  f"Ratio={var_blus/var_orig:.3f}")
        
        # Compute CCA on both
        from sklearn.cross_decomposition import CCA
        
        cca_orig = CCA(n_components=3)
        cca_orig.fit(X_orig, Y_orig)
        U_orig, V_orig = cca_orig.transform(X_orig, Y_orig)
        r_orig = [np.corrcoef(U_orig[:, k], V_orig[:, k])[0, 1] for k in range(3)]
        
        cca_blus = CCA(n_components=3)
        cca_blus.fit(X_blus, Y_blus)
        U_blus, V_blus = cca_blus.transform(X_blus, Y_blus)
        r_blus = [np.corrcoef(U_blus[:, k], V_blus[:, k])[0, 1] for k in range(3)]
        
        print(f"\n  Canonical correlations:")
        print(f"    Original: {[f'{r:.3f}' for r in r_orig]}")
        print(f"    BLUS:     {[f'{r:.3f}' for r in r_blus]}")
        
        results.append({
            'state': state,
            'n_orig': X_orig.shape[0],
            'n_blus': X_blus.shape[0],
            'r1_orig': r_orig[0],
            'r1_blus': r_blus[0],
            'r2_orig': r_orig[1],
            'r2_blus': r_blus[1],
            'r3_orig': r_orig[2],
            'r3_blus': r_blus[2]
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'theil_effect_analysis.csv', index=False)
    
    return results_df


def analyze_subject_variability(merged_df: pd.DataFrame, output_dir: Path):
    """Analyze subject-level variability in each state."""
    
    print("\n" + "=" * 80)
    print("SUBJECT-LEVEL VARIABILITY ANALYSIS")
    print("=" * 80)
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    
    for state in ['RS', 'DMT']:
        state_data = merged_df[merged_df['state'] == state].copy()
        
        print(f"\n{state} State:")
        print("-" * 40)
        
        # Subject means and SDs
        subject_stats = state_data.groupby('subject')[physio_vars].agg(['mean', 'std'])
        
        print("\nSubject means (physiological):")
        for var in physio_vars:
            means = subject_stats[(var, 'mean')].values
            print(f"  {var}: range=[{means.min():.2f}, {means.max():.2f}], "
                  f"SD of means={means.std():.3f}")
        
        print("\nSubject SDs (within-subject variability):")
        for var in physio_vars:
            sds = subject_stats[(var, 'std')].values
            print(f"  {var}: mean SD={sds.mean():.3f}, range=[{sds.min():.3f}, {sds.max():.3f}]")


def plot_diagnostic_figures(merged_df: pd.DataFrame, variance_df: pd.DataFrame, 
                           output_dir: Path):
    """Generate diagnostic visualizations."""
    
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTIC FIGURES")
    print("=" * 80)
    
    # Figure 1: ICC comparison between states
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, var_type in enumerate(['physio', 'tet']):
        ax = axes[idx]
        type_data = variance_df[variance_df['type'] == var_type]
        
        # Pivot for grouped bar chart
        pivot_data = type_data.pivot(index='variable', columns='state', values='icc')
        
        x = np.arange(len(pivot_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pivot_data['RS'], width, label='RS', color='steelblue')
        bars2 = ax.bar(x + width/2, pivot_data['DMT'], width, label='DMT', color='coral')
        
        ax.set_ylabel('ICC (Intraclass Correlation)')
        ax.set_title(f'{"Physiological" if var_type == "physio" else "TET Affective"} Variables')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
        ax.legend()
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='ICC=0.5')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Variance Structure: Between-Subject vs Within-Subject\n'
                 '(Higher ICC = More between-subject variance)', fontsize=12)
    plt.tight_layout()
    
    fig_path = output_dir / 'diagnostic_icc_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")
    
    # Figure 2: Subject trajectories
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    
    for row, state in enumerate(['RS', 'DMT']):
        state_data = merged_df[merged_df['state'] == state].copy()
        
        for col, var in enumerate(physio_vars):
            ax = axes[row, col]
            
            for subject in state_data['subject'].unique():
                subj_data = state_data[state_data['subject'] == subject]
                ax.plot(subj_data['window'], subj_data[var], 
                       alpha=0.7, linewidth=1, label=subject)
            
            ax.set_xlabel('Time Window (30s bins)')
            ax.set_ylabel(var)
            ax.set_title(f'{state} - {var}')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Subject Trajectories: Physiological Variables by State', fontsize=12)
    plt.tight_layout()
    
    fig_path = output_dir / 'diagnostic_subject_trajectories.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    """Run diagnostic analysis."""
    
    print("=" * 80)
    print("DIAGNOSTIC ANALYSIS: Understanding CCA Theil Results")
    print("=" * 80)
    
    # Output directory
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
    
    print(f"  Loaded {len(merged_df)} observations")
    print(f"  Subjects: {merged_df['subject'].nunique()}")
    
    # Initialize analyzer
    analyzer = TETPhysioCCAAnalyzer(merged_df)
    analyzer.fit_cca(n_components=3)
    
    # Run analyses
    variance_df = analyze_variance_structure(merged_df, output_dir)
    theil_df = analyze_theil_effect(analyzer, output_dir)
    analyze_subject_variability(merged_df, output_dir)
    plot_diagnostic_figures(merged_df, variance_df, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nKey findings:")
    
    # Compare ICC between states
    physio_icc_rs = variance_df[(variance_df['state'] == 'RS') & 
                                (variance_df['type'] == 'physio')]['icc'].mean()
    physio_icc_dmt = variance_df[(variance_df['state'] == 'DMT') & 
                                 (variance_df['type'] == 'physio')]['icc'].mean()
    
    print(f"\n1. Average ICC (physiological variables):")
    print(f"   RS:  {physio_icc_rs:.3f} ({physio_icc_rs*100:.1f}% between-subject variance)")
    print(f"   DMT: {physio_icc_dmt:.3f} ({physio_icc_dmt*100:.1f}% between-subject variance)")
    
    if physio_icc_dmt > physio_icc_rs:
        print(f"   → DMT has MORE between-subject variance than RS")
        print(f"   → Theil removes more variance in DMT, leaving less signal")
    else:
        print(f"   → RS has MORE between-subject variance than DMT")
    
    # Compare canonical correlations
    print(f"\n2. Canonical correlations (Mode 1):")
    for _, row in theil_df.iterrows():
        print(f"   {row['state']}: Original r={row['r1_orig']:.3f}, "
              f"BLUS r={row['r1_blus']:.3f}")
    
    print(f"\n3. Interpretation:")
    print(f"   - RS shows HIGHER r after Theil ({theil_df[theil_df['state']=='RS']['r1_blus'].values[0]:.3f})")
    print(f"     → Strong within-subject coupling exists")
    print(f"   - DMT shows similar r after Theil ({theil_df[theil_df['state']=='DMT']['r1_blus'].values[0]:.3f})")
    print(f"     → But permutation test is not significant")
    print(f"     → Suggests high variability in coupling patterns across subjects")
    
    print(f"\nDiagnostic files saved to: {output_dir}")
    
    return variance_df, theil_df


if __name__ == '__main__':
    main()
