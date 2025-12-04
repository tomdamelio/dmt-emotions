#!/usr/bin/env python3
"""
Deep investigation into why Theil method produces different results for RS vs DMT.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.cross_decomposition import CCA
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def investigate_permutation_distributions(analyzer, output_dir):
    """Compare the null distributions for RS vs DMT."""
    
    print("\n" + "=" * 80)
    print("PERMUTATION NULL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    for state in ['RS', 'DMT']:
        print(f"\n{state} State:")
        print("-" * 40)
        
        # Get data
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
        
        # Compute observed statistic
        cca = CCA(n_components=3)
        cca.fit(X_blus, Y_blus)
        U, V = cca.transform(X_blus, Y_blus)
        observed_r = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        
        # Compute Wilks' Lambda
        r_all = [np.corrcoef(U[:, k], V[:, k])[0, 1] for k in range(3)]
        observed_wilks = np.prod([1 - r**2 for r in r_all])
        
        print(f"  Observed r (Mode 1): {observed_r:.4f}")
        print(f"  Observed Wilks' Lambda: {observed_wilks:.4f}")
        
        # Generate permutation distribution (sample of 200 for speed)
        n_perm = 200
        unique_subjects = np.unique(blus_ids)
        n_subjects = len(unique_subjects)
        
        perm_wilks = []
        perm_r = []
        
        all_derangements = list(analyzer._generate_all_derangements(n_subjects))
        sample_derangements = all_derangements[:n_perm]
        
        for derangement in sample_derangements:
            X_perm, Y_perm = analyzer._apply_derangement(
                X_blus, Y_blus, blus_ids, derangement
            )
            
            cca_perm = CCA(n_components=3)
            cca_perm.fit(X_perm, Y_perm)
            U_perm, V_perm = cca_perm.transform(X_perm, Y_perm)
            
            r_perm = [np.corrcoef(U_perm[:, k], V_perm[:, k])[0, 1] for k in range(3)]
            wilks_perm = np.prod([1 - r**2 for r in r_perm])
            
            perm_wilks.append(wilks_perm)
            perm_r.append(r_perm[0])
        
        perm_wilks = np.array(perm_wilks)
        perm_r = np.array(perm_r)
        
        # Statistics of null distribution
        print(f"\n  Null distribution (Wilks' Lambda):")
        print(f"    Mean: {perm_wilks.mean():.4f}")
        print(f"    SD:   {perm_wilks.std():.4f}")
        print(f"    Min:  {perm_wilks.min():.4f}")
        print(f"    Max:  {perm_wilks.max():.4f}")
        
        # Where does observed fall?
        percentile = (perm_wilks <= observed_wilks).mean() * 100
        print(f"    Observed percentile: {percentile:.1f}%")
        
        print(f"\n  Null distribution (r Mode 1):")
        print(f"    Mean: {perm_r.mean():.4f}")
        print(f"    SD:   {perm_r.std():.4f}")
        print(f"    Min:  {perm_r.min():.4f}")
        print(f"    Max:  {perm_r.max():.4f}")
        
        # Key insight: how much of the null distribution exceeds observed?
        pct_exceeds = (perm_r >= observed_r).mean() * 100
        print(f"    % permutations with r >= observed: {pct_exceeds:.1f}%")


def investigate_subject_means(merged_df):
    """Investigate the subject means that Theil removes."""
    
    print("\n" + "=" * 80)
    print("SUBJECT MEANS ANALYSIS (What Theil Removes)")
    print("=" * 80)
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    tet_vars = ['emotional_intensity_z']
    
    for state in ['RS', 'DMT']:
        print(f"\n{state} State:")
        print("-" * 40)
        
        state_data = merged_df[merged_df['state'] == state]
        
        # Subject means
        subj_means = state_data.groupby('subject')[physio_vars + tet_vars].mean()
        
        print("\nSubject means (what Theil removes):")
        print(subj_means.round(3).to_string())
        
        # Correlation of subject means
        print("\nCorrelation of subject means (physio vs emotional_intensity):")
        for var in physio_vars:
            r, p = stats.pearsonr(subj_means[var], subj_means['emotional_intensity_z'])
            print(f"  {var}: r={r:.3f}, p={p:.3f}")
        
        # Variance of subject means
        print("\nVariance of subject means:")
        for var in physio_vars + tet_vars:
            print(f"  {var}: {subj_means[var].var():.4f}")


def investigate_within_subject_structure(merged_df):
    """Investigate the within-subject structure that remains after Theil."""
    
    print("\n" + "=" * 80)
    print("WITHIN-SUBJECT STRUCTURE (What Remains After Theil)")
    print("=" * 80)
    
    physio_vars = ['HR', 'SMNA_AUC', 'RVT']
    
    for state in ['RS', 'DMT']:
        print(f"\n{state} State:")
        print("-" * 40)
        
        state_data = merged_df[merged_df['state'] == state].copy()
        
        # Demean within subject (what Theil approximately does)
        for var in physio_vars + ['emotional_intensity_z']:
            state_data[f'{var}_demeaned'] = state_data.groupby('subject')[var].transform(
                lambda x: x - x.mean()
            )
        
        # Correlation of demeaned variables
        print("\nCorrelation of demeaned variables (within-subject):")
        for var in physio_vars:
            r, p = stats.pearsonr(
                state_data[f'{var}_demeaned'],
                state_data['emotional_intensity_z_demeaned']
            )
            print(f"  {var}: r={r:.3f}, p={p:.3f}")
        
        # Per-subject correlations of demeaned variables
        print("\nPer-subject correlations (demeaned):")
        for subject in state_data['subject'].unique():
            subj_data = state_data[state_data['subject'] == subject]
            r, _ = stats.pearsonr(
                subj_data['HR_demeaned'],
                subj_data['emotional_intensity_z_demeaned']
            )
            print(f"  Subject {subject}: HR-EmotInt r={r:.3f}")


def main():
    print("=" * 80)
    print("DEEP INVESTIGATION: Why Theil Produces Different Results")
    print("=" * 80)
    
    output_dir = Path('results/tet/physio_correlation/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = TETPhysioDataLoader(
        composite_physio_path='results/composite/arousal_index_long.csv',
        tet_path='results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec=30
    )
    loader.load_physiological_data()
    loader.load_tet_data()
    merged_df = loader.merge_datasets()
    
    # Initialize analyzer
    analyzer = TETPhysioCCAAnalyzer(merged_df)
    analyzer.fit_cca(n_components=3)
    
    # Run investigations
    investigate_subject_means(merged_df)
    investigate_within_subject_structure(merged_df)
    investigate_permutation_distributions(analyzer, output_dir)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)


if __name__ == '__main__':
    main()
