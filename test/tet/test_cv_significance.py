#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for CV significance testing implementation.

This script verifies that the compute_cv_significance() method works correctly
with synthetic data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def create_synthetic_data(n_subjects=7, n_obs_per_subject=50):
    """
    Create synthetic physiological-TET data for testing.
    
    Args:
        n_subjects: Number of subjects
        n_obs_per_subject: Observations per subject
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    
    data_list = []
    
    for subject_id in range(1, n_subjects + 1):
        for obs_id in range(n_obs_per_subject):
            # Create correlated physiological and TET data
            # Simulate a moderate correlation (r ~ 0.5)
            physio_base = np.random.randn(3)
            tet_base = 0.5 * physio_base[:, np.newaxis] + 0.5 * np.random.randn(3, 6)
            
            data_list.append({
                'subject': f'S{subject_id:02d}',
                'session_id': 1,
                'state': 'DMT',
                'dose': 'High',
                't_bin': obs_id + 1,
                'window': obs_id + 1,
                'HR': physio_base[0],
                'SMNA_AUC': physio_base[1],
                'RVT': physio_base[2],
                'pleasantness_z': tet_base[0, 0],
                'unpleasantness_z': tet_base[0, 1],
                'emotional_intensity_z': tet_base[0, 2],
                'interoception_z': tet_base[0, 3],
                'bliss_z': tet_base[0, 4],
                'anxiety_z': tet_base[0, 5]
            })
    
    return pd.DataFrame(data_list)


def test_cv_significance():
    """Test the compute_cv_significance() method."""
    print("=" * 80)
    print("Testing CV Significance Implementation")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    data = create_synthetic_data(n_subjects=7, n_obs_per_subject=50)
    print(f"   Created data: {len(data)} observations, {data['subject'].nunique()} subjects")
    
    # Initialize analyzer
    print("\n2. Initializing CCA analyzer...")
    analyzer = TETPhysioCCAAnalyzer(data)
    
    # Fit CCA
    print("\n3. Fitting CCA...")
    analyzer.fit_cca(n_components=2)
    print("   CCA fitted successfully")
    
    # Run LOSO cross-validation
    print("\n4. Running LOSO cross-validation...")
    cv_results = analyzer.loso_cross_validation('DMT', n_components=2)
    print(f"   CV completed: {len(cv_results)} fold results")
    
    # Compute CV significance
    print("\n5. Computing CV significance...")
    try:
        cv_significance = analyzer.compute_cv_significance()
        print("   ✓ CV significance computed successfully")
        
        # Display results
        print("\n" + "=" * 80)
        print("CV SIGNIFICANCE RESULTS")
        print("=" * 80)
        print(cv_significance.to_string(index=False))
        
        # Verify expected columns
        expected_cols = [
            'state', 'canonical_variate', 'n_folds', 'mean_r_oos', 'sd_r_oos',
            't_statistic', 'p_value_t_test', 'p_value_wilcoxon', 
            'success_rate', 'n_positive_folds', 'significant', 'interpretation'
        ]
        
        missing_cols = set(expected_cols) - set(cv_significance.columns)
        if missing_cols:
            print(f"\n❌ ERROR: Missing columns: {missing_cols}")
            return False
        
        print("\n✓ All expected columns present")
        
        # Verify data types
        assert cv_significance['n_folds'].dtype in [np.int64, np.int32], "n_folds should be integer"
        assert cv_significance['significant'].dtype == bool, "significant should be boolean"
        assert cv_significance['interpretation'].dtype == object, "interpretation should be string"
        
        print("✓ Data types correct")
        
        # Verify statistical values are reasonable
        for _, row in cv_significance.iterrows():
            assert 0 <= row['p_value_t_test'] <= 1, "p-value should be in [0, 1]"
            assert 0 <= row['success_rate'] <= 1, "success_rate should be in [0, 1]"
            assert row['n_positive_folds'] <= row['n_folds'], "n_positive_folds <= n_folds"
        
        print("✓ Statistical values are reasonable")
        
        print("\n" + "=" * 80)
        print("TEST PASSED: CV significance implementation works correctly!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_cv_significance()
    sys.exit(0 if success else 1)
