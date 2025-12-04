# -*- coding: utf-8 -*-
"""
Test FDR shading implementation in TET time series visualizer.

This script verifies that the FDR correction and shading work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from scripts.tet.time_series_visualizer import TETTimeSeriesVisualizer

def test_fdr_correction():
    """Test Benjamini-Hochberg FDR correction."""
    print("Testing FDR correction...")
    
    # Create dummy visualizer to access BH method
    # Include all required dimensions
    dimensions = ['emotional_intensity_z', 'interoception_z', 'anxiety_z', 
                  'unpleasantness_z', 'pleasantness_z', 'bliss_z', 'valence_index_z']
    
    dummy_data = pd.DataFrame({
        'subject': ['S01'] * 10,
        'state': ['RS'] * 5 + ['DMT'] * 5,
        'dose': ['Alta'] * 5 + ['Baja'] * 5,
        't_bin': list(range(5)) + list(range(5)),
        **{dim: np.random.randn(10) for dim in dimensions}
    })
    
    dummy_lme = pd.DataFrame({
        'dimension': dimensions,
        'effect': ['state[T.DMT]'] * len(dimensions),
        'beta': [0.5] * len(dimensions),
        'significant': [True] * len(dimensions)
    })
    
    dummy_contrasts = pd.DataFrame()
    dummy_tc = pd.DataFrame({
        'dimension': dimensions * 10,
        'state': ['DMT'] * len(dimensions) * 10,
        'dose': ['Alta'] * len(dimensions) * 5 + ['Baja'] * len(dimensions) * 5,
        't_bin': list(range(5)) * len(dimensions) * 2,
        't_sec': [i * 4 for i in range(5)] * len(dimensions) * 2,
        'mean': np.random.randn(len(dimensions) * 10),
        'sem': np.random.rand(len(dimensions) * 10) * 0.1
    })
    
    viz = TETTimeSeriesVisualizer(dummy_data, dummy_lme, dummy_contrasts, dummy_tc, 
                                   dimensions=dimensions)
    
    # Test with known p-values
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.50, 1.0]
    p_fdr = viz._benjamini_hochberg_correction(p_values)
    
    print(f"  Raw p-values: {p_values}")
    print(f"  FDR-adjusted: {[f'{p:.4f}' for p in p_fdr]}")
    
    # Check that FDR adjustment is monotonic
    assert all(p_fdr[i] <= p_fdr[i+1] for i in range(len(p_fdr)-1)), "FDR not monotonic!"
    
    # Check that FDR >= raw p-values
    assert all(p_fdr[i] >= p_values[i] for i in range(len(p_values))), "FDR < raw p-value!"
    
    print("  ✓ FDR correction working correctly")
    print()


def test_dose_interaction_detection():
    """Test dose interaction detection with FDR."""
    print("Testing dose interaction detection with FDR...")
    
    # Load actual data
    data_path = 'results/tet/preprocessed/tet_preprocessed.csv'
    if not os.path.exists(data_path):
        print(f"  ⚠ Data not found: {data_path}")
        print("  Skipping test")
        return
    
    data = pd.read_csv(data_path)
    
    # Load LME results
    lme_path = 'results/tet/lme/lme_results.csv'
    if not os.path.exists(lme_path):
        print(f"  ⚠ LME results not found: {lme_path}")
        print("  Skipping test")
        return
    
    lme_results = pd.read_csv(lme_path)
    
    # Load contrasts
    contrasts_path = 'results/tet/lme/lme_contrasts.csv'
    if not os.path.exists(contrasts_path):
        print(f"  ⚠ Contrasts not found: {contrasts_path}")
        print("  Skipping test")
        return
    
    lme_contrasts = pd.read_csv(contrasts_path)
    
    # Load time courses
    tc_path = 'results/tet/descriptive/time_course_all_dimensions.csv'
    if not os.path.exists(tc_path):
        print(f"  ⚠ Time courses not found: {tc_path}")
        print("  Skipping test")
        return
    
    time_courses = pd.read_csv(tc_path)
    
    # Create visualizer
    viz = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)
    
    # Check dose interaction results
    dose_interactions = viz.dose_interaction_bins
    
    print(f"  Total time bins tested: {len(dose_interactions)}")
    print(f"  Significant bins (p_FDR < 0.05): {dose_interactions['dose_effect_sig'].sum()}")
    
    # Show summary per dimension with contiguous segment grouping
    for dimension in viz.dimensions:
        dim_data = dose_interactions[dose_interactions['dimension'] == dimension]
        n_sig = dim_data['dose_effect_sig'].sum()
        n_total = len(dim_data)
        
        if n_sig > 0:
            min_p_fdr = dim_data[dim_data['dose_effect_sig']]['p_fdr'].min()
            
            # Get significant bins and group consecutive ones
            sig_bins = sorted(dim_data[dim_data['dose_effect_sig']]['t_bin'].values)
            
            # Group consecutive bins
            bin_groups = []
            if len(sig_bins) > 0:
                current_group = [sig_bins[0]]
                for i in range(1, len(sig_bins)):
                    if sig_bins[i] == current_group[-1] + 1:
                        current_group.append(sig_bins[i])
                    else:
                        bin_groups.append(current_group)
                        current_group = [sig_bins[i]]
                bin_groups.append(current_group)
            
            print(f"  {dimension}: {n_sig}/{n_total} bins significant (min p_FDR={min_p_fdr:.4f})")
            print(f"    → {len(bin_groups)} contiguous segment(s)")
            for j, group in enumerate(bin_groups, 1):
                t_start = group[0] * 4 / 60
                t_end = (group[-1] + 1) * 4 / 60
                print(f"       Segment {j}: bins {group[0]}-{group[-1]} ({t_start:.2f}-{t_end:.2f} min)")
    
    print("  ✓ Dose interaction detection complete")
    print()


def test_figure_generation():
    """Test figure generation with FDR shading."""
    print("Testing figure generation with FDR shading...")
    
    # Load actual data
    data_path = 'results/tet/preprocessed/tet_preprocessed.csv'
    if not os.path.exists(data_path):
        print(f"  ⚠ Data not found: {data_path}")
        print("  Skipping test")
        return
    
    data = pd.read_csv(data_path)
    
    # Load LME results
    lme_path = 'results/tet/lme/lme_results.csv'
    if not os.path.exists(lme_path):
        print(f"  ⚠ LME results not found: {lme_path}")
        print("  Skipping test")
        return
    
    lme_results = pd.read_csv(lme_path)
    
    # Load contrasts
    contrasts_path = 'results/tet/lme/lme_contrasts.csv'
    if not os.path.exists(contrasts_path):
        print(f"  ⚠ Contrasts not found: {contrasts_path}")
        print("  Skipping test")
        return
    
    lme_contrasts = pd.read_csv(contrasts_path)
    
    # Load time courses
    tc_path = 'results/tet/descriptive/time_course_all_dimensions.csv'
    if not os.path.exists(tc_path):
        print(f"  ⚠ Time courses not found: {tc_path}")
        print("  Skipping test")
        return
    
    time_courses = pd.read_csv(tc_path)
    
    # Create visualizer
    viz = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)
    
    # Generate test figure
    output_path = 'test/tet/test_fdr_shading_output.png'
    viz.export_figure(output_path, dpi=150, export_fdr_report=True)
    
    print(f"  ✓ Figure saved to: {output_path}")
    
    # Check that FDR report was created
    report_path = output_path.replace('.png', '_fdr_report.txt')
    if os.path.exists(report_path):
        print(f"  ✓ FDR report saved to: {report_path}")
        
        # Show first few lines
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:15]
        print("\n  First lines of FDR report:")
        for line in lines:
            print(f"    {line.rstrip()}")
    else:
        print(f"  ⚠ FDR report not found: {report_path}")
    
    print()


if __name__ == '__main__':
    print("=" * 70)
    print("Testing FDR Shading Implementation")
    print("=" * 70)
    print()
    
    test_fdr_correction()
    test_dose_interaction_detection()
    test_figure_generation()
    
    print("=" * 70)
    print("All tests complete!")
    print("=" * 70)
