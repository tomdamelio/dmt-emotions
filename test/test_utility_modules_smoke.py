"""
Smoke tests for utility modules created in tasks 1-7.

These tests verify that all utility modules can be imported and basic
functionality works without errors.
"""

import sys
import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, '.')

from scripts import (
    alternative_statistics,
    baseline_comparator,
    phase_analyzer,
    feature_extractor,
    enhanced_visualizer,
    statistical_reporter
)


class TestAlternativeStatistics:
    """Smoke tests for alternative_statistics module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert alternative_statistics is not None
    
    def test_compute_pointwise_uncorrected_basic(self):
        """Test basic functionality of compute_pointwise_uncorrected."""
        # Create simple test data
        np.random.seed(42)
        data_high = np.random.randn(10, 18)
        data_low = np.random.randn(10, 18)
        
        result = alternative_statistics.compute_pointwise_uncorrected(
            data_high, data_low
        )
        
        # Verify result structure
        assert 't_stats' in result
        assert 'p_values' in result
        assert 'significant_mask' in result
        assert 'significant_segments' in result
        assert len(result['t_stats']) == 18
        assert len(result['p_values']) == 18
    
    def test_compute_one_tailed_tests_basic(self):
        """Test basic functionality of compute_one_tailed_tests."""
        np.random.seed(42)
        data_high = np.random.randn(10, 18) + 1.0  # Shift high dose up
        data_low = np.random.randn(10, 18)
        
        result = alternative_statistics.compute_one_tailed_tests(
            data_high, data_low, alternative='greater'
        )
        
        # Verify result structure
        assert 't_stats' in result
        assert 'p_values' in result
        assert 'significant_mask' in result
        assert 'significant_segments' in result


class TestBaselineComparator:
    """Smoke tests for baseline_comparator module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert baseline_comparator is not None
    
    def test_compute_baseline_summary_stats_basic(self):
        """Test basic functionality of compute_baseline_summary_stats."""
        # Create simple test data with subject column
        df = pd.DataFrame({
            'subject': ['sub-01'] * 3 + ['sub-02'] * 2,
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = baseline_comparator.compute_baseline_summary_stats(df)
        
        # Verify result structure
        assert 'mean' in result
        assert 'std' in result
        assert 'median' in result
        assert 'q25' in result
        assert 'q75' in result
        assert 'n_samples' in result
        assert 'n_subjects' in result
        assert result['n_samples'] == 5
        assert result['n_subjects'] == 2


class TestPhaseAnalyzer:
    """Smoke tests for phase_analyzer module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert phase_analyzer is not None
    
    def test_define_temporal_phases_basic(self):
        """Test basic functionality of define_temporal_phases."""
        phases = phase_analyzer.define_temporal_phases(
            total_duration_sec=540,
            phase_boundaries=[0, 180, 540]
        )
        
        # Verify result structure
        assert len(phases) == 2
        assert phases[0] == (0, 180)
        assert phases[1] == (180, 540)


class TestFeatureExtractor:
    """Smoke tests for feature_extractor module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert feature_extractor is not None
    
    def test_extract_peak_amplitude_basic(self):
        """Test basic functionality of extract_peak_amplitude."""
        # Create simple test signal with known peak
        t = np.linspace(0, 540, 100)
        signal = np.sin(t / 100) + 2.0
        
        result = feature_extractor.extract_peak_amplitude(t, signal)
        
        # Verify result structure
        assert 'peak_amplitude' in result
        assert 'peak_time_sec' in result
        assert isinstance(result['peak_amplitude'], (int, float))
        assert isinstance(result['peak_time_sec'], (int, float))


class TestEnhancedVisualizer:
    """Smoke tests for enhanced_visualizer module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert enhanced_visualizer is not None
    
    def test_module_has_required_functions(self):
        """Test that module has required functions."""
        assert hasattr(enhanced_visualizer, 'add_significance_markers')
        assert hasattr(enhanced_visualizer, 'apply_homogeneous_aesthetics')
        assert hasattr(enhanced_visualizer, 'save_figure_vector')


class TestStatisticalReporter:
    """Smoke tests for statistical_reporter module."""
    
    def test_import(self):
        """Test that module imports successfully."""
        assert statistical_reporter is not None
    
    def test_format_ttest_result_basic(self):
        """Test basic functionality of format_ttest_result."""
        result = statistical_reporter.format_ttest_result(
            t_stat=2.5,
            df=17,
            p_value=0.023,
            cohens_d=0.6
        )
        
        # Verify result is a string
        assert isinstance(result, str)
        assert 't(' in result
        assert 'p' in result
        assert 'd' in result
    
    def test_format_lme_result_basic(self):
        """Test basic functionality of format_lme_result."""
        result = statistical_reporter.format_lme_result(
            beta=0.5,
            ci_lower=0.2,
            ci_upper=0.8,
            p_fdr=0.01,
            parameter_name='State[T.DMT]'
        )
        
        # Verify result is a string
        assert isinstance(result, str)
        assert 'β' in result or 'beta' in result.lower()
        assert 'CI' in result
        assert 'p' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
