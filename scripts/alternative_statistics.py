# -*- coding: utf-8 -*-
"""
Alternative Statistical Testing Module for DMT Physiological Analysis.

This module provides less conservative statistical criteria for time-to-time comparisons
in physiological signals (HR, SMNA, RVT). It implements:
  1. Pointwise uncorrected paired t-tests (no FDR correction)
  2. One-tailed paired t-tests for directional hypotheses (High > Low)

Scientific Rationale:
- FDR correction is conservative and may obscure dose-dependent temporal dynamics
- One-tailed tests are appropriate when directional hypotheses are scientifically justified
- One-tailed tests should be tried first (more principled than removing correction)
- Uncorrected tests provide exploratory insights but require careful interpretation

Usage:
    from scripts.alternative_statistics import (
        compute_pointwise_uncorrected,
        compute_one_tailed_tests
    )
    
    # One-tailed test (preferred for directional hypotheses)
    results_one_tailed = compute_one_tailed_tests(
        data_high, data_low, alternative='greater'
    )
    
    # Uncorrected test (use only if one-tailed fails)
    results_uncorrected = compute_pointwise_uncorrected(data_high, data_low)

Author: DMT Analysis Pipeline
Date: 2026-01-10
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


def compute_pointwise_uncorrected(
    data_high: np.ndarray,
    data_low: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Compute pointwise paired t-tests without FDR correction.
    
    This function performs paired t-tests at each timepoint without applying
    multiple comparison correction. Results should be clearly labeled as
    "uncorrected p < .05" to distinguish from FDR-corrected findings.
    
    Scientific Context:
    - Use only when one-tailed tests fail to show expected effects
    - Provides exploratory insights into temporal dynamics
    - Requires careful interpretation due to inflated Type I error rate
    - Should be accompanied by effect size estimates
    
    Args:
        data_high: Array of shape (n_subjects, n_timepoints) for High dose condition
        data_low: Array of shape (n_subjects, n_timepoints) for Low dose condition
        alpha: Significance threshold (default 0.05)
    
    Returns:
        Dictionary with keys:
            - 't_stats': t-statistics at each timepoint (n_timepoints,)
            - 'p_values': Uncorrected p-values at each timepoint (n_timepoints,)
            - 'significant_mask': Boolean mask of significant timepoints (n_timepoints,)
            - 'significant_segments': List of (start_idx, end_idx) tuples for 
                                     contiguous significant segments
            - 'test_type': String identifier 'uncorrected'
            - 'alpha': Significance threshold used
            - 'n_subjects': Number of subjects
            - 'n_timepoints': Number of timepoints
    
    Raises:
        ValueError: If input arrays have incompatible shapes or insufficient data
    
    Example:
        >>> data_high = np.random.randn(10, 18)  # 10 subjects, 18 timepoints
        >>> data_low = np.random.randn(10, 18)
        >>> results = compute_pointwise_uncorrected(data_high, data_low)
        >>> print(f"Significant timepoints: {np.sum(results['significant_mask'])}")
    
    References:
        - Requirements 1.1, 1.4, 1.5
    """
    # Input validation
    if data_high.shape != data_low.shape:
        raise ValueError(
            f"Input arrays must have same shape. "
            f"Got data_high: {data_high.shape}, data_low: {data_low.shape}"
        )
    
    if data_high.ndim != 2:
        raise ValueError(
            f"Input arrays must be 2D (n_subjects, n_timepoints). "
            f"Got {data_high.ndim}D"
        )
    
    n_subjects, n_timepoints = data_high.shape
    
    if n_subjects < 2:
        raise ValueError(
            f"Need at least 2 subjects for paired t-test. Got {n_subjects}"
        )
    
    # Initialize output arrays
    t_stats = np.zeros(n_timepoints)
    p_values = np.zeros(n_timepoints)
    
    # Compute paired t-test at each timepoint
    for timepoint_idx in range(n_timepoints):
        high_values = data_high[:, timepoint_idx]
        low_values = data_low[:, timepoint_idx]
        
        # Check for NaN values
        valid_mask = ~(np.isnan(high_values) | np.isnan(low_values))
        
        if np.sum(valid_mask) < 2:
            # Not enough valid data for this timepoint
            t_stats[timepoint_idx] = np.nan
            p_values[timepoint_idx] = np.nan
            continue
        
        # Perform two-tailed paired t-test
        t_stat, p_value = stats.ttest_rel(
            high_values[valid_mask],
            low_values[valid_mask],
            alternative='two-sided'
        )
        
        t_stats[timepoint_idx] = t_stat
        p_values[timepoint_idx] = p_value
    
    # Create significance mask
    significant_mask = p_values < alpha
    
    # Find contiguous significant segments
    significant_segments = _find_contiguous_segments(significant_mask)
    
    return {
        't_stats': t_stats,
        'p_values': p_values,
        'significant_mask': significant_mask,
        'significant_segments': significant_segments,
        'test_type': 'uncorrected',
        'alpha': alpha,
        'n_subjects': n_subjects,
        'n_timepoints': n_timepoints
    }


def compute_one_tailed_tests(
    data_high: np.ndarray,
    data_low: np.ndarray,
    alternative: str = 'greater',
    alpha: float = 0.05,
    apply_fdr: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compute one-tailed paired t-tests for directional hypotheses with optional FDR correction.
    
    This function performs one-tailed paired t-tests at each timepoint to test
    directional hypotheses (e.g., High > Low for arousal measures). One-tailed
    tests have greater statistical power than two-tailed tests when the direction
    of effect is scientifically justified.
    
    IMPORTANT: By default, applies FDR correction to control for multiple comparisons
    across timepoints. This is the recommended approach as it combines the power
    of one-tailed tests with proper Type I error control.
    
    Scientific Context:
    - Preferred over uncorrected tests when directional hypothesis exists
    - Appropriate for SMNA and RVT where High dose expected to increase arousal
    - Effectively halves p-values compared to two-tailed tests
    - FDR correction controls false discovery rate across multiple timepoints
    - Should be tried before considering uncorrected tests
    
    Args:
        data_high: Array of shape (n_subjects, n_timepoints) for High dose condition
        data_low: Array of shape (n_subjects, n_timepoints) for Low dose condition
        alternative: Direction of hypothesis:
            - 'greater': Test if High > Low (default, for arousal measures)
            - 'less': Test if High < Low (for inhibitory measures)
        alpha: Significance threshold (default 0.05)
        apply_fdr: Whether to apply FDR correction (default True, recommended)
    
    Returns:
        Dictionary with keys:
            - 't_stats': t-statistics at each timepoint (n_timepoints,)
            - 'p_values': One-tailed p-values at each timepoint (n_timepoints,)
            - 'p_values_fdr': FDR-corrected p-values (if apply_fdr=True)
            - 'significant_mask': Boolean mask based on FDR-corrected p-values
            - 'significant_segments': List of (start_idx, end_idx) tuples for 
                                     contiguous significant segments
            - 'test_type': String identifier 'one_tailed_greater_fdr' or similar
            - 'alternative': Direction of hypothesis tested
            - 'alpha': Significance threshold used
            - 'fdr_corrected': Boolean indicating if FDR was applied
            - 'n_subjects': Number of subjects
            - 'n_timepoints': Number of timepoints
    
    Raises:
        ValueError: If input arrays have incompatible shapes, insufficient data,
                   or invalid alternative hypothesis
    
    Example:
        >>> data_high = np.random.randn(10, 18) + 0.5  # Higher mean
        >>> data_low = np.random.randn(10, 18)
        >>> results = compute_one_tailed_tests(
        ...     data_high, data_low, alternative='greater', apply_fdr=True
        ... )
        >>> print(f"Directional hypothesis: High > Low (FDR-corrected)")
        >>> print(f"Significant timepoints: {np.sum(results['significant_mask'])}")
    
    References:
        - Requirements 1.2, 1.3, 1.5
    """
    # Input validation
    if data_high.shape != data_low.shape:
        raise ValueError(
            f"Input arrays must have same shape. "
            f"Got data_high: {data_high.shape}, data_low: {data_low.shape}"
        )
    
    if data_high.ndim != 2:
        raise ValueError(
            f"Input arrays must be 2D (n_subjects, n_timepoints). "
            f"Got {data_high.ndim}D"
        )
    
    if alternative not in ['greater', 'less']:
        raise ValueError(
            f"alternative must be 'greater' or 'less'. Got '{alternative}'"
        )
    
    n_subjects, n_timepoints = data_high.shape
    
    if n_subjects < 2:
        raise ValueError(
            f"Need at least 2 subjects for paired t-test. Got {n_subjects}"
        )
    
    # Initialize output arrays
    t_stats = np.zeros(n_timepoints)
    p_values = np.zeros(n_timepoints)
    
    # Compute one-tailed paired t-test at each timepoint
    for timepoint_idx in range(n_timepoints):
        high_values = data_high[:, timepoint_idx]
        low_values = data_low[:, timepoint_idx]
        
        # Check for NaN values
        valid_mask = ~(np.isnan(high_values) | np.isnan(low_values))
        
        if np.sum(valid_mask) < 2:
            # Not enough valid data for this timepoint
            t_stats[timepoint_idx] = np.nan
            p_values[timepoint_idx] = np.nan
            continue
        
        # Perform one-tailed paired t-test
        t_stat, p_value = stats.ttest_rel(
            high_values[valid_mask],
            low_values[valid_mask],
            alternative=alternative
        )
        
        t_stats[timepoint_idx] = t_stat
        p_values[timepoint_idx] = p_value
    
    # Apply FDR correction if requested
    if apply_fdr:
        # Get valid (non-NaN) p-values
        valid_pvals = ~np.isnan(p_values)
        
        if np.sum(valid_pvals) > 0:
            # Apply Benjamini-Hochberg FDR correction
            p_values_fdr = np.full_like(p_values, np.nan)
            valid_indices = np.where(valid_pvals)[0]
            
            # Extract valid p-values and apply correction
            pvals_to_correct = p_values[valid_pvals].tolist()
            corrected = _benjamini_hochberg_correction(pvals_to_correct)
            
            # Put corrected values back
            p_values_fdr[valid_indices] = corrected
            
            # Create significance mask based on FDR-corrected p-values
            significant_mask = p_values_fdr < alpha
        else:
            p_values_fdr = np.full_like(p_values, np.nan)
            significant_mask = np.zeros(n_timepoints, dtype=bool)
        
        test_type = f'one_tailed_{alternative}_fdr'
    else:
        # No FDR correction - use raw p-values
        p_values_fdr = None
        significant_mask = p_values < alpha
        test_type = f'one_tailed_{alternative}'
    
    # Find contiguous significant segments
    significant_segments = _find_contiguous_segments(significant_mask)
    
    result = {
        't_stats': t_stats,
        'p_values': p_values,
        'significant_mask': significant_mask,
        'significant_segments': significant_segments,
        'test_type': test_type,
        'alternative': alternative,
        'alpha': alpha,
        'fdr_corrected': apply_fdr,
        'n_subjects': n_subjects,
        'n_timepoints': n_timepoints
    }
    
    # Add FDR-corrected p-values if computed
    if apply_fdr:
        result['p_values_fdr'] = p_values_fdr
    
    return result


def _benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.
    
    This function implements the Benjamini-Hochberg procedure for controlling
    the False Discovery Rate (FDR) when performing multiple hypothesis tests.
    
    Algorithm:
    1. Sort p-values in ascending order
    2. For each p-value at rank i (1-indexed), compute adjusted p-value as:
       p_adj[i] = min(p[i] * n / i, p_adj[i+1])
    3. Work backwards to ensure monotonicity
    
    Args:
        p_values: List of raw p-values from multiple tests
    
    Returns:
        List of FDR-corrected p-values in the same order as input
    
    Example:
        >>> p_vals = [0.01, 0.04, 0.03, 0.005]
        >>> p_adj = _benjamini_hochberg_correction(p_vals)
        >>> print(p_adj)  # [0.0133, 0.04, 0.04, 0.02]
    
    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society: Series B, 57(1), 289-300.
    """
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    
    # Get sorted indices
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    # Initialize adjusted p-values array
    adjusted_p = np.zeros(n, dtype=float)
    
    # Work backwards from largest to smallest p-value
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            # Largest p-value: adjusted = raw * n / rank
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            # Ensure monotonicity: take minimum of current adjustment and next
            adjusted_p[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted_p[sorted_indices[i + 1]]
            )
    
    # Ensure no adjusted p-value exceeds 1.0
    return np.minimum(adjusted_p, 1.0).tolist()


def _find_contiguous_segments(
    mask: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Find contiguous segments of True values in a boolean mask.
    
    Helper function to identify continuous runs of significant timepoints.
    
    Args:
        mask: Boolean array indicating significant timepoints
    
    Returns:
        List of (start_idx, end_idx) tuples for each contiguous segment.
        end_idx is inclusive (Python convention would be exclusive, but we
        use inclusive for clarity in reporting time segments).
    
    Example:
        >>> mask = np.array([False, True, True, False, True, False])
        >>> segments = _find_contiguous_segments(mask)
        >>> print(segments)  # [(1, 2), (4, 4)]
    """
    if len(mask) == 0:
        return []
    
    segments = []
    in_segment = False
    start_idx = None
    
    for idx, is_significant in enumerate(mask):
        if is_significant and not in_segment:
            # Start of new segment
            start_idx = idx
            in_segment = True
        elif not is_significant and in_segment:
            # End of current segment
            segments.append((start_idx, idx - 1))
            in_segment = False
    
    # Handle case where mask ends with True
    if in_segment:
        segments.append((start_idx, len(mask) - 1))
    
    return segments


def format_alternative_results(
    results: Dict[str, np.ndarray],
    time_array: Optional[np.ndarray] = None,
    time_unit: str = 'minutes'
) -> str:
    """
    Format alternative statistics results for reporting.
    
    Creates a human-readable summary of uncorrected or one-tailed test results,
    including significant segments and their timing.
    
    Args:
        results: Dictionary from compute_pointwise_uncorrected() or 
                compute_one_tailed_tests()
        time_array: Optional array of time values for each timepoint.
                   If None, uses timepoint indices.
        time_unit: Unit of time for reporting (default 'minutes')
    
    Returns:
        Formatted string summarizing the results
    
    Example:
        >>> results = compute_one_tailed_tests(data_high, data_low)
        >>> time_array = np.arange(0, 9, 0.5)  # 0-9 minutes in 30-sec bins
        >>> report = format_alternative_results(results, time_array)
        >>> print(report)
    """
    test_type = results['test_type']
    alpha = results['alpha']
    n_significant = np.sum(results['significant_mask'])
    n_timepoints = results['n_timepoints']
    
    # Create header
    if test_type == 'uncorrected':
        header = f"Uncorrected Pointwise Paired t-tests (α = {alpha})"
        warning = "⚠️  Results are UNCORRECTED for multiple comparisons"
    elif 'one_tailed' in test_type:
        direction = results['alternative']
        header = f"One-Tailed Paired t-tests: High {'>' if direction == 'greater' else '<'} Low (α = {alpha})"
        warning = f"Directional hypothesis: {direction}"
    else:
        header = f"Alternative Statistical Test (α = {alpha})"
        warning = ""
    
    lines = [
        "=" * 70,
        header,
        "=" * 70,
        warning,
        "",
        f"Number of subjects: {results['n_subjects']}",
        f"Number of timepoints: {n_timepoints}",
        f"Significant timepoints: {n_significant} ({100 * n_significant / n_timepoints:.1f}%)",
        ""
    ]
    
    # Report significant segments
    if len(results['significant_segments']) > 0:
        lines.append("Significant Segments:")
        lines.append("-" * 70)
        
        for seg_idx, (start_idx, end_idx) in enumerate(results['significant_segments'], 1):
            if time_array is not None:
                start_time = time_array[start_idx]
                end_time = time_array[end_idx]
                lines.append(
                    f"  Segment {seg_idx}: {start_time:.2f}-{end_time:.2f} {time_unit} "
                    f"(indices {start_idx}-{end_idx})"
                )
            else:
                lines.append(
                    f"  Segment {seg_idx}: indices {start_idx}-{end_idx}"
                )
    else:
        lines.append("No significant segments found.")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
