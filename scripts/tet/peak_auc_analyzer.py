# -*- coding: utf-8 -*-
"""
TET Peak and AUC Analyzer Module

This module provides functionality for computing peak values, time-to-peak, and
area under the curve (AUC) metrics for TET dimensions, and performing statistical
comparisons between dose conditions using non-parametric tests.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
from scipy.stats import wilcoxon
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETPeakAUCAnalyzer:
    """
    Analyzes peak values and area under curve (AUC) for TET dimensions.
    
    This class computes metrics (peak, time_to_peak, AUC) for each session-dimension
    combination and performs statistical comparisons between High and Low dose
    conditions using Wilcoxon signed-rank tests with bootstrap confidence intervals.
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data with z-scored dimensions
        dimensions (List[str]): List of z-scored dimension column names
        metrics_df (pd.DataFrame): Computed metrics for all sessions
        results_df (pd.DataFrame): Statistical test results
    
    Example:
        >>> import pandas as pd
        >>> from tet.peak_auc_analyzer import TETPeakAUCAnalyzer
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/tet_preprocessed.csv')
        >>> 
        >>> # Define z-scored dimensions
        >>> dimensions = [col for col in data.columns if col.endswith('_z')]
        >>> 
        >>> # Initialize analyzer
        >>> analyzer = TETPeakAUCAnalyzer(data, dimensions)
        >>> 
        >>> # Compute metrics
        >>> metrics = analyzer.compute_metrics()
        >>> 
        >>> # Perform statistical tests
        >>> results = analyzer.perform_tests()
        >>> 
        >>> # Export results
        >>> paths = analyzer.export_results('results/tet/peak_auc')
    """
    
    def __init__(self, data: pd.DataFrame, dimensions: List[str]):
        """
        Initialize analyzer with preprocessed TET data.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data with columns for subject,
                session_id, state, dose, t_bin, t_sec, and z-scored dimensions
            dimensions (List[str]): List of z-scored dimension column names
                (e.g., ['pleasantness_z', 'anxiety_z', ...])
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.metrics_df = None
        self.results_df = None
        
        logger.info(f"Initialized TETPeakAUCAnalyzer with {len(data)} rows, "
                   f"{data['subject'].nunique()} subjects, "
                   f"{len(dimensions)} dimensions")

    def compute_metrics(self) -> pd.DataFrame:
        """
        Compute peak, time_to_peak, and AUC metrics for all DMT sessions.
        
        For each session-dimension combination in DMT sessions (0-9 minutes):
        - Peak value: Maximum z-score
        - Time to peak: Time in minutes when peak occurs
        - AUC 0-9: Area under curve using trapezoidal integration
        
        AUC Interpretation with Z-scores:
            Since dimensions are z-scored within subjects, AUC represents the
            cumulative deviation from the subject's personal baseline across all
            dimensions and sessions. Positive AUC indicates sustained elevation
            above baseline, while negative AUC indicates sustained suppression.
            The magnitude reflects both intensity and duration of the experience.
        
        Returns:
            pd.DataFrame: Metrics with columns:
                - subject: Subject identifier
                - session: Session identifier (e.g., "DMT_Baja_1")
                - dose: Dose level ("Baja" or "Alta")
                - dimension: Dimension name
                - peak: Maximum z-score value
                - time_to_peak_min: Time to peak in minutes
                - auc_0_9: Area under curve (0-9 minutes)
        """
        logger.info("Computing peak and AUC metrics...")
        
        # Filter to DMT sessions only
        dmt_data = self.data[self.data['state'] == 'DMT'].copy()
        
        # Filter to first 9 minutes (t_bin <= 18 at 30s resolution)
        # Note: t_bin is 0-indexed, so t_bin=18 corresponds to 9 minutes
        dmt_data = dmt_data[dmt_data['t_bin'] <= 18].copy()
        
        logger.info(f"Filtered to {len(dmt_data)} DMT observations (0-9 minutes)")
        
        # Initialize list to store metrics
        metrics_list = []
        
        # Group by subject, session_id, dose
        session_groups = dmt_data.groupby(['subject', 'session_id', 'dose'])
        
        for (subject, session_id, dose), session_data in session_groups:
            # Process each dimension
            for dimension in self.dimensions:
                # Get dimension values and time
                z_values = session_data[dimension].values
                t_sec = session_data['t_sec'].values
                t_min = t_sec / 60  # Convert to minutes
                
                # Compute peak value
                peak = np.max(z_values)
                
                # Compute time to peak (in minutes)
                peak_idx = np.argmax(z_values)
                time_to_peak_min = t_min[peak_idx]
                
                # Compute AUC using trapezoidal integration
                auc_0_9 = trapezoid(z_values, t_min)
                
                # Store metrics
                metrics_list.append({
                    'subject': subject,
                    'session': session_id,
                    'dose': dose,
                    'dimension': dimension,
                    'peak': peak,
                    'time_to_peak_min': time_to_peak_min,
                    'auc_0_9': auc_0_9
                })
        
        # Create DataFrame
        self.metrics_df = pd.DataFrame(metrics_list)
        
        # Log summary statistics
        n_sessions = self.metrics_df.groupby(['subject', 'session']).ngroups
        n_dimensions = len(self.dimensions)
        n_metrics = len(self.metrics_df)
        
        logger.info(f"Computed metrics: {n_metrics} rows "
                   f"({n_sessions} sessions × {n_dimensions} dimensions)")
        logger.info(f"  Peak range: [{self.metrics_df['peak'].min():.2f}, "
                   f"{self.metrics_df['peak'].max():.2f}]")
        logger.info(f"  AUC range: [{self.metrics_df['auc_0_9'].min():.2f}, "
                   f"{self.metrics_df['auc_0_9'].max():.2f}]")
        
        return self.metrics_df

    def _compute_effect_size_r(
        self, 
        high_dose: np.ndarray, 
        low_dose: np.ndarray, 
        n_bootstrap: int = 2000
    ) -> Tuple[float, float, float]:
        """
        Compute effect size r with 95% bootstrap confidence intervals.
        
        Effect size r is computed as r = Z / sqrt(N), where Z is the z-score
        from the Wilcoxon signed-rank test and N is the number of paired observations.
        
        Effect Size Interpretation:
            - r ≈ 0.1: Small effect
            - r ≈ 0.3: Medium effect
            - r ≈ 0.5: Large effect
            
            The sign of r indicates the direction of the effect (positive = High > Low).
            Bootstrap confidence intervals provide uncertainty estimates that account
            for sampling variability in the paired observations.
        
        Bootstrap confidence intervals are computed by resampling paired observations
        with replacement and computing the effect size for each bootstrap sample.
        
        Args:
            high_dose (np.ndarray): High dose values (paired observations)
            low_dose (np.ndarray): Low dose values (paired observations)
            n_bootstrap (int): Number of bootstrap iterations (default: 2000)
        
        Returns:
            Tuple[float, float, float]: (effect_r, ci_lower, ci_upper)
                - effect_r: Observed effect size
                - ci_lower: 2.5th percentile of bootstrap distribution
                - ci_upper: 97.5th percentile of bootstrap distribution
        """
        try:
            # Compute observed effect size
            statistic, p_value = wilcoxon(high_dose, low_dose)
            
            # Convert Wilcoxon statistic to z-score
            # For large N, Wilcoxon statistic is approximately normal
            n = len(high_dose)
            mean_w = n * (n + 1) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            z_score = (statistic - mean_w) / std_w
            
            # Compute effect size r
            effect_r = z_score / np.sqrt(n)
            
            # Bootstrap confidence intervals
            bootstrap_effects = []
            
            for _ in range(n_bootstrap):
                # Resample paired observations with replacement
                indices = np.random.choice(n, size=n, replace=True)
                high_boot = high_dose[indices]
                low_boot = low_dose[indices]
                
                try:
                    # Compute Wilcoxon test for bootstrap sample
                    stat_boot, _ = wilcoxon(high_boot, low_boot)
                    
                    # Convert to z-score and effect size
                    z_boot = (stat_boot - mean_w) / std_w
                    r_boot = z_boot / np.sqrt(n)
                    
                    bootstrap_effects.append(r_boot)
                except:
                    # Skip failed bootstrap samples
                    continue
            
            # Compute 95% CI from bootstrap distribution
            if len(bootstrap_effects) > 0:
                ci_lower = np.percentile(bootstrap_effects, 2.5)
                ci_upper = np.percentile(bootstrap_effects, 97.5)
            else:
                # If all bootstrap samples failed, return NaN
                ci_lower = np.nan
                ci_upper = np.nan
            
            return effect_r, ci_lower, ci_upper
            
        except Exception as e:
            logger.warning(f"Effect size computation failed: {e}")
            return np.nan, np.nan, np.nan

    def perform_tests(self) -> pd.DataFrame:
        """
        Perform Wilcoxon signed-rank tests comparing High vs Low dose.
        
        For each metric type (peak, time_to_peak_min, auc_0_9) and each dimension:
        - Pivot data to get paired observations (High vs Low dose per subject)
        - Remove subjects with missing data
        - Skip if n < 3 pairs
        - Perform Wilcoxon signed-rank test
        - Compute effect size with bootstrap CI
        - Apply BH-FDR correction separately for each metric type
        
        Returns:
            pd.DataFrame: Test results with columns:
                - dimension: Dimension name
                - metric: Metric type ("peak", "time_to_peak_min", "auc_0_9")
                - n_pairs: Number of paired observations
                - statistic: Wilcoxon test statistic
                - p_value: Raw p-value
                - p_fdr: FDR-corrected p-value
                - effect_r: Effect size r
                - ci_lower: 95% CI lower bound
                - ci_upper: 95% CI upper bound
                - significant: Significance after FDR correction (p_fdr < 0.05)
        """
        if self.metrics_df is None:
            raise ValueError("Must call compute_metrics() before perform_tests()")
        
        logger.info("Performing statistical tests...")
        
        # Metric types to test
        metric_types = ['peak', 'time_to_peak_min', 'auc_0_9']
        
        # Initialize list to store results
        results_list = []
        
        # Process each metric type
        for metric_type in metric_types:
            logger.info(f"Testing metric: {metric_type}")
            
            # Store p-values for FDR correction
            p_values_for_fdr = []
            test_results_temp = []
            
            # Process each dimension
            for dimension in self.dimensions:
                # Filter to current dimension
                dim_data = self.metrics_df[
                    self.metrics_df['dimension'] == dimension
                ].copy()
                
                # Pivot to get High vs Low dose per subject
                # Each subject should have 2 DMT sessions (1 High, 1 Low)
                pivot_data = dim_data.pivot_table(
                    index='subject',
                    columns='dose',
                    values=metric_type,
                    aggfunc='mean'  # Average if multiple sessions per dose
                )
                
                # Check if both Alta and Baja columns exist
                if 'Alta' not in pivot_data.columns or 'Baja' not in pivot_data.columns:
                    logger.warning(f"Skipping {dimension} - {metric_type}: "
                                 f"missing dose condition")
                    continue
                
                # Remove subjects with missing data
                complete_data = pivot_data.dropna()
                
                # Get paired arrays
                high_dose = complete_data['Alta'].values
                low_dose = complete_data['Baja'].values
                n_pairs = len(high_dose)
                
                # Skip if insufficient data
                if n_pairs < 3:
                    logger.warning(f"Skipping {dimension} - {metric_type}: "
                                 f"n={n_pairs} < 3 pairs")
                    continue
                
                try:
                    # Perform Wilcoxon signed-rank test
                    statistic, p_value = wilcoxon(high_dose, low_dose)
                    
                    # Compute effect size with bootstrap CI
                    effect_r, ci_lower, ci_upper = self._compute_effect_size_r(
                        high_dose, low_dose, n_bootstrap=2000
                    )
                    
                    # Store results
                    test_results_temp.append({
                        'dimension': dimension,
                        'metric': metric_type,
                        'n_pairs': n_pairs,
                        'statistic': statistic,
                        'p_value': p_value,
                        'effect_r': effect_r,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper
                    })
                    
                    p_values_for_fdr.append(p_value)
                    
                except Exception as e:
                    logger.warning(f"Test failed for {dimension} - {metric_type}: {e}")
                    continue
            
            # Apply BH-FDR correction for this metric type
            if len(p_values_for_fdr) > 0:
                reject, p_fdr, _, _ = multipletests(
                    p_values_for_fdr, 
                    method='fdr_bh'
                )
                
                # Add FDR-corrected p-values and significance
                for i, result in enumerate(test_results_temp):
                    result['p_fdr'] = p_fdr[i]
                    result['significant'] = reject[i]
                    results_list.append(result)
                
                # Log summary
                n_significant = np.sum(reject)
                logger.info(f"  {metric_type}: {n_significant}/{len(p_values_for_fdr)} "
                          f"significant after FDR correction")
            else:
                logger.warning(f"No valid tests for {metric_type}")
        
        # Create results DataFrame
        self.results_df = pd.DataFrame(results_list)
        
        # Log overall summary
        if len(self.results_df) > 0:
            total_tests = len(self.results_df)
            total_significant = self.results_df['significant'].sum()
            logger.info(f"Total tests: {total_tests}, "
                       f"significant: {total_significant} "
                       f"({100*total_significant/total_tests:.1f}%)")
        else:
            logger.warning("No valid test results")
        
        return self.results_df

    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export metrics and test results to CSV files.
        
        Creates output directory if it doesn't exist and exports:
        - peak_auc_metrics.csv: Raw metrics for all sessions
        - peak_auc_tests.csv: Statistical test results
        
        Args:
            output_dir (str): Directory to save output files
        
        Returns:
            Dict[str, str]: Dictionary mapping file types to file paths
                - 'metrics': Path to metrics CSV
                - 'tests': Path to test results CSV
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {}
        
        # Export metrics
        if self.metrics_df is not None:
            metrics_path = os.path.join(output_dir, 'peak_auc_metrics.csv')
            self.metrics_df.to_csv(metrics_path, index=False)
            output_paths['metrics'] = metrics_path
            logger.info(f"Exported metrics to: {metrics_path}")
        else:
            logger.warning("No metrics to export (call compute_metrics() first)")
        
        # Export test results
        if self.results_df is not None:
            tests_path = os.path.join(output_dir, 'peak_auc_tests.csv')
            self.results_df.to_csv(tests_path, index=False)
            output_paths['tests'] = tests_path
            logger.info(f"Exported test results to: {tests_path}")
        else:
            logger.warning("No test results to export (call perform_tests() first)")
        
        return output_paths
