# -*- coding: utf-8 -*-
"""
TET State Dose Analyzer Module

This module provides statistical testing functionality for dose effects on
experiential state occupancy metrics. It implements classical paired t-tests,
across-sessions-within-subject permutation tests, and interaction effect evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETStateDoseAnalyzer:
    """
    Statistical testing for dose effects on experiential state occupancy.
    
    This class performs statistical tests to evaluate dose effects (High vs Low)
    on state occupancy metrics derived from clustering/HMM analyses. It implements:
    
    1. Classical paired t-tests for dose comparisons
    2. Across-sessions-within-subject permutation tests
    3. Interaction effect evaluation (Dose × State)
    4. Multiple comparison corrections (BH-FDR)
    
    The across-sessions-within-subject permutation test preserves within-session
    temporal structure by reshuffling entire sessions across permutation iterations,
    making it appropriate for temporal state models like GLHMM.
    
    Interpretation of Interaction Effects (High vs Low × DMT vs RS):
        The interaction effect tests whether dose effects differ between experimental
        states (DMT vs RS). It is computed as:
        
        Interaction = (High - Low | DMT) - (High - Low | RS)
        
        Interpretation:
        - Positive interaction: Dose effect is stronger in DMT than RS. This suggests
          that the drug amplifies dose-dependent differences in state occupancy.
        - Negative interaction: Dose effect is weaker in DMT than RS, or reversed.
          This could indicate ceiling/floor effects or compensatory mechanisms.
        - Non-significant interaction: Dose effects are similar across DMT and RS,
          suggesting dose-dependent differences are not specific to the drug state.
        
        The permutation test provides a non-parametric p-value that accounts for
        the temporal structure of the data by preserving within-session order while
        reshuffling session labels across permutations.
    
    Statistical Approach:
        - Classical t-tests: Parametric tests assuming normality. Fast and powerful
          when assumptions are met. Appropriate for normally distributed metrics.
        
        - Permutation tests: Non-parametric tests that make no distributional
          assumptions. More robust to violations of normality and outliers.
          Preserve temporal structure by reshuffling entire sessions rather than
          individual time points. Computationally intensive but provide exact
          p-values under the null hypothesis of no dose effect.
        
        - FDR correction: Controls the expected proportion of false positives among
          rejected hypotheses. More powerful than Bonferroni correction when testing
          multiple hypotheses. Applied separately to each family of tests.
    
    Attributes:
        state_metrics (pd.DataFrame): State occupancy metrics from clustering
        cluster_probabilities (pd.DataFrame): Cluster/state probabilities
        test_results (pd.DataFrame): Classical t-test results
        permutation_results (pd.DataFrame): Permutation test results
        interaction_results (pd.DataFrame): Interaction effect results
    
    Example:
        >>> import pandas as pd
        >>> from tet.state_dose_analyzer import TETStateDoseAnalyzer
        >>> 
        >>> # Load state metrics
        >>> metrics = pd.read_csv('results/tet/clustering/clustering_state_metrics.csv')
        >>> probs = pd.read_csv('results/tet/clustering/clustering_glhmm_probabilities.csv')
        >>> 
        >>> # Initialize analyzer
        >>> analyzer = TETStateDoseAnalyzer(
        ...     state_metrics=metrics,
        ...     cluster_probabilities=probs
        ... )
        >>> 
        >>> # Perform classical tests
        >>> classical_results = analyzer.compute_pairwise_tests()
        >>> 
        >>> # Perform permutation tests
        >>> perm_results = analyzer.apply_glhmm_permutation_test(n_permutations=1000)
        >>> 
        >>> # Evaluate interaction effects
        >>> interaction_results = analyzer.evaluate_interaction_effects(n_permutations=1000)
        >>> 
        >>> # Apply FDR correction
        >>> analyzer.apply_fdr_correction()
        >>> 
        >>> # Export results
        >>> paths = analyzer.export_results('results/tet/clustering')
    """
    
    def __init__(
        self,
        state_metrics: pd.DataFrame,
        cluster_probabilities: Optional[pd.DataFrame] = None
    ):
        """
        Initialize state dose analyzer with occupancy metrics and probabilities.
        
        Args:
            state_metrics (pd.DataFrame): State occupancy metrics with columns:
                - subject: Subject identifier
                - session_id: Session identifier
                - state: Experimental state (RS or DMT)
                - dose: Dose level (Baja or Alta / Low or High)
                - method: Clustering method ('KMeans' or 'GLHMM')
                - cluster_state: Cluster/state identifier
                - fractional_occupancy: Proportion of time in state
                - n_visits: Number of visits to state
                - mean_dwell_time: Mean consecutive time bins in state
            cluster_probabilities (pd.DataFrame, optional): Cluster/state probabilities
                with columns for subject, session_id, state, dose, t_bin, and
                probability columns (e.g., gamma_state_0, gamma_state_1, ...)
        """
        self.state_metrics = state_metrics.copy()
        self.cluster_probabilities = cluster_probabilities.copy() if cluster_probabilities is not None else None
        
        # Initialize storage for results
        self.test_results = None
        self.permutation_results = None
        self.interaction_results = None
        
        logger.info(f"Initialized TETStateDoseAnalyzer with {len(state_metrics)} metric rows")
        
        if self.cluster_probabilities is not None:
            logger.info(f"  Cluster probabilities: {len(cluster_probabilities)} rows")

    def compute_pairwise_tests(self) -> pd.DataFrame:
        """
        Perform classical paired t-tests for dose effects on state occupancy metrics.
        
        For each state occupancy metric (fractional_occupancy, n_visits, mean_dwell_time)
        and each cluster state, this method:
        1. Pivots data to get paired observations (High vs Low dose per subject)
        2. Removes subjects with missing data
        3. Performs paired t-test
        4. Computes mean difference and 95% confidence interval
        
        The paired t-test is appropriate because each subject experiences both
        dose conditions, allowing within-subject comparisons that control for
        individual differences.
        
        Note: BH-FDR correction is applied separately via apply_fdr_correction().
        
        Returns:
            pd.DataFrame: Test results with columns:
                - metric: Metric name (fractional_occupancy, n_visits, mean_dwell_time)
                - method: Clustering method ('KMeans' or 'GLHMM')
                - cluster_state: Cluster/state identifier
                - n_pairs: Number of paired observations
                - mean_high: Mean value for High dose
                - mean_low: Mean value for Low dose
                - mean_diff: Mean difference (High - Low)
                - t_statistic: t-statistic from paired t-test
                - p_value: p-value from paired t-test
                - ci_lower: Lower bound of 95% CI for mean difference
                - ci_upper: Upper bound of 95% CI for mean difference
        """
        logger.info("Computing classical paired t-tests for dose effects...")
        
        # Metrics to test
        metrics = ['fractional_occupancy', 'n_visits', 'mean_dwell_time']
        
        test_results_list = []
        
        # Get unique methods and cluster states
        methods = self.state_metrics['method'].unique()
        
        for method in methods:
            method_data = self.state_metrics[self.state_metrics['method'] == method]
            cluster_states = method_data['cluster_state'].unique()
            
            for cluster_state in cluster_states:
                cluster_data = method_data[method_data['cluster_state'] == cluster_state]
                
                for metric in metrics:
                    logger.info(f"  Testing {method} cluster {cluster_state}, metric: {metric}")
                    
                    # Pivot to get High vs Low dose per subject
                    # Filter to DMT sessions only for dose comparison
                    dmt_data = cluster_data[cluster_data['state'] == 'DMT']
                    
                    # Create pivot table
                    pivot = dmt_data.pivot_table(
                        index='subject',
                        columns='dose',
                        values=metric,
                        aggfunc='mean'  # Average across sessions if multiple
                    )
                    
                    # Check if we have both dose levels
                    if 'Alta' not in pivot.columns and 'High' not in pivot.columns:
                        logger.warning(f"    No High dose data for {method} cluster {cluster_state}")
                        continue
                    if 'Baja' not in pivot.columns and 'Low' not in pivot.columns:
                        logger.warning(f"    No Low dose data for {method} cluster {cluster_state}")
                        continue
                    
                    # Handle both Spanish and English dose labels
                    high_col = 'Alta' if 'Alta' in pivot.columns else 'High'
                    low_col = 'Baja' if 'Baja' in pivot.columns else 'Low'
                    
                    # Remove subjects with missing data
                    complete_data = pivot[[high_col, low_col]].dropna()
                    
                    n_pairs = len(complete_data)
                    
                    if n_pairs < 3:
                        logger.warning(f"    Insufficient pairs (n={n_pairs}) for {method} cluster {cluster_state}")
                        continue
                    
                    # Extract paired observations
                    high_values = complete_data[high_col].values
                    low_values = complete_data[low_col].values
                    
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(high_values, low_values)
                    
                    # Compute mean difference and CI
                    diff = high_values - low_values
                    mean_diff = np.mean(diff)
                    se_diff = stats.sem(diff)
                    
                    # 95% CI for mean difference
                    ci = stats.t.interval(
                        0.95,
                        df=n_pairs - 1,
                        loc=mean_diff,
                        scale=se_diff
                    )
                    
                    # Store results
                    test_results_list.append({
                        'metric': metric,
                        'method': method,
                        'cluster_state': cluster_state,
                        'n_pairs': n_pairs,
                        'mean_high': np.mean(high_values),
                        'mean_low': np.mean(low_values),
                        'mean_diff': mean_diff,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'ci_lower': ci[0],
                        'ci_upper': ci[1]
                    })
                    
                    logger.info(f"    n={n_pairs}, t={t_stat:.3f}, p={p_value:.4f}, "
                               f"mean_diff={mean_diff:.4f}")
        
        # Create results DataFrame
        self.test_results = pd.DataFrame(test_results_list)
        
        if len(self.test_results) > 0:
            logger.info(f"Completed {len(self.test_results)} paired t-tests")
            
            # Log summary
            n_sig = np.sum(self.test_results['p_value'] < 0.05)
            logger.info(f"  {n_sig} tests significant at p < 0.05 (uncorrected)")
        else:
            logger.warning("No valid t-tests were performed")
        
        return self.test_results

    def apply_glhmm_permutation_test(
        self,
        n_permutations: int = 1000,
        random_seed: int = 22
    ) -> pd.DataFrame:
        """
        Apply across-sessions-within-subject permutation test for dose effects.
        
        This permutation test preserves within-session temporal structure by
        reshuffling entire sessions across permutation iterations. For each subject:
        1. Reshuffle session labels (dose assignments) across permutations
        2. Recompute state occupancy metrics for each permutation
        3. Build null distribution from permuted results
        4. Compute p-values as rank-based comparisons of empirical metrics
        
        The test is applied to:
        - Fractional occupancy
        - Number of visits
        - Mean dwell times
        
        This approach is appropriate for temporal state models (GLHMM) where
        within-session temporal order must be preserved.
        
        Reference:
            notebooks/Testing_across_sessions_within_subject.ipynb
            glhmm.statistics.test_across_sessions_within_subject()
        
        Args:
            n_permutations (int): Number of permutation iterations (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 22)
        
        Returns:
            pd.DataFrame: Permutation test results with columns:
                - metric: Metric name
                - method: Clustering method
                - cluster_state: Cluster/state identifier
                - observed_stat: Observed test statistic (mean difference High - Low)
                - p_value_perm: Permutation-based p-value
                - n_permutations: Number of permutations performed
        """
        logger.info(f"Applying across-sessions-within-subject permutation test "
                   f"({n_permutations} permutations)...")
        
        np.random.seed(random_seed)
        
        # Metrics to test
        metrics = ['fractional_occupancy', 'n_visits', 'mean_dwell_time']
        
        perm_results_list = []
        
        # Get unique methods and cluster states
        methods = self.state_metrics['method'].unique()
        
        for method in methods:
            method_data = self.state_metrics[self.state_metrics['method'] == method]
            cluster_states = method_data['cluster_state'].unique()
            
            for cluster_state in cluster_states:
                cluster_data = method_data[method_data['cluster_state'] == cluster_state]
                
                # Filter to DMT sessions only
                dmt_data = cluster_data[cluster_data['state'] == 'DMT'].copy()
                
                if len(dmt_data) == 0:
                    logger.warning(f"  No DMT data for {method} cluster {cluster_state}")
                    continue
                
                for metric in metrics:
                    logger.info(f"  Permutation test: {method} cluster {cluster_state}, metric: {metric}")
                    
                    # Compute observed statistic (mean difference High - Low)
                    # Handle both Spanish and English dose labels
                    high_data = dmt_data[dmt_data['dose'].isin(['Alta', 'High'])]
                    low_data = dmt_data[dmt_data['dose'].isin(['Baja', 'Low'])]
                    
                    if len(high_data) == 0 or len(low_data) == 0:
                        logger.warning(f"    Missing dose data for {method} cluster {cluster_state}")
                        continue
                    
                    # Compute subject-level means
                    high_means = high_data.groupby('subject')[metric].mean()
                    low_means = low_data.groupby('subject')[metric].mean()
                    
                    # Get common subjects
                    common_subjects = high_means.index.intersection(low_means.index)
                    
                    if len(common_subjects) < 3:
                        logger.warning(f"    Insufficient subjects (n={len(common_subjects)})")
                        continue
                    
                    # Observed statistic: mean difference
                    observed_diff = (high_means[common_subjects] - low_means[common_subjects]).mean()
                    
                    # Permutation test
                    null_distribution = []
                    
                    for perm_idx in range(n_permutations):
                        # For each subject, randomly shuffle dose labels across sessions
                        perm_data = dmt_data.copy()
                        
                        # Group by subject and shuffle dose labels within subject
                        for subject in perm_data['subject'].unique():
                            subject_mask = perm_data['subject'] == subject
                            subject_doses = perm_data.loc[subject_mask, 'dose'].values
                            
                            # Shuffle dose labels
                            np.random.shuffle(subject_doses)
                            perm_data.loc[subject_mask, 'dose'] = subject_doses
                        
                        # Recompute means for permuted data
                        perm_high_data = perm_data[perm_data['dose'].isin(['Alta', 'High'])]
                        perm_low_data = perm_data[perm_data['dose'].isin(['Baja', 'Low'])]
                        
                        perm_high_means = perm_high_data.groupby('subject')[metric].mean()
                        perm_low_means = perm_low_data.groupby('subject')[metric].mean()
                        
                        # Get common subjects for this permutation
                        perm_common = perm_high_means.index.intersection(perm_low_means.index)
                        
                        if len(perm_common) > 0:
                            perm_diff = (perm_high_means[perm_common] - perm_low_means[perm_common]).mean()
                            null_distribution.append(perm_diff)
                    
                    # Compute p-value
                    # Two-tailed test: proportion of permutations with |diff| >= |observed|
                    null_distribution = np.array(null_distribution)
                    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))
                    
                    # Store results
                    perm_results_list.append({
                        'metric': metric,
                        'method': method,
                        'cluster_state': cluster_state,
                        'observed_stat': observed_diff,
                        'p_value_perm': p_value,
                        'n_permutations': len(null_distribution)
                    })
                    
                    logger.info(f"    observed={observed_diff:.4f}, p_perm={p_value:.4f}")
        
        # Create results DataFrame
        self.permutation_results = pd.DataFrame(perm_results_list)
        
        if len(self.permutation_results) > 0:
            logger.info(f"Completed {len(self.permutation_results)} permutation tests")
            
            # Log summary
            n_sig = np.sum(self.permutation_results['p_value_perm'] < 0.05)
            logger.info(f"  {n_sig} tests significant at p < 0.05 (uncorrected)")
        else:
            logger.warning("No valid permutation tests were performed")
        
        return self.permutation_results

    def evaluate_interaction_effects(
        self,
        n_permutations: int = 1000,
        random_seed: int = 22
    ) -> pd.DataFrame:
        """
        Evaluate interaction effects: (High - Low in DMT) - (High - Low in RS).
        
        This method tests whether dose effects differ between DMT and RS states,
        which would indicate a State × Dose interaction. The interaction effect
        is computed as the difference of differences:
        
        Interaction = (High - Low | DMT) - (High - Low | RS)
        
        For each state occupancy metric and cluster state:
        1. Compute High vs Low difference within DMT sessions
        2. Compute High vs Low difference within RS sessions
        3. Compute interaction as difference of differences
        4. Apply across-sessions-within-subject permutation test
        
        The permutation test preserves within-session temporal structure by
        reshuffling entire sessions across permutation iterations.
        
        Reference:
            notebooks/Testing_across_sessions_within_subject.ipynb
            Section on interaction effects and permutation testing
        
        Args:
            n_permutations (int): Number of permutation iterations (default: 1000)
            random_seed (int): Random seed for reproducibility (default: 22)
        
        Returns:
            pd.DataFrame: Interaction effect results with columns:
                - metric: Metric name
                - method: Clustering method
                - cluster_state: Cluster/state identifier
                - interaction_effect: Observed interaction effect
                - dmt_diff: High - Low difference in DMT
                - rs_diff: High - Low difference in RS
                - p_value_perm: Permutation-based p-value
                - n_permutations: Number of permutations performed
        """
        logger.info(f"Evaluating interaction effects (State × Dose) "
                   f"with {n_permutations} permutations...")
        
        np.random.seed(random_seed)
        
        # Metrics to test
        metrics = ['fractional_occupancy', 'n_visits', 'mean_dwell_time']
        
        interaction_results_list = []
        
        # Get unique methods and cluster states
        methods = self.state_metrics['method'].unique()
        
        for method in methods:
            method_data = self.state_metrics[self.state_metrics['method'] == method]
            cluster_states = method_data['cluster_state'].unique()
            
            for cluster_state in cluster_states:
                cluster_data = method_data[method_data['cluster_state'] == cluster_state]
                
                for metric in metrics:
                    logger.info(f"  Interaction test: {method} cluster {cluster_state}, metric: {metric}")
                    
                    # Separate DMT and RS data
                    dmt_data = cluster_data[cluster_data['state'] == 'DMT'].copy()
                    rs_data = cluster_data[cluster_data['state'] == 'RS'].copy()
                    
                    if len(dmt_data) == 0 or len(rs_data) == 0:
                        logger.warning(f"    Missing DMT or RS data")
                        continue
                    
                    # Compute observed interaction effect
                    # DMT: High - Low
                    dmt_high = dmt_data[dmt_data['dose'].isin(['Alta', 'High'])]
                    dmt_low = dmt_data[dmt_data['dose'].isin(['Baja', 'Low'])]
                    
                    dmt_high_means = dmt_high.groupby('subject')[metric].mean()
                    dmt_low_means = dmt_low.groupby('subject')[metric].mean()
                    
                    dmt_common = dmt_high_means.index.intersection(dmt_low_means.index)
                    
                    if len(dmt_common) < 3:
                        logger.warning(f"    Insufficient DMT subjects (n={len(dmt_common)})")
                        continue
                    
                    dmt_diff = (dmt_high_means[dmt_common] - dmt_low_means[dmt_common]).mean()
                    
                    # RS: High - Low
                    rs_high = rs_data[rs_data['dose'].isin(['Alta', 'High'])]
                    rs_low = rs_data[rs_data['dose'].isin(['Baja', 'Low'])]
                    
                    rs_high_means = rs_high.groupby('subject')[metric].mean()
                    rs_low_means = rs_low.groupby('subject')[metric].mean()
                    
                    rs_common = rs_high_means.index.intersection(rs_low_means.index)
                    
                    if len(rs_common) < 3:
                        logger.warning(f"    Insufficient RS subjects (n={len(rs_common)})")
                        continue
                    
                    rs_diff = (rs_high_means[rs_common] - rs_low_means[rs_common]).mean()
                    
                    # Interaction effect
                    observed_interaction = dmt_diff - rs_diff
                    
                    # Permutation test for interaction
                    null_distribution = []
                    
                    for perm_idx in range(n_permutations):
                        # Shuffle dose labels within each subject, separately for DMT and RS
                        perm_dmt = dmt_data.copy()
                        perm_rs = rs_data.copy()
                        
                        # Shuffle DMT dose labels
                        for subject in perm_dmt['subject'].unique():
                            subject_mask = perm_dmt['subject'] == subject
                            subject_doses = perm_dmt.loc[subject_mask, 'dose'].values
                            np.random.shuffle(subject_doses)
                            perm_dmt.loc[subject_mask, 'dose'] = subject_doses
                        
                        # Shuffle RS dose labels
                        for subject in perm_rs['subject'].unique():
                            subject_mask = perm_rs['subject'] == subject
                            subject_doses = perm_rs.loc[subject_mask, 'dose'].values
                            np.random.shuffle(subject_doses)
                            perm_rs.loc[subject_mask, 'dose'] = subject_doses
                        
                        # Recompute DMT difference
                        perm_dmt_high = perm_dmt[perm_dmt['dose'].isin(['Alta', 'High'])]
                        perm_dmt_low = perm_dmt[perm_dmt['dose'].isin(['Baja', 'Low'])]
                        
                        perm_dmt_high_means = perm_dmt_high.groupby('subject')[metric].mean()
                        perm_dmt_low_means = perm_dmt_low.groupby('subject')[metric].mean()
                        
                        perm_dmt_common = perm_dmt_high_means.index.intersection(perm_dmt_low_means.index)
                        
                        if len(perm_dmt_common) > 0:
                            perm_dmt_diff = (perm_dmt_high_means[perm_dmt_common] - 
                                           perm_dmt_low_means[perm_dmt_common]).mean()
                        else:
                            continue
                        
                        # Recompute RS difference
                        perm_rs_high = perm_rs[perm_rs['dose'].isin(['Alta', 'High'])]
                        perm_rs_low = perm_rs[perm_rs['dose'].isin(['Baja', 'Low'])]
                        
                        perm_rs_high_means = perm_rs_high.groupby('subject')[metric].mean()
                        perm_rs_low_means = perm_rs_low.groupby('subject')[metric].mean()
                        
                        perm_rs_common = perm_rs_high_means.index.intersection(perm_rs_low_means.index)
                        
                        if len(perm_rs_common) > 0:
                            perm_rs_diff = (perm_rs_high_means[perm_rs_common] - 
                                          perm_rs_low_means[perm_rs_common]).mean()
                        else:
                            continue
                        
                        # Permuted interaction
                        perm_interaction = perm_dmt_diff - perm_rs_diff
                        null_distribution.append(perm_interaction)
                    
                    # Compute p-value
                    null_distribution = np.array(null_distribution)
                    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_interaction))
                    
                    # Store results
                    interaction_results_list.append({
                        'metric': metric,
                        'method': method,
                        'cluster_state': cluster_state,
                        'interaction_effect': observed_interaction,
                        'dmt_diff': dmt_diff,
                        'rs_diff': rs_diff,
                        'p_value_perm': p_value,
                        'n_permutations': len(null_distribution)
                    })
                    
                    logger.info(f"    interaction={observed_interaction:.4f}, "
                               f"p_perm={p_value:.4f}")
        
        # Create results DataFrame
        self.interaction_results = pd.DataFrame(interaction_results_list)
        
        if len(self.interaction_results) > 0:
            logger.info(f"Completed {len(self.interaction_results)} interaction tests")
            
            # Log summary
            n_sig = np.sum(self.interaction_results['p_value_perm'] < 0.05)
            logger.info(f"  {n_sig} tests significant at p < 0.05 (uncorrected)")
        else:
            logger.warning("No valid interaction tests were performed")
        
        return self.interaction_results

    def apply_fdr_correction(self, alpha: float = 0.05) -> None:
        """
        Apply Benjamini-Hochberg FDR correction to all test results.
        
        This method applies BH-FDR correction separately to:
        1. Classical t-test p-values
        2. Permutation test p-values
        3. Interaction effect p-values
        
        The correction controls the false discovery rate (expected proportion
        of false positives among rejected hypotheses) at the specified alpha level.
        
        Results DataFrames are updated in-place with p_fdr columns.
        
        Args:
            alpha (float): FDR significance level (default: 0.05)
        """
        logger.info(f"Applying BH-FDR correction (alpha={alpha})...")
        
        # Apply FDR to classical t-tests
        if self.test_results is not None and len(self.test_results) > 0:
            logger.info("  Correcting classical t-test p-values...")
            
            p_values = self.test_results['p_value'].values
            reject, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
            
            self.test_results['p_fdr'] = p_fdr
            self.test_results['significant'] = reject
            
            n_sig = np.sum(reject)
            logger.info(f"    {n_sig}/{len(p_values)} tests significant at FDR < {alpha}")
        
        # Apply FDR to permutation tests
        if self.permutation_results is not None and len(self.permutation_results) > 0:
            logger.info("  Correcting permutation test p-values...")
            
            p_values = self.permutation_results['p_value_perm'].values
            reject, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
            
            self.permutation_results['p_fdr'] = p_fdr
            self.permutation_results['significant'] = reject
            
            n_sig = np.sum(reject)
            logger.info(f"    {n_sig}/{len(p_values)} tests significant at FDR < {alpha}")
        
        # Apply FDR to interaction tests
        if self.interaction_results is not None and len(self.interaction_results) > 0:
            logger.info("  Correcting interaction test p-values...")
            
            p_values = self.interaction_results['p_value_perm'].values
            reject, p_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
            
            self.interaction_results['p_fdr'] = p_fdr
            self.interaction_results['significant'] = reject
            
            n_sig = np.sum(reject)
            logger.info(f"    {n_sig}/{len(p_values)} tests significant at FDR < {alpha}")
        
        logger.info("FDR correction complete")

    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export statistical test results to CSV files.
        
        Creates output directory if it doesn't exist and exports:
        - clustering_dose_tests_classical.csv: Classical paired t-test results
        - clustering_dose_tests_permutation.csv: Permutation test results
        - clustering_interaction_effects.csv: Interaction effect results
        
        Args:
            output_dir (str): Directory to save output files
        
        Returns:
            Dict[str, str]: Dictionary mapping file types to file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {}
        
        # Export classical t-test results
        if self.test_results is not None and len(self.test_results) > 0:
            classical_path = os.path.join(output_dir, 'clustering_dose_tests_classical.csv')
            self.test_results.to_csv(classical_path, index=False)
            output_paths['classical_tests'] = classical_path
            logger.info(f"Exported classical test results to: {classical_path}")
        else:
            logger.warning("No classical test results to export")
        
        # Export permutation test results
        if self.permutation_results is not None and len(self.permutation_results) > 0:
            perm_path = os.path.join(output_dir, 'clustering_dose_tests_permutation.csv')
            self.permutation_results.to_csv(perm_path, index=False)
            output_paths['permutation_tests'] = perm_path
            logger.info(f"Exported permutation test results to: {perm_path}")
        else:
            logger.warning("No permutation test results to export")
        
        # Export interaction effect results
        if self.interaction_results is not None and len(self.interaction_results) > 0:
            interaction_path = os.path.join(output_dir, 'clustering_interaction_effects.csv')
            self.interaction_results.to_csv(interaction_path, index=False)
            output_paths['interaction_effects'] = interaction_path
            logger.info(f"Exported interaction effect results to: {interaction_path}")
        else:
            logger.warning("No interaction effect results to export")
        
        if len(output_paths) > 0:
            logger.info(f"Exported {len(output_paths)} result files to: {output_dir}")
        else:
            logger.warning("No results were exported")
        
        return output_paths
