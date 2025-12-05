# -*- coding: utf-8 -*-
"""
TET Physiological-Affective Correlation Analyzer

This module implements correlation analysis between physiological signals
(HR, SMNA AUC, RVT) and affective TET dimensions to test hypotheses about
autonomic correlates of subjective emotional experiences.

Classes:
    TETPhysioCorrelationAnalyzer: Compute correlations, regressions, and hypothesis tests
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TETPhysioCorrelationAnalyzer:
    """
    Analyzer for computing correlations between physiological signals and TET affective dimensions.
    
    This class implements:
    - Pearson correlations between TET and physiological measures
    - FDR correction for multiple comparisons
    - PCA on physiological signals to extract PC1 (arousal index)
    - Regression analysis predicting TET from physiological PC1
    - Steiger's Z-test comparing arousal vs valence coupling
    
    Attributes:
        data (pd.DataFrame): Merged physiological-TET dataset
        physio_measures (list): List of physiological measure names
        tet_affective (list): List of TET affective dimension names
        correlation_results (pd.DataFrame): Correlation analysis results
        regression_results (pd.DataFrame): Regression analysis results
        arousal_valence_hypothesis (pd.DataFrame): Hypothesis test results
        pca_loadings (pd.DataFrame): PCA loadings for documentation
    
    Example:
        >>> analyzer = TETPhysioCorrelationAnalyzer(merged_data)
        >>> corr_results = analyzer.compute_correlations(by_state=True)
        >>> pca_loadings = analyzer.load_pca_loadings()
        >>> reg_results = analyzer.regression_analysis(by_state=True)
        >>> hypothesis_results = analyzer.test_arousal_valence_hypothesis()
        >>> analyzer.export_results('results/tet/physio_correlation')
    """
    
    def __init__(self, merged_data: pd.DataFrame):
        """
        Initialize correlation analyzer.
        
        Args:
            merged_data: Merged physiological-TET dataset with columns:
                - subject, session_id, state, dose, t_bin, t_sec
                - HR_z, SMNA_AUC_z, RVT_z (z-scored physiological measures)
                - *_z (z-scored TET dimensions)
                - ArousalIndex (PC1 from physiological PCA, already computed)
        
        Note:
            All data is already z-scored within subject. No additional
            standardization is needed for correlation analysis.
            
            ArousalIndex (PC1) is already computed and available in merged_data.
            No need to recompute PCA - use existing ArousalIndex column.
        """
        self.data = merged_data.copy()
        
        # Physiological measures (already z-scored)
        self.physio_measures = ['HR_z', 'SMNA_AUC_z', 'RVT_z']
        
        # TET affective dimensions (with _z suffix for z-scored versions)
        self.tet_affective = [dim + '_z' for dim in config.TET_AFFECTIVE_COLUMNS]
        
        # Validate required columns
        required_cols = (
            ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec', 'ArousalIndex'] +
            self.physio_measures + self.tet_affective
        )
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add computed indices
        if 'valence_index_z' not in self.data.columns:
            self.data['valence_index_z'] = (
                self.data['pleasantness_z'] - self.data['unpleasantness_z']
            )
        
        # Initialize result storage
        self.correlation_results = None
        self.regression_results = None
        self.arousal_valence_hypothesis = None
        self.pca_loadings = None
        
        logger.info(f"Initialized TETPhysioCorrelationAnalyzer with {len(self.data)} observations")
        logger.info(f"Physiological measures: {self.physio_measures}")
        logger.info(f"TET affective dimensions: {self.tet_affective}")
    
    def compute_correlations(self, by_state: bool = True) -> pd.DataFrame:
        """
        Compute Pearson correlations between TET and physiological measures.
        
        Analysis families (for FDR correction):
        1. Arousal-Physiology: emotional_intensity_z vs (HR_z, SMNA_AUC_z, RVT_z) = 3 tests
        2. Valence-Physiology: valence_index_z vs (HR_z, SMNA_AUC_z, RVT_z) = 3 tests
        3. All Affective-Physiology: 6 TET dims × 3 physio measures = 18 tests
        
        Args:
            by_state: If True, compute correlations separately for RS and DMT states
        
        Returns:
            DataFrame with columns:
            - tet_dimension: TET dimension name
            - physio_measure: Physiological measure name
            - state: RS or DMT (if by_state=True)
            - r: Pearson correlation coefficient
            - p_value: Two-tailed p-value
            - p_fdr: FDR-corrected p-value
            - ci_lower, ci_upper: 95% confidence interval
            - n_obs: Number of observations
        
        Note:
            FDR correction is applied within each analysis family separately.
        """
        logger.info("Computing correlations between TET and physiological measures...")
        
        results = []
        states = self.data['state'].unique() if by_state else [None]
        
        for state in states:
            # Filter data by state
            if state is not None:
                state_data = self.data[self.data['state'] == state].copy()
                state_label = state
            else:
                state_data = self.data.copy()
                state_label = 'All'
            
            # Compute correlations for each TET-physio pair
            # Include all affective dimensions plus valence_index_z (emotional_intensity_z already in tet_affective)
            tet_dims_to_test = self.tet_affective + ['valence_index_z']
            for tet_dim in tet_dims_to_test:
                for physio_measure in self.physio_measures:
                    # Extract valid pairs (drop NaN)
                    valid_mask = (
                        state_data[tet_dim].notna() & 
                        state_data[physio_measure].notna()
                    )
                    x = state_data.loc[valid_mask, tet_dim].values
                    y = state_data.loc[valid_mask, physio_measure].values
                    
                    n_obs = len(x)
                    
                    if n_obs < 30:
                        logger.warning(
                            f"Low sample size for {tet_dim} vs {physio_measure} "
                            f"in state {state_label}: N={n_obs}"
                        )
                    
                    if n_obs >= 3:
                        # Compute Pearson correlation
                        r, p_value = stats.pearsonr(x, y)
                        
                        # Compute 95% CI using Fisher z-transform
                        z = np.arctanh(r)
                        se_z = 1 / np.sqrt(n_obs - 3)
                        ci_z_lower = z - 1.96 * se_z
                        ci_z_upper = z + 1.96 * se_z
                        ci_lower = np.tanh(ci_z_lower)
                        ci_upper = np.tanh(ci_z_upper)
                        
                        results.append({
                            'tet_dimension': tet_dim,
                            'physio_measure': physio_measure,
                            'state': state_label,
                            'r': r,
                            'p_value': p_value,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'n_obs': n_obs
                        })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction within each analysis family
        results_df = self._apply_fdr_correction(results_df)
        
        self.correlation_results = results_df
        
        logger.info(f"Computed {len(results_df)} correlations")
        n_sig = (results_df['p_fdr'] < 0.05).sum()
        logger.info(f"Significant correlations (p_fdr < 0.05): {n_sig}")
        
        return results_df
    
    def _apply_fdr_correction(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply FDR correction within each analysis family.
        
        Analysis families:
        1. Arousal-Physiology: emotional_intensity_z vs physio
        2. Valence-Physiology: valence_index_z vs physio
        3. All Affective-Physiology: other TET dims vs physio
        
        Args:
            results_df: DataFrame with correlation results
        
        Returns:
            DataFrame with p_fdr column added
        """
        results_df = results_df.copy()
        results_df['p_fdr'] = np.nan
        
        # Define analysis families
        families = {
            'arousal': results_df['tet_dimension'] == 'emotional_intensity_z',
            'valence': results_df['tet_dimension'] == 'valence_index_z',
            'affective': results_df['tet_dimension'].isin(self.tet_affective)
        }
        
        # Apply FDR correction within each family
        for family_name, family_mask in families.items():
            if family_mask.sum() > 0:
                p_values = results_df.loc[family_mask, 'p_value'].values
                _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
                results_df.loc[family_mask, 'p_fdr'] = p_fdr
                
                n_sig = (p_fdr < 0.05).sum()
                logger.info(
                    f"FDR correction for {family_name} family: "
                    f"{n_sig}/{len(p_fdr)} significant"
                )
        
        return results_df
    
    def load_pca_loadings(self) -> pd.DataFrame:
        """
        Load PCA loadings from composite arousal index analysis.
        
        PC1 (ArousalIndex) is already computed and available in merged_data.
        This method loads the PCA loadings for documentation purposes.
        
        Returns:
            DataFrame with PCA loadings for HR_z, SMNA_AUC_z, RVT_z
        
        Note:
            Loadings file should be at results/composite/pca_loadings_pc1.csv
            If file doesn't exist, logs warning and returns None.
        """
        loadings_path = Path('results/composite/pca_loadings_pc1.csv')
        
        if not loadings_path.exists():
            logger.warning(
                f"PCA loadings file not found at {loadings_path}. "
                "ArousalIndex (PC1) is still available in merged_data."
            )
            return None
        
        try:
            loadings_df = pd.read_csv(loadings_path)
            self.pca_loadings = loadings_df
            
            logger.info("Loaded PCA loadings for documentation")
            logger.info(f"PC1 loadings:\n{loadings_df}")
            
            return loadings_df
        
        except Exception as e:
            logger.error(f"Error loading PCA loadings: {e}")
            return None
    
    def regression_analysis(self, by_state: bool = True) -> pd.DataFrame:
        """
        Fit linear regression models predicting TET from physiological PC1 (ArousalIndex).
        
        Models:
        1. emotional_intensity_z ~ ArousalIndex
        2. valence_index_z ~ ArousalIndex
        3. Each TET_AFFECTIVE dimension ~ ArousalIndex
        
        Args:
            by_state: If True, fit models separately for RS and DMT states
        
        Returns:
            DataFrame with columns:
            - outcome_variable: TET dimension name
            - predictor: 'ArousalIndex'
            - state: RS or DMT (if by_state=True)
            - beta: Standardized regression coefficient
            - r_squared: Proportion of variance explained
            - p_value: Significance of beta
            - ci_lower, ci_upper: 95% CI for beta
            - n_obs: Number of observations
        
        Note:
            ArousalIndex is already standardized (PC1 from z-scored inputs).
            TET dimensions are also z-scored, so beta is standardized.
        """
        logger.info("Performing regression analysis: TET ~ ArousalIndex...")
        
        results = []
        states = self.data['state'].unique() if by_state else [None]
        
        for state in states:
            # Filter data by state
            if state is not None:
                state_data = self.data[self.data['state'] == state].copy()
                state_label = state
            else:
                state_data = self.data.copy()
                state_label = 'All'
            
            # Fit regression for each TET dimension
            # Include all affective dimensions plus valence_index_z (emotional_intensity_z already in tet_affective)
            tet_dims_to_test = self.tet_affective + ['valence_index_z']
            for tet_dim in tet_dims_to_test:
                # Extract valid pairs (drop NaN)
                valid_mask = (
                    state_data[tet_dim].notna() & 
                    state_data['ArousalIndex'].notna()
                )
                y = state_data.loc[valid_mask, tet_dim].values
                X = state_data.loc[valid_mask, 'ArousalIndex'].values
                
                n_obs = len(y)
                
                if n_obs < 30:
                    logger.warning(
                        f"Low sample size for {tet_dim} ~ ArousalIndex "
                        f"in state {state_label}: N={n_obs}"
                    )
                
                if n_obs >= 3:
                    # Add constant for intercept
                    X_with_const = sm.add_constant(X)
                    
                    # Fit OLS regression
                    model = sm.OLS(y, X_with_const).fit()
                    
                    # Extract statistics
                    beta = model.params[1]  # Slope coefficient
                    r_squared = model.rsquared
                    p_value = model.pvalues[1]
                    conf_int = model.conf_int()
                    ci_lower, ci_upper = conf_int[1, 0], conf_int[1, 1]
                    
                    results.append({
                        'outcome_variable': tet_dim,
                        'predictor': 'ArousalIndex',
                        'state': state_label,
                        'beta': beta,
                        'r_squared': r_squared,
                        'p_value': p_value,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'n_obs': n_obs
                    })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.regression_results = results_df
        
        logger.info(f"Fitted {len(results_df)} regression models")
        n_sig = (results_df['p_value'] < 0.05).sum()
        logger.info(f"Significant regressions (p < 0.05): {n_sig}")
        
        return results_df
    
    def test_arousal_valence_hypothesis(self) -> pd.DataFrame:
        """
        Test hypothesis: arousal-ArousalIndex correlation > valence-ArousalIndex correlation.
        
        Uses Steiger's Z-test for comparing dependent correlations:
        - H0: |r_arousal_ArousalIndex| = |r_valence_ArousalIndex|
        - H1: |r_arousal_ArousalIndex| > |r_valence_ArousalIndex|
        
        Process:
        1. Compute r_arousal_ArousalIndex and r_valence_ArousalIndex
        2. Compute r_arousal_valence (correlation between arousal and valence)
        3. Apply Steiger's Z-test formula
        4. Compute one-tailed p-value (directional hypothesis)
        5. Repeat for each state (RS, DMT)
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - r_arousal_arousalindex: Arousal-ArousalIndex correlation
            - r_valence_arousalindex: Valence-ArousalIndex correlation
            - r_arousal_valence: Arousal-valence correlation
            - z_statistic: Steiger's Z
            - p_value: One-tailed p-value
            - conclusion: 'arousal > valence' or 'no difference'
        """
        logger.info("Testing arousal vs valence hypothesis using Steiger's Z-test...")
        
        results = []
        
        for state in self.data['state'].unique():
            state_data = self.data[self.data['state'] == state].copy()
            
            # Extract valid observations
            valid_mask = (
                state_data['emotional_intensity_z'].notna() &
                state_data['valence_index_z'].notna() &
                state_data['ArousalIndex'].notna()
            )
            
            arousal = state_data.loc[valid_mask, 'emotional_intensity_z'].values
            valence = state_data.loc[valid_mask, 'valence_index_z'].values
            arousal_index = state_data.loc[valid_mask, 'ArousalIndex'].values
            
            n = len(arousal)
            
            if n < 30:
                logger.warning(f"Low sample size for hypothesis test in state {state}: N={n}")
            
            if n >= 3:
                # Compute correlations
                r_arousal_arousalindex, _ = stats.pearsonr(arousal, arousal_index)
                r_valence_arousalindex, _ = stats.pearsonr(valence, arousal_index)
                r_arousal_valence, _ = stats.pearsonr(arousal, valence)
                
                # Apply Steiger's Z-test
                z1 = np.arctanh(r_arousal_arousalindex)
                z2 = np.arctanh(r_valence_arousalindex)
                r_jk = r_arousal_valence
                
                # Compute standard error
                denominator = np.sqrt(
                    (2 * (n - 1) * (1 - r_jk)) / 
                    ((n - 3) * (1 + r_jk))
                )
                
                z_statistic = (z1 - z2) / denominator
                
                # One-tailed p-value (directional hypothesis: arousal > valence)
                p_value = 1 - stats.norm.cdf(z_statistic)
                
                # Conclusion
                conclusion = 'arousal > valence' if p_value < 0.05 else 'no difference'
                
                results.append({
                    'state': state,
                    'r_arousal_arousalindex': r_arousal_arousalindex,
                    'r_valence_arousalindex': r_valence_arousalindex,
                    'r_arousal_valence': r_arousal_valence,
                    'z_statistic': z_statistic,
                    'p_value': p_value,
                    'conclusion': conclusion,
                    'n_obs': n
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.arousal_valence_hypothesis = results_df
        
        logger.info("Hypothesis test results:")
        for _, row in results_df.iterrows():
            logger.info(
                f"  {row['state']}: Z={row['z_statistic']:.3f}, "
                f"p={row['p_value']:.4f}, {row['conclusion']}"
            )
        
        return results_df
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export all analysis results to CSV files.
        
        Args:
            output_dir: Directory to save results
        
        Returns:
            Dictionary mapping file types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Export correlation results
        if self.correlation_results is not None:
            corr_path = output_path / 'correlation_results.csv'
            self.correlation_results.to_csv(corr_path, index=False)
            file_paths['correlation_results'] = str(corr_path)
            logger.info(f"Exported correlation results to {corr_path}")
        
        # Export regression results
        if self.regression_results is not None:
            reg_path = output_path / 'regression_results.csv'
            self.regression_results.to_csv(reg_path, index=False)
            file_paths['regression_results'] = str(reg_path)
            logger.info(f"Exported regression results to {reg_path}")
        
        # Export arousal-valence hypothesis results
        if self.arousal_valence_hypothesis is not None:
            hyp_path = output_path / 'arousal_valence_hypothesis.csv'
            self.arousal_valence_hypothesis.to_csv(hyp_path, index=False)
            file_paths['arousal_valence_hypothesis'] = str(hyp_path)
            logger.info(f"Exported hypothesis test results to {hyp_path}")
        
        # Copy PCA loadings for documentation
        if self.pca_loadings is not None:
            pca_path = output_path / 'pca_loadings_pc1.csv'
            self.pca_loadings.to_csv(pca_path, index=False)
            file_paths['pca_loadings'] = str(pca_path)
            logger.info(f"Copied PCA loadings to {pca_path}")
        
        logger.info(f"Exported {len(file_paths)} result files to {output_dir}")
        
        return file_paths
