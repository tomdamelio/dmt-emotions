"""
Canonical Correlation Analysis (CCA) for physiological-TET integration.

This module implements CCA to identify shared latent dimensions between
physiological signals (HR, SMNA_AUC, RVT) and affective TET dimensions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.cross_decomposition import CCA
from scipy import stats
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


class TETPhysioCCAAnalyzer:
    """
    Canonical Correlation Analysis for physiological-TET relationships.
    
    CCA finds linear combinations of physiological measures and TET dimensions
    that maximize their correlation, revealing shared latent dimensions.
    
    Attributes:
        data: Merged physiological-TET dataset
        physio_measures: List of physiological measure names
        tet_affective: List of TET affective dimension names
        cca_models: Dict mapping state to fitted CCA models
        canonical_correlations: Dict storing canonical correlation results
        canonical_loadings: Dict storing canonical loading results
    
    Example:
        >>> analyzer = TETPhysioCCAAnalyzer(merged_data)
        >>> analyzer.fit_cca(n_components=2)
        >>> variates_df = analyzer.extract_canonical_variates()
        >>> loadings_df = analyzer.compute_canonical_loadings()
        >>> analyzer.export_results('results/tet/physio_correlation')
    """
    
    def __init__(self, merged_data: pd.DataFrame):
        """
        Initialize CCA analyzer.
        
        Args:
            merged_data: Merged physio-TET dataset from TETPhysioDataLoader
                Must contain columns for physiological measures and TET dimensions
        """
        self.data = merged_data.copy()
        self.physio_measures = ['HR', 'SMNA_AUC', 'RVT']
        self.tet_affective = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        
        # Storage for results
        self.cca_models: Dict[str, CCA] = {}
        self.canonical_correlations: Dict[str, np.ndarray] = {}
        self.canonical_loadings: Dict[str, pd.DataFrame] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def prepare_matrices(self, state: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare physiological and TET matrices for CCA.
        
        Process:
        1. Filter data by state (RS or DMT)
        2. Extract physiological matrix X (n_obs × 3)
        3. Extract TET affective matrix Y (n_obs × 6)
        4. Remove rows with missing values (complete cases only)
        5. Standardize both matrices (z-score each column)
        
        Args:
            state: 'RS' or 'DMT'
        
        Returns:
            X: Physiological matrix (n_obs × 3), standardized
            Y: TET affective matrix (n_obs × 6), standardized
        
        Raises:
            ValueError: If insufficient observations (< 100) after filtering
        """
        # Filter by state
        state_data = self.data[self.data['state'] == state].copy()
        
        # Extract physiological matrix X (n_obs × 3)
        X = state_data[self.physio_measures].values
        
        # Extract TET affective matrix Y (n_obs × 6)
        Y = state_data[self.tet_affective].values
        
        # Remove rows with missing values (complete cases only)
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X = X[valid_mask]
        Y = Y[valid_mask]
        
        n_obs = X.shape[0]
        
        # Check minimum sample size
        if n_obs < 100:
            raise ValueError(
                f"Insufficient observations for CCA in {state} state: "
                f"{n_obs} < 100 (minimum required)"
            )
        
        # Standardize both matrices (z-score each column)
        # TET dimensions already z-scored within subject, but re-standardize for CCA
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_standardized = (X - X_mean) / X_std
        
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_standardized = (Y - Y_mean) / Y_std
        
        self.logger.info(
            f"Prepared matrices for {state}: X shape {X_standardized.shape}, "
            f"Y shape {Y_standardized.shape}, N={n_obs}"
        )
        
        return X_standardized, Y_standardized
    
    def fit_cca(self, n_components: int = 2) -> Dict[str, CCA]:
        """
        Fit Canonical Correlation Analysis for each state.
        
        CCA finds linear combinations:
        - U = X @ W_x (canonical variates for physio)
        - V = Y @ W_y (canonical variates for TET)
        
        Such that corr(U_i, V_i) is maximized for each pair i.
        
        Process:
        1. For each state (RS, DMT):
           a. Prepare matrices X (physio) and Y (TET)
           b. Fit CCA with n_components canonical variates
           c. Extract canonical correlations (r_i)
           d. Store fitted model
        
        Args:
            n_components: Number of canonical variate pairs (default: 2)
        
        Returns:
            Dict mapping state to fitted CCA model
        
        Interpretation:
        - Canonical variate 1: Strongest shared dimension
        - Canonical variate 2: Second strongest (orthogonal to 1)
        - Canonical correlation r_i: Strength of relationship for pair i
        """
        states = ['RS', 'DMT']
        
        for state in states:
            try:
                # Prepare matrices
                X, Y = self.prepare_matrices(state)
                
                # Fit CCA
                cca = CCA(n_components=n_components)
                cca.fit(X, Y)
                
                # Transform to canonical variates
                U, V = cca.transform(X, Y)
                
                # Compute canonical correlations
                canonical_corrs = np.array([
                    np.corrcoef(U[:, i], V[:, i])[0, 1]
                    for i in range(n_components)
                ])
                
                # Store results
                self.cca_models[state] = cca
                self.canonical_correlations[state] = canonical_corrs
                
                self.logger.info(
                    f"Fitted CCA for {state}: "
                    f"canonical correlations = {canonical_corrs}"
                )
                
            except ValueError as e:
                self.logger.warning(f"Failed to fit CCA for {state}: {e}")
                continue
        
        return self.cca_models
    
    def extract_canonical_variates(self) -> pd.DataFrame:
        """
        Extract canonical correlations and test significance.
        
        For each state and canonical variate pair:
        1. Extract canonical correlation r_i
        2. Compute Wilks' Lambda for significance test
        3. Compute degrees of freedom
        4. Compute p-value (chi-square approximation)
        
        Wilks' Lambda test:
        - H0: No relationship between X and Y
        - Lambda = product of (1 - r_i²) for all i
        - Chi-square statistic: -n * ln(Lambda)
        - df = p * q (p = dim(X), q = dim(Y))
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1 or 2
            - canonical_correlation: r_i
            - wilks_lambda: Wilks' Lambda statistic
            - chi_square: Chi-square test statistic
            - df: Degrees of freedom
            - p_value: Significance of canonical correlation
        """
        results = []
        
        for state in self.cca_models.keys():
            # Get canonical correlations
            canonical_corrs = self.canonical_correlations[state]
            n_components = len(canonical_corrs)
            
            # Get sample size
            state_data = self.data[self.data['state'] == state]
            X = state_data[self.physio_measures].values
            Y = state_data[self.tet_affective].values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
            n_obs = valid_mask.sum()
            
            # Dimensions
            p = len(self.physio_measures)  # 3
            q = len(self.tet_affective)    # 6
            df = p * q  # 18
            
            # Compute Wilks' Lambda for all canonical correlations
            wilks_lambda = np.prod([1 - r**2 for r in canonical_corrs])
            
            # Chi-square test statistic
            chi_square = -n_obs * np.log(wilks_lambda)
            
            # P-value
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            # Store results for each canonical variate
            for i, r in enumerate(canonical_corrs):
                results.append({
                    'state': state,
                    'canonical_variate': i + 1,
                    'canonical_correlation': r,
                    'wilks_lambda': wilks_lambda,
                    'chi_square': chi_square,
                    'df': df,
                    'p_value': p_value,
                    'n_obs': n_obs
                })
        
        results_df = pd.DataFrame(results)
        
        self.logger.info(
            f"Extracted canonical variates: "
            f"{len(results_df)} variate pairs across {len(self.cca_models)} states"
        )
        
        return results_df
    
    def compute_canonical_loadings(self) -> pd.DataFrame:
        """
        Compute canonical loadings (structure coefficients).
        
        Canonical loadings = correlations between original variables
        and canonical variates.
        
        For physiological variables:
        - Loading_X_i = corr(X_j, U_i) for each variable j and variate i
        
        For TET variables:
        - Loading_Y_i = corr(Y_k, V_i) for each variable k and variate i
        
        Interpretation:
        - High loading (|r| > 0.3): Variable contributes strongly to variate
        - Sign indicates direction of relationship
        - Loadings reveal "meaning" of each canonical dimension
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1 or 2
            - variable_set: 'physio' or 'tet'
            - variable_name: Original variable name
            - loading: Canonical loading (correlation)
        
        Example interpretation:
            Canonical Variate 1 (r = 0.65):
            Physio loadings: HR (0.82), SMNA (0.71), RVT (0.45)
            TET loadings: emotional_intensity (0.78), anxiety (0.62), 
                          interoception (0.54)
            → Interpretation: "Autonomic arousal" dimension
        """
        loadings_list = []
        
        for state, cca_model in self.cca_models.items():
            # Prepare matrices
            X, Y = self.prepare_matrices(state)
            
            # Transform to canonical variates
            U, V = cca_model.transform(X, Y)
            
            n_components = U.shape[1]
            
            # Compute loadings for physiological variables
            # Loading = correlation between original variable and canonical variate
            for i in range(n_components):
                for j, var_name in enumerate(self.physio_measures):
                    loading = np.corrcoef(X[:, j], U[:, i])[0, 1]
                    loadings_list.append({
                        'state': state,
                        'canonical_variate': i + 1,
                        'variable_set': 'physio',
                        'variable_name': var_name,
                        'loading': loading
                    })
            
            # Compute loadings for TET variables
            for i in range(n_components):
                for k, var_name in enumerate(self.tet_affective):
                    loading = np.corrcoef(Y[:, k], V[:, i])[0, 1]
                    # Clean variable name (remove _z suffix)
                    clean_name = var_name.replace('_z', '')
                    loadings_list.append({
                        'state': state,
                        'canonical_variate': i + 1,
                        'variable_set': 'tet',
                        'variable_name': clean_name,
                        'loading': loading
                    })
        
        loadings_df = pd.DataFrame(loadings_list)
        
        # Store for later use
        self.canonical_loadings = loadings_df
        
        self.logger.info(
            f"Computed canonical loadings: {len(loadings_df)} loadings "
            f"across {len(self.cca_models)} states"
        )
        
        return loadings_df
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export CCA results to CSV files.
        
        Creates:
        - cca_results.csv: Canonical correlations and significance tests
        - cca_loadings.csv: Canonical loadings for all variables
        
        Args:
            output_dir: Directory to save results
        
        Returns:
            Dict mapping file types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Export canonical variates (correlations and significance)
        if self.canonical_correlations:
            variates_df = self.extract_canonical_variates()
            variates_path = output_path / 'cca_results.csv'
            variates_df.to_csv(variates_path, index=False)
            file_paths['cca_results'] = str(variates_path)
            self.logger.info(f"Exported CCA results to {variates_path}")
        
        # Export canonical loadings
        if isinstance(self.canonical_loadings, pd.DataFrame):
            loadings_path = output_path / 'cca_loadings.csv'
            self.canonical_loadings.to_csv(loadings_path, index=False)
            file_paths['cca_loadings'] = str(loadings_path)
            self.logger.info(f"Exported CCA loadings to {loadings_path}")
        
        return file_paths
