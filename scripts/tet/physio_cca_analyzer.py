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
        self.permutation_results: Dict[str, pd.DataFrame] = {}
        self.cv_results: Dict[str, pd.DataFrame] = {}
        
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
    
    def _subject_level_shuffle(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        subject_ids: np.ndarray,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform subject-level shuffling for permutation testing.
        
        This method preserves within-subject temporal structure while
        randomly pairing physiological and TET data across subjects.
        
        Process:
        1. Identify unique subjects
        2. For each subject i:
           a. Keep subject i's physiological matrix intact
           b. Randomly pair with subject j's TET matrix (i ≠ j)
           c. Maintain temporal order within each subject
        
        Args:
            X: Physiological matrix (n_obs × 3)
            Y: TET affective matrix (n_obs × 6)
            subject_ids: Subject identifiers for each observation
            random_state: Random seed for reproducibility
        
        Returns:
            X_perm: Original X (unchanged)
            Y_perm: Shuffled Y with preserved temporal structure
        
        Note:
            This approach maintains temporal autocorrelation within subjects
            while breaking the cross-subject physiological-affective coupling.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        # Create permutation mapping (i -> j where i ≠ j)
        permuted_subjects = unique_subjects.copy()
        
        # Shuffle until no subject maps to itself
        valid_permutation = False
        max_attempts = 100
        attempt = 0
        
        while not valid_permutation and attempt < max_attempts:
            np.random.shuffle(permuted_subjects)
            # Check that no subject maps to itself
            if not np.any(unique_subjects == permuted_subjects):
                valid_permutation = True
            attempt += 1
        
        if not valid_permutation:
            # Fallback: use derangement algorithm
            # Simple swap-based derangement
            for i in range(n_subjects):
                if unique_subjects[i] == permuted_subjects[i]:
                    # Find another position to swap with
                    for j in range(i + 1, n_subjects):
                        if unique_subjects[j] != permuted_subjects[i] and \
                           unique_subjects[i] != permuted_subjects[j]:
                            permuted_subjects[i], permuted_subjects[j] = \
                                permuted_subjects[j], permuted_subjects[i]
                            break
        
        # Create mapping dictionary
        subject_mapping = dict(zip(unique_subjects, permuted_subjects))
        
        # Apply permutation to Y
        Y_perm = np.zeros_like(Y)
        
        for subject in unique_subjects:
            # Get indices for this subject's data
            subject_mask = subject_ids == subject
            
            # Get the subject this one is paired with
            paired_subject = subject_mapping[subject]
            paired_mask = subject_ids == paired_subject
            
            # Copy paired subject's Y data to this subject's position
            # This preserves temporal order within each subject
            Y_perm[subject_mask] = Y[paired_mask]
        
        return X, Y_perm
    
    def _fit_permuted_cca(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        subject_ids: np.ndarray,
        n_components: int,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit CCA on permuted data.
        
        Args:
            X: Physiological matrix (n_obs × 3)
            Y: TET affective matrix (n_obs × 6)
            subject_ids: Subject identifiers
            n_components: Number of canonical variates
            random_state: Random seed
        
        Returns:
            Array of permuted canonical correlations (n_components,)
        """
        # Shuffle at subject level
        X_perm, Y_perm = self._subject_level_shuffle(X, Y, subject_ids, random_state)
        
        # Fit CCA on permuted data
        cca_perm = CCA(n_components=n_components)
        cca_perm.fit(X_perm, Y_perm)
        
        # Transform to canonical variates
        U_perm, V_perm = cca_perm.transform(X_perm, Y_perm)
        
        # Compute canonical correlations
        canonical_corrs_perm = np.array([
            np.corrcoef(U_perm[:, i], V_perm[:, i])[0, 1]
            for i in range(n_components)
        ])
        
        return canonical_corrs_perm
    
    def _compute_permutation_pvalues(
        self,
        observed_corrs: np.ndarray,
        permuted_corrs: np.ndarray
    ) -> np.ndarray:
        """
        Compute empirical p-values from permutation distribution.
        
        P-value = (count of permuted r >= observed r + 1) / (n_permutations + 1)
        
        The +1 in numerator and denominator accounts for the observed value
        being part of the permutation distribution.
        
        Args:
            observed_corrs: Observed canonical correlations (n_components,)
            permuted_corrs: Permuted correlations (n_permutations × n_components)
        
        Returns:
            Array of p-values (n_components,)
        """
        n_permutations = permuted_corrs.shape[0]
        n_components = observed_corrs.shape[0]
        
        p_values = np.zeros(n_components)
        
        for i in range(n_components):
            # Count permutations where r_perm >= r_observed
            count = np.sum(permuted_corrs[:, i] >= observed_corrs[i])
            
            # Compute empirical p-value (one-tailed test)
            p_values[i] = (count + 1) / (n_permutations + 1)
        
        return p_values
    
    def permutation_test(
        self,
        n_permutations: int = 100,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform subject-level permutation testing for CCA.
        
        This method validates CCA canonical correlation significance by
        randomly pairing subjects' physiological and TET data while
        preserving within-subject temporal structure.
        
        Process:
        1. For each state (RS, DMT):
           a. Extract observed canonical correlations
           b. For each permutation:
              - Shuffle subject pairings (i -> j where i ≠ j)
              - Fit CCA on permuted data
              - Extract permuted canonical correlations
           c. Compute empirical p-values
        
        Args:
            n_permutations: Number of permutation iterations (default: 100)
                Use 100 for debugging (~2 min), 1000 for publication (~15 min)
            random_state: Random seed for reproducibility (default: 42)
        
        Returns:
            Dict mapping state to DataFrame with columns:
            - canonical_variate: 1, 2, ...
            - observed_r: Observed canonical correlation
            - permutation_p_value: Empirical p-value
            - n_permutations: Number of permutations performed
        
        Note:
            Start with n_permutations=100 for debugging to catch bugs quickly.
            Scale to n_permutations=1000 for final publication results.
        
        Example:
            >>> analyzer.fit_cca(n_components=2)
            >>> perm_results = analyzer.permutation_test(n_permutations=100)
            >>> print(perm_results['DMT'])
        """
        if not self.cca_models:
            raise ValueError(
                "Must fit CCA models first using fit_cca() before permutation testing"
            )
        
        self.logger.info(
            f"Starting permutation testing with {n_permutations} iterations "
            f"(random_state={random_state})"
        )
        
        results_dict = {}
        
        for state in self.cca_models.keys():
            self.logger.info(f"Permutation testing for {state} state...")
            
            # Prepare matrices and get subject IDs
            X, Y = self.prepare_matrices(state)
            
            # Get subject IDs for this state
            state_data = self.data[self.data['state'] == state].copy()
            subject_ids_full = state_data['subject'].values
            
            # Filter to valid observations (same as prepare_matrices)
            X_full = state_data[self.physio_measures].values
            Y_full = state_data[self.tet_affective].values
            valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
            subject_ids = subject_ids_full[valid_mask]
            
            # Get observed canonical correlations
            observed_corrs = self.canonical_correlations[state]
            n_components = len(observed_corrs)
            
            # Initialize storage for permuted correlations
            permuted_corrs = np.zeros((n_permutations, n_components))
            
            # Perform permutations
            for perm_idx in range(n_permutations):
                # Set seed for this permutation
                perm_seed = random_state + perm_idx if random_state is not None else None
                
                # Fit CCA on permuted data
                permuted_corrs[perm_idx] = self._fit_permuted_cca(
                    X, Y, subject_ids, n_components, perm_seed
                )
                
                # Log progress every 10 permutations
                if (perm_idx + 1) % 10 == 0:
                    self.logger.info(
                        f"  Completed {perm_idx + 1}/{n_permutations} permutations"
                    )
            
            # Compute empirical p-values
            p_values = self._compute_permutation_pvalues(observed_corrs, permuted_corrs)
            
            # Create results DataFrame
            results = []
            for i in range(n_components):
                results.append({
                    'state': state,
                    'canonical_variate': i + 1,
                    'observed_r': observed_corrs[i],
                    'permutation_p_value': p_values[i],
                    'n_permutations': n_permutations
                })
            
            results_df = pd.DataFrame(results)
            results_dict[state] = results_df
            
            self.logger.info(
                f"Permutation testing complete for {state}: "
                f"p-values = {p_values}"
            )
        
        # Store results
        self.permutation_results = results_dict
        
        return results_dict
    
    def plot_permutation_distributions(
        self,
        output_dir: str,
        alpha: float = 0.05
    ) -> Dict[str, str]:
        """
        Generate permutation null distribution plots.
        
        For each canonical variate:
        1. Plot histogram of permuted correlations
        2. Mark observed correlation with vertical line
        3. Shade rejection region (top alpha%)
        4. Annotate with p-value
        
        Args:
            output_dir: Directory to save figures
            alpha: Significance level for rejection region (default: 0.05)
        
        Returns:
            Dict mapping state to figure file path
        
        Raises:
            ValueError: If permutation testing has not been performed
        """
        if not self.permutation_results:
            raise ValueError(
                "Must perform permutation testing first using permutation_test()"
            )
        
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        figure_paths = {}
        
        for state in self.permutation_results.keys():
            results_df = self.permutation_results[state]
            n_components = len(results_df)
            
            # Get permuted correlations for this state
            # Need to recompute to get full distribution
            X, Y = self.prepare_matrices(state)
            state_data = self.data[self.data['state'] == state].copy()
            subject_ids_full = state_data['subject'].values
            X_full = state_data[self.physio_measures].values
            Y_full = state_data[self.tet_affective].values
            valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
            subject_ids = subject_ids_full[valid_mask]
            
            # Get n_permutations from results
            n_permutations = results_df['n_permutations'].iloc[0]
            
            # Recompute permuted correlations to get full distribution
            permuted_corrs = np.zeros((n_permutations, n_components))
            for perm_idx in range(n_permutations):
                permuted_corrs[perm_idx] = self._fit_permuted_cca(
                    X, Y, subject_ids, n_components, 42 + perm_idx
                )
            
            # Create figure with subplots for each canonical variate
            fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 5))
            if n_components == 1:
                axes = [axes]
            
            for i in range(n_components):
                ax = axes[i]
                
                # Get observed correlation and p-value
                observed_r = results_df.iloc[i]['observed_r']
                p_value = results_df.iloc[i]['permutation_p_value']
                
                # Plot histogram of permuted correlations
                ax.hist(
                    permuted_corrs[:, i],
                    bins=30,
                    color='lightgray',
                    edgecolor='black',
                    alpha=0.7,
                    label='Permuted'
                )
                
                # Mark observed correlation
                ax.axvline(
                    observed_r,
                    color='red',
                    linewidth=2,
                    linestyle='--',
                    label=f'Observed (r={observed_r:.3f})'
                )
                
                # Shade rejection region (top alpha%)
                rejection_threshold = np.percentile(permuted_corrs[:, i], (1 - alpha) * 100)
                ax.axvspan(
                    rejection_threshold,
                    permuted_corrs[:, i].max(),
                    alpha=0.2,
                    color='red',
                    label=f'Rejection region (α={alpha})'
                )
                
                # Annotate with p-value
                ax.text(
                    0.05, 0.95,
                    f'p = {p_value:.3f}',
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
                
                # Labels and title
                ax.set_xlabel('Canonical Correlation', fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(
                    f'Canonical Variate {i + 1}',
                    fontsize=12,
                    fontweight='bold'
                )
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            # Overall title
            fig.suptitle(
                f'Permutation Null Distributions - {state} State',
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()
            
            # Save figure
            fig_path = output_path / f'permutation_null_distributions_{state.lower()}.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            figure_paths[state] = str(fig_path)
            self.logger.info(f"Saved permutation distribution plot to {fig_path}")
        
        return figure_paths
    
    def _generate_loso_folds(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        subject_ids: np.ndarray
    ):
        """
        Generate Leave-One-Subject-Out (LOSO) cross-validation folds.
        
        For each unique subject:
        - Train set: all subjects except held-out
        - Test set: held-out subject only
        
        Args:
            X: Physiological matrix (n_obs × 3)
            Y: TET affective matrix (n_obs × 6)
            subject_ids: Subject identifiers for each observation
        
        Yields:
            Tuple of (X_train, Y_train, X_test, Y_test, subject_id)
        """
        unique_subjects = np.unique(subject_ids)
        
        for held_out_subject in unique_subjects:
            # Create train/test masks
            test_mask = subject_ids == held_out_subject
            train_mask = ~test_mask
            
            # Split data
            X_train = X[train_mask]
            Y_train = Y[train_mask]
            X_test = X[test_mask]
            Y_test = Y[test_mask]
            
            yield X_train, Y_train, X_test, Y_test, held_out_subject
    
    def _compute_oos_correlation(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        n_components: int,
        global_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute out-of-sample correlation for one LOSO fold.
        
        CRITICAL: Sign Flipping Handling
        - CCA canonical variates have arbitrary signs
        - Before applying weights to test subject, check sign consistency
        - If correlation with global weights is negative, flip signs
        - This prevents sign cancellation when averaging across folds
        
        CRITICAL: Low Variance Safety
        - Handle cases where test subject has zero/near-zero variance
        - If correlation calculation fails (NaN), flag as invalid fold
        - Log warning with subject ID and continue to next fold
        
        Args:
            X_train: Training physiological matrix (n_train × 3)
            Y_train: Training TET matrix (n_train × 6)
            X_test: Test physiological matrix (n_test × 3)
            Y_test: Test TET matrix (n_test × 6)
            n_components: Number of canonical variates
            global_weights: Optional tuple of (W_x_global, W_y_global) for sign alignment
        
        Returns:
            Array of out-of-sample correlations (n_components,)
            Returns NaN for components with invalid correlations
        """
        try:
            # Fit CCA on training data
            cca_train = CCA(n_components=n_components)
            cca_train.fit(X_train, Y_train)
            
            # Extract canonical weights
            W_x = cca_train.x_weights_  # (3 × n_components)
            W_y = cca_train.y_weights_  # (6 × n_components)
            
            # Sign flipping: align with global weights if provided
            if global_weights is not None:
                W_x_global, W_y_global = global_weights
                
                for i in range(n_components):
                    # Compute correlation between train weights and global weights
                    corr_x = np.corrcoef(W_x[:, i], W_x_global[:, i])[0, 1]
                    corr_y = np.corrcoef(W_y[:, i], W_y_global[:, i])[0, 1]
                    
                    # If correlation is negative, flip signs
                    if corr_x < 0:
                        W_x[:, i] = -W_x[:, i]
                    if corr_y < 0:
                        W_y[:, i] = -W_y[:, i]
            
            # Transform test data using training weights
            U_test = X_test @ W_x  # (n_test × n_components)
            V_test = Y_test @ W_y  # (n_test × n_components)
            
            # Compute out-of-sample correlations
            r_oos = np.zeros(n_components)
            
            for i in range(n_components):
                # Check for zero/near-zero variance
                if np.std(U_test[:, i]) < 1e-10 or np.std(V_test[:, i]) < 1e-10:
                    r_oos[i] = np.nan
                    continue
                
                # Compute correlation
                try:
                    r_oos[i] = np.corrcoef(U_test[:, i], V_test[:, i])[0, 1]
                    
                    # Check for NaN
                    if np.isnan(r_oos[i]):
                        continue
                        
                except Exception:
                    r_oos[i] = np.nan
            
            return r_oos
            
        except Exception as e:
            self.logger.warning(f"Failed to compute OOS correlation: {e}")
            return np.full(n_components, np.nan)
    
    def loso_cross_validation(
        self,
        state: str,
        n_components: int = 2
    ) -> pd.DataFrame:
        """
        Perform Leave-One-Subject-Out (LOSO) cross-validation for CCA.
        
        This method assesses CCA generalization by training on N-1 subjects
        and testing on the held-out subject.
        
        Process:
        1. For each subject:
           a. Train CCA on all other subjects
           b. Extract canonical weights (W_x, W_y)
           c. Handle sign flipping (align with global weights)
           d. Transform held-out subject's data
           e. Compute out-of-sample correlation
           f. Handle low variance cases (flag as invalid)
        2. Compute summary statistics across folds
        
        Args:
            state: 'RS' or 'DMT'
            n_components: Number of canonical variates (default: 2)
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1, 2, ...
            - fold_id: Subject held out
            - r_oos: Out-of-sample correlation (NaN if invalid)
        
        Technical Note:
            Sign indeterminacy is a fundamental property of CCA that must
            be handled to avoid spurious results. We align signs with the
            global (full-sample) CCA weights.
        
        Example:
            >>> analyzer.fit_cca(n_components=2)
            >>> cv_results = analyzer.loso_cross_validation('DMT')
            >>> print(cv_results.groupby('canonical_variate')['r_oos'].mean())
        """
        if state not in self.cca_models:
            raise ValueError(
                f"Must fit CCA for {state} state first using fit_cca()"
            )
        
        self.logger.info(
            f"Starting LOSO cross-validation for {state} state "
            f"with {n_components} components"
        )
        
        # Prepare matrices and get subject IDs
        X, Y = self.prepare_matrices(state)
        
        state_data = self.data[self.data['state'] == state].copy()
        subject_ids_full = state_data['subject'].values
        
        # Filter to valid observations (same as prepare_matrices)
        X_full = state_data[self.physio_measures].values
        Y_full = state_data[self.tet_affective].values
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
        subject_ids = subject_ids_full[valid_mask]
        
        # Get global weights for sign alignment
        global_model = self.cca_models[state]
        W_x_global = global_model.x_weights_
        W_y_global = global_model.y_weights_
        global_weights = (W_x_global, W_y_global)
        
        # Perform LOSO cross-validation
        cv_records = []
        
        for X_train, Y_train, X_test, Y_test, subject_id in self._generate_loso_folds(
            X, Y, subject_ids
        ):
            # Compute out-of-sample correlations
            r_oos = self._compute_oos_correlation(
                X_train, Y_train, X_test, Y_test,
                n_components, global_weights
            )
            
            # Store results for each canonical variate
            for i in range(n_components):
                cv_records.append({
                    'state': state,
                    'canonical_variate': i + 1,
                    'fold_id': subject_id,
                    'r_oos': r_oos[i]
                })
            
            # Log progress
            if np.all(np.isnan(r_oos)):
                self.logger.warning(
                    f"  Subject {subject_id}: All correlations invalid (low variance)"
                )
            else:
                self.logger.info(
                    f"  Subject {subject_id}: r_oos = {r_oos}"
                )
        
        cv_df = pd.DataFrame(cv_records)
        
        # Store results
        if state not in self.cv_results:
            self.cv_results[state] = cv_df
        else:
            self.cv_results[state] = pd.concat(
                [self.cv_results[state], cv_df],
                ignore_index=True
            )
        
        self.logger.info(
            f"LOSO cross-validation complete for {state}: "
            f"{len(cv_df)} fold results"
        )
        
        return cv_df
    
    def _summarize_cv_results(self, state: str) -> pd.DataFrame:
        """
        Compute cross-validation summary statistics.
        
        RECOMMENDED: Fisher Z-Transformation for Averaging Correlations
        - Convert each r_oos to Fisher z: z = np.arctanh(r_oos)
        - Compute mean_z and sd_z across valid folds (excluding NaN)
        - Convert back to correlation: mean_r_oos = np.tanh(mean_z)
        - Rationale: Correlation distribution is non-normal, especially with small N
        
        For each canonical variate:
        - Compute mean and SD of out-of-sample correlations
        - Compute min and max r_oos across valid folds
        - Compare to in-sample r (overfitting check)
        - Report number of valid and excluded folds
        
        Args:
            state: 'RS' or 'DMT'
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1, 2, ...
            - mean_r_oos: Mean out-of-sample correlation (Fisher Z-transformed)
            - sd_r_oos: Standard deviation of r_oos
            - min_r_oos: Minimum r_oos across valid folds
            - max_r_oos: Maximum r_oos across valid folds
            - in_sample_r: In-sample canonical correlation
            - overfitting_index: (in_sample_r - mean_r_oos) / in_sample_r
            - n_valid_folds: Number of valid folds
            - n_excluded_folds: Number of excluded folds (NaN)
        
        Technical Note:
            Fisher Z-transformation is statistically more rigorous than
            simple averaging of correlations.
        """
        if state not in self.cv_results:
            raise ValueError(
                f"Must perform LOSO cross-validation for {state} first"
            )
        
        cv_df = self.cv_results[state]
        in_sample_corrs = self.canonical_correlations[state]
        
        summary_records = []
        
        for variate_idx in cv_df['canonical_variate'].unique():
            variate_data = cv_df[cv_df['canonical_variate'] == variate_idx]
            
            # Get r_oos values (excluding NaN)
            r_oos_values = variate_data['r_oos'].values
            valid_mask = ~np.isnan(r_oos_values)
            r_oos_valid = r_oos_values[valid_mask]
            
            n_valid = valid_mask.sum()
            n_excluded = (~valid_mask).sum()
            
            if n_valid == 0:
                # All folds invalid
                summary_records.append({
                    'state': state,
                    'canonical_variate': variate_idx,
                    'mean_r_oos': np.nan,
                    'sd_r_oos': np.nan,
                    'min_r_oos': np.nan,
                    'max_r_oos': np.nan,
                    'in_sample_r': in_sample_corrs[variate_idx - 1],
                    'overfitting_index': np.nan,
                    'n_valid_folds': n_valid,
                    'n_excluded_folds': n_excluded
                })
                continue
            
            # Fisher Z-transformation for averaging
            z_values = np.arctanh(r_oos_valid)
            mean_z = np.mean(z_values)
            sd_z = np.std(z_values, ddof=1) if n_valid > 1 else 0.0
            
            # Convert back to correlation
            mean_r_oos = np.tanh(mean_z)
            
            # Compute SD in correlation space (for reporting)
            sd_r_oos = np.std(r_oos_valid, ddof=1) if n_valid > 1 else 0.0
            
            # Min and max
            min_r_oos = np.min(r_oos_valid)
            max_r_oos = np.max(r_oos_valid)
            
            # In-sample correlation
            in_sample_r = in_sample_corrs[variate_idx - 1]
            
            # Overfitting index
            if in_sample_r != 0:
                overfitting_index = (in_sample_r - mean_r_oos) / in_sample_r
            else:
                overfitting_index = np.nan
            
            summary_records.append({
                'state': state,
                'canonical_variate': variate_idx,
                'mean_r_oos': mean_r_oos,
                'sd_r_oos': sd_r_oos,
                'min_r_oos': min_r_oos,
                'max_r_oos': max_r_oos,
                'in_sample_r': in_sample_r,
                'overfitting_index': overfitting_index,
                'n_valid_folds': n_valid,
                'n_excluded_folds': n_excluded
            })
        
        summary_df = pd.DataFrame(summary_records)
        
        self.logger.info(
            f"Computed CV summary for {state}: "
            f"{len(summary_df)} canonical variates"
        )
        
        return summary_df
    
    def compute_cv_significance(self) -> pd.DataFrame:
        """
        Perform statistical significance testing on cross-validation correlations.
        
        This method tests whether out-of-sample correlations are significantly
        greater than zero using both parametric (t-test) and non-parametric
        (Wilcoxon) approaches.
        
        Process:
        1. For each state and canonical variate:
           a. Extract r_oos values from valid folds
           b. Apply Fisher Z-transformation: z = arctanh(r)
           c. Perform one-sample t-test: H0: z = 0, H1: z > 0
           d. Perform Wilcoxon signed-rank test as robust alternative
           e. Compute success rate (proportion of folds with r_oos > 0)
        
        Fisher Z-Transformation Rationale:
        - Raw correlations are not normally distributed
        - Fisher Z-transform normalizes the distribution
        - Enables valid parametric testing with t-test
        - Especially important for small sample sizes (N=7 folds)
        
        One-Tailed Test Rationale:
        - We test if r_oos > 0 (positive prediction)
        - Negative correlations are as bad as zero for our purposes
        - One-tailed test is more powerful for directional hypothesis
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1, 2, ...
            - n_folds: Number of valid folds
            - mean_r_oos: Mean out-of-sample correlation
            - sd_r_oos: Standard deviation of r_oos
            - t_statistic: T-statistic from one-sample t-test
            - p_value_t_test: One-tailed p-value from t-test
            - p_value_wilcoxon: One-tailed p-value from Wilcoxon test
            - success_rate: Proportion of folds with r_oos > 0
            - n_positive_folds: Count of folds with r_oos > 0
            - significant: Boolean (True if p_value_t_test < 0.05)
            - interpretation: Text interpretation of results
        
        Raises:
            ValueError: If cross-validation has not been performed
        
        Example:
            >>> analyzer.fit_cca(n_components=2)
            >>> analyzer.loso_cross_validation('DMT')
            >>> sig_results = analyzer.compute_cv_significance()
            >>> print(sig_results[sig_results['significant']])
        """
        if not self.cv_results:
            raise ValueError(
                "Must perform LOSO cross-validation first using loso_cross_validation()"
            )
        
        self.logger.info("Computing cross-validation significance tests...")
        
        significance_records = []
        
        for state in self.cv_results.keys():
            cv_df = self.cv_results[state]
            
            for variate_idx in cv_df['canonical_variate'].unique():
                variate_data = cv_df[cv_df['canonical_variate'] == variate_idx]
                
                # Get r_oos values (excluding NaN)
                r_oos_values = variate_data['r_oos'].values
                valid_mask = ~np.isnan(r_oos_values)
                r_oos_valid = r_oos_values[valid_mask]
                
                n_valid = valid_mask.sum()
                
                if n_valid < 3:
                    # Insufficient data for statistical testing
                    self.logger.warning(
                        f"{state} CV{variate_idx}: Only {n_valid} valid folds, "
                        "skipping significance testing (minimum 3 required)"
                    )
                    significance_records.append({
                        'state': state,
                        'canonical_variate': variate_idx,
                        'n_folds': n_valid,
                        'mean_r_oos': np.nan,
                        'sd_r_oos': np.nan,
                        't_statistic': np.nan,
                        'p_value_t_test': np.nan,
                        'p_value_wilcoxon': np.nan,
                        'success_rate': np.nan,
                        'n_positive_folds': 0,
                        'significant': False,
                        'interpretation': 'Insufficient Data'
                    })
                    continue
                
                # Clip r_oos to avoid infinities in Fisher Z-transform
                r_oos_clipped = np.clip(r_oos_valid, -0.99999, 0.99999)
                
                # Fisher Z-transformation
                z_scores = np.arctanh(r_oos_clipped)
                
                # One-sample t-test (one-tailed: greater than 0)
                t_stat, p_value_t = stats.ttest_1samp(
                    z_scores,
                    popmean=0,
                    alternative='greater'
                )
                
                # Wilcoxon signed-rank test (robust alternative)
                # Test if r_oos values are significantly greater than 0
                try:
                    w_stat, p_value_w = stats.wilcoxon(
                        r_oos_valid - 0,
                        alternative='greater'
                    )
                except ValueError as e:
                    # Handle case where all values are zero or identical
                    self.logger.warning(
                        f"{state} CV{variate_idx}: Wilcoxon test failed ({e}), "
                        "using p=1.0"
                    )
                    p_value_w = 1.0
                
                # Compute success rate
                n_positive = np.sum(r_oos_valid > 0)
                success_rate = n_positive / n_valid
                
                # Compute mean and SD
                mean_r_oos = np.mean(r_oos_valid)
                sd_r_oos = np.std(r_oos_valid, ddof=1) if n_valid > 1 else 0.0
                
                # Determine significance
                significant = p_value_t < 0.05
                
                # Interpretation
                if p_value_t < 0.05:
                    interpretation = 'Significant Generalization'
                elif p_value_t < 0.10:
                    interpretation = 'Trend'
                else:
                    interpretation = 'Not Significant'
                
                significance_records.append({
                    'state': state,
                    'canonical_variate': variate_idx,
                    'n_folds': n_valid,
                    'mean_r_oos': mean_r_oos,
                    'sd_r_oos': sd_r_oos,
                    't_statistic': t_stat,
                    'p_value_t_test': p_value_t,
                    'p_value_wilcoxon': p_value_w,
                    'success_rate': success_rate,
                    'n_positive_folds': n_positive,
                    'significant': significant,
                    'interpretation': interpretation
                })
                
                self.logger.info(
                    f"{state} CV{variate_idx}: mean_r={mean_r_oos:.3f}, "
                    f"t={t_stat:.2f}, p={p_value_t:.4f} ({interpretation})"
                )
        
        significance_df = pd.DataFrame(significance_records)
        
        self.logger.info(
            f"Computed CV significance for {len(significance_df)} canonical variates"
        )
        
        return significance_df
    
    def plot_cv_diagnostics(
        self,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate cross-validation diagnostic plots.
        
        Creates three diagnostic plots:
        1. Box plots of r_oos distributions per canonical variate
        2. Scatter plot of in-sample vs mean out-of-sample r
        3. Bar chart showing overfitting index per variate
        
        Args:
            output_dir: Directory to save figures
        
        Returns:
            Dict mapping plot type to file path
        
        Raises:
            ValueError: If cross-validation has not been performed
        """
        if not self.cv_results:
            raise ValueError(
                "Must perform LOSO cross-validation first using loso_cross_validation()"
            )
        
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        figure_paths = {}
        
        # Combine all states for plotting
        all_cv_data = []
        all_summaries = []
        
        for state in self.cv_results.keys():
            cv_df = self.cv_results[state]
            all_cv_data.append(cv_df)
            
            summary_df = self._summarize_cv_results(state)
            all_summaries.append(summary_df)
        
        combined_cv = pd.concat(all_cv_data, ignore_index=True)
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        
        # Plot 1: Box plots of r_oos distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, state in enumerate(['RS', 'DMT']):
            ax = axes[idx]
            state_data = combined_cv[combined_cv['state'] == state]
            
            # Prepare data for box plot
            box_data = []
            labels = []
            for variate in sorted(state_data['canonical_variate'].unique()):
                variate_data = state_data[state_data['canonical_variate'] == variate]
                r_oos_valid = variate_data['r_oos'].dropna().values
                box_data.append(r_oos_valid)
                labels.append(f'CV{variate}')
            
            # Create box plot
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            # Add horizontal line at 0
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Canonical Variate', fontsize=11)
            ax.set_ylabel('Out-of-Sample Correlation', fontsize=11)
            ax.set_title(f'{state} State', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(
            'LOSO Cross-Validation: r_oos Distributions',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        
        fig_path = output_path / 'cca_cross_validation_boxplots.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths['boxplots'] = str(fig_path)
        self.logger.info(f"Saved CV boxplots to {fig_path}")
        
        # Plot 2: In-sample vs out-of-sample scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = {'RS': 'blue', 'DMT': 'red'}
        markers = {1: 'o', 2: 's'}
        
        for _, row in combined_summary.iterrows():
            ax.scatter(
                row['in_sample_r'],
                row['mean_r_oos'],
                color=colors[row['state']],
                marker=markers[row['canonical_variate']],
                s=100,
                alpha=0.7,
                label=f"{row['state']} CV{row['canonical_variate']}"
            )
            
            # Add error bars (SD)
            ax.errorbar(
                row['in_sample_r'],
                row['mean_r_oos'],
                yerr=row['sd_r_oos'],
                color=colors[row['state']],
                alpha=0.5,
                capsize=5
            )
        
        # Add identity line
        max_r = max(
            combined_summary['in_sample_r'].max(),
            combined_summary['mean_r_oos'].max()
        )
        min_r = min(
            combined_summary['in_sample_r'].min(),
            combined_summary['mean_r_oos'].min()
        )
        ax.plot([min_r, max_r], [min_r, max_r], 'k--', linewidth=2, alpha=0.5, label='Identity')
        
        ax.set_xlabel('In-Sample Correlation', fontsize=12)
        ax.set_ylabel('Mean Out-of-Sample Correlation', fontsize=12)
        ax.set_title(
            'In-Sample vs Out-of-Sample Correlations',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig_path = output_path / 'cca_cross_validation_scatter.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths['scatter'] = str(fig_path)
        self.logger.info(f"Saved CV scatter plot to {fig_path}")
        
        # Plot 3: Overfitting index bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for grouped bar chart
        x_positions = []
        bar_heights = []
        bar_colors = []
        bar_labels = []
        
        x_pos = 0
        for state in ['RS', 'DMT']:
            state_summary = combined_summary[combined_summary['state'] == state]
            
            for _, row in state_summary.iterrows():
                x_positions.append(x_pos)
                bar_heights.append(row['overfitting_index'])
                bar_colors.append(colors[state])
                bar_labels.append(f"{state}\nCV{row['canonical_variate']}")
                x_pos += 1
            
            x_pos += 0.5  # Add space between states
        
        bars = ax.bar(x_positions, bar_heights, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, fontsize=10)
        
        # Add horizontal line at 0
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add reference line at 0.1 (10% overfitting threshold)
        ax.axhline(0.1, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='10% threshold')
        
        ax.set_ylabel('Overfitting Index', fontsize=12)
        ax.set_title(
            'Overfitting Index: (In-Sample - OOS) / In-Sample',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig_path = output_path / 'cca_cross_validation_overfitting.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        figure_paths['overfitting'] = str(fig_path)
        self.logger.info(f"Saved CV overfitting plot to {fig_path}")
        
        return figure_paths
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export CCA results to CSV files.
        
        Creates:
        - cca_results.csv: Canonical correlations and significance tests
        - cca_loadings.csv: Canonical loadings for all variables
        - cca_redundancy_indices.csv: Redundancy indices for each canonical variate
        - cca_permutation_pvalues.csv: Permutation test p-values (if available)
        - cca_cross_validation_folds.csv: Per-fold CV results (if available)
        - cca_cross_validation_summary.csv: CV summary statistics (if available)
        
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
        
        # Export redundancy indices
        if self.cca_models:
            redundancy_dfs = []
            for state in self.cca_models.keys():
                redundancy_df = self.compute_redundancy_index(state)
                redundancy_dfs.append(redundancy_df)
            
            if redundancy_dfs:
                redundancy_combined = pd.concat(redundancy_dfs, ignore_index=True)
                redundancy_path = output_path / 'cca_redundancy_indices.csv'
                redundancy_combined.to_csv(redundancy_path, index=False)
                file_paths['cca_redundancy_indices'] = str(redundancy_path)
                self.logger.info(f"Exported redundancy indices to {redundancy_path}")
                
                # Add interpretation column
                redundancy_combined['interpretation'] = redundancy_combined.apply(
                    self._interpret_redundancy, axis=1
                )
                redundancy_interpreted_path = output_path / 'cca_redundancy_indices_interpreted.csv'
                redundancy_combined.to_csv(redundancy_interpreted_path, index=False)
                file_paths['cca_redundancy_indices_interpreted'] = str(redundancy_interpreted_path)
        
        # Export permutation test results
        if self.permutation_results:
            # Combine all states into single DataFrame
            perm_dfs = []
            for state, df in self.permutation_results.items():
                perm_dfs.append(df)
            
            if perm_dfs:
                perm_combined = pd.concat(perm_dfs, ignore_index=True)
                perm_path = output_path / 'cca_permutation_pvalues.csv'
                perm_combined.to_csv(perm_path, index=False)
                file_paths['cca_permutation_pvalues'] = str(perm_path)
                self.logger.info(f"Exported permutation p-values to {perm_path}")
                
                # Export full permutation distributions
                self._export_permutation_distributions(output_path)
                file_paths['cca_permutation_distributions'] = str(
                    output_path / 'cca_permutation_distributions.csv'
                )
        
        # Export cross-validation results
        if self.cv_results:
            # Export per-fold results
            cv_dfs = []
            for state, df in self.cv_results.items():
                cv_dfs.append(df)
            
            if cv_dfs:
                cv_combined = pd.concat(cv_dfs, ignore_index=True)
                cv_folds_path = output_path / 'cca_cross_validation_folds.csv'
                cv_combined.to_csv(cv_folds_path, index=False)
                file_paths['cca_cross_validation_folds'] = str(cv_folds_path)
                self.logger.info(f"Exported CV fold results to {cv_folds_path}")
            
            # Export summary statistics
            summary_dfs = []
            for state in self.cv_results.keys():
                summary_df = self._summarize_cv_results(state)
                summary_dfs.append(summary_df)
            
            if summary_dfs:
                summary_combined = pd.concat(summary_dfs, ignore_index=True)
                cv_summary_path = output_path / 'cca_cross_validation_summary.csv'
                summary_combined.to_csv(cv_summary_path, index=False)
                file_paths['cca_cross_validation_summary'] = str(cv_summary_path)
                self.logger.info(f"Exported CV summary to {cv_summary_path}")
            
            # Export CV significance results
            try:
                cv_significance_df = self.compute_cv_significance()
                cv_sig_path = output_path / 'cca_cv_significance.csv'
                cv_significance_df.to_csv(cv_sig_path, index=False)
                file_paths['cca_cv_significance'] = str(cv_sig_path)
                self.logger.info(f"Exported CV significance results to {cv_sig_path}")
            except ValueError as e:
                self.logger.warning(f"Could not compute CV significance: {e}")
        
        return file_paths
    
    def _interpret_redundancy(self, row: pd.Series) -> str:
        """
        Interpret redundancy index magnitude.
        
        Interpretation guidelines:
        - High (> 15%): Strong shared variance
        - Moderate (10-15%): Meaningful relationship
        - Low (5-10%): Weak relationship
        - Very Low (< 5%): Minimal shared variance, potential overfitting
        
        Args:
            row: DataFrame row with redundancy values
        
        Returns:
            Interpretation string
        """
        if row['canonical_variate'] == 'Total':
            # For total row, interpret based on total redundancy
            redundancy = max(
                row['redundancy_Y_given_X'],
                row['redundancy_X_given_Y']
            )
        else:
            # For individual variates, interpret based on average
            redundancy = (
                row['redundancy_Y_given_X'] + row['redundancy_X_given_Y']
            ) / 2
        
        if pd.isna(redundancy):
            return 'N/A'
        elif redundancy > 0.15:
            return 'High'
        elif redundancy > 0.10:
            return 'Moderate'
        elif redundancy > 0.05:
            return 'Low'
        else:
            return 'Very Low'
    
    def _export_permutation_distributions(self, output_path: Path) -> None:
        """
        Export full permutation distributions to CSV.
        
        Creates a CSV with all permuted correlations for each state and
        canonical variate, enabling further analysis and visualization.
        
        Args:
            output_path: Directory to save the file
        """
        distribution_records = []
        
        for state in self.permutation_results.keys():
            results_df = self.permutation_results[state]
            n_components = len(results_df)
            n_permutations = results_df['n_permutations'].iloc[0]
            
            # Recompute permuted correlations to get full distribution
            X, Y = self.prepare_matrices(state)
            state_data = self.data[self.data['state'] == state].copy()
            subject_ids_full = state_data['subject'].values
            X_full = state_data[self.physio_measures].values
            Y_full = state_data[self.tet_affective].values
            valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
            subject_ids = subject_ids_full[valid_mask]
            
            # Compute permuted correlations
            for perm_idx in range(n_permutations):
                permuted_corrs = self._fit_permuted_cca(
                    X, Y, subject_ids, n_components, 42 + perm_idx
                )
                
                for i in range(n_components):
                    distribution_records.append({
                        'state': state,
                        'canonical_variate': i + 1,
                        'permutation_id': perm_idx,
                        'permuted_correlation': permuted_corrs[i]
                    })
        
        # Create DataFrame and export
        distribution_df = pd.DataFrame(distribution_records)
        dist_path = output_path / 'cca_permutation_distributions.csv'
        distribution_df.to_csv(dist_path, index=False)
        self.logger.info(f"Exported permutation distributions to {dist_path}")
    
    def compute_redundancy_index(self, state: str) -> pd.DataFrame:
        """
        Compute redundancy index for CCA canonical variates.
        
        The redundancy index quantifies the percentage of variance in one
        variable set explained by the canonical variates from the other set.
        
        Redundancy Index Formula:
        - Redundancy of Y given X = r_c² × R²(Y|U)
        - Redundancy of X given Y = r_c² × R²(X|V)
        
        Where:
        - r_c = canonical correlation for variate pair
        - R²(Y|U) = average R² from regressing each y_j on U_i
        - R²(X|V) = average R² from regressing each x_k on V_i
        - U_i = canonical variate for physio (X @ W_x)
        - V_i = canonical variate for TET (Y @ W_y)
        
        Interpretation:
        - Redundancy > 10%: Meaningful shared variance
        - Redundancy < 5%: Weak relationship, potential overfitting
        - Total redundancy = sum across all canonical variates
        
        Args:
            state: 'RS' or 'DMT'
        
        Returns:
            DataFrame with columns:
            - state: RS or DMT
            - canonical_variate: 1, 2, ...
            - r_canonical: Canonical correlation
            - var_explained_Y_by_U: Average R² of TET variables by physio variate
            - var_explained_X_by_V: Average R² of physio variables by TET variate
            - redundancy_Y_given_X: % variance in TET explained by physio
            - redundancy_X_given_Y: % variance in physio explained by TET
        
        Example:
            >>> analyzer.fit_cca(n_components=2)
            >>> redundancy_df = analyzer.compute_redundancy_index('DMT')
            >>> print(redundancy_df)
        """
        if state not in self.cca_models:
            raise ValueError(
                f"Must fit CCA for {state} state first using fit_cca()"
            )
        
        self.logger.info(f"Computing redundancy index for {state} state")
        
        # Prepare matrices
        X, Y = self.prepare_matrices(state)
        
        # Get CCA model and transform to canonical variates
        cca_model = self.cca_models[state]
        U, V = cca_model.transform(X, Y)
        
        # Get canonical correlations
        canonical_corrs = self.canonical_correlations[state]
        n_components = len(canonical_corrs)
        
        # Compute variance explained for each canonical variate
        results = []
        
        for i in range(n_components):
            # Get canonical correlation
            r_c = canonical_corrs[i]
            
            # Compute variance explained in Y by U_i
            var_explained_Y = self._compute_variance_explained(Y, U[:, i])
            
            # Compute variance explained in X by V_i
            var_explained_X = self._compute_variance_explained(X, V[:, i])
            
            # Compute redundancy indices
            redundancy_Y_given_X = (r_c ** 2) * var_explained_Y
            redundancy_X_given_Y = (r_c ** 2) * var_explained_X
            
            results.append({
                'state': state,
                'canonical_variate': i + 1,
                'r_canonical': r_c,
                'var_explained_Y_by_U': var_explained_Y,
                'var_explained_X_by_V': var_explained_X,
                'redundancy_Y_given_X': redundancy_Y_given_X,
                'redundancy_X_given_Y': redundancy_X_given_Y
            })
        
        # Create DataFrame
        redundancy_df = pd.DataFrame(results)
        
        # Compute total redundancy (sum across all variates)
        total_redundancy_Y = redundancy_df['redundancy_Y_given_X'].sum()
        total_redundancy_X = redundancy_df['redundancy_X_given_Y'].sum()
        
        # Add total row
        total_row = {
            'state': state,
            'canonical_variate': 'Total',
            'r_canonical': np.nan,
            'var_explained_Y_by_U': np.nan,
            'var_explained_X_by_V': np.nan,
            'redundancy_Y_given_X': total_redundancy_Y,
            'redundancy_X_given_Y': total_redundancy_X
        }
        redundancy_df = pd.concat(
            [redundancy_df, pd.DataFrame([total_row])],
            ignore_index=True
        )
        
        self.logger.info(
            f"Redundancy index computed for {state}: "
            f"Total TET variance explained by physio = {total_redundancy_Y:.1%}, "
            f"Total physio variance explained by TET = {total_redundancy_X:.1%}"
        )
        
        return redundancy_df
    
    def _compute_variance_explained(
        self,
        Y: np.ndarray,
        U: np.ndarray
    ) -> float:
        """
        Compute average variance explained in Y by canonical variate U.
        
        For each variable y_j in Y:
        1. Regress y_j on U: y_j = β₀ + β₁ * U + ε
        2. Compute R² = 1 - (SS_residual / SS_total)
        3. Average R² across all variables
        
        Args:
            Y: Variable matrix (n_obs × n_vars)
            U: Canonical variate (n_obs,)
        
        Returns:
            Average R² across all variables in Y
        """
        n_vars = Y.shape[1]
        r_squared_values = []
        
        for j in range(n_vars):
            y_j = Y[:, j]
            
            # Compute correlation between y_j and U
            # R² for simple linear regression = r²
            r = np.corrcoef(y_j, U)[0, 1]
            r_squared = r ** 2
            
            r_squared_values.append(r_squared)
        
        # Average R² across all variables
        avg_r_squared = np.mean(r_squared_values)
        
        return avg_r_squared
    
    def plot_redundancy_indices(
        self,
        output_dir: str
    ) -> str:
        """
        Generate redundancy index visualization.
        
        Creates a grouped bar chart showing redundancy indices for each
        canonical variate pair, with separate panels for RS and DMT states.
        
        Visualization:
        - X-axis: Canonical variates (CV1, CV2)
        - Y-axis: Redundancy index (%)
        - Blue bars: Physio → TET (variance in TET explained by physio)
        - Red bars: TET → Physio (variance in physio explained by TET)
        - Horizontal reference line at 10% (meaningful threshold)
        - Annotations showing exact percentages
        
        Args:
            output_dir: Directory to save figure
        
        Returns:
            Path to saved figure
        
        Raises:
            ValueError: If redundancy indices have not been computed
        
        Example:
            >>> analyzer.fit_cca(n_components=2)
            >>> redundancy_df_rs = analyzer.compute_redundancy_index('RS')
            >>> redundancy_df_dmt = analyzer.compute_redundancy_index('DMT')
            >>> fig_path = analyzer.plot_redundancy_indices('results/tet/physio_correlation')
        """
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute redundancy indices for both states if not already done
        redundancy_data = {}
        
        for state in ['RS', 'DMT']:
            if state in self.cca_models:
                redundancy_df = self.compute_redundancy_index(state)
                # Exclude total row for plotting
                redundancy_data[state] = redundancy_df[
                    redundancy_df['canonical_variate'] != 'Total'
                ].copy()
        
        if not redundancy_data:
            raise ValueError(
                "Must fit CCA models first using fit_cca() before plotting redundancy"
            )
        
        # Create figure with subplots for each state
        n_states = len(redundancy_data)
        fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 6))
        
        if n_states == 1:
            axes = [axes]
        
        for idx, (state, redundancy_df) in enumerate(redundancy_data.items()):
            ax = axes[idx]
            
            # Prepare data for grouped bar chart
            n_variates = len(redundancy_df)
            x_positions = np.arange(n_variates)
            bar_width = 0.35
            
            # Extract redundancy values (convert to percentages)
            redundancy_physio_to_tet = redundancy_df['redundancy_Y_given_X'].values * 100
            redundancy_tet_to_physio = redundancy_df['redundancy_X_given_Y'].values * 100
            
            # Create grouped bars
            bars1 = ax.bar(
                x_positions - bar_width/2,
                redundancy_physio_to_tet,
                bar_width,
                label='Physio → TET',
                color='steelblue',
                alpha=0.8,
                edgecolor='black'
            )
            
            bars2 = ax.bar(
                x_positions + bar_width/2,
                redundancy_tet_to_physio,
                bar_width,
                label='TET → Physio',
                color='coral',
                alpha=0.8,
                edgecolor='black'
            )
            
            # Add horizontal reference line at 10%
            ax.axhline(
                10,
                color='gray',
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label='10% threshold'
            )
            
            # Annotate bars with exact percentages
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.5,
                            f'{height:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=9
                        )
            
            # Set x-axis labels
            variate_labels = [f'CV{int(v)}' for v in redundancy_df['canonical_variate']]
            ax.set_xticks(x_positions)
            ax.set_xticklabels(variate_labels, fontsize=11)
            
            # Labels and title
            ax.set_xlabel('Canonical Variate', fontsize=12)
            ax.set_ylabel('Redundancy Index (%)', fontsize=12)
            ax.set_title(
                f'{state} State',
                fontsize=13,
                fontweight='bold'
            )
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits to accommodate annotations
            max_redundancy = max(
                redundancy_physio_to_tet.max(),
                redundancy_tet_to_physio.max()
            )
            ax.set_ylim(0, max_redundancy * 1.15)
        
        # Overall title
        fig.suptitle(
            'CCA Redundancy Indices: Shared Variance Between Physio and TET',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        
        # Save figure
        fig_path = output_path / 'cca_redundancy_indices.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved redundancy indices plot to {fig_path}")
        
        return str(fig_path)

    # =========================================================================
    # STEPWISE PERMUTATION TESTING (Winkler et al., 2020)
    # =========================================================================
    
    def _generate_all_derangements(self, n: int):
        """
        Generate all derangements (permutations with no fixed points) of n elements.
        
        A derangement is a permutation where no element appears in its original position.
        For n=7, there are exactly 1854 derangements.
        
        Uses itertools to generate all permutations and filters for derangements.
        
        Args:
            n: Number of elements
        
        Yields:
            Tuple representing a derangement (permutation indices)
        
        Example:
            For n=3: yields (1,2,0), (2,0,1) - the only 2 derangements
        """
        from itertools import permutations
        
        original = tuple(range(n))
        for perm in permutations(original):
            # Check if it's a derangement (no fixed points)
            if all(perm[i] != i for i in range(n)):
                yield perm
    
    def _count_derangements(self, n: int) -> int:
        """
        Count the number of derangements of n elements using subfactorial formula.
        
        !n = n! * sum_{k=0}^{n} (-1)^k / k!
        
        For n=7: !7 = 1854
        
        Args:
            n: Number of elements
        
        Returns:
            Number of derangements (subfactorial of n)
        """
        from math import factorial
        
        if n == 0:
            return 1
        if n == 1:
            return 0
        
        # Subfactorial formula: !n = (n-1) * (!(n-1) + !(n-2))
        # Or: !n = n! * sum_{k=0}^{n} (-1)^k / k!
        result = 0
        for k in range(n + 1):
            result += ((-1) ** k) / factorial(k)
        return round(factorial(n) * result)
    
    def _apply_derangement(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        subject_ids: np.ndarray,
        derangement: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a specific derangement to shuffle Y at subject level.
        
        Args:
            X: Physiological matrix (n_obs × p)
            Y: TET affective matrix (n_obs × q)
            subject_ids: Subject identifiers for each observation
            derangement: Tuple of indices representing the derangement
        
        Returns:
            X_perm: Original X (unchanged)
            Y_perm: Shuffled Y according to derangement
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        # Create mapping: subject i gets Y data from subject derangement[i]
        Y_perm = np.zeros_like(Y)
        
        for i, subject in enumerate(unique_subjects):
            # Get indices for this subject's data
            subject_mask = subject_ids == subject
            
            # Get the subject this one is paired with
            paired_idx = derangement[i]
            paired_subject = unique_subjects[paired_idx]
            paired_mask = subject_ids == paired_subject
            
            # Copy paired subject's Y data to this subject's position
            Y_perm[subject_mask] = Y[paired_mask]
        
        return X.copy(), Y_perm
    
    def _compute_theil_residuals(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        subject_ids: np.ndarray,
        rows_to_remove: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute BLUS residuals using Theil's method (Winkler et al., 2020, Section 2.7).
        
        The Theil method computes Best Linear Unbiased residuals with Scalar covariance
        (BLUS) that respect the block structure of the data. Unlike Huh-Jhun, this method
        maintains a one-to-one mapping between original observations and residuals,
        preserving exchangeability within blocks.
        
        Formula (adapted for CCA):
            Q_Z = R_Z @ S' @ (S @ R_Z @ S')^(-1/2)
        
        Where:
            R_Z = I - Z @ (Z'Z)^(-1) @ Z'  (residual-forming matrix)
            S = selection matrix (identity with some rows removed)
            ^(-1/2) = positive definite matrix square root inverse
        
        The BLUS residuals are: Y_tilde = Q_Z' @ Y
        
        Args:
            Y: Left-side matrix (n_obs × q) - TET affective dimensions
            X: Right-side matrix (n_obs × p) - Physiological measures
            Z: Nuisance/confound matrix (n_obs × r) - e.g., subject means
               If None, uses subject indicator matrix for block structure
            subject_ids: Subject identifiers for each observation
            rows_to_remove: Indices of rows to remove for selection matrix S
                           If None, removes one observation per subject
        
        Returns:
            Y_blus: BLUS residuals for Y (n_selected × q)
            X_blus: BLUS residuals for X (n_selected × p)  
            selected_subject_ids: Subject IDs for selected observations
        
        References:
            Theil, H. (1965). The analysis of disturbances in regression analysis.
            Theil, H. (1968). A simplification of the BLUS procedure.
            Magnus, J. R., & Sinha, A. K. (2005). On Theil's errors.
            Winkler et al. (2020). NeuroImage, Section 2.7.
        """
        n_obs = Y.shape[0]
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        # If Z is None, create subject indicator matrix (block structure)
        if Z is None:
            # Create dummy coding for subjects (nuisance = subject means)
            Z = np.zeros((n_obs, n_subjects))
            for i, subj in enumerate(unique_subjects):
                Z[subject_ids == subj, i] = 1
        
        r = Z.shape[1]  # Number of nuisance variables
        
        # Determine rows to remove
        # Strategy: remove one observation per subject to maintain block balance
        if rows_to_remove is None:
            # Remove first observation from each subject
            rows_to_remove = []
            for subj in unique_subjects:
                subj_indices = np.where(subject_ids == subj)[0]
                rows_to_remove.append(subj_indices[0])
            rows_to_remove = np.array(rows_to_remove)
        
        # Ensure we remove at least r rows (rank of Z)
        n_to_remove = max(r, len(rows_to_remove))
        if len(rows_to_remove) < n_to_remove:
            # Add more rows to remove if needed
            all_indices = np.arange(n_obs)
            available = np.setdiff1d(all_indices, rows_to_remove)
            additional = available[:n_to_remove - len(rows_to_remove)]
            rows_to_remove = np.concatenate([rows_to_remove, additional])
        
        rows_to_remove = rows_to_remove[:n_to_remove]
        
        # Create selection matrix S (N' × N where N' = N - n_to_remove)
        rows_to_keep = np.setdiff1d(np.arange(n_obs), rows_to_remove)
        n_selected = len(rows_to_keep)
        
        # S is implemented as index selection rather than explicit matrix
        # S @ A is equivalent to A[rows_to_keep, :]
        
        # Compute residual-forming matrix R_Z = I - Z @ (Z'Z)^(-1) @ Z'
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)
        R_Z = np.eye(n_obs) - Z @ ZtZ_inv @ Z.T
        
        # Compute S @ R_Z @ S' (selected rows and columns of R_Z)
        S_RZ_St = R_Z[np.ix_(rows_to_keep, rows_to_keep)]
        
        # Compute (S @ R_Z @ S')^(-1/2) using eigendecomposition
        # For positive semi-definite matrix, use eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S_RZ_St)
        
        # Handle numerical issues: set small/negative eigenvalues to small positive
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Compute inverse square root: V @ diag(1/sqrt(lambda)) @ V'
        sqrt_inv = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Compute Q_Z = R_Z @ S' @ (S @ R_Z @ S')^(-1/2)
        # R_Z @ S' selects columns of R_Z corresponding to kept rows
        RZ_St = R_Z[:, rows_to_keep]
        Q_Z = RZ_St @ sqrt_inv
        
        # Compute BLUS residuals: Y_tilde = Q_Z' @ Y
        Y_blus = Q_Z.T @ Y
        X_blus = Q_Z.T @ X
        
        # Get subject IDs for selected observations
        # The BLUS residuals correspond to the selected rows
        selected_subject_ids = subject_ids[rows_to_keep]
        
        self.logger.debug(
            f"Theil BLUS: {n_obs} obs -> {n_selected} residuals, "
            f"removed {n_to_remove} rows (r={r} nuisance vars)"
        )
        
        return Y_blus, X_blus, selected_subject_ids
    
    def _compute_theil_residuals_for_blocks(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        subject_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simplified Theil residuals for block-structured data (repeated measures).
        
        For data with repeated measures within subjects, this method:
        1. Removes subject-level means (nuisance)
        2. Maintains one-to-one mapping to preserve block exchangeability
        3. Removes one observation per subject to achieve BLUS properties
        
        This is the recommended approach for CCA with repeated measures
        when permuting at the subject level.
        
        Args:
            Y: Left-side matrix (n_obs × q)
            X: Right-side matrix (n_obs × p)
            subject_ids: Subject identifiers
        
        Returns:
            Y_blus: BLUS residuals for Y
            X_blus: BLUS residuals for X
            block_ids: Block (subject) IDs for each residual observation
        """
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        # Create subject indicator matrix Z
        n_obs = Y.shape[0]
        Z = np.zeros((n_obs, n_subjects))
        for i, subj in enumerate(unique_subjects):
            Z[subject_ids == subj, i] = 1
        
        # For balanced designs, remove first observation from each subject
        rows_to_remove = []
        for subj in unique_subjects:
            subj_indices = np.where(subject_ids == subj)[0]
            rows_to_remove.append(subj_indices[0])
        rows_to_remove = np.array(rows_to_remove)
        
        return self._compute_theil_residuals(
            Y, X, Z, subject_ids, rows_to_remove
        )
    
    def _compute_wilks_lambda(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        start_mode: int = 0
    ) -> float:
        """
        Compute Wilks' Lambda statistic for CCA.
        
        Wilks' Lambda = product of (1 - r_i²) for canonical correlations i >= start_mode
        
        For stepwise testing, we compute Lambda only for modes from start_mode onwards,
        effectively testing the residual association after removing earlier modes.
        
        Args:
            X: Physiological matrix (n_obs × p), standardized
            Y: TET affective matrix (n_obs × q), standardized
            start_mode: Starting mode index (0-based). For mode k, use start_mode=k-1
        
        Returns:
            Wilks' Lambda statistic (smaller = stronger association)
        """
        n_obs = X.shape[0]
        p = X.shape[1]  # Number of physio variables
        q = Y.shape[1]  # Number of TET variables
        
        # Maximum number of canonical correlations
        n_components = min(p, q)
        
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
        
        # Compute Wilks' Lambda for modes from start_mode onwards
        # Lambda = product of (1 - r_i²) for i >= start_mode
        wilks_lambda = np.prod([
            1 - r**2 for r in canonical_corrs[start_mode:]
        ])
        
        return wilks_lambda
    
    def _compute_roys_largest_root(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        mode: int = 0
    ) -> float:
        """
        Compute Roy's Largest Root statistic for a specific CCA mode.
        
        Roy's Largest Root = r_k² / (1 - r_k²) for mode k
        This is the eigenvalue of the canonical correlation matrix.
        
        Args:
            X: Physiological matrix (n_obs × p), standardized
            Y: TET affective matrix (n_obs × q), standardized
            mode: Mode index (0-based) to compute statistic for
        
        Returns:
            Roy's Largest Root statistic (larger = stronger association)
        """
        n_components = min(X.shape[1], Y.shape[1])
        
        if mode >= n_components:
            return 0.0
        
        # Fit CCA
        cca = CCA(n_components=n_components)
        cca.fit(X, Y)
        
        # Transform to canonical variates
        U, V = cca.transform(X, Y)
        
        # Compute canonical correlation for this mode
        r = np.corrcoef(U[:, mode], V[:, mode])[0, 1]
        
        # Roy's Largest Root = r² / (1 - r²)
        if abs(r) >= 1.0:
            return np.inf
        
        roys_root = (r**2) / (1 - r**2)
        
        return roys_root
    
    def _deflate_matrices(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_modes_to_remove: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deflate matrices by removing variance explained by first k modes.
        
        This implements the stepwise approach from Winkler 2020:
        For testing mode k, we first remove the variance explained by modes 1 to k-1.
        
        The correct approach (per Winkler 2020) is to:
        1. Fit CCA to get canonical variates U and V
        2. Regress out U from X and V from Y
        3. Use residuals for testing subsequent modes
        
        Args:
            X: Physiological matrix (n_obs × p)
            Y: TET affective matrix (n_obs × q)
            n_modes_to_remove: Number of modes to deflate (k-1 for testing mode k)
        
        Returns:
            X_deflated: Deflated physiological matrix
            Y_deflated: Deflated TET matrix
        """
        if n_modes_to_remove == 0:
            return X.copy(), Y.copy()
        
        # Fit CCA with modes to remove
        cca = CCA(n_components=n_modes_to_remove)
        cca.fit(X, Y)
        
        # Get canonical variates
        U, V = cca.transform(X, Y)
        
        # Deflate X by regressing out U (canonical variates from X side)
        # X_deflated = X - U @ (U.T @ U)^-1 @ U.T @ X
        # Since U columns are orthogonal: X_deflated = X - U @ U.T @ X / n
        # More robust: use least squares projection
        
        # For each column of X, regress out U
        X_deflated = np.zeros_like(X)
        for j in range(X.shape[1]):
            # Regress X[:, j] on U
            coeffs, _, _, _ = np.linalg.lstsq(U, X[:, j], rcond=None)
            X_deflated[:, j] = X[:, j] - U @ coeffs
        
        # For each column of Y, regress out V
        Y_deflated = np.zeros_like(Y)
        for j in range(Y.shape[1]):
            # Regress Y[:, j] on V
            coeffs, _, _, _ = np.linalg.lstsq(V, Y[:, j], rcond=None)
            Y_deflated[:, j] = Y[:, j] - V @ coeffs
        
        return X_deflated, Y_deflated
    
    def stepwise_permutation_test(
        self,
        state: str,
        n_permutations: int = 5000,
        statistic: str = 'wilks',
        permutation_type: str = 'row',
        use_theil: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Perform Stepwise Permutation Testing for CCA (Winkler et al., 2020).
        
        This method implements Algorithm 1 from Winkler et al. (2020) for
        rigorous significance testing of canonical correlations with FWER control.
        
        Key features:
        - Tests each mode sequentially after deflating previous modes
        - Supports both row-wise and subject-level permutation
        - Uses Theil's method for BLUS residuals (Section 2.7) to respect block structure
        - Applies cumulative max correction for FWER control
        
        Algorithm (with Theil method for subject-level permutation):
        1. Compute Theil BLUS residuals to remove nuisance (subject means)
           while preserving block exchangeability
        2. Compute observed statistics on BLUS residuals
        3. For each derangement (exact enumeration for n=7):
           a. Permute BLUS residuals at subject level
           b. For each mode k = 1 to K:
              - Deflate matrices by removing modes 1 to k-1
              - Compute test statistic on deflated permuted data
        4. Compute raw p-values: p_k = (count + 1) / (n_derangements + 1)
        5. Apply cumulative max correction: p_corrected_k = max(p_1, ..., p_k)
        
        Args:
            state: 'RS' or 'DMT'
            n_permutations: Number of permutation iterations (default: 5000)
                For subject-level, uses all exact derangements instead
            statistic: Test statistic to use ('wilks' or 'roys')
                - 'wilks': Wilks' Lambda (tests all remaining modes jointly)
                - 'roys': Roy's Largest Root (tests single mode)
            permutation_type: Type of permutation ('row' or 'subject')
                - 'row': Permute rows of Y independently (standard approach)
                - 'subject': Permute subject pairings with Theil BLUS residuals
            use_theil: Whether to use Theil's method for BLUS residuals (default: True)
                Only applies when permutation_type='subject'
                Recommended for repeated measures / block designs
            random_state: Random seed for reproducibility
        
        Returns:
            DataFrame with columns:
            - mode: Canonical mode (1, 2, ...)
            - observed_r: Observed canonical correlation
            - observed_statistic: Observed test statistic
            - raw_p_value: Uncorrected permutation p-value
            - fwer_p_value: FWER-corrected p-value (cumulative max)
            - significant: Boolean indicating significance at alpha=0.05
            - n_permutations: Number of permutations performed
            - method: 'theil' or 'standard'
        
        References:
            Winkler, A. M., et al. (2020). Permutation inference for canonical 
            correlation analysis. NeuroImage, 220, 117065.
            Theil, H. (1965). The analysis of disturbances in regression analysis.
        """
        self.logger.info(
            f"Starting Stepwise Permutation Test for {state} state "
            f"(n_permutations={n_permutations}, statistic={statistic})"
        )
        
        # Prepare matrices
        X, Y = self.prepare_matrices(state)
        n_obs = X.shape[0]
        p = X.shape[1]  # 3 physio measures
        q = Y.shape[1]  # 6 TET dimensions
        n_modes = min(p, q)  # Maximum 3 modes
        
        # Get subject IDs for subject-level permutation
        state_data = self.data[self.data['state'] == state].copy()
        subject_ids_full = state_data['subject'].values
        X_full = state_data[self.physio_measures].values
        Y_full = state_data[self.tet_affective].values
        valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
        subject_ids = subject_ids_full[valid_mask]
        
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        
        self.logger.info(f"  Data: {n_obs} observations, {n_subjects} subjects")
        self.logger.info(f"  Testing {n_modes} canonical modes")
        
        # =====================================================================
        # STEP 0: Apply Theil's method for BLUS residuals (if requested)
        # =====================================================================
        method_used = 'standard'
        
        if permutation_type == 'subject' and use_theil:
            self.logger.info("  Applying Theil's method for BLUS residuals...")
            self.logger.info("    (Removes subject means while preserving block exchangeability)")
            
            Y_blus, X_blus, blus_subject_ids = self._compute_theil_residuals_for_blocks(
                Y, X, subject_ids
            )
            
            # Update data for permutation testing
            X_for_perm = X_blus
            Y_for_perm = Y_blus
            subject_ids_for_perm = blus_subject_ids
            n_obs_perm = X_blus.shape[0]
            method_used = 'theil'
            
            self.logger.info(
                f"    BLUS residuals: {n_obs} -> {n_obs_perm} observations "
                f"({n_obs - n_obs_perm} removed for BLUS)"
            )
        else:
            X_for_perm = X
            Y_for_perm = Y
            subject_ids_for_perm = subject_ids
            n_obs_perm = n_obs
        
        # =====================================================================
        # STEP 1: Compute observed statistics for all modes
        # =====================================================================
        self.logger.info("  Computing observed statistics...")
        
        # Fit full CCA to get observed canonical correlations
        # Use BLUS residuals if Theil method was applied
        cca_full = CCA(n_components=n_modes)
        cca_full.fit(X_for_perm, Y_for_perm)
        U_obs, V_obs = cca_full.transform(X_for_perm, Y_for_perm)
        
        observed_r = np.array([
            np.corrcoef(U_obs[:, k], V_obs[:, k])[0, 1]
            for k in range(n_modes)
        ])
        
        # Compute observed test statistics for each mode (stepwise)
        observed_stats = np.zeros(n_modes)
        
        for k in range(n_modes):
            # Deflate matrices by removing modes 0 to k-1
            X_deflated, Y_deflated = self._deflate_matrices(
                X_for_perm, Y_for_perm, n_modes_to_remove=k
            )
            
            if statistic == 'wilks':
                observed_stats[k] = self._compute_wilks_lambda(
                    X_deflated, Y_deflated, start_mode=0
                )
            else:  # roys
                observed_stats[k] = self._compute_roys_largest_root(
                    X_deflated, Y_deflated, mode=0
                )
        
        self.logger.info(f"  Observed canonical correlations: {observed_r}")
        self.logger.info(f"  Observed {statistic} statistics: {observed_stats}")
        
        # =====================================================================
        # STEP 2: Permutation loop
        # =====================================================================
        
        if permutation_type == 'subject':
            # Use ALL exact derangements for subject-level permutation
            n_derangements = self._count_derangements(n_subjects)
            self.logger.info(
                f"  Using ALL {n_derangements} exact derangements for {n_subjects} subjects"
            )
            
            # Generate all derangements
            all_derangements = list(self._generate_all_derangements(n_subjects))
            actual_n_permutations = len(all_derangements)
            
            # Storage for permuted statistics
            permuted_stats = np.zeros((actual_n_permutations, n_modes))
            
            for j, derangement in enumerate(all_derangements):
                # Apply derangement to BLUS residuals (or original data)
                X_perm, Y_perm = self._apply_derangement(
                    X_for_perm, Y_for_perm, subject_ids_for_perm, derangement
                )
                
                # Stepwise: compute statistic for each mode on permuted data
                for k in range(n_modes):
                    X_perm_deflated, Y_perm_deflated = self._deflate_matrices(
                        X_perm, Y_perm, n_modes_to_remove=k
                    )
                    
                    if statistic == 'wilks':
                        permuted_stats[j, k] = self._compute_wilks_lambda(
                            X_perm_deflated, Y_perm_deflated, start_mode=0
                        )
                    else:  # roys
                        permuted_stats[j, k] = self._compute_roys_largest_root(
                            X_perm_deflated, Y_perm_deflated, mode=0
                        )
                
                # Progress logging
                if (j + 1) % 200 == 0:
                    self.logger.info(
                        f"    Completed {j + 1}/{actual_n_permutations} derangements"
                    )
            
            # Update n_permutations to actual count
            n_permutations = actual_n_permutations
            
        else:  # row-wise permutation
            self.logger.info(f"  Running {n_permutations} row-wise permutations...")
            
            # Storage for permuted statistics
            permuted_stats = np.zeros((n_permutations, n_modes))
            
            np.random.seed(random_state)
            
            for j in range(n_permutations):
                # Standard row permutation
                perm_indices = np.random.permutation(n_obs_perm)
                X_perm = X_for_perm.copy()
                Y_perm = Y_for_perm[perm_indices]
                
                # Stepwise: compute statistic for each mode on permuted data
                for k in range(n_modes):
                    # Deflate permuted matrices
                    X_perm_deflated, Y_perm_deflated = self._deflate_matrices(
                        X_perm, Y_perm, n_modes_to_remove=k
                    )
                    
                    if statistic == 'wilks':
                        permuted_stats[j, k] = self._compute_wilks_lambda(
                            X_perm_deflated, Y_perm_deflated, start_mode=0
                        )
                    else:  # roys
                        permuted_stats[j, k] = self._compute_roys_largest_root(
                            X_perm_deflated, Y_perm_deflated, mode=0
                        )
                
                # Progress logging
                if (j + 1) % 500 == 0:
                    self.logger.info(f"    Completed {j + 1}/{n_permutations} permutations")
        
        # =====================================================================
        # STEP 3: Compute raw p-values
        # =====================================================================
        self.logger.info("  Computing raw p-values...")
        
        raw_p_values = np.zeros(n_modes)
        
        for k in range(n_modes):
            if statistic == 'wilks':
                # For Wilks' Lambda, smaller is more extreme (reject if perm <= obs)
                count = np.sum(permuted_stats[:, k] <= observed_stats[k])
            else:  # roys
                # For Roy's Root, larger is more extreme (reject if perm >= obs)
                count = np.sum(permuted_stats[:, k] >= observed_stats[k])
            
            raw_p_values[k] = (count + 1) / (n_permutations + 1)
        
        self.logger.info(f"  Raw p-values: {raw_p_values}")
        
        # =====================================================================
        # STEP 4: Apply cumulative max correction (FWER control)
        # =====================================================================
        self.logger.info("  Applying FWER correction (cumulative max)...")
        
        fwer_p_values = np.zeros(n_modes)
        
        for k in range(n_modes):
            # p_corrected_k = max(p_1, ..., p_k)
            fwer_p_values[k] = np.max(raw_p_values[:k + 1])
        
        self.logger.info(f"  FWER-corrected p-values: {fwer_p_values}")
        
        # =====================================================================
        # STEP 5: Create results DataFrame
        # =====================================================================
        results = []
        
        for k in range(n_modes):
            results.append({
                'state': state,
                'mode': k + 1,
                'observed_r': observed_r[k],
                'observed_statistic': observed_stats[k],
                'statistic_type': statistic,
                'permutation_type': permutation_type,
                'method': method_used,
                'raw_p_value': raw_p_values[k],
                'fwer_p_value': fwer_p_values[k],
                'significant': fwer_p_values[k] < 0.05,
                'n_permutations': n_permutations,
                'n_obs_tested': n_obs_perm
            })
        
        results_df = pd.DataFrame(results)
        
        # Store results
        if not hasattr(self, 'stepwise_permutation_results'):
            self.stepwise_permutation_results = {}
        self.stepwise_permutation_results[state] = results_df
        
        self.logger.info(
            f"Stepwise permutation test complete for {state}. "
            f"Significant modes: {results_df[results_df['significant']]['mode'].tolist()}"
        )
        
        return results_df
    
    def run_stepwise_permutation_both_states(
        self,
        n_permutations: int = 5000,
        statistic: str = 'wilks',
        permutation_type: str = 'row',
        use_theil: bool = True,
        random_state: int = 42,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run stepwise permutation test for both RS and DMT states.
        
        Args:
            n_permutations: Number of permutation iterations
            statistic: Test statistic ('wilks' or 'roys')
            permutation_type: Type of permutation ('row' or 'subject')
            use_theil: Whether to use Theil's method for BLUS residuals
            random_state: Random seed
            output_dir: Optional directory to save results
        
        Returns:
            Combined DataFrame with results for both states
        """
        results_list = []
        
        for state in ['RS', 'DMT']:
            try:
                state_results = self.stepwise_permutation_test(
                    state=state,
                    n_permutations=n_permutations,
                    statistic=statistic,
                    permutation_type=permutation_type,
                    use_theil=use_theil,
                    random_state=random_state
                )
                results_list.append(state_results)
            except Exception as e:
                self.logger.error(f"Failed stepwise permutation for {state}: {e}")
                continue
        
        if not results_list:
            raise ValueError("Stepwise permutation failed for all states")
        
        combined_df = pd.concat(results_list, ignore_index=True)
        
        # Export if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            csv_path = output_path / 'cca_stepwise_permutation_results.csv'
            combined_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved stepwise permutation results to {csv_path}")
        
        return combined_df
    
    def plot_stepwise_permutation_results(
        self,
        output_dir: str,
        alpha: float = 0.05
    ) -> str:
        """
        Generate visualization of stepwise permutation test results.
        
        Creates a bar plot showing raw and FWER-corrected p-values for each mode,
        with significance threshold indicated.
        
        Args:
            output_dir: Directory to save figure
            alpha: Significance threshold (default: 0.05)
        
        Returns:
            Path to saved figure
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'stepwise_permutation_results'):
            raise ValueError(
                "Must run stepwise_permutation_test() first"
            )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine results from both states
        results_list = []
        for state, df in self.stepwise_permutation_results.items():
            results_list.append(df)
        
        combined_df = pd.concat(results_list, ignore_index=True)
        
        # Create figure
        states = combined_df['state'].unique()
        n_states = len(states)
        
        fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5))
        if n_states == 1:
            axes = [axes]
        
        for idx, state in enumerate(states):
            ax = axes[idx]
            state_df = combined_df[combined_df['state'] == state]
            
            modes = state_df['mode'].values
            raw_p = state_df['raw_p_value'].values
            fwer_p = state_df['fwer_p_value'].values
            
            x = np.arange(len(modes))
            width = 0.35
            
            # Bar plots
            bars1 = ax.bar(x - width/2, raw_p, width, label='Raw p-value', 
                          color='steelblue', alpha=0.8)
            bars2 = ax.bar(x + width/2, fwer_p, width, label='FWER p-value', 
                          color='coral', alpha=0.8)
            
            # Significance threshold
            ax.axhline(alpha, color='red', linestyle='--', linewidth=2, 
                      label=f'α = {alpha}')
            
            # Annotate significant modes
            for i, (raw, fwer) in enumerate(zip(raw_p, fwer_p)):
                if fwer < alpha:
                    ax.annotate('*', xy=(x[i] + width/2, fwer), 
                               ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # Labels
            ax.set_xlabel('Canonical Mode', fontsize=12)
            ax.set_ylabel('p-value', fontsize=12)
            ax.set_title(f'{state} State\nStepwise Permutation Test', 
                        fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Mode {m}' for m in modes])
            ax.legend(fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = output_path / 'cca_stepwise_permutation_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved stepwise permutation plot to {fig_path}")
        
        return str(fig_path)
