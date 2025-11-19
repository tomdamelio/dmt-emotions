"""
Independent Component Analysis (ICA) for TET data.

This module implements group-level ICA on z-scored TET dimensions to identify
statistically independent sources of experiential variation that may not be
captured by variance-based PCA.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.decomposition import FastICA
from pathlib import Path


class TETICAAnalyzer:
    """
    Perform group-level Independent Component Analysis on TET data.
    
    ICA identifies statistically independent sources based on non-Gaussianity,
    complementary to PCA which identifies sources based on variance. This may
    reveal latent experiential patterns masked by the dominant variance structure.
    
    Example:
        >>> analyzer = TETICAAnalyzer(
        ...     data=preprocessed_df,
        ...     dimensions=['pleasantness_z', 'anxiety_z', ...],
        ...     n_components=3,
        ...     pca_scores=pca_scores_df,
        ...     random_state=42
        ... )
        >>> ica_model = analyzer.fit_ica()
        >>> ic_scores = analyzer.transform_data()
        >>> mixing_matrix = analyzer.get_mixing_matrix()
        >>> correlations = analyzer.compute_pca_correlation()
        >>> analyzer.export_results('results/tet/ica')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        dimensions: List[str],
        n_components: int,
        pca_scores: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ):
        """
        Initialize ICA analyzer.
        
        Args:
            data: Preprocessed TET data with z-scored dimensions
            dimensions: List of z-scored dimension column names
            n_components: Number of components (same as PCA for comparison)
            pca_scores: Optional PC scores for correlation analysis
            random_state: Random seed for reproducibility
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.n_components = n_components
        self.pca_scores = pca_scores
        self.random_state = random_state
        
        # Storage for results
        self.ica_model = None  # Single ICA model for all data
        self.ic_scores = None  # DataFrame with IC scores
        self.mixing_matrix_df = None
        self.pca_correlation_df = None
    
    def fit_ica(self) -> FastICA:
        """
        Fit group-level ICA on z-scored dimensions across all subjects.
        
        Process:
        1. Extract z-scored dimension matrix from all data
           Shape: (n_total_timepoints × n_dimensions) = (16200 × 15)
        2. Fit FastICA with specified n_components
        3. Store fitted model and mixing matrix
        
        Algorithm: FastICA
        - Maximizes non-Gaussianity of sources
        - Uses fixed-point iteration
        - Converges to statistically independent components
        
        Returns:
            Fitted ICA model
        """
        # Extract z-scored dimension matrix
        X = self.data[self.dimensions].values
        
        # Fit FastICA
        self.ica_model = FastICA(
            n_components=self.n_components,
            algorithm='parallel',
            whiten='unit-variance',
            max_iter=1000,
            tol=1e-4,
            random_state=self.random_state
        )
        
        self.ica_model.fit(X)
        
        return self.ica_model
    
    def transform_data(self) -> pd.DataFrame:
        """
        Transform z-scored dimensions to IC scores for all data.
        
        Process:
        1. Extract z-scored dimension matrix from all data
        2. Transform using fitted ICA model: IC_scores = (X - mean) @ W.T
        3. Create DataFrame with IC scores
        4. Add metadata columns (subject, session_id, state, dose, t_bin, t_sec)
        5. Return complete DataFrame
        
        Returns:
            DataFrame with columns:
            - subject, session_id, state, dose, t_bin, t_sec
            - IC1, IC2, ..., ICn (one column per component)
        """
        if self.ica_model is None:
            raise ValueError("ICA model not fitted. Call fit_ica() first.")
        
        # Extract z-scored dimension matrix
        X = self.data[self.dimensions].values
        
        # Transform to IC scores
        ic_scores_array = self.ica_model.transform(X)
        
        # Create DataFrame with IC scores
        ic_columns = [f'IC{i+1}' for i in range(self.n_components)]
        ic_scores_df = pd.DataFrame(ic_scores_array, columns=ic_columns)
        
        # Add metadata columns
        metadata_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
        for col in metadata_cols:
            if col in self.data.columns:
                ic_scores_df[col] = self.data[col].values
        
        # Reorder columns: metadata first, then IC scores
        ordered_cols = [col for col in metadata_cols if col in ic_scores_df.columns] + ic_columns
        ic_scores_df = ic_scores_df[ordered_cols]
        
        self.ic_scores = ic_scores_df
        return self.ic_scores
    
    def get_mixing_matrix(self) -> pd.DataFrame:
        """
        Extract ICA mixing matrix from group-level ICA.
        
        The mixing matrix A shows how each independent component
        contributes to each observed dimension:
            X = A @ S
        where X = observed dimensions, S = independent components
        
        Mixing matrix coefficients are analogous to PCA loadings
        but represent linear combinations that produce independent sources.
        
        Returns:
            DataFrame with columns:
            - component: Component name (IC1, IC2, ...)
            - dimension: Original dimension name
            - mixing_coef: Mixing coefficient (weight)
        """
        if self.ica_model is None:
            raise ValueError("ICA model not fitted. Call fit_ica() first.")
        
        # Extract mixing matrix: shape (n_dimensions × n_components) = (15 × n_components)
        mixing_matrix = self.ica_model.mixing_
        
        # Convert to long format DataFrame
        rows = []
        for i in range(self.n_components):
            component_name = f'IC{i+1}'
            for j, dimension in enumerate(self.dimensions):
                rows.append({
                    'component': component_name,
                    'dimension': dimension,
                    'mixing_coef': mixing_matrix[j, i]
                })
        
        self.mixing_matrix_df = pd.DataFrame(rows)
        return self.mixing_matrix_df
    
    def compute_pca_correlation(self) -> pd.DataFrame:
        """
        Compute correlations between ICA component scores and PCA component scores.
        
        Purpose:
        - Assess overlap vs complementarity between ICA and PCA
        - High correlation (|r| > 0.7): ICA component similar to PC
        - Low correlation (|r| < 0.3): ICA reveals distinct structure
        
        Returns:
            DataFrame with columns:
            - ic_component: ICA component name (IC1, IC2, ...)
            - pc_component: PCA component name (PC1, PC2, ...)
            - correlation: Pearson correlation coefficient
            - abs_correlation: Absolute correlation (for sorting)
        """
        if self.ic_scores is None:
            raise ValueError("IC scores not computed. Call transform_data() first.")
        
        if self.pca_scores is None:
            raise ValueError("PCA scores not provided. Pass pca_scores to __init__().")
        
        # Get IC column names
        ic_columns = [f'IC{i+1}' for i in range(self.n_components)]
        
        # Get PC column names from pca_scores
        pc_columns = [col for col in self.pca_scores.columns if col.startswith('PC')]
        
        # Merge IC and PC scores on metadata columns for alignment
        metadata_cols = ['subject', 'session_id', 't_bin']
        merged = self.ic_scores.merge(
            self.pca_scores,
            on=metadata_cols,
            how='inner'
        )
        
        # Compute correlations between each IC and each PC
        rows = []
        for ic_col in ic_columns:
            for pc_col in pc_columns:
                correlation = merged[ic_col].corr(merged[pc_col])
                rows.append({
                    'ic_component': ic_col,
                    'pc_component': pc_col,
                    'correlation': correlation,
                    'abs_correlation': abs(correlation)
                })
        
        self.pca_correlation_df = pd.DataFrame(rows)
        return self.pca_correlation_df
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export ICA results to CSV files.
        
        Creates:
        - ica_mixing_matrix.csv: Mixing coefficients for all components
        - ica_scores.csv: IC scores for all time points
        - ica_pca_correlation.csv: Correlations between IC and PC scores
        
        Args:
            output_dir: Directory to save output files
        
        Returns:
            Dict mapping file types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Export mixing matrix
        if self.mixing_matrix_df is not None:
            mixing_path = output_path / 'ica_mixing_matrix.csv'
            self.mixing_matrix_df.to_csv(mixing_path, index=False)
            output_files['mixing_matrix'] = str(mixing_path)
        
        # Export IC scores
        if self.ic_scores is not None:
            scores_path = output_path / 'ica_scores.csv'
            self.ic_scores.to_csv(scores_path, index=False)
            output_files['scores'] = str(scores_path)
        
        # Export PCA correlations
        if self.pca_correlation_df is not None:
            corr_path = output_path / 'ica_pca_correlation.csv'
            self.pca_correlation_df.to_csv(corr_path, index=False)
            output_files['pca_correlation'] = str(corr_path)
        
        return output_files
