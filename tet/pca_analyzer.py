# -*- coding: utf-8 -*-
"""
TET PCA Analyzer Module

This module provides Principal Component Analysis (PCA) functionality for TET data
to identify principal modes of experiential variation.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETPCAAnalyzer:
    """
    Performs group-level PCA on z-scored TET dimensions.
    
    This class performs a single group-level PCA on z-scored dimensions across
    all subjects and time points to identify principal modes of experiential
    variation. The PCA is fit on all data simultaneously, allowing direct
    comparison of PC scores across subjects.
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data with z-scored dimensions
        dimensions (List[str]): List of z-scored dimension column names
        variance_threshold (float): Target cumulative variance (default: 0.75)
        pca_model (PCA): Fitted sklearn PCA model
        n_components (int): Number of components retained
        pc_scores (pd.DataFrame): DataFrame with PC scores and metadata
        loadings_df (pd.DataFrame): Component loadings in long format
        variance_df (pd.DataFrame): Variance explained by each component
    
    Example:
        >>> import pandas as pd
        >>> from tet.pca_analyzer import TETPCAAnalyzer
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
        >>> 
        >>> # Get z-scored dimension columns
        >>> z_dims = [col for col in data.columns if col.endswith('_z') 
        ...           and col not in ['valence_index_z']]
        >>> 
        >>> # Initialize and fit PCA
        >>> analyzer = TETPCAAnalyzer(data, z_dims, variance_threshold=0.75)
        >>> analyzer.fit_pca()
        >>> analyzer.transform_data()
        >>> analyzer.get_loadings()
        >>> analyzer.get_variance_explained()
        >>> 
        >>> # Export results
        >>> analyzer.export_results('results/tet/pca')
    """
    
    def __init__(self, data: pd.DataFrame, dimensions: List[str], 
                 variance_threshold: float = 0.75, n_components: int = None):
        """
        Initialize PCA analyzer.
        
        Args:
            data: Preprocessed TET data with z-scored dimensions
            dimensions: List of z-scored dimension column names (e.g., ['pleasantness_z', ...])
            variance_threshold: Target cumulative variance (default: 0.75 = 75%)
            n_components: Fixed number of components to retain (default: None, use variance_threshold)
        """
        self.data = data.copy()
        self.dimensions = dimensions
        self.variance_threshold = variance_threshold
        self.n_components_fixed = n_components  # Fixed number if specified
        self.pca_model = None  # Single PCA model for all data
        self.n_components = None  # Number of components retained
        self.pc_scores = None  # DataFrame with PC scores
        self.loadings_df = None
        self.variance_df = None
        self.group_mean = None  # Group-level mean for standardization
        self.group_std = None  # Group-level std for standardization
        
        if n_components is not None:
            logger.info(f"Initialized TETPCAAnalyzer with {len(data)} observations, "
                       f"{len(dimensions)} dimensions, n_components={n_components} (fixed)")
        else:
            logger.info(f"Initialized TETPCAAnalyzer with {len(data)} observations, "
                       f"{len(dimensions)} dimensions, variance_threshold={variance_threshold}")
    
    def fit_pca(self) -> PCA:
        """
        Fit single group-level PCA on z-scored dimensions across all subjects.
        
        Process:
        1. Extract z-scored dimension matrix from all data
           Shape: (n_total_timepoints × n_dimensions) = (16200 × 15)
        2. Fit initial PCA with n_components=None to determine variance structure
        3. Calculate cumulative variance explained
        4. Determine n_components where cumulative variance ≥ variance_threshold
        5. Ensure at least 2 components (PC1, PC2) are retained
        6. Refit PCA with selected n_components
        7. Store fitted model and n_components
        
        Returns:
            Fitted PCA model
        """
        logger.info("Fitting group-level PCA...")
        
        # Extract z-scored dimension matrix (already z-scored within subjects)
        X = self.data[self.dimensions].values
        logger.info(f"Input matrix shape: {X.shape}")
        
        # Check for missing values
        if np.any(np.isnan(X)):
            n_missing = np.sum(np.isnan(X))
            logger.warning(f"Found {n_missing} missing values, dropping rows with NaN")
            valid_mask = ~np.any(np.isnan(X), axis=1)
            X = X[valid_mask]
            logger.info(f"Matrix shape after dropping NaN: {X.shape}")
        
        # Apply second z-scoring at group level (across all subjects)
        # This ensures each dimension has mean=0 and std=1 across the entire dataset
        # Making loadings more interpretable across dimensions
        logger.info("Applying group-level z-scoring (across subjects)...")
        self.group_mean = np.mean(X, axis=0)
        self.group_std = np.std(X, axis=0, ddof=1)
        X = (X - self.group_mean) / self.group_std
        logger.info(f"  Group-level standardization complete")
        logger.info(f"  Mean per dimension after standardization: {np.mean(X, axis=0).round(6)}")  # Should be ~0
        logger.info(f"  Std per dimension after standardization: {np.std(X, axis=0, ddof=1).round(6)}")  # Should be ~1
        
        # Determine number of components
        if self.n_components_fixed is not None:
            # Use fixed number of components
            n_components = self.n_components_fixed
            logger.info(f"Using fixed n_components: {n_components}")
            
            # Fit PCA to get variance info
            pca_full = PCA(n_components=None)
            pca_full.fit(X)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            logger.info(f"Cumulative variance explained with {n_components} components: {cumsum_var[n_components-1]:.2%}")
        else:
            # Fit initial PCA with all components to determine variance structure
            pca_full = PCA(n_components=None)
            pca_full.fit(X)
            
            # Calculate cumulative variance explained
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find n_components for target variance
            n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
            n_components = max(2, n_components)  # Ensure at least PC1 and PC2
            
            # Log variance information
            logger.info(f"Variance threshold: {self.variance_threshold:.2%}")
            logger.info(f"Selected n_components: {n_components}")
            logger.info(f"Cumulative variance explained: {cumsum_var[n_components-1]:.2%}")
        
        # Refit with selected n_components for efficiency
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(X)
        self.n_components = n_components
        
        logger.info(f"PCA fitting complete: {n_components} components retained")
        
        return self.pca_model
    
    def transform_data(self) -> pd.DataFrame:
        """
        Transform z-scored dimensions to PC scores for all data.
        
        Process:
        1. Extract z-scored dimension matrix from all data
        2. Transform using fitted PCA model: PC_scores = X @ components.T
        3. Create DataFrame with PC scores
        4. Add metadata columns (subject, session_id, state, dose, t_bin, t_sec)
        5. Store as self.pc_scores
        
        Returns:
            DataFrame with columns:
            - subject, session_id, state, dose, t_bin, t_sec
            - PC1, PC2, ..., PCn (one column per retained component)
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        
        logger.info("Transforming data to PC scores...")
        
        # Extract z-scored dimension matrix
        X = self.data[self.dimensions].values
        
        # Handle missing values (use same mask as fit)
        valid_mask = ~np.any(np.isnan(X), axis=1)
        
        # Apply group-level standardization using saved parameters
        X[valid_mask] = (X[valid_mask] - self.group_mean) / self.group_std
        
        # Transform using fitted PCA model
        pc_scores_array = self.pca_model.transform(X[valid_mask])
        
        # Create DataFrame with PC scores
        pc_columns = [f'PC{i+1}' for i in range(self.n_components)]
        pc_scores_df = pd.DataFrame(pc_scores_array, columns=pc_columns)
        
        # Add metadata columns
        metadata_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
        metadata = self.data.loc[valid_mask, metadata_cols].reset_index(drop=True)
        
        # Combine metadata and PC scores
        self.pc_scores = pd.concat([metadata, pc_scores_df], axis=1)
        
        logger.info(f"Transformation complete: {len(self.pc_scores)} observations, "
                   f"{self.n_components} components")
        
        return self.pc_scores
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Extract PCA loadings (component weights) from group-level PCA.
        
        Loadings represent the contribution of each original dimension
        to each principal component. Since this is group-level PCA,
        loadings are the same for all subjects.
        
        Returns:
            DataFrame with columns:
            - component: Component name (PC1, PC2, ...)
            - dimension: Original dimension name
            - loading: Loading value (weight)
        
        Format (long format):
            component | dimension           | loading
            ----------|---------------------|--------
            PC1       | pleasantness_z      | 0.45
            PC1       | anxiety_z           | -0.32
            PC1       | complex_imagery_z   | 0.28
            PC2       | pleasantness_z      | 0.12
            PC2       | anxiety_z           | 0.67
            ...
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        
        logger.info("Extracting PCA loadings...")
        
        # Extract components matrix (n_components × n_dimensions)
        components = self.pca_model.components_
        
        # Convert to long format
        loadings_list = []
        for i in range(self.n_components):
            component_name = f'PC{i+1}'
            for j, dimension in enumerate(self.dimensions):
                loadings_list.append({
                    'component': component_name,
                    'dimension': dimension,
                    'loading': components[i, j]
                })
        
        self.loadings_df = pd.DataFrame(loadings_list)
        
        logger.info(f"Extracted loadings: {len(self.loadings_df)} rows "
                   f"({self.n_components} components × {len(self.dimensions)} dimensions)")
        
        return self.loadings_df
    
    def get_variance_explained(self) -> pd.DataFrame:
        """
        Extract variance explained by each component from group-level PCA.
        
        Returns:
            DataFrame with columns:
            - component: Component name (PC1, PC2, ...)
            - variance_explained: Proportion of variance (0-1)
            - cumulative_variance: Cumulative proportion (0-1)
        
        Example:
            component | variance_explained | cumulative_variance
            ----------|--------------------|--------------------
            PC1       | 0.35               | 0.35
            PC2       | 0.22               | 0.57
            PC3       | 0.18               | 0.75
        """
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call fit_pca() first.")
        
        logger.info("Extracting variance explained...")
        
        # Extract variance explained ratios
        variance_ratios = self.pca_model.explained_variance_ratio_
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(variance_ratios)
        
        # Create DataFrame
        variance_data = []
        for i in range(self.n_components):
            variance_data.append({
                'component': f'PC{i+1}',
                'variance_explained': variance_ratios[i],
                'cumulative_variance': cumulative_variance[i]
            })
        
        self.variance_df = pd.DataFrame(variance_data)
        
        logger.info(f"Variance explained: PC1={variance_ratios[0]:.2%}, "
                   f"Total={cumulative_variance[-1]:.2%}")
        
        return self.variance_df
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export PCA results to CSV files.
        
        Creates:
        - pca_loadings.csv: Component loadings for all subjects
        - pca_variance_explained.csv: Variance explained by each component
        - pca_scores.csv: PC scores for all time points
        
        Args:
            output_dir: Directory to save output files
        
        Returns:
            Dict mapping file types to file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = {}
        
        # Export loadings
        if self.loadings_df is not None:
            loadings_path = os.path.join(output_dir, 'pca_loadings.csv')
            self.loadings_df.to_csv(loadings_path, index=False)
            output_paths['loadings'] = loadings_path
            logger.info(f"Exported loadings to {loadings_path}")
        else:
            logger.warning("Loadings not computed, skipping export")
        
        # Export variance explained
        if self.variance_df is not None:
            variance_path = os.path.join(output_dir, 'pca_variance_explained.csv')
            self.variance_df.to_csv(variance_path, index=False)
            output_paths['variance'] = variance_path
            logger.info(f"Exported variance explained to {variance_path}")
        else:
            logger.warning("Variance not computed, skipping export")
        
        # Export PC scores
        if self.pc_scores is not None:
            scores_path = os.path.join(output_dir, 'pca_scores.csv')
            self.pc_scores.to_csv(scores_path, index=False)
            output_paths['scores'] = scores_path
            logger.info(f"Exported PC scores to {scores_path}")
        else:
            logger.warning("PC scores not computed, skipping export")
        
        logger.info(f"Export complete: {len(output_paths)} files saved to {output_dir}")
        
        return output_paths
