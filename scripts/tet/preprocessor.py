# -*- coding: utf-8 -*-
"""
TET Preprocessor Module

This module provides preprocessing functionality for TET (Temporal Experience Tracking)
data including session trimming, within-subject standardization, and composite index creation.
"""

import pandas as pd
import numpy as np
from typing import List
import logging
import sys
import os

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETPreprocessor:
    """
    Preprocesses TET (Temporal Experience Tracking) data.
    
    This class handles preprocessing of TET data including:
    - Session trimming to analysis windows (0-10 min RS, 0-20 min DMT)
    - Global within-subject standardization (z-scores)
    - Creation of composite indices (affect, imagery, self)
    
    Attributes:
        data (pd.DataFrame): TET data to preprocess
        dimension_columns (List[str]): List of dimension column names
    
    Example:
        >>> from tet.data_loader import TETDataLoader
        >>> from tet.preprocessor import TETPreprocessor
        >>> import config
        >>> 
        >>> # Load data
        >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
        >>> data = loader.load_data()
        >>> 
        >>> # Preprocess
        >>> preprocessor = TETPreprocessor(data, config.TET_DIMENSION_COLUMNS)
        >>> data_preprocessed = preprocessor.preprocess_all()
    """
    
    def __init__(self, data: pd.DataFrame, dimension_columns: List[str]):
        """
        Initialize preprocessor with TET data.
        
        Args:
            data (pd.DataFrame): TET data with columns for subject, session_id,
                state, dose, t_bin, t_sec, and dimension ratings
            dimension_columns (List[str]): List of the 15 dimension column names
        """
        self.data = data.copy()
        self.dimension_columns = dimension_columns
        logger.info(f"Initialized TETPreprocessor with {len(data)} rows, "
                   f"{data['subject'].nunique()} subjects")
    
    def trim_sessions(self) -> pd.DataFrame:
        """
        Trim sessions to analysis time windows.
        
        Trims RS sessions to 0-10 minutes (0-600s, 150 points @ 0.25 Hz)
        and DMT sessions to 0-20 minutes (0-1200s, 300 points @ 0.25 Hz).
        
        Returns:
            pd.DataFrame: Trimmed data preserving original 0.25 Hz sampling rate
        """
        # Filter by time windows
        mask_rs = (self.data['state'] == 'RS') & (self.data['t_sec'] < 600)
        mask_dmt = (self.data['state'] == 'DMT') & (self.data['t_sec'] < 1200)
        
        trimmed = self.data[mask_rs | mask_dmt].copy()
        
        # Log trimming results
        n_before = len(self.data)
        n_after = len(trimmed)
        n_removed = n_before - n_after
        
        logger.info(f"Trimmed sessions: {n_before} → {n_after} rows ({n_removed} removed)")
        logger.info(f"  RS: {len(trimmed[trimmed['state']=='RS'])} points "
                   f"({trimmed[trimmed['state']=='RS'].groupby(['subject','session_id']).size().unique()} per session)")
        logger.info(f"  DMT: {len(trimmed[trimmed['state']=='DMT'])} points "
                   f"({trimmed[trimmed['state']=='DMT'].groupby(['subject','session_id']).size().unique()} per session)")
        
        return trimmed
    
    def create_valence_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create valence variables as aliases for pleasantness and unpleasantness.
        
        Args:
            data (pd.DataFrame): Data to add valence variables to
            
        Returns:
            pd.DataFrame: Data with valence_pos and valence_neg columns added
        """
        data = data.copy()
        
        # Create valence variables
        data['valence_pos'] = data['pleasantness']
        data['valence_neg'] = data['unpleasantness']
        
        logger.info("Created valence variables (valence_pos, valence_neg)")
        
        return data
    
    def standardize_within_subject(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global within-subject z-scores for all dimensions.
        
        For each subject, computes z-scores using the global mean and standard
        deviation across all 15 dimensions and all 4 sessions. This controls
        for individual differences in scale usage patterns.
        
        Args:
            data (pd.DataFrame): Data to standardize
            
        Returns:
            pd.DataFrame: Data with z-scored dimension columns (suffix _z)
        """
        data = data.copy()
        
        # Process each subject
        for subject in data['subject'].unique():
            # Get all dimension values for this subject (all sessions, all dimensions)
            subject_mask = data['subject'] == subject
            subject_data = data.loc[subject_mask, self.dimension_columns]
            
            # Stack all values into single array
            all_values = subject_data.values.flatten()
            
            # Compute global statistics
            global_mean = np.mean(all_values)
            global_std = np.std(all_values, ddof=1)
            
            # Handle edge case: zero std (all values identical)
            if global_std == 0:
                logger.warning(f"Subject {subject}: global_std = 0, setting all z-scores to 0")
                global_std = 1  # Avoid division by zero
            
            # Standardize each dimension
            for dim in self.dimension_columns:
                z_col = f"{dim}_z"
                data.loc[subject_mask, z_col] = (
                    (data.loc[subject_mask, dim] - global_mean) / global_std
                )
            
            logger.debug(f"Subject {subject}: global_mean={global_mean:.3f}, "
                        f"global_std={global_std:.3f}")
        
        logger.info(f"Standardized {len(self.dimension_columns)} dimensions "
                   f"for {data['subject'].nunique()} subjects")
        
        return data
    
    def create_composite_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite indices from z-scored dimensions.
        
        Creates three composite indices:
        - affect_index_z: mean(pleasantness_z, bliss_z) - mean(anxiety_z, unpleasantness_z)
        - imagery_index_z: mean(elementary_imagery_z, complex_imagery_z)
        - self_index_z: -disembodiment_z + selfhood_z
        
        Args:
            data (pd.DataFrame): Data with z-scored dimensions
            
        Returns:
            pd.DataFrame: Data with composite index columns added
        """
        data = data.copy()
        
        # Affect index: positive affect - negative affect
        positive_affect = (data['pleasantness_z'] + data['bliss_z']) / 2
        negative_affect = (data['anxiety_z'] + data['unpleasantness_z']) / 2
        data['affect_index_z'] = positive_affect - negative_affect
        
        # Imagery index: mean of elementary and complex imagery
        data['imagery_index_z'] = (
            (data['elementary_imagery_z'] + data['complex_imagery_z']) / 2
        )
        
        # Self index: inverted disembodiment + selfhood
        # Higher values = more self-integration (embodied, strong sense of self)
        data['self_index_z'] = -data['disembodiment_z'] + data['selfhood_z']
        
        logger.info("Created composite indices: affect_index_z, imagery_index_z, self_index_z")
        
        return data
    
    def preprocess_all(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Executes all preprocessing steps in order:
        1. Trim sessions to analysis windows
        2. Create valence variables
        3. Standardize within subject (global z-scores)
        4. Create composite indices
        
        Returns:
            pd.DataFrame: Fully preprocessed data with:
                - Original dimension columns
                - Z-scored dimension columns (suffix _z)
                - Valence variables (valence_pos, valence_neg)
                - Composite indices (affect_index_z, imagery_index_z, self_index_z)
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Trim sessions
        data = self.trim_sessions()
        
        # Step 2: Create valence variables
        data = self.create_valence_variables(data)
        
        # Step 3: Standardize within subject
        data = self.standardize_within_subject(data)
        
        # Step 4: Create composite indices
        data = self.create_composite_indices(data)
        
        # Log final statistics
        n_rows = len(data)
        n_subjects = data['subject'].nunique()
        n_sessions = data.groupby(['subject', 'session_id']).ngroups
        n_cols = len(data.columns)
        
        logger.info(f"Preprocessing complete: {n_rows} rows, {n_subjects} subjects, "
                   f"{n_sessions} sessions, {n_cols} columns")
        
        return data
