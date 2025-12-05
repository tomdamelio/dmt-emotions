# -*- coding: utf-8 -*-
"""
TET Preprocessing Metadata Module

This module provides functionality for generating and documenting preprocessing
metadata including parameters, composite index definition, and summary statistics.
"""

from typing import Dict, Any
from datetime import datetime
import pandas as pd
import sys
import os

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class PreprocessingMetadata:
    """
    Generates metadata documentation for TET preprocessing.
    
    This class creates comprehensive metadata documenting all preprocessing
    steps, parameters, and composite index definition for reproducibility
    and traceability.
    
    Example:
        >>> from tet.metadata import PreprocessingMetadata
        >>> metadata_gen = PreprocessingMetadata()
        >>> metadata = metadata_gen.generate_metadata(data_preprocessed)
        >>> 
        >>> import json
        >>> with open('preprocessing_metadata.json', 'w') as f:
        ...     json.dump(metadata, f, indent=2)
    """
    
    def __init__(self):
        """Initialize metadata generator."""
        pass
    
    def generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive preprocessing metadata.
        
        Creates metadata documenting:
        - Trimming windows and sampling rate
        - Standardization method
        - Composite index formula and directionality
        - Summary statistics
        - Timestamp
        
        Args:
            data (pd.DataFrame): Preprocessed TET data
            
        Returns:
            Dictionary containing all preprocessing metadata
        """
        # Summary statistics
        n_subjects = data['subject'].nunique()
        n_sessions = data.groupby(['subject', 'session_id']).ngroups
        n_time_points = len(data)
        n_dimensions = len(config.TET_DIMENSION_COLUMNS)
        
        # Count by state
        n_rs_points = len(data[data['state'] == 'RS'])
        n_dmt_points = len(data[data['state'] == 'DMT'])
        
        # Composite index definition
        composite_indices = {
            'valence_index_z': {
                'formula': 'pleasantness_z - unpleasantness_z',
                'components': {
                    'positive': ['pleasantness'],
                    'negative': ['unpleasantness']
                },
                'interpretation': 'Higher values indicate more positive affective valence (pleasant). '
                                 'Lower values indicate more negative affective valence (unpleasant).',
                'directionality': 'pleasantness - unpleasantness'
            }
        }
        
        # Compile metadata
        metadata = {
            'preprocessing_version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'random_seed': 22,
            'data_summary': {
                'n_subjects': n_subjects,
                'n_sessions': n_sessions,
                'n_time_points_total': n_time_points,
                'n_time_points_rs': n_rs_points,
                'n_time_points_dmt': n_dmt_points,
                'n_dimensions': n_dimensions
            },
            'trimming': {
                'method': 'time-based',
                'rs_window': '0-600 seconds (0-10 minutes)',
                'dmt_window': '0-1200 seconds (0-20 minutes)',
                'sampling_rate_hz': config.TET_SAMPLING_RATE_HZ,
                'sampling_interval_sec': config.TET_SAMPLING_INTERVAL_SEC,
                'expected_points': {
                    'rs': 150,
                    'dmt': 300
                }
            },
            'standardization': {
                'method': 'global_within_subject',
                'description': 'Z-scores computed using global mean and std across all 15 dimensions '
                              'and all 4 sessions for each subject. This controls for individual '
                              'differences in scale usage patterns.',
                'formula': '(value - global_mean) / global_std',
                'scope': 'all_dimensions_all_sessions_per_subject'
            },
            'valence_variables': {
                'valence_pos': 'pleasantness',
                'valence_neg': 'unpleasantness'
            },
            'composite_indices': composite_indices,
            'dimensions': config.TET_DIMENSION_COLUMNS
        }
        
        return metadata
