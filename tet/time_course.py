# -*- coding: utf-8 -*-
"""
TET Time Course Analysis Module

This module provides functionality for computing group-level time courses
(mean ± SEM) for TET dimensions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging
import sys
import os

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Configure logging
logger = logging.getLogger(__name__)


class TETTimeCourseAnalyzer:
    """
    Computes group-level time courses for TET dimensions.
    
    This class computes mean and standard error of the mean (SEM) for each
    dimension at each time bin, grouped by state and dose.
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data
        dimensions (List[str]): List of dimension column names to analyze
    
    Example:
        >>> from tet.time_course import TETTimeCourseAnalyzer
        >>> import pandas as pd
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
        >>> 
        >>> # Create analyzer
        >>> analyzer = TETTimeCourseAnalyzer(data)
        >>> 
        >>> # Compute time courses
        >>> time_courses = analyzer.compute_all_time_courses()
        >>> 
        >>> # Export to CSV
        >>> analyzer.export_time_courses('results/tet/descriptive')
    """
    
    def __init__(self, data: pd.DataFrame, dimensions: Optional[List[str]] = None):
        """
        Initialize time course analyzer.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data
            dimensions (Optional[List[str]]): List of dimensions to analyze.
                If None, uses all z-scored dimensions and composite index.
        """
        self.data = data
        
        # Default: all z-scored dimensions + composite index
        if dimensions is None:
            z_dims = [f"{dim}_z" for dim in config.TET_DIMENSION_COLUMNS]
            composite_dims = ['valence_index_z']
            self.dimensions = z_dims + composite_dims
        else:
            self.dimensions = dimensions
        
        logger.info(f"Initialized TETTimeCourseAnalyzer with {len(self.dimensions)} dimensions")
    
    def compute_time_course(self, dimension: str) -> pd.DataFrame:
        """
        Compute time course for a single dimension.
        
        Args:
            dimension (str): Dimension column name
            
        Returns:
            pd.DataFrame: Time course data with columns:
                - dimension: dimension name
                - state: 'RS' or 'DMT'
                - dose: dose level
                - t_bin: time bin index
                - t_sec: time in seconds
                - mean: mean value across subjects
                - sem: standard error of the mean
                - n: number of observations
        """
        if dimension not in self.data.columns:
            raise ValueError(f"Dimension '{dimension}' not found in data")
        
        # Group by state, dose, t_bin
        grouped = self.data.groupby(['state', 'dose', 't_bin', 't_sec'])
        
        # Compute mean and SEM
        time_course = grouped[dimension].agg([
            ('mean', 'mean'),
            ('sem', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            ('n', 'count')
        ]).reset_index()
        
        # Add dimension column
        time_course.insert(0, 'dimension', dimension)
        
        return time_course
    
    def compute_all_time_courses(self) -> pd.DataFrame:
        """
        Compute time courses for all dimensions.
        
        Returns:
            pd.DataFrame: Combined time course data for all dimensions
        """
        logger.info(f"Computing time courses for {len(self.dimensions)} dimensions...")
        
        time_courses = []
        for i, dimension in enumerate(self.dimensions, 1):
            logger.debug(f"  [{i}/{len(self.dimensions)}] {dimension}")
            tc = self.compute_time_course(dimension)
            time_courses.append(tc)
        
        # Concatenate all time courses
        all_time_courses = pd.concat(time_courses, ignore_index=True)
        
        logger.info(f"Computed {len(all_time_courses)} time course points")
        
        return all_time_courses
    
    def export_time_courses(
        self,
        output_dir: str,
        filename: str = 'time_course_all_dimensions.csv'
    ) -> str:
        """
        Compute and export time courses to CSV.
        
        Args:
            output_dir (str): Output directory path
            filename (str): Output filename (default: 'time_course_all_dimensions.csv')
            
        Returns:
            str: Path to exported file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute time courses
        time_courses = self.compute_all_time_courses()
        
        # Export to CSV
        output_path = os.path.join(output_dir, filename)
        time_courses.to_csv(output_path, index=False)
        
        logger.info(f"Exported time courses to: {output_path}")
        
        return output_path
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for time courses.
        
        Returns:
            pd.DataFrame: Summary statistics including:
                - dimension
                - state
                - n_time_bins
                - mean_n_per_bin (average number of subjects per bin)
        """
        time_courses = self.compute_all_time_courses()
        
        summary = time_courses.groupby(['dimension', 'state']).agg({
            't_bin': 'nunique',
            'n': 'mean'
        }).reset_index()
        
        summary.columns = ['dimension', 'state', 'n_time_bins', 'mean_n_per_bin']
        
        return summary
