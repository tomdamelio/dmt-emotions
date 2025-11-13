# -*- coding: utf-8 -*-
"""
TET Session Metrics Module

This module provides functionality for computing session-level summary metrics
for TET dimensions including peak value, time to peak, AUC, and slopes.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import sys
import os

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Configure logging
logger = logging.getLogger(__name__)


class TETSessionMetrics:
    """
    Computes session-level summary metrics for TET dimensions.
    
    This class computes metrics including:
    - Peak value: maximum value in the session
    - Time to peak: time when peak occurs
    - AUC (0-9 min): area under curve from 0-540 seconds
    - Slope (0-2 min): linear slope from 0-120 seconds
    - Slope (5-9 min): linear slope from 300-540 seconds
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data
        dimensions (List[str]): List of dimension column names to analyze
    
    Example:
        >>> from tet.session_metrics import TETSessionMetrics
        >>> import pandas as pd
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
        >>> 
        >>> # Create analyzer
        >>> analyzer = TETSessionMetrics(data)
        >>> 
        >>> # Compute session metrics
        >>> metrics = analyzer.compute_all_session_metrics()
        >>> 
        >>> # Export to CSV
        >>> analyzer.export_session_metrics('results/tet/descriptive')
    """
    
    def __init__(self, data: pd.DataFrame, dimensions: Optional[List[str]] = None):
        """
        Initialize session metrics analyzer.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data
            dimensions (Optional[List[str]]): List of dimensions to analyze.
                If None, uses all z-scored dimensions and composite indices.
        """
        self.data = data
        
        # Default: all z-scored dimensions + composite indices
        if dimensions is None:
            z_dims = [f"{dim}_z" for dim in config.TET_DIMENSION_COLUMNS]
            composite_dims = ['affect_index_z', 'imagery_index_z', 'self_index_z']
            self.dimensions = z_dims + composite_dims
        else:
            self.dimensions = dimensions
        
        logger.info(f"Initialized TETSessionMetrics with {len(self.dimensions)} dimensions")
    
    def compute_session_metrics(
        self,
        subject: str,
        session_id: int,
        state: str,
        dose: str,
        dimension: str
    ) -> Dict[str, Any]:
        """
        Compute summary metrics for a single session and dimension.
        
        Args:
            subject (str): Subject ID
            session_id (int): Session number
            state (str): 'RS' or 'DMT'
            dose (str): Dose level
            dimension (str): Dimension column name
            
        Returns:
            Dict[str, Any]: Dictionary with metrics:
                - subject, session_id, state, dose, dimension
                - peak_value, time_to_peak
                - auc_0_9min
                - slope_0_2min, slope_5_9min
        """
        # Get session data
        session_data = self.data[
            (self.data['subject'] == subject) &
            (self.data['session_id'] == session_id) &
            (self.data['state'] == state)
        ].copy()
        
        if len(session_data) == 0:
            logger.warning(f"No data for {subject}, session {session_id}, {state}")
            return None
        
        # Sort by time
        session_data = session_data.sort_values('t_sec')
        
        # Initialize metrics
        metrics = {
            'subject': subject,
            'session_id': session_id,
            'state': state,
            'dose': dose,
            'dimension': dimension
        }
        
        # 1. Peak value and time to peak
        peak_value = session_data[dimension].max()
        peak_idx = session_data[dimension].idxmax()
        time_to_peak = session_data.loc[peak_idx, 't_sec']
        
        metrics['peak_value'] = peak_value
        metrics['time_to_peak'] = time_to_peak
        
        # 2. AUC (0-9 min = 0-540 seconds)
        auc_data = session_data[session_data['t_sec'] <= 540]
        if len(auc_data) >= 2:
            auc = np.trapz(auc_data[dimension], auc_data['t_sec'])
            metrics['auc_0_9min'] = auc
        else:
            metrics['auc_0_9min'] = np.nan
        
        # 3. Slope (0-2 min = 0-120 seconds)
        slope_0_2_data = session_data[session_data['t_sec'] <= 120]
        if len(slope_0_2_data) >= 2:
            slope_0_2 = np.polyfit(slope_0_2_data['t_sec'], slope_0_2_data[dimension], 1)[0]
            metrics['slope_0_2min'] = slope_0_2
        else:
            metrics['slope_0_2min'] = np.nan
        
        # 4. Slope (5-9 min = 300-540 seconds)
        slope_5_9_data = session_data[
            (session_data['t_sec'] >= 300) & (session_data['t_sec'] <= 540)
        ]
        if len(slope_5_9_data) >= 2:
            slope_5_9 = np.polyfit(slope_5_9_data['t_sec'], slope_5_9_data[dimension], 1)[0]
            metrics['slope_5_9min'] = slope_5_9
        else:
            metrics['slope_5_9min'] = np.nan
        
        return metrics
    
    def compute_all_session_metrics(self) -> pd.DataFrame:
        """
        Compute session metrics for all sessions and dimensions.
        
        Returns:
            pd.DataFrame: Session metrics with columns:
                - subject, session_id, state, dose, dimension
                - peak_value, time_to_peak
                - auc_0_9min
                - slope_0_2min, slope_5_9min
        """
        logger.info("Computing session metrics...")
        
        # Get unique sessions
        sessions = self.data.groupby(['subject', 'session_id', 'state', 'dose']).size().reset_index()[
            ['subject', 'session_id', 'state', 'dose']
        ]
        
        logger.info(f"  {len(sessions)} sessions × {len(self.dimensions)} dimensions = {len(sessions) * len(self.dimensions)} metrics to compute")
        
        # Compute metrics for all sessions and dimensions
        all_metrics = []
        total = len(sessions) * len(self.dimensions)
        count = 0
        
        for _, session in sessions.iterrows():
            for dimension in self.dimensions:
                count += 1
                if count % 100 == 0:
                    logger.debug(f"  Progress: {count}/{total} ({count/total*100:.1f}%)")
                
                metrics = self.compute_session_metrics(
                    subject=session['subject'],
                    session_id=session['session_id'],
                    state=session['state'],
                    dose=session['dose'],
                    dimension=dimension
                )
                
                if metrics is not None:
                    all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        logger.info(f"Computed {len(metrics_df)} session metrics")
        
        return metrics_df
    
    def export_session_metrics(
        self,
        output_dir: str,
        filename: str = 'session_metrics_all_dimensions.csv'
    ) -> str:
        """
        Compute and export session metrics to CSV.
        
        Args:
            output_dir (str): Output directory path
            filename (str): Output filename (default: 'session_metrics_all_dimensions.csv')
            
        Returns:
            str: Path to exported file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute session metrics
        metrics = self.compute_all_session_metrics()
        
        # Export to CSV
        output_path = os.path.join(output_dir, filename)
        metrics.to_csv(output_path, index=False)
        
        logger.info(f"Exported session metrics to: {output_path}")
        
        return output_path
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics for session metrics.
        
        Returns:
            pd.DataFrame: Summary statistics including:
                - dimension
                - state
                - n_sessions
                - mean_peak_value
                - mean_time_to_peak
                - mean_auc_0_9min
        """
        metrics = self.compute_all_session_metrics()
        
        summary = metrics.groupby(['dimension', 'state']).agg({
            'subject': 'count',
            'peak_value': 'mean',
            'time_to_peak': 'mean',
            'auc_0_9min': 'mean'
        }).reset_index()
        
        summary.columns = [
            'dimension', 'state', 'n_sessions',
            'mean_peak_value', 'mean_time_to_peak', 'mean_auc_0_9min'
        ]
        
        return summary
