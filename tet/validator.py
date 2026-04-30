"""TET Data Validator

This module provides validation functionality for TET (Temporal Experience Tracking) data.
It checks data integrity including session lengths, dimension ranges, and subject completeness.
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class TETDataValidator:
    """Validates TET data integrity and quality.
    
    This class performs multiple validation checks on TET data including:
    - Session length validation (RS: 20 bins, DMT: 40 bins)
    - Dimension range validation (all values in [0, 10])
    - Subject completeness validation (4 sessions per subject)
    - Value clamping for out-of-range values
    
    Attributes:
        data: DataFrame containing TET data
        dimension_columns: List of dimension column names to validate
    
    Example:
        >>> validator = TETDataValidator(data, dimension_cols)
        >>> results = validator.validate_all()
        >>> if len(results['range_violations']) > 0:
        ...     clean_data, log = validator.clamp_out_of_range_values()
    """
    
    def __init__(self, data: pd.DataFrame, dimension_columns: List[str]):
        """Initialize validator with data and dimension column names.
        
        Args:
            data: DataFrame containing TET data with columns for subject, 
                  session_id, state, dose, t_bin, and dimension ratings
            dimension_columns: List of column names for the 15 dimensions
        """
        self.data = data.copy()
        self.dimension_columns = dimension_columns

    def validate_session_lengths(self) -> Dict[str, List[str]]:
        """Validate that RS and DMT sessions have expected bin counts.
        
        Groups data by subject, session_id, and state, then counts time bins
        per session. Identifies sessions with incorrect lengths based on
        EXPECTED_SESSION_LENGTHS from config.py.
        
        Returns:
            Dictionary mapping session identifiers (subject_session_state) to 
            list of issue descriptions. Empty dict if no issues found.
            
        Example:
            >>> issues = validator.validate_session_lengths()
            >>> # {'S01_1_RS': ['Expected 150 bins, found 148']}
        """
        issues = {}
        
        # Expected session lengths from config
        expected_lengths = config.EXPECTED_SESSION_LENGTHS
        
        # Group by subject, session_id, state and count bins
        session_counts = self.data.groupby(
            ['subject', 'session_id', 'state']
        )['t_bin'].count().reset_index()
        session_counts.columns = ['subject', 'session_id', 'state', 'n_bins']
        
        # Check each session
        for _, row in session_counts.iterrows():
            expected = expected_lengths[row['state']]
            actual = row['n_bins']
            
            if actual != expected:
                session_key = f"{row['subject']}_{row['session_id']}_{row['state']}"
                issue_msg = f"Expected {expected} bins, found {actual}"
                issues[session_key] = [issue_msg]
        
        return issues

    def validate_dimension_ranges(self) -> pd.DataFrame:
        """Check all dimension columns for values outside [0, 10] range.
        
        Scans all dimension columns and identifies any values that fall
        outside the expected [0, 10] range.
        
        Returns:
            DataFrame with columns: subject, session_id, t_bin, dimension, value
            Contains only rows with out-of-range values. Empty DataFrame if 
            all values are valid.
            
        Example:
            >>> violations = validator.validate_dimension_ranges()
            >>> print(f"Found {len(violations)} out-of-range values")
        """
        violations = []
        
        for dim_col in self.dimension_columns:
            # Find values outside [0, 10]
            mask = (self.data[dim_col] < 0) | (self.data[dim_col] > 10)
            out_of_range = self.data[mask]
            
            if len(out_of_range) > 0:
                for _, row in out_of_range.iterrows():
                    violations.append({
                        'subject': row['subject'],
                        'session_id': row['session_id'],
                        't_bin': row['t_bin'],
                        'dimension': dim_col,
                        'value': row[dim_col]
                    })
        
        return pd.DataFrame(violations)

    def validate_subject_completeness(self) -> Dict[str, List[str]]:
        """Validate each subject has 2 RS and 2 DMT sessions (4 total).
        
        Groups by subject and counts unique (state, session_id) combinations
        to ensure each subject has complete data.
        
        Returns:
            Dictionary mapping subject IDs to list of missing session types.
            Empty dict if all subjects have complete data.
            
        Example:
            >>> issues = validator.validate_subject_completeness()
            >>> # {'S05': ['Missing 1 RS session', 'Missing 1 DMT session']}
        """
        issues = {}
        
        # Group by subject and count sessions by state
        subject_sessions = self.data.groupby(['subject', 'state'])['session_id'].nunique().reset_index()
        subject_sessions.columns = ['subject', 'state', 'n_sessions']
        
        # Pivot to get RS and DMT counts per subject
        session_pivot = subject_sessions.pivot(
            index='subject', 
            columns='state', 
            values='n_sessions'
        ).fillna(0).astype(int)
        
        # Check each subject
        for subject in session_pivot.index:
            subject_issues = []
            
            rs_count = session_pivot.loc[subject, 'RS'] if 'RS' in session_pivot.columns else 0
            dmt_count = session_pivot.loc[subject, 'DMT'] if 'DMT' in session_pivot.columns else 0
            
            if rs_count < 2:
                missing = 2 - rs_count
                subject_issues.append(f"Missing {missing} RS session(s)")
            elif rs_count > 2:
                extra = rs_count - 2
                subject_issues.append(f"Extra {extra} RS session(s)")
                
            if dmt_count < 2:
                missing = 2 - dmt_count
                subject_issues.append(f"Missing {missing} DMT session(s)")
            elif dmt_count > 2:
                extra = dmt_count - 2
                subject_issues.append(f"Extra {extra} DMT session(s)")
            
            if subject_issues:
                issues[subject] = subject_issues
        
        return issues

    def clamp_out_of_range_values(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clamp dimension values outside [0, 10] to nearest boundary.
        
        Uses numpy.clip to clamp all dimension values to the valid [0, 10] range.
        Creates a detailed log of all adjustments made.
        
        Returns:
            Tuple of (clamped_data, adjustment_log) where:
            - clamped_data: DataFrame with clamped values
            - adjustment_log: DataFrame documenting all changes with columns:
              subject, session_id, t_bin, dimension, original_value, clamped_value
              
        Example:
            >>> clean_data, log = validator.clamp_out_of_range_values()
            >>> print(f"Clamped {len(log)} values")
            >>> clean_data.to_csv('cleaned_data.csv')
        """
        clamped_data = self.data.copy()
        adjustments = []
        
        # Clamp each dimension column
        for dim_col in self.dimension_columns:
            # Find values that need clamping
            mask = (clamped_data[dim_col] < 0) | (clamped_data[dim_col] > 10)
            
            if mask.any():
                # Record adjustments before clamping
                for idx in clamped_data[mask].index:
                    row = clamped_data.loc[idx]
                    original_value = row[dim_col]
                    clamped_value = np.clip(original_value, 0, 10)
                    
                    adjustments.append({
                        'subject': row['subject'],
                        'session_id': row['session_id'],
                        't_bin': row['t_bin'],
                        'dimension': dim_col,
                        'original_value': original_value,
                        'clamped_value': clamped_value
                    })
                
                # Apply clamping
                clamped_data[dim_col] = np.clip(clamped_data[dim_col], 0, 10)
        
        adjustment_log = pd.DataFrame(adjustments)
        
        return clamped_data, adjustment_log

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks and aggregate results.
        
        Orchestrates all validation methods and compiles results into a
        structured dictionary with summary statistics and detailed findings.
        
        Returns:
            Dictionary containing:
            - summary: Dict with n_subjects, n_sessions, n_time_bins
            - session_length_issues: Dict from validate_session_lengths()
            - range_violations: DataFrame from validate_dimension_ranges()
            - completeness_issues: Dict from validate_subject_completeness()
            - timestamp: ISO format timestamp of validation run
            
        Example:
            >>> results = validator.validate_all()
            >>> print(f"Validated {results['summary']['n_subjects']} subjects")
            >>> if results['range_violations'].empty:
            ...     print("All values in valid range")
        """
        # Run all validation checks
        session_length_issues = self.validate_session_lengths()
        range_violations = self.validate_dimension_ranges()
        completeness_issues = self.validate_subject_completeness()
        
        # Calculate summary statistics
        n_subjects = self.data['subject'].nunique()
        n_sessions = self.data.groupby(['subject', 'session_id']).ngroups
        n_time_bins = len(self.data)
        n_dimensions = len(self.dimension_columns)
        
        # Compile results
        results = {
            'summary': {
                'n_subjects': n_subjects,
                'n_sessions': n_sessions,
                'n_time_bins': n_time_bins,
                'n_dimensions': n_dimensions
            },
            'session_length_issues': session_length_issues,
            'range_violations': range_violations,
            'completeness_issues': completeness_issues,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
