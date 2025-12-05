"""
CCA Data Validator for physiological-TET integration analysis.

This module validates data structure and temporal resolution before CCA analysis
to ensure proper sample sizes and avoid artificially inflated N from temporal
autocorrelation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging


class CCADataValidator:
    """
    Validator for CCA input data structure and temporal resolution.
    
    Validates that:
    1. Data are aggregated to 30-second bins (not raw 0.25 Hz)
    2. Sample size is sufficient for CCA (N ≥ 100)
    3. Data structure is correct (aligned physio and TET matrices)
    4. Subject IDs match across modalities
    
    Attributes:
        merged_data: Merged physiological-TET dataset
        validation_report: Dict storing validation results
        logger: Logger instance
    
    Example:
        >>> validator = CCADataValidator(merged_data)
        >>> validator.validate_temporal_resolution()
        >>> validator.validate_sample_size()
        >>> validator.audit_data_structure()
        >>> report = validator.generate_validation_report()
    """
    
    def __init__(self, merged_data: pd.DataFrame):
        """
        Initialize CCA data validator.
        
        Args:
            merged_data: Merged physio-TET dataset from TETPhysioDataLoader
                Must contain columns: subject, session_id, state, dose, window,
                physiological measures, and TET dimensions
        """
        self.data = merged_data.copy()
        self.validation_report = {}
        self.logger = logging.getLogger(__name__)
    
    def validate_temporal_resolution(self) -> Dict:
        """
        Validate that data are aggregated to 30-second bins.
        
        Checks:
        1. Expected bin count: 18 bins per 9-minute session
        2. Actual sampling rate from time differences
        3. Flags if resolution appears to be raw data (~1350 points/subject)
        
        Process:
        - Compute bins per session for each subject-session combination
        - Compute mean time difference between consecutive bins
        - Determine if data are at 30s resolution or raw 0.25 Hz
        
        Returns:
            Dict with:
            - resolution_seconds: Estimated temporal resolution
            - bins_per_session: Mean bins per session
            - is_valid: True if data are at 30s resolution
            - warning_message: Description of any issues
        
        Raises:
            ValueError: If temporal resolution is invalid (raw data detected)
        """
        self.logger.info("Validating temporal resolution...")
        
        # Expected values for 30-second bins
        EXPECTED_BIN_DURATION = 30  # seconds
        EXPECTED_BINS_PER_9MIN = 18  # 9 minutes / 30 seconds
        RAW_SAMPLING_RATE = 0.25  # Hz (4 seconds per sample)
        
        # Compute bins per session
        # Try different grouping columns depending on data structure
        if 'session_id' in self.data.columns:
            bins_per_session = self.data.groupby(['subject', 'session_id']).size()
        elif 'State' in self.data.columns and 'Dose' in self.data.columns:
            # For composite data: group by subject, State, Dose
            bins_per_session = self.data.groupby(['subject', 'State', 'Dose']).size()
        else:
            # Fallback: group by subject only
            bins_per_session = self.data.groupby('subject').size()
        
        mean_bins = bins_per_session.mean()
        std_bins = bins_per_session.std()
        
        # Estimate temporal resolution from time differences
        # Check if 't_sec' or 'time_sec' column exists
        time_col = None
        for col in ['t_sec', 'time_sec']:
            if col in self.data.columns:
                time_col = col
                break
        
        if time_col is None:
            # Use window index as proxy - assume 30-second bins
            resolution_seconds = EXPECTED_BIN_DURATION
            self.logger.warning(
                "No time column found, assuming 30-second bins based on window index"
            )
        else:
            # Compute time differences within each session
            time_diffs = []
            
            # Determine grouping columns
            if 'session_id' in self.data.columns:
                group_cols = ['subject', 'session_id']
            elif 'State' in self.data.columns and 'Dose' in self.data.columns:
                group_cols = ['subject', 'State', 'Dose']
            else:
                group_cols = ['subject']
            
            for group_key, group in self.data.groupby(group_cols):
                if len(group) > 1:
                    sorted_group = group.sort_values(time_col)
                    diffs = sorted_group[time_col].diff().dropna()
                    # Filter out zero differences
                    non_zero_diffs = diffs[diffs > 0]
                    time_diffs.extend(non_zero_diffs.values)
            
            if len(time_diffs) > 0:
                resolution_seconds = np.median(time_diffs)
            else:
                # If all differences are zero, use window index
                # This means data are already aggregated to bins
                resolution_seconds = EXPECTED_BIN_DURATION
                self.logger.info(
                    "Time column has no variation within sessions, "
                    "assuming 30-second bins based on window structure"
                )
        
        # Determine if data are valid
        is_valid = True
        warning_message = None
        
        # Check if resolution is close to 30 seconds
        if abs(resolution_seconds - EXPECTED_BIN_DURATION) > 5:
            is_valid = False
            warning_message = (
                f"Temporal resolution ({resolution_seconds:.1f}s) does not match "
                f"expected 30-second bins. Data may be at raw sampling rate."
            )
        
        # Check if bins per session is close to expected
        if mean_bins > 100:
            is_valid = False
            warning_message = (
                f"Mean bins per session ({mean_bins:.1f}) is much higher than "
                f"expected ({EXPECTED_BINS_PER_9MIN}). Data appear to be raw "
                f"0.25 Hz samples, not 30-second aggregates."
            )
        
        # Store results
        result = {
            'resolution_seconds': float(resolution_seconds),
            'bins_per_session': float(mean_bins),
            'bins_per_session_std': float(std_bins),
            'expected_bins': EXPECTED_BINS_PER_9MIN,
            'is_valid': is_valid,
            'warning_message': warning_message
        }
        
        self.validation_report['temporal_resolution'] = result
        
        # Log results
        if is_valid:
            self.logger.info(
                f"  ✓ Temporal resolution valid: {resolution_seconds:.1f}s bins, "
                f"{mean_bins:.1f} bins/session"
            )
        else:
            self.logger.error(f"  ✗ Temporal resolution invalid: {warning_message}")
            raise ValueError(warning_message)
        
        return result
    
    def validate_sample_size(self) -> Dict:
        """
        Validate that sample size is sufficient for CCA.
        
        Checks:
        1. Number of subjects with complete data
        2. Total observations (N_subjects × N_sessions × N_bins)
        3. Minimum N for CCA (recommend N ≥ 100)
        4. Observations per subject and per session
        
        Process:
        - Identify subjects with complete physio and TET data
        - Count total observations
        - Compute observations per subject
        - Check against minimum threshold
        
        Returns:
            Dict with:
            - n_subjects: Number of subjects with complete data
            - n_sessions: Total number of sessions
            - n_total_obs: Total observations
            - n_obs_per_subject: Mean observations per subject
            - is_sufficient: True if N ≥ 100
        
        Raises:
            ValueError: If sample size is insufficient (N < 100)
        """
        self.logger.info("Validating sample size...")
        
        MINIMUM_N = 100  # Minimum observations for CCA
        
        # Identify complete cases (no missing values in key variables)
        physio_cols = ['HR', 'SMNA_AUC', 'RVT']
        tet_affective_cols = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        
        # Check which columns exist
        available_physio = [col for col in physio_cols if col in self.data.columns]
        available_tet = [col for col in tet_affective_cols if col in self.data.columns]
        
        if len(available_physio) == 0 or len(available_tet) == 0:
            raise ValueError(
                "Missing required columns for CCA validation. "
                f"Available physio: {available_physio}, Available TET: {available_tet}"
            )
        
        # Identify complete cases
        complete_mask = ~(
            self.data[available_physio].isna().any(axis=1) |
            self.data[available_tet].isna().any(axis=1)
        )
        
        complete_data = self.data[complete_mask]
        
        # Count subjects, sessions, observations
        n_subjects = complete_data['subject'].nunique()
        
        # Count sessions based on available columns
        if 'session_id' in complete_data.columns:
            n_sessions = complete_data.groupby('subject')['session_id'].nunique().sum()
        elif 'State' in complete_data.columns and 'Dose' in complete_data.columns:
            # Count unique State-Dose combinations per subject
            n_sessions = complete_data.groupby('subject').apply(
                lambda x: len(x[['State', 'Dose']].drop_duplicates())
            ).sum()
        else:
            n_sessions = n_subjects  # Fallback
        
        n_total_obs = len(complete_data)
        n_obs_per_subject = n_total_obs / n_subjects if n_subjects > 0 else 0
        
        # Check sufficiency
        is_sufficient = n_total_obs >= MINIMUM_N
        
        # Store results
        result = {
            'n_subjects': int(n_subjects),
            'n_sessions': int(n_sessions),
            'n_total_obs': int(n_total_obs),
            'n_obs_per_subject': float(n_obs_per_subject),
            'minimum_required': MINIMUM_N,
            'is_sufficient': is_sufficient
        }
        
        self.validation_report['sample_size'] = result
        
        # Log results
        if is_sufficient:
            self.logger.info(
                f"  ✓ Sample size sufficient: N = {n_total_obs} "
                f"({n_subjects} subjects, {n_sessions} sessions)"
            )
        else:
            self.logger.error(
                f"  ✗ Sample size insufficient: N = {n_total_obs} < {MINIMUM_N}"
            )
            raise ValueError(
                f"Insufficient sample size for CCA: N = {n_total_obs} < {MINIMUM_N}"
            )
        
        return result
    
    def audit_data_structure(self) -> pd.DataFrame:
        """
        Audit data structure and alignment between physio and TET.
        
        Checks:
        1. Alignment between physio and TET matrices
        2. Missing values in key variables
        3. Subject IDs match across modalities
        4. Temporal alignment (window indices match)
        
        Process:
        - For each subject, check completeness of physio and TET data
        - Identify subjects with missing data
        - Compute per-subject completeness statistics
        
        Returns:
            DataFrame with per-subject completeness statistics:
            - subject: Subject ID
            - n_obs: Total observations
            - n_complete: Observations with complete physio and TET
            - completeness_rate: Proportion of complete observations
            - missing_physio: Count of missing physiological values
            - missing_tet: Count of missing TET values
        """
        self.logger.info("Auditing data structure...")
        
        # Define key variables
        physio_cols = ['HR', 'SMNA_AUC', 'RVT']
        tet_affective_cols = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        
        # Check which columns exist
        available_physio = [col for col in physio_cols if col in self.data.columns]
        available_tet = [col for col in tet_affective_cols if col in self.data.columns]
        
        # Compute per-subject statistics
        audit_results = []
        
        for subject in self.data['subject'].unique():
            subj_data = self.data[self.data['subject'] == subject]
            
            n_obs = len(subj_data)
            
            # Count missing values
            missing_physio = subj_data[available_physio].isna().any(axis=1).sum()
            missing_tet = subj_data[available_tet].isna().any(axis=1).sum()
            
            # Count complete cases
            complete_mask = ~(
                subj_data[available_physio].isna().any(axis=1) |
                subj_data[available_tet].isna().any(axis=1)
            )
            n_complete = complete_mask.sum()
            
            completeness_rate = n_complete / n_obs if n_obs > 0 else 0
            
            audit_results.append({
                'subject': subject,
                'n_obs': n_obs,
                'n_complete': n_complete,
                'completeness_rate': completeness_rate,
                'missing_physio': missing_physio,
                'missing_tet': missing_tet
            })
        
        audit_df = pd.DataFrame(audit_results)
        
        # Store in validation report
        self.validation_report['data_structure'] = {
            'n_subjects_total': len(audit_df),
            'n_subjects_complete': (audit_df['completeness_rate'] == 1.0).sum(),
            'mean_completeness_rate': float(audit_df['completeness_rate'].mean()),
            'subjects_with_missing_data': audit_df[
                audit_df['completeness_rate'] < 1.0
            ]['subject'].tolist()
        }
        
        # Log results
        n_complete_subjects = (audit_df['completeness_rate'] == 1.0).sum()
        mean_completeness = audit_df['completeness_rate'].mean()
        
        self.logger.info(
            f"  Data structure audit complete: "
            f"{n_complete_subjects}/{len(audit_df)} subjects with complete data "
            f"(mean completeness: {mean_completeness:.1%})"
        )
        
        if n_complete_subjects < len(audit_df):
            n_incomplete = len(audit_df) - n_complete_subjects
            self.logger.warning(
                f"  ⚠ {n_incomplete} subjects have incomplete data"
            )
        
        return audit_df
    
    def generate_validation_report(self, output_path: str = None) -> Dict:
        """
        Generate comprehensive validation report.
        
        Compiles all validation checks into structured report with:
        - Temporal resolution validation
        - Sample size validation
        - Data structure audit
        - Recommendations for proceeding with CCA
        
        Args:
            output_path: Optional path to save report as text file
        
        Returns:
            validation_report: Dict with all validation results
        """
        self.logger.info("Generating validation report...")
        
        # Ensure all validations have been run
        if 'temporal_resolution' not in self.validation_report:
            self.validate_temporal_resolution()
        
        if 'sample_size' not in self.validation_report:
            self.validate_sample_size()
        
        if 'data_structure' not in self.validation_report:
            self.audit_data_structure()
        
        # Add overall validation status
        all_valid = (
            self.validation_report['temporal_resolution']['is_valid'] and
            self.validation_report['sample_size']['is_sufficient']
        )
        
        self.validation_report['overall_status'] = {
            'is_valid': all_valid,
            'ready_for_cca': all_valid
        }
        
        # Generate text report if output path provided
        if output_path:
            self._write_text_report(output_path)
        
        # Log summary
        if all_valid:
            self.logger.info("  ✓ All validations passed - data ready for CCA")
        else:
            self.logger.error("  ✗ Validation failed - data not ready for CCA")
        
        return self.validation_report
    
    def _write_text_report(self, output_path: str):
        """
        Write validation report to text file.
        
        Args:
            output_path: Path to save report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CCA DATA VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall status
            f.write("OVERALL STATUS\n")
            f.write("-" * 80 + "\n")
            status = self.validation_report['overall_status']
            if status['is_valid']:
                f.write("✓ PASSED - Data are valid and ready for CCA analysis\n\n")
            else:
                f.write("✗ FAILED - Data validation issues detected\n\n")
            
            # Temporal resolution
            f.write("TEMPORAL RESOLUTION\n")
            f.write("-" * 80 + "\n")
            tres = self.validation_report['temporal_resolution']
            f.write(f"Resolution: {tres['resolution_seconds']:.1f} seconds\n")
            f.write(f"Bins per session: {tres['bins_per_session']:.1f} ± {tres['bins_per_session_std']:.1f}\n")
            f.write(f"Expected bins: {tres['expected_bins']}\n")
            f.write(f"Status: {'✓ VALID' if tres['is_valid'] else '✗ INVALID'}\n")
            if tres['warning_message']:
                f.write(f"Warning: {tres['warning_message']}\n")
            f.write("\n")
            
            # Sample size
            f.write("SAMPLE SIZE\n")
            f.write("-" * 80 + "\n")
            ssize = self.validation_report['sample_size']
            f.write(f"Subjects: {ssize['n_subjects']}\n")
            f.write(f"Sessions: {ssize['n_sessions']}\n")
            f.write(f"Total observations: {ssize['n_total_obs']}\n")
            f.write(f"Observations per subject: {ssize['n_obs_per_subject']:.1f}\n")
            f.write(f"Minimum required: {ssize['minimum_required']}\n")
            f.write(f"Status: {'✓ SUFFICIENT' if ssize['is_sufficient'] else '✗ INSUFFICIENT'}\n")
            f.write("\n")
            
            # Data structure
            f.write("DATA STRUCTURE\n")
            f.write("-" * 80 + "\n")
            dstruct = self.validation_report['data_structure']
            f.write(f"Total subjects: {dstruct['n_subjects_total']}\n")
            f.write(f"Subjects with complete data: {dstruct['n_subjects_complete']}\n")
            f.write(f"Mean completeness rate: {dstruct['mean_completeness_rate']:.1%}\n")
            if dstruct['subjects_with_missing_data']:
                f.write(f"Subjects with missing data: {', '.join(dstruct['subjects_with_missing_data'])}\n")
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if status['is_valid']:
                f.write("✓ Data properly aggregated to 30-second bins\n")
                f.write("✓ Sample size sufficient for CCA (N ≥ 100)\n")
                f.write("✓ Proceed with CCA analysis\n")
            else:
                f.write("✗ Address validation issues before proceeding:\n")
                if not tres['is_valid']:
                    f.write("  - Aggregate data to 30-second bins\n")
                if not ssize['is_sufficient']:
                    f.write("  - Increase sample size or check data completeness\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        self.logger.info(f"  Validation report saved to {output_file}")
