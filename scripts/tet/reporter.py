"""TET Validation Reporter

This module provides reporting functionality for TET data validation results.
It generates human-readable reports and exports validation logs.
"""

import os
from typing import Dict, Any
import pandas as pd


class ValidationReporter:
    """Generates validation reports for TET data quality checks.
    
    This class takes validation results from TETDataValidator and creates
    comprehensive reports including summary statistics, detailed issue lists,
    and adjustment logs.
    
    Attributes:
        validation_results: Dictionary containing validation results from TETDataValidator
        output_dir: Directory path where reports will be saved
    
    Example:
        >>> reporter = ValidationReporter(validation_results, 'results/tet/validation')
        >>> report_path = reporter.generate_report()
        >>> print(f"Report saved to: {report_path}")
    """
    
    def __init__(self, validation_results: Dict[str, Any], output_dir: str):
        """Initialize reporter with validation results and output directory.
        
        Args:
            validation_results: Dictionary containing validation results with keys:
                - summary: Dict with n_subjects, n_sessions, n_time_bins, n_dimensions
                - session_length_issues: Dict mapping sessions to issues
                - range_violations: DataFrame with out-of-range values
                - completeness_issues: Dict mapping subjects to missing sessions
                - adjustments_made: DataFrame with clamping adjustments (optional)
                - timestamp: ISO format timestamp
            output_dir: Path to directory where reports will be saved
        """
        self.validation_results = validation_results
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_summary_stats(self) -> Dict[str, int]:
        """Generate summary statistics from validation results.
        
        Extracts key counts from validation results including number of subjects,
        sessions, time bins, and dimensions.
        
        Returns:
            Dictionary with keys:
            - n_subjects: Number of unique subjects
            - n_sessions: Number of unique sessions
            - n_time_bins: Total number of time bin observations
            - n_dimensions: Number of dimension columns validated
            
        Example:
            >>> stats = reporter.generate_summary_stats()
            >>> print(f"Validated {stats['n_subjects']} subjects")
        """
        return self.validation_results.get('summary', {})

    def generate_report(self) -> str:
        """Generate comprehensive validation report.
        
        Creates a text report with sections for:
        - Summary statistics
        - Session length validation results
        - Dimension range validation results
        - Subject completeness results
        - List of all adjustments made (if any)
        
        Also exports adjustment log as CSV if adjustments were made.
        
        Returns:
            Path to the generated validation_report.txt file
            
        Example:
            >>> report_path = reporter.generate_report()
            >>> with open(report_path) as f:
            ...     print(f.read())
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TET DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Timestamp
        timestamp = self.validation_results.get('timestamp', 'Unknown')
        report_lines.append(f"Validation Date: {timestamp}")
        report_lines.append("")
        
        # Summary Statistics
        report_lines.append("-" * 80)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        summary = self.generate_summary_stats()
        report_lines.append(f"Number of Subjects:    {summary.get('n_subjects', 0)}")
        report_lines.append(f"Number of Sessions:    {summary.get('n_sessions', 0)}")
        report_lines.append(f"Number of Time Bins:   {summary.get('n_time_bins', 0)}")
        report_lines.append(f"Number of Dimensions:  {summary.get('n_dimensions', 0)}")
        report_lines.append("")
        
        # Session Length Validation
        report_lines.append("-" * 80)
        report_lines.append("SESSION LENGTH VALIDATION")
        report_lines.append("-" * 80)
        session_issues = self.validation_results.get('session_length_issues', {})
        if session_issues:
            report_lines.append(f"Found {len(session_issues)} session(s) with incorrect length:")
            report_lines.append("")
            for session_key, issues in sorted(session_issues.items()):
                report_lines.append(f"  {session_key}:")
                for issue in issues:
                    report_lines.append(f"    - {issue}")
        else:
            report_lines.append("✓ All sessions have correct length")
        report_lines.append("")
        
        # Dimension Range Validation
        report_lines.append("-" * 80)
        report_lines.append("DIMENSION RANGE VALIDATION")
        report_lines.append("-" * 80)
        range_violations = self.validation_results.get('range_violations', pd.DataFrame())
        if isinstance(range_violations, pd.DataFrame) and not range_violations.empty:
            n_violations = len(range_violations)
            report_lines.append(f"Found {n_violations} value(s) outside [0, 10] range:")
            report_lines.append("")
            
            # Group by dimension for summary
            violations_by_dim = range_violations.groupby('dimension').size()
            for dim, count in violations_by_dim.items():
                report_lines.append(f"  {dim}: {count} violation(s)")
            
            report_lines.append("")
            report_lines.append("Details of violations:")
            for _, row in range_violations.head(20).iterrows():
                report_lines.append(
                    f"  Subject {row['subject']}, Session {row['session_id']}, "
                    f"t_bin {row['t_bin']}: {row['dimension']} = {row['value']:.2f}"
                )
            
            if len(range_violations) > 20:
                report_lines.append(f"  ... and {len(range_violations) - 20} more violations")
        else:
            report_lines.append("✓ All dimension values within valid [0, 10] range")
        report_lines.append("")
        
        # Subject Completeness Validation
        report_lines.append("-" * 80)
        report_lines.append("SUBJECT COMPLETENESS VALIDATION")
        report_lines.append("-" * 80)
        completeness_issues = self.validation_results.get('completeness_issues', {})
        if completeness_issues:
            report_lines.append(f"Found {len(completeness_issues)} subject(s) with incomplete data:")
            report_lines.append("")
            for subject, issues in sorted(completeness_issues.items()):
                report_lines.append(f"  {subject}:")
                for issue in issues:
                    report_lines.append(f"    - {issue}")
        else:
            report_lines.append("✓ All subjects have complete data (2 RS + 2 DMT sessions)")
        report_lines.append("")
        
        # Adjustments Made
        report_lines.append("-" * 80)
        report_lines.append("ADJUSTMENTS MADE")
        report_lines.append("-" * 80)
        adjustments = self.validation_results.get('adjustments_made', pd.DataFrame())
        if isinstance(adjustments, pd.DataFrame) and not adjustments.empty:
            n_adjustments = len(adjustments)
            report_lines.append(f"Clamped {n_adjustments} out-of-range value(s) to [0, 10]:")
            report_lines.append("")
            
            # Export adjustments to CSV
            adjustments_path = os.path.join(self.output_dir, 'validation_adjustments.csv')
            adjustments.to_csv(adjustments_path, index=False)
            report_lines.append(f"Full adjustment log saved to: validation_adjustments.csv")
            report_lines.append("")
            
            # Show summary by dimension
            adjustments_by_dim = adjustments.groupby('dimension').size()
            for dim, count in adjustments_by_dim.items():
                report_lines.append(f"  {dim}: {count} adjustment(s)")
        else:
            report_lines.append("✓ No adjustments needed - all values within valid range")
        report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Write report to file
        report_path = os.path.join(self.output_dir, 'validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        return report_path
