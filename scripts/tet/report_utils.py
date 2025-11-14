"""
Utility functions for TET report generation.

This module provides helper functions for report generation, including
update detection and file timestamp comparison.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def check_results_updated(
    results_dir: str,
    report_path: str,
    required_files: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Check if analysis results are newer than the report file.
    
    This function compares timestamps of result files against the report file
    to determine if the report needs to be regenerated.
    
    Args:
        results_dir: Directory containing analysis results
        report_path: Path to the comprehensive report file
        required_files: Optional list of specific files to check.
                       If None, checks all CSV files in results_dir.
    
    Returns:
        Tuple of (needs_update, newer_files):
            - needs_update: True if report should be regenerated
            - newer_files: List of files that are newer than report
    
    Examples:
        >>> needs_update, newer = check_results_updated('results/tet', 'docs/report.md')
        >>> if needs_update:
        ...     print(f"Report outdated. Newer files: {newer}")
    """
    results_path = Path(results_dir)
    report_file = Path(report_path)
    
    # If report doesn't exist, it needs to be generated
    if not report_file.exists():
        logger.info(f"Report file does not exist: {report_path}")
        return True, ["Report file missing"]
    
    # Get report modification time
    report_mtime = report_file.stat().st_mtime
    logger.debug(f"Report modification time: {report_mtime}")
    
    # Determine which files to check
    if required_files is None:
        # Find all CSV files recursively in results directory
        result_files = list(results_path.rglob('*.csv'))
        logger.debug(f"Found {len(result_files)} CSV files in {results_dir}")
    else:
        # Use specified files
        result_files = [results_path / f for f in required_files]
        logger.debug(f"Checking {len(result_files)} specified files")
    
    # Check each result file
    newer_files = []
    for result_file in result_files:
        if not result_file.exists():
            logger.debug(f"Result file does not exist: {result_file}")
            continue
        
        result_mtime = result_file.stat().st_mtime
        
        if result_mtime > report_mtime:
            rel_path = result_file.relative_to(results_path)
            newer_files.append(str(rel_path))
            logger.debug(f"Newer file found: {rel_path}")
    
    needs_update = len(newer_files) > 0
    
    if needs_update:
        logger.info(f"Report is outdated. {len(newer_files)} newer files found.")
    else:
        logger.info("Report is up to date.")
    
    return needs_update, newer_files


def get_result_file_groups(results_dir: str) -> dict:
    """
    Get organized groups of result files by analysis type.
    
    Args:
        results_dir: Directory containing analysis results
    
    Returns:
        Dictionary mapping analysis types to lists of file paths
    
    Examples:
        >>> groups = get_result_file_groups('results/tet')
        >>> print(groups.keys())
        dict_keys(['descriptive', 'lme', 'peak_auc', 'pca', 'clustering'])
    """
    results_path = Path(results_dir)
    
    file_groups = {
        'descriptive': [],
        'lme': [],
        'peak_auc': [],
        'pca': [],
        'clustering': []
    }
    
    # Descriptive statistics files
    descriptive_dir = results_path / 'descriptive'
    if descriptive_dir.exists():
        file_groups['descriptive'] = [
            str(f.relative_to(results_path))
            for f in descriptive_dir.glob('*.csv')
        ]
    
    # LME results files
    lme_dir = results_path / 'lme'
    if lme_dir.exists():
        file_groups['lme'] = [
            str(f.relative_to(results_path))
            for f in lme_dir.glob('*.csv')
        ]
    
    # Peak/AUC results files
    peak_auc_dir = results_path / 'peak_auc'
    if peak_auc_dir.exists():
        file_groups['peak_auc'] = [
            str(f.relative_to(results_path))
            for f in peak_auc_dir.glob('*.csv')
        ]
    
    # PCA results files
    pca_dir = results_path / 'pca'
    if pca_dir.exists():
        file_groups['pca'] = [
            str(f.relative_to(results_path))
            for f in pca_dir.glob('*.csv')
        ]
    
    # Clustering results files
    clustering_dir = results_path / 'clustering'
    if clustering_dir.exists():
        file_groups['clustering'] = [
            str(f.relative_to(results_path))
            for f in clustering_dir.glob('*.csv')
        ]
    
    return file_groups


def should_regenerate_report(
    results_dir: str,
    report_path: str,
    force: bool = False
) -> Tuple[bool, str]:
    """
    Determine if report should be regenerated.
    
    Args:
        results_dir: Directory containing analysis results
        report_path: Path to the comprehensive report file
        force: If True, always regenerate regardless of timestamps
    
    Returns:
        Tuple of (should_regenerate, reason):
            - should_regenerate: True if report should be regenerated
            - reason: Human-readable explanation
    
    Examples:
        >>> should_regen, reason = should_regenerate_report('results/tet', 'docs/report.md')
        >>> if should_regen:
        ...     print(f"Regenerating report: {reason}")
    """
    if force:
        return True, "Force regeneration requested"
    
    report_file = Path(report_path)
    
    if not report_file.exists():
        return True, "Report file does not exist"
    
    # Check if results are newer
    needs_update, newer_files = check_results_updated(results_dir, report_path)
    
    if needs_update:
        if len(newer_files) == 1:
            reason = f"1 result file is newer than report: {newer_files[0]}"
        else:
            reason = f"{len(newer_files)} result files are newer than report"
        return True, reason
    
    return False, "Report is up to date"


def format_file_list(files: List[str], max_display: int = 10) -> str:
    """
    Format a list of files for display.
    
    Args:
        files: List of file paths
        max_display: Maximum number of files to display
    
    Returns:
        Formatted string with file list
    
    Examples:
        >>> files = ['file1.csv', 'file2.csv', 'file3.csv']
        >>> print(format_file_list(files, max_display=2))
        - file1.csv
        - file2.csv
        ... and 1 more file(s)
    """
    if not files:
        return "  (none)"
    
    lines = []
    for i, f in enumerate(files[:max_display]):
        lines.append(f"  - {f}")
    
    if len(files) > max_display:
        remaining = len(files) - max_display
        lines.append(f"  ... and {remaining} more file(s)")
    
    return "\n".join(lines)
