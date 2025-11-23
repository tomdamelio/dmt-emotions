"""
Create CCA Validation Summary Table

This script creates a comprehensive validation summary table combining:
- Permutation test results
- Cross-validation performance metrics
- Redundancy indices (when available)
- Integrated decision for each canonical variate

Author: TET Analysis Pipeline
Date: November 21, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_validation_results(results_dir: Path) -> dict:
    """
    Load all CCA validation result files.
    
    Args:
        results_dir: Directory containing CCA validation results
        
    Returns:
        Dictionary with validation DataFrames
    """
    results = {}
    
    # Load permutation results
    perm_file = results_dir / 'cca_permutation_pvalues.csv'
    if perm_file.exists():
        results['permutation'] = pd.read_csv(perm_file)
        logger.info(f"Loaded permutation results: {len(results['permutation'])} rows")
    else:
        logger.warning(f"Permutation file not found: {perm_file}")
        results['permutation'] = None
    
    # Load cross-validation summary
    cv_file = results_dir / 'cca_cross_validation_summary.csv'
    if cv_file.exists():
        results['cross_validation'] = pd.read_csv(cv_file)
        logger.info(f"Loaded CV results: {len(results['cross_validation'])} rows")
    else:
        logger.warning(f"CV file not found: {cv_file}")
        results['cross_validation'] = None
    
    # Load CV significance results
    cv_sig_file = results_dir / 'cca_cv_significance.csv'
    if cv_sig_file.exists():
        results['cv_significance'] = pd.read_csv(cv_sig_file)
        logger.info(f"Loaded CV significance results: {len(results['cv_significance'])} rows")
    else:
        logger.warning(f"CV significance file not found: {cv_sig_file}")
        results['cv_significance'] = None
    
    # Load redundancy indices (may not exist yet)
    redundancy_file = results_dir / 'cca_redundancy_indices.csv'
    if redundancy_file.exists():
        results['redundancy'] = pd.read_csv(redundancy_file)
        logger.info(f"Loaded redundancy results: {len(results['redundancy'])} rows")
    else:
        logger.warning(f"Redundancy file not found: {redundancy_file}")
        results['redundancy'] = None
    
    return results


def create_validation_summary(results: dict) -> pd.DataFrame:
    """
    Create comprehensive validation summary table.
    
    Args:
        results: Dictionary with validation DataFrames
        
    Returns:
        Summary DataFrame with all validation metrics
    """
    # Start with permutation results
    if results['permutation'] is not None:
        summary = results['permutation'].copy()
    else:
        raise ValueError("Permutation results are required")
    
    # Merge cross-validation results
    if results['cross_validation'] is not None:
        cv_cols = ['state', 'canonical_variate', 'mean_r_oos', 'sd_r_oos', 
                   'in_sample_r', 'overfitting_index', 'n_valid_folds']
        cv_data = results['cross_validation'][cv_cols]
        
        summary = summary.merge(
            cv_data,
            on=['state', 'canonical_variate'],
            how='left'
        )
    
    # Merge CV significance results
    if results['cv_significance'] is not None:
        cv_sig_cols = ['state', 'canonical_variate', 'p_value_t_test', 
                       'p_value_wilcoxon', 'significant', 'interpretation']
        cv_sig_data = results['cv_significance'][cv_sig_cols]
        
        summary = summary.merge(
            cv_sig_data,
            on=['state', 'canonical_variate'],
            how='left',
            suffixes=('', '_cv_sig')
        )
    else:
        # Add placeholder columns
        summary['p_value_t_test'] = np.nan
        summary['p_value_wilcoxon'] = np.nan
        summary['significant'] = False
        summary['interpretation'] = 'Not Computed'
    
    # Merge redundancy results if available
    if results['redundancy'] is not None:
        # Filter out 'Total' rows for merging
        redundancy_data = results['redundancy'][
            results['redundancy']['canonical_variate'] != 'Total'
        ].copy()
        
        # Convert canonical_variate to int for merging
        redundancy_data['canonical_variate'] = redundancy_data['canonical_variate'].astype(int)
        
        redundancy_cols = ['state', 'canonical_variate', 'redundancy_Y_given_X', 
                          'redundancy_X_given_Y']
        
        summary = summary.merge(
            redundancy_data[redundancy_cols],
            on=['state', 'canonical_variate'],
            how='left'
        )
    else:
        # Add placeholder columns
        summary['redundancy_Y_given_X'] = np.nan
        summary['redundancy_X_given_Y'] = np.nan
    
    return summary


def add_decision_column(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add integrated decision column based on validation criteria.
    
    Decision criteria (updated with CV significance):
    - Accept: p_perm < 0.05 AND cv_p_value < 0.05 AND mean_r_oos > 0.3 AND redundancy > 0.10
    - Promising: p_perm < 0.15 AND cv_p_value < 0.05 AND mean_r_oos > 0.3
    - Caution: Meets some but not all criteria
    - Reject: Fails multiple criteria
    
    Args:
        summary: Validation summary DataFrame
        
    Returns:
        DataFrame with decision column added
    """
    decisions = []
    
    for _, row in summary.iterrows():
        p_perm = row['permutation_p_value']
        p_cv = row.get('p_value_t_test', np.nan)
        r_oos = row.get('mean_r_oos', np.nan)
        overfit = row.get('overfitting_index', np.nan)
        redundancy = row.get('redundancy_Y_given_X', np.nan)
        
        # Count criteria met
        criteria_met = 0
        criteria_total = 0
        
        # Permutation test
        if not pd.isna(p_perm):
            criteria_total += 1
            if p_perm < 0.05:
                criteria_met += 1
        
        # CV significance test
        if not pd.isna(p_cv):
            criteria_total += 1
            if p_cv < 0.05:
                criteria_met += 1
        
        # Cross-validation performance
        if not pd.isna(r_oos) and not pd.isna(overfit):
            criteria_total += 2
            if r_oos > 0.3:
                criteria_met += 1
            if overfit < 0.3:
                criteria_met += 1
        
        # Redundancy (if available)
        if not pd.isna(redundancy):
            criteria_total += 1
            if redundancy > 0.10:
                criteria_met += 1
        
        # Make decision
        if criteria_total == 0:
            decision = "Insufficient Data"
        elif criteria_met == criteria_total:
            decision = "✅ Accept"
        elif p_perm < 0.15 and p_cv < 0.05 and r_oos > 0.3:
            decision = "⚠️ Promising"
        elif criteria_met >= criteria_total / 2:
            decision = "⚠️ Caution"
        else:
            decision = "❌ Reject"
        
        # Special cases
        if not pd.isna(r_oos) and r_oos < 0:
            decision = "❌ Reject (Negative r_oos)"
        elif not pd.isna(overfit) and overfit > 0.5:
            decision = "❌ Reject (Severe Overfitting)"
        elif not pd.isna(p_cv) and p_cv >= 0.10:
            decision = "❌ Reject (No Significant Generalization)"
        
        decisions.append(decision)
    
    summary['decision'] = decisions
    return summary


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Format summary table for readability.
    
    Args:
        summary: Raw summary DataFrame
        
    Returns:
        Formatted DataFrame
    """
    # Reorder columns
    column_order = [
        'state',
        'canonical_variate',
        'observed_r',
        'permutation_p_value',
        'mean_r_oos',
        'sd_r_oos',
        'p_value_t_test',
        'significant',
        'overfitting_index',
        'redundancy_Y_given_X',
        'redundancy_X_given_Y',
        'n_valid_folds',
        'interpretation',
        'decision'
    ]
    
    # Select and reorder columns (only those that exist)
    available_cols = [col for col in column_order if col in summary.columns]
    formatted = summary[available_cols].copy()
    
    # Round numeric columns
    numeric_cols = ['observed_r', 'permutation_p_value', 'mean_r_oos', 'sd_r_oos',
                   'p_value_t_test', 'overfitting_index', 'redundancy_Y_given_X', 
                   'redundancy_X_given_Y']
    
    for col in numeric_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(3)
    
    # Rename columns for clarity
    formatted = formatted.rename(columns={
        'canonical_variate': 'CV',
        'observed_r': 'r_observed',
        'permutation_p_value': 'p_perm',
        'p_value_t_test': 'cv_p_value',
        'significant': 'cv_significant',
        'redundancy_Y_given_X': 'Redundancy_TET|Physio',
        'redundancy_X_given_Y': 'Redundancy_Physio|TET',
        'n_valid_folds': 'n_folds'
    })
    
    return formatted


def main():
    """Main execution function."""
    # Define paths
    results_dir = Path('results/tet/physio_correlation')
    output_file = results_dir / 'cca_validation_summary_table.csv'
    
    logger.info("=" * 80)
    logger.info("Creating CCA Validation Summary Table")
    logger.info("=" * 80)
    
    # Load validation results
    logger.info("\nLoading validation results...")
    results = load_validation_results(results_dir)
    
    # Create summary table
    logger.info("\nCreating validation summary...")
    summary = create_validation_summary(results)
    
    # Add decision column
    logger.info("Adding integrated decision column...")
    summary = add_decision_column(summary)
    
    # Format table
    logger.info("Formatting table...")
    formatted_summary = format_summary_table(summary)
    
    # Export
    logger.info(f"\nExporting to: {output_file}")
    formatted_summary.to_csv(output_file, index=False)
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY TABLE")
    logger.info("=" * 80)
    print("\n" + formatted_summary.to_string(index=False))
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    decision_counts = formatted_summary['decision'].value_counts()
    logger.info("\nDecision Distribution:")
    for decision, count in decision_counts.items():
        logger.info(f"  {decision}: {count}")
    
    # Highlight key findings
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)
    
    accept_or_promising = formatted_summary[
        formatted_summary['decision'].str.contains('Accept|Promising', na=False)
    ]
    
    if len(accept_or_promising) > 0:
        logger.info("\nCanonical variates with evidence for robust coupling:")
        for _, row in accept_or_promising.iterrows():
            logger.info(
                f"  {row['state']} CV{row['CV']}: "
                f"r = {row['r_observed']:.3f}, "
                f"p = {row['p_perm']:.3f}, "
                f"r_oos = {row['mean_r_oos']:.3f}, "
                f"overfitting = {row['overfitting_index']:.3f}"
            )
    else:
        logger.info("\nNo canonical variates met criteria for robust coupling.")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nValidation summary table saved to: {output_file}")


if __name__ == '__main__':
    main()
