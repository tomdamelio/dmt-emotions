#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for CCA data validator.

This script tests the CCADataValidator class with actual merged data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import logging
from scripts.tet.cca_data_validator import CCADataValidator


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_validator():
    """Test CCA data validator with actual data."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Testing CCA Data Validator")
    logger.info("=" * 80)
    
    # Load merged data
    merged_data_path = 'results/tet/physio_correlation/merged_physio_tet_data.csv'
    
    if not Path(merged_data_path).exists():
        logger.error(f"Merged data file not found: {merged_data_path}")
        logger.info("Please run the physiological correlation analysis first.")
        return
    
    logger.info(f"\nLoading merged data from {merged_data_path}...")
    merged_df = pd.read_csv(merged_data_path)
    logger.info(f"  Loaded {len(merged_df)} observations")
    logger.info(f"  Columns: {list(merged_df.columns)}")
    
    # Initialize validator
    logger.info("\nInitializing CCA data validator...")
    validator = CCADataValidator(merged_df)
    
    # Test temporal resolution validation
    logger.info("\n" + "-" * 80)
    logger.info("Test 1: Temporal Resolution Validation")
    logger.info("-" * 80)
    try:
        tres_result = validator.validate_temporal_resolution()
        logger.info("✓ Temporal resolution validation passed")
        logger.info(f"  Resolution: {tres_result['resolution_seconds']:.1f}s")
        logger.info(f"  Bins per session: {tres_result['bins_per_session']:.1f}")
    except ValueError as e:
        logger.error(f"✗ Temporal resolution validation failed: {e}")
    
    # Test sample size validation
    logger.info("\n" + "-" * 80)
    logger.info("Test 2: Sample Size Validation")
    logger.info("-" * 80)
    try:
        ssize_result = validator.validate_sample_size()
        logger.info("✓ Sample size validation passed")
        logger.info(f"  Subjects: {ssize_result['n_subjects']}")
        logger.info(f"  Total observations: {ssize_result['n_total_obs']}")
        logger.info(f"  Obs per subject: {ssize_result['n_obs_per_subject']:.1f}")
    except ValueError as e:
        logger.error(f"✗ Sample size validation failed: {e}")
    
    # Test data structure audit
    logger.info("\n" + "-" * 80)
    logger.info("Test 3: Data Structure Audit")
    logger.info("-" * 80)
    audit_df = validator.audit_data_structure()
    logger.info("✓ Data structure audit completed")
    logger.info(f"  Subjects audited: {len(audit_df)}")
    logger.info(f"  Mean completeness: {audit_df['completeness_rate'].mean():.1%}")
    
    # Display subjects with incomplete data
    incomplete = audit_df[audit_df['completeness_rate'] < 1.0]
    if len(incomplete) > 0:
        logger.info(f"  Subjects with incomplete data: {len(incomplete)}")
        logger.info("\n  Details:")
        for _, row in incomplete.iterrows():
            logger.info(
                f"    {row['subject']}: {row['completeness_rate']:.1%} complete "
                f"({row['n_complete']}/{row['n_obs']} obs)"
            )
    
    # Generate validation report
    logger.info("\n" + "-" * 80)
    logger.info("Test 4: Generate Validation Report")
    logger.info("-" * 80)
    
    output_dir = Path('test/tet')
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'cca_validation_test_report.txt'
    
    validation_report = validator.generate_validation_report(str(report_path))
    
    logger.info(f"✓ Validation report generated: {report_path}")
    logger.info(f"  Overall status: {'VALID' if validation_report['overall_status']['is_valid'] else 'INVALID'}")
    
    # Display report summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    
    if validation_report['overall_status']['is_valid']:
        logger.info("✓ ALL CHECKS PASSED - Data ready for CCA")
    else:
        logger.info("✗ VALIDATION FAILED - Address issues before CCA")
    
    logger.info("\nDetailed results:")
    logger.info(f"  Temporal resolution: {'✓ VALID' if validation_report['temporal_resolution']['is_valid'] else '✗ INVALID'}")
    logger.info(f"  Sample size: {'✓ SUFFICIENT' if validation_report['sample_size']['is_sufficient'] else '✗ INSUFFICIENT'}")
    logger.info(f"  Data completeness: {validation_report['data_structure']['mean_completeness_rate']:.1%}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Test completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    setup_logging()
    test_validator()
