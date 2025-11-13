#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Data Validation Script

This script validates Temporal Experience Tracking (TET) data from DMT 
psychopharmacology studies. It performs comprehensive data quality checks
including session length validation, dimension range validation, and subject
completeness validation.

Usage:
    python scripts/validate_tet_data.py
    python scripts/validate_tet_data.py --input-file data/tet/tet_data.csv
    python scripts/validate_tet_data.py --output-dir results/tet/validation --verbose

Requirements: 1.1, 1.6
"""

import sys
import os
import argparse
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import TET modules
from tet.data_loader import TETDataLoader
from tet.validator import TETDataValidator
from tet.reporter import ValidationReporter

# Import configuration
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    # Default .mat directory
    default_mat_dir = os.path.join('..', 'data', 'original', 'reports', 'resampled')
    
    parser = argparse.ArgumentParser(
        description='Validate TET (Temporal Experience Tracking) data quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default .mat directory
  python scripts/validate_tet_data.py
  
  # Specify custom .mat directory
  python scripts/validate_tet_data.py --mat-dir ../data/tet/mat_files
  
  # Load from CSV file instead
  python scripts/validate_tet_data.py --input-file data/tet/tet_data.csv
  
  # Specify custom output directory with verbose output
  python scripts/validate_tet_data.py --output-dir results/validation --verbose
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Path to TET data CSV file (alternative to --mat-dir)'
    )
    
    parser.add_argument(
        '--mat-dir',
        type=str,
        default=default_mat_dir,
        help=f'Directory containing .mat files (default: {default_mat_dir})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(config.TET_RESULTS_DIR, 'validation'),
        help=f'Directory for validation reports (default: results/tet/validation)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose console output'
    )
    
    return parser.parse_args()


def main():
    """Main validation workflow.
    
    This function orchestrates the complete validation pipeline:
    1. Load TET data from CSV file
    2. Run all validation checks
    3. Apply clamping to out-of-range values if needed
    4. Generate comprehensive validation report
    5. Print summary to console
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    try:
        # Print header
        print("=" * 80)
        print("TET DATA VALIDATION")
        print("=" * 80)
        print()
        
        # Step 1: Load data
        if args.input_file:
            logger.info(f"Loading TET data from CSV: {args.input_file}")
            print(f"[*] Loading data from CSV: {args.input_file}")
            loader = TETDataLoader(file_path=args.input_file)
        else:
            logger.info(f"Loading TET data from .mat directory: {args.mat_dir}")
            print(f"[*] Loading data from .mat directory: {args.mat_dir}")
            loader = TETDataLoader(mat_dir=args.mat_dir)
        
        data = loader.load_data()
        
        print(f"[OK] Successfully loaded {len(data)} rows")
        print()
        
        # Step 2: Run validation
        logger.info("Running validation checks...")
        print("[*] Running validation checks...")
        print()
        
        validator = TETDataValidator(data, config.TET_DIMENSION_COLUMNS)
        validation_results = validator.validate_all()
        
        # Step 3: Apply clamping if needed
        range_violations = validation_results['range_violations']
        if isinstance(range_violations, pd.DataFrame) and not range_violations.empty:
            n_violations = len(range_violations)
            logger.info(f"Found {n_violations} out-of-range values, applying clamping...")
            print(f"[WARNING] Found {n_violations} out-of-range values")
            print("          Applying clamping to [0, 10] range...")
            
            data_clean, adjustments = validator.clamp_out_of_range_values()
            validation_results['adjustments_made'] = adjustments
            
            print(f"[OK] Clamped {len(adjustments)} values")
            print()
        else:
            logger.info("No out-of-range values found")
            print("[OK] All values within valid [0, 10] range")
            print()
        
        # Step 4: Generate report
        logger.info(f"Generating validation report in: {args.output_dir}")
        print(f"[*] Generating validation report...")
        
        reporter = ValidationReporter(validation_results, args.output_dir)
        report_path = reporter.generate_report()
        
        print(f"[OK] Report saved to: {report_path}")
        print()
        
        # Step 5: Print summary to console
        print("-" * 80)
        print("VALIDATION SUMMARY")
        print("-" * 80)
        
        summary = validation_results['summary']
        print(f"Subjects:    {summary['n_subjects']}")
        print(f"Sessions:    {summary['n_sessions']}")
        print(f"Time Bins:   {summary['n_time_bins']}")
        print(f"Dimensions:  {summary['n_dimensions']}")
        print()
        
        # Report issues found
        session_issues = validation_results['session_length_issues']
        completeness_issues = validation_results['completeness_issues']
        
        total_issues = (
            len(session_issues) + 
            len(completeness_issues) + 
            (len(range_violations) if not range_violations.empty else 0)
        )
        
        if total_issues == 0:
            print("[SUCCESS] No data quality issues found!")
        else:
            print(f"[WARNING] Found {total_issues} issue(s):")
            if session_issues:
                print(f"          - {len(session_issues)} session(s) with incorrect length")
            if completeness_issues:
                print(f"          - {len(completeness_issues)} subject(s) with incomplete data")
            if not range_violations.empty:
                print(f"          - {len(range_violations)} value(s) outside valid range (clamped)")
        
        print()
        print(f"Full report: {report_path}")
        print("=" * 80)
        
        logger.info("Validation complete")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] {e}")
        print("\nPlease check that the input file exists at the specified path.")
        return 1
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n[ERROR] {e}")
        print("\nPlease check that the input file has the correct format and columns.")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        print(f"\n[ERROR] Unexpected error: {type(e).__name__}")
        print(f"        {e}")
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    # Import pandas here to avoid import at module level
    import pandas as pd
    
    sys.exit(main())
