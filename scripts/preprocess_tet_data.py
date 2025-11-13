#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Data Preprocessing Script

This script preprocesses Temporal Experience Tracking (TET) data including
session trimming, within-subject standardization, and composite index creation.

Usage:
    python scripts/preprocess_tet_data.py
    python scripts/preprocess_tet_data.py --mat-dir ../data/original/reports/resampled
    python scripts/preprocess_tet_data.py --output-dir results/tet/preprocessed --verbose

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
"""

import sys
import os
import argparse
import logging
import json

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import TET modules
from tet.data_loader import TETDataLoader
from tet.validator import TETDataValidator
from tet.preprocessor import TETPreprocessor
from tet.metadata import PreprocessingMetadata

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
    default_mat_dir = os.path.join('..', 'data', 'original', 'reports', 'resampled')
    default_output_dir = os.path.join(config.TET_RESULTS_DIR, 'preprocessed')
    
    parser = argparse.ArgumentParser(
        description='Preprocess TET (Temporal Experience Tracking) data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default .mat directory
  python scripts/preprocess_tet_data.py
  
  # Specify custom .mat directory
  python scripts/preprocess_tet_data.py --mat-dir ../data/tet/mat_files
  
  # Specify custom output directory with verbose output
  python scripts/preprocess_tet_data.py --output-dir results/preprocessed --verbose
        """
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
        default=default_output_dir,
        help=f'Directory for preprocessed outputs (default: {default_output_dir})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose console output'
    )
    
    return parser.parse_args()


def main():
    """
    Main preprocessing workflow.
    
    This function orchestrates the complete preprocessing pipeline:
    1. Load TET data from .mat files
    2. Validate data quality
    3. Apply clamping to out-of-range values if needed
    4. Trim sessions to analysis windows
    5. Standardize within subjects (global z-scores)
    6. Create composite indices
    7. Generate metadata
    8. Save preprocessed data and metadata
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Print header
        print("=" * 80)
        print("TET DATA PREPROCESSING")
        print("=" * 80)
        print()
        
        # Step 1: Load data
        logger.info(f"Loading TET data from .mat directory: {args.mat_dir}")
        print(f"[1/6] Loading data from .mat directory: {args.mat_dir}")
        
        loader = TETDataLoader(mat_dir=args.mat_dir)
        data = loader.load_data()
        
        print(f"      ✓ Loaded {len(data)} rows, {data['subject'].nunique()} subjects")
        print()
        
        # Step 2: Validate data
        logger.info("Validating data quality...")
        print("[2/6] Validating data quality...")
        
        validator = TETDataValidator(data, config.TET_DIMENSION_COLUMNS)
        validation_results = validator.validate_all()
        
        # Apply clamping if needed
        range_violations = validation_results['range_violations']
        if isinstance(range_violations, pd.DataFrame) and not range_violations.empty:
            n_violations = len(range_violations)
            logger.info(f"Found {n_violations} out-of-range values, applying clamping...")
            print(f"      ⚠ Found {n_violations} out-of-range values, clamping...")
            
            data, adjustments = validator.clamp_out_of_range_values()
            print(f"      ✓ Clamped {len(adjustments)} values")
        else:
            logger.info("No out-of-range values found")
            print("      ✓ All values within valid range")
        
        print()
        
        # Step 3: Preprocess data
        logger.info("Preprocessing data...")
        print("[3/6] Preprocessing data...")
        print("      - Trimming sessions to analysis windows")
        print("      - Creating valence variables")
        print("      - Standardizing within subjects (global z-scores)")
        print("      - Creating composite indices")
        
        preprocessor = TETPreprocessor(data, config.TET_DIMENSION_COLUMNS)
        data_preprocessed = preprocessor.preprocess_all()
        
        print(f"      ✓ Preprocessing complete: {len(data_preprocessed)} rows, "
              f"{len(data_preprocessed.columns)} columns")
        print()
        
        # Step 4: Generate metadata
        logger.info("Generating metadata...")
        print("[4/6] Generating metadata...")
        
        metadata_gen = PreprocessingMetadata()
        metadata = metadata_gen.generate_metadata(data_preprocessed)
        
        print("      ✓ Metadata generated")
        print()
        
        # Step 5: Save preprocessed data
        logger.info(f"Saving preprocessed data to: {args.output_dir}")
        print("[5/6] Saving preprocessed data...")
        
        output_csv = os.path.join(args.output_dir, 'tet_preprocessed.csv')
        data_preprocessed.to_csv(output_csv, index=False)
        
        print(f"      ✓ Saved to: {output_csv}")
        print()
        
        # Step 6: Save metadata
        logger.info("Saving metadata...")
        print("[6/6] Saving metadata...")
        
        metadata_json = os.path.join(args.output_dir, 'preprocessing_metadata.json')
        with open(metadata_json, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      ✓ Saved to: {metadata_json}")
        print()
        
        # Print summary
        print("-" * 80)
        print("PREPROCESSING SUMMARY")
        print("-" * 80)
        
        summary = metadata['data_summary']
        print(f"Subjects:     {summary['n_subjects']}")
        print(f"Sessions:     {summary['n_sessions']}")
        print(f"Time Points:  {summary['n_time_points_total']}")
        print(f"  - RS:       {summary['n_time_points_rs']}")
        print(f"  - DMT:      {summary['n_time_points_dmt']}")
        print(f"Dimensions:   {summary['n_dimensions']}")
        print()
        
        print("New columns created:")
        print(f"  - Z-scored dimensions: {summary['n_dimensions']} columns (suffix _z)")
        print(f"  - Valence variables: 2 columns (valence_pos, valence_neg)")
        print(f"  - Composite indices: 3 columns (affect_index_z, imagery_index_z, self_index_z)")
        print()
        
        print(f"Output files:")
        print(f"  - {output_csv}")
        print(f"  - {metadata_json}")
        print()
        print("=" * 80)
        
        logger.info("Preprocessing complete")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] {e}")
        print("\nPlease check that the input directory exists at the specified path.")
        return 1
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n[ERROR] {e}")
        print("\nPlease check that the input files have the correct format and columns.")
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
