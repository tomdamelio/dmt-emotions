#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Peak and AUC Analysis Script

This script computes peak values, time-to-peak, and area under curve (AUC) metrics
for TET dimensions and performs statistical comparisons between dose conditions.

Usage:
    python scripts/compute_peak_auc.py
    python scripts/compute_peak_auc.py --input results/tet/preprocessed/tet_preprocessed.csv
    python scripts/compute_peak_auc.py --output results/tet/peak_auc --verbose

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import sys
import os
import argparse
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import TET modules
from tet.peak_auc_analyzer import TETPeakAUCAnalyzer

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
    default_input = os.path.join(config.TET_RESULTS_DIR, 'preprocessed', 'tet_preprocessed.csv')
    default_output = os.path.join(config.TET_RESULTS_DIR, 'peak_auc')
    
    parser = argparse.ArgumentParser(
        description='Compute peak and AUC metrics for TET dimensions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default input file
  python scripts/compute_peak_auc.py
  
  # Specify custom input file
  python scripts/compute_peak_auc.py --input data/tet_preprocessed.csv
  
  # Specify custom output directory with verbose output
  python scripts/compute_peak_auc.py --output results/peak_auc --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=default_input,
        help=f'Path to preprocessed TET data CSV (default: {default_input})'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=default_output,
        help=f'Directory for output files (default: {default_output})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose console output'
    )
    
    return parser.parse_args()


def main():
    """
    Main peak and AUC analysis workflow.
    
    This function orchestrates the complete analysis pipeline:
    1. Load preprocessed TET data
    2. Initialize TETPeakAUCAnalyzer with z-scored dimensions
    3. Compute metrics (peak, time_to_peak, AUC) for all DMT sessions
    4. Perform statistical tests with bootstrap confidence intervals
    5. Export results to CSV files
    6. Print summary to console
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Print header
        print("=" * 80)
        print("TET PEAK AND AUC ANALYSIS")
        print("=" * 80)
        print()
        
        # Step 1: Load preprocessed data
        logger.info(f"Loading preprocessed data from: {args.input}")
        print(f"[1/4] Loading preprocessed data...")
        
        import pandas as pd
        data = pd.read_csv(args.input)
        
        print(f"      ✓ Loaded {len(data)} rows, {data['subject'].nunique()} subjects")
        print()
        
        # Step 2: Initialize analyzer
        logger.info("Initializing TETPeakAUCAnalyzer...")
        print("[2/4] Initializing analyzer...")
        
        # Get z-scored dimension columns
        z_dimensions = [col for col in data.columns if col.endswith('_z') 
                       and col.replace('_z', '') in config.TET_DIMENSION_COLUMNS]
        
        print(f"      ✓ Found {len(z_dimensions)} z-scored dimensions")
        
        analyzer = TETPeakAUCAnalyzer(data, z_dimensions)
        print()
        
        # Step 3: Compute metrics
        logger.info("Computing peak and AUC metrics...")
        print("[3/4] Computing metrics...")
        print("      - Peak values (maximum z-score)")
        print("      - Time to peak (minutes)")
        print("      - AUC 0-9 minutes (trapezoidal integration)")
        
        metrics = analyzer.compute_metrics()
        
        n_sessions = metrics.groupby(['subject', 'session']).ngroups
        print(f"      ✓ Computed metrics for {n_sessions} DMT sessions")
        print()
        
        # Step 4: Perform statistical tests
        logger.info("Performing statistical tests...")
        print("[4/4] Performing statistical tests...")
        print("      - Wilcoxon signed-rank tests (High vs Low dose)")
        print("      - Bootstrap confidence intervals (2000 iterations)")
        print("      - BH-FDR correction per metric type")
        
        results = analyzer.perform_tests()
        
        # Count significant results per metric
        if len(results) > 0:
            sig_by_metric = results.groupby('metric')['significant'].sum()
            total_by_metric = results.groupby('metric').size()
            
            print()
            print("      Statistical test results:")
            for metric in ['peak', 'time_to_peak_min', 'auc_0_9']:
                if metric in sig_by_metric.index:
                    n_sig = sig_by_metric[metric]
                    n_total = total_by_metric[metric]
                    pct = 100 * n_sig / n_total if n_total > 0 else 0
                    print(f"        {metric:20s}: {n_sig}/{n_total} significant ({pct:.1f}%)")
        
        print()
        
        # Step 5: Export results
        logger.info(f"Exporting results to: {args.output}")
        print("[5/5] Exporting results...")
        
        output_paths = analyzer.export_results(args.output)
        
        for file_type, path in output_paths.items():
            print(f"      ✓ {file_type}: {path}")
        
        print()
        
        # Print summary
        print("-" * 80)
        print("ANALYSIS SUMMARY")
        print("-" * 80)
        
        print(f"Subjects:        {data['subject'].nunique()}")
        print(f"DMT Sessions:    {n_sessions}")
        print(f"Dimensions:      {len(z_dimensions)}")
        print(f"Metrics:         3 (peak, time_to_peak, AUC)")
        print()
        
        if len(results) > 0:
            total_tests = len(results)
            total_significant = results['significant'].sum()
            pct_sig = 100 * total_significant / total_tests if total_tests > 0 else 0
            
            print(f"Statistical Tests:")
            print(f"  Total tests:   {total_tests}")
            print(f"  Significant:   {total_significant} ({pct_sig:.1f}%)")
            print()
            
            # Show top significant results by effect size
            sig_results = results[results['significant']].copy()
            if len(sig_results) > 0:
                sig_results = sig_results.sort_values('effect_r', ascending=False, key=abs)
                
                print("Top significant effects (by |effect_r|):")
                print()
                print(f"{'Dimension':<25} {'Metric':<20} {'Effect r':>10} {'p_fdr':>10}")
                print("-" * 70)
                
                for _, row in sig_results.head(10).iterrows():
                    dim_name = row['dimension'].replace('_z', '')
                    print(f"{dim_name:<25} {row['metric']:<20} "
                          f"{row['effect_r']:>10.3f} {row['p_fdr']:>10.4f}")
        else:
            print("No statistical tests performed (insufficient data)")
        
        print()
        print(f"Output files:")
        for file_type, path in output_paths.items():
            print(f"  - {path}")
        print()
        print("=" * 80)
        
        logger.info("Analysis complete")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n[ERROR] {e}")
        print("\nPlease check that the input file exists at the specified path.")
        print("You may need to run preprocess_tet_data.py first.")
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
    sys.exit(main())
