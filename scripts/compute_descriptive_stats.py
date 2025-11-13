#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Descriptive Statistics Script

This script computes descriptive statistics and time course summaries for TET data:
1. Group-level time courses (mean ± SEM by time bin)
2. Session-level summary metrics (peak, AUC, slopes)

Usage:
    python compute_descriptive_stats.py [options]
    
Example:
    python compute_descriptive_stats.py --input results/tet/preprocessed/tet_preprocessed.csv
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.time_course import TETTimeCourseAnalyzer
from tet.session_metrics import TETSessionMetrics
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main workflow for computing descriptive statistics.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compute descriptive statistics and time course summaries for TET data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_descriptive_stats.py
  python compute_descriptive_stats.py --input results/tet/preprocessed/tet_preprocessed.csv
  python compute_descriptive_stats.py --output-dir results/tet/descriptive --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data (default: results/tet/preprocessed/tet_preprocessed.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/tet/descriptive',
        help='Output directory for results (default: results/tet/descriptive)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    try:
        print("=" * 80)
        print("TET DESCRIPTIVE STATISTICS")
        print("=" * 80)
        
        # Load preprocessed data
        logger.info(f"Loading preprocessed data from: {args.input}")
        print(f"\n[1/4] Loading preprocessed data...")
        
        if not os.path.exists(args.input):
            print(f"❌ ERROR: Input file not found: {args.input}")
            print("   Run: python scripts/preprocess_tet_data.py")
            return 1
        
        data = pd.read_csv(args.input)
        print(f"      ✓ Loaded {len(data):,} rows, {len(data.columns)} columns")
        print(f"      ✓ {data['subject'].nunique()} subjects, {data.groupby(['subject', 'session_id']).ngroups} sessions")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Compute time courses
        print(f"\n[2/4] Computing group-level time courses...")
        logger.info("Computing time courses...")
        
        tc_analyzer = TETTimeCourseAnalyzer(data)
        print(f"      - Analyzing {len(tc_analyzer.dimensions)} dimensions")
        
        tc_path = tc_analyzer.export_time_courses(str(output_dir))
        print(f"      ✓ Saved: {Path(tc_path).name}")
        
        # Get time course summary
        tc_summary = tc_analyzer.get_summary_stats()
        print(f"      ✓ Computed {len(tc_summary)} time course summaries")
        
        # Compute session metrics
        print(f"\n[3/4] Computing session-level summary metrics...")
        logger.info("Computing session metrics...")
        
        sm_analyzer = TETSessionMetrics(data)
        print(f"      - Analyzing {len(sm_analyzer.dimensions)} dimensions")
        print(f"      - Computing peak, time_to_peak, AUC, slopes...")
        
        sm_path = sm_analyzer.export_session_metrics(str(output_dir))
        print(f"      ✓ Saved: {Path(sm_path).name}")
        
        # Get session metrics summary
        sm_summary = sm_analyzer.get_summary_stats()
        print(f"      ✓ Computed {len(sm_summary)} session metric summaries")
        
        # Print summary
        print(f"\n[4/4] Summary")
        print("-" * 80)
        
        # Load results to get counts
        time_courses = pd.read_csv(tc_path)
        session_metrics = pd.read_csv(sm_path)
        
        print(f"\nTime Courses:")
        print(f"  Total data points: {len(time_courses):,}")
        print(f"  Dimensions: {time_courses['dimension'].nunique()}")
        print(f"  States: {', '.join(time_courses['state'].unique())}")
        print(f"  Time bins per state:")
        for state in time_courses['state'].unique():
            n_bins = time_courses[time_courses['state'] == state]['t_bin'].nunique()
            print(f"    {state}: {n_bins} bins")
        
        print(f"\nSession Metrics:")
        print(f"  Total metrics: {len(session_metrics):,}")
        print(f"  Dimensions: {session_metrics['dimension'].nunique()}")
        print(f"  Sessions: {session_metrics.groupby(['subject', 'session_id']).ngroups}")
        print(f"  Metrics per session: peak_value, time_to_peak, auc_0_9min, slope_0_2min, slope_5_9min")
        
        print(f"\nOutput Files:")
        print(f"  {output_dir / 'time_course_all_dimensions.csv'}")
        print(f"  {output_dir / 'session_metrics_all_dimensions.csv'}")
        
        print("\n" + "=" * 80)
        logger.info("Descriptive statistics computation complete")
        print("✓ Descriptive statistics computation complete!")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: File not found - {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Computation failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: Computation failed - {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
