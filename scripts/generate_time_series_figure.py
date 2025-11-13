#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Time Series Visualization Script

This script generates time series plots with statistical annotations showing
dose effects over time.

Usage:
    python generate_time_series_figure.py [options]
    
Example:
    python generate_time_series_figure.py --output results/tet/figures/time_series.png
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.time_series_visualizer import TETTimeSeriesVisualizer
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main workflow for time series visualization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate time series plots with statistical annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_time_series_figure.py
  python generate_time_series_figure.py --output results/tet/figures/time_series.png
  python generate_time_series_figure.py --dpi 600 --verbose
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data (default: results/tet/preprocessed/tet_preprocessed.csv)'
    )
    
    parser.add_argument(
        '--lme-results',
        type=str,
        default='results/tet/lme/lme_results.csv',
        help='Path to LME results (default: results/tet/lme/lme_results.csv)'
    )
    
    parser.add_argument(
        '--lme-contrasts',
        type=str,
        default='results/tet/lme/lme_contrasts.csv',
        help='Path to LME contrasts (default: results/tet/lme/lme_contrasts.csv)'
    )
    
    parser.add_argument(
        '--time-courses',
        type=str,
        default='results/tet/descriptive/time_course_all_dimensions.csv',
        help='Path to time course data (default: results/tet/descriptive/time_course_all_dimensions.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/figures/time_series_with_annotations.png',
        help='Output file path (default: results/tet/figures/time_series_with_annotations.png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution in DPI (default: 300)'
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
        print("TET TIME SERIES VISUALIZATION")
        print("=" * 80)
        
        # Check input files
        print(f"\n[1/4] Checking input files...")
        
        required_files = {
            'Preprocessed data': args.data,
            'LME results': args.lme_results,
            'LME contrasts': args.lme_contrasts,
            'Time courses': args.time_courses
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                print(f"      ❌ ERROR: {name} not found: {path}")
                return 1
            else:
                print(f"      ✓ {name}: {Path(path).name}")
        
        # Load data
        print(f"\n[2/4] Loading data...")
        logger.info("Loading data files...")
        
        data = pd.read_csv(args.data)
        print(f"      ✓ Preprocessed data: {len(data):,} rows")
        
        lme_results = pd.read_csv(args.lme_results)
        print(f"      ✓ LME results: {len(lme_results)} rows")
        
        lme_contrasts = pd.read_csv(args.lme_contrasts)
        print(f"      ✓ LME contrasts: {len(lme_contrasts)} rows")
        
        time_courses = pd.read_csv(args.time_courses)
        print(f"      ✓ Time courses: {len(time_courses):,} rows")
        
        # Create visualizer
        print(f"\n[3/4] Creating visualizer...")
        logger.info("Initializing visualizer...")
        
        visualizer = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)
        print(f"      ✓ Visualizer initialized")
        print(f"      ✓ {len(visualizer.dimensions)} dimensions to plot")
        print(f"      ✓ Dimensions ordered by State effect strength")
        print(f"      ✓ Strongest: {visualizer.ordered_dimensions[0]}")
        
        # Generate and export figure
        print(f"\n[4/4] Generating figure...")
        logger.info("Generating figure...")
        
        print(f"      - Creating multi-panel figure (5×3 grid)")
        print(f"      - Resolution: {args.dpi} DPI")
        print(f"      - This may take 10-20 seconds...")
        
        output_path = visualizer.export_figure(args.output, dpi=args.dpi)
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"      ✓ Figure generated and saved")
        print(f"      ✓ File size: {file_size_mb:.2f} MB")
        
        # Summary
        print(f"\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        print(f"\nFigure Components:")
        print(f"  - 15 dimension panels (5 rows × 3 columns)")
        print(f"  - Ordered by State effect strength (strongest first)")
        print(f"  - Blue line: 20mg dose (Low)")
        print(f"  - Red line: 40mg dose (High)")
        print(f"  - Grey shading: SEM around mean trajectories")
        print(f"  - Grey background: Significant DMT vs RS effect (p<0.05)")
        print(f"  - Black bars (top): Time bins with significant dose differences (p<0.05)")
        print(f"  - Dashed line: DMT onset")
        
        # Count time bins with dose differences
        n_dose_bins = visualizer.dose_interaction_bins['dose_effect_sig'].sum()
        n_total_bins = len(visualizer.dose_interaction_bins)
        print(f"\nStatistical Annotations:")
        print(f"  - {n_dose_bins}/{n_total_bins} time bins with significant dose differences")
        
        print(f"\nOutput:")
        print(f"  File: {output_path}")
        print(f"  Resolution: {args.dpi} DPI")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        print("\n" + "=" * 80)
        logger.info("Visualization complete")
        print("✓ Visualization complete!")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: File not found - {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: Visualization failed - {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
