#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Time Series Plotting Script

This script generates annotated time series plots showing dose effects over time
with statistical annotations (Requirement 8.1).

Usage:
    python plot_time_series.py [options]
    
Example:
    python plot_time_series.py --output results/tet/figures/timeseries_all_dimensions.png
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


def plot_annotated_time_series(
    data_path: str,
    lme_results_path: str,
    lme_contrasts_path: str,
    time_courses_path: str,
    output_path: str,
    dpi: int = 300
) -> str:
    """
    Generate annotated time series plots.
    
    This function creates publication-ready time series figures showing:
    - Mean trajectories with SEM shading for Low (20mg) and High (40mg) doses
    - RS baseline as reference point
    - Grey background shading for time bins where DMT differs from RS baseline
    - Black horizontal bars for time bins with significant State:Dose interactions
    - Dimensions ordered by strength of State main effect
    
    Args:
        data_path (str): Path to preprocessed TET data
        lme_results_path (str): Path to LME results CSV
        lme_contrasts_path (str): Path to LME contrasts CSV
        time_courses_path (str): Path to time course data CSV
        output_path (str): Output file path for figure
        dpi (int): Resolution in dots per inch (default: 300)
        
    Returns:
        str: Path to exported figure
        
    Example:
        >>> plot_annotated_time_series(
        ...     'results/tet/preprocessed/tet_preprocessed.csv',
        ...     'results/tet/lme/lme_results.csv',
        ...     'results/tet/lme/lme_contrasts.csv',
        ...     'results/tet/descriptive/time_course_all_dimensions.csv',
        ...     'results/tet/figures/timeseries_all_dimensions.png'
        ... )
    """
    logger.info("Loading data files...")
    
    # Load data
    data = pd.read_csv(data_path)
    lme_results = pd.read_csv(lme_results_path)
    lme_contrasts = pd.read_csv(lme_contrasts_path)
    time_courses = pd.read_csv(time_courses_path)
    
    logger.info(f"  Preprocessed data: {len(data):,} rows")
    logger.info(f"  LME results: {len(lme_results)} rows")
    logger.info(f"  Time courses: {len(time_courses):,} rows")
    
    # Create visualizer
    logger.info("Initializing visualizer...")
    visualizer = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)
    
    logger.info(f"  {len(visualizer.dimensions)} dimensions to plot")
    logger.info(f"  Dimensions ordered by State effect strength")
    
    # Generate and export figure
    logger.info("Generating figure...")
    output_path = visualizer.export_figure(output_path, dpi=dpi)
    
    logger.info(f"Figure exported to: {output_path}")
    
    return output_path


def main():
    """
    Main workflow for time series visualization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate annotated time series plots (Requirement 8.1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_time_series.py
  python plot_time_series.py --output results/tet/figures/timeseries_all_dimensions.png
  python plot_time_series.py --dpi 600 --verbose
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
        default='results/tet/figures/timeseries_all_dimensions.png',
        help='Output file path (default: results/tet/figures/timeseries_all_dimensions.png)'
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
        print("TET TIME SERIES VISUALIZATION (Requirement 8.1)")
        print("=" * 80)
        
        # Check input files
        print(f"\n[1/3] Checking input files...")
        
        required_files = {
            'Preprocessed data': args.data,
            'LME results': args.lme_results,
            'LME contrasts': args.lme_contrasts,
            'Time courses': args.time_courses
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                print(f"      ❌ ERROR: {name} not found: {path}")
                print(f"\n   Required files:")
                print(f"   1. Run: python scripts/preprocess_tet_data.py")
                print(f"   2. Run: python scripts/compute_descriptive_stats.py")
                print(f"   3. Run: python scripts/fit_lme_models.py")
                return 1
            else:
                print(f"      ✓ {name}: {Path(path).name}")
        
        # Generate figure
        print(f"\n[2/3] Generating annotated time series figure...")
        print(f"      - Custom layout: 2 large panels (Arousal, Valence) + 5 small panels")
        print(f"      - Row 1: Arousal (Emotional Intensity), Valence (Pleasantness-Unpleasantness)")
        print(f"      - Row 2: Interoception, Anxiety, Unpleasantness, Pleasantness, Bliss")
        print(f"      - Resolution: {args.dpi} DPI")
        print(f"      - Blue line: Low dose (20mg)")
        print(f"      - Red line: High dose (40mg)")
        print(f"      - Grey shading: SEM around mean trajectories")
        print(f"      - Grey background: Significant DMT vs RS effect")
        print(f"      - Black bars: Significant dose differences at specific time bins")
        print(f"      - This may take 10-20 seconds...")
        
        output_path = plot_annotated_time_series(
            args.data,
            args.lme_results,
            args.lme_contrasts,
            args.time_courses,
            args.output,
            args.dpi
        )
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"      ✓ Figure generated and saved")
        print(f"      ✓ File size: {file_size_mb:.2f} MB")
        
        # Summary
        print(f"\n[3/3] Summary")
        print("-" * 80)
        
        print(f"\nOutput:")
        print(f"  File: {output_path}")
        print(f"  Resolution: {args.dpi} DPI")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        print(f"\nFigure Components:")
        print(f"  ✓ Custom 2-row layout:")
        print(f"    - Row 1: Arousal (Emotional Intensity), Valence (Pleasantness-Unpleasantness)")
        print(f"    - Row 2: Interoception, Anxiety, Unpleasantness, Pleasantness, Bliss")
        print(f"  ✓ Low dose (20mg) in blue, High dose (40mg) in red")
        print(f"  ✓ SEM shading around mean trajectories")
        print(f"  ✓ Grey dashed line at DMT onset (t=0)")
        print(f"  ✓ Grey background for significant DMT vs RS effects")
        print(f"  ✓ Black bars for significant dose differences")
        print(f"  ✓ X-axis: Time in minutes (0-20 for DMT)")
        print(f"  ✓ Y-axis: Z-scored intensity")
        
        print("\n" + "=" * 80)
        logger.info("Visualization complete")
        print("✓ Time series visualization complete!")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: File not found - {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: Visualization failed - {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
