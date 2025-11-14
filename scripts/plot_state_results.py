# -*- coding: utf-8 -*-
"""
Plot State Clustering and Modelling Results

This script generates publication-ready figures for TET clustering and state modelling results:
1. KMeans centroid profiles (Fig. 3.5-like)
2. Time-course cluster probability plots (Fig. 3.6-like)
3. GLHMM state time-course plots
4. KMeans-GLHMM correspondence heatmap

Usage:
    python scripts/plot_state_results.py
    python scripts/plot_state_results.py --input-dir results/tet/clustering --output-dir results/tet/figures
    python scripts/plot_state_results.py --k 2 --include-rs --subset-glhmm-states 0 1
"""

import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime

# Add scripts/tet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tet'))

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_visualization import TETStateVisualization
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate state clustering and modelling visualization figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python scripts/plot_state_results.py

  # Specify custom input/output directories
  python scripts/plot_state_results.py \\
      --input-dir results/tet/clustering \\
      --output-dir results/tet/figures

  # Generate plots for k=2 clusters with RS comparison
  python scripts/plot_state_results.py \\
      --k 2 \\
      --include-rs

  # Plot subset of GLHMM states (e.g., S=2 solution)
  python scripts/plot_state_results.py \\
      --subset-glhmm-states 0 1

  # Generate all plots with high resolution
  python scripts/plot_state_results.py \\
      --dpi 600 \\
      --verbose
        """
    )
    
    parser.add_argument(
        '--preprocessed-data',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data (default: results/tet/preprocessed/tet_preprocessed.csv)'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='results/tet/clustering',
        help='Directory containing clustering results (default: results/tet/clustering)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/tet/figures',
        help='Output directory for figures (default: results/tet/figures)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='Number of KMeans clusters to plot (default: 2)'
    )
    
    parser.add_argument(
        '--include-rs',
        action='store_true',
        help='Include RS condition in time-course plots'
    )
    
    parser.add_argument(
        '--subset-glhmm-states',
        type=int,
        nargs='+',
        default=None,
        help='Subset of GLHMM states to plot (e.g., 0 1 for S=2 solution)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for saved figures (default: 300)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("TET STATE CLUSTERING VISUALIZATION")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Preprocessed data: {args.preprocessed_data}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"KMeans clusters (k): {args.k}")
    logger.info(f"Include RS: {args.include_rs}")
    logger.info(f"GLHMM state subset: {args.subset_glhmm_states}")
    logger.info(f"DPI: {args.dpi}")
    logger.info("=" * 80)
    
    # Load preprocessed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)
    
    if not os.path.exists(args.preprocessed_data):
        logger.error(f"Preprocessed data not found: {args.preprocessed_data}")
        logger.error("Please run preprocess_tet_data.py first.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(args.preprocessed_data)
        logger.info(f"Loaded preprocessed data: {len(data)} rows")
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        sys.exit(1)
    
    # Load KMeans assignments
    kmeans_path = os.path.join(args.input_dir, 'clustering_kmeans_assignments.csv')
    kmeans_assignments = None
    
    if os.path.exists(kmeans_path):
        try:
            kmeans_assignments = pd.read_csv(kmeans_path)
            logger.info(f"Loaded KMeans assignments: {len(kmeans_assignments)} rows")
        except Exception as e:
            logger.warning(f"Failed to load KMeans assignments: {e}")
    else:
        logger.warning(f"KMeans assignments not found: {kmeans_path}")
    
    # Load GLHMM Viterbi paths
    viterbi_path = os.path.join(args.input_dir, 'clustering_glhmm_viterbi.csv')
    glhmm_viterbi = None
    
    if os.path.exists(viterbi_path):
        try:
            glhmm_viterbi = pd.read_csv(viterbi_path)
            logger.info(f"Loaded GLHMM Viterbi paths: {len(glhmm_viterbi)} rows")
        except Exception as e:
            logger.warning(f"Failed to load GLHMM Viterbi paths: {e}")
    else:
        logger.warning(f"GLHMM Viterbi paths not found: {viterbi_path}")
    
    # Load GLHMM probabilities
    gamma_path = os.path.join(args.input_dir, 'clustering_glhmm_probabilities.csv')
    glhmm_probabilities = None
    
    if os.path.exists(gamma_path):
        try:
            glhmm_probabilities = pd.read_csv(gamma_path)
            logger.info(f"Loaded GLHMM probabilities: {len(glhmm_probabilities)} rows")
        except Exception as e:
            logger.warning(f"Failed to load GLHMM probabilities: {e}")
    else:
        logger.warning(f"GLHMM probabilities not found: {gamma_path}")
    
    # Initialize visualizer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: INITIALIZING VISUALIZER")
    logger.info("=" * 80)
    
    try:
        viz = TETStateVisualization(
            data=data,
            kmeans_assignments=kmeans_assignments,
            glhmm_viterbi=glhmm_viterbi,
            glhmm_probabilities=glhmm_probabilities
        )
    except Exception as e:
        logger.error(f"Failed to initialize visualizer: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    generated_files = []
    
    # 1. KMeans centroid profiles
    if kmeans_assignments is not None:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: GENERATING KMEANS CENTROID PROFILES")
        logger.info("=" * 80)
        
        try:
            centroid_path = viz.plot_kmeans_centroid_profiles(
                k=args.k,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            if centroid_path:
                generated_files.append(centroid_path)
                logger.info(f"✓ Generated centroid profile plot")
        except Exception as e:
            logger.error(f"Failed to generate centroid profiles: {e}")
    
    # 2. KMeans cluster time courses
    if kmeans_assignments is not None:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: GENERATING KMEANS TIME-COURSE PLOTS")
        logger.info("=" * 80)
        
        try:
            timecourse_path = viz.plot_kmeans_cluster_timecourses(
                k=args.k,
                include_rs=args.include_rs,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            if timecourse_path:
                generated_files.append(timecourse_path)
                logger.info(f"✓ Generated cluster time-course plot")
        except Exception as e:
            logger.error(f"Failed to generate cluster time courses: {e}")
    
    # 3. GLHMM state time courses
    if glhmm_probabilities is not None:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: GENERATING GLHMM STATE TIME-COURSE PLOTS")
        logger.info("=" * 80)
        
        try:
            states_to_plot = ['RS', 'DMT'] if args.include_rs else ['DMT']
            
            glhmm_path = viz.plot_glhmm_state_timecourses(
                states_to_plot=states_to_plot,
                subset_states=args.subset_glhmm_states,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            if glhmm_path:
                generated_files.append(glhmm_path)
                logger.info(f"✓ Generated GLHMM state time-course plot")
        except Exception as e:
            logger.error(f"Failed to generate GLHMM time courses: {e}")
    
    # 4. KMeans-GLHMM correspondence
    if kmeans_assignments is not None and glhmm_viterbi is not None:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: GENERATING KMEANS-GLHMM CORRESPONDENCE")
        logger.info("=" * 80)
        
        try:
            fig_path, csv_path = viz.plot_kmeans_glhmm_crosswalk(
                k=args.k,
                output_dir=args.output_dir,
                dpi=args.dpi
            )
            if fig_path:
                generated_files.append(fig_path)
                logger.info(f"✓ Generated correspondence heatmap")
            if csv_path:
                generated_files.append(csv_path)
                logger.info(f"✓ Generated correspondence table")
        except Exception as e:
            logger.error(f"Failed to generate correspondence plots: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nGenerated {len(generated_files)} output files:")
    for path in generated_files:
        logger.info(f"  {path}")
    
    logger.info(f"\nOutput directory: {args.output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nVisualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
