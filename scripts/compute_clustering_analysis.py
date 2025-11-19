# -*- coding: utf-8 -*-
"""
Compute Clustering and State Modelling Analysis for TET Data

This script performs clustering and state modelling analysis on preprocessed TET data:
1. Fits KMeans clustering (k = 2, 3, 4)
2. Fits GLHMM models (S = 2, 3, 4)
3. Evaluates models and selects optimal k and S
4. Performs bootstrap stability analysis
5. Computes state occupancy metrics
6. Performs statistical tests for dose effects
7. Evaluates interaction effects (State × Dose)

Usage:
    python scripts/compute_clustering_analysis.py --input results/tet/tet_preprocessed.csv --output results/tet/clustering
    python scripts/compute_clustering_analysis.py --input results/tet/tet_preprocessed.csv --output results/tet/clustering --n-bootstrap 500 --n-permutations 500
    python scripts/compute_clustering_analysis.py --input results/tet/tet_preprocessed.csv --output results/tet/clustering --skip-glhmm-permutation
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add scripts/tet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tet'))

# Add project root to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_model_analyzer import TETStateModelAnalyzer
from state_dose_analyzer import TETStateDoseAnalyzer
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
        description='Compute clustering and state modelling analysis for TET data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python scripts/compute_clustering_analysis.py

  # Specify custom input/output paths
  python scripts/compute_clustering_analysis.py \\
      --input results/tet/tet_preprocessed.csv \\
      --output results/tet/clustering

  # Faster debug run with fewer iterations
  python scripts/compute_clustering_analysis.py \\
      --n-bootstrap 100 \\
      --n-permutations 100 \\
      --skip-glhmm-permutation

  # Full analysis with verbose output
  python scripts/compute_clustering_analysis.py \\
      --n-bootstrap 1000 \\
      --n-permutations 1000 \\
      --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data CSV file (default: results/tet/preprocessed/tet_preprocessed.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/clustering',
        help='Output directory for clustering results (default: results/tet/clustering)'
    )
    
    parser.add_argument(
        '--state-values',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='Number of states to test for clustering (default: 2 3 4)'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap iterations for stability analysis (default: 1000)'
    )
    
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutation iterations for statistical tests (default: 1000)'
    )
    
    parser.add_argument(
        '--skip-glhmm-permutation',
        action='store_true',
        help='Skip GLHMM permutation tests (for faster debug runs)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=22,
        help='Random seed for reproducibility (default: 22)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip comprehensive report generation (for faster debugging)'
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
    logger.info("TET CLUSTERING AND STATE MODELLING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"State values to test: {args.state_values}")
    logger.info(f"Bootstrap iterations: {args.n_bootstrap}")
    logger.info(f"Permutation iterations: {args.n_permutations}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info(f"Skip GLHMM permutation: {args.skip_glhmm_permutation}")
    logger.info("=" * 80)
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load preprocessed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING PREPROCESSED DATA")
    logger.info("=" * 80)
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.error("Please run preprocess_tet_data.py first to generate preprocessed data.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(data)} rows from {args.input}")
        logger.info(f"  Subjects: {data['subject'].nunique()}")
        logger.info(f"  Sessions: {data['session_id'].nunique()}")
        logger.info(f"  States: {data['state'].unique()}")
        logger.info(f"  Doses: {data['dose'].unique()}")
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        sys.exit(1)
    
    # Identify z-scored dimensions
    z_dimensions = [col for col in data.columns if col.endswith('_z') and 
                   col not in ['valence_index_z']]
    
    logger.info(f"  Z-scored dimensions: {len(z_dimensions)}")
    logger.info(f"    {', '.join(z_dimensions[:5])}...")
    
    # Initialize state model analyzer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: INITIALIZING STATE MODEL ANALYZER")
    logger.info("=" * 80)
    
    analyzer = TETStateModelAnalyzer(
        data=data,
        dimensions=z_dimensions,
        subject_id_col='subject',
        session_id_col='session_id',
        time_col='t_bin',
        state_values=args.state_values,
        random_seed=args.random_seed
    )
    
    # Fit KMeans clustering
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: FITTING KMEANS CLUSTERING")
    logger.info("=" * 80)
    
    try:
        kmeans_results = analyzer.fit_kmeans()
        logger.info(f"Successfully fitted KMeans for k={list(kmeans_results.keys())}")
    except Exception as e:
        logger.error(f"KMeans fitting failed: {e}")
        logger.error("Continuing without KMeans results...")
        kmeans_results = {}
    
    # Fit GLHMM models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: FITTING GLHMM MODELS")
    logger.info("=" * 80)
    logger.info("Note: GLHMM fitting requires the glhmm library.")
    logger.info("      If not installed, this step will be skipped.")
    
    try:
        glhmm_results = analyzer.fit_glhmm()
        if len(glhmm_results) > 0:
            logger.info(f"Successfully fitted GLHMM for S={list(glhmm_results.keys())}")
        else:
            logger.warning("GLHMM fitting returned no results (library may not be available)")
    except Exception as e:
        logger.error(f"GLHMM fitting failed: {e}")
        logger.error("Continuing without GLHMM results...")
        glhmm_results = {}
    
    # Evaluate models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: EVALUATING MODELS")
    logger.info("=" * 80)
    
    try:
        evaluation = analyzer.evaluate_models()
        logger.info(f"Model evaluation complete:")
        logger.info(f"\n{evaluation.to_string()}")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        sys.exit(1)
    
    # Select optimal models
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: SELECTING OPTIMAL MODELS")
    logger.info("=" * 80)
    
    try:
        optimal_k, optimal_S = analyzer.select_optimal_models()
        logger.info(f"Optimal KMeans: k={optimal_k}")
        logger.info(f"Optimal GLHMM: S={optimal_S}")
    except Exception as e:
        logger.error(f"Model selection failed: {e}")
        sys.exit(1)
    
    # Perform bootstrap stability analysis
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: BOOTSTRAP STABILITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Performing {args.n_bootstrap} bootstrap iterations...")
    logger.info("This may take several minutes...")
    
    try:
        stability = analyzer.bootstrap_stability(n_bootstrap=args.n_bootstrap)
        logger.info(f"Bootstrap stability analysis complete:")
        logger.info(f"\n{stability.to_string()}")
    except Exception as e:
        logger.error(f"Bootstrap stability analysis failed: {e}")
        logger.error("Continuing without stability results...")
        stability = None
    
    # Compute state occupancy metrics
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: COMPUTING STATE OCCUPANCY METRICS")
    logger.info("=" * 80)
    
    try:
        metrics = analyzer.compute_state_metrics()
        logger.info(f"Computed state metrics:")
        logger.info(f"  Total rows: {len(metrics)}")
        logger.info(f"  Methods: {metrics['method'].unique()}")
        logger.info(f"  Metrics: fractional_occupancy, n_visits, mean_dwell_time")
    except Exception as e:
        logger.error(f"State metrics computation failed: {e}")
        sys.exit(1)
    
    # Export clustering results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: EXPORTING CLUSTERING RESULTS")
    logger.info("=" * 80)
    
    try:
        clustering_paths = analyzer.export_results(args.output)
        logger.info(f"Exported {len(clustering_paths)} clustering result files:")
        for file_type, path in clustering_paths.items():
            logger.info(f"  {file_type}: {path}")
    except Exception as e:
        logger.error(f"Failed to export clustering results: {e}")
        sys.exit(1)
    
    # Initialize statistical analyzer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 10: INITIALIZING STATISTICAL ANALYZER")
    logger.info("=" * 80)
    
    # Load GLHMM probabilities if available
    glhmm_probs = None
    glhmm_probs_path = clustering_paths.get('glhmm_probabilities')
    if glhmm_probs_path and os.path.exists(glhmm_probs_path):
        try:
            glhmm_probs = pd.read_csv(glhmm_probs_path)
            logger.info(f"Loaded GLHMM probabilities: {len(glhmm_probs)} rows")
        except Exception as e:
            logger.warning(f"Failed to load GLHMM probabilities: {e}")
    
    stat_analyzer = TETStateDoseAnalyzer(
        state_metrics=metrics,
        cluster_probabilities=glhmm_probs
    )
    
    # Compute classical paired t-tests
    logger.info("\n" + "=" * 80)
    logger.info("STEP 11: CLASSICAL PAIRED T-TESTS")
    logger.info("=" * 80)
    
    try:
        classical_results = stat_analyzer.compute_pairwise_tests()
        logger.info(f"Completed {len(classical_results)} classical t-tests")
    except Exception as e:
        logger.error(f"Classical t-tests failed: {e}")
        logger.error("Continuing without classical test results...")
        classical_results = None
    
    # Apply permutation tests
    logger.info("\n" + "=" * 80)
    logger.info("STEP 12: PERMUTATION TESTS")
    logger.info("=" * 80)
    logger.info(f"Performing {args.n_permutations} permutation iterations...")
    logger.info("This may take several minutes...")
    
    if args.skip_glhmm_permutation:
        logger.info("Skipping GLHMM permutation tests (--skip-glhmm-permutation flag set)")
        perm_results = None
    else:
        try:
            perm_results = stat_analyzer.apply_glhmm_permutation_test(
                n_permutations=args.n_permutations,
                random_seed=args.random_seed
            )
            logger.info(f"Completed {len(perm_results)} permutation tests")
        except Exception as e:
            logger.error(f"Permutation tests failed: {e}")
            logger.error("Continuing without permutation test results...")
            perm_results = None
    
    # Evaluate interaction effects
    logger.info("\n" + "=" * 80)
    logger.info("STEP 13: INTERACTION EFFECTS (STATE × DOSE)")
    logger.info("=" * 80)
    logger.info(f"Performing {args.n_permutations} permutation iterations...")
    
    if args.skip_glhmm_permutation:
        logger.info("Skipping interaction tests (--skip-glhmm-permutation flag set)")
        interaction_results = None
    else:
        try:
            interaction_results = stat_analyzer.evaluate_interaction_effects(
                n_permutations=args.n_permutations,
                random_seed=args.random_seed
            )
            logger.info(f"Completed {len(interaction_results)} interaction tests")
        except Exception as e:
            logger.error(f"Interaction tests failed: {e}")
            logger.error("Continuing without interaction test results...")
            interaction_results = None
    
    # Apply FDR correction
    logger.info("\n" + "=" * 80)
    logger.info("STEP 14: FDR CORRECTION")
    logger.info("=" * 80)
    
    try:
        stat_analyzer.apply_fdr_correction(alpha=0.05)
        logger.info("FDR correction applied to all test results")
    except Exception as e:
        logger.error(f"FDR correction failed: {e}")
        logger.error("Continuing without FDR correction...")
    
    # Export statistical test results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 15: EXPORTING STATISTICAL TEST RESULTS")
    logger.info("=" * 80)
    
    try:
        stat_paths = stat_analyzer.export_results(args.output)
        logger.info(f"Exported {len(stat_paths)} statistical result files:")
        for file_type, path in stat_paths.items():
            logger.info(f"  {file_type}: {path}")
    except Exception as e:
        logger.error(f"Failed to export statistical results: {e}")
        logger.error("Some results may not have been saved.")
    
    # Generate comprehensive report (if not skipped)
    if not args.skip_report:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 16: GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 80)
        
        try:
            # Import report generator
            sys.path.insert(0, str(Path(__file__).parent))
            from generate_comprehensive_report import main as generate_report_main
            
            # Save original sys.argv
            original_argv = sys.argv.copy()
            
            # Set up arguments for report generation
            sys.argv = [
                'generate_comprehensive_report.py',
                '--results-dir', str(Path(args.output).parent),
                '--output', 'docs/tet_comprehensive_results.md'
            ]
            
            if args.verbose:
                sys.argv.append('--verbose')
            
            # Generate report
            report_exit_code = generate_report_main()
            
            # Restore original sys.argv
            sys.argv = original_argv
            
            if report_exit_code == 0:
                logger.info("✓ Comprehensive report generated successfully")
            else:
                logger.warning("⚠ Report generation completed with warnings")
                
        except Exception as e:
            logger.error(f"✗ Failed to generate comprehensive report: {e}")
            logger.info("Continuing without report generation...")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("Skipping comprehensive report generation (--skip-report flag)")
        logger.info("=" * 80)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nClustering Results:")
    logger.info(f"  Optimal KMeans: k={optimal_k}")
    logger.info(f"  Optimal GLHMM: S={optimal_S}")
    
    if stability is not None and len(stability) > 0:
        logger.info(f"\nStability (Bootstrap ARI):")
        for _, row in stability.iterrows():
            logger.info(f"  {row['method']} (n={row['n_states']}): "
                       f"ARI = {row['mean_ari']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]")
    
    if classical_results is not None and len(classical_results) > 0:
        n_sig_classical = np.sum(classical_results.get('significant', False))
        logger.info(f"\nClassical T-Tests:")
        logger.info(f"  Total tests: {len(classical_results)}")
        logger.info(f"  Significant (FDR < 0.05): {n_sig_classical}")
    
    if perm_results is not None and len(perm_results) > 0:
        n_sig_perm = np.sum(perm_results.get('significant', False))
        logger.info(f"\nPermutation Tests:")
        logger.info(f"  Total tests: {len(perm_results)}")
        logger.info(f"  Significant (FDR < 0.05): {n_sig_perm}")
    
    if interaction_results is not None and len(interaction_results) > 0:
        n_sig_interaction = np.sum(interaction_results.get('significant', False))
        logger.info(f"\nInteraction Effects:")
        logger.info(f"  Total tests: {len(interaction_results)}")
        logger.info(f"  Significant (FDR < 0.05): {n_sig_interaction}")
    
    logger.info(f"\nOutput Directory: {args.output}")
    logger.info(f"  Total files exported: {len(clustering_paths) + len(stat_paths)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)
