#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Stepwise Permutation Test for CCA (Winkler et al., 2020).

This script executes the stepwise permutation testing algorithm to rigorously
assess the significance of canonical correlations with FWER control.

Usage:
    python scripts/run_stepwise_permutation_test.py [options]

Options:
    --n-permutations INT    Number of permutations (default: 5000)
    --statistic STR         Test statistic: 'wilks' or 'roys' (default: wilks)
    --output PATH           Output directory (default: results/tet/physio_correlation)
    --verbose               Enable verbose logging

Example:
    # Quick test (100 permutations, ~2 min)
    python scripts/run_stepwise_permutation_test.py --n-permutations 100 --verbose
    
    # Publication-ready (5000 permutations, ~30-60 min)
    python scripts/run_stepwise_permutation_test.py --n-permutations 5000 --verbose

References:
    Winkler, A. M., et al. (2020). Permutation inference for canonical 
    correlation analysis. NeuroImage, 220, 117065.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Stepwise Permutation Test for CCA (Winkler 2020)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutation iterations'
    )
    parser.add_argument(
        '--statistic',
        type=str,
        choices=['wilks', 'roys'],
        default='wilks',
        help='Test statistic to use'
    )
    parser.add_argument(
        '--permutation-type',
        type=str,
        choices=['row', 'subject'],
        default='row',
        help='Permutation type: row (standard) or subject (preserves within-subject structure)'
    )
    parser.add_argument(
        '--use-theil',
        action='store_true',
        default=True,
        help='Use Theil method for BLUS residuals (recommended for repeated measures)'
    )
    parser.add_argument(
        '--no-theil',
        action='store_true',
        help='Disable Theil method (use standard deflation)'
    )
    parser.add_argument(
        '--physio-composite',
        type=str,
        default='results/composite/arousal_index_long.csv',
        help='Path to composite physiological data'
    )
    parser.add_argument(
        '--tet-data',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/physio_correlation',
        help='Output directory'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Determine if Theil method should be used
    use_theil = args.use_theil and not args.no_theil
    
    logger.info("=" * 80)
    logger.info("STEPWISE PERMUTATION TEST FOR CCA (Winkler et al., 2020)")
    logger.info("=" * 80)
    logger.info(f"  Permutations: {args.n_permutations}")
    logger.info(f"  Statistic: {args.statistic}")
    logger.info(f"  Permutation type: {args.permutation_type}")
    logger.info(f"  Use Theil BLUS: {use_theil}")
    logger.info(f"  Random state: {args.random_state}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 80)
    
    loader = TETPhysioDataLoader(
        composite_physio_path=args.physio_composite,
        tet_path=args.tet_data,
        target_bin_duration_sec=30
    )
    
    # Load and merge
    physio_df = loader.load_physiological_data()
    tet_df = loader.load_tet_data()
    merged_df = loader.merge_datasets()
    
    logger.info(f"  Loaded {len(merged_df)} observations")
    logger.info(f"  Subjects: {merged_df['subject'].nunique()}")
    
    # =========================================================================
    # Initialize CCA analyzer and fit models
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Fitting CCA models")
    logger.info("=" * 80)
    
    analyzer = TETPhysioCCAAnalyzer(merged_df)
    analyzer.fit_cca(n_components=3)  # Fit with max components
    
    # =========================================================================
    # Run stepwise permutation test
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running Stepwise Permutation Test")
    logger.info("=" * 80)
    
    results_df = analyzer.run_stepwise_permutation_both_states(
        n_permutations=args.n_permutations,
        statistic=args.statistic,
        permutation_type=args.permutation_type,
        use_theil=use_theil,
        random_state=args.random_state,
        output_dir=str(output_dir)
    )
    
    # =========================================================================
    # Display results
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS: Stepwise Permutation Test")
    logger.info("=" * 80)
    
    print("\n" + results_df.to_string(index=False))
    
    # Summary interpretation
    logger.info("\n" + "-" * 80)
    logger.info("INTERPRETATION")
    logger.info("-" * 80)
    
    for state in results_df['state'].unique():
        state_df = results_df[results_df['state'] == state]
        sig_modes = state_df[state_df['significant']]['mode'].tolist()
        
        if sig_modes:
            logger.info(f"  {state}: Significant modes = {sig_modes}")
            for mode in sig_modes:
                row = state_df[state_df['mode'] == mode].iloc[0]
                logger.info(
                    f"    Mode {mode}: r = {row['observed_r']:.3f}, "
                    f"p_raw = {row['raw_p_value']:.4f}, "
                    f"p_FWER = {row['fwer_p_value']:.4f}"
                )
        else:
            logger.info(f"  {state}: No significant modes (all p_FWER > 0.05)")
    
    # =========================================================================
    # Generate visualization
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Generating visualization")
    logger.info("=" * 80)
    
    fig_path = analyzer.plot_stepwise_permutation_results(str(output_dir))
    logger.info(f"  Saved figure to: {fig_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    
    return results_df


if __name__ == '__main__':
    main()
