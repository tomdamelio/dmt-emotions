#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Linear Mixed Effects (LME) Modeling Script

This script fits LME models to test dose and state effects on TET dimensions:
1. Fits LME models for each dimension
2. Applies FDR correction
3. Computes dose contrasts within states

Usage:
    python fit_lme_models.py [options]
    
Example:
    python fit_lme_models.py --input results/tet/preprocessed/tet_preprocessed.csv
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.lme_analyzer import TETLMEAnalyzer
from tet.contrast_analyzer import TETContrastAnalyzer
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main workflow for LME modeling.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Fit Linear Mixed Effects models to TET data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fit_lme_models.py
  python fit_lme_models.py --input results/tet/preprocessed/tet_preprocessed.csv
  python fit_lme_models.py --output-dir results/tet/lme --verbose
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
        default='results/tet/lme',
        help='Output directory for results (default: results/tet/lme)'
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
        print("TET LINEAR MIXED EFFECTS MODELING")
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
        
        # Fit LME models
        print(f"\n[2/4] Fitting LME models...")
        logger.info("Fitting LME models...")
        
        lme_analyzer = TETLMEAnalyzer(data)
        print(f"      - Analyzing {len(lme_analyzer.dimensions)} dimensions")
        print(f"      - Analysis window: t_bin 0-18 (0-540 seconds, 0-9 minutes)")
        print(f"      - Model: Y ~ State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c + (1|Subject)")
        print(f"      - This may take 1-2 minutes...")
        
        lme_results = lme_analyzer.fit_all_dimensions()
        
        if len(lme_results) == 0:
            print(f"      ❌ ERROR: No models were successfully fitted")
            return 1
        
        print(f"      ✓ Fitted {len(lme_analyzer.models)} models")
        
        # Save LME results
        lme_path = lme_analyzer.export_results(str(output_dir))
        print(f"      ✓ Saved: {Path(lme_path).name}")
        
        # Show summary
        summary = lme_analyzer.get_summary()
        if len(summary) > 0:
            print(f"      ✓ Significant effects found in {len(summary)} dimensions")
        else:
            print(f"      ⚠ No significant effects found after FDR correction")
        
        # Compute contrasts
        print(f"\n[3/4] Computing dose contrasts within states...")
        logger.info("Computing contrasts...")
        
        contrast_analyzer = TETContrastAnalyzer(lme_analyzer.models)
        print(f"      - DMT High vs Low dose")
        print(f"      - RS High vs Low dose")
        
        contrasts = contrast_analyzer.compute_all_contrasts()
        
        # Save contrasts
        contrast_path = contrast_analyzer.export_contrasts(str(output_dir))
        print(f"      ✓ Saved: {Path(contrast_path).name}")
        
        # Show contrast summary
        contrast_summary = contrast_analyzer.get_summary()
        if len(contrast_summary) > 0:
            print(f"      ✓ Significant contrasts:")
            for _, row in contrast_summary.iterrows():
                print(f"        - {row['contrast']}: {row['n_significant']} dimensions")
        else:
            print(f"      ⚠ No significant contrasts found after FDR correction")
        
        # Print summary
        print(f"\n[4/4] Summary")
        print("-" * 80)
        
        # Load results to get counts
        lme_results = pd.read_csv(lme_path)
        contrasts = pd.read_csv(contrast_path)
        
        print(f"\nLME Results:")
        print(f"  Total results: {len(lme_results):,}")
        print(f"  Dimensions analyzed: {lme_results['dimension'].nunique()}")
        print(f"  Fixed effects per dimension: {len(lme_results[lme_results['dimension'] == lme_results['dimension'].iloc[0]])}")
        
        # Count significant effects
        sig_results = lme_results[lme_results['significant'] == True]
        if len(sig_results) > 0:
            print(f"  Significant effects (FDR < 0.05): {len(sig_results)}")
            print(f"\n  Top significant effects:")
            top_sig = sig_results.nsmallest(5, 'p_fdr')[['dimension', 'effect', 'beta', 'p_fdr']]
            for _, row in top_sig.iterrows():
                print(f"    - {row['dimension']}: {row['effect']} (β={row['beta']:.3f}, FDR={row['p_fdr']:.4f})")
        
        print(f"\nDose Contrasts:")
        print(f"  Total contrasts: {len(contrasts)}")
        print(f"  Dimensions: {contrasts['dimension'].nunique()}")
        print(f"  Contrast types: {', '.join(contrasts['contrast'].unique())}")
        
        # Count significant contrasts
        sig_contrasts = contrasts[contrasts['significant'] == True]
        if len(sig_contrasts) > 0:
            print(f"  Significant contrasts (FDR < 0.05): {len(sig_contrasts)}")
            print(f"\n  Significant contrasts by type:")
            for contrast_type in sig_contrasts['contrast'].unique():
                n = len(sig_contrasts[sig_contrasts['contrast'] == contrast_type])
                print(f"    - {contrast_type}: {n} dimensions")
        
        print(f"\nOutput Files:")
        print(f"  {output_dir / 'lme_results.csv'}")
        print(f"  {output_dir / 'lme_contrasts.csv'}")
        
        print("\n" + "=" * 80)
        logger.info("LME modeling complete")
        print("✓ LME modeling complete!")
        print("=" * 80)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ ERROR: File not found - {e}")
        return 1
        
    except Exception as e:
        logger.error(f"Modeling failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: Modeling failed - {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
