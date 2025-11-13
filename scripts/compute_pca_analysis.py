#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TET PCA Analysis Script

This script performs Principal Component Analysis (PCA) on preprocessed TET data
and fits Linear Mixed Effects (LME) models to the resulting principal component scores.

Usage:
    python scripts/compute_pca_analysis.py [options]

Example:
    # Run with default settings (75% variance threshold, PC1 and PC2)
    python scripts/compute_pca_analysis.py
    
    # Specify custom variance threshold
    python scripts/compute_pca_analysis.py --variance-threshold 0.80
    
    # Analyze additional components
    python scripts/compute_pca_analysis.py --components PC1 PC2 PC3
    
    # Verbose output
    python scripts/compute_pca_analysis.py --verbose

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import argparse
import logging
import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Import TET PCA analyzers
sys.path.append(os.path.join(os.path.dirname(__file__), 'tet'))
from tet.pca_analyzer import TETPCAAnalyzer
from tet.pca_lme_analyzer import TETPCALMEAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Perform PCA analysis on TET data and fit LME models to PC scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=os.path.join(config.TET_RESULTS_DIR, 'preprocessed', 'tet_preprocessed.csv'),
        help='Path to preprocessed TET data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=os.path.join(config.TET_RESULTS_DIR, 'pca'),
        help='Output directory for PCA results'
    )
    
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.75,
        help='Target cumulative variance explained (0.70-0.80 recommended)'
    )
    
    parser.add_argument(
        '--components',
        nargs='+',
        default=['PC1', 'PC2'],
        help='List of principal components to analyze with LME (e.g., PC1 PC2 PC3)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    
    return parser.parse_args()


def load_preprocessed_data(input_path: str) -> pd.DataFrame:
    """
    Load preprocessed TET data.
    
    Args:
        input_path: Path to preprocessed TET data CSV
    
    Returns:
        pd.DataFrame: Preprocessed TET data
    
    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading preprocessed data from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    data = pd.read_csv(input_path)
    logger.info(f"  Loaded {len(data)} rows, {len(data.columns)} columns")
    
    # Verify required columns exist
    required_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get z-scored dimension columns (exclude composite indices)
    z_dims = [col for col in data.columns 
              if col.endswith('_z') 
              and col not in ['affect_index_z', 'imagery_index_z', 'self_index_z']]
    
    if len(z_dims) == 0:
        raise ValueError("No z-scored dimension columns found in data")
    
    logger.info(f"  Found {len(z_dims)} z-scored dimensions")
    
    return data


def main():
    """
    Main PCA analysis workflow.
    
    Steps:
    1. Parse command-line arguments
    2. Load preprocessed TET data
    3. Initialize TETPCAAnalyzer with z-scored dimensions
    4. Fit group-level PCA
    5. Transform data to PC scores
    6. Extract loadings and variance explained
    7. Export PCA results (loadings, variance, scores)
    8. Initialize TETPCALMEAnalyzer with PC scores
    9. Prepare PC data for LME
    10. Fit LME models for specified components
    11. Extract LME results
    12. Export LME results
    13. Print summary to console
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 80)
    logger.info("TET PCA ANALYSIS")
    logger.info("=" * 80)
    
    try:
        # =====================================================================
        # STEP 1: Load preprocessed data
        # =====================================================================
        data = load_preprocessed_data(args.input)
        
        # Get z-scored dimension columns
        z_dims = [col for col in data.columns 
                  if col.endswith('_z') 
                  and col not in ['affect_index_z', 'imagery_index_z', 'self_index_z']]
        
        logger.info(f"Z-scored dimensions: {', '.join(z_dims)}")
        
        # =====================================================================
        # STEP 2: Initialize TETPCAAnalyzer
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FITTING GROUP-LEVEL PCA")
        logger.info("=" * 80)
        
        pca_analyzer = TETPCAAnalyzer(
            data=data,
            dimensions=z_dims,
            variance_threshold=args.variance_threshold
        )
        
        # =====================================================================
        # STEP 3: Fit PCA
        # =====================================================================
        pca_model = pca_analyzer.fit_pca()
        
        # =====================================================================
        # STEP 4: Transform data to PC scores
        # =====================================================================
        pc_scores = pca_analyzer.transform_data()
        
        # =====================================================================
        # STEP 5: Extract loadings and variance explained
        # =====================================================================
        loadings_df = pca_analyzer.get_loadings()
        variance_df = pca_analyzer.get_variance_explained()
        
        # =====================================================================
        # STEP 6: Export PCA results
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING PCA RESULTS")
        logger.info("=" * 80)
        
        pca_output_paths = pca_analyzer.export_results(args.output)
        
        for file_type, path in pca_output_paths.items():
            logger.info(f"  {file_type}: {path}")
        
        # =====================================================================
        # STEP 7: Print PCA summary
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PCA SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"Number of components retained: {pca_analyzer.n_components}")
        logger.info(f"Variance threshold: {args.variance_threshold:.1%}")
        logger.info(f"Cumulative variance explained: {variance_df['cumulative_variance'].iloc[-1]:.1%}")
        logger.info("")
        logger.info("Variance explained by component:")
        for _, row in variance_df.iterrows():
            logger.info(f"  {row['component']}: {row['variance_explained']:.1%} "
                       f"(cumulative: {row['cumulative_variance']:.1%})")
        
        # =====================================================================
        # STEP 8: Initialize TETPCALMEAnalyzer
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("FITTING LME MODELS FOR PC SCORES")
        logger.info("=" * 80)
        
        # Validate requested components exist
        available_components = [f'PC{i+1}' for i in range(pca_analyzer.n_components)]
        invalid_components = [c for c in args.components if c not in available_components]
        if invalid_components:
            logger.warning(f"Requested components not available: {invalid_components}")
            logger.warning(f"Available components: {available_components}")
            # Filter to valid components
            valid_components = [c for c in args.components if c in available_components]
            if not valid_components:
                logger.error("No valid components to analyze")
                return 1
            args.components = valid_components
        
        lme_analyzer = TETPCALMEAnalyzer(
            pc_scores=pc_scores,
            components=args.components
        )
        
        # =====================================================================
        # STEP 9: Fit LME models
        # =====================================================================
        models = lme_analyzer.fit_pc_models()
        
        if not models:
            logger.error("No LME models fitted successfully")
            return 1
        
        # =====================================================================
        # STEP 10: Extract LME results
        # =====================================================================
        lme_results = lme_analyzer.extract_results()
        
        # =====================================================================
        # STEP 11: Export LME results
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING LME RESULTS")
        logger.info("=" * 80)
        
        lme_output_paths = lme_analyzer.export_results(args.output)
        
        for file_type, path in lme_output_paths.items():
            logger.info(f"  {file_type}: {path}")
        
        # =====================================================================
        # STEP 12: Print LME summary
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("LME SUMMARY")
        logger.info("=" * 80)
        
        # Count significant effects (p < 0.05) for each component
        for component in args.components:
            component_results = lme_results[lme_results['component'] == component]
            n_significant = (component_results['p_value'] < 0.05).sum()
            logger.info(f"{component}: {n_significant} significant effects (p < 0.05)")
            
            # Show significant effects
            sig_effects = component_results[component_results['p_value'] < 0.05]
            if len(sig_effects) > 0:
                logger.info(f"  Significant effects:")
                for _, row in sig_effects.iterrows():
                    logger.info(f"    {row['effect']}: β={row['beta']:.3f}, "
                               f"p={row['p_value']:.4f}")
        
        # =====================================================================
        # STEP 13: Success
        # =====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PCA ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
