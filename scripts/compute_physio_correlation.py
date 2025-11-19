#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for physiological-TET correlation analysis.

This script orchestrates the complete physiological-affective integration analysis,
including:
- Loading and merging physiological and TET data
- Computing correlations between TET affective dimensions and physiological measures
- Regression analysis predicting TET from physiological PC1 (ArousalIndex)
- Testing arousal vs valence hypothesis using Steiger's Z-test
- Canonical Correlation Analysis (CCA) to identify shared latent dimensions
- Generating publication-ready visualizations

Usage:
    python scripts/compute_physio_correlation.py [options]

Options:
    --tet-data PATH              Path to preprocessed TET data
    --physio-composite PATH      Path to composite physiological data
    --pca-loadings PATH          Path to PCA loadings file
    --output PATH                Output directory for results
    --n-cca-components INT       Number of CCA components (default: 2)
    --by-state                   Compute analyses separately by state (default: True)
    --verbose                    Enable verbose logging

Example:
    python scripts/compute_physio_correlation.py --verbose

Author: TET Analysis Pipeline
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_correlation_analyzer import TETPhysioCorrelationAnalyzer
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer
from scripts.tet.physio_visualizer import TETPhysioVisualizer
import config


def setup_logging(verbose: bool = False):
    """
    Configure logging for the analysis.
    
    Args:
        verbose: If True, set logging level to DEBUG
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('results/tet/physio_correlation/analysis.log', mode='w')
        ]
    )


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Physiological-TET correlation analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input file paths
    parser.add_argument(
        '--tet-data',
        type=str,
        default='results/tet/preprocessed/tet_preprocessed.csv',
        help='Path to preprocessed TET data'
    )
    parser.add_argument(
        '--physio-composite',
        type=str,
        default='results/composite/arousal_index_long.csv',
        help='Path to composite physiological data (with PC1)'
    )
    parser.add_argument(
        '--pca-loadings',
        type=str,
        default='results/composite/pca_loadings_pc1.csv',
        help='Path to PCA loadings file (for documentation)'
    )
    
    # Output directory
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/physio_correlation',
        help='Output directory for results'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--n-cca-components',
        type=int,
        default=2,
        help='Number of CCA components to extract'
    )
    parser.add_argument(
        '--by-state',
        action='store_true',
        default=True,
        help='Compute analyses separately by state (RS vs DMT)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """
    Main analysis workflow for physiological-TET correlation analysis.
    
    Process:
    1. Load composite physiological data (HR, SMNA AUC, RVT with PC1)
    2. Load and aggregate TET data to 30-second bins
    3. Merge datasets on (subject, state, dose, window)
    4. Compute correlations (arousal, valence, all affective dims vs physio)
    5. Load PCA loadings for documentation
    6. Fit regression models (TET ~ ArousalIndex)
    7. Test arousal vs valence hypothesis (Steiger's Z-test)
    8. Perform Canonical Correlation Analysis (CCA)
    9. Generate visualizations (heatmaps, scatter plots, CCA loadings)
    10. Export all results
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("PHYSIOLOGICAL-TET CORRELATION ANALYSIS")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load and merge data
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading and merging data")
    logger.info("=" * 80)
    
    try:
        # Initialize data loader
        loader = TETPhysioDataLoader(
            composite_physio_path=args.physio_composite,
            tet_path=args.tet_data,
            target_bin_duration_sec=30
        )
        
        # Load physiological data
        logger.info("\nLoading composite physiological data...")
        physio_df = loader.load_physiological_data()
        logger.info(f"  Loaded {len(physio_df)} physiological observations")
        logger.info(f"  Subjects: {physio_df['subject'].nunique()}")
        logger.info(f"  States: {physio_df['State'].unique()}")
        logger.info(f"  Doses: {physio_df['Dose'].unique()}")
        
        # Load TET data
        logger.info("\nLoading and aggregating TET data...")
        tet_df = loader.load_tet_data()
        logger.info(f"  Loaded {len(tet_df)} TET observations (30s bins)")
        logger.info(f"  Subjects: {tet_df['subject'].nunique()}")
        logger.info(f"  Windows: {tet_df['window'].nunique()}")
        
        # Merge datasets
        logger.info("\nMerging physiological and TET datasets...")
        merged_df = loader.merge_datasets()
        logger.info(f"  Merged dataset: {len(merged_df)} observations")
        logger.info(f"  Subjects: {merged_df['subject'].nunique()}")
        logger.info(f"  Sessions: {merged_df.groupby('subject')['session_id'].nunique().sum()}")
        
        # Add t_bin column (alias for window) for compatibility with analyzer
        merged_df['t_bin'] = merged_df['window']
        
        # Export merged data
        merged_path = output_dir / 'merged_physio_tet_data.csv'
        loader.export_merged_data(merged_path)
        logger.info(f"  Exported merged data to {merged_path}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # =========================================================================
    # STEP 2: Correlation analysis
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Correlation analysis")
    logger.info("=" * 80)
    
    try:
        # Initialize correlation analyzer
        corr_analyzer = TETPhysioCorrelationAnalyzer(merged_df)
        
        # Compute correlations
        logger.info("\nComputing correlations between TET and physiological measures...")
        corr_results = corr_analyzer.compute_correlations(by_state=args.by_state)
        
        # Log summary
        n_total = len(corr_results)
        n_sig = (corr_results['p_fdr'] < 0.05).sum()
        logger.info(f"  Total correlations computed: {n_total}")
        logger.info(f"  Significant correlations (p_fdr < 0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
        
        # Log top correlations
        logger.info("\n  Top 5 strongest correlations:")
        top_corrs = corr_results.nlargest(5, 'r')[
            ['tet_dimension', 'physio_measure', 'state', 'r', 'p_fdr']
        ]
        for _, row in top_corrs.iterrows():
            logger.info(
                f"    {row['tet_dimension']} vs {row['physio_measure']} ({row['state']}): "
                f"r = {row['r']:.3f}, p_fdr = {row['p_fdr']:.4f}"
            )
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise
    
    # =========================================================================
    # STEP 3: Load PCA loadings for documentation
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Loading PCA loadings")
    logger.info("=" * 80)
    
    try:
        logger.info("\nLoading PCA loadings for documentation...")
        pca_loadings = corr_analyzer.load_pca_loadings()
        
        if pca_loadings is not None:
            logger.info("  PCA loadings loaded successfully")
            logger.info(f"  PC1 loadings:\n{pca_loadings}")
        else:
            logger.warning("  PCA loadings file not found (non-critical)")
        
    except Exception as e:
        logger.warning(f"Could not load PCA loadings: {e}")
    
    # =========================================================================
    # STEP 4: Regression analysis
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Regression analysis")
    logger.info("=" * 80)
    
    try:
        logger.info("\nFitting regression models: TET ~ ArousalIndex...")
        reg_results = corr_analyzer.regression_analysis(by_state=args.by_state)
        
        # Log summary
        n_total = len(reg_results)
        n_sig = (reg_results['p_value'] < 0.05).sum()
        logger.info(f"  Total regressions fitted: {n_total}")
        logger.info(f"  Significant regressions (p < 0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
        
        # Log top regressions
        logger.info("\n  Top 5 strongest regressions (by R²):")
        top_regs = reg_results.nlargest(5, 'r_squared')[
            ['outcome_variable', 'state', 'beta', 'r_squared', 'p_value']
        ]
        for _, row in top_regs.iterrows():
            logger.info(
                f"    {row['outcome_variable']} ({row['state']}): "
                f"β = {row['beta']:.3f}, R² = {row['r_squared']:.3f}, p = {row['p_value']:.4f}"
            )
        
    except Exception as e:
        logger.error(f"Error in regression analysis: {e}")
        raise
    
    # =========================================================================
    # STEP 5: Test arousal vs valence hypothesis
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Testing arousal vs valence hypothesis")
    logger.info("=" * 80)
    
    try:
        logger.info("\nTesting hypothesis: arousal-ArousalIndex > valence-ArousalIndex...")
        hypothesis_results = corr_analyzer.test_arousal_valence_hypothesis()
        
        # Log results
        logger.info("\n  Hypothesis test results:")
        for _, row in hypothesis_results.iterrows():
            logger.info(f"\n  {row['state']} State:")
            logger.info(f"    r(arousal, ArousalIndex) = {row['r_arousal_arousalindex']:.3f}")
            logger.info(f"    r(valence, ArousalIndex) = {row['r_valence_arousalindex']:.3f}")
            logger.info(f"    r(arousal, valence) = {row['r_arousal_valence']:.3f}")
            logger.info(f"    Steiger's Z = {row['z_statistic']:.3f}")
            logger.info(f"    p-value = {row['p_value']:.4f}")
            logger.info(f"    Conclusion: {row['conclusion']}")
        
    except Exception as e:
        logger.error(f"Error in hypothesis testing: {e}")
        raise
    
    # =========================================================================
    # STEP 6: Canonical Correlation Analysis (CCA)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Canonical Correlation Analysis")
    logger.info("=" * 80)
    
    try:
        # Initialize CCA analyzer
        cca_analyzer = TETPhysioCCAAnalyzer(merged_df)
        
        # Fit CCA
        logger.info(f"\nFitting CCA with {args.n_cca_components} components...")
        cca_models = cca_analyzer.fit_cca(n_components=args.n_cca_components)
        logger.info(f"  Fitted CCA for {len(cca_models)} states")
        
        # Extract canonical variates
        logger.info("\nExtracting canonical variates and testing significance...")
        cca_variates = cca_analyzer.extract_canonical_variates()
        
        # Log results
        logger.info("\n  Canonical correlation results:")
        for _, row in cca_variates.iterrows():
            sig_marker = '*' if row['p_value'] < 0.05 else ''
            logger.info(
                f"    {row['state']} - CV{row['canonical_variate']}: "
                f"r = {row['canonical_correlation']:.3f}{sig_marker}, "
                f"p = {row['p_value']:.4f}"
            )
        
        # Compute canonical loadings
        logger.info("\nComputing canonical loadings...")
        cca_loadings = cca_analyzer.compute_canonical_loadings()
        logger.info(f"  Computed {len(cca_loadings)} canonical loadings")
        
        # Log top loadings for CV1
        logger.info("\n  Top loadings for Canonical Variate 1:")
        for state in cca_loadings['state'].unique():
            logger.info(f"\n    {state} State:")
            state_cv1 = cca_loadings[
                (cca_loadings['state'] == state) &
                (cca_loadings['canonical_variate'] == 1)
            ].copy()
            
            # Top physio loadings
            physio_top = state_cv1[state_cv1['variable_set'] == 'physio'].nlargest(3, 'loading')
            logger.info("      Physiological:")
            for _, row in physio_top.iterrows():
                logger.info(f"        {row['variable_name']}: {row['loading']:.3f}")
            
            # Top TET loadings
            tet_top = state_cv1[state_cv1['variable_set'] == 'tet'].nlargest(3, 'loading')
            logger.info("      TET Affective:")
            for _, row in tet_top.iterrows():
                logger.info(f"        {row['variable_name']}: {row['loading']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in CCA: {e}")
        raise
    
    # =========================================================================
    # STEP 7: Generate visualizations
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Generating visualizations")
    logger.info("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = TETPhysioVisualizer()
        
        # Generate correlation heatmaps
        logger.info("\nGenerating correlation heatmaps...")
        heatmap_paths = visualizer.plot_correlation_heatmaps(
            corr_results,
            str(output_dir)
        )
        logger.info(f"  Generated {len(heatmap_paths)} heatmap(s)")
        
        # Generate regression scatter plots
        logger.info("\nGenerating regression scatter plots...")
        # Create PC1 scores DataFrame for plotting
        pc1_scores = merged_df[['subject', 'session_id', 't_bin', 'ArousalIndex']].copy()
        pc1_scores = pc1_scores.rename(columns={'ArousalIndex': 'physio_PC1'})
        pc1_scores['t_bin'] = merged_df['window']  # Use window as t_bin for merging
        
        scatter_paths = visualizer.plot_regression_scatter(
            merged_df,
            pc1_scores,
            reg_results,
            str(output_dir)
        )
        logger.info(f"  Generated {len(scatter_paths)} scatter plot(s)")
        
        # Generate CCA loading plots
        logger.info("\nGenerating CCA loading plots...")
        cca_paths = visualizer.plot_cca_loadings(
            cca_loadings,
            cca_variates,
            str(output_dir)
        )
        logger.info(f"  Generated {len(cca_paths)} CCA plot(s)")
        
        # Export all figures
        all_figures = visualizer.export_figures(str(output_dir))
        logger.info(f"\n  Total figures generated: {len(all_figures)}")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise
    
    # =========================================================================
    # STEP 8: Export results
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Exporting results")
    logger.info("=" * 80)
    
    try:
        # Export correlation and regression results
        logger.info("\nExporting correlation and regression results...")
        corr_files = corr_analyzer.export_results(str(output_dir))
        logger.info(f"  Exported {len(corr_files)} file(s):")
        for file_type, file_path in corr_files.items():
            logger.info(f"    - {file_type}: {file_path}")
        
        # Export CCA results
        logger.info("\nExporting CCA results...")
        cca_files = cca_analyzer.export_results(str(output_dir))
        logger.info(f"  Exported {len(cca_files)} file(s):")
        for file_type, file_path in cca_files.items():
            logger.info(f"    - {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    logger.info("\nKey findings:")
    logger.info(f"  - Computed {len(corr_results)} correlations")
    logger.info(f"  - Significant correlations: {(corr_results['p_fdr'] < 0.05).sum()}")
    logger.info(f"  - Fitted {len(reg_results)} regression models")
    logger.info(f"  - Significant regressions: {(reg_results['p_value'] < 0.05).sum()}")
    logger.info(f"  - CCA canonical correlations:")
    for _, row in cca_variates.iterrows():
        sig = '*' if row['p_value'] < 0.05 else ''
        logger.info(f"      {row['state']} CV{row['canonical_variate']}: r = {row['canonical_correlation']:.3f}{sig}")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info(f"  - CSV files: {len(corr_files) + len(cca_files)}")
    logger.info(f"  - Figures: {len(all_figures)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)
