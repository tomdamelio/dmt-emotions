#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupling Analysis Pipeline: Physiology-Subjective Experience Integration.

This pipeline performs all analyses for the "Coupling between physiology and 
subjective experience" section of the paper, including:

1. Linear correlations between physiological markers and TET affective dimensions
2. Regression analysis: TET dimensions ~ Arousal Index (PC1)
3. Canonical Correlation Analysis (CCA) with permutation testing
4. Leave-One-Subject-Out (LOSO) cross-validation
5. Publication-ready visualizations (Figure 4, Figure S5)

Results reported in paper:
- DMT: Emotional Intensity correlates with HR (r=.44), SMNA (r=.36), RVT (r=.43)
- DMT: Arousal Index explains 30.5% variance in Emotional Intensity (beta=0.32)
- RS: Arousal Index does not predict Emotional Intensity (R2=0.4%, p=.294)
- CCA: DMT shows generalizable coupling (r_oos=.49, p=.008)
- CCA: RS shows idiosyncratic coupling (r_oos=-.28, fails cross-validation)

Usage:
    python pipelines/run_coupling_analysis.py [--n-permutations N] [--verbose]
    
    Default: 10,000 permutations (publication-ready, ~30-60 min)
    For quick testing: --n-permutations 100 (~2 min)

Prerequisites:
    Run these pipelines first:
    - run_ecg_hr_analysis.py
    - run_eda_smna_analysis.py
    - run_resp_rvt_analysis.py
    - run_composite_arousal_index.py
    - run_tet_analysis.py

Author: DMT-Emotions Pipeline
Date: 2025
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tet.physio_data_loader import TETPhysioDataLoader
from tet.physio_correlation_analyzer import TETPhysioCorrelationAnalyzer
from tet.physio_cca_analyzer import TETPhysioCCAAnalyzer
from tet.physio_visualizer import TETPhysioVisualizer


# Output directories
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'coupling'
FIGURES_DIR = OUTPUT_DIR / 'figures'


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('coupling_analysis')
    logger.setLevel(level)
    logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    
    # File handler with UTF-8 encoding for Windows compatibility
    log_file = OUTPUT_DIR / 'coupling_analysis.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    return logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run physiology-subjective experience coupling analysis'
    )
    parser.add_argument(
        '--n-permutations', type=int, default=10000,
        help='Number of permutations for CCA testing (default: 10000)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def load_data(logger: logging.Logger) -> pd.DataFrame:
    """
    Load and merge physiological and TET data.
    
    Returns:
        Merged DataFrame with physiological measures and TET dimensions
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading and merging data")
    logger.info("=" * 80)
    
    # Initialize data loader
    loader = TETPhysioDataLoader(
        composite_physio_path='results/composite/arousal_index_long.csv',
        tet_path='results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec=30
    )
    
    # Load physiological data
    logger.info("\nLoading composite physiological data...")
    physio_df = loader.load_physiological_data()
    logger.info(f"  Loaded {len(physio_df)} physiological observations")
    logger.info(f"  Subjects: {physio_df['subject'].nunique()}")
    
    # Load TET data
    logger.info("\nLoading TET data (aggregated to 30s bins)...")
    tet_df = loader.load_tet_data()
    logger.info(f"  Loaded {len(tet_df)} TET observations")
    
    # Merge datasets
    logger.info("\nMerging datasets...")
    merged_df = loader.merge_datasets()
    merged_df['t_bin'] = merged_df['window']
    
    logger.info(f"  Merged dataset: {len(merged_df)} observations")
    logger.info(f"  Subjects: {merged_df['subject'].nunique()}")
    logger.info(f"  States: {merged_df['state'].unique()}")
    
    # Export merged data
    merged_path = OUTPUT_DIR / 'merged_physio_tet_data.csv'
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"  Saved: {merged_path}")
    
    return merged_df


def run_correlation_analysis(
    merged_df: pd.DataFrame, 
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute correlations between TET dimensions and physiological measures.
    
    Paper results:
    - DMT: Emotional Intensity ~ HR (r=.44), SMNA (r=.36), RVT (r=.43)
    - DMT: Valence Index ~ HR (r=-.15)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Correlation Analysis")
    logger.info("=" * 80)
    
    analyzer = TETPhysioCorrelationAnalyzer(merged_df)
    corr_results = analyzer.compute_correlations(by_state=True)
    
    # Summary statistics
    n_total = len(corr_results)
    n_sig = (corr_results['p_fdr'] < 0.05).sum()
    
    logger.info(f"\n  Total correlations: {n_total}")
    logger.info(f"  Significant (p_FDR < 0.05): {n_sig} ({n_sig/n_total*100:.1f}%)")
    
    # Report key findings for paper
    logger.info("\n  Key findings (DMT state):")
    dmt_corrs = corr_results[corr_results['state'] == 'DMT']
    
    for physio in ['HR', 'SMNA_AUC', 'RVT']:
        ei_corr = dmt_corrs[
            (dmt_corrs['tet_dimension'] == 'emotional_intensity_z') &
            (dmt_corrs['physio_measure'] == physio)
        ]
        if len(ei_corr) > 0:
            r = ei_corr['r'].values[0]
            p = ei_corr['p_fdr'].values[0]
            logger.info(f"    Emotional Intensity ~ {physio}: r = {r:.2f}, p_FDR = {p:.3f}")
    
    # Valence-HR correlation
    val_hr = dmt_corrs[
        (dmt_corrs['tet_dimension'] == 'valence_index_z') &
        (dmt_corrs['physio_measure'] == 'HR')
    ]
    if len(val_hr) > 0:
        r = val_hr['r'].values[0]
        p = val_hr['p_fdr'].values[0]
        logger.info(f"    Valence Index ~ HR: r = {r:.2f}, p_FDR = {p:.3f}")
    
    # Export results
    corr_path = OUTPUT_DIR / 'correlations_tet_physio.csv'
    corr_results.to_csv(corr_path, index=False)
    logger.info(f"\n  Saved: {corr_path}")
    
    return corr_results


def run_regression_analysis(
    merged_df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Regression analysis: TET dimensions ~ Arousal Index (PC1).
    
    Paper results:
    - RS: Emotional Intensity ~ Arousal Index: R2=0.4%, beta=-0.04, p=.294
    - DMT: Emotional Intensity ~ Arousal Index: R2=30.5%, beta=0.32, p<.001
    - RS: Valence Index ~ Arousal Index: R2=6.2%, beta=-0.34
    - DMT: Valence Index ~ Arousal Index: R2=2.1%, beta=-0.15
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Regression Analysis (TET ~ Arousal Index)")
    logger.info("=" * 80)
    
    analyzer = TETPhysioCorrelationAnalyzer(merged_df)
    reg_results = analyzer.regression_analysis(by_state=True)
    
    # Report key findings
    logger.info("\n  Key findings:")
    
    for state in ['RS', 'DMT']:
        logger.info(f"\n  {state} State:")
        state_regs = reg_results[reg_results['state'] == state]
        
        for outcome in ['emotional_intensity_z', 'valence_index_z']:
            outcome_reg = state_regs[state_regs['outcome_variable'] == outcome]
            if len(outcome_reg) > 0:
                beta = outcome_reg['beta'].values[0]
                r2 = outcome_reg['r_squared'].values[0] * 100
                p = outcome_reg['p_value'].values[0]
                ci_low = outcome_reg['ci_lower'].values[0]
                ci_high = outcome_reg['ci_upper'].values[0]
                
                outcome_name = 'Emotional Intensity' if 'emotional' in outcome else 'Valence Index'
                p_str = f'{p:.3f}' if p >= 0.001 else '<.001'
                
                logger.info(
                    f"    {outcome_name}: R2={r2:.1f}%, beta={beta:.2f} "
                    f"[{ci_low:.2f}, {ci_high:.2f}], p={p_str}"
                )
    
    # Export results
    reg_path = OUTPUT_DIR / 'regression_tet_arousal_index.csv'
    reg_results.to_csv(reg_path, index=False)
    logger.info(f"\n  Saved: {reg_path}")
    
    return reg_results


def run_cca_analysis(
    merged_df: pd.DataFrame,
    n_permutations: int,
    logger: logging.Logger
) -> dict:
    """
    Canonical Correlation Analysis with permutation testing and cross-validation.
    
    Paper results:
    - RS: r=.85, p_FWER=.014, but cross-validation fails (r_oos=-.28)
    - DMT: r=.74, p_FWER=.357, but cross-validation succeeds (r_oos=.49, p=.008)
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Canonical Correlation Analysis")
    logger.info("=" * 80)
    
    cca_analyzer = TETPhysioCCAAnalyzer(merged_df)
    results = {}
    
    # Fit CCA
    logger.info("\n  Fitting CCA models...")
    cca_models = cca_analyzer.fit_cca(n_components=2)
    logger.info(f"    Fitted CCA for {len(cca_models)} states")
    
    # Extract canonical variates
    logger.info("\n  Extracting canonical variates...")
    cca_variates = cca_analyzer.extract_canonical_variates()
    results['variates'] = cca_variates
    
    for _, row in cca_variates.iterrows():
        logger.info(
            f"    {row['state']} CV{row['canonical_variate']}: "
            f"r = {row['canonical_correlation']:.2f}"
        )
    
    # Compute canonical loadings
    logger.info("\n  Computing canonical loadings...")
    cca_loadings = cca_analyzer.compute_canonical_loadings()
    results['loadings'] = cca_loadings
    
    # Report top loadings for DMT CV1
    logger.info("\n  DMT CV1 loadings (Figure 4A-B):")
    dmt_cv1 = cca_loadings[
        (cca_loadings['state'] == 'DMT') & 
        (cca_loadings['canonical_variate'] == 1)
    ]
    
    logger.info("    Physiological:")
    physio_loadings = dmt_cv1[dmt_cv1['variable_set'] == 'physio'].sort_values(
        'loading', ascending=False
    )
    for _, row in physio_loadings.iterrows():
        logger.info(f"      {row['variable_name']}: {row['loading']:.2f}")
    
    logger.info("    Affective:")
    tet_loadings = dmt_cv1[dmt_cv1['variable_set'] == 'tet'].sort_values(
        'loading', ascending=False
    ).head(3)
    for _, row in tet_loadings.iterrows():
        logger.info(f"      {row['variable_name']}: {row['loading']:.2f}")
    
    # Export loadings
    loadings_path = OUTPUT_DIR / 'cca_loadings.csv'
    cca_loadings.to_csv(loadings_path, index=False)
    logger.info(f"\n  Saved: {loadings_path}")
    
    # Permutation testing
    logger.info(f"\n  Running permutation test (n={n_permutations})...")
    perm_results = cca_analyzer.permutation_test(
        n_permutations=n_permutations,
        random_state=42
    )
    results['permutation'] = perm_results
    
    logger.info("\n  Permutation test results:")
    for state, perm_df in perm_results.items():
        cv1_row = perm_df[perm_df['canonical_variate'] == 1].iloc[0]
        logger.info(
            f"    {state} CV1: r_obs = {cv1_row['observed_r']:.2f}, "
            f"p_FWER = {cv1_row['permutation_p_value']:.3f}"
        )
    
    # Export permutation results
    perm_all = pd.concat([
        df.assign(state=state) for state, df in perm_results.items()
    ])
    perm_path = OUTPUT_DIR / 'cca_permutation_pvalues.csv'
    perm_all.to_csv(perm_path, index=False)
    
    # LOSO Cross-validation
    logger.info("\n  Running Leave-One-Subject-Out cross-validation...")
    cv_summary_all = []
    cv_folds_all = []
    
    for state in cca_models.keys():
        # loso_cross_validation returns a DataFrame with state already included
        cv_results_df = cca_analyzer.loso_cross_validation(state=state, n_components=2)
        cv_summary = cca_analyzer._summarize_cv_results(state)
        
        # Append fold results (already a DataFrame)
        cv_folds_all.append(cv_results_df)
        
        cv_summary['state'] = state
        cv_summary_all.append(cv_summary)
    
    cv_folds_df = pd.concat(cv_folds_all, ignore_index=True) if cv_folds_all else pd.DataFrame()
    cv_summary_df = pd.concat(cv_summary_all, ignore_index=True)
    
    results['cv_folds'] = cv_folds_df
    results['cv_summary'] = cv_summary_df
    
    logger.info("\n  Cross-validation results (Figure 4C):")
    for _, row in cv_summary_df[cv_summary_df['canonical_variate'] == 1].iterrows():
        logger.info(
            f"    {row['state']} CV1: mean r_oos = {row['mean_r_oos']:.2f}, "
            f"in-sample r = {row['in_sample_r']:.2f}"
        )
    
    # Export CV results
    cv_folds_path = OUTPUT_DIR / 'cca_cross_validation_folds.csv'
    cv_folds_df.to_csv(cv_folds_path, index=False)
    
    cv_summary_path = OUTPUT_DIR / 'cca_cross_validation_summary.csv'
    cv_summary_df.to_csv(cv_summary_path, index=False)
    
    # CV significance testing
    logger.info("\n  Testing cross-validation significance...")
    try:
        cv_significance = cca_analyzer.compute_cv_significance()
        results['cv_significance'] = cv_significance
        
        for _, row in cv_significance.iterrows():
            sig = '**' if row['p_value_t_test'] < 0.01 else ('*' if row['p_value_t_test'] < 0.05 else '')
            logger.info(
                f"    {row['state']} CV{row['canonical_variate']}: "
                f"t = {row['t_statistic']:.2f}, p = {row['p_value_t_test']:.3f}{sig}"
            )
        
        cv_sig_path = OUTPUT_DIR / 'cca_cv_significance.csv'
        cv_significance.to_csv(cv_sig_path, index=False)
    except Exception as e:
        logger.warning(f"    Could not compute CV significance: {e}")
    
    logger.info(f"\n  Saved all CCA results to: {OUTPUT_DIR}")
    
    return results


def generate_visualizations(
    merged_df: pd.DataFrame,
    corr_results: pd.DataFrame,
    reg_results: pd.DataFrame,
    cca_results: dict,
    logger: logging.Logger
) -> None:
    """
    Generate publication-ready visualizations.
    
    Outputs:
    - Figure 4: CCA combined figure (loadings, CV, scatterplot)
    - Figure S5: PC1 composite 4-panel (regression scatter plots)
    - Correlation heatmaps
    - Permutation null distributions
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Generating Visualizations")
    logger.info("=" * 80)
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    visualizer = TETPhysioVisualizer()
    
    # Correlation heatmaps
    logger.info("\n  Generating correlation heatmaps...")
    try:
        heatmap_paths = visualizer.plot_correlation_heatmaps(
            corr_results, str(OUTPUT_DIR)
        )
        logger.info(f"    Saved: {heatmap_paths}")
    except Exception as e:
        logger.warning(f"    Could not generate heatmaps: {e}")
    
    # PC1 composite 4-panel figure (Figure S5)
    logger.info("\n  Generating PC1 composite figure (Figure S5)...")
    try:
        # Create PC1 scores DataFrame
        pc1_scores = merged_df[['subject', 'session_id', 't_bin', 'ArousalIndex']].copy()
        pc1_scores = pc1_scores.rename(columns={'ArousalIndex': 'physio_PC1'})
        
        composite_path = visualizer.plot_pc1_composite_figure(
            merged_df, pc1_scores, reg_results, str(OUTPUT_DIR)
        )
        logger.info(f"    Saved: {composite_path}")
    except Exception as e:
        logger.warning(f"    Could not generate PC1 composite: {e}")
    
    # CCA loadings visualization
    logger.info("\n  Generating CCA loading plots...")
    try:
        if 'loadings' in cca_results and 'variates' in cca_results:
            cca_analyzer = TETPhysioCCAAnalyzer(merged_df)
            cca_analyzer.fit_cca(n_components=2)
            
            # Generate biplot and bar charts
            visualizer.plot_cca_loadings(
                cca_results['loadings'],
                cca_results['variates'],
                str(OUTPUT_DIR)
            )
            logger.info(f"    Saved CCA loading plots to: {FIGURES_DIR}")
    except Exception as e:
        logger.warning(f"    Could not generate CCA loadings: {e}")
    
    logger.info(f"\n  All visualizations saved to: {FIGURES_DIR}")


def generate_summary_report(
    corr_results: pd.DataFrame,
    reg_results: pd.DataFrame,
    cca_results: dict,
    logger: logging.Logger
) -> None:
    """Generate a summary report with key statistics for the paper."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Key Statistics for Paper")
    logger.info("=" * 80)
    
    report_lines = [
        "# Coupling Analysis Summary Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Linear Correlations (DMT State)",
        ""
    ]
    
    # Correlation results
    dmt_corrs = corr_results[corr_results['state'] == 'DMT']
    
    for physio in ['HR', 'SMNA_AUC', 'RVT']:
        ei_corr = dmt_corrs[
            (dmt_corrs['tet_dimension'] == 'emotional_intensity_z') &
            (dmt_corrs['physio_measure'] == physio)
        ]
        if len(ei_corr) > 0:
            r = ei_corr['r'].values[0]
            p = ei_corr['p_fdr'].values[0]
            report_lines.append(f"- Emotional Intensity ~ {physio}: r = {r:.2f}, p_FDR = {p:.3f}")
    
    report_lines.extend(["", "## 2. Regression Analysis (TET ~ Arousal Index)", ""])
    
    for state in ['RS', 'DMT']:
        state_regs = reg_results[reg_results['state'] == state]
        ei_reg = state_regs[state_regs['outcome_variable'] == 'emotional_intensity_z']
        
        if len(ei_reg) > 0:
            beta = ei_reg['beta'].values[0]
            r2 = ei_reg['r_squared'].values[0] * 100
            p = ei_reg['p_value'].values[0]
            ci_low = ei_reg['ci_lower'].values[0]
            ci_high = ei_reg['ci_upper'].values[0]
            p_str = f'{p:.3f}' if p >= 0.001 else '<.001'
            
            report_lines.append(
                f"- {state}: Emotional Intensity ~ Arousal Index: "
                f"R2 = {r2:.1f}%, beta = {beta:.2f} [{ci_low:.2f}, {ci_high:.2f}], p = {p_str}"
            )
    
    report_lines.extend(["", "## 3. Canonical Correlation Analysis", ""])
    
    # CCA results
    if 'cv_summary' in cca_results:
        cv_summary = cca_results['cv_summary']
        cv1_summary = cv_summary[cv_summary['canonical_variate'] == 1]
        
        for _, row in cv1_summary.iterrows():
            report_lines.append(
                f"- {row['state']} CV1: in-sample r = {row['in_sample_r']:.2f}, "
                f"mean r_oos = {row['mean_r_oos']:.2f}"
            )
    
    if 'permutation' in cca_results:
        report_lines.append("")
        report_lines.append("### Permutation Test Results:")
        for state, perm_df in cca_results['permutation'].items():
            cv1_row = perm_df[perm_df['canonical_variate'] == 1].iloc[0]
            report_lines.append(
                f"- {state} CV1: p_FWER = {cv1_row['permutation_p_value']:.3f}"
            )
    
    if 'cv_significance' in cca_results:
        report_lines.append("")
        report_lines.append("### Cross-Validation Significance:")
        for _, row in cca_results['cv_significance'].iterrows():
            if row['canonical_variate'] == 1:
                sig = '**' if row['p_value_t_test'] < 0.01 else ('*' if row['p_value_t_test'] < 0.05 else 'ns')
                report_lines.append(
                    f"- {row['state']} CV1: t = {row['t_statistic']:.2f}, "
                    f"p = {row['p_value_t_test']:.3f} ({sig})"
                )
    
    # Write report
    report_path = OUTPUT_DIR / 'coupling_analysis_summary.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n  Summary report saved: {report_path}")
    
    # Print key findings
    logger.info("\n" + "-" * 80)
    logger.info("KEY FINDINGS FOR PAPER:")
    logger.info("-" * 80)
    for line in report_lines[4:]:
        if line.startswith('- '):
            logger.info(f"  {line}")


def main():
    """Main pipeline execution."""
    args = parse_args()
    logger = setup_logging(args.verbose)
    
    start_time = datetime.now()
    
    logger.info("=" * 80)
    logger.info("COUPLING ANALYSIS PIPELINE")
    logger.info("Physiology-Subjective Experience Integration")
    logger.info("=" * 80)
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Permutations: {args.n_permutations}")
    
    try:
        # Step 1: Load data
        merged_df = load_data(logger)
        
        # Step 2: Correlation analysis
        corr_results = run_correlation_analysis(merged_df, logger)
        
        # Step 3: Regression analysis
        reg_results = run_regression_analysis(merged_df, logger)
        
        # Step 4: CCA analysis
        cca_results = run_cca_analysis(merged_df, args.n_permutations, logger)
        
        # Step 5: Generate visualizations
        generate_visualizations(
            merged_df, corr_results, reg_results, cca_results, logger
        )
        
        # Step 6: Generate summary report
        generate_summary_report(corr_results, reg_results, cca_results, logger)
        
        # Done
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info("\nNext step: Run pipelines/run_figures.py to generate final figures")
        
    except Exception as e:
        logger.error(f"\nPIPELINE FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
