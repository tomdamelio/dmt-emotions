#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Analysis Pipeline Orchestrator

This script provides a single entry point for running the complete TET
(Temporal Experience Tracking) analysis pipeline. It executes all analysis
stages in the correct sequence and manages dependencies between stages.

Usage:
    # Run complete pipeline
    python scripts/run_tet_analysis.py
    
    # Run specific stages
    python scripts/run_tet_analysis.py --stages preprocessing descriptive lme
    
    # Skip specific stages
    python scripts/run_tet_analysis.py --skip-stages clustering
    
    # Run from specific stage onward
    python scripts/run_tet_analysis.py --from-stage pca
    
    # Dry run (validate without executing)
    python scripts/run_tet_analysis.py --dry-run
    
    # Verbose output
    python scripts/run_tet_analysis.py --verbose

Requirements: 10.3, 10.4, 10.13, 10.14, 10.15
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to path (pipelines/ is one level below root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import config


class PipelineValidator:
    """Validates pipeline dependencies and inputs."""
    
    def __init__(self, logger):
        """Initialize validator with logger."""
        self.logger = logger
    
    def validate_stage_inputs(self, stage_name: str) -> Tuple[bool, str]:
        """
        Validate inputs required for a specific stage.
        
        Args:
            stage_name: Name of the stage to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        validators = {
            'preprocessing': self._validate_raw_data,
            'descriptive': self._validate_preprocessed_data,
            'lme': self._validate_preprocessed_data,
            'pca': self._validate_preprocessed_data,
            'ica': self._validate_ica_inputs,
            'physio_correlation': self._validate_physio_correlation_inputs,
            'clustering': self._validate_preprocessed_data,
            'figures': self._validate_analysis_results,
            'report': self._validate_all_results
        }
        
        validator = validators.get(stage_name)
        if validator:
            return validator()
        return True, ""
    
    def _validate_raw_data(self) -> Tuple[bool, str]:
        """Validate raw TET data files exist."""
        # Check if raw data directory exists
        raw_data_dir = Path(config.REPORTS_DATA)
        
        if not raw_data_dir.exists():
            return False, (
                f"Raw data directory not found: {raw_data_dir}\n"
                "Please ensure TET data files are in the correct location."
            )
        
        # Check for at least one .mat file
        mat_files = list(raw_data_dir.glob('*.mat'))
        if not mat_files:
            return False, (
                f"No .mat files found in: {raw_data_dir}\n"
                "Please ensure TET data files are present."
            )
        
        self.logger.debug(f"Found {len(mat_files)} .mat files in raw data directory")
        return True, ""
    
    def _validate_preprocessed_data(self) -> Tuple[bool, str]:
        """Validate preprocessed data exists."""
        required = Path('results/tet/preprocessed/tet_preprocessed.csv')
        
        if not required.exists():
            return False, (
                f"Preprocessed data not found: {required}\n"
                "Run preprocessing stage first:\n"
                "  python pipelines/run_tet_analysis.py --stages preprocessing"
            )
        
        self.logger.debug(f"Preprocessed data found: {required}")
        return True, ""
    
    def _validate_ica_inputs(self) -> Tuple[bool, str]:
        """Validate PCA results exist for ICA analysis."""
        required = [
            'results/tet/preprocessed/tet_preprocessed.csv',
            'results/tet/pca/pca_scores.csv',
            'results/tet/pca/pca_loadings.csv',
            'results/tet/pca/pca_lme_results.csv'
        ]
        
        missing = [f for f in required if not Path(f).exists()]
        
        if missing:
            return False, (
                f"Missing required files for ICA:\n" +
                "\n".join(f"  - {f}" for f in missing) +
                "\n\nRun PCA analysis first:\n"
                "  python pipelines/run_tet_analysis.py --stages pca"
            )
        
        self.logger.debug("ICA inputs validated")
        return True, ""
    
    def _validate_physio_correlation_inputs(self) -> Tuple[bool, str]:
        """Validate inputs for physiological-TET correlation analysis."""
        required = [
            'results/tet/preprocessed/tet_preprocessed.csv',
            'results/composite/arousal_index_long.csv',
            'results/composite/pca_loadings_pc1.csv'
        ]
        
        missing = [f for f in required if not Path(f).exists()]
        
        if missing:
            # Check if TET preprocessing is missing
            if 'results/tet/preprocessed/tet_preprocessed.csv' in missing:
                return False, (
                    "Missing TET preprocessed data:\n"
                    "  - results/tet/preprocessed/tet_preprocessed.csv\n\n"
                    "Run TET preprocessing first:\n"
                    "  python pipelines/run_tet_analysis.py --stages preprocessing"
                )
            
            # Check if composite physiological files are missing
            composite_missing = [f for f in missing if 'composite' in f]
            if composite_missing:
                return False, (
                    f"Missing composite physiological data files:\n" +
                    "\n".join(f"  - {f}" for f in composite_missing) +
                    "\n\nRun physiological preprocessing and composite arousal index pipeline:\n"
                    "  python pipelines/run_composite_arousal_index.py\n\n"
                    "This will generate the required composite physiological data with PC1 (ArousalIndex)."
                )
        
        self.logger.debug("Physiological-TET correlation inputs validated")
        return True, ""
    
    def _validate_analysis_results(self) -> Tuple[bool, str]:
        """Validate analysis results exist for figure generation."""
        required = [
            'results/tet/lme/lme_results.csv',
        ]
        
        missing = [f for f in required if not Path(f).exists()]
        
        if missing:
            return False, (
                f"Missing analysis results:\n" +
                "\n".join(f"  - {f}" for f in missing) +
                "\n\nRun analysis stages first:\n"
                "  python scripts/run_tet_analysis.py --stages lme"
            )
        
        self.logger.debug("Analysis results validated")
        return True, ""
    
    def _validate_all_results(self) -> Tuple[bool, str]:
        """Validate all results exist for report generation."""
        required_dirs = [
            'results/tet/descriptive',
            'results/tet/lme',
        ]
        
        missing = [d for d in required_dirs if not Path(d).exists()]
        
        if missing:
            return False, (
                f"Missing result directories:\n" +
                "\n".join(f"  - {d}" for d in missing) +
                "\n\nRun complete pipeline first:\n"
                "  python scripts/run_tet_analysis.py"
            )
        
        self.logger.debug("All results validated")
        return True, ""


class TETAnalysisPipeline:
    """
    Orchestrates complete TET analysis pipeline.
    
    This class manages the execution of all TET analysis stages in the correct
    sequence, handles dependencies, validates inputs, and logs progress.
    
    Attributes:
        config: Configuration object
        logger: Logger instance
        validator: PipelineValidator instance
        stages: List of (stage_name, stage_function) tuples
        results: Dictionary tracking stage execution results
    
    Example:
        >>> pipeline = TETAnalysisPipeline()
        >>> pipeline.run()
    """
    
    def __init__(self, config_module=None, log_path='results/tet/pipeline_execution.log'):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config_module: Configuration module (default: config)
            log_path: Path to log file
        """
        self.config = config_module or config
        self.log_path = Path(log_path)
        self.logger = self._setup_logging()
        self.validator = PipelineValidator(self.logger)
        self.stages = self._define_stages()
        self.results = {}
    
    def _setup_logging(self, verbose=False) -> logging.Logger:
        """
        Set up logging configuration.
        
        Args:
            verbose: Enable debug logging
            
        Returns:
            Configured logger instance
        """
        # Create results directory if needed
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        level = logging.DEBUG if verbose else logging.INFO
        
        # File handler
        file_handler = logging.FileHandler(self.log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Root logger
        logger = logging.getLogger('tet_pipeline')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _define_stages(self) -> List[Tuple[str, callable]]:
        """
        Define pipeline stages in execution order.
        
        Returns:
            List of (stage_name, stage_function) tuples
        """
        return [
            ('preprocessing', self._run_preprocessing),
            ('descriptive', self._run_descriptive_stats),
            ('lme', self._run_lme_models),
            ('pca', self._run_pca_analysis),
            ('ica', self._run_ica_analysis),
            ('physio_correlation', self._run_physio_correlation),
            ('clustering', self._run_clustering_analysis),
            ('figures', self._run_figure_generation),
            ('report', self._run_report_generation)
        ]
    
    def run(self, stages: Optional[List[str]] = None, 
            skip_stages: Optional[List[str]] = None,
            from_stage: Optional[str] = None,
            dry_run: bool = False) -> Dict[str, str]:
        """
        Execute pipeline stages.
        
        Args:
            stages: List of specific stages to run (None = all)
            skip_stages: List of stages to skip
            from_stage: Start from this stage onward
            dry_run: Validate without executing
            
        Returns:
            Dictionary mapping stage names to status ('success', 'failed', 'skipped')
        """
        self.logger.info("=" * 80)
        self.logger.info("TET ANALYSIS PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log file: {self.log_path}")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - Validation only, no execution")
        
        self.logger.info("=" * 80)
        
        # Determine which stages to run
        stages_to_run = self._determine_stages(stages, skip_stages, from_stage)
        
        self.logger.info(f"\nStages to execute: {', '.join(stages_to_run)}")
        self.logger.info("=" * 80)
        
        # Execute stages
        for i, (stage_name, stage_func) in enumerate(self.stages, 1):
            if stage_name not in stages_to_run:
                self.results[stage_name] = 'skipped'
                continue
            
            self.logger.info(f"\nStage {i}/{len(self.stages)}: {stage_name.upper()}")
            self.logger.info("-" * 80)
            
            # Validate inputs
            is_valid, error_msg = self.validator.validate_stage_inputs(stage_name)
            if not is_valid:
                self.logger.error(f"Validation failed for stage '{stage_name}':")
                self.logger.error(error_msg)
                self.results[stage_name] = 'failed'
                
                if stage_name in ['preprocessing', 'descriptive', 'lme']:
                    # Critical stages - stop pipeline
                    self.logger.error("Critical stage failed. Stopping pipeline.")
                    break
                else:
                    # Non-critical - continue with warning
                    self.logger.warning("Non-critical stage failed. Continuing...")
                    continue
            
            if dry_run:
                self.logger.info(f"✓ Validation passed for '{stage_name}'")
                self.results[stage_name] = 'validated'
                continue
            
            # Execute stage
            start_time = time.time()
            try:
                stage_func()
                elapsed = time.time() - start_time
                self.logger.info(f"✓ Stage '{stage_name}' completed ({elapsed:.1f}s)")
                self.results[stage_name] = 'success'
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"✗ Stage '{stage_name}' failed ({elapsed:.1f}s): {e}", exc_info=True)
                self.results[stage_name] = 'failed'
                
                if stage_name in ['preprocessing', 'descriptive', 'lme']:
                    self.logger.error("Critical stage failed. Stopping pipeline.")
                    break
                else:
                    self.logger.warning("Non-critical stage failed. Continuing...")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _determine_stages(self, stages: Optional[List[str]], 
                         skip_stages: Optional[List[str]],
                         from_stage: Optional[str]) -> List[str]:
        """Determine which stages to run based on arguments."""
        all_stage_names = [name for name, _ in self.stages]
        
        if stages:
            # Run only specified stages
            return [s for s in stages if s in all_stage_names]
        
        if from_stage:
            # Run from specified stage onward
            try:
                start_idx = all_stage_names.index(from_stage)
                stages_to_run = all_stage_names[start_idx:]
            except ValueError:
                self.logger.error(f"Invalid stage name: {from_stage}")
                return []
        else:
            # Run all stages
            stages_to_run = all_stage_names.copy()
        
        # Remove skipped stages
        if skip_stages:
            stages_to_run = [s for s in stages_to_run if s not in skip_stages]
        
        return stages_to_run
    
    def _run_preprocessing(self):
        """Execute preprocessing stage."""
        self.logger.info("Running TET data preprocessing...")
        from preprocess_tet_data import main as preprocess_main
        
        # Save original sys.argv
        original_argv = sys.argv.copy()
        sys.argv = ['preprocess_tet_data.py']
        
        try:
            preprocess_main()
        finally:
            sys.argv = original_argv
    
    def _run_descriptive_stats(self):
        """Execute descriptive statistics stage."""
        self.logger.info("Computing descriptive statistics...")
        from compute_descriptive_stats import main as descriptive_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['compute_descriptive_stats.py']
        
        try:
            descriptive_main()
        finally:
            sys.argv = original_argv
    
    def _run_lme_models(self):
        """Execute LME modeling stage."""
        self.logger.info("Fitting LME models...")
        from fit_lme_models import main as lme_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['fit_lme_models.py']
        
        try:
            lme_main()
        finally:
            sys.argv = original_argv
    

    def _run_pca_analysis(self):
        """Execute PCA analysis stage."""
        self.logger.info("Computing PCA analysis...")
        from compute_pca_analysis import main as pca_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['compute_pca_analysis.py']
        
        try:
            pca_main()
        finally:
            sys.argv = original_argv
    
    def _run_ica_analysis(self):
        """Execute ICA analysis stage."""
        self.logger.info("Computing ICA analysis...")
        from compute_ica_analysis import main as ica_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['compute_ica_analysis.py']
        
        try:
            ica_main()
        finally:
            sys.argv = original_argv
    
    def _run_physio_correlation(self):
        """Execute physiological-TET correlation analysis stage."""
        self.logger.info("Computing physiological-TET correlation analysis...")
        from compute_physio_correlation import main as physio_corr_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['compute_physio_correlation.py']
        
        try:
            physio_corr_main()
        finally:
            sys.argv = original_argv
    
    def _run_clustering_analysis(self):
        """Execute clustering analysis stage."""
        self.logger.info("Computing clustering analysis...")
        from compute_clustering_analysis import main as clustering_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['compute_clustering_analysis.py', '--skip-report']
        
        try:
            clustering_main()
        finally:
            sys.argv = original_argv
    
    def _run_figure_generation(self):
        """Execute figure generation stage."""
        self.logger.info("Generating figures...")
        from generate_all_figures import main as figures_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['generate_all_figures.py', '--skip-report']
        
        try:
            figures_main()
        finally:
            sys.argv = original_argv
    
    def _run_report_generation(self):
        """Execute report generation stage."""
        self.logger.info("Generating comprehensive report...")
        from generate_comprehensive_report import main as report_main
        
        original_argv = sys.argv.copy()
        sys.argv = ['generate_comprehensive_report.py', '--force']
        
        try:
            report_main()
        finally:
            sys.argv = original_argv
    
    def _print_summary(self):
        """Print pipeline execution summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        
        success_count = sum(1 for status in self.results.values() if status == 'success')
        failed_count = sum(1 for status in self.results.values() if status == 'failed')
        skipped_count = sum(1 for status in self.results.values() if status == 'skipped')
        
        self.logger.info(f"Total stages: {len(self.results)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Failed: {failed_count}")
        self.logger.info(f"  Skipped: {skipped_count}")
        
        if self.results:
            self.logger.info("\nStage Results:")
            for stage_name, status in self.results.items():
                symbol = "✓" if status == 'success' else "✗" if status == 'failed' else "⊘"
                self.logger.info(f"  {symbol} {stage_name}: {status}")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log file: {self.log_path}")
        self.logger.info("=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run TET analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python scripts/run_tet_analysis.py

  # Run specific stages
  python scripts/run_tet_analysis.py --stages preprocessing descriptive lme

  # Skip specific stages
  python scripts/run_tet_analysis.py --skip-stages clustering

  # Run from specific stage onward
  python scripts/run_tet_analysis.py --from-stage pca

  # Run only preprocessing
  python scripts/run_tet_analysis.py --preprocessing-only

  # Dry run (validate without executing)
  python scripts/run_tet_analysis.py --dry-run

  # Verbose output
  python scripts/run_tet_analysis.py --verbose
        """
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['preprocessing', 'descriptive', 'lme', 
                 'pca', 'ica', 'physio_correlation', 'clustering', 'figures', 'report'],
        help='Specific stages to run'
    )
    
    parser.add_argument(
        '--skip-stages',
        nargs='+',
        choices=['preprocessing', 'descriptive', 'lme', 
                 'pca', 'ica', 'physio_correlation', 'clustering', 'figures', 'report'],
        help='Stages to skip'
    )
    
    parser.add_argument(
        '--from-stage',
        choices=['preprocessing', 'descriptive', 'lme', 
                 'pca', 'ica', 'physio_correlation', 'clustering', 'figures', 'report'],
        help='Start from this stage onward'
    )
    
    parser.add_argument(
        '--preprocessing-only',
        action='store_true',
        help='Run only preprocessing stage'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate without executing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle preprocessing-only flag
    if args.preprocessing_only:
        args.stages = ['preprocessing']
    
    # Initialize pipeline
    pipeline = TETAnalysisPipeline()
    
    # Update logging level if verbose
    if args.verbose:
        pipeline.logger.setLevel(logging.DEBUG)
        for handler in pipeline.logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Run pipeline
    try:
        results = pipeline.run(
            stages=args.stages,
            skip_stages=args.skip_stages,
            from_stage=args.from_stage,
            dry_run=args.dry_run
        )
        
        # Return exit code based on results
        failed_count = sum(1 for status in results.values() if status == 'failed')
        return 1 if failed_count > 0 else 0
        
    except KeyboardInterrupt:
        pipeline.logger.info("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        pipeline.logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
