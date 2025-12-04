#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master Figure Generation Script for TET Analysis

This script generates all publication-ready figures for the TET analysis pipeline.
It orchestrates the generation of multiple figure types and provides comprehensive
error handling and reporting.

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --input results/tet --output results/tet/figures
    python scripts/generate_all_figures.py --figures time-series lme peak-auc
    python scripts/generate_all_figures.py --dpi 600 --verbose

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FigureGenerationReport:
    """Track figure generation status and create summary reports."""
    
    def __init__(self):
        self.generated = []
        self.skipped = []
        self.failed = []
        self.start_time = datetime.now()
    
    def add_generated(self, figure_type: str, path: str):
        """Record a successfully generated figure."""
        self.generated.append((figure_type, path))
        logger.info(f"✓ Generated: {figure_type} -> {Path(path).name}")
    
    def add_skipped(self, figure_type: str, reason: str):
        """Record a skipped figure."""
        self.skipped.append((figure_type, reason))
        logger.warning(f"⊘ Skipped: {figure_type} - {reason}")
    
    def add_failed(self, figure_type: str, error: str):
        """Record a failed figure generation."""
        self.failed.append((figure_type, error))
        logger.error(f"✗ Failed: {figure_type} - {error}")
    
    def get_summary(self) -> str:
        """Generate summary report text."""
        duration = datetime.now() - self.start_time
        
        summary = []
        summary.append("=" * 80)
        summary.append("FIGURE GENERATION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Duration: {duration.total_seconds():.1f} seconds")
        summary.append(f"Generated: {len(self.generated)} figures")
        summary.append(f"Skipped: {len(self.skipped)} figures")
        summary.append(f"Failed: {len(self.failed)} figures")
        summary.append("")
        
        if self.generated:
            summary.append("Generated Figures:")
            summary.append("-" * 80)
            for fig_type, path in self.generated:
                summary.append(f"  ✓ {fig_type}: {Path(path).name}")
            summary.append("")
        
        if self.skipped:
            summary.append("Skipped Figures:")
            summary.append("-" * 80)
            for fig_type, reason in self.skipped:
                summary.append(f"  ⊘ {fig_type}: {reason}")
            summary.append("")
        
        if self.failed:
            summary.append("Failed Figures:")
            summary.append("-" * 80)
            for fig_type, error in self.failed:
                summary.append(f"  ✗ {fig_type}: {error}")
            summary.append("")
        
        summary.append("=" * 80)
        
        return "\n".join(summary)


def check_required_files(file_paths: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if required input files exist.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        Tuple of (all_exist, missing_files)
    """
    missing = []
    for path in file_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    return len(missing) == 0, missing


def generate_time_series_figures(
    input_dir: str,
    output_dir: str,
    dpi: int,
    report: FigureGenerationReport
) -> None:
    """
    Generate annotated time series plots (Requirement 8.1).
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory for output figures
        dpi: Resolution in DPI
        report: Report object to track status
    """
    figure_type = "Time Series (Req 8.1)"
    
    try:
        # Check required files
        required_files = [
            os.path.join(input_dir, 'preprocessed', 'tet_preprocessed.csv'),
            os.path.join(input_dir, 'lme', 'lme_results.csv'),
            os.path.join(input_dir, 'lme', 'lme_contrasts.csv'),
            os.path.join(input_dir, 'descriptive', 'time_course_all_dimensions.csv')
        ]
        
        all_exist, missing = check_required_files(required_files)
        
        if not all_exist:
            report.add_skipped(figure_type, f"Missing files: {', '.join([Path(f).name for f in missing])}")
            return
        
        # Import and run
        from plot_time_series import plot_annotated_time_series
        
        output_path = os.path.join(output_dir, 'timeseries_all_dimensions.png')
        
        plot_annotated_time_series(
            data_path=required_files[0],
            lme_results_path=required_files[1],
            lme_contrasts_path=required_files[2],
            time_courses_path=required_files[3],
            output_path=output_path,
            dpi=dpi
        )
        
        report.add_generated(figure_type, output_path)
        
    except Exception as e:
        report.add_failed(figure_type, str(e))


def generate_lme_coefficient_figures(
    input_dir: str,
    output_dir: str,
    dpi: int,
    report: FigureGenerationReport
) -> None:
    """
    Generate LME coefficient forest plots (Requirement 8.2).
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory for output figures
        dpi: Resolution in DPI
        report: Report object to track status
    """
    figure_type = "LME Coefficients (Req 8.2)"
    
    try:
        # Check required files
        lme_results_path = os.path.join(input_dir, 'lme', 'lme_results.csv')
        
        if not os.path.exists(lme_results_path):
            report.add_skipped(figure_type, f"Missing file: {Path(lme_results_path).name}")
            return
        
        # Import and run
        from plot_lme_coefficients import load_lme_results, prepare_plotting_data, plot_coefficient_forest
        
        # Load and prepare data
        lme_results = load_lme_results(lme_results_path)
        plot_data = prepare_plotting_data(lme_results)
        
        if not plot_data:
            report.add_skipped(figure_type, "No data available for plotting")
            return
        
        # Generate plot
        output_path = os.path.join(output_dir, 'lme_coefficients_forest.png')
        plot_coefficient_forest(plot_data, output_path)
        
        report.add_generated(figure_type, output_path)
        
    except Exception as e:
        report.add_failed(figure_type, str(e))





def generate_pca_figures(
    input_dir: str,
    output_dir: str,
    dpi: int,
    report: FigureGenerationReport
) -> None:
    """
    Generate PCA figures (Requirements 8.4, 8.5).
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory for output figures
        dpi: Resolution in DPI
        report: Report object to track status
    """
    figure_type_scree = "PCA Scree Plot (Req 8.4)"
    figure_type_loadings = "PCA Loadings Heatmap (Req 8.5)"
    figure_type_timeseries = "PCA Time Series (PC1 & PC2)"
    
    try:
        # Check required files
        pca_dir = os.path.join(input_dir, 'pca')
        required_files = [
            os.path.join(pca_dir, 'pca_variance_explained.csv'),
            os.path.join(pca_dir, 'pca_loadings.csv')
        ]
        
        all_exist, missing = check_required_files(required_files)
        
        if not all_exist:
            report.add_skipped(figure_type_scree, f"Missing files: {', '.join([Path(f).name for f in missing])}")
            report.add_skipped(figure_type_loadings, f"Missing files: {', '.join([Path(f).name for f in missing])}")
            report.add_skipped(figure_type_timeseries, f"Missing files: {', '.join([Path(f).name for f in missing])}")
            return
        
        # Run plot_pca_simple.py script (generates scree plot + loadings heatmap)
        import subprocess
        
        script_path = os.path.join(os.path.dirname(__file__), 'plot_pca_simple.py')
        cmd = [
            sys.executable,
            script_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check if output files were created
            scree_output = os.path.join(output_dir, 'pca_scree_plot.png')
            loadings_output = os.path.join(output_dir, 'pca_loadings_heatmap.png')
            
            if os.path.exists(scree_output):
                report.add_generated(figure_type_scree, scree_output)
            else:
                report.add_failed(figure_type_scree, "Output file not created")
            
            if os.path.exists(loadings_output):
                report.add_generated(figure_type_loadings, loadings_output)
            else:
                report.add_failed(figure_type_loadings, "Output file not created")
        else:
            error_msg = result.stderr if result.stderr else "Script execution failed"
            report.add_failed("PCA Figures (Req 8.4, 8.5)", error_msg)
        
        # Generate PCA time series figure
        pca_scores_path = os.path.join(pca_dir, 'pca_scores.csv')
        if os.path.exists(pca_scores_path):
            try:
                from plot_pca_timeseries import compute_pc_time_courses, plot_pc_timeseries
                import pandas as pd
                
                # Load PC scores
                pc_scores = pd.read_csv(pca_scores_path)
                
                # Compute time courses
                time_courses = compute_pc_time_courses(pc_scores)
                
                # Generate figure
                timeseries_output = os.path.join(output_dir, 'pca_timeseries.png')
                plot_pc_timeseries(time_courses, timeseries_output, dpi=dpi)
                
                if os.path.exists(timeseries_output):
                    report.add_generated(figure_type_timeseries, timeseries_output)
                else:
                    report.add_failed(figure_type_timeseries, "Output file not created")
                    
            except Exception as e:
                report.add_failed(figure_type_timeseries, f"Error generating time series: {str(e)}")
        else:
            report.add_skipped(figure_type_timeseries, f"Missing file: pca_scores.csv")
        
    except Exception as e:
        report.add_failed("PCA Figures (Req 8.4, 8.5)", str(e))


def generate_clustering_figures(
    input_dir: str,
    output_dir: str,
    dpi: int,
    report: FigureGenerationReport
) -> None:
    """
    Generate clustering figures (Requirements 8.6, 8.7).
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory for output figures
        dpi: Resolution in DPI
        report: Report object to track status
    """
    figure_type = "Clustering Figures (Req 8.6, 8.7)"
    
    try:
        # Check required files
        clustering_dir = os.path.join(input_dir, 'clustering')
        preprocessed_data = os.path.join(input_dir, 'preprocessed', 'tet_preprocessed.csv')
        
        required_files = [
            preprocessed_data,
            os.path.join(clustering_dir, 'clustering_kmeans_assignments.csv')
        ]
        
        all_exist, missing = check_required_files(required_files)
        
        if not all_exist:
            report.add_skipped(figure_type, f"Missing files: {', '.join([Path(f).name for f in missing])}")
            return
        
        # Import and run
        from plot_state_results import main as plot_state_main
        
        # Note: plot_state_results.py uses argparse, so we'd need to refactor
        # For now, we skip with informative message
        report.add_skipped(figure_type, "Use plot_state_results.py directly for clustering figures")
        
    except Exception as e:
        report.add_failed(figure_type, str(e))


def generate_glhmm_figures(
    input_dir: str,
    output_dir: str,
    dpi: int,
    report: FigureGenerationReport
) -> None:
    """
    Generate GLHMM figures (Requirement 8.8) - Optional/Future Work.
    
    Args:
        input_dir: Directory containing input data
        output_dir: Directory for output figures
        dpi: Resolution in DPI
        report: Report object to track status
    """
    figure_type = "GLHMM Figures (Req 8.8)"
    
    # GLHMM is optional/future work
    report.add_skipped(figure_type, "GLHMM analysis is optional future work (not yet implemented)")


def create_html_index(
    output_dir: str,
    report: FigureGenerationReport
) -> str:
    """
    Create HTML index linking to all generated figures.
    
    Args:
        output_dir: Directory containing figures
        report: Report object with generation status
        
    Returns:
        Path to HTML index file
    """
    html_path = os.path.join(output_dir, 'index.html')
    
    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html>")
    html_content.append("<head>")
    html_content.append("    <meta charset='utf-8'>")
    html_content.append("    <title>TET Analysis Figures</title>")
    html_content.append("    <style>")
    html_content.append("        body { font-family: Arial, sans-serif; margin: 40px; }")
    html_content.append("        h1 { color: #333; }")
    html_content.append("        h2 { color: #666; margin-top: 30px; }")
    html_content.append("        .figure { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }")
    html_content.append("        .figure img { max-width: 100%; height: auto; }")
    html_content.append("        .status { padding: 5px 10px; border-radius: 3px; font-weight: bold; }")
    html_content.append("        .generated { background-color: #d4edda; color: #155724; }")
    html_content.append("        .skipped { background-color: #fff3cd; color: #856404; }")
    html_content.append("        .failed { background-color: #f8d7da; color: #721c24; }")
    html_content.append("    </style>")
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append("    <h1>TET Analysis Figures</h1>")
    html_content.append(f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
    
    # Summary
    html_content.append("    <h2>Summary</h2>")
    html_content.append("    <ul>")
    html_content.append(f"        <li>Generated: {len(report.generated)} figures</li>")
    html_content.append(f"        <li>Skipped: {len(report.skipped)} figures</li>")
    html_content.append(f"        <li>Failed: {len(report.failed)} figures</li>")
    html_content.append("    </ul>")
    
    # Generated figures
    if report.generated:
        html_content.append("    <h2>Generated Figures</h2>")
        for fig_type, path in report.generated:
            rel_path = os.path.relpath(path, output_dir)
            html_content.append("    <div class='figure'>")
            html_content.append(f"        <h3>{fig_type} <span class='status generated'>✓ Generated</span></h3>")
            html_content.append(f"        <p><a href='{rel_path}'>{Path(path).name}</a></p>")
            html_content.append(f"        <img src='{rel_path}' alt='{fig_type}'>")
            html_content.append("    </div>")
    
    # Skipped figures
    if report.skipped:
        html_content.append("    <h2>Skipped Figures</h2>")
        html_content.append("    <ul>")
        for fig_type, reason in report.skipped:
            html_content.append(f"        <li><span class='status skipped'>⊘ Skipped</span> {fig_type}: {reason}</li>")
        html_content.append("    </ul>")
    
    # Failed figures
    if report.failed:
        html_content.append("    <h2>Failed Figures</h2>")
        html_content.append("    <ul>")
        for fig_type, error in report.failed:
            html_content.append(f"        <li><span class='status failed'>✗ Failed</span> {fig_type}: {error}</li>")
            html_content.append("    </ul>")
    
    html_content.append("</body>")
    html_content.append("</html>")
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))
    
    logger.info(f"Created HTML index: {html_path}")
    
    return html_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate all TET analysis figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures with default settings
  python scripts/generate_all_figures.py

  # Specify custom input/output directories
  python scripts/generate_all_figures.py --input results/tet --output results/tet/figures

  # Generate only specific figure types
  python scripts/generate_all_figures.py --figures time-series lme peak-auc

  # High resolution output
  python scripts/generate_all_figures.py --dpi 600 --verbose
  
  # Skip report generation for faster debugging
  python scripts/generate_all_figures.py --skip-report
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet',
        help='Input directory containing analysis results (default: results/tet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/figures',
        help='Output directory for figures (default: results/tet/figures)'
    )
    
    parser.add_argument(
        '--figures',
        type=str,
        nargs='+',
        choices=['time-series', 'lme', 'pca', 'clustering', 'glhmm', 'all'],
        default=['all'],
        help='Figure types to generate (default: all)'
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
    
    parser.add_argument(
        '--skip-report',
        action='store_true',
        help='Skip comprehensive report generation (for faster debugging)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize report
    report = FigureGenerationReport()
    
    # Print header
    print("=" * 80)
    print("TET ANALYSIS - MASTER FIGURE GENERATION")
    print("=" * 80)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Resolution: {args.dpi} DPI")
    print(f"Figure types: {', '.join(args.figures)}")
    print("=" * 80)
    print()
    
    # Determine which figures to generate
    generate_all = 'all' in args.figures
    
    # Generate figures
    if generate_all or 'time-series' in args.figures:
        logger.info("Generating time series figures...")
        generate_time_series_figures(args.input, args.output, args.dpi, report)
    
    if generate_all or 'lme' in args.figures:
        logger.info("Generating LME coefficient figures...")
        generate_lme_coefficient_figures(args.input, args.output, args.dpi, report)
    

    if generate_all or 'pca' in args.figures:
        logger.info("Generating PCA figures...")
        generate_pca_figures(args.input, args.output, args.dpi, report)
    
    if generate_all or 'clustering' in args.figures:
        logger.info("Generating clustering figures...")
        generate_clustering_figures(args.input, args.output, args.dpi, report)
    
    if generate_all or 'glhmm' in args.figures:
        logger.info("Checking GLHMM figures...")
        generate_glhmm_figures(args.input, args.output, args.dpi, report)
    
    # Create HTML index
    logger.info("Creating HTML index...")
    html_path = create_html_index(args.output, report)
    
    # Print summary
    print()
    print(report.get_summary())
    print()
    print(f"HTML index: {html_path}")
    print()
    
    # Generate comprehensive report (unless skipped)
    if not args.skip_report:
        logger.info("=" * 80)
        logger.info("Generating comprehensive results report...")
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
                '--results-dir', args.input,
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
        logger.info("Skipping comprehensive report generation (--skip-report flag)")
    
    # Return exit code
    if report.failed:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())
