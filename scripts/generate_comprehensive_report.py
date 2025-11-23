"""
Generate Comprehensive TET Results Report

This script generates a comprehensive markdown report synthesizing all TET analysis results.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tet.results_synthesizer import TETResultsSynthesizer


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Generate comprehensive TET results report'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/tet',
        help='Path to TET results directory (default: results/tet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/tet_comprehensive_results.md',
        help='Path to output markdown file (default: docs/tet_comprehensive_results.md)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if report is up to date'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if report needs updating, do not regenerate'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("TET Comprehensive Results Report Generation")
    logger.info("=" * 80)
    
    try:
        # Import update detection utilities
        from tet.report_utils import should_regenerate_report, format_file_list, check_results_updated
        
        # Check if report needs updating
        should_regen, reason = should_regenerate_report(
            args.results_dir,
            args.output,
            force=args.force
        )
        
        logger.info(f"\nUpdate check: {reason}")
        
        # If check-only mode, just report and exit
        if args.check_only:
            if should_regen:
                logger.info("\n✓ Report needs updating")
                _, newer_files = check_results_updated(args.results_dir, args.output)
                if newer_files and newer_files[0] != "Report file missing":
                    logger.info(f"\nNewer files ({len(newer_files)}):")
                    logger.info(format_file_list(newer_files))
                return 0
            else:
                logger.info("\n✓ Report is up to date")
                return 0
        
        # Skip regeneration if not needed
        if not should_regen:
            logger.info("\n✓ Report is up to date. Skipping regeneration.")
            logger.info("   Use --force to regenerate anyway.")
            return 0
        
        logger.info("\nProceeding with report generation...")
        
        # Initialize synthesizer
        synthesizer = TETResultsSynthesizer(
            results_dir=args.results_dir,
            output_path=args.output
        )
        
        # Load all results
        logger.info("\nLoading analysis results...")
        synthesizer.load_all_results()
        
        # Generate report
        logger.info("\nGenerating comprehensive report...")
        report_path = synthesizer.generate_report()
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output file: {report_path}")
        logger.info(f"File size: {report_path.stat().st_size:,} bytes")
        
        # Count loaded components
        loaded_components = []
        if synthesizer.descriptive_data is not None:
            loaded_components.append("Descriptive Statistics")
        if synthesizer.lme_results is not None:
            loaded_components.append("LME Results")
        if synthesizer.pca_results is not None:
            loaded_components.append("PCA Results")
        if synthesizer.clustering_results is not None:
            loaded_components.append("Clustering Results")
        if synthesizer.physio_correlation_results is not None:
            loaded_components.append("Physiological-Affective Integration")
        
        logger.info(f"\nLoaded components ({len(loaded_components)}/5):")
        for component in loaded_components:
            logger.info(f"  ✓ {component}")
        
        missing_components = set([
            "Descriptive Statistics", "LME Results",
            "PCA Results", "Clustering Results",
            "Physiological-Affective Integration"
        ]) - set(loaded_components)
        
        if missing_components:
            logger.info(f"\nMissing components ({len(missing_components)}):")
            for component in missing_components:
                logger.info(f"  ✗ {component}")
        
        logger.info("\n" + "=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\nError generating report: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
