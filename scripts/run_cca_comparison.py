"""
Run CCA Analysis at Different Temporal Resolutions

This script runs the CCA analysis at two different temporal resolutions:
1. 30-second bins (standard, aggregated)
2. 4-second bins (high resolution, interpolated)

This allows comparison of results to determine if temporal resolution
affects the detection of physiological-affective coupling.

Author: TET Analysis Pipeline
Date: November 22, 2025
"""

import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_cca_analysis(bin_size: int, n_permutations: int = 1000):
    """
    Run CCA analysis with specified temporal resolution.
    
    Args:
        bin_size: Temporal bin size in seconds (4 or 30)
        n_permutations: Number of permutation iterations
    """
    logger.info("=" * 80)
    logger.info(f"Running CCA Analysis: {bin_size}s bins, {n_permutations} permutations")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(f'results/tet/physio_correlation_{bin_size}s')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    cmd = [
        'python',
        'scripts/compute_physio_correlation.py',
        '--bin-size', str(bin_size),
        '--n-permutations', str(n_permutations),
        '--output', str(output_dir),
        '--compute-redundancy'
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✅ Analysis complete for {bin_size}s bins")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Analysis failed for {bin_size}s bins: {e}")
        return False


def create_validation_summary(bin_size: int):
    """
    Create validation summary table for specified resolution.
    
    Args:
        bin_size: Temporal bin size in seconds
    """
    logger.info(f"\nCreating validation summary for {bin_size}s bins...")
    
    # Modify the summary script to use the correct input directory
    cmd = [
        'python',
        '-c',
        f"""
import sys
sys.path.insert(0, 'scripts/tet')
from create_cca_validation_summary import load_validation_results, create_validation_summary, add_decision_column, format_summary_table
from pathlib import Path
import pandas as pd

results_dir = Path('results/tet/physio_correlation_{bin_size}s')
output_file = results_dir / 'cca_validation_summary_table.csv'

# Load and process
results = load_validation_results(results_dir)
summary = create_validation_summary(results)
summary = add_decision_column(summary)
formatted = format_summary_table(summary)

# Export
formatted.to_csv(output_file, index=False)
print(f"Saved to: {{output_file}}")
print(formatted.to_string(index=False))
"""
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"✅ Validation summary created for {bin_size}s bins")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create summary for {bin_size}s bins: {e}")
        return False


def main():
    """Main execution."""
    logger.info("\n" + "=" * 80)
    logger.info("CCA TEMPORAL RESOLUTION COMPARISON")
    logger.info("=" * 80)
    
    # Configuration
    n_permutations = 1000  # Use 1000 for publication-quality results
    
    # Run analyses
    results = {}
    
    # 1. High resolution (4s bins)
    logger.info("\n\n" + "=" * 80)
    logger.info("ANALYSIS 1: HIGH RESOLUTION (4-second bins)")
    logger.info("=" * 80)
    results['4s'] = run_cca_analysis(bin_size=4, n_permutations=n_permutations)
    
    if results['4s']:
        create_validation_summary(bin_size=4)
    
    # 2. Standard resolution (30s bins)
    logger.info("\n\n" + "=" * 80)
    logger.info("ANALYSIS 2: STANDARD RESOLUTION (30-second bins)")
    logger.info("=" * 80)
    results['30s'] = run_cca_analysis(bin_size=30, n_permutations=n_permutations)
    
    if results['30s']:
        create_validation_summary(bin_size=30)
    
    # Summary
    logger.info("\n\n" + "=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)
    
    logger.info("\nResults:")
    logger.info(f"  4s bins:  {'✅ Success' if results['4s'] else '❌ Failed'}")
    logger.info(f"  30s bins: {'✅ Success' if results['30s'] else '❌ Failed'}")
    
    logger.info("\nOutput directories:")
    logger.info(f"  4s:  results/tet/physio_correlation_4s/")
    logger.info(f"  30s: results/tet/physio_correlation_30s/")
    
    logger.info("\nNext steps:")
    logger.info("  1. Compare validation summary tables")
    logger.info("  2. Compare permutation p-values")
    logger.info("  3. Compare redundancy indices")
    logger.info("  4. Determine which resolution reveals stronger coupling")


if __name__ == '__main__':
    main()
