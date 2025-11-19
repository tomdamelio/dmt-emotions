"""
Main script for ICA analysis of TET data.

This script performs Independent Component Analysis on preprocessed TET data,
fits LME models to IC scores, and compares results with PCA.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root and scripts directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from tet.ica_analyzer import TETICAAnalyzer
from tet.ica_lme_analyzer import TETICALMEAnalyzer
from tet.ica_comparator import TETICAComparator
import config


def main():
    """Main ICA analysis workflow."""
    parser = argparse.ArgumentParser(
        description='Perform ICA analysis on TET data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=str(Path(config.TET_RESULTS_DIR) / 'preprocessed' / 'tet_preprocessed.csv'),
        help='Path to preprocessed TET data'
    )
    parser.add_argument(
        '--pca-dir',
        type=str,
        default=str(Path(config.TET_RESULTS_DIR) / 'pca'),
        help='Directory containing PCA results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(Path(config.TET_RESULTS_DIR) / 'ica'),
        help='Output directory for ICA results'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        default=['IC1', 'IC2'],
        help='IC components to analyze with LME (default: IC1 IC2)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ICA ANALYSIS FOR TET DATA")
    print("=" * 80)
    print()
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.input}...")
    data = pd.read_csv(args.input)
    print(f"  Loaded {len(data)} observations")
    print()
    
    # Load PCA results
    print(f"Loading PCA results from {args.pca_dir}...")
    pca_dir = Path(args.pca_dir)
    
    try:
        pca_scores = pd.read_csv(pca_dir / 'pca_scores.csv')
        pca_loadings = pd.read_csv(pca_dir / 'pca_loadings.csv')
        pca_variance = pd.read_csv(pca_dir / 'pca_variance_explained.csv')
        pca_lme = pd.read_csv(pca_dir / 'pca_lme_results.csv')
        
        n_pca_components = len(pca_variance)
        print(f"  Loaded PCA results with {n_pca_components} components")
        print()
    except FileNotFoundError as e:
        print(f"Error: PCA results not found. Run PCA analysis first.")
        print(f"  Missing file: {e.filename}")
        sys.exit(1)
    
    # Get z-scored dimension columns
    z_dimensions = [col for col in data.columns if col.endswith('_z') and col != 'time_c']
    print(f"Found {len(z_dimensions)} z-scored dimensions")
    print()
    
    # Initialize ICA analyzer
    print(f"Initializing ICA analyzer with {n_pca_components} components...")
    ica_analyzer = TETICAAnalyzer(
        data=data,
        dimensions=z_dimensions,
        n_components=n_pca_components,
        pca_scores=pca_scores,
        random_state=args.random_state
    )
    
    # Fit ICA
    print("Fitting ICA model...")
    ica_model = ica_analyzer.fit_ica()
    print("  ICA model fitted successfully")
    print()
    
    # Transform data to IC scores
    print("Transforming data to IC scores...")
    ic_scores = ica_analyzer.transform_data()
    print(f"  Generated IC scores: {ic_scores.shape}")
    print()
    
    # Extract mixing matrix
    print("Extracting mixing matrix...")
    mixing_matrix = ica_analyzer.get_mixing_matrix()
    print(f"  Mixing matrix shape: {len(mixing_matrix)} rows")
    print()
    
    # Compute PCA correlations
    print("Computing IC-PC correlations...")
    pca_correlation = ica_analyzer.compute_pca_correlation()
    print("  Top IC-PC correlations:")
    top_corrs = pca_correlation.nlargest(3, 'abs_correlation')
    for _, row in top_corrs.iterrows():
        print(f"    {row['ic_component']} vs {row['pc_component']}: r = {row['correlation']:.3f}")
    print()
    
    # Export ICA results
    print(f"Exporting ICA results to {output_dir}...")
    ica_files = ica_analyzer.export_results(str(output_dir))
    for file_type, file_path in ica_files.items():
        print(f"  {file_type}: {file_path}")
    print()
    
    # Fit LME models to IC scores
    print(f"Fitting LME models for {args.components}...")
    ica_lme_analyzer = TETICALMEAnalyzer(
        ic_scores=ic_scores,
        components=args.components
    )
    
    models = ica_lme_analyzer.fit_ic_models()
    print(f"  Fitted {len(models)} LME models")
    print()
    
    # Extract LME results
    print("Extracting LME results...")
    ica_lme_results = ica_lme_analyzer.extract_results()
    
    # Count significant effects
    n_sig = (ica_lme_results['p_value'] < 0.05).sum()
    print(f"  Found {n_sig} significant effects (p < 0.05)")
    print()
    
    # Export LME results
    print(f"Exporting LME results...")
    lme_files = ica_lme_analyzer.export_results(str(output_dir))
    for file_type, file_path in lme_files.items():
        print(f"  {file_type}: {file_path}")
    print()
    
    # Compare ICA and PCA
    print("Comparing ICA and PCA results...")
    comparator = TETICAComparator(
        ica_results={
            'mixing_matrix': mixing_matrix,
            'scores': ic_scores,
            'lme_results': ica_lme_results,
            'pca_correlation': pca_correlation
        },
        pca_results={
            'loadings': pca_loadings,
            'scores': pca_scores,
            'lme_results': pca_lme
        }
    )
    
    # Compare loadings
    loading_comparison = comparator.compare_loadings()
    print(f"  Loading comparison: {len(loading_comparison)} component pairs")
    print()
    
    # Compare LME results
    lme_comparison = comparator.compare_lme_results()
    if not lme_comparison.empty and 'agreement' in lme_comparison.columns:
        n_convergent = (lme_comparison['agreement'] == 'convergent').sum()
        n_divergent = lme_comparison['agreement'].str.contains('divergent').sum()
        print(f"  LME comparison:")
        print(f"    Convergent effects: {n_convergent}")
        print(f"    Divergent effects: {n_divergent}")
    else:
        print(f"  LME comparison: No comparable effects found")
    print()
    
    # Generate visualizations
    print("Generating comparison visualizations...")
    figures_dir = output_dir / 'figures'
    figure_files = comparator.generate_visualizations(str(figures_dir))
    for fig_path in figure_files:
        print(f"  {fig_path}")
    print()
    
    # Generate comparison report
    print("Generating comparison report...")
    report_path = output_dir / 'ica_pca_comparison_report.md'
    comparator.generate_report(str(report_path))
    print(f"  Report saved to: {report_path}")
    print()
    
    print("=" * 80)
    print("ICA ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
