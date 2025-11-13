# -*- coding: utf-8 -*-
"""
Quick test script for TETPCAAnalyzer

Tests basic functionality of the PCA analyzer with preprocessed TET data.
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.pca_analyzer import TETPCAAnalyzer
import config

def main():
    """Test PCA analyzer with preprocessed data."""
    
    print("=" * 80)
    print("QUICK TEST: TET PCA Analyzer")
    print("=" * 80)
    
    # Load preprocessed data
    input_file = 'results/tet/preprocessed/tet_preprocessed.csv'
    
    if not os.path.exists(input_file):
        print(f"\n❌ Error: Preprocessed data not found at {input_file}")
        print("   Run scripts/preprocess_tet_data.py first")
        return
    
    print(f"\n📂 Loading preprocessed data from {input_file}...")
    data = pd.read_csv(input_file)
    print(f"   Loaded {len(data)} rows, {data['subject'].nunique()} subjects")
    
    # Get z-scored dimension columns (exclude composite indices)
    z_dims = [col for col in data.columns if col.endswith('_z') 
              and col not in ['affect_index_z', 'imagery_index_z', 'self_index_z']]
    
    print(f"\n📊 Found {len(z_dims)} z-scored dimensions:")
    for dim in z_dims:
        print(f"   - {dim}")
    
    # Initialize PCA analyzer
    print(f"\n🔧 Initializing TETPCAAnalyzer (variance_threshold=0.75)...")
    analyzer = TETPCAAnalyzer(data, z_dims, variance_threshold=0.75)
    
    # Fit PCA
    print(f"\n🔬 Fitting group-level PCA...")
    analyzer.fit_pca()
    print(f"   ✓ Retained {analyzer.n_components} components")
    
    # Transform data
    print(f"\n🔄 Transforming data to PC scores...")
    pc_scores = analyzer.transform_data()
    print(f"   ✓ Generated PC scores: {pc_scores.shape}")
    print(f"\n   First few rows:")
    print(pc_scores.head())
    
    # Get loadings
    print(f"\n📈 Extracting PCA loadings...")
    loadings = analyzer.get_loadings()
    print(f"   ✓ Loadings shape: {loadings.shape}")
    print(f"\n   Top 5 loadings for PC1:")
    pc1_loadings = loadings[loadings['component'] == 'PC1'].copy()
    pc1_loadings['abs_loading'] = pc1_loadings['loading'].abs()
    pc1_top = pc1_loadings.nlargest(5, 'abs_loading')
    for _, row in pc1_top.iterrows():
        print(f"   {row['dimension']:30s}: {row['loading']:+.3f}")
    
    # Get variance explained
    print(f"\n📊 Extracting variance explained...")
    variance = analyzer.get_variance_explained()
    print(f"   ✓ Variance explained:")
    for _, row in variance.iterrows():
        print(f"   {row['component']}: {row['variance_explained']:.2%} "
              f"(cumulative: {row['cumulative_variance']:.2%})")
    
    # Export results
    output_dir = 'results/tet/pca'
    print(f"\n💾 Exporting results to {output_dir}...")
    output_paths = analyzer.export_results(output_dir)
    print(f"   ✓ Exported {len(output_paths)} files:")
    for file_type, path in output_paths.items():
        print(f"   - {file_type}: {path}")
    
    print("\n" + "=" * 80)
    print("✅ PCA ANALYZER TEST COMPLETE")
    print("=" * 80)
    
    # Summary statistics
    print(f"\n📋 Summary:")
    print(f"   - Input observations: {len(data)}")
    print(f"   - Input dimensions: {len(z_dims)}")
    print(f"   - Components retained: {analyzer.n_components}")
    print(f"   - Total variance explained: {variance['cumulative_variance'].iloc[-1]:.2%}")
    print(f"   - PC1 variance: {variance['variance_explained'].iloc[0]:.2%}")
    print(f"   - PC2 variance: {variance['variance_explained'].iloc[1]:.2%}")

if __name__ == '__main__':
    main()


def test_pca_lme():
    """Test PCA LME analyzer with PC scores."""
    
    print("\n" + "=" * 80)
    print("QUICK TEST: TET PCA LME Analyzer")
    print("=" * 80)
    
    # Import here to avoid circular dependency
    from tet.pca_lme_analyzer import TETPCALMEAnalyzer
    
    # Load PC scores
    input_file = 'results/tet/pca/pca_scores.csv'
    
    if not os.path.exists(input_file):
        print(f"\n❌ Error: PC scores not found at {input_file}")
        print("   Run the PCA test first to generate PC scores")
        return
    
    print(f"\n📂 Loading PC scores from {input_file}...")
    pc_scores = pd.read_csv(input_file)
    print(f"   Loaded {len(pc_scores)} rows, {pc_scores['subject'].nunique()} subjects")
    
    # Get available PC columns
    pc_cols = [col for col in pc_scores.columns if col.startswith('PC')]
    print(f"\n📊 Found {len(pc_cols)} PC columns: {', '.join(pc_cols[:5])}")
    
    # Initialize PCA LME analyzer (default: PC1, PC2)
    print(f"\n🔧 Initializing TETPCALMEAnalyzer...")
    analyzer = TETPCALMEAnalyzer(pc_scores, components=['PC1', 'PC2'])
    
    # Fit LME models
    print(f"\n🔬 Fitting LME models for PC1 and PC2...")
    models = analyzer.fit_pc_models()
    print(f"   ✓ Fitted {len(models)} models")
    
    # Extract results
    print(f"\n📊 Extracting results...")
    results = analyzer.extract_results()
    print(f"   ✓ Extracted {len(results)} rows")
    print(f"\n   Results preview:")
    print(results.head(10))
    
    # Export results
    output_dir = 'results/tet/pca'
    print(f"\n💾 Exporting results to {output_dir}...")
    output_paths = analyzer.export_results(output_dir)
    print(f"   ✓ Exported {len(output_paths)} files:")
    for file_type, path in output_paths.items():
        print(f"   - {file_type}: {path}")
    
    print("\n" + "=" * 80)
    print("✅ PCA LME ANALYZER TEST COMPLETE")
    print("=" * 80)
    
    # Summary statistics
    print(f"\n📋 Summary:")
    print(f"   - Components analyzed: {len(models)}")
    print(f"   - Total effects: {len(results)}")
    print(f"   - Effects per component: {len(results) // len(models)}")
    
    # Show significant effects (p < 0.05)
    sig_results = results[results['p_value'] < 0.05]
    if len(sig_results) > 0:
        print(f"\n   Significant effects (p < 0.05):")
        for _, row in sig_results.iterrows():
            print(f"   - {row['component']}: {row['effect']} "
                  f"(β={row['beta']:.3f}, p={row['p_value']:.4f})")
    else:
        print(f"\n   No significant effects at p < 0.05")


if __name__ == '__main__':
    # Run PCA test first
    main()
    
    # Then run PCA LME test
    test_pca_lme()
