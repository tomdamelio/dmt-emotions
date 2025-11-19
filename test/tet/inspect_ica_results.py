"""
Inspection script for ICA analysis results.

This script loads and displays ICA results to help interpret the analysis
and assess whether ICA reveals patterns beyond PCA.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import config


def main():
    """Main inspection workflow."""
    print("=" * 80)
    print("ICA RESULTS INSPECTION")
    print("=" * 80)
    print()
    
    ica_dir = Path(config.TET_RESULTS_DIR) / 'ica'
    
    if not ica_dir.exists():
        print(f"Error: ICA results directory not found: {ica_dir}")
        print("Run ICA analysis first:")
        print("  python scripts/compute_ica_analysis.py")
        return
    
    # 1. Load and display mixing matrix
    print("1. ICA MIXING MATRIX")
    print("-" * 80)
    
    mixing_path = ica_dir / 'ica_mixing_matrix.csv'
    if mixing_path.exists():
        mixing_df = pd.read_csv(mixing_path)
        
        # Show top mixing coefficients for IC1 and IC2
        for ic in ['IC1', 'IC2']:
            ic_mixing = mixing_df[mixing_df['component'] == ic].copy()
            ic_mixing['abs_mixing'] = ic_mixing['mixing_coef'].abs()
            ic_mixing = ic_mixing.sort_values('abs_mixing', ascending=False)
            
            print(f"\n{ic} - Top Mixing Coefficients:")
            print(ic_mixing[['dimension', 'mixing_coef']].head(5).to_string(index=False))
    else:
        print(f"  Mixing matrix not found: {mixing_path}")
    
    print()
    
    # 2. Load and display IC-PC correlations
    print("2. IC-PC ALIGNMENT")
    print("-" * 80)
    
    corr_path = ica_dir / 'ica_pca_correlation.csv'
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        
        print("\nIC-PC Correlation Matrix:")
        corr_pivot = corr_df.pivot(index='ic_component', columns='pc_component', values='correlation')
        print(corr_pivot.to_string())
        
        print("\n\nStrongest IC-PC Alignments:")
        top_corrs = corr_df.nlargest(5, 'abs_correlation')
        print(top_corrs[['ic_component', 'pc_component', 'correlation']].to_string(index=False))
        
        # Interpretation
        print("\n\nInterpretation:")
        high_corr = (corr_df['abs_correlation'] > 0.7).sum()
        low_corr = (corr_df['abs_correlation'] < 0.3).sum()
        print(f"  High correlations (|r| > 0.7): {high_corr}")
        print(f"  Low correlations (|r| < 0.3): {low_corr}")
        
        if low_corr > 0:
            print(f"\n  → ICA reveals {low_corr} components with distinct structure from PCA")
        else:
            print("\n  → ICA components show substantial overlap with PCA")
    else:
        print(f"  Correlation file not found: {corr_path}")
    
    print()
    
    # 3. Load and display LME results
    print("3. ICA LME RESULTS")
    print("-" * 80)
    
    lme_path = ica_dir / 'ica_lme_results.csv'
    if lme_path.exists():
        lme_df = pd.read_csv(lme_path)
        
        # Show significant effects
        sig_effects = lme_df[lme_df['p_value'] < 0.05].copy()
        
        print(f"\nSignificant Effects (p < 0.05): {len(sig_effects)}")
        
        if not sig_effects.empty:
            print("\nTop Significant Effects:")
            sig_effects = sig_effects.sort_values('p_value')
            print(sig_effects[['component', 'effect', 'beta', 'p_value']].head(10).to_string(index=False))
        else:
            print("  No significant effects found")
    else:
        print(f"  LME results not found: {lme_path}")
    
    print()
    
    # 4. Load and display comparison report summary
    print("4. ICA-PCA COMPARISON SUMMARY")
    print("-" * 80)
    
    report_path = ica_dir / 'ica_pca_comparison_report.md'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract executive summary
        in_summary = False
        summary_lines = []
        for line in lines:
            if '## 1. Executive Summary' in line:
                in_summary = True
                continue
            elif in_summary and line.startswith('## '):
                break
            elif in_summary:
                summary_lines.append(line.rstrip())
        
        print('\n'.join(summary_lines))
        
        print(f"\n\nFull report available at: {report_path}")
    else:
        print(f"  Comparison report not found: {report_path}")
    
    print()
    
    # 5. Highlight key findings
    print("5. KEY FINDINGS")
    print("-" * 80)
    print()
    
    if corr_path.exists() and lme_path.exists():
        corr_df = pd.read_csv(corr_path)
        lme_df = pd.read_csv(lme_path)
        
        # Check if ICA reveals patterns beyond PC1/PC2
        low_corr_components = corr_df[corr_df['abs_correlation'] < 0.3]['ic_component'].unique()
        
        if len(low_corr_components) > 0:
            print("✓ ICA reveals experiential structure beyond PCA:")
            for ic in low_corr_components:
                print(f"  - {ic} shows low correlation with all PCs")
            print()
            print("  Recommendation: Include ICA in main analysis")
        else:
            print("✗ ICA components show substantial overlap with PCA")
            print()
            print("  Recommendation: PCA sufficient for main analysis")
            print("  (ICA can be reported as supplementary)")
    
    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
