# -*- coding: utf-8 -*-
"""
Inspection Script for PCA Analysis Results

This script loads and displays PCA analysis results including:
- Variance explained by each component
- Top loadings for PC1 and PC2
- Significant LME effects for PC1 and PC2
- Interpretation guide for PC effects

Usage:
    python scripts/inspect_pca_results.py
    python scripts/inspect_pca_results.py --results-dir results/tet/pca
    python scripts/inspect_pca_results.py --top-n 5
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def load_variance_explained(results_dir: str) -> pd.DataFrame:
    """
    Load variance explained by each component.
    
    Args:
        results_dir: Directory containing PCA results
    
    Returns:
        DataFrame with variance explained
    """
    variance_path = os.path.join(results_dir, 'pca_variance_explained.csv')
    
    if not os.path.exists(variance_path):
        raise FileNotFoundError(f"Variance file not found: {variance_path}")
    
    return pd.read_csv(variance_path)


def load_loadings(results_dir: str) -> pd.DataFrame:
    """
    Load PCA loadings.
    
    Args:
        results_dir: Directory containing PCA results
    
    Returns:
        DataFrame with loadings
    """
    loadings_path = os.path.join(results_dir, 'pca_loadings.csv')
    
    if not os.path.exists(loadings_path):
        raise FileNotFoundError(f"Loadings file not found: {loadings_path}")
    
    return pd.read_csv(loadings_path)


def load_lme_results(results_dir: str) -> pd.DataFrame:
    """
    Load LME results for PC scores.
    
    Args:
        results_dir: Directory containing PCA results
    
    Returns:
        DataFrame with LME results
    """
    lme_path = os.path.join(results_dir, 'pca_lme_results.csv')
    
    if not os.path.exists(lme_path):
        raise FileNotFoundError(f"LME results file not found: {lme_path}")
    
    return pd.read_csv(lme_path)


def display_variance_explained(variance_df: pd.DataFrame):
    """
    Display variance explained by each component.
    
    Args:
        variance_df: DataFrame with variance explained
    """
    print("\n" + "="*80)
    print("VARIANCE EXPLAINED BY EACH COMPONENT")
    print("="*80)
    
    for _, row in variance_df.iterrows():
        component = row['component']
        var_exp = row['variance_explained'] * 100
        cum_var = row['cumulative_variance'] * 100
        
        # Create bar visualization
        bar_length = int(var_exp / 2)  # Scale to fit in terminal
        bar = "█" * bar_length
        
        print(f"\n{component}:")
        print(f"  Variance explained: {var_exp:6.2f}% {bar}")
        print(f"  Cumulative:         {cum_var:6.2f}%")
    
    print("\n" + "-"*80)
    total_var = variance_df['cumulative_variance'].iloc[-1] * 100
    print(f"Total variance explained by {len(variance_df)} components: {total_var:.2f}%")
    print("="*80)


def display_top_loadings(loadings_df: pd.DataFrame, component: str, top_n: int = 10):
    """
    Display top loadings for a specific component.
    
    Args:
        loadings_df: DataFrame with loadings
        component: Component name (e.g., 'PC1')
        top_n: Number of top loadings to display
    """
    # Filter for component
    comp_loadings = loadings_df[loadings_df['component'] == component].copy()
    
    # Sort by absolute loading value
    comp_loadings['abs_loading'] = comp_loadings['loading'].abs()
    comp_loadings = comp_loadings.sort_values('abs_loading', ascending=False)
    
    print(f"\n{component} - Top {top_n} Loadings:")
    print("-" * 60)
    print(f"{'Dimension':<25} {'Loading':>10} {'Direction':>15}")
    print("-" * 60)
    
    for i, row in comp_loadings.head(top_n).iterrows():
        dimension = row['dimension'].replace('_z', '')
        loading = row['loading']
        direction = "Positive +" if loading > 0 else "Negative -"
        
        # Create bar visualization
        bar_length = int(abs(loading) * 20)  # Scale for visualization
        bar = "█" * bar_length
        
        print(f"{dimension:<25} {loading:>10.3f} {direction:>15}")
        print(f"{'':25} {bar}")


def display_lme_results(lme_df: pd.DataFrame, component: str, alpha: float = 0.05):
    """
    Display significant LME effects for a component.
    
    Args:
        lme_df: DataFrame with LME results
        component: Component name (e.g., 'PC1')
        alpha: Significance threshold
    """
    # Filter for component
    comp_results = lme_df[lme_df['component'] == component].copy()
    
    # Identify significant effects
    comp_results['significant'] = comp_results['p_value'] < alpha
    
    print(f"\n{component} - LME Fixed Effects:")
    print("-" * 80)
    print(f"{'Effect':<35} {'Beta':>10} {'95% CI':>20} {'p-value':>10} {'Sig':>5}")
    print("-" * 80)
    
    for _, row in comp_results.iterrows():
        effect = row['effect']
        beta = row['beta']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        p_value = row['p_value']
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        ci_str = f"[{ci_lower:6.3f}, {ci_upper:6.3f}]"
        p_str = f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"
        
        print(f"{effect:<35} {beta:>10.3f} {ci_str:>20} {p_str:>10} {sig:>5}")
    
    print("-" * 80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")


def display_interpretation_guide():
    """
    Display interpretation guide for PC effects.
    """
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE FOR PRINCIPAL COMPONENTS")
    print("="*80)
    
    print("\n1. VARIANCE EXPLAINED")
    print("-" * 80)
    print("   - Shows how much of the total variance in the 15 TET dimensions")
    print("     is captured by each principal component")
    print("   - Higher variance = more important component")
    print("   - Cumulative variance shows total explained by all components")
    
    print("\n2. LOADINGS")
    print("-" * 80)
    print("   - Loadings represent the contribution of each original dimension")
    print("     to the principal component")
    print("   - Positive loading: dimension increases with PC score")
    print("   - Negative loading: dimension decreases with PC score")
    print("   - Larger absolute value = stronger contribution")
    print("   - Loadings help interpret what each PC represents")
    
    print("\n3. LME FIXED EFFECTS")
    print("-" * 80)
    print("   - Tests whether PC scores differ by experimental conditions")
    print("   - Main effects:")
    print("     • State (DMT vs RS): Does DMT change this PC?")
    print("     • Dose (Alta vs Baja): Does dose affect this PC?")
    print("     • Time: Does this PC change over time?")
    print("   - Interactions:")
    print("     • State:Dose: Is the DMT effect moderated by dose?")
    print("     • State:Time: Does the time course differ between DMT and RS?")
    print("     • Dose:Time: Does the time course differ by dose?")
    
    print("\n4. INTERPRETATION STRATEGY")
    print("-" * 80)
    print("   Step 1: Examine loadings to understand what each PC represents")
    print("   Step 2: Look at variance explained to assess importance")
    print("   Step 3: Check LME results for significant experimental effects")
    print("   Step 4: Combine loadings + effects to interpret findings")
    print("   ")
    print("   Example:")
    print("   - If PC1 has high loadings on pleasantness, bliss (positive)")
    print("     and anxiety, unpleasantness (negative)")
    print("   - And PC1 shows significant State effect (DMT > RS)")
    print("   - Interpretation: DMT increases positive affect dimension")
    
    print("\n5. DIMENSIONALITY REDUCTION BENEFITS")
    print("-" * 80)
    print("   - Reduces 15 correlated dimensions to 2-3 independent components")
    print("   - Captures major modes of experiential variation")
    print("   - Reduces multiple testing burden (fewer tests)")
    print("   - Reveals underlying structure in subjective experience")
    
    print("\n" + "="*80)


def main():
    """Main inspection workflow."""
    parser = argparse.ArgumentParser(
        description='Inspect PCA analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=os.path.join(config.TET_RESULTS_DIR, 'pca'),
        help='Directory containing PCA results (default: results/tet/pca)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top loadings to display (default: 10)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance threshold for LME effects (default: 0.05)'
    )
    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        default=['PC1', 'PC2'],
        help='Components to display (default: PC1 PC2)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("PCA ANALYSIS RESULTS INSPECTION")
    print("="*80)
    print(f"\nResults directory: {args.results_dir}")
    
    try:
        # Load data
        print("\nLoading results...")
        variance_df = load_variance_explained(args.results_dir)
        loadings_df = load_loadings(args.results_dir)
        lme_df = load_lme_results(args.results_dir)
        print("✓ All files loaded successfully")
        
        # Display variance explained
        display_variance_explained(variance_df)
        
        # Display loadings and LME results for each component
        for component in args.components:
            print("\n" + "="*80)
            print(f"COMPONENT: {component}")
            print("="*80)
            
            # Check if component exists
            if component not in loadings_df['component'].values:
                print(f"\n⚠ Warning: {component} not found in results")
                continue
            
            # Display top loadings
            display_top_loadings(loadings_df, component, args.top_n)
            
            # Display LME results
            if component in lme_df['component'].values:
                display_lme_results(lme_df, component, args.alpha)
            else:
                print(f"\n⚠ Warning: No LME results found for {component}")
        
        # Display interpretation guide
        display_interpretation_guide()
        
        print("\n✓ Inspection complete")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have run the PCA analysis first:")
        print("  python scripts/compute_pca_analysis.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
