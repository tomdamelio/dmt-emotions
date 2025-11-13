#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inspection of peak and AUC analysis results

This script loads and displays peak_auc_tests.csv results, showing:
- Summary statistics by metric type
- Significant dimensions with effect sizes
- Top dimensions by effect size magnitude
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("PEAK AND AUC ANALYSIS RESULTS INSPECTION")
print("=" * 70)

# Define file path
results_file = 'results/tet/peak_auc/peak_auc_tests.csv'

# Check if file exists
if not os.path.exists(results_file):
    print(f"\nERROR: Results file not found: {results_file}")
    print("Please run scripts/compute_peak_auc.py first.")
    exit(1)

# Load test results
print(f"\n1. LOADING RESULTS")
print(f"   File: {results_file}")
results = pd.read_csv(results_file)
print(f"   Shape: {results.shape}")
print(f"   Columns: {list(results.columns)}")

# Summary by metric type
print(f"\n2. SUMMARY BY METRIC TYPE")
print(f"   Total tests: {len(results)}")

for metric in results['metric'].unique():
    metric_data = results[results['metric'] == metric]
    n_total = len(metric_data)
    n_sig = (metric_data['significant'] == True).sum()
    pct_sig = 100 * n_sig / n_total if n_total > 0 else 0
    
    print(f"\n   {metric}:")
    print(f"     Total tests: {n_total}")
    print(f"     Significant (FDR < 0.05): {n_sig} ({pct_sig:.1f}%)")
    print(f"     Mean n_pairs: {metric_data['n_pairs'].mean():.1f}")
    print(f"     Effect size range: [{metric_data['effect_r'].min():.3f}, "
          f"{metric_data['effect_r'].max():.3f}]")

# Significant dimensions with effect sizes
print(f"\n3. SIGNIFICANT DIMENSIONS (FDR < 0.05)")
sig_results = results[results['significant'] == True].sort_values('p_fdr')
print(f"   Total: {len(sig_results)}")

if len(sig_results) > 0:
    print(f"\n   By metric type:")
    for metric in sig_results['metric'].unique():
        metric_sig = sig_results[sig_results['metric'] == metric]
        print(f"\n     {metric}: {len(metric_sig)} dimensions")
        
        # Show dimensions with effect sizes
        display_cols = ['dimension', 'effect_r', 'ci_lower', 'ci_upper', 'p_fdr']
        print(metric_sig[display_cols].to_string(index=False))
else:
    print("   No significant results found.")

# Top dimensions by effect size
print(f"\n4. TOP DIMENSIONS BY EFFECT SIZE MAGNITUDE")

for metric in results['metric'].unique():
    metric_data = results[results['metric'] == metric].copy()
    
    # Sort by absolute effect size
    metric_data['abs_effect_r'] = metric_data['effect_r'].abs()
    top_dims = metric_data.nlargest(5, 'abs_effect_r')
    
    print(f"\n   {metric} (top 5):")
    display_cols = ['dimension', 'effect_r', 'ci_lower', 'ci_upper', 
                   'p_value', 'p_fdr', 'significant']
    print(top_dims[display_cols].to_string(index=False))

# Effect size interpretation guide
print(f"\n5. EFFECT SIZE INTERPRETATION")
print(f"   Effect size r (Cohen's guidelines):")
print(f"     Small:  |r| ≈ 0.1")
print(f"     Medium: |r| ≈ 0.3")
print(f"     Large:  |r| ≈ 0.5")
print(f"\n   Positive r: High dose > Low dose")
print(f"   Negative r: High dose < Low dose")

# Distribution of effect sizes
print(f"\n6. EFFECT SIZE DISTRIBUTION")
for metric in results['metric'].unique():
    metric_data = results[results['metric'] == metric]
    effect_sizes = metric_data['effect_r'].values
    
    print(f"\n   {metric}:")
    print(f"     Mean: {np.mean(effect_sizes):.3f}")
    print(f"     Median: {np.median(effect_sizes):.3f}")
    print(f"     Std: {np.std(effect_sizes):.3f}")
    print(f"     Range: [{np.min(effect_sizes):.3f}, {np.max(effect_sizes):.3f}]")
    
    # Count by magnitude
    small = np.sum(np.abs(effect_sizes) < 0.3)
    medium = np.sum((np.abs(effect_sizes) >= 0.3) & (np.abs(effect_sizes) < 0.5))
    large = np.sum(np.abs(effect_sizes) >= 0.5)
    
    print(f"     Small (|r| < 0.3): {small}")
    print(f"     Medium (0.3 ≤ |r| < 0.5): {medium}")
    print(f"     Large (|r| ≥ 0.5): {large}")

# Dimensions with consistent effects across metrics
print(f"\n7. DIMENSIONS WITH CONSISTENT EFFECTS")
print(f"   (Significant across multiple metrics)")

# Count significant results per dimension
sig_counts = sig_results.groupby('dimension').size().sort_values(ascending=False)

if len(sig_counts) > 0:
    print(f"\n   Dimensions significant in multiple metrics:")
    for dim, count in sig_counts.items():
        if count > 1:
            dim_results = sig_results[sig_results['dimension'] == dim]
            metrics_list = ', '.join(dim_results['metric'].values)
            mean_effect = dim_results['effect_r'].mean()
            print(f"     {dim}: {count} metrics ({metrics_list}), "
                  f"mean effect_r = {mean_effect:.3f}")
else:
    print("   No dimensions significant in multiple metrics.")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
