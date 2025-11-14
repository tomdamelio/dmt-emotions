#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test of TET Peak and AUC Analyzer"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from scripts.tet.peak_auc_analyzer import TETPeakAUCAnalyzer
import config

# Load preprocessed data
print("Loading preprocessed data...")
data_path = 'results/tet/preprocessed/tet_preprocessed.csv'

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    print("Run preprocess_tet_data.py first")
    sys.exit(1)

data = pd.read_csv(data_path)
print(f"Loaded: {len(data)} rows, {data['subject'].nunique()} subjects")

# Get z-scored dimensions (exclude composite indices)
z_dimensions = [col for col in data.columns 
                if col.endswith('_z') 
                and 'index' not in col]
print(f"Z-scored dimensions: {len(z_dimensions)}")

# Initialize analyzer
print("\nInitializing analyzer...")
analyzer = TETPeakAUCAnalyzer(data, z_dimensions)

# Compute metrics
print("\nComputing metrics...")
metrics = analyzer.compute_metrics()
print(f"Metrics computed: {len(metrics)} rows")
print(f"  Sessions: {metrics.groupby(['subject', 'session']).ngroups}")
print(f"  Dimensions: {metrics['dimension'].nunique()}")

# Show sample metrics
print("\nSample metrics (first dimension, first 5 sessions):")
sample = metrics[metrics['dimension'] == z_dimensions[0]].head()
print(sample[['subject', 'dose', 'peak', 'time_to_peak_min', 'auc_0_9']])

# Perform statistical tests
print("\nPerforming statistical tests...")
results = analyzer.perform_tests()
print(f"Tests performed: {len(results)}")
print(f"Significant results: {results['significant'].sum()}")

# Show significant results
if results['significant'].sum() > 0:
    print("\nSignificant results (p_fdr < 0.05):")
    sig_results = results[results['significant']].sort_values('p_fdr')
    print(sig_results[['dimension', 'metric', 'n_pairs', 'p_fdr', 'effect_r']])
else:
    print("\nNo significant results found")

# Show top effects by effect size
print("\nTop 5 effects by absolute effect size:")
top_effects = results.nlargest(5, 'effect_r', keep='all')
print(top_effects[['dimension', 'metric', 'effect_r', 'ci_lower', 'ci_upper', 'p_fdr']])

print("\n✓ Peak and AUC analyzer test complete!")
