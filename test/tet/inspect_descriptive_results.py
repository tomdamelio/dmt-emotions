#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inspection of descriptive statistics results
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("DESCRIPTIVE STATISTICS RESULTS INSPECTION")
print("=" * 70)

# Load time course data
print("\n1. TIME COURSE DATA")
tc = pd.read_csv('results/tet/descriptive/time_course_all_dimensions.csv')
print(f"   Shape: {tc.shape}")
print(f"   Columns: {list(tc.columns)}")
print(f"\n   Sample (first 5 rows):")
print(tc.head().to_string(index=False))

print(f"\n   Dimensions analyzed: {tc['dimension'].nunique()}")
print(f"   States: {tc['state'].unique()}")
print(f"   Time bins:")
for state in tc['state'].unique():
    n_bins = tc[tc['state'] == state]['t_bin'].nunique()
    print(f"     {state}: {n_bins} bins")

# Load session metrics data
print("\n2. SESSION METRICS DATA")
sm = pd.read_csv('results/tet/descriptive/session_metrics_all_dimensions.csv')
print(f"   Shape: {sm.shape}")
print(f"   Columns: {list(sm.columns)}")
print(f"\n   Sample (first 3 rows):")
print(sm.head(3).to_string(index=False))

print(f"\n   Dimensions analyzed: {sm['dimension'].nunique()}")
print(f"   Sessions: {sm.groupby(['subject', 'session_id']).ngroups}")
print(f"   States: {sm['state'].unique()}")

# Check for missing values
print("\n3. DATA QUALITY")
print(f"   Time course missing values: {tc.isnull().sum().sum()}")
print(f"   Session metrics missing values:")
for col in ['peak_value', 'time_to_peak', 'auc_0_9min', 'slope_0_2min', 'slope_5_9min']:
    n_missing = sm[col].isnull().sum()
    if n_missing > 0:
        print(f"     {col}: {n_missing} ({n_missing/len(sm)*100:.1f}%)")

# Sample statistics
print("\n4. SAMPLE STATISTICS (affect_index_z)")
affect_tc = tc[tc['dimension'] == 'affect_index_z']
affect_sm = sm[sm['dimension'] == 'affect_index_z']

print(f"\n   Time Course (affect_index_z):")
print(f"     Mean range: [{affect_tc['mean'].min():.3f}, {affect_tc['mean'].max():.3f}]")
print(f"     SEM range: [{affect_tc['sem'].min():.3f}, {affect_tc['sem'].max():.3f}]")

print(f"\n   Session Metrics (affect_index_z):")
print(f"     Peak value range: [{affect_sm['peak_value'].min():.3f}, {affect_sm['peak_value'].max():.3f}]")
print(f"     Time to peak range: [{affect_sm['time_to_peak'].min():.0f}s, {affect_sm['time_to_peak'].max():.0f}s]")
print(f"     AUC (0-9 min) range: [{affect_sm['auc_0_9min'].min():.1f}, {affect_sm['auc_0_9min'].max():.1f}]")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
