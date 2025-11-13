#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test of LME analyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.lme_analyzer import TETLMEAnalyzer
import pandas as pd

print("=" * 70)
print("QUICK LME TEST")
print("=" * 70)

# Load data
print("\n1. Loading preprocessed data...")
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
print(f"   ✓ Loaded {len(data)} rows")

# Test with just 2 dimensions
print("\n2. Testing LME with 2 dimensions...")
test_dims = ['pleasantness_z', 'anxiety_z']

analyzer = TETLMEAnalyzer(data, dimensions=test_dims)
print(f"   ✓ Analyzer initialized")
print(f"   ✓ Data prepared: {len(analyzer.data_lme)} rows")

# Fit models
print("\n3. Fitting models...")
results = analyzer.fit_all_dimensions()
print(f"   ✓ Results: {len(results)} rows")

# Show results
print("\n4. Sample results:")
print(results[['dimension', 'effect', 'beta', 'p_value', 'p_fdr', 'significant']].head(10).to_string(index=False))

# Summary
print("\n5. Summary:")
summary = analyzer.get_summary()
print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
