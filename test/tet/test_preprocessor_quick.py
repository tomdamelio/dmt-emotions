#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick test of TET preprocessor"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.tet.data_loader import TETDataLoader
from scripts.tet.validator import TETDataValidator
from scripts.tet.preprocessor import TETPreprocessor
import config

# Load data
print("Loading data...")
loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
data = loader.load_data()
print(f"Loaded: {len(data)} rows")

# Validate and clamp
print("\nValidating...")
validator = TETDataValidator(data, config.TET_DIMENSION_COLUMNS)
results = validator.validate_all()
if len(results['range_violations']) > 0:
    data, _ = validator.clamp_out_of_range_values()
    print(f"Clamped {len(results['range_violations'])} values")

# Preprocess
print("\nPreprocessing...")
preprocessor = TETPreprocessor(data, config.TET_DIMENSION_COLUMNS)
data_preprocessed = preprocessor.preprocess_all()

# Verify output
print(f"\nOutput shape: {data_preprocessed.shape}")
print(f"Columns: {len(data_preprocessed.columns)}")

# Check z-scored columns
z_cols = [col for col in data_preprocessed.columns if '_z' in col]
print(f"Z-scored columns: {len(z_cols)}")

# Check composite indices
composite_cols = [col for col in data_preprocessed.columns if 'index' in col]
print(f"Composite indices: {composite_cols}")

# Verify standardization for one subject
subject = data_preprocessed['subject'].iloc[0]
subj_data = data_preprocessed[data_preprocessed['subject'] == subject]
z_values = subj_data[[col for col in z_cols if col != 'affect_index_z' and col != 'imagery_index_z' and col != 'self_index_z']].values.flatten()
print(f"\nSubject {subject} z-score stats:")
print(f"  Mean: {z_values.mean():.6f} (should be ≈0)")
print(f"  Std: {z_values.std():.6f} (should be ≈1)")

print("\n✓ Preprocessor test complete!")
