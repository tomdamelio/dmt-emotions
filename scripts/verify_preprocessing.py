#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification Script for TET Preprocessing (Requirement 2)

This script verifies that all preprocessing steps work correctly:
1. Session trimming
2. Valence variables
3. Global within-subject standardization
4. Composite index
5. Metadata generation
"""

import sys
import os
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tet.data_loader import TETDataLoader
from tet.validator import TETDataValidator
from tet.preprocessor import TETPreprocessor
from tet.metadata import PreprocessingMetadata
import config

print("=" * 70)
print("VERIFICATION: TET PREPROCESSING (REQUIREMENT 2)")
print("=" * 70)

# Load preprocessed data
print("\n1. Loading preprocessed data...")
data_path = 'results/tet/preprocessed/tet_preprocessed.csv'
if not os.path.exists(data_path):
    print(f"❌ ERROR: Preprocessed data not found at {data_path}")
    print("   Run: python scripts/preprocess_tet_data.py")
    sys.exit(1)

data = pd.read_csv(data_path)
print(f"✓ Loaded {len(data)} rows, {len(data.columns)} columns")

# Load metadata
metadata_path = 'results/tet/preprocessed/preprocessing_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"✓ Loaded metadata")

# Verification 1: Session Trimming
print("\n" + "=" * 70)
print("VERIFICATION 1: SESSION TRIMMING")
print("=" * 70)

rs_data = data[data['state'] == 'RS']
dmt_data = data[data['state'] == 'DMT']

print(f"\nRS sessions:")
rs_sessions = rs_data.groupby(['subject', 'session_id'])
for (subj, sess), group in list(rs_sessions)[:3]:
    t_range = f"{group['t_sec'].min():.1f} - {group['t_sec'].max():.1f}s"
    print(f"  Subject {subj}, Session {sess}: {len(group)} points, t_sec range: {t_range}")

rs_max_time = rs_data['t_sec'].max()
rs_expected_points = 150  # 600s / 4s = 150 points @ 0.25 Hz

print(f"\n✓ RS max time: {rs_max_time}s (expected: <600s)")
print(f"✓ RS points per session: {rs_data.groupby(['subject', 'session_id']).size().unique()} (expected: 150)")

if rs_max_time >= 600:
    print(f"❌ ERROR: RS sessions not properly trimmed (max time: {rs_max_time}s)")
else:
    print("✓ RS sessions properly trimmed to 0-600s")

print(f"\nDMT sessions:")
dmt_sessions = dmt_data.groupby(['subject', 'session_id'])
for (subj, sess), group in list(dmt_sessions)[:3]:
    t_range = f"{group['t_sec'].min():.1f} - {group['t_sec'].max():.1f}s"
    print(f"  Subject {subj}, Session {sess}: {len(group)} points, t_sec range: {t_range}")

dmt_max_time = dmt_data['t_sec'].max()
dmt_expected_points = 300  # 1200s / 4s = 300 points @ 0.25 Hz

print(f"\n✓ DMT max time: {dmt_max_time}s (expected: <1200s)")
print(f"✓ DMT points per session: {dmt_data.groupby(['subject', 'session_id']).size().unique()} (expected: 300)")

if dmt_max_time >= 1200:
    print(f"❌ ERROR: DMT sessions not properly trimmed (max time: {dmt_max_time}s)")
else:
    print("✓ DMT sessions properly trimmed to 0-1200s")

# Verification 2: Valence Variables
print("\n" + "=" * 70)
print("VERIFICATION 2: VALENCE VARIABLES")
print("=" * 70)

if 'valence_pos' not in data.columns or 'valence_neg' not in data.columns:
    print("❌ ERROR: Valence variables not found")
else:
    # Check if valence_pos == pleasantness
    valence_pos_match = (data['valence_pos'] == data['pleasantness']).all()
    valence_neg_match = (data['valence_neg'] == data['unpleasantness']).all()
    
    print(f"valence_pos == pleasantness: {valence_pos_match}")
    print(f"valence_neg == unpleasantness: {valence_neg_match}")
    
    if valence_pos_match and valence_neg_match:
        print("✓ Valence variables correctly created")
    else:
        print("❌ ERROR: Valence variables do not match expected values")

# Verification 3: Global Within-Subject Standardization
print("\n" + "=" * 70)
print("VERIFICATION 3: GLOBAL WITHIN-SUBJECT STANDARDIZATION")
print("=" * 70)

z_cols = [f"{dim}_z" for dim in config.TET_DIMENSION_COLUMNS]

# Check that all z-score columns exist
missing_z_cols = [col for col in z_cols if col not in data.columns]
if missing_z_cols:
    print(f"❌ ERROR: Missing z-score columns: {missing_z_cols}")
else:
    print(f"✓ All {len(z_cols)} z-score columns present")

# Check z-score properties for each subject
print("\nZ-score properties per subject:")
print(f"{'Subject':<10} {'Mean':<12} {'Std':<12} {'Status'}")
print("-" * 50)

all_good = True
for subject in sorted(data['subject'].unique()):
    subj_data = data[data['subject'] == subject]
    z_values = subj_data[z_cols].values.flatten()
    
    z_mean = np.mean(z_values)
    z_std = np.std(z_values, ddof=1)
    
    # Check if mean ≈ 0 and std ≈ 1 (tolerance: 0.01)
    mean_ok = abs(z_mean) < 0.01
    std_ok = abs(z_std - 1.0) < 0.01
    
    status = "✓" if (mean_ok and std_ok) else "❌"
    if not (mean_ok and std_ok):
        all_good = False
    
    print(f"{subject:<10} {z_mean:>11.6f} {z_std:>11.6f} {status}")

if all_good:
    print("\n✓ All subjects have proper standardization (mean≈0, std≈1)")
else:
    print("\n❌ ERROR: Some subjects have improper standardization")

# Verification 4: Composite Indices
print("\n" + "=" * 70)
print("VERIFICATION 4: COMPOSITE INDEX")
print("=" * 70)

composite_indices = ['valence_index_z']

# Check that composite index exists
missing_composites = [col for col in composite_indices if col not in data.columns]
if missing_composites:
    print(f"❌ ERROR: Missing composite index: {missing_composites}")
else:
    print(f"✓ Composite index present: valence_index_z")

# Verify formula
print("\nVerifying composite index formula:")

# valence_index_z = pleasantness_z - unpleasantness_z
valence_computed = data['pleasantness_z'] - data['unpleasantness_z']
valence_match = np.allclose(data['valence_index_z'], valence_computed, rtol=1e-5)
print(f"  valence_index_z formula: {'✓' if valence_match else '❌'}")

if valence_match:
    print("\n✓ Composite index formula correct")
else:
    print("\n❌ ERROR: Composite index formula incorrect")

# Show sample values
print("\nSample composite index values (first 5 rows):")
print(data[['subject', 'state', 't_sec'] + composite_indices].head())

# Verification 5: Metadata
print("\n" + "=" * 70)
print("VERIFICATION 5: METADATA")
print("=" * 70)

required_metadata_keys = [
    'preprocessing_version',
    'timestamp',
    'data_summary',
    'trimming',
    'standardization',
    'valence_variables',
    'composite_indices',
    'dimensions'
]

missing_keys = [key for key in required_metadata_keys if key not in metadata]
if missing_keys:
    print(f"❌ ERROR: Missing metadata keys: {missing_keys}")
else:
    print(f"✓ All required metadata keys present")

# Check composite definition
print("\nComposite index definition in metadata:")
for idx_name, idx_def in metadata['composite_indices'].items():
    print(f"  {idx_name}:")
    print(f"    Formula: {idx_def['formula']}")

# Summary Statistics
print("\nSummary statistics from metadata:")
summary = metadata['data_summary']
print(f"  Subjects: {summary['n_subjects']}")
print(f"  Sessions: {summary['n_sessions']}")
print(f"  Time points: {summary['n_time_points_total']} ({summary['n_time_points_rs']} RS, {summary['n_time_points_dmt']} DMT)")
print(f"  Dimensions: {summary['n_dimensions']}")

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

checks = {
    "Session trimming": rs_max_time < 600 and dmt_max_time < 1200,
    "Valence variables": valence_pos_match and valence_neg_match,
    "Z-score standardization": all_good,
    "Composite index": valence_match,
    "Metadata": len(missing_keys) == 0
}

print("\nVerification Results:")
for check_name, passed in checks.items():
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"  {check_name:<30} {status}")

all_passed = all(checks.values())
print("\n" + "=" * 70)
if all_passed:
    print("✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓")
    print("Requirement 2 (Preprocessing) is working correctly!")
else:
    print("❌❌❌ SOME VERIFICATIONS FAILED ❌❌❌")
    print("Please review the errors above.")
print("=" * 70)

sys.exit(0 if all_passed else 1)
