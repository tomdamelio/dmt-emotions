#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inspection of preprocessed TET data
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("PREPROCESSED TET DATA INSPECTION")
print("=" * 70)

# Load data
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')

print(f"\n1. DATA SHAPE")
print(f"   Rows: {len(data):,}")
print(f"   Columns: {len(data.columns)}")

print(f"\n2. COLUMN GROUPS")
metadata_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
raw_dims = [c for c in data.columns if not c.endswith('_z') and c not in metadata_cols and c not in ['valence_pos', 'valence_neg', 'affect_index_z', 'imagery_index_z', 'self_index_z']]
z_dims = [c for c in data.columns if c.endswith('_z') and 'index' not in c]
composites = ['affect_index_z', 'imagery_index_z', 'self_index_z']
valence = ['valence_pos', 'valence_neg']

print(f"   Metadata: {len(metadata_cols)} columns")
print(f"   Raw dimensions: {len(raw_dims)} columns")
print(f"   Z-scored dimensions: {len(z_dims)} columns")
print(f"   Valence variables: {len(valence)} columns")
print(f"   Composite indices: {len(composites)} columns")

print(f"\n3. DATA BY STATE")
print(f"   RS:  {len(data[data['state'] == 'RS']):,} rows ({len(data[data['state'] == 'RS']) / len(data) * 100:.1f}%)")
print(f"   DMT: {len(data[data['state'] == 'DMT']):,} rows ({len(data[data['state'] == 'DMT']) / len(data) * 100:.1f}%)")

print(f"\n4. SUBJECTS")
subjects = sorted(data['subject'].unique())
print(f"   N = {len(subjects)}")
print(f"   IDs: {', '.join(subjects)}")

print(f"\n5. SESSIONS PER SUBJECT")
sessions_per_subj = data.groupby('subject')['session_id'].nunique()
print(f"   Min: {sessions_per_subj.min()}, Max: {sessions_per_subj.max()}, Mean: {sessions_per_subj.mean():.1f}")

print(f"\n6. TIME RANGE")
print(f"   RS:  {data[data['state'] == 'RS']['t_sec'].min():.0f} - {data[data['state'] == 'RS']['t_sec'].max():.0f} seconds")
print(f"   DMT: {data[data['state'] == 'DMT']['t_sec'].min():.0f} - {data[data['state'] == 'DMT']['t_sec'].max():.0f} seconds")

print(f"\n7. RAW DIMENSION RANGES")
for dim in raw_dims[:5]:  # Show first 5
    print(f"   {dim:<25} [{data[dim].min():.2f}, {data[dim].max():.2f}]")
print(f"   ... ({len(raw_dims) - 5} more)")

print(f"\n8. Z-SCORE RANGES")
z_values = data[z_dims].values.flatten()
print(f"   Global: [{np.min(z_values):.2f}, {np.max(z_values):.2f}]")
print(f"   Mean: {np.mean(z_values):.6f}, Std: {np.std(z_values, ddof=1):.6f}")

print(f"\n9. COMPOSITE INDEX RANGES")
for comp in composites:
    print(f"   {comp:<20} [{data[comp].min():.2f}, {data[comp].max():.2f}]")

print(f"\n10. MISSING VALUES")
missing = data.isnull().sum()
if missing.sum() == 0:
    print(f"   ✓ No missing values")
else:
    print(f"   ❌ {missing.sum()} missing values found:")
    print(missing[missing > 0])

print(f"\n11. SAMPLE DATA (first 3 rows)")
print(data[['subject', 'state', 't_sec', 'pleasantness', 'pleasantness_z', 'affect_index_z']].head(3).to_string(index=False))

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
