# -*- coding: utf-8 -*-
"""
Clean Emotiphai SCR event files to respect session duration cuts.

This script scans EDA Emotiphai per-event CSVs produced by preprocess_phys.py
and removes rows with SCR_Onsets_Emotiphai beyond the expected session length.

Rules (sampling rate = from config, default 250 Hz):
- DMT sessions (files containing "_dmt_"): keep onsets <= 20:15 (DURACIONES_ESPERADAS['DMT'])
- Resting sessions (files containing "_rs_"): keep onsets <= 10:15 (DURACIONES_ESPERADAS['Reposo'])

Targets:
- {DERIVATIVES_DATA}/phys/eda/dmt_high/*_emotiphai_scr.csv
- {DERIVATIVES_DATA}/phys/eda/dmt_low/*_emotiphai_scr.csv

Usage:
    python scripts/clean_emotiphai_scr.py
"""

import os
import sys
import glob
import pandas as pd

# Add project root for config import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import DERIVATIVES_DATA, NEUROKIT_PARAMS, DURACIONES_ESPERADAS


def main() -> None:
    sr = int(NEUROKIT_PARAMS.get('sampling_rate_default', 250))
    dmt_sec = float(DURACIONES_ESPERADAS['DMT'])
    rs_sec = float(DURACIONES_ESPERADAS['Reposo'])
    dmt_thresh = int(dmt_sec * sr)
    rs_thresh = int(rs_sec * sr)

    base_dirs = [
        os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high'),
        os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low'),
    ]

    total_files = 0
    modified_files = 0
    total_rows_removed = 0

    print("🔍 Cleaning Emotiphai SCR files...")
    print(f"   Sampling rate: {sr} Hz | DMT limit: {dmt_thresh} samples | RS limit: {rs_thresh} samples")

    for target_dir in base_dirs:
        if not os.path.isdir(target_dir):
            print(f"   ⚠️  Directory not found: {target_dir}")
            continue

        pattern = os.path.join(target_dir, '*_emotiphai_scr.csv')
        for csv_path in glob.glob(pattern):
            total_files += 1
            fname = os.path.basename(csv_path)

            # Decide threshold based on filename
            if '_dmt_' in fname:
                threshold = dmt_thresh
            elif '_rs_' in fname:
                threshold = rs_thresh
            else:
                print(f"   ⚠️  Could not infer session type from filename, skipping: {fname}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"   ❌ Failed to read {fname}: {e}")
                continue

            if 'SCR_Onsets_Emotiphai' not in df.columns:
                print(f"   ⚠️  Column 'SCR_Onsets_Emotiphai' not found in {fname} - skipping")
                continue

            before = len(df)
            df_clean = df[df['SCR_Onsets_Emotiphai'] <= threshold].copy()
            removed = before - len(df_clean)

            if removed > 0:
                try:
                    df_clean.to_csv(csv_path, index=False)
                    modified_files += 1
                    total_rows_removed += removed
                    print(f"   ✂️  Cleaned {fname}: removed {removed} rows (kept {len(df_clean)})")
                except Exception as e:
                    print(f"   ❌ Failed to write cleaned file {fname}: {e}")
            else:
                print(f"   ✅ OK {fname}: no rows beyond threshold")

    print("\n📊 Summary:")
    print(f"   Files scanned: {total_files}")
    print(f"   Files modified: {modified_files}")
    print(f"   Total rows removed: {total_rows_removed}")


if __name__ == '__main__':
    main()


