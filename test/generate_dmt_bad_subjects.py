# -*- coding: utf-8 -*-
"""
Generate per-signal lists of subjects rejected for any DMT condition.

This script parses validation_log.json (root) and collects subjects that have
category == "bad" OR empty ("") for at least one DMT recording (high or low)
per signal type.

Output JSON structure (written to test/dmt_bad_subjects.json by default):
{
  "eda": ["S05", "S12", ...],  # subject rejected if any DMT file is bad or empty
  "ecg": ["S03", ...],
  "resp": ["S10", ...]
}

Usage:
  python test/generate_dmt_bad_subjects.py
  python test/generate_dmt_bad_subjects.py --out test/my_bad_subjects.json
"""

import os
import sys
import json
import argparse
from typing import Dict, List


def load_validation_log(path: str) -> Dict:
    """Load validation_log.json, returning an empty structure if missing."""
    if not os.path.exists(path):
        print(f"⚠️  validation_log not found: {path}")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading validation log: {e}")
        return {}


def collect_bad_subjects_for_dmt(validation_data: Dict) -> Dict[str, List[str]]:
    """
    Scan validation_data for subjects with any DMT file marked as category == 'bad'.

    Returns a dict with keys 'eda', 'ecg', 'resp' and values as unique subject lists.
    """
    result: Dict[str, List[str]] = {"eda": [], "ecg": [], "resp": []}
    subjects = (validation_data or {}).get('subjects', {})

    for subject, files in subjects.items():
        # files: dict[file_key -> dict[signal_type -> entry]]
        for file_key, per_signal in files.items():
            # We only consider DMT files (both sessions, any condition)
            # Expected file_key like: 'dmt_session1_high', 'dmt_session2_low'
            if not isinstance(file_key, str):
                continue
            if not file_key.startswith('dmt_'):
                continue

            if not isinstance(per_signal, dict):
                continue

            for signal_type, entry in per_signal.items():
                if signal_type not in result:
                    # Only track eda/ecg/resp
                    continue
                try:
                    category = (entry or {}).get('category', '')
                    category_norm = category.strip().lower() if isinstance(category, str) else ''
                except Exception:
                    category_norm = ''

                # Reject if DMT file category is 'bad' OR empty
                if category_norm in ('bad', ''):
                    subj_std = subject.upper()
                    if not subj_std.startswith('S'):
                        subj_std = f"S{subj_std}"
                    if subj_std not in result[signal_type]:
                        result[signal_type].append(subj_std)

    # Sort lists for readability
    for k in result:
        result[k].sort()
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate per-signal BAD subjects for DMT recordings from validation_log.json')
    parser.add_argument('--log', type=str, default='validation_log.json', help='Path to validation_log.json')
    parser.add_argument('--out', type=str, default=os.path.join('test', 'dmt_bad_subjects.json'), help='Output JSON path')
    args = parser.parse_args()

    validation = load_validation_log(args.log)
    bad_subjects = collect_bad_subjects_for_dmt(validation)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(bad_subjects, f, indent=2, ensure_ascii=False)
        print(f"✅ Wrote BAD subjects JSON: {args.out}")
        print(f"   EDA:  {len(bad_subjects['eda'])} subjects")
        print(f"   ECG:  {len(bad_subjects['ecg'])} subjects")
        print(f"   RESP: {len(bad_subjects['resp'])} subjects")
    except Exception as e:
        print(f"❌ Error writing output JSON: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


