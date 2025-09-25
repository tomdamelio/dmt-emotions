# -*- coding: utf-8 -*-
"""
Minimal reproduction to debug NeuroKit2 plotting errors for EDA across subjects/sessions.

Usage examples:
  python test/debug_plot_s13_eda_dmt_high.py --subject S13 --reg dmt --cond high --session session2
  python test/debug_plot_s13_eda_dmt_high.py -s s05 --reg rs --cond low   # session inferred from config

This script:
- Loads the CSV and corresponding _info.json for EDA
- Prints lengths and potential mismatches in info arrays (including deep scan)
- Performs detailed SCR pairing analysis
- Attempts nk.eda_plot(signal_df, info_dict) to reproduce the error
- Retries with sanitized/reconciled info, and with df-fixed markers if needed
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import argparse


# Allow importing config from project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import DERIVATIVES_DATA, get_dosis_sujeto, NEUROKIT_PARAMS


def is_sequence_like(value):
    """Return True if value behaves like a 1D sequence of numbers/booleans."""
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return True
    return False


def summarize_info_lengths(info_dict, signal_length):
    """Summarize lengths of sequence-like entries in info dict for mismatch detection."""
    lengths = []
    try:
        for k, v in (info_dict or {}).items():
            if isinstance(v, dict):
                # Recurse one level to catch typical nested keys like 'EDA' / 'SCR'
                for kk, vv in v.items():
                    if is_sequence_like(vv):
                        try:
                            lvv = len(vv)
                        except Exception:
                            lvv = None
                        lengths.append((f"{k}.{kk}", lvv, lvv == signal_length))
            else:
                if is_sequence_like(v):
                    try:
                        lv = len(v)
                    except Exception:
                        lv = None
                    lengths.append((k, lv, lv == signal_length))
    except Exception as e:
        print(f"⚠️  Error summarizing info dict lengths: {e}")
    return lengths


def walk_and_report_lengths(obj, target_lengths=None, path_prefix=""):
    """Recursively walk dict-like structures and print sequence lengths.
    If target_lengths is provided (e.g., {111, 112}), only report matches.
    """
    if target_lengths is None:
        target_lengths = set()
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{path_prefix}.{k}" if path_prefix else str(k)
                if isinstance(v, dict):
                    walk_and_report_lengths(v, target_lengths, path)
                else:
                    if is_sequence_like(v):
                        try:
                            lv = len(v)
                        except Exception:
                            lv = None
                        if not target_lengths or (lv in target_lengths):
                            print(f"   - {path}: length={lv}")
        else:
            # Not a dict; nothing to walk
            pass
    except Exception as e:
        print(f"⚠️  Error walking info: {e}")


def analyze_scr_pairing(info_dict, signal_length):
    """Compute detailed pairing diagnostics for SCR events.
    Returns a dict with counts and reasons for dropped events.
    """
    events = get_scr_events_from_info(info_dict)
    diag = {
        'counts_raw': {k: (len(v) if is_sequence_like(v) else None) for k, v in events.items()},
        'in_bounds_onsets': 0,
        'in_bounds_peaks': 0,
        'paired_valid': 0,
        'dropped_reasons': {'onset_oob': 0, 'peak_oob': 0, 'peak_before_onset': 0}
    }
    on = events.get('SCR_Onsets', [])
    pk = events.get('SCR_Peaks', [])
    amp = events.get('SCR_Amplitude', [])
    n = min(len(on), len(pk), len(amp)) if amp else min(len(on), len(pk))
    if n == 0:
        return diag
    in_bounds_on = []
    in_bounds_pk = []
    reasons = {'onset_oob': [], 'peak_oob': [], 'peak_before_onset': []}
    for i in range(n):
        onset = int(on[i])
        peak = int(pk[i])
        onset_ok = (0 <= onset < signal_length)
        peak_ok = (0 <= peak < signal_length)
        if not onset_ok:
            reasons['onset_oob'].append(i)
        if not peak_ok:
            reasons['peak_oob'].append(i)
        if onset_ok and peak_ok:
            in_bounds_on.append(i)
            in_bounds_pk.append(i)
    diag['in_bounds_onsets'] = len(in_bounds_on)
    diag['in_bounds_peaks'] = len(in_bounds_pk)
    # Paired valid: within bounds and chronological
    paired = 0
    for i in range(n):
        onset = int(on[i])
        peak = int(pk[i])
        if not (0 <= onset < signal_length):
            continue
        if not (0 <= peak < signal_length):
            continue
        if peak < onset:
            reasons['peak_before_onset'].append(i)
            continue
        paired += 1
    diag['paired_valid'] = paired
    for key in reasons:
        diag['dropped_reasons'][key] = len(reasons[key])
    # Show tail examples for visibility
    diag['tail_examples'] = {
        'onsets_last5': list(on[-5:]) if len(on) >= 5 else list(on),
        'peaks_last5': list(pk[-5:]) if len(pk) >= 5 else list(pk)
    }
    return diag


def get_scr_events_from_info(info_dict):
    """
    Extract SCR event arrays from the info dict.
    Returns a dict with keys among:
    ['SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude', 'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime']
    Only includes keys present and sequence-like, converted to lists.
    """
    scr_keys = [
        'SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude',
        'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime'
    ]
    events = {}
    if not isinstance(info_dict, dict):
        return events
    for key in scr_keys:
        val = info_dict.get(key, None)
        if is_sequence_like(val):
            try:
                events[key] = list(val)
            except Exception:
                pass
    return events


def reconcile_scr_events_with_trim(events, signal_length):
    """
    Given SCR event arrays and the trimmed signal length, reconcile for consistency:
    - Align arrays to a common min length among provided arrays
    - Drop events whose onset or peak is outside [0, signal_length-1]
    - Ensure all arrays end up with the same final length
    Returns (reconciled_events, report_dict)
    """
    report = {
        'initial_lengths': {},
        'min_length': None,
        'dropped_out_of_bounds': 0,
        'final_length': None,
        'kept_indices': []  # indices kept after mask
    }
    if not events:
        return events, report

    # Record initial lengths and cast types
    for k, v in events.items():
        report['initial_lengths'][k] = len(v)
        try:
            if k in ['SCR_Onsets', 'SCR_Peaks']:
                events[k] = [int(x) for x in v]
            else:
                events[k] = [float(x) for x in v]
        except Exception:
            pass

    # Align to common min length
    lengths = [len(v) for v in events.values()]
    min_len = min(lengths) if lengths else 0
    report['min_length'] = min_len
    trimmed = {k: v[:min_len] for k, v in events.items()}

    # Build mask for in-bounds onsets/peaks
    mask_valid = [True] * min_len
    on = trimmed.get('SCR_Onsets', None)
    pk = trimmed.get('SCR_Peaks', None)
    if on is not None:
        for i, x in enumerate(on):
            if x < 0 or x >= signal_length:
                mask_valid[i] = False
    if pk is not None:
        for i, x in enumerate(pk):
            if x < 0 or x >= signal_length:
                mask_valid[i] = False

    # Apply mask to all arrays
    if any(not m for m in mask_valid):
        report['dropped_out_of_bounds'] = sum(1 for m in mask_valid if not m)
    reconciled = {}
    kept_indices = [i for i, m in enumerate(mask_valid) if m]
    for k, v in trimmed.items():
        reconciled[k] = [vv for vv, m in zip(v, mask_valid) if m]

    # Final align again to ensure identical lengths
    if reconciled:
        final_min = min(len(v) for v in reconciled.values())
        for k in list(reconciled.keys()):
            reconciled[k] = reconciled[k][:final_min]
        # Truncate kept_indices accordingly (kept order preserved)
        kept_indices = kept_indices[:final_min]
        report['final_length'] = final_min
    else:
        report['final_length'] = 0

    report['kept_indices'] = kept_indices
    return reconciled, report


def infer_session_from_config(subject, cond):
    """Infer session1/session2 from subject dose mapping and condition (high/low)."""
    try:
        dose_session1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_session1 = 'Alta'
    if (cond == 'high' and dose_session1 == 'Alta') or (cond == 'low' and dose_session1 == 'Baja'):
        return 'session1'
    return 'session2'


def build_paths(subject, reg, session, cond):
    """Build CSV and info paths for EDA given parameters."""
    base_dir = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high' if cond == 'high' else 'dmt_low')
    filename = f"{subject}_{reg}_{session}_{cond}.csv"
    csv_path = os.path.join(base_dir, filename)
    info_path = csv_path.replace('.csv', '_info.json')
    return csv_path, info_path


def main():
    parser = argparse.ArgumentParser(description='Debug NeuroKit2 EDA plotting by subject/session/condition.')
    parser.add_argument('--subject', '-s', type=str, default='S13', help='Subject code, e.g., S13 or s13')
    parser.add_argument('--reg', type=str, choices=['dmt', 'rs'], default='dmt', help='Registro: dmt or rs')
    parser.add_argument('--cond', type=str, choices=['high', 'low'], default='high', help='Condition: high or low')
    parser.add_argument('--session', type=str, choices=['session1', 'session2'], default=None, help='Session, optional (inferred if omitted)')
    args = parser.parse_args()

    subject = args.subject.upper()
    if not subject.startswith('S'):
        subject = f"S{subject}"
    reg = args.reg.lower()
    cond = args.cond.lower()
    session = args.session if args.session else infer_session_from_config(subject, cond)

    csv_path, info_path = build_paths(subject, reg, session, cond)

    print(f"\n🧪 Minimal debug for {subject} EDA {reg.upper()} {session} {cond}")
    print(f"   CSV path:  {csv_path}")
    print(f"   INFO path: {info_path}")

    if not os.path.exists(csv_path):
        # Provide helpful alternatives for common typos/variants
        base_dir_high = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high')
        alt1 = os.path.join(base_dir_high, f'{subject}_dmt_session1_high.csv')
        alt2 = os.path.join(base_dir_high, f'{subject}_dmt_session2_high.csv')
        print("❌ CSV not found.")
        print(f"   Tried: {csv_path}")
        print(f"   Alt1:  {alt1} -> {'exists' if os.path.exists(alt1) else 'missing'}")
        print(f"   Alt2:  {alt2} -> {'exists' if os.path.exists(alt2) else 'missing'}")
        sys.exit(1)

    # Load CSV
    df_full = pd.read_csv(csv_path)
    if 'time' not in df_full.columns:
        print("❌ Column 'time' not found in CSV")
        sys.exit(1)
    t = df_full['time']
    df = df_full.drop('time', axis=1)
    print(f"✅ Loaded CSV with {len(df)} samples and {len(df.columns)} variables")
    print(f"   Columns: {list(df.columns)}")

    # Load info dict
    info = None
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print("✅ Loaded info JSON")
        except Exception as e:
            print(f"⚠️  Failed to load info JSON: {e}")
    else:
        print("⚠️  Info JSON not found")

    # Summarize potential length mismatches
    lengths = summarize_info_lengths(info, len(df))
    mismatches = [(k, l, ok) for (k, l, ok) in lengths if l is not None and not ok]
    print(f"\n🔍 Info dict sequence lengths (showing mismatches):")
    if mismatches:
        for k, l, _ in mismatches[:20]:
            print(f"   - {k}: length={l} vs signal={len(df)}")
        if len(mismatches) > 20:
            print(f"   ... and {len(mismatches) - 20} more")
    else:
        print("   No obvious mismatches found (by simple length check)")

    # Deep walk: find any arrays specifically of length 111 or 112 anywhere in info
    print("\n🔎 Deep scan for arrays of length 111 or 112 (any nesting):")
    walk_and_report_lengths(info, target_lengths={111, 112})

    # Detailed SCR pairing analysis to explain where 111 comes from
    print("\n🧮 Detailed SCR pairing analysis:")
    diag = analyze_scr_pairing(info, len(df))
    print(f"   Raw counts: {diag['counts_raw']}")
    print(f"   In-bounds onsets: {diag['in_bounds_onsets']} | in-bounds peaks: {diag['in_bounds_peaks']}")
    print(f"   Paired valid events (onset/peak within bounds, peak≥onset): {diag['paired_valid']}")
    print(f"   Dropped reasons: {diag['dropped_reasons']}")
    print(f"   Tail examples -> onsets: {diag['tail_examples']['onsets_last5']}, peaks: {diag['tail_examples']['peaks_last5']}")

    # Attempt plotting with NeuroKit2
    plt.ion()
    try:
        plt.close('all')
        print("\n📈 Trying nk.eda_plot(df, info)...")
        nk.eda_plot(df, info)
        plt.suptitle(f"{subject} - EDA - {reg.upper()} - {session} - {cond}", fontsize=12)
        plt.tight_layout()
        out_dir = os.path.join('test', 'plots')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{subject.lower()}___eda___{reg}___{session}___{cond}_debug.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.3)
        print(f"✅ Plot generated and saved: {out_path}")
    except Exception as e:
        print(f"⚠️  NeuroKit2 plotting failed: {e}")

        # Try a sanitized info dict that drops entries with mismatched lengths
        if info is not None:
            print("\n🔧 Retrying with sanitized info (dropping mismatched-length arrays)...")
            sanitized = {}
            for k, v in info.items():
                if isinstance(v, dict):
                    sub = {}
                    for kk, vv in v.items():
                        if is_sequence_like(vv):
                            try:
                                if len(vv) == len(df):
                                    sub[kk] = vv
                            except Exception:
                                pass
                        else:
                            sub[kk] = vv
                    sanitized[k] = sub
                else:
                    if not is_sequence_like(v):
                        sanitized[k] = v
                    else:
                        try:
                            if len(v) == len(df):
                                sanitized[k] = v
                        except Exception:
                            pass

            try:
                plt.close('all')
                nk.eda_plot(df, sanitized)
                plt.suptitle(f"{subject} - EDA - {reg.upper()} - {session} - {cond} (sanitized)", fontsize=12)
                plt.tight_layout()
                out_dir = os.path.join('test', 'plots')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'{subject.lower()}___eda___{reg}___{session}___{cond}_debug_sanitized.png')
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.show(block=False)
                plt.pause(0.3)
                print(f"✅ Plot (sanitized) generated and saved: {out_path}")
            except Exception as e2:
                print(f"❌ Plot still failing after sanitization: {e2}")

            # Reconcile SCR event arrays post-trim (drop out-of-bounds onsets/peaks and realign)
            try:
                events = get_scr_events_from_info(info)
                if events:
                    reconciled, report = reconcile_scr_events_with_trim(events, len(df))
                    print("\n🧪 Reconciliation report:")
                    print(f"   Initial lengths: {report['initial_lengths']}")
                    print(f"   Common min length: {report['min_length']}")
                    print(f"   Dropped out-of-bounds: {report['dropped_out_of_bounds']}")
                    print(f"   Final aligned length: {report['final_length']}")

                    fixed_info = dict(info)
                    for k, v in reconciled.items():
                        fixed_info[k] = v

                    plt.close('all')
                    print("\n🔧 Retrying with reconciled SCR events (post-trim)...")
                    nk.eda_plot(df, fixed_info)
                    plt.suptitle(f"{subject} - EDA - {reg.upper()} - {session} - {cond} (reconciled)", fontsize=12)
                    plt.tight_layout()
                    out_dir = os.path.join('test', 'plots')
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{subject.lower()}___eda___{reg}___{session}___{cond}_debug_reconciled.png")
                    plt.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.show(block=False)
                    plt.pause(0.3)
                    print(f"✅ Plot (reconciled) generated and saved: {out_path}")
                else:
                    print("ℹ️  No SCR event arrays found in info to reconcile.")
            except Exception as e3:
                print(f"❌ Plot still failing after reconciliation: {e3}")

            # Final attempt: derive events only from df and use a minimal info dict
            try:
                info_slim = {'sampling_rate': NEUROKIT_PARAMS['sampling_rate_default']}
                plt.close('all')
                print("\n🔧 Retrying with minimal info (sampling_rate only, ignoring event arrays)...")
                nk.eda_plot(df, info_slim)
                plt.suptitle(f"{subject} - EDA - {reg.upper()} - {session} - {cond} (info_slim)", fontsize=12)
                plt.tight_layout()
                out_dir = os.path.join('test', 'plots')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{subject.lower()}___eda___{reg}___{session}___{cond}_debug_info_slim.png")
                plt.savefig(out_path, dpi=200, bbox_inches='tight')
                plt.show(block=False)
                plt.pause(0.3)
                print(f"✅ Plot (info_slim) generated and saved: {out_path}")
            except Exception as e4:
                print(f"❌ Plot still failing with info_slim: {e4}")

            # Force df sample-level markers to match reconciled events exactly, then plot
            try:
                events = get_scr_events_from_info(info)
                if events:
                    reconciled, report = reconcile_scr_events_with_trim(events, len(df))
                    onsets_kept = reconciled.get('SCR_Onsets', [])
                    peaks_kept = reconciled.get('SCR_Peaks', [])

                    df_fixed = df.copy()
                    # Rebuild sample-level binary markers from reconciled indices
                    n_samples = len(df_fixed)
                    on_vec = np.zeros(n_samples, dtype=int)
                    pk_vec = np.zeros(n_samples, dtype=int)
                    for idx in onsets_kept:
                        if 0 <= idx < n_samples:
                            on_vec[idx] = 1
                    for idx in peaks_kept:
                        if 0 <= idx < n_samples:
                            pk_vec[idx] = 1
                    if 'SCR_Onsets' in df_fixed.columns:
                        df_fixed['SCR_Onsets'] = on_vec
                    else:
                        df_fixed.insert(len(df_fixed.columns), 'SCR_Onsets', on_vec)
                    if 'SCR_Peaks' in df_fixed.columns:
                        df_fixed['SCR_Peaks'] = pk_vec
                    else:
                        df_fixed.insert(len(df_fixed.columns), 'SCR_Peaks', pk_vec)

                    fixed_info2 = {'sampling_rate': NEUROKIT_PARAMS['sampling_rate_default']}
                    # Also pass reconciled event arrays explicitly
                    for k, v in reconciled.items():
                        fixed_info2[k] = v

                    plt.close('all')
                    print("\n🔧 Retrying with df-fixed markers + reconciled events...")
                    nk.eda_plot(df_fixed, fixed_info2)
                    plt.suptitle(f"{subject} - EDA - {reg.upper()} - {session} - {cond} (df_fixed)", fontsize=12)
                    plt.tight_layout()
                    out_dir = os.path.join('test', 'plots')
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{subject.lower()}___eda___{reg}___{session}___{cond}_debug_df_fixed.png")
                    plt.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.show(block=False)
                    plt.pause(0.3)
                    print(f"✅ Plot (df_fixed) generated and saved: {out_path}")
                else:
                    print("ℹ️  No SCR event arrays found to build df-fixed markers.")
            except Exception as e5:
                print(f"❌ Plot still failing with df-fixed approach: {e5}")


if __name__ == '__main__':
    main()


