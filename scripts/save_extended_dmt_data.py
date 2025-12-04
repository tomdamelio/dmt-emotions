# -*- coding: utf-8 -*-
"""
Save Extended DMT Data (~19 minutes) for Composite Analysis.

This script extracts and saves extended DMT data (High vs Low, ~19 minutes)
from the three physiological modalities (HR, SMNA, RVT) for use in composite
arousal index analysis.

Outputs: results/composite/extended_dmt_*.csv

Run:
  python scripts/save_extended_dmt_data.py
"""

import os
import sys
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_ECG,
    SUJETOS_VALIDADOS_EDA,
    SUJETOS_VALIDADOS_RESP,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Configuration
WINDOW_SIZE_SEC = 30
LIMIT_SEC = 1150.0  # ~19 minutes
TOTAL_WINDOWS = int(np.floor(LIMIT_SEC / WINDOW_SIZE_SEC))  # ~38 windows
MIN_SAMPLES_PER_WINDOW = 10

OUT_DIR = './results/composite/'


#############################
# Helper functions (copied from individual scripts)
#############################

def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


#############################
# ECG/HR functions
#############################

def build_ecg_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv")
    return high_csv, low_csv


def build_rs_ecg_path(subject: str, session: str) -> str:
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}.csv")


def load_ecg_csv(csv_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if 'ECG_Rate' not in df.columns:
        return None
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    return df


def zscore_with_subject_baseline_hr(t_rs_high, y_rs_high, t_dmt_high, y_dmt_high,
                                     t_rs_low, y_rs_low, t_dmt_low, y_dmt_low):
    """Z-score all sessions from subject using combined baseline."""
    # Remove NaNs only
    mask_rs_h = ~np.isnan(y_rs_high)
    mask_dmt_h = ~np.isnan(y_dmt_high)
    mask_rs_l = ~np.isnan(y_rs_low)
    mask_dmt_l = ~np.isnan(y_dmt_low)
    
    y_rs_h_clean = y_rs_high[mask_rs_h]
    y_dmt_h_clean = y_dmt_high[mask_dmt_h]
    y_rs_l_clean = y_rs_low[mask_rs_l]
    y_dmt_l_clean = y_dmt_low[mask_dmt_l]
    
    y_subject_clean = np.concatenate([y_rs_h_clean, y_dmt_h_clean, y_rs_l_clean, y_dmt_l_clean])
    
    if len(y_subject_clean) < MIN_SAMPLES_PER_WINDOW * 4:
        return None, None, None, None, False
    
    mu = np.nanmean(y_subject_clean)
    sigma = np.nanstd(y_subject_clean, ddof=1)
    
    if sigma == 0.0 or not np.isfinite(sigma):
        return None, None, None, None, False
    
    y_rs_high_z = np.full_like(y_rs_high, np.nan, dtype=float)
    y_dmt_high_z = np.full_like(y_dmt_high, np.nan, dtype=float)
    y_rs_low_z = np.full_like(y_rs_low, np.nan, dtype=float)
    y_dmt_low_z = np.full_like(y_dmt_low, np.nan, dtype=float)
    
    y_rs_high_z[mask_rs_h] = (y_rs_h_clean - mu) / sigma
    y_dmt_high_z[mask_dmt_h] = (y_dmt_h_clean - mu) / sigma
    y_rs_low_z[mask_rs_l] = (y_rs_l_clean - mu) / sigma
    y_dmt_low_z[mask_dmt_l] = (y_dmt_l_clean - mu) / sigma
    
    return y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, True


def compute_hr_mean_per_window(t, hr, window_idx):
    start_time = window_idx * WINDOW_SIZE_SEC
    end_time = (window_idx + 1) * WINDOW_SIZE_SEC
    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None
    hr_win = hr[mask]
    hr_win_valid = hr_win[~np.isnan(hr_win)]
    if len(hr_win_valid) < MIN_SAMPLES_PER_WINDOW:
        return None
    return float(np.nanmean(hr_win_valid))


#############################
# EDA/SMNA functions
#############################

def build_cvx_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high_cvx_decomposition.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low_cvx_decomposition.csv")
    return high_csv, low_csv


def build_rs_cvx_path(subject: str, session: str) -> str:
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}_cvx_decomposition.csv")


def load_cvx_smna(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if 'SMNA' not in df.columns:
        return None
    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)
    smna = pd.to_numeric(df['SMNA'], errors='coerce').fillna(0.0).to_numpy()
    return t, smna


def zscore_with_subject_baseline_smna(t_rs_high, y_rs_high, t_dmt_high, y_dmt_high,
                                      t_rs_low, y_rs_low, t_dmt_low, y_dmt_low):
    """Z-score all sessions from subject using combined baseline."""
    y_all = np.concatenate([y_rs_high, y_dmt_high, y_rs_low, y_dmt_low])
    valid_all = ~np.isnan(y_all)
    
    if np.sum(valid_all) < MIN_SAMPLES_PER_WINDOW * 4:
        return None, None, None, None, False
    
    mu = np.nanmean(y_all)
    sigma = np.nanstd(y_all, ddof=1)
    
    if sigma == 0.0 or not np.isfinite(sigma):
        return None, None, None, None, False
    
    y_rs_high_z = (y_rs_high - mu) / sigma
    y_dmt_high_z = (y_dmt_high - mu) / sigma
    y_rs_low_z = (y_rs_low - mu) / sigma
    y_dmt_low_z = (y_dmt_low - mu) / sigma
    
    return y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, True


def compute_auc_window(t, y, window_idx):
    start_time = window_idx * WINDOW_SIZE_SEC
    end_time = (window_idx + 1) * WINDOW_SIZE_SEC
    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None
    t_win = t[mask]
    y_win = y[mask]
    if len(t_win) < MIN_SAMPLES_PER_WINDOW:
        return None
    return float(np.trapezoid(y_win, t_win))


#############################
# RESP/RVT functions
#############################

def build_resp_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv")
    return high_csv, low_csv


def build_rs_resp_path(subject: str, session: str) -> str:
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}.csv")


def load_resp_csv(csv_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if 'RSP_RVT' not in df.columns:
        return None
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rvt_default', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    return df


def zscore_with_subject_baseline_rvt(t_rs_high, y_rs_high, t_dmt_high, y_dmt_high,
                                     t_rs_low, y_rs_low, t_dmt_low, y_dmt_low):
    """Z-score all sessions from subject using combined baseline."""
    y_all = np.concatenate([y_rs_high, y_dmt_high, y_rs_low, y_dmt_low])
    valid_all = ~np.isnan(y_all)
    
    if np.sum(valid_all) < MIN_SAMPLES_PER_WINDOW * 4:
        return None, None, None, None, False
    
    mu = np.nanmean(y_all)
    sigma = np.nanstd(y_all, ddof=1)
    
    if sigma == 0.0 or not np.isfinite(sigma):
        return None, None, None, None, False
    
    y_rs_high_z = (y_rs_high - mu) / sigma
    y_dmt_high_z = (y_dmt_high - mu) / sigma
    y_rs_low_z = (y_rs_low - mu) / sigma
    y_dmt_low_z = (y_dmt_low - mu) / sigma
    
    return y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, True


def compute_rvt_mean_per_window(t, rvt, window_idx):
    start_time = window_idx * WINDOW_SIZE_SEC
    end_time = (window_idx + 1) * WINDOW_SIZE_SEC
    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None
    rvt_win = rvt[mask]
    rvt_win_valid = rvt_win[~np.isnan(rvt_win)]
    if len(rvt_win_valid) < MIN_SAMPLES_PER_WINDOW:
        return None
    return float(np.nanmean(rvt_win_valid))


#############################
# Main extraction functions
#############################

def extract_extended_hr() -> pd.DataFrame:
    """Extract extended DMT HR data (~19 minutes)."""
    print("Extracting extended HR data...")
    rows: List[Dict] = []
    
    for subject in SUJETOS_VALIDADOS_ECG:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_ecg_paths(subject, high_session, low_session)
            d_high = load_ecg_csv(p_high)
            d_low = load_ecg_csv(p_low)
            
            if d_high is None or d_low is None:
                continue
            
            # Load RS for z-scoring
            p_rsh = build_rs_ecg_path(subject, high_session)
            p_rsl = build_rs_ecg_path(subject, low_session)
            r_high = load_ecg_csv(p_rsh)
            r_low = load_ecg_csv(p_rsl)
            
            if r_high is None or r_low is None:
                continue
            
            # Extract data
            th = d_high['time'].to_numpy()
            yh = pd.to_numeric(d_high['ECG_Rate'], errors='coerce').to_numpy()
            tl = d_low['time'].to_numpy()
            yl = pd.to_numeric(d_low['ECG_Rate'], errors='coerce').to_numpy()
            trh = r_high['time'].to_numpy()
            yrh = pd.to_numeric(r_high['ECG_Rate'], errors='coerce').to_numpy()
            trl = r_low['time'].to_numpy()
            yrl = pd.to_numeric(r_low['ECG_Rate'], errors='coerce').to_numpy()
            
            # Z-score using subject baseline
            yrh_z, yh_z, yrl_z, yl_z, success = zscore_with_subject_baseline_hr(
                trh, yrh, th, yh, trl, yrl, tl, yl
            )
            
            if not success:
                continue
            
            # Compute per-window means
            for window_idx in range(TOTAL_WINDOWS):
                hr_h = compute_hr_mean_per_window(th, yh_z, window_idx)
                hr_l = compute_hr_mean_per_window(tl, yl_z, window_idx)
                
                if hr_h is not None and hr_l is not None:
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'High',
                        'HR': hr_h
                    })
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'Low',
                        'HR': hr_l
                    })
        
        except Exception as e:
            print(f"  {subject}: {str(e)}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"  Extracted {len(df)} rows from {df['subject'].nunique()} subjects")
    return df


def extract_extended_smna() -> pd.DataFrame:
    """Extract extended DMT SMNA data (~19 minutes)."""
    print("Extracting extended SMNA data...")
    rows: List[Dict] = []
    
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_smna(p_high)
            d_low = load_cvx_smna(p_low)
            
            if d_high is None or d_low is None:
                continue
            
            # Load RS for z-scoring
            p_rsh = build_rs_cvx_path(subject, high_session)
            p_rsl = build_rs_cvx_path(subject, low_session)
            r_high = load_cvx_smna(p_rsh)
            r_low = load_cvx_smna(p_rsl)
            
            if r_high is None or r_low is None:
                continue
            
            # Extract data
            th, yh = d_high
            tl, yl = d_low
            trh, yrh = r_high
            trl, yrl = r_low
            
            # Z-score using subject baseline
            yrh_z, yh_z, yrl_z, yl_z, success = zscore_with_subject_baseline_smna(
                trh, yrh, th, yh, trl, yrl, tl, yl
            )
            
            if not success:
                continue
            
            # Compute per-window AUC
            for window_idx in range(TOTAL_WINDOWS):
                auc_h = compute_auc_window(th, yh_z, window_idx)
                auc_l = compute_auc_window(tl, yl_z, window_idx)
                
                if auc_h is not None and auc_l is not None:
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'High',
                        'SMNA_AUC': auc_h
                    })
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'Low',
                        'SMNA_AUC': auc_l
                    })
        
        except Exception as e:
            print(f"  {subject}: {str(e)}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"  Extracted {len(df)} rows from {df['subject'].nunique()} subjects")
    return df


def extract_extended_rvt() -> pd.DataFrame:
    """Extract extended DMT RVT data (~19 minutes)."""
    print("Extracting extended RVT data...")
    rows: List[Dict] = []
    
    for subject in SUJETOS_VALIDADOS_RESP:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_resp_paths(subject, high_session, low_session)
            d_high = load_resp_csv(p_high)
            d_low = load_resp_csv(p_low)
            
            if d_high is None or d_low is None:
                continue
            
            # Load RS for z-scoring
            p_rsh = build_rs_resp_path(subject, high_session)
            p_rsl = build_rs_resp_path(subject, low_session)
            r_high = load_resp_csv(p_rsh)
            r_low = load_resp_csv(p_rsl)
            
            if r_high is None or r_low is None:
                continue
            
            # Extract data
            th = d_high['time'].to_numpy()
            yh = pd.to_numeric(d_high['RSP_RVT'], errors='coerce').to_numpy()
            tl = d_low['time'].to_numpy()
            yl = pd.to_numeric(d_low['RSP_RVT'], errors='coerce').to_numpy()
            trh = r_high['time'].to_numpy()
            yrh = pd.to_numeric(r_high['RSP_RVT'], errors='coerce').to_numpy()
            trl = r_low['time'].to_numpy()
            yrl = pd.to_numeric(r_low['RSP_RVT'], errors='coerce').to_numpy()
            
            # Z-score using subject baseline
            yrh_z, yh_z, yrl_z, yl_z, success = zscore_with_subject_baseline_rvt(
                trh, yrh, th, yh, trl, yrl, tl, yl
            )
            
            if not success:
                continue
            
            # Compute per-window means
            for window_idx in range(TOTAL_WINDOWS):
                rvt_h = compute_rvt_mean_per_window(th, yh_z, window_idx)
                rvt_l = compute_rvt_mean_per_window(tl, yl_z, window_idx)
                
                if rvt_h is not None and rvt_l is not None:
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'High',
                        'RVT': rvt_h
                    })
                    rows.append({
                        'subject': subject,
                        'window': window_idx + 1,
                        'Dose': 'Low',
                        'RVT': rvt_l
                    })
        
        except Exception as e:
            print(f"  {subject}: {str(e)}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"  Extracted {len(df)} rows from {df['subject'].nunique()} subjects")
    return df


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("EXTRACTING EXTENDED DMT DATA FOR COMPOSITE ANALYSIS")
    print("=" * 80 + "\n")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Extract each modality
    df_hr = extract_extended_hr()
    df_smna = extract_extended_smna()
    df_rvt = extract_extended_rvt()
    
    # Save to CSV
    hr_path = os.path.join(OUT_DIR, 'extended_dmt_hr_z.csv')
    smna_path = os.path.join(OUT_DIR, 'extended_dmt_smna_z.csv')
    rvt_path = os.path.join(OUT_DIR, 'extended_dmt_rvt_z.csv')
    
    df_hr.to_csv(hr_path, index=False)
    df_smna.to_csv(smna_path, index=False)
    df_rvt.to_csv(rvt_path, index=False)
    
    print(f"\n✓ Saved extended DMT data:")
    print(f"  - {hr_path}")
    print(f"  - {smna_path}")
    print(f"  - {rvt_path}")
    print()


if __name__ == '__main__':
    main()
