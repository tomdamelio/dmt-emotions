# -*- coding: utf-8 -*-
"""
Unified RSP_RVT (Respiratory Volume per Time) Analysis: LME modeling and visualization (first 9 minutes).

This script processes respiration-derived respiratory volume per time (RSP_RVT) from NeuroKit RSP_RVT column:
  1) Build long-format per-30-second window RSP_RVT dataset (mean RVT per window, 0–540s)
  2) Fit LME with State × Dose and time effects; apply BH-FDR per family
  3) Create coefficient, marginal means, interaction, diagnostics plots
  4) Write model summary as TXT and figure captions
  5) Generate group-level timecourse plot for the first 9 minutes with FDR

Outputs are written to: results/resp/rvt/

Run:
  python scripts/run_resp_rvt_analysis.py
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_RESP,  
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
except Exception:
    mixedlm = None

try:
    from scipy import stats as scistats
except Exception:
    scistats = None


#############################
# Plot aesthetics & centralized hyperparameters
#############################

# Centralized font sizes and legend settings
AXES_TITLE_SIZE = 29
AXES_LABEL_SIZE = 36
TICK_LABEL_SIZE = 28
TICK_LABEL_SIZE_SMALL = 24

# Legend sizes (global and a smaller variant for dense, per-subject panels)
LEGEND_FONTSIZE = 18
LEGEND_FONTSIZE_SMALL = 14
LEGEND_MARKERSCALE = 1.6
LEGEND_BORDERPAD = 0.6
LEGEND_HANDLELENGTH = 3.0
LEGEND_LABELSPACING = 0.7
LEGEND_BORDERAXESPAD = 0.9

# Stacked per-subject figure specific sizes
STACKED_AXES_LABEL_SIZE = 22
STACKED_TICK_LABEL_SIZE = 14
STACKED_SUBJECT_FONTSIZE = 30

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.titlepad': 8.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': LEGEND_FONTSIZE,
    'legend.borderpad': LEGEND_BORDERPAD,
    'legend.handlelength': LEGEND_HANDLELENGTH,
    'xtick.labelsize': TICK_LABEL_SIZE_SMALL,
    'ytick.labelsize': TICK_LABEL_SIZE_SMALL,
})

# Respiration modality uses green color scheme from tab20c palette
# High and Low are consistent across RS and DMT using first and third gradients of green
# tab20c has 20 colors in 5 groups of 4 gradients each
# Green group: indices 8-11 (darkest to lightest)
tab20c_colors = plt.cm.tab20c.colors
COLOR_RS_HIGH = tab20c_colors[8]   # First green gradient (darkest/most intense) for High
COLOR_RS_LOW = tab20c_colors[10]   # Third green gradient (lighter) for Low
COLOR_DMT_HIGH = tab20c_colors[8]  # Same intense green for High
COLOR_DMT_LOW = tab20c_colors[10]  # Same lighter green for Low

# Analysis window: first 9 minutes (18 windows of 30 seconds each)
N_WINDOWS = 18  # 30-second windows: 0-30s, 30-60s, ..., 510-540s
WINDOW_SIZE_SEC = 30  # 30-second windows
MAX_TIME_SEC = N_WINDOWS * WINDOW_SIZE_SEC  # 540 seconds = 9 minutes

# Z-scoring configuration
USE_ZSCORE = True  # If True: z-score using session/subject baseline; If False: use absolute RVT values
ZSCORE_BY_SUBJECT = True  # If True: z-score using all sessions of subject; If False: z-score each session independently
EXPORT_ABSOLUTE_SCALE = True  # Also export absolute scale for QC (only when USE_ZSCORE=True)

# Optional trims (in seconds) - set to None to disable
RS_TRIM_START = None  # Trim start of RS (e.g., 5.0 for first 5 seconds)
DMT_TRIM_START = None  # Trim start of DMT (e.g., 5.0 for first 5 seconds)

# Minimum samples per window to accept a session
MIN_SAMPLES_PER_WINDOW = 10

# Baseline correction flag (deprecated - use z-scoring instead)
BASELINE_CORRECTION = False


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_resp_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    """Build paths to DMT High and Low RESP CSVs."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv")
    return high_csv, low_csv


def build_rs_resp_path(subject: str, session: str) -> str:
    """Build path to RS RESP CSV."""
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)  # 'Alta' or 'Baja'
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}.csv")


def load_resp_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load RESP CSV and validate required columns."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    
    # Validate required columns
    if 'RSP_RVT' not in df.columns:
        return None
    
    # Reconstruct time if missing
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rvt_default', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    
    return df


def compute_rvt_mean_per_window(t: np.ndarray, rvt: np.ndarray, window_idx: int) -> Optional[float]:
    """Compute mean RSP_RVT for a specific 30-second window.
    
    Parameters:
        t: Time array (seconds)
        rvt: RVT values (can be absolute or z-scored)
        window_idx: Window index (0-based, each window is 30 seconds)
    """
    start_time = window_idx * WINDOW_SIZE_SEC
    end_time = (window_idx + 1) * WINDOW_SIZE_SEC
    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None
    rvt_win = rvt[mask]
    
    # Check if we're dealing with z-scored data (values can be negative)
    has_negative = np.any(rvt_win < 0)
    if has_negative:
        # Z-scored data: just remove NaNs
        rvt_win_valid = rvt_win[~np.isnan(rvt_win)]
    else:
        # Absolute data: apply physiological range filter
        rvt_win_valid = rvt_win[(rvt_win > 0) & (rvt_win < 50000)]  # Physiological range (RVT = Amplitude × Rate)
    
    if len(rvt_win_valid) < 2:
        return None
    return float(np.nanmean(rvt_win_valid))


def zscore_with_session_baseline(t_rs: np.ndarray, y_rs: np.ndarray,
                                 t_dmt: np.ndarray, y_dmt: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Z-score both RS and DMT using the entire session (RS + DMT) as baseline.
    
    This approach normalizes the entire session together, using mu and sigma
    computed from the concatenated RS + DMT data.
    
    Returns:
        (y_rs_z, y_dmt_z, diagnostics_dict)
    """
    diagnostics = {'scalable': False, 'mu': np.nan, 'sigma': np.nan, 'reason': ''}
    
    # Apply optional trims
    if RS_TRIM_START is not None:
        mask_rs = t_rs >= RS_TRIM_START
        t_rs = t_rs[mask_rs]
        y_rs = y_rs[mask_rs]
    
    if DMT_TRIM_START is not None:
        mask_dmt = t_dmt >= DMT_TRIM_START
        t_dmt = t_dmt[mask_dmt]
        y_dmt = y_dmt[mask_dmt]
    
    # Check minimum samples
    if len(y_rs) < MIN_SAMPLES_PER_WINDOW or len(y_dmt) < MIN_SAMPLES_PER_WINDOW:
        diagnostics['reason'] = f'Insufficient samples: RS={len(y_rs)}, DMT={len(y_dmt)}'
        return None, None, diagnostics
    
    # Filter physiological range
    y_rs_valid = y_rs[(y_rs > 0) & (y_rs < 50000)]
    y_dmt_valid = y_dmt[(y_dmt > 0) & (y_dmt < 50000)]
    
    # Concatenate entire session (RS + DMT) to compute mu and sigma
    y_session = np.concatenate([y_rs_valid, y_dmt_valid])
    
    if len(y_session) < MIN_SAMPLES_PER_WINDOW * 2:
        diagnostics['reason'] = f'Session insufficient valid samples: {len(y_session)}'
        return None, None, diagnostics
    
    # Compute mu and sigma from entire session
    mu = np.nanmean(y_session)
    sigma = np.nanstd(y_session, ddof=1)
    
    # Check sigma validity
    if sigma == 0.0 or not np.isfinite(sigma):
        diagnostics['reason'] = f'Invalid sigma: {sigma}'
        return None, None, diagnostics
    
    # Apply z-scoring to both RS and DMT using session parameters
    y_rs_z = (y_rs - mu) / sigma
    y_dmt_z = (y_dmt - mu) / sigma
    
    diagnostics['scalable'] = True
    diagnostics['mu'] = float(mu)
    diagnostics['sigma'] = float(sigma)
    diagnostics['n_session'] = len(y_session)
    diagnostics['n_rs'] = len(y_rs_valid)
    diagnostics['n_dmt'] = len(y_dmt_valid)
    
    return y_rs_z, y_dmt_z, diagnostics


def zscore_with_subject_baseline(t_rs_high: np.ndarray, y_rs_high: np.ndarray,
                                 t_dmt_high: np.ndarray, y_dmt_high: np.ndarray,
                                 t_rs_low: np.ndarray, y_rs_low: np.ndarray,
                                 t_dmt_low: np.ndarray, y_dmt_low: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Z-score all sessions of a subject using combined baseline (all RS + all DMT).
    
    This approach normalizes both sessions together, using mu and sigma
    computed from all data (RS_high + DMT_high + RS_low + DMT_low).
    
    Returns:
        (y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, diagnostics_dict)
    """
    diagnostics = {'scalable': False, 'mu': np.nan, 'sigma': np.nan, 'reason': ''}
    
    # Apply optional trims
    if RS_TRIM_START is not None:
        mask_rs_h = t_rs_high >= RS_TRIM_START
        t_rs_high = t_rs_high[mask_rs_h]
        y_rs_high = y_rs_high[mask_rs_h]
        
        mask_rs_l = t_rs_low >= RS_TRIM_START
        t_rs_low = t_rs_low[mask_rs_l]
        y_rs_low = y_rs_low[mask_rs_l]
    
    if DMT_TRIM_START is not None:
        mask_dmt_h = t_dmt_high >= DMT_TRIM_START
        t_dmt_high = t_dmt_high[mask_dmt_h]
        y_dmt_high = y_dmt_high[mask_dmt_h]
        
        mask_dmt_l = t_dmt_low >= DMT_TRIM_START
        t_dmt_low = t_dmt_low[mask_dmt_l]
        y_dmt_low = y_dmt_low[mask_dmt_l]
    
    # Check minimum samples per session
    if (len(y_rs_high) < MIN_SAMPLES_PER_WINDOW or len(y_dmt_high) < MIN_SAMPLES_PER_WINDOW or
        len(y_rs_low) < MIN_SAMPLES_PER_WINDOW or len(y_dmt_low) < MIN_SAMPLES_PER_WINDOW):
        diagnostics['reason'] = f'Insufficient samples: RS_high={len(y_rs_high)}, DMT_high={len(y_dmt_high)}, RS_low={len(y_rs_low)}, DMT_low={len(y_dmt_low)}'
        return None, None, None, None, diagnostics
    
    # Filter physiological range for all arrays
    y_rs_high_valid = y_rs_high[(y_rs_high > 0) & (y_rs_high < 50000)]
    y_dmt_high_valid = y_dmt_high[(y_dmt_high > 0) & (y_dmt_high < 50000)]
    y_rs_low_valid = y_rs_low[(y_rs_low > 0) & (y_rs_low < 50000)]
    y_dmt_low_valid = y_dmt_low[(y_dmt_low > 0) & (y_dmt_low < 50000)]
    
    # Concatenate ALL subject data (both sessions) to compute mu and sigma
    y_all = np.concatenate([y_rs_high_valid, y_dmt_high_valid, y_rs_low_valid, y_dmt_low_valid])
    
    if len(y_all) < MIN_SAMPLES_PER_WINDOW * 4:
        diagnostics['reason'] = f'Subject insufficient valid samples: {len(y_all)}'
        return None, None, None, None, diagnostics
    
    # Compute mu and sigma from ALL subject data
    mu = np.nanmean(y_all)
    sigma = np.nanstd(y_all, ddof=1)
    
    # Check sigma validity
    if sigma == 0.0 or not np.isfinite(sigma):
        diagnostics['reason'] = f'Invalid sigma: {sigma}'
        return None, None, None, None, diagnostics
    
    # Apply z-scoring to all sessions using subject-level parameters
    y_rs_high_z = (y_rs_high - mu) / sigma
    y_dmt_high_z = (y_dmt_high - mu) / sigma
    y_rs_low_z = (y_rs_low - mu) / sigma
    y_dmt_low_z = (y_dmt_low - mu) / sigma
    
    diagnostics['scalable'] = True
    diagnostics['mu'] = float(mu)
    diagnostics['sigma'] = float(sigma)
    diagnostics['n_all'] = len(y_all)
    diagnostics['n_rs_high'] = len(y_rs_high_valid)
    diagnostics['n_dmt_high'] = len(y_dmt_high_valid)
    diagnostics['n_rs_low'] = len(y_rs_low_valid)
    diagnostics['n_dmt_low'] = len(y_dmt_low_valid)
    
    return y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, diagnostics


def prepare_long_data_resp_rvt() -> pd.DataFrame:
    """Build long-format per-30-second window RSP_RVT table (first 9 minutes = 18 windows).
    
    If USE_ZSCORE=True: z-scores using session or subject baseline.
    If ZSCORE_BY_SUBJECT=True: uses all sessions of subject for normalization.
    If ZSCORE_BY_SUBJECT=False: uses each session independently for normalization.
    If USE_ZSCORE=False: uses absolute RVT values.
    """
    rows: List[Dict] = []
    qc_log: List[str] = []
    
    for subject in SUJETOS_VALIDADOS_RESP:
        high_session, low_session = determine_sessions(subject)
        
        # Load all data for this subject
        dmt_high_path, dmt_low_path = build_resp_paths(subject, high_session, low_session)
        rs_high_path = build_rs_resp_path(subject, high_session)
        rs_low_path = build_rs_resp_path(subject, low_session)

        dmt_high = load_resp_csv(dmt_high_path)
        dmt_low = load_resp_csv(dmt_low_path)
        rs_high = load_resp_csv(rs_high_path)
        rs_low = load_resp_csv(rs_low_path)
        
        # Check if all data is available
        if any(x is None for x in (dmt_high, dmt_low, rs_high, rs_low)):
            qc_log.append(f"{subject}: Missing data files")
            continue

        # Extract time and RSP_RVT
        t_dmt_high = dmt_high['time'].to_numpy()
        rvt_dmt_high_abs = pd.to_numeric(dmt_high['RSP_RVT'], errors='coerce').to_numpy()
        t_dmt_low = dmt_low['time'].to_numpy()
        rvt_dmt_low_abs = pd.to_numeric(dmt_low['RSP_RVT'], errors='coerce').to_numpy()
        t_rs_high = rs_high['time'].to_numpy()
        rvt_rs_high_abs = pd.to_numeric(rs_high['RSP_RVT'], errors='coerce').to_numpy()
        t_rs_low = rs_low['time'].to_numpy()
        rvt_rs_low_abs = pd.to_numeric(rs_low['RSP_RVT'], errors='coerce').to_numpy()

        if USE_ZSCORE:
            if ZSCORE_BY_SUBJECT:
                # Z-score using ALL sessions of subject as baseline
                rvt_rs_high_z, rvt_dmt_high_z, rvt_rs_low_z, rvt_dmt_low_z, diag = zscore_with_subject_baseline(
                    t_rs_high, rvt_rs_high_abs, t_dmt_high, rvt_dmt_high_abs,
                    t_rs_low, rvt_rs_low_abs, t_dmt_low, rvt_dmt_low_abs
                )
                
                if not diag['scalable']:
                    qc_log.append(f"{subject}: Not scalable (subject-level): {diag['reason']}")
                    continue
                
                # Process each 30-second window with z-scored data for BOTH sessions
                for window_idx in range(N_WINDOWS):
                    window_label = window_idx + 1
                    
                    # Compute window means for z-scored data
                    rvt_dmt_h_z = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_z, window_idx)
                    rvt_dmt_l_z = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_z, window_idx)
                    rvt_rs_h_z = compute_rvt_mean_per_window(t_rs_high, rvt_rs_high_z, window_idx)
                    rvt_rs_l_z = compute_rvt_mean_per_window(t_rs_low, rvt_rs_low_z, window_idx)
                    
                    if None not in (rvt_dmt_h_z, rvt_dmt_l_z, rvt_rs_h_z, rvt_rs_l_z):
                        rows.extend([
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_z, 'Scale': 'z'},
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'RSP_RVT': rvt_rs_h_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'RSP_RVT': rvt_rs_l_z, 'Scale': 'z'},
                        ])
                        
                        # Optional: absolute scale for QC
                        if EXPORT_ABSOLUTE_SCALE:
                            rvt_dmt_h_abs = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_abs, window_idx)
                            rvt_dmt_l_abs = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_abs, window_idx)
                            rvt_rs_h_abs = compute_rvt_mean_per_window(t_rs_high, rvt_rs_high_abs, window_idx)
                            rvt_rs_l_abs = compute_rvt_mean_per_window(t_rs_low, rvt_rs_low_abs, window_idx)
                            
                            if None not in (rvt_dmt_h_abs, rvt_dmt_l_abs, rvt_rs_h_abs, rvt_rs_l_abs):
                                rows.extend([
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'RSP_RVT': rvt_rs_h_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'RSP_RVT': rvt_rs_l_abs, 'Scale': 'abs'},
                                ])
            
            else:
                # Z-score each session independently (session-level normalization)
                # Process HIGH session
                rvt_rs_high_z, rvt_dmt_high_z, diag_high = zscore_with_session_baseline(
                    t_rs_high, rvt_rs_high_abs, t_dmt_high, rvt_dmt_high_abs
                )
                
                # Process LOW session
                rvt_rs_low_z, rvt_dmt_low_z, diag_low = zscore_with_session_baseline(
                    t_rs_low, rvt_rs_low_abs, t_dmt_low, rvt_dmt_low_abs
                )
                
                if not (diag_high['scalable'] and diag_low['scalable']):
                    if not diag_high['scalable']:
                        qc_log.append(f"{subject} HIGH session not scalable: {diag_high['reason']}")
                    if not diag_low['scalable']:
                        qc_log.append(f"{subject} LOW session not scalable: {diag_low['reason']}")
                    continue
                
                # Process each 30-second window with z-scored data
                for window_idx in range(N_WINDOWS):
                    window_label = window_idx + 1
                    
                    rvt_dmt_h_z = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_z, window_idx)
                    rvt_dmt_l_z = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_z, window_idx)
                    rvt_rs_h_z = compute_rvt_mean_per_window(t_rs_high, rvt_rs_high_z, window_idx)
                    rvt_rs_l_z = compute_rvt_mean_per_window(t_rs_low, rvt_rs_low_z, window_idx)
                    
                    if None not in (rvt_dmt_h_z, rvt_dmt_l_z, rvt_rs_h_z, rvt_rs_l_z):
                        rows.extend([
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_z, 'Scale': 'z'},
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'RSP_RVT': rvt_rs_h_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'RSP_RVT': rvt_rs_l_z, 'Scale': 'z'},
                        ])
                        
                        # Optional: absolute scale for QC
                        if EXPORT_ABSOLUTE_SCALE:
                            rvt_dmt_h_abs = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_abs, window_idx)
                            rvt_dmt_l_abs = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_abs, window_idx)
                            rvt_rs_h_abs = compute_rvt_mean_per_window(t_rs_high, rvt_rs_high_abs, window_idx)
                            rvt_rs_l_abs = compute_rvt_mean_per_window(t_rs_low, rvt_rs_low_abs, window_idx)
                            
                            if None not in (rvt_dmt_h_abs, rvt_dmt_l_abs, rvt_rs_h_abs, rvt_rs_l_abs):
                                rows.extend([
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'RSP_RVT': rvt_rs_h_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'RSP_RVT': rvt_rs_l_abs, 'Scale': 'abs'},
                                ])
        
        else:
            # Use absolute values (no z-scoring)
            for window_idx in range(N_WINDOWS):
                window_label = window_idx + 1
                
                rvt_dmt_h = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_abs, window_idx)
                rvt_dmt_l = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_abs, window_idx)
                rvt_rs_h = compute_rvt_mean_per_window(t_rs_high, rvt_rs_high_abs, window_idx)
                rvt_rs_l = compute_rvt_mean_per_window(t_rs_low, rvt_rs_low_abs, window_idx)
                
                if None not in (rvt_dmt_h, rvt_dmt_l, rvt_rs_h, rvt_rs_l):
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l, 'Scale': 'abs'},
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'RSP_RVT': rvt_rs_h, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'RSP_RVT': rvt_rs_l, 'Scale': 'abs'},
                    ])

    if not rows:
        raise ValueError('No valid RSP_RVT data found for any subject!')
    
    # Save QC log
    if USE_ZSCORE and qc_log:
        print(f"Warning: {len(qc_log)} sessions/subjects excluded from z-scoring:")
        for msg in qc_log:
            print(f"  {msg}")

    df = pd.DataFrame(rows)
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['Scale'] = pd.Categorical(df['Scale'], categories=['z', 'abs'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['window_c'] = df['window'] - df['window'].mean()
    return df


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    if mixedlm is None:
        return None, {'error': 'statsmodels not available'}
    # Filter to appropriate scale (z-scored if USE_ZSCORE=True, else absolute)
    scale_to_use = 'z' if USE_ZSCORE else 'abs'
    df_model = df[df['Scale'] == scale_to_use].copy()
    if len(df_model) == 0:
        return None, {'error': f'No {scale_to_use}-scaled data available'}
    try:
        formula = 'RSP_RVT ~ State * Dose + window_c + State:window_c + Dose:window_c'
        model = mixedlm(formula, df_model, groups=df_model['subject'])  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fitted = model.fit()
            convergence_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        return None, {'error': str(e)}
    diagnostics = {
        'aic': getattr(fitted, 'aic', np.nan),
        'bic': getattr(fitted, 'bic', np.nan),
        'loglik': getattr(fitted, 'llf', np.nan),
        'n_obs': getattr(fitted, 'nobs', len(df_model)),
        'n_groups': len(df_model['subject'].unique()),
        'convergence_warnings': convergence_warnings,
        'random_effects_var': getattr(fitted, 'cov_re', None),
        'residual_var': getattr(fitted, 'scale', np.nan),
    }
    return fitted, diagnostics


def plot_model_diagnostics(fitted_model, df: pd.DataFrame, output_dir: str) -> None:
    if fitted_model is None:
        return
    fitted_vals = fitted_model.fittedvalues
    residuals = fitted_model.resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    try:
        if scistats is not None:
            scistats.probplot(residuals, dist='norm', plot=axes[0, 1])
    except Exception:
        pass
    subject_means = df.groupby('subject', observed=False).apply(lambda x: residuals[x.index].mean(), include_groups=False)
    axes[1, 0].bar(range(len(subject_means)), subject_means.values, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('Mean Residual')
    window_residuals = df.groupby('window', observed=False).apply(lambda x: residuals[x.index].mean(), include_groups=False)
    axes[1, 1].plot(window_residuals.index, window_residuals.values, 'o-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Window (30s)')
    axes[1, 1].set_ylabel('Mean Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lme_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    order = np.argsort(p_array)
    sorted_p = p_array[order]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[order[i]] = sorted_p[i]
        else:
            adjusted[order[i]] = min(sorted_p[i] * n / (i + 1), adjusted[order[i + 1]])
    return np.minimum(adjusted, 1.0).tolist()


def hypothesis_testing_with_fdr(fitted_model) -> Dict:
    if fitted_model is None:
        return {}
    params = fitted_model.params
    pvalues = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    stderr = fitted_model.bse
    results: Dict[str, Dict] = {
        'all_params': params.to_dict(),
        'all_pvalues': pvalues.to_dict(),
        'all_stderr': stderr.to_dict(),
        'conf_int': conf_int.to_dict(),
    }
    families: Dict[str, List[str]] = {'State': [], 'Dose': [], 'Interaction': []}
    for p in ['State[T.DMT]', 'State[T.DMT]:window_c']:
        if p in pvalues.index:
            families['State'].append(p)
    for p in ['Dose[T.High]', 'Dose[T.High]:window_c']:
        if p in pvalues.index:
            families['Dose'].append(p)
    for p in ['State[T.DMT]:Dose[T.High]']:
        if p in pvalues.index:
            families['Interaction'].append(p)
    fdr_results: Dict[str, Dict] = {}
    for fam, plist in families.items():
        if not plist:
            continue
        raw = [pvalues[p] for p in plist]
        adj = benjamini_hochberg_correction(raw)
        fam_dict: Dict[str, Dict] = {}
        for i, p in enumerate(plist):
            fam_dict[p] = {
                'beta': float(params[p]),
                'se': float(stderr[p]),
                'p_raw': float(pvalues[p]),
                'p_fdr': float(adj[i]),
                'ci_lower': float(conf_int.loc[p, 0]),
                'ci_upper': float(conf_int.loc[p, 1]),
            }
        fdr_results[fam] = fam_dict
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(stderr['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'description': 'High - Low within RS',
        }
    if all(k in params.index for k in ['Dose[T.High]', 'State[T.DMT]:Dose[T.High]']):
        # Efecto total de High vs Low dentro de DMT
        beta_dmt = float(params['Dose[T.High]']) + float(params['State[T.DMT]:Dose[T.High]'])
        # SE aproximado usando propagación de errores (asumiendo covarianza ≈ 0)
        se_dmt = np.sqrt(float(stderr['Dose[T.High]'])**2 + float(stderr['State[T.DMT]:Dose[T.High]'])**2)
        contrasts['High_Low_within_DMT'] = {
            'beta': beta_dmt,
            'se': se_dmt,
            'p_raw': np.nan,  # Requeriría test de Wald explícito
            'description': 'High - Low within DMT (simple effect)',
        }
        contrasts['Interaction_DMT_vs_RS'] = {
            'beta': float(params['State[T.DMT]:Dose[T.High]']),
            'se': float(stderr['State[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['State[T.DMT]:Dose[T.High]']),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }
    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    return results


def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, df: pd.DataFrame, output_dir: str) -> str:
    df_z = df[df['Scale'] == 'z']
    zscore_mode = "subject-level (all sessions)" if ZSCORE_BY_SUBJECT else "session-level (independent)"
    lines: List[str] = [
        '=' * 80,
        'LME ANALYSIS REPORT: RSP_RVT (Respiratory Volume per Time) by 30-second Windows (first 9 minutes)',
        '=' * 80,
        '',
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df_z)} observations from {len(df_z['subject'].unique())} subjects",
        '',
        'DESIGN:',
        '  Within-subjects 2×2: State (RS vs DMT) × Dose (Low vs High)',
        '  Time windows: 18 thirty-second windows (0-540 seconds = 9 minutes)',
        '  Dependent variable: Mean RSP_RVT per 30-second window',
        f'  Z-scoring: {USE_ZSCORE}',
        f'  Z-scoring mode: {zscore_mode if USE_ZSCORE else "N/A"}',
        f'  {"Subject parameters (mu, sigma) computed from all sessions (RS_high + DMT_high + RS_low + DMT_low)" if USE_ZSCORE and ZSCORE_BY_SUBJECT else "Session parameters (mu, sigma) computed per session (RS + DMT)" if USE_ZSCORE else "Absolute RVT values (a.u./min)"}',
        '',
        'MODEL SPECIFICATION:',
        '  Fixed effects: RSP_RVT ~ State*Dose + window_c + State:window_c + Dose:window_c',
        '  Random effects: ~ 1 | subject',
        '  Where window_c = window - mean(window) [centered time]',
        f'  Scale: {"z-scored" if USE_ZSCORE else "absolute"}',
        '',
    ]
    if fitted_model is not None:
        lines.extend([
            'MODEL FIT STATISTICS:',
            f"  AIC: {diagnostics.get('aic', np.nan):.2f}",
            f"  BIC: {diagnostics.get('bic', np.nan):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', np.nan):.6f}",
            '',
        ])
        warns = diagnostics.get('convergence_warnings', [])
        if warns:
            lines.extend(['CONVERGENCE WARNINGS:', *[f'  - {w}' for w in warns], ''])
        else:
            lines.append('Model converged without warnings\n')
    if 'fdr_families' in hypothesis_results:
        lines.extend(['HYPOTHESIS TESTING RESULTS (with BH-FDR correction):', '=' * 60, ''])
        for fam, famres in hypothesis_results['fdr_families'].items():
            lines.extend([f'FAMILY {fam.upper()}:', '-' * 30])
            for param, res in famres.items():
                sig = '***' if res['p_fdr'] < 0.001 else '**' if res['p_fdr'] < 0.01 else '*' if res['p_fdr'] < 0.05 else ''
                lines.extend([
                    f'  {param}:',
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {sig}",
                    '',
                ])
            lines.append('')
    if 'conditional_contrasts' in hypothesis_results:
        lines.extend(['CONDITIONAL CONTRASTS:', '-' * 30])
        for _, res in hypothesis_results['conditional_contrasts'].items():
            sig = '***' if res['p_raw'] < 0.001 else '**' if res['p_raw'] < 0.01 else '*' if res['p_raw'] < 0.05 else ''
            lines.extend([f"  {res['description']}:", f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {sig}", ''])
    lines.extend(['', 'DATA SUMMARY (Z-SCORED):', '-' * 30])
    cell = df_z.groupby(['State', 'Dose'], observed=False)['RSP_RVT'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Cell means (RSP_RVT by State × Dose):', str(cell), ''])
    trend = df_z.groupby('window', observed=False)['RSP_RVT'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Time trend (RSP_RVT by 30-second window):', str(trend), ''])
    
    # QC check: RS by dose
    lines.extend(['', 'QC CHECK: RS by Dose:', '-' * 30])
    rs_only = df_z[df_z['State'] == 'RS']
    rs_by_dose = rs_only.groupby('Dose', observed=False)['RSP_RVT'].agg(['count', 'mean', 'std']).round(4)
    lines.extend([str(rs_by_dose), ''])
    if USE_ZSCORE:
        if ZSCORE_BY_SUBJECT:
            lines.extend(['Note: Z-scoring uses ALL sessions of subject (RS_high + DMT_high + RS_low + DMT_low),',
                          'so RS values reflect subject-level normalization. High and Low RS should be similar',
                          'since they share the same normalization parameters.', ''])
        else:
            lines.extend(['Note: Z-scoring uses entire session (RS+DMT) independently per session,', 
                          'so RS values reflect session-level normalization. High and Low RS may differ',
                          'since each session has its own normalization parameters.', ''])
    lines.extend(['', '=' * 80])
    out_path = os.path.join(output_dir, 'lme_analysis_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def load_lme_results_from_report(report_path: str) -> Dict:
    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    coefficients: Dict[str, Dict] = {}
    current_family: Optional[str] = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('FAMILY STATE:'):
            current_family = 'State'
        elif line.startswith('FAMILY DOSE:'):
            current_family = 'Dose'
        elif line.startswith('FAMILY INTERACTION:'):
            current_family = 'Interaction'
        elif line.startswith('CONDITIONAL CONTRASTS:'):
            current_family = 'Contrasts'
        if current_family and line.endswith(':') and not line.startswith('FAMILY') and not line.startswith('CONDITIONAL'):
            name = line.rstrip(':').strip()
            if i + 3 < len(lines):
                beta_line = lines[i + 1].strip()
                ci_line = lines[i + 2].strip()
                p_line = lines[i + 3].strip()
                try:
                    if beta_line.startswith('β ='):
                        parts = beta_line.split(',')
                        beta = float(parts[0].split('=')[1].strip())
                        se = float(parts[1].split('=')[1].strip())
                    if ci_line.startswith('95% CI:'):
                        txt = ci_line.replace('95% CI:', '').replace('[', '').replace(']', '').strip()
                        ci_lower = float(txt.split(',')[0])
                        ci_upper = float(txt.split(',')[1])
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw = float([p for p in p_parts if 'p_raw' in p][0].split('=')[1].strip().split()[0])
                        p_fdr_text = [p for p in p_parts if 'p_FDR' in p][0].split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        significance = '***' if '***' in p_fdr_text else '**' if '**' in p_fdr_text else '*' if '*' in p_fdr_text else ''
                    coefficients[name] = {
                        'family': current_family,
                        'beta': beta,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_raw': p_raw,
                        'p_fdr': p_fdr,
                        'significance': significance,
                    }
                except Exception:
                    continue
    return coefficients


def prepare_coefficient_data(coefficients: Dict) -> pd.DataFrame:
    order = [
        'State[T.DMT]',
        'Dose[T.High]',
        'State[T.DMT]:window_c',
        'Dose[T.High]:window_c',
        'State[T.DMT]:Dose[T.High]'
    ]
    labels = {
        'State[T.DMT]': 'State (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'State[T.DMT]:window_c': 'State × Time',
        'Dose[T.High]:window_c': 'Dose × Time',
        'State[T.DMT]:Dose[T.High]': 'State × Dose'
    }
    # Use Respiration modality color (green tones from tab20c) with distinct shades for visual distinction
    # Green group from tab20c: indices 8-11 (darkest to lightest)
    fam_colors = {
        'State': tab20c_colors[8],      # First green gradient (darkest/most intense)
        'Dose': tab20c_colors[9],      # Second green gradient (medium)
        'Interaction': tab20c_colors[10],  # Third green gradient (lighter)
    }
    rows: List[Dict] = []
    for i, p in enumerate(order):
        if p in coefficients:
            c = coefficients[p]
            rows.append({
                'parameter': p,
                'label': labels.get(p, p),
                'beta': c['beta'],
                'se': c['se'],
                'ci_lower': c['ci_lower'],
                'ci_upper': c['ci_upper'],
                'p_raw': c['p_raw'],
                'p_fdr': c['p_fdr'],
                'significance': c['significance'],
                'family': c['family'],
                'order': i,
                'significant': c['p_fdr'] < 0.05,
                'color': fam_colors.get(c['family'], '#666666'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No coefficient data to plot')
    return df


def create_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    coef_df = coef_df.sort_values('order')
    y_positions = np.arange(len(coef_df))
    for idx, row in coef_df.iterrows():
        y_pos = y_positions[coef_df.index.get_loc(idx)]
        # Tamaño uniforme para todos los elementos
        linewidth = 6.5
        alpha = 1.0
        marker_size = 200
        # Línea del CI muy gruesa
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], color=row['color'], linewidth=linewidth, alpha=alpha)
        # Círculo del coeficiente grande con borde del mismo color
        ax.scatter(row['beta'], y_pos, color=row['color'], s=marker_size, alpha=alpha, edgecolors=row['color'], linewidths=3.5, zorder=3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_df['label'], fontsize=33)
    ax.set_xlabel('Coefficient Estimate (β)\nwith 95% CI')
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    # Add significance asterisks based on FDR-corrected p-values
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for idx, row in coef_df.iterrows():
        y_pos = y_positions[coef_df.index.get_loc(idx)]
        p_fdr = row.get('p_fdr', 1.0)
        if p_fdr < 0.001:
            sig_marker = '***'
        elif p_fdr < 0.01:
            sig_marker = '**'
        elif p_fdr < 0.05:
            sig_marker = '*'
        else:
            sig_marker = ''
        if sig_marker:
            # Position asterisks to the right of the CI
            x_pos = row['ci_upper'] + x_range * 0.02
            ax.text(x_pos, y_pos, sig_marker, fontsize=40, fontweight='bold',
                   va='center', ha='left', color=row['color'])
    plt.subplots_adjust(left=0.28, right=0.92)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    # Use appropriate scale based on USE_ZSCORE
    scale_to_use = 'z' if USE_ZSCORE else 'abs'
    df_scale = df[df['Scale'] == scale_to_use]
    grouped = df_scale.groupby(['window', 'State', 'Dose'], observed=False)['RSP_RVT']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['window', 'State', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['State'].astype(str) + '_' + stats_df['Dose'].astype(str)
    alpha = 1 - confidence
    t_critical = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1) if scistats is not None else 1.96
    stats_df['ci_lower'] = stats_df['mean'] - t_critical * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_critical * stats_df['se']
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    return stats_df


def create_marginal_means_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    conditions = stats_df['condition'].unique()
    for condition in sorted(conditions):
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        if len(cond_data) == 0:
            continue
        if condition == 'RS_Low':
            color = COLOR_RS_LOW
        elif condition == 'RS_High':
            color = COLOR_RS_HIGH
        elif condition == 'DMT_Low':
            color = COLOR_DMT_LOW
        elif condition == 'DMT_High':
            color = COLOR_DMT_HIGH
        else:
            color = '#666666'
        ax.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('_', ' '), marker='o', markersize=5)
        ax.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('RSP_RVT (Z-scored)')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_state_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    state_means = stats_df.groupby(['minute', 'State'], observed=False).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    state_se = stats_df.groupby(['minute', 'State'], observed=False)['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    state_means = state_means.merge(state_se, on=['minute', 'State'], how='left')
    t_crit = 1.96
    state_means['ci_lower'] = state_means['mean'] - t_crit * state_means['se']
    state_means['ci_upper'] = state_means['mean'] + t_crit * state_means['se']
    fig, ax = plt.subplots(figsize=(10, 6))
    for state, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        state_data = state_means[state_means['State'] == state].sort_values('minute')
        ax.plot(state_data['minute'], state_data['mean'], color=color, linewidth=3, label=f'{state}', marker='o', markersize=6)
        ax.fill_between(state_data['minute'], state_data['ci_lower'], state_data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('RSP_RVT (Z-scored)')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('RS_', ''), marker='o', markersize=4)
        ax1.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('RSP_RVT (Z-scored)')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    ticks = list(range(1, N_MINUTES + 1))
    ax1.set_xticks(ticks)
    ax1.set_xlim(0.8, N_MINUTES + 0.2)
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('DMT_', ''), marker='o', markersize=4)
        ax2.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    ax2.set_xticks(ticks)
    ax2.set_xlim(0.8, N_MINUTES + 0.2)
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    table = coef_df[['label', 'beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'significance', 'family']].copy()
    numeric = ['beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr']
    table[numeric] = table[numeric].round(4)
    table['interpretation'] = table.apply(lambda r: (
        f"{'Significant' if r['p_fdr'] < 0.05 else 'Non-significant'} "
        f"{'increase' if r['beta'] > 0 else 'decrease'} in RSP_RVT"), axis=1)
    table.to_csv(output_path, index=False)


def create_model_summary_txt(diagnostics: Dict, coef_df: pd.DataFrame, output_path: str) -> None:
    lines: List[str] = [
        'LME MODEL SUMMARY',
        '=' * 60,
        '',
        'Fixed Effects Formula:',
        'RSP_RVT ~ State*Dose + minute_c + State:minute_c + Dose:minute_c',
        '',
        'Random Effects: ~ 1 | subject',
        '',
        'Model Fit Statistics:',
        f"AIC: {diagnostics.get('aic', np.nan):.2f}",
        f"BIC: {diagnostics.get('bic', np.nan):.2f}",
        f"Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
        f"N observations: {diagnostics.get('n_obs', 'N/A')}",
        f"N subjects: {diagnostics.get('n_groups', 'N/A')}",
        '',
        'Significant Fixed Effects (p_FDR < 0.05):',
    ]
    sig = coef_df[coef_df['p_fdr'] < 0.05]
    if len(sig) == 0:
        lines.append('• None')
    else:
        for _, row in sig.iterrows():
            lines.append(f"• {row['label']}: β = {row['beta']:.3f} {row['significance']}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(t_grid, np.nan, dtype=float)
    t_valid = t[valid]
    y_valid = y[valid]
    yg = np.full_like(t_grid, np.nan, dtype=float)
    mask = (t_grid >= t_valid[0]) & (t_grid <= t_valid[-1])
    if np.any(mask):
        yg[mask] = np.interp(t_grid[mask], t_valid, y_valid)
    return yg


def _compute_fdr_significant_segments(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05, alternative: str = 'two-sided') -> List[Tuple[float, float]]:
    """Return contiguous x-intervals where High vs Low differ after BH-FDR."""
    if scistats is None:
        return []
    n_time = A.shape[1]
    pvals = np.full(n_time, np.nan, dtype=float)
    for t in range(n_time):
        a = A[:, t]
        b = B[:, t]
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) >= 2:
            try:
                _, p = scistats.ttest_rel(a[mask], b[mask], alternative=alternative)
                pvals[t] = float(p)
            except Exception:
                pvals[t] = np.nan
    valid = ~np.isnan(pvals)
    if not np.any(valid):
        return []
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid].tolist())
    adj[valid] = np.array(adj_vals)
    sig = adj < alpha
    segs: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segs.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return segs


def _compute_fdr_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05, alternative: str = 'two-sided') -> Dict:
    """Compute paired t-test across time, apply BH-FDR, and summarize results."""
    result: Dict[str, object] = {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    if scistats is None:
        print("Warning: scipy.stats not available for FDR computation")
        return result
    
    n_time = A.shape[1]
    pvals = np.full(n_time, np.nan, dtype=float)
    
    # Compute t-tests at each time point
    for t in range(n_time):
        a = A[:, t]
        b = B[:, t]
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) >= 2:
            try:
                _, p = scistats.ttest_rel(a[mask], b[mask], alternative=alternative)
                pvals[t] = float(p)
            except Exception:
                pvals[t] = np.nan
    
    valid_idx = np.where(~np.isnan(pvals))[0]
    if len(valid_idx) == 0:
        print("Warning: No valid p-values computed for FDR")
        return result
    
    # Apply BH-FDR correction
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid_idx].tolist())
    adj[valid_idx] = np.array(adj_vals, dtype=float)
    
    # Find significant time points
    sig = adj < alpha
    n_sig = np.sum(sig)
    print(f"FDR analysis: {n_sig}/{len(sig)} time points significant (alpha={alpha}, alternative={alternative})")
    
    # Find contiguous segments of significance
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segments.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    
    print(f"Found {len(segments)} significant segments")
    if segments:
        for j, (x0, x1) in enumerate(segments):
            print(f"  Segment {j+1}: {x0:.1f}s - {x1:.1f}s ({x0/60:.2f} - {x1/60:.2f} min)")
    
    result['pvals'] = pvals.tolist()
    result['pvals_adj'] = adj.tolist()
    result['sig_mask'] = sig.tolist()
    result['segments'] = segments
    return result


def create_combined_summary_plot(out_dir: str) -> Optional[str]:
    """Create RS+DMT summary using per-30-second-window mean RVT (first 9 minutes = 18 windows).
    
    Uses z-scored data if USE_ZSCORE=True, else absolute RVT values.
    If ZSCORE_BY_SUBJECT=True: uses subject-level normalization.
    If ZSCORE_BY_SUBJECT=False: uses session-level normalization.
    
    Saves: results/resp/rvt/plots/all_subs_resp_rvt.png
    """
    # Build per-subject per-window mean RVT matrices for RS and DMT (High/Low)
    H_RS, L_RS, H_DMT, L_DMT = [], [], [], []
    for subject in SUJETOS_VALIDADOS_RESP:
        try:
            high_session, low_session = determine_sessions(subject)
            # DMT paths
            p_high, p_low = build_resp_paths(subject, high_session, low_session)
            d_high = load_resp_csv(p_high)
            d_low = load_resp_csv(p_low)
            # RS paths
            p_rsh = build_rs_resp_path(subject, high_session)
            p_rsl = build_rs_resp_path(subject, low_session)
            r_high = load_resp_csv(p_rsh)
            r_low = load_resp_csv(p_rsl)
            if any(x is None for x in (d_high, d_low, r_high, r_low)):
                continue
            
            th_abs = d_high['time'].to_numpy()
            yh_abs = pd.to_numeric(d_high['RSP_RVT'], errors='coerce').to_numpy()
            tl_abs = d_low['time'].to_numpy()
            yl_abs = pd.to_numeric(d_low['RSP_RVT'], errors='coerce').to_numpy()
            trh_abs = r_high['time'].to_numpy()
            yrh_abs = pd.to_numeric(r_high['RSP_RVT'], errors='coerce').to_numpy()
            trl_abs = r_low['time'].to_numpy()
            yrl_abs = pd.to_numeric(r_low['RSP_RVT'], errors='coerce').to_numpy()
            
            # Apply z-scoring if enabled
            if USE_ZSCORE:
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    yrh_z, yh_z, yrl_z, yl_z, diag = zscore_with_subject_baseline(
                        trh_abs, yrh_abs, th_abs, yh_abs,
                        trl_abs, yrl_abs, tl_abs, yl_abs
                    )
                    if not diag['scalable']:
                        continue
                    yh, yl, yrh, yrl = yh_z, yl_z, yrh_z, yrl_z
                else:
                    # Z-score each session independently
                    yrh_z, yh_z, diag_h = zscore_with_session_baseline(trh_abs, yrh_abs, th_abs, yh_abs)
                    yrl_z, yl_z, diag_l = zscore_with_session_baseline(trl_abs, yrl_abs, tl_abs, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        continue
                    yh, yl, yrh, yrl = yh_z, yl_z, yrh_z, yrl_z
            else:
                # Use absolute values
                yh, yl, yrh, yrl = yh_abs, yl_abs, yrh_abs, yrl_abs
            
            # Compute per-30-second-window mean RVT 1..18
            rvt_dmt_h = [compute_rvt_mean_per_window(th_abs, yh, m) for m in range(N_WINDOWS)]
            rvt_dmt_l = [compute_rvt_mean_per_window(tl_abs, yl, m) for m in range(N_WINDOWS)]
            rvt_rs_h = [compute_rvt_mean_per_window(trh_abs, yrh, m) for m in range(N_WINDOWS)]
            rvt_rs_l = [compute_rvt_mean_per_window(trl_abs, yrl, m) for m in range(N_WINDOWS)]
            if None in rvt_dmt_h or None in rvt_dmt_l or None in rvt_rs_h or None in rvt_rs_l:
                continue
            H_RS.append(np.array(rvt_rs_h, dtype=float))
            L_RS.append(np.array(rvt_rs_l, dtype=float))
            H_DMT.append(np.array(rvt_dmt_h, dtype=float))
            L_DMT.append(np.array(rvt_dmt_l, dtype=float))
        except Exception:
            continue
    
    if not (H_RS and L_RS and H_DMT and L_DMT):
        return None
    
    H_RS = np.vstack(H_RS); L_RS = np.vstack(L_RS)
    H_DMT = np.vstack(H_DMT); L_DMT = np.vstack(L_DMT)

    def mean_sem(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])

    rs_mean_h, rs_sem_h = mean_sem(H_RS)
    rs_mean_l, rs_sem_l = mean_sem(L_RS)
    dmt_mean_h, dmt_sem_h = mean_sem(H_DMT)
    dmt_mean_l, dmt_sem_l = mean_sem(L_DMT)

    x = np.arange(1, N_WINDOWS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    c_rs_high, c_rs_low = COLOR_RS_HIGH, COLOR_RS_LOW
    
    # RS panel
    rs_segs = _compute_fdr_significant_segments(H_RS, L_RS, x)
    for w0, w1 in rs_segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax1.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    # Convert window indices to time in minutes for x-axis
    time_minutes = (x - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
    l1 = ax1.plot(time_minutes, rs_mean_h, color=c_rs_high, lw=2.0, label='High dose (40mg)')[0]
    ax1.fill_between(time_minutes, rs_mean_h - rs_sem_h, rs_mean_h + rs_sem_h, color=c_rs_high, alpha=0.25)
    l2 = ax1.plot(time_minutes, rs_mean_l, color=c_rs_low, lw=2.0, label='Low dose (20mg)')[0]
    ax1.fill_between(time_minutes, rs_mean_l - rs_sem_l, rs_mean_l + rs_sem_l, color=c_rs_low, alpha=0.25)
    leg1 = ax1.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg1.get_frame().set_facecolor('white'); leg1.get_frame().set_alpha(0.9)
    ax1.set_xlabel('Time (minutes)')
    # Use green color from tab20c for Respiration (RESP/RVT modality) - only first line colored
    ax1.text(-0.20, 0.5, 'Respiration', transform=ax1.transAxes, 
             fontsize=28, fontweight='bold', color=tab20c_colors[8],
             rotation=90, va='center', ha='center')
    ax1.text(-0.12, 0.5, 'RVT (Z-scored)', transform=ax1.transAxes, 
             fontsize=28, fontweight='normal', color='black', 
             rotation=90, va='center', ha='center')
    ax1.set_title('Resting State (RS)', fontweight='bold')
    ax1.grid(True, which='major', axis='y', alpha=0.25); ax1.grid(False, which='major', axis='x')
    
    # DMT panel
    dmt_segs = _compute_fdr_significant_segments(H_DMT, L_DMT, x, alternative='greater')
    for w0, w1 in dmt_segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax2.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    l3 = ax2.plot(time_minutes, dmt_mean_h, color=c_dmt_high, lw=2.0, label='High dose (40mg)')[0]
    ax2.fill_between(time_minutes, dmt_mean_h - dmt_sem_h, dmt_mean_h + dmt_sem_h, color=c_dmt_high, alpha=0.25)
    l4 = ax2.plot(time_minutes, dmt_mean_l, color=c_dmt_low, lw=2.0, label='Low dose (20mg)')[0]
    ax2.fill_between(time_minutes, dmt_mean_l - dmt_sem_l, dmt_mean_l + dmt_sem_l, color=c_dmt_low, alpha=0.25)
    leg2 = ax2.legend([l3, l4], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg2.get_frame().set_facecolor('white'); leg2.get_frame().set_alpha(0.9)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title('DMT', fontweight='bold')
    ax2.grid(True, which='major', axis='y', alpha=0.25); ax2.grid(False, which='major', axis='x')

    for ax in (ax1, ax2):
        ticks = list(range(0, 10))  # 0-9 minutes
        ax.set_xticks(ticks)
        ax.set_xlim(-0.2, 9.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_resp_rvt.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    # Write FDR report
    try:
        lines: List[str] = [
            'FDR COMPARISON: RVT High vs Low (All Subjects, RS and DMT)',
            'Alpha = 0.05',
            ''
        ]
        def _sect(name: str, segs: List[Tuple[float, float]]):
            lines.append(f'PANEL {name}:')
            lines.append(f'  Significant window ranges (count={len(segs)}):')
            if len(segs) == 0:
                lines.append('    - None')
            for (w0, w1) in segs:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
                t1 = w1 * WINDOW_SIZE_SEC / 60.0
                lines.append(f"    - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
            lines.append('')
        _sect('RS', rs_segs)
        _sect('DMT', dmt_segs)
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_resp_rvt.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path


def create_dmt_only_20min_plot(out_dir: str) -> Optional[str]:
    """Create DMT-only extended plot using per-30-second-window mean RVT (~19 minutes).
    
    Uses z-scored data if USE_ZSCORE=True, else absolute RVT values.
    If ZSCORE_BY_SUBJECT=True: uses subject-level normalization.
    If ZSCORE_BY_SUBJECT=False: uses session-level normalization.
    Saves results/resp/rvt/plots/all_subs_dmt_resp_rvt.png
    """
    limit_sec = 1150.0
    total_windows = int(np.floor(limit_sec / WINDOW_SIZE_SEC))  # ~38 windows
    H_list, L_list = [], []
    print(f"  Processing {len(SUJETOS_VALIDADOS_RESP)} subjects for DMT-only plot...")
    for subject in SUJETOS_VALIDADOS_RESP:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_resp_paths(subject, high_session, low_session)
            d_high = load_resp_csv(p_high)
            d_low = load_resp_csv(p_low)
            if d_high is None or d_low is None:
                print(f"    {subject}: Missing DMT data")
                continue
            th_abs = d_high['time'].to_numpy()
            yh_abs = pd.to_numeric(d_high['RSP_RVT'], errors='coerce').to_numpy()
            tl_abs = d_low['time'].to_numpy()
            yl_abs = pd.to_numeric(d_low['RSP_RVT'], errors='coerce').to_numpy()
            
            # Apply z-scoring if enabled (need RS data for baseline)
            if USE_ZSCORE:
                # Load RS data for z-scoring baseline
                p_rsh = build_rs_resp_path(subject, high_session)
                p_rsl = build_rs_resp_path(subject, low_session)
                r_high = load_resp_csv(p_rsh)
                r_low = load_resp_csv(p_rsl)
                if r_high is None or r_low is None:
                    print(f"    {subject}: Missing RS data for z-scoring")
                    continue
                trh_abs = r_high['time'].to_numpy()
                yrh_abs = pd.to_numeric(r_high['RSP_RVT'], errors='coerce').to_numpy()
                trl_abs = r_low['time'].to_numpy()
                yrl_abs = pd.to_numeric(r_low['RSP_RVT'], errors='coerce').to_numpy()
                
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    _, yh_z, _, yl_z, diag = zscore_with_subject_baseline(
                        trh_abs, yrh_abs, th_abs, yh_abs,
                        trl_abs, yrl_abs, tl_abs, yl_abs
                    )
                    if not diag['scalable']:
                        print(f"    {subject}: Not scalable (subject-level) - {diag.get('reason', 'unknown')}")
                        continue
                    yh, yl = yh_z, yl_z
                else:
                    # Z-score each session independently
                    _, yh_z, diag_h = zscore_with_session_baseline(trh_abs, yrh_abs, th_abs, yh_abs)
                    _, yl_z, diag_l = zscore_with_session_baseline(trl_abs, yrl_abs, tl_abs, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        print(f"    {subject}: Not scalable (session-level)")
                        continue
                    yh, yl = yh_z, yl_z
            else:
                # Use absolute values
                yh, yl = yh_abs, yl_abs
            
            rvt_h = [compute_rvt_mean_per_window(th_abs, yh, m) for m in range(total_windows)]
            rvt_l = [compute_rvt_mean_per_window(tl_abs, yl, m) for m in range(total_windows)]
            if None in rvt_h or None in rvt_l:
                print(f"    {subject}: Missing window data (None in rvt_h={None in rvt_h}, rvt_l={None in rvt_l})")
                continue
            print(f"    {subject}: OK - added to DMT plot")
            H_list.append(np.array(rvt_h, dtype=float))
            L_list.append(np.array(rvt_l, dtype=float))
        except Exception as e:
            print(f"    {subject}: Exception - {str(e)}")
            continue
    
    if not (H_list and L_list):
        print(f"  No valid data: H_list={len(H_list)}, L_list={len(L_list)}")
        return None
    print(f"  Found {len(H_list)} subjects with valid data")
    
    H = np.vstack(H_list); L = np.vstack(L_list)
    mean_h = np.nanmean(H, axis=0); mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    x = np.arange(1, total_windows + 1)
    # Convert window indices to time in minutes for x-axis
    time_minutes = (x - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    
    segs = _compute_fdr_significant_segments(H, L, x, alternative='greater')
    for w0, w1 in segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    l1 = ax.plot(time_minutes, mean_h, color=c_dmt_high, lw=2.0, label='High dose (40mg)')[0]
    ax.fill_between(time_minutes, mean_h - sem_h, mean_h + sem_h, color=c_dmt_high, alpha=0.25)
    l2 = ax.plot(time_minutes, mean_l, color=c_dmt_low, lw=2.0, label='Low dose (20mg)')[0]
    ax.fill_between(time_minutes, mean_l - sem_l, mean_l + sem_l, color=c_dmt_low, alpha=0.25)
    leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg.get_frame().set_facecolor('white'); leg.get_frame().set_alpha(0.9)
    ax.set_xlabel('Time (minutes)')
    # Use green color from tab20c for Respiration (RESP/RVT modality) - only first line colored
    ax.text(-0.20, 0.5, 'Respiration', transform=ax.transAxes, 
            fontsize=28, fontweight='bold', color=tab20c_colors[8],
            rotation=90, va='center', ha='center')
    ylabel_text = 'RVT (Z-scored)' if USE_ZSCORE else 'RVT (a.u./min)'
    ax.text(-0.12, 0.5, ylabel_text, transform=ax.transAxes, 
            fontsize=28, fontweight='normal', color='black', 
            rotation=90, va='center', ha='center')
    ax.set_title('DMT', fontweight='bold')
    ax.grid(True, which='major', axis='y', alpha=0.25); ax.grid(False, which='major', axis='x')
    ticks = list(range(0, 20))  # 0-19 minutes
    ax.set_xticks(ticks); ax.set_xlim(-0.2, 19.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_dmt_resp_rvt.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    # Write FDR report
    try:
        lines: List[str] = [
            'FDR COMPARISON: RVT High vs Low (DMT only)',
            'Alpha = 0.05',
            ''
        ]
        lines.append(f"Significant window ranges (count={len(segs)}):")
        if len(segs) == 0:
            lines.append('  - None')
        for (w0, w1) in segs:
            t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
            t1 = w1 * WINDOW_SIZE_SEC / 60.0
            lines.append(f"  - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_dmt_resp_rvt.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path


def create_stacked_subjects_plot(out_dir: str) -> Optional[str]:
    """Create a stacked per-subject figure (RS left, DMT right) using per-30-second-window mean RVT (1..18).
    
    Uses z-scored data if USE_ZSCORE=True, else absolute RVT values.
    If ZSCORE_BY_SUBJECT=True: uses subject-level normalization.
    If ZSCORE_BY_SUBJECT=False: uses session-level normalization.
    
    Saves results/resp/rvt/plots/stacked_subs_resp_rvt.png
    """
    rows: List[Dict] = []
    print(f"  Processing {len(SUJETOS_VALIDADOS_RESP)} subjects for stacked plot...")
    for subject in SUJETOS_VALIDADOS_RESP:
        try:
            # DMT High/Low by session mapping
            high_session, low_session = determine_sessions(subject)
            p_dmt_h, p_dmt_l = build_resp_paths(subject, high_session, low_session)
            dmt_h = load_resp_csv(p_dmt_h)
            dmt_l = load_resp_csv(p_dmt_l)
            if dmt_h is None or dmt_l is None:
                print(f"    {subject}: Missing DMT data")
                continue
            th_abs = dmt_h['time'].to_numpy()
            yh_abs = pd.to_numeric(dmt_h['RSP_RVT'], errors='coerce').to_numpy()
            tl_abs = dmt_l['time'].to_numpy()
            yl_abs = pd.to_numeric(dmt_l['RSP_RVT'], errors='coerce').to_numpy()

            # RS session1/session2, map to High/Low using recorded dose per session
            p_rs1 = build_rs_resp_path(subject, 'session1')
            p_rs2 = build_rs_resp_path(subject, 'session2')
            rs1 = load_resp_csv(p_rs1)
            rs2 = load_resp_csv(p_rs2)
            if rs1 is None or rs2 is None:
                print(f"    {subject}: Missing RS data")
                continue
            t1_abs = rs1['time'].to_numpy()
            y1_abs = pd.to_numeric(rs1['RSP_RVT'], errors='coerce').to_numpy()
            t2_abs = rs2['time'].to_numpy()
            y2_abs = pd.to_numeric(rs2['RSP_RVT'], errors='coerce').to_numpy()
            
            # Apply z-scoring if enabled
            if USE_ZSCORE:
                # Determine which RS session is high/low
                try:
                    dose_s1 = get_dosis_sujeto(subject, 1)
                except Exception:
                    dose_s1 = 'Alta'
                cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
                
                if cond1 == 'High':
                    # session1=high, session2=low
                    trh_abs, yrh_abs = t1_abs, y1_abs
                    trl_abs, yrl_abs = t2_abs, y2_abs
                else:
                    # session1=low, session2=high
                    trh_abs, yrh_abs = t2_abs, y2_abs
                    trl_abs, yrl_abs = t1_abs, y1_abs
                
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    yrh_z, yh_z, yrl_z, yl_z, diag = zscore_with_subject_baseline(
                        trh_abs, yrh_abs, th_abs, yh_abs,
                        trl_abs, yrl_abs, tl_abs, yl_abs
                    )
                    if not diag['scalable']:
                        print(f"    {subject}: Not scalable (subject-level) - {diag.get('reason', 'unknown')}")
                        continue
                    yh, yl = yh_z, yl_z
                    if cond1 == 'High':
                        y1, y2 = yrh_z, yrl_z
                    else:
                        y1, y2 = yrl_z, yrh_z
                else:
                    # Z-score each session independently
                    yrh_z, yh_z, diag_h = zscore_with_session_baseline(trh_abs, yrh_abs, th_abs, yh_abs)
                    yrl_z, yl_z, diag_l = zscore_with_session_baseline(trl_abs, yrl_abs, tl_abs, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        print(f"    {subject}: Not scalable (session-level)")
                        continue
                    yh, yl = yh_z, yl_z
                    if cond1 == 'High':
                        y1, y2 = yrh_z, yrl_z
                    else:
                        y1, y2 = yrl_z, yrh_z
            else:
                # Use absolute values
                yh, yl, y1, y2 = yh_abs, yl_abs, y1_abs, y2_abs
            
            # Compute mean RVT from z-scored or absolute data
            rvt_dmt_h = [compute_rvt_mean_per_window(th_abs, yh, m) for m in range(N_WINDOWS)]
            rvt_dmt_l = [compute_rvt_mean_per_window(tl_abs, yl, m) for m in range(N_WINDOWS)]
            rvt_rs1 = [compute_rvt_mean_per_window(t1_abs, y1, m) for m in range(N_WINDOWS)]
            rvt_rs2 = [compute_rvt_mean_per_window(t2_abs, y2, m) for m in range(N_WINDOWS)]
            
            if None in rvt_dmt_h or None in rvt_dmt_l or None in rvt_rs1 or None in rvt_rs2:
                print(f"    {subject}: Missing window data")
                continue
            print(f"    {subject}: OK - added to stacked plot")

            try:
                dose_s1 = get_dosis_sujeto(subject, 1)
            except Exception:
                dose_s1 = 'Alta'
            cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
            cond2 = 'Low' if cond1 == 'High' else 'High'
            if cond1 == 'High':
                rvt_rs_h, rvt_rs_l = rvt_rs1, rvt_rs2
            else:
                rvt_rs_h, rvt_rs_l = rvt_rs2, rvt_rs1

            rows.append({
                'subject': subject,
                'windows': list(range(1, N_WINDOWS + 1)),
                'rs_high': np.asarray(rvt_rs_h, dtype=float),
                'rs_low': np.asarray(rvt_rs_l, dtype=float),
                'dmt_high': np.asarray(rvt_dmt_h, dtype=float),
                'dmt_low': np.asarray(rvt_dmt_l, dtype=float),
            })
        except Exception as e:
            print(f"    {subject}: Exception - {str(e)}")
            continue

    if not rows:
        print(f"  No valid subjects found for stacked plot")
        return None

    print(f"  Found {len(rows)} subjects with valid data")
    n = len(rows)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(18, max(6.0, 3.2 * n)),
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.8, 'wspace': 0.35}
    )
    if n == 1:
        axes = np.array([axes])

    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    c_rs_high, c_rs_low = COLOR_RS_HIGH, COLOR_RS_LOW
    time_ticks = list(range(0, 10))  # 0-9 minutes

    from matplotlib.lines import Line2D

    for i, row in enumerate(rows):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]

        # Convert window indices to time in minutes for x-axis
        time_minutes = (np.array(row['windows']) - 0.5) * WINDOW_SIZE_SEC / 60.0

        # RS panel
        ax_rs.plot(time_minutes, row['rs_high'], color=c_rs_high, lw=1.8, marker='o', markersize=4)
        ax_rs.plot(time_minutes, row['rs_low'], color=c_rs_low, lw=1.8, marker='o', markersize=4)
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ylabel = 'RVT (Z-scored)' if USE_ZSCORE else 'RVT (a.u./min)'
        ax_rs.set_ylabel(r'$\mathbf{Respiration}$' + f'\n{ylabel}', fontsize=12)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(-0.2, 9.2)
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=c_rs_high, lw=1.8, marker='o', markersize=4, label='High dose (40mg)'),
            Line2D([0], [0], color=c_rs_low, lw=1.8, marker='o', markersize=4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)

        # DMT panel
        ax_dmt.plot(time_minutes, row['dmt_high'], color=c_dmt_high, lw=1.8, marker='o', markersize=4)
        ax_dmt.plot(time_minutes, row['dmt_low'], color=c_dmt_low, lw=1.8, marker='o', markersize=4)
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ylabel = 'RVT (Z-scored)' if USE_ZSCORE else 'RVT (a.u./min)'
        ax_dmt.set_ylabel(r'$\mathbf{Respiration}$' + f'\n{ylabel}', fontsize=12)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(-0.2, 9.2)
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=c_dmt_high, lw=1.8, marker='o', markersize=4, label='High dose (40mg)'),
            Line2D([0], [0], color=c_dmt_low, lw=1.8, marker='o', markersize=4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)

        ax_rs.set_xticks(time_ticks)
        ax_dmt.set_xticks(time_ticks)

    # Apply tight_layout first to finalize subplot positions
    fig.tight_layout(pad=2.0)

    # Add subject codes centered between columns, aligned to final layout
    for i, row in enumerate(rows):
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 1].get_position()
        y_center = (pos_left.y0 + pos_left.y1) / 2.0
        x_center = (pos_left.x1 + pos_right.x0) / 2.0
        fig.text(
            x_center,
            y_center + 0.02,
            row['subject'],
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=STACKED_SUBJECT_FONTSIZE,
            transform=fig.transFigure,
        )
    out_path = os.path.join(out_dir, 'plots', 'stacked_subs_resp_rvt.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def generate_captions_file(output_dir: str) -> None:
    captions = [
        'Figure: LME Coefficients (RSP_RVT)\n\n'
        'Point estimates (β) and 95% CIs for fixed effects from the mixed model. '
        'Reference line at zero aids interpretation. Significant effects are visually emphasized.',
        '',
        'Figure: Marginal Means Over Time (RS vs DMT × High vs Low)\n\n'
        'Group-level mean ± 95% CI of RSP_RVT (a.u./min) across the first 9 minutes for each condition (RS Low/High, DMT Low/High). '
        'Legends indicate dose levels; shading shows uncertainty.',
        '',
        'Figure: Main State Effect Over Time\n\n'
        'Mean ± 95% CI for RS and DMT (averaged across dose) across minutes 0–8. '
        'Illustrates overall state separation and temporal trend.',
        '',
        'Figure: State × Dose Interaction (Panels)\n\n'
        'Left: RS Low vs High; Right: DMT Low vs High. Lines show mean ± 95% CI across minutes 0–8. '
        'Highlights how dose effects differ between states.',
        '',
        'Figure: Group-level RSP_RVT Timecourse (9 min)\n\n'
        'Two panels (RS, DMT) showing mean ± SEM over time; High vs Low dose with legends. '
        'Gray shading indicates FDR-significant differences (High vs Low) across time. '
        'Time axis in minutes (0–9).',
        '',
        'Figure: DMT-only RSP_RVT Timecourse (~19 min)\n\n'
        'Extended timecourse plot showing DMT High vs Low over approximately 19 minutes. '
        'Gray shading indicates FDR-significant differences (High vs Low) across time. '
        'Mean ± SEM for all subjects. Time axis in minutes (0–19).',
        '',
        'Figure: Stacked Per-Subject RSP_RVT Timecourse (9 min)\n\n'
        'Individual subject traces for RS (left) and DMT (right) conditions. '
        'High/Low dose traces shown in respective colors. Subject codes centered between panels.',
    ]
    with open(os.path.join(output_dir, 'captions_resp_rvt.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def prepare_extended_long_data_resp_rvt() -> pd.DataFrame:
    """Build long-format per-30-second window RSP_RVT table for extended DMT (~19 min = 38 windows).
    
    This function exports DMT-only data for extended time range, used by composite analysis.
    Uses same z-scoring approach as prepare_long_data_resp_rvt().
    """
    limit_sec = 1150.0
    total_windows = int(np.floor(limit_sec / WINDOW_SIZE_SEC))  # ~38 windows
    rows: List[Dict] = []
    qc_log: List[str] = []
    
    for subject in SUJETOS_VALIDADOS_RESP:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_resp_paths(subject, high_session, low_session)
        rs_high_path = build_rs_resp_path(subject, high_session)
        rs_low_path = build_rs_resp_path(subject, low_session)

        dmt_high = load_resp_csv(dmt_high_path)
        dmt_low = load_resp_csv(dmt_low_path)
        rs_high = load_resp_csv(rs_high_path)
        rs_low = load_resp_csv(rs_low_path)
        
        if any(x is None for x in (dmt_high, dmt_low, rs_high, rs_low)):
            continue

        t_dmt_high = dmt_high['time'].to_numpy()
        rvt_dmt_high_abs = pd.to_numeric(dmt_high['RSP_RVT'], errors='coerce').to_numpy()
        t_dmt_low = dmt_low['time'].to_numpy()
        rvt_dmt_low_abs = pd.to_numeric(dmt_low['RSP_RVT'], errors='coerce').to_numpy()
        t_rs_high = rs_high['time'].to_numpy()
        rvt_rs_high_abs = pd.to_numeric(rs_high['RSP_RVT'], errors='coerce').to_numpy()
        t_rs_low = rs_low['time'].to_numpy()
        rvt_rs_low_abs = pd.to_numeric(rs_low['RSP_RVT'], errors='coerce').to_numpy()

        if USE_ZSCORE:
            if ZSCORE_BY_SUBJECT:
                rvt_rs_high_z, rvt_dmt_high_z, rvt_rs_low_z, rvt_dmt_low_z, diag = zscore_with_subject_baseline(
                    t_rs_high, rvt_rs_high_abs, t_dmt_high, rvt_dmt_high_abs,
                    t_rs_low, rvt_rs_low_abs, t_dmt_low, rvt_dmt_low_abs
                )
                if not diag['scalable']:
                    qc_log.append(f"{subject}: Not scalable: {diag['reason']}")
                    continue
                rvt_dmt_high_z_use, rvt_dmt_low_z_use = rvt_dmt_high_z, rvt_dmt_low_z
            else:
                rvt_rs_high_z, rvt_dmt_high_z, diag_high = zscore_with_session_baseline(
                    t_rs_high, rvt_rs_high_abs, t_dmt_high, rvt_dmt_high_abs
                )
                rvt_rs_low_z, rvt_dmt_low_z, diag_low = zscore_with_session_baseline(
                    t_rs_low, rvt_rs_low_abs, t_dmt_low, rvt_dmt_low_abs
                )
                if not (diag_high['scalable'] and diag_low['scalable']):
                    continue
                rvt_dmt_high_z_use, rvt_dmt_low_z_use = rvt_dmt_high_z, rvt_dmt_low_z
            
            # Process extended windows (DMT only)
            for window_idx in range(total_windows):
                window_label = window_idx + 1
                rvt_dmt_h_z = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_z_use, window_idx)
                rvt_dmt_l_z = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_z_use, window_idx)
                
                if rvt_dmt_h_z is not None and rvt_dmt_l_z is not None:
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_z, 'Scale': 'z'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_z, 'Scale': 'z'},
                    ])
                    if EXPORT_ABSOLUTE_SCALE:
                        rvt_dmt_h_abs = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_abs, window_idx)
                        rvt_dmt_l_abs = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_abs, window_idx)
                        if rvt_dmt_h_abs is not None and rvt_dmt_l_abs is not None:
                            rows.extend([
                                {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_abs, 'Scale': 'abs'},
                                {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_abs, 'Scale': 'abs'},
                            ])
        else:
            for window_idx in range(total_windows):
                window_label = window_idx + 1
                rvt_dmt_h_abs = compute_rvt_mean_per_window(t_dmt_high, rvt_dmt_high_abs, window_idx)
                rvt_dmt_l_abs = compute_rvt_mean_per_window(t_dmt_low, rvt_dmt_low_abs, window_idx)
                if rvt_dmt_h_abs is not None and rvt_dmt_l_abs is not None:
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'RSP_RVT': rvt_dmt_h_abs, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'RSP_RVT': rvt_dmt_l_abs, 'Scale': 'abs'},
                    ])

    if not rows:
        raise ValueError('No valid extended RSP_RVT data found!')
    
    if qc_log:
        print(f"  Warning: {len(qc_log)} subjects excluded from extended data")

    df = pd.DataFrame(rows)
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['Scale'] = pd.Categorical(df['Scale'], categories=['z', 'abs'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['window_c'] = df['window'] - df['window'].mean()
    return df


def main() -> bool:
    out_dir = os.path.join('results', 'resp', 'rvt')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    try:
        # Data preparation
        scale_mode = "z-scoring (subject baseline)" if USE_ZSCORE and ZSCORE_BY_SUBJECT else "z-scoring (session baseline)" if USE_ZSCORE else "absolute RVT values"
        print(f"Preparing long-format data with {scale_mode}...")
        df = prepare_long_data_resp_rvt()
        
        # Save all scales
        df.to_csv(os.path.join(out_dir, 'resp_rvt_minute_long_data_all_scales.csv'), index=False)
        print(f"  [OK] Saved all scales: {len(df)} rows")
        
        # Save primary scale data separately
        scale_to_use = 'z' if USE_ZSCORE else 'abs'
        df_primary = df[df['Scale'] == scale_to_use]
        scale_filename = 'resp_rvt_minute_long_data_z.csv' if USE_ZSCORE else 'resp_rvt_minute_long_data_abs.csv'
        df_primary.to_csv(os.path.join(out_dir, scale_filename), index=False)
        scale_desc = "z-scored" if USE_ZSCORE else "absolute"
        print(f"  [OK] Saved {scale_desc} data: {len(df_primary)} rows from {len(df_primary['subject'].unique())} subjects")
        
        # Export extended DMT data (~19 minutes)
        print("Preparing extended DMT data (~19 minutes)...")
        df_extended = prepare_extended_long_data_resp_rvt()
        df_extended.to_csv(os.path.join(out_dir, 'resp_rvt_extended_dmt_all_scales.csv'), index=False)
        df_ext_primary = df_extended[df_extended['Scale'] == scale_to_use]
        ext_filename = 'resp_rvt_extended_dmt_z.csv' if USE_ZSCORE else 'resp_rvt_extended_dmt_abs.csv'
        df_ext_primary.to_csv(os.path.join(out_dir, ext_filename), index=False)
        print(f"  [OK] Saved extended DMT data: {len(df_ext_primary)} rows from {len(df_ext_primary['subject'].unique())} subjects")
        
        # QC check: RS by dose
        rs_qc = df_primary[df_primary['State'] == 'RS'].groupby('Dose', observed=False)['RSP_RVT'].agg(['count', 'mean', 'std']).round(4)
        with open(os.path.join(out_dir, 'qc_rs_by_dose.txt'), 'w') as f:
            f.write('QC CHECK: RS RSP_RVT by Dose\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'Scale: {scale_desc}\n\n')
            f.write(str(rs_qc) + '\n\n')
            if USE_ZSCORE:
                if ZSCORE_BY_SUBJECT:
                    f.write('Note: Z-scoring uses ALL sessions of subject (RS_high + DMT_high + RS_low + DMT_low).\n')
                    f.write('Both sessions share the same normalization parameters (mu and sigma).\n')
                    f.write('RS High and RS Low should have similar means (both near 0) since they use\n')
                    f.write('the same subject-level baseline.\n')
                else:
                    f.write('Note: Z-scoring uses entire session (RS + DMT concatenated) as baseline.\n')
                    f.write('Each session is normalized independently using mu and sigma computed\n')
                    f.write('from the full session data (RS + DMT together).\n')
                    f.write('RS values reflect session-level normalization, not pure baseline.\n')
            else:
                f.write('Note: Using absolute RVT values (a.u./min) without normalization.\n')
        print(f"  [OK] QC check saved")
        
        # LME model (uses appropriate scale)
        print(f"Fitting LME model on {scale_desc} data...")
        fitted, diagnostics = fit_lme_model(df)
        plot_model_diagnostics(fitted, df_primary, plots_dir)
        
        # Hypothesis testing + report
        print("Performing hypothesis testing with FDR correction...")
        hyp = hypothesis_testing_with_fdr(fitted)
        report_path = generate_report(fitted, diagnostics, hyp, df, out_dir)
        print(f"  Report saved: {report_path}")
        
        # Coefficients
        print("Generating plots...")
        coefs = load_lme_results_from_report(report_path)
        coef_df = prepare_coefficient_data(coefs)
        create_coefficient_plot(coef_df, os.path.join(plots_dir, 'lme_coefficient_plot.png'))
        create_effect_sizes_table(coef_df, os.path.join(plots_dir, 'effect_sizes_table.csv'))
        
        # Summary statistics
        stats_df = compute_empirical_means_and_ci(df)
        overall = stats_df.groupby('condition').agg({
            'mean': 'mean',
            'se': lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1)),
            'n': 'mean'
        }).round(4)
        overall['ci_lower'] = overall['mean'] - 1.96 * overall['se']
        overall['ci_upper'] = overall['mean'] + 1.96 * overall['se']
        overall.to_csv(os.path.join(plots_dir, 'summary_statistics.csv'))
        
        # Model summary txt
        create_model_summary_txt(diagnostics, coef_df, os.path.join(out_dir, 'model_summary.txt'))
        
        # Combined summary (9 min with FDR)
        print("Creating combined summary plot with FDR...")
        create_combined_summary_plot(out_dir)
        
        # DMT-only extended (19 min with FDR)
        print("Creating DMT-only extended plot...")
        dmt_plot = create_dmt_only_20min_plot(out_dir)
        if dmt_plot:
            print(f"  OK DMT-only plot saved: {dmt_plot}")
        else:
            print("  WARNING: DMT-only plot could not be created (insufficient data)")
        
        # Stacked per-subject
        print("Creating stacked per-subject plot...")
        stacked_plot = create_stacked_subjects_plot(out_dir)
        if stacked_plot:
            print(f"  OK Stacked plot saved: {stacked_plot}")
        else:
            print("  WARNING: Stacked plot could not be created (insufficient data)")
        
        # Captions
        generate_captions_file(out_dir)
        
        print(f"\nOK Respiration RVT analysis complete! Results in: {out_dir}")
        if USE_ZSCORE:
            print(f"  - Z-scored window means used for LME modeling")
            if ZSCORE_BY_SUBJECT:
                print(f"  - Subject-level normalization: ALL sessions used as baseline")
                print(f"  - Mu and sigma computed from RS_high + DMT_high + RS_low + DMT_low")
                print(f"  - Both sessions of each subject share same normalization parameters")
            else:
                print(f"  - Session-level normalization: Each session normalized independently")
                print(f"  - Mu and sigma computed per session from RS + DMT")
                print(f"  - Each session has its own normalization parameters")
            print(f"  - Continuous plots show z-scored data")
        else:
            print(f"  - Absolute RVT values (a.u./min) used for LME modeling")
            print(f"  - No normalization applied")
            print(f"  - Continuous plots show absolute values")
    except Exception as e:
        print(f'Respiration RVT analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)

