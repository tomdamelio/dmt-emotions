# -*- coding: utf-8 -*-
"""
Unified SMNA Analysis: LME modeling and visualization (first 9 minutes).

This script combines and streamlines the SMNA AUC LME analysis and plotting into
one reproducible pipeline that:
  1) Prepares long-format SMNA AUC data by minute (0-8) across conditions
  2) Fits an LME model with State × Dose and time effects
  3) Applies BH-FDR to families of hypotheses
  4) Produces publication-ready plots with consistent aesthetics
  5) Writes a plain-text model summary and captions for figures

Outputs are written to: results/eda/smna/

Run:
  python scripts/run_eda_smna_analysis.py
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
# Only set TkAgg if no backend is already set (allows Agg when imported from run_figures.py)
if matplotlib.get_backend() == 'agg' or not matplotlib.get_backend():
    pass  # Keep current backend
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass  # Ignore if backend already set
import matplotlib.pyplot as plt
import seaborn as sns  # optional

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_EDA,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import shapiro
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

#############################
# Plot aesthetics & centralized hyperparameters
#############################

# Import centralized figure configuration
try:
    from figure_config import (
        FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK_LABEL,
        FONT_SIZE_LEGEND, FONT_SIZE_PANEL_LABEL, FONT_SIZE_ANNOTATION,
        FONT_SIZE_TITLE_SMALL, FONT_SIZE_AXIS_LABEL_SMALL, 
        FONT_SIZE_TICK_LABEL_SMALL, FONT_SIZE_LEGEND_SMALL,
        LINE_WIDTH, MARKER_SIZE, LEGEND_MARKERSCALE, LEGEND_BORDERPAD,
        LEGEND_HANDLELENGTH, LEGEND_LABELSPACING, LEGEND_BORDERAXESPAD,
        COLOR_EDA_HIGH, COLOR_EDA_LOW, DOUBLE_COL_WIDTH,
        apply_rcparams, add_panel_label, style_legend
    )
    AXES_TITLE_SIZE = FONT_SIZE_TITLE
    AXES_LABEL_SIZE = FONT_SIZE_AXIS_LABEL
    TICK_LABEL_SIZE = FONT_SIZE_TICK_LABEL
    TICK_LABEL_SIZE_SMALL = FONT_SIZE_TICK_LABEL_SMALL
    LEGEND_FONTSIZE = FONT_SIZE_LEGEND
    LEGEND_FONTSIZE_SMALL = FONT_SIZE_LEGEND_SMALL
    STACKED_AXES_LABEL_SIZE = FONT_SIZE_AXIS_LABEL_SMALL
    STACKED_TICK_LABEL_SIZE = FONT_SIZE_TICK_LABEL_SMALL
    STACKED_SUBJECT_FONTSIZE = FONT_SIZE_TITLE
    apply_rcparams()
except ImportError:
    AXES_TITLE_SIZE = 10
    AXES_LABEL_SIZE = 9
    TICK_LABEL_SIZE = 8
    TICK_LABEL_SIZE_SMALL = 7
    LEGEND_FONTSIZE = 8
    LEGEND_FONTSIZE_SMALL = 7
    LEGEND_MARKERSCALE = 1.2
    LEGEND_BORDERPAD = 0.4
    LEGEND_HANDLELENGTH = 2.0
    LEGEND_LABELSPACING = 0.5
    LEGEND_BORDERAXESPAD = 0.5
    STACKED_AXES_LABEL_SIZE = 8
    STACKED_TICK_LABEL_SIZE = 7
    STACKED_SUBJECT_FONTSIZE = 10

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.titlepad': 6.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.fontsize': LEGEND_FONTSIZE,
    'legend.borderpad': LEGEND_BORDERPAD,
    'legend.handlelength': LEGEND_HANDLELENGTH,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
})

# EDA SMNA modality uses green color scheme from tab20c palette
tab20c_colors = plt.cm.tab20c.colors
try:
    COLOR_RS_HIGH = COLOR_EDA_HIGH
    COLOR_RS_LOW = COLOR_EDA_LOW
    COLOR_DMT_HIGH = COLOR_EDA_HIGH
    COLOR_DMT_LOW = COLOR_EDA_LOW
except NameError:
    # EDA uses orange family (indices 4-7)
    COLOR_RS_HIGH = tab20c_colors[4]
    COLOR_RS_LOW = tab20c_colors[6]
    COLOR_DMT_HIGH = tab20c_colors[4]
    COLOR_DMT_LOW = tab20c_colors[6]

# Analysis window: first 9 minutes (18 windows of 30 seconds each)
N_WINDOWS = 18  # 30-second windows: 0-30s, 30-60s, ..., 510-540s
WINDOW_SIZE_SEC = 30  # 30-second windows
MAX_TIME_SEC = N_WINDOWS * WINDOW_SIZE_SEC  # 540 seconds = 9 minutes

# Z-scoring configuration: use RS as baseline per session
USE_RS_ZSCORE = True  # If True: z-score using session baseline (RS+DMT); If False: use absolute AUC values
ZSCORE_BY_SUBJECT = True  # If True: z-score using all sessions of subject; If False: z-score each session independently
EXPORT_ABSOLUTE_SCALE = True  # Also export absolute scale for QC (only when USE_RS_ZSCORE=True)

# Optional trims (in seconds) - set to None to disable
RS_TRIM_START = None  # Trim start of RS (e.g., 5.0 for first 5 seconds)
DMT_TRIM_START = None  # Trim start of DMT (e.g., 5.0 for first 5 seconds)

# Minimum samples per window to accept a session
MIN_SAMPLES_PER_WINDOW = 10


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_cvx_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    """Build full paths for DMT high and DMT low CVX decomposition CSVs for EDA."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high_cvx_decomposition.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low_cvx_decomposition.csv")
    return high_csv, low_csv


def build_rs_cvx_path(subject: str, session: str) -> str:
    """Build RS CVX decomposition path for a given subject/session using session dose."""
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)  # 'Alta' or 'Baja'
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}_cvx_decomposition.csv")


def load_cvx_smna(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load time and SMNA arrays from CVX decomposition CSV.

    Returns (t, smna) or None if missing/invalid.
    """
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


def zscore_with_session_baseline(t_rs: np.ndarray, y_rs: np.ndarray, 
                                 t_dmt: np.ndarray, y_dmt: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Z-score both RS and DMT using the entire session (RS + DMT) as baseline.
    
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
    
    # Concatenate entire session (RS + DMT) to compute mu and sigma
    y_session = np.concatenate([y_rs, y_dmt])
    valid_session = ~np.isnan(y_session)
    
    if np.sum(valid_session) < MIN_SAMPLES_PER_WINDOW * 2:
        diagnostics['reason'] = f'Session insufficient valid samples: {np.sum(valid_session)}'
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
    diagnostics['n_session'] = int(np.sum(valid_session))
    diagnostics['n_rs'] = int(np.sum(~np.isnan(y_rs)))
    diagnostics['n_dmt'] = int(np.sum(~np.isnan(y_dmt)))
    
    return y_rs_z, y_dmt_z, diagnostics


def zscore_with_subject_baseline(t_rs_high: np.ndarray, y_rs_high: np.ndarray,
                                 t_dmt_high: np.ndarray, y_dmt_high: np.ndarray,
                                 t_rs_low: np.ndarray, y_rs_low: np.ndarray,
                                 t_dmt_low: np.ndarray, y_dmt_low: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Z-score all sessions of a subject using combined baseline (all RS + all DMT).
    
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
    
    # Concatenate ALL subject data (both sessions) to compute mu and sigma
    y_all = np.concatenate([y_rs_high, y_dmt_high, y_rs_low, y_dmt_low])
    valid_all = ~np.isnan(y_all)
    
    if np.sum(valid_all) < MIN_SAMPLES_PER_WINDOW * 4:
        diagnostics['reason'] = f'Subject insufficient valid samples: {np.sum(valid_all)}'
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
    diagnostics['n_all'] = int(np.sum(valid_all))
    diagnostics['n_rs_high'] = int(np.sum(~np.isnan(y_rs_high)))
    diagnostics['n_dmt_high'] = int(np.sum(~np.isnan(y_dmt_high)))
    diagnostics['n_rs_low'] = int(np.sum(~np.isnan(y_rs_low)))
    diagnostics['n_dmt_low'] = int(np.sum(~np.isnan(y_dmt_low)))
    
    return y_rs_high_z, y_dmt_high_z, y_rs_low_z, y_dmt_low_z, diagnostics


def compute_auc_window(t: np.ndarray, y: np.ndarray, window_idx: int) -> Optional[float]:
    """Compute AUC for a specific 30-second window.
    
    Parameters:
        t: Time array (seconds)
        y: SMNA values
        window_idx: Window index (0-based, each window is 30 seconds)
    """
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


def prepare_long_data() -> pd.DataFrame:
    """Prepare data in long format for LME analysis (first 9 minutes = 18 windows of 30s).
    
    If USE_RS_ZSCORE=True: z-scores using session or subject baseline (RS+DMT).
    If ZSCORE_BY_SUBJECT=True: uses all sessions of subject for normalization.
    If ZSCORE_BY_SUBJECT=False: uses each session independently for normalization.
    If USE_RS_ZSCORE=False: uses absolute AUC values.

    Returns:
        DataFrame with columns: subject, session, window, State, Dose, AUC, Scale, window_c
    """
    rows: List[Dict] = []
    qc_log: List[str] = []

    for subject in SUJETOS_VALIDADOS_EDA:
        high_session, low_session = determine_sessions(subject)

        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs_high_path = build_rs_cvx_path(subject, high_session)
        rs_low_path = build_rs_cvx_path(subject, low_session)

        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs_high = load_cvx_smna(rs_high_path)
        rs_low = load_cvx_smna(rs_low_path)

        if None in (dmt_high, dmt_low, rs_high, rs_low):
            qc_log.append(f"{subject}: Missing data files")
            continue

        t_dmt_high, smna_dmt_high_abs = dmt_high
        t_dmt_low, smna_dmt_low_abs = dmt_low
        t_rs_high, smna_rs_high_abs = rs_high
        t_rs_low, smna_rs_low_abs = rs_low

        if USE_RS_ZSCORE:
            if ZSCORE_BY_SUBJECT:
                # Z-score using ALL sessions of subject as baseline
                smna_rs_high_z, smna_dmt_high_z, smna_rs_low_z, smna_dmt_low_z, diag = zscore_with_subject_baseline(
                    t_rs_high, smna_rs_high_abs, t_dmt_high, smna_dmt_high_abs,
                    t_rs_low, smna_rs_low_abs, t_dmt_low, smna_dmt_low_abs
                )
                
                if not diag['scalable']:
                    qc_log.append(f"{subject}: Not scalable (subject-level): {diag['reason']}")
                    continue
                
                # Process each 30-second window with z-scored data for BOTH sessions
                for window_idx in range(N_WINDOWS):
                    window_label = window_idx + 1
                    
                    # Compute AUC for z-scored data
                    auc_dmt_high_z = compute_auc_window(t_dmt_high, smna_dmt_high_z, window_idx)
                    auc_dmt_low_z = compute_auc_window(t_dmt_low, smna_dmt_low_z, window_idx)
                    auc_rs_high_z = compute_auc_window(t_rs_high, smna_rs_high_z, window_idx)
                    auc_rs_low_z = compute_auc_window(t_rs_low, smna_rs_low_z, window_idx)
                    
                    if None not in (auc_dmt_high_z, auc_dmt_low_z, auc_rs_high_z, auc_rs_low_z):
                        rows.extend([
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low_z, 'Scale': 'z'},
                            {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'AUC': auc_rs_high_z, 'Scale': 'z'},
                            {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low_z, 'Scale': 'z'},
                        ])
                        
                        # Optional: absolute scale for QC
                        if EXPORT_ABSOLUTE_SCALE:
                            auc_dmt_high_abs = compute_auc_window(t_dmt_high, smna_dmt_high_abs, window_idx)
                            auc_dmt_low_abs = compute_auc_window(t_dmt_low, smna_dmt_low_abs, window_idx)
                            auc_rs_high_abs = compute_auc_window(t_rs_high, smna_rs_high_abs, window_idx)
                            auc_rs_low_abs = compute_auc_window(t_rs_low, smna_rs_low_abs, window_idx)
                            
                            if None not in (auc_dmt_high_abs, auc_dmt_low_abs, auc_rs_high_abs, auc_rs_low_abs):
                                rows.extend([
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'AUC': auc_rs_high_abs, 'Scale': 'abs'},
                                    {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low_abs, 'Scale': 'abs'},
                                ])
            
            else:
                # Z-score each session independently (original behavior)
                # Process HIGH session
                smna_rs_high_z, smna_dmt_high_z, diag_high = zscore_with_session_baseline(
                    t_rs_high, smna_rs_high_abs, t_dmt_high, smna_dmt_high_abs
                )
                
                if diag_high['scalable']:
                    for window_idx in range(N_WINDOWS):
                        window_label = window_idx + 1
                        
                        auc_dmt_high_z = compute_auc_window(t_dmt_high, smna_dmt_high_z, window_idx)
                        auc_rs_high_z = compute_auc_window(t_rs_high, smna_rs_high_z, window_idx)
                        
                        if auc_dmt_high_z is not None and auc_rs_high_z is not None:
                            rows.extend([
                                {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high_z, 'Scale': 'z'},
                                {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'AUC': auc_rs_high_z, 'Scale': 'z'},
                            ])
                            
                            if EXPORT_ABSOLUTE_SCALE:
                                auc_dmt_high_abs = compute_auc_window(t_dmt_high, smna_dmt_high_abs, window_idx)
                                auc_rs_high_abs = compute_auc_window(t_rs_high, smna_rs_high_abs, window_idx)
                                if auc_dmt_high_abs is not None and auc_rs_high_abs is not None:
                                    rows.extend([
                                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high_abs, 'Scale': 'abs'},
                                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'AUC': auc_rs_high_abs, 'Scale': 'abs'},
                                    ])
                else:
                    qc_log.append(f"{subject} HIGH session not scalable: {diag_high['reason']}")
                
                # Process LOW session
                smna_rs_low_z, smna_dmt_low_z, diag_low = zscore_with_session_baseline(
                    t_rs_low, smna_rs_low_abs, t_dmt_low, smna_dmt_low_abs
                )
                
                if diag_low['scalable']:
                    for window_idx in range(N_WINDOWS):
                        window_label = window_idx + 1
                        
                        auc_dmt_low_z = compute_auc_window(t_dmt_low, smna_dmt_low_z, window_idx)
                        auc_rs_low_z = compute_auc_window(t_rs_low, smna_rs_low_z, window_idx)
                        
                        if auc_dmt_low_z is not None and auc_rs_low_z is not None:
                            rows.extend([
                                {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low_z, 'Scale': 'z'},
                                {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low_z, 'Scale': 'z'},
                            ])
                            
                            if EXPORT_ABSOLUTE_SCALE:
                                auc_dmt_low_abs = compute_auc_window(t_dmt_low, smna_dmt_low_abs, window_idx)
                                auc_rs_low_abs = compute_auc_window(t_rs_low, smna_rs_low_abs, window_idx)
                                if auc_dmt_low_abs is not None and auc_rs_low_abs is not None:
                                    rows.extend([
                                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low_abs, 'Scale': 'abs'},
                                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low_abs, 'Scale': 'abs'},
                                    ])
                else:
                    qc_log.append(f"{subject} LOW session not scalable: {diag_low['reason']}")
        
        else:
            # Use absolute values (no z-scoring)
            for window_idx in range(N_WINDOWS):
                window_label = window_idx + 1
                
                auc_dmt_high = compute_auc_window(t_dmt_high, smna_dmt_high_abs, window_idx)
                auc_dmt_low = compute_auc_window(t_dmt_low, smna_dmt_low_abs, window_idx)
                auc_rs_high = compute_auc_window(t_rs_high, smna_rs_high_abs, window_idx)
                auc_rs_low = compute_auc_window(t_rs_low, smna_rs_low_abs, window_idx)
                
                if None not in (auc_dmt_high, auc_dmt_low, auc_rs_high, auc_rs_low):
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low, 'Scale': 'abs'},
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'RS', 'Dose': 'High', 'AUC': auc_rs_high, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low, 'Scale': 'abs'},
                    ])

    if not rows:
        raise ValueError("No valid SMNA data found for any subject!")
    
    # Save QC log
    if USE_RS_ZSCORE and qc_log:
        print(f"Warning: {len(qc_log)} sessions/subjects excluded from z-scoring:")
        for msg in qc_log:
            print(f"  {msg}")

    df = pd.DataFrame(rows)

    # Set categorical variables with proper ordering
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['Scale'] = pd.Categorical(df['Scale'], categories=['z', 'abs'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])

    # Centered window
    df['window_c'] = df['window'] - df['window'].mean()

    return df


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    """Fit the LME model with specified fixed and random effects."""
    # Filter to appropriate scale (z-scored if USE_RS_ZSCORE=True, else absolute)
    scale_to_use = 'z' if USE_RS_ZSCORE else 'abs'
    df_model = df[df['Scale'] == scale_to_use].copy()
    if len(df_model) == 0:
        return None, {'error': f'No {scale_to_use}-scaled data available'}
    try:
        formula = "AUC ~ State * Dose + window_c + State:window_c + Dose:window_c"
        model = mixedlm(formula, df_model, groups=df_model["subject"])  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitted = model.fit()
            convergence_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        return None, {'error': str(e)}

    diagnostics = {
        'aic': getattr(fitted, 'aic', np.nan),
        'bic': getattr(fitted, 'bic', np.nan),
        'loglik': getattr(fitted, 'llf', np.nan),
        'n_obs': getattr(fitted, 'nobs', len(df)),
        'n_groups': len(df['subject'].unique()),
        'convergence_warnings': convergence_warnings,
        'random_effects_var': getattr(fitted, 'cov_re', None),
        'residual_var': getattr(fitted, 'scale', np.nan),
    }
    return fitted, diagnostics


def plot_model_diagnostics(fitted_model, df: pd.DataFrame, output_dir: str) -> None:
    """Create diagnostic plots (titles removed)."""
    if fitted_model is None:
        return

    fitted_vals = fitted_model.fittedvalues
    residuals = fitted_model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')

    # Q-Q plot of residuals
    try:
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    except Exception:
        pass

    # Residuals by subject
    subject_means = df.groupby('subject', observed=False).apply(lambda x: residuals[x.index].mean(), include_groups=False)
    axes[1, 0].bar(range(len(subject_means)), subject_means.values, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('Mean Residual')

    # Residuals by window
    window_residuals = df.groupby('window', observed=False).apply(lambda x: residuals[x.index].mean(), include_groups=False)
    axes[1, 1].plot(window_residuals.index, window_residuals.values, 'o-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Window (30s)')
    axes[1, 1].set_ylabel('Mean Residual')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'lme_diagnostics.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted_p[sorted_indices[i]] = min(sorted_p[i] * n / (i + 1), adjusted_p[sorted_indices[i + 1]])
    return np.minimum(adjusted_p, 1.0).tolist()


def _compute_fdr_significant_segments(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05, alternative: str = 'two-sided') -> List[Tuple[float, float]]:
    if not SCIPY_AVAILABLE:
        return []
    from scipy import stats as scistats
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
    if not SCIPY_AVAILABLE:
        return {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    from scipy import stats as scistats
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
        return {'alpha': alpha, 'pvals': pvals.tolist(), 'pvals_adj': [], 'sig_mask': [], 'segments': []}
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
    return {'alpha': alpha, 'pvals': pvals.tolist(), 'pvals_adj': adj.tolist(), 'sig_mask': sig.tolist(), 'segments': segs}

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

    # Family (i): State effects
    for param in ['State[T.DMT]', 'State[T.DMT]:window_c']:
        if param in pvalues.index:
            families['State'].append(param)

    # Family (ii): Dose effects
    for param in ['Dose[T.High]', 'Dose[T.High]:window_c']:
        if param in pvalues.index:
            families['Dose'].append(param)

    # Family (iii): Interaction
    for param in ['State[T.DMT]:Dose[T.High]']:
        if param in pvalues.index:
            families['Interaction'].append(param)

    fdr_results: Dict[str, Dict] = {}
    for family_name, param_list in families.items():
        if not param_list:
            continue
        family_pvals = [pvalues[param] for param in param_list]
        fdr_pvals = benjamini_hochberg_correction(family_pvals)
        family_dict: Dict[str, Dict] = {}
        for i, param in enumerate(param_list):
            family_dict[param] = {
                'beta': float(params[param]),
                'se': float(stderr[param]),
                'p_raw': float(pvalues[param]),
                'p_fdr': float(fdr_pvals[i]),
                'ci_lower': float(conf_int.loc[param, 0]),
                'ci_upper': float(conf_int.loc[param, 1]),
            }
        fdr_results[family_name] = family_dict

    # Conditional contrasts
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(stderr['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'description': 'High - Low within RS',
        }
    if all(p in params.index for p in ['Dose[T.High]', 'State[T.DMT]:Dose[T.High]']):
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': float(params['State[T.DMT]:Dose[T.High]']),
            'se': float(stderr['State[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['State[T.DMT]:Dose[T.High]']),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }

    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    return results


def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, df: pd.DataFrame, output_dir: str) -> str:
    """Generate comprehensive analysis report (TXT)."""
    df_z = df[df['Scale'] == 'z']
    zscore_mode = "subject-level (all sessions)" if ZSCORE_BY_SUBJECT else "session-level (independent)"
    report_lines: List[str] = []

    report_lines.extend([
        "=" * 80,
        "LME ANALYSIS REPORT: SMNA AUC by 30-second Windows (first 9 minutes)",
        "=" * 80,
        "",
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df_z)} observations from {len(df_z['subject'].unique())} subjects",
        "",
        "DESIGN:",
        "  Within-subjects 2×2: State (RS vs DMT) × Dose (Low vs High)",
        "  Time windows: 18 thirty-second windows (0-540 seconds = 9 minutes)",
        "  Dependent variable: AUC of SMNA signal per 30-second window",
        f"  Z-scoring: {USE_RS_ZSCORE}",
        f"  Z-scoring mode: {zscore_mode}",
        "  Z-scoring baseline: Computed using RS + DMT data",
        f"  {'Subject parameters (mu, sigma) computed from all sessions (RS_high + DMT_high + RS_low + DMT_low)' if ZSCORE_BY_SUBJECT else 'Session parameters (mu, sigma) computed per session (RS + DMT)'}",
        "",
        "MODEL SPECIFICATION:",
        "  Fixed effects: AUC ~ State*Dose + window_c + State:window_c + Dose:window_c",
        "  Random effects: ~ 1 | subject",
        "  Where window_c = window - mean(window) [centered time]",
        f"  Scale: {'z-scored' if USE_RS_ZSCORE else 'absolute'}",
        "",
    ])

    if fitted_model is not None:
        report_lines.extend([
            "MODEL FIT STATISTICS:",
            f"  AIC: {diagnostics.get('aic', np.nan):.2f}",
            f"  BIC: {diagnostics.get('bic', np.nan):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', np.nan):.6f}",
            "",
        ])
        warnings_list = diagnostics.get('convergence_warnings', [])
        if warnings_list:
            report_lines.extend(["CONVERGENCE WARNINGS:"] + [f"  - {w}" for w in warnings_list] + [""])
        else:
            report_lines.append("Model converged without warnings\n")

    if 'fdr_families' in hypothesis_results:
        report_lines.extend([
            "HYPOTHESIS TESTING RESULTS (with BH-FDR correction):",
            "=" * 60,
            "",
        ])
        for family_name, family_results in hypothesis_results['fdr_families'].items():
            report_lines.extend([f"FAMILY {family_name.upper()}:", "-" * 30])
            for param, res in family_results.items():
                significance = "***" if res['p_fdr'] < 0.001 else "**" if res['p_fdr'] < 0.01 else "*" if res['p_fdr'] < 0.05 else ""
                report_lines.extend([
                    f"  {param}:",
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {significance}",
                    "",
                ])
            report_lines.append("")

    if 'conditional_contrasts' in hypothesis_results:
        report_lines.extend(["CONDITIONAL CONTRASTS:", "-" * 30])
        for _, res in hypothesis_results['conditional_contrasts'].items():
            significance = "***" if res['p_raw'] < 0.001 else "**" if res['p_raw'] < 0.01 else "*" if res['p_raw'] < 0.05 else ""
            report_lines.extend([
                f"  {res['description']}:",
                f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {significance}",
                "",
            ])

    # Data summary
    report_lines.extend(["", "DATA SUMMARY (Z-SCORED):", "-" * 30])
    summary_stats = df_z.groupby(['State', 'Dose'], observed=False)['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend(["Cell means (AUC by State × Dose):", str(summary_stats), ""]) 
    time_stats = df_z.groupby('window', observed=False)['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend(["Time trend (AUC by 30-second window):", str(time_stats), ""]) 
    
    # QC check: RS by dose
    report_lines.extend(['', 'QC CHECK: RS by Dose:', '-' * 30])
    rs_only = df_z[df_z['State'] == 'RS']
    rs_by_dose = rs_only.groupby('Dose', observed=False)['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend([str(rs_by_dose), ''])
    if ZSCORE_BY_SUBJECT:
        report_lines.extend(['Note: Z-scoring uses ALL sessions of subject (RS_high + DMT_high + RS_low + DMT_low),',
                             'so RS values reflect subject-level normalization. High and Low RS should be similar',
                             'since they share the same normalization parameters.', ''])
    else:
        report_lines.extend(['Note: Z-scoring uses entire session (RS+DMT) independently per session,', 
                             'so RS values reflect session-level normalization. High and Low RS may differ',
                             'since each session has its own normalization parameters.', ''])

    report_lines.extend(["", "=" * 80])

    report_path = os.path.join(output_dir, 'lme_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return report_path


def load_lme_results_from_report(report_path: str) -> Dict:
    """Parse the LME analysis report to extract coefficients for plotting."""
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
            param_name = line.rstrip(':').strip()
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
                        ci_text = ci_line.replace('95% CI:', '').replace('[', '').replace(']', '').strip()
                        ci_lower = float(ci_text.split(',')[0])
                        ci_upper = float(ci_text.split(',')[1])
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw = float([p for p in p_parts if 'p_raw' in p][0].split('=')[1].strip().split()[0])
                        p_fdr_text = [p for p in p_parts if 'p_FDR' in p][0].split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        significance = ''
                        if '***' in p_fdr_text:
                            significance = '***'
                        elif '**' in p_fdr_text:
                            significance = '**'
                        elif '*' in p_fdr_text:
                            significance = '*'
                    coefficients[param_name] = {
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
    param_order = [
        'State[T.DMT]',
        'Dose[T.High]',
        'State[T.DMT]:window_c',
        'Dose[T.High]:window_c',
        'State[T.DMT]:Dose[T.High]'
    ]
    param_labels = {
        'State[T.DMT]': 'State (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'State[T.DMT]:window_c': 'State × Time',
        'Dose[T.High]:window_c': 'Dose × Time',
        'State[T.DMT]:Dose[T.High]': 'State × Dose'
    }
    # Use EDA modality color (blue tones from tab20c) with distinct shades for visual distinction
    # Blue group from tab20c: indices 4-7 (darkest to lightest)
    family_colors = {
        'State': tab20c_colors[4],      # First blue gradient (darkest/most intense)
        'Dose': tab20c_colors[5],      # Second blue gradient (medium)
        'Interaction': tab20c_colors[6],  # Third blue gradient (lighter)
    }

    rows: List[Dict] = []
    for i, param in enumerate(param_order):
        if param in coefficients:
            c = coefficients[param]
            rows.append({
                'parameter': param,
                'label': param_labels.get(param, param),
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
                'color': family_colors.get(c['family'], '#666666'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No coefficient data to plot')
    return df


def create_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    # Use height that matches timeseries plots when assembled
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 0.45, DOUBLE_COL_WIDTH * 0.35))
    coef_df = coef_df.sort_values('order')
    y_positions = np.arange(len(coef_df))

    for _, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
        # Tamaño uniforme para todos los elementos
        linewidth = 4.0
        alpha = 1.0
        marker_size = 30  # Reduced from 60
        # Línea del CI
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], color=row['color'], linewidth=linewidth, alpha=alpha)
        # Círculo del coeficiente con borde del mismo color
        ax.scatter(row['beta'], y_pos, color=row['color'], s=marker_size, alpha=alpha, edgecolors=row['color'], linewidths=1.5, zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_df['label'], fontsize=FONT_SIZE_TICK_LABEL)
    ax.set_xlabel('Coefficient Estimate (β)\nwith 95% CI', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(axis='x', labelsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    # Add significance asterisks based on FDR-corrected p-values
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for _, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
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
            ax.text(x_pos, y_pos, sig_marker, fontsize=FONT_SIZE_TITLE, fontweight='bold',
                   va='center', ha='left', color=row['color'])
    plt.subplots_adjust(left=0.35, right=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    # Use appropriate scale based on USE_RS_ZSCORE
    scale_to_use = 'z' if USE_RS_ZSCORE else 'abs'
    df_scale = df[df['Scale'] == scale_to_use]
    grouped = df_scale.groupby(['window', 'State', 'Dose'], observed=False)['AUC']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['window', 'State', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['State'].astype(str) + '_' + stats_df['Dose'].astype(str)
    alpha = 1 - confidence
    from scipy import stats as scistats  # local import in case scipy missing at top
    t_critical = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1)
    stats_df['ci_lower'] = stats_df['mean'] - t_critical * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_critical * stats_df['se']
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    return stats_df


def create_marginal_means_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    conditions = stats_df['condition'].unique()
    for condition in sorted(conditions):
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('window')
        if len(cond_data) == 0:
            continue
        # Map colors based on condition
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

        # Convert window index to time in minutes for x-axis
        time_minutes = (cond_data['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
        ax.plot(time_minutes, cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('_', ' '), marker='o', markersize=5)
        ax.fill_between(time_minutes, cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)

    ax.set_xlabel('Time (minutes)')
    ylabel = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'
    ax.set_ylabel(ylabel)
    ticks = list(range(0, 10))  # 0-9 minutes
    ax.set_xticks(ticks)
    ax.set_xlim(-0.2, 9.2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_state_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    state_means = stats_df.groupby(['window', 'State']).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    # Rough SE aggregation
    state_se = stats_df.groupby(['window', 'State'])['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    state_means = state_means.merge(state_se, on=['window', 'State'], how='left')
    t_crit = 1.96
    state_means['ci_lower'] = state_means['mean'] - t_crit * state_means['se']
    state_means['ci_upper'] = state_means['mean'] + t_crit * state_means['se']

    fig, ax = plt.subplots(figsize=(10, 6))
    # Ensure legend order: DMT first, RS second
    for state, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        state_data = state_means[state_means['State'] == state].sort_values('window')
        # Convert window index to time in minutes for x-axis
        time_minutes = (state_data['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
        ax.plot(time_minutes, state_data['mean'], color=color, linewidth=3, label=f'{state}', marker='o', markersize=6)
        ax.fill_between(time_minutes, state_data['ci_lower'], state_data['ci_upper'], color=color, alpha=0.2)

    ax.set_xlabel('Time (minutes)')
    ylabel = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'
    ax.set_ylabel(ylabel)
    ticks = list(range(0, 10))  # 0-9 minutes
    ax.set_xticks(ticks)
    ax.set_xlim(-0.2, 9.2)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str, df_raw: Optional[pd.DataFrame] = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    ylabel = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'

    # RS panel (High above Low in legend)
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('window')
        # Convert window index to time in minutes for x-axis
        time_minutes = (cond_data['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
        ax1.plot(time_minutes, cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('RS_', ''), marker='o', markersize=4)
        ax1.fill_between(time_minutes, cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel(ylabel)
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    ticks = list(range(0, 10))  # 0-9 minutes
    ax1.set_xticks(ticks)
    ax1.set_xlim(-0.2, 9.2)
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)

    # DMT panel (High above Low in legend)
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('window')
        # Convert window index to time in minutes for x-axis
        time_minutes = (cond_data['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
        ax2.plot(time_minutes, cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('DMT_', ''), marker='o', markersize=4)
        ax2.fill_between(time_minutes, cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    ax2.set_xticks(ticks)
    ax2.set_xlim(-0.2, 9.2)
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)

    # Optional: add FDR-based shading if raw df provided
    if df_raw is not None:
        try:
            subjects = list(df_raw['subject'].cat.categories) if 'category' in str(df_raw['subject'].dtype) else list(df_raw['subject'].unique())
            n_time = N_WINDOWS
            # RS arrays
            H = np.full((len(subjects), n_time), np.nan, dtype=float)
            L = np.full((len(subjects), n_time), np.nan, dtype=float)
            for si, subj in enumerate(subjects):
                sdf = df_raw[df_raw['subject'] == subj]
                for window_idx in range(1, N_WINDOWS + 1):
                    row_h = sdf[(sdf['State'] == 'RS') & (sdf['Dose'] == 'High') & (sdf['window'] == window_idx)]['AUC']
                    row_l = sdf[(sdf['State'] == 'RS') & (sdf['Dose'] == 'Low') & (sdf['window'] == window_idx)]['AUC']
                    if len(row_h) == 1:
                        H[si, window_idx - 1] = float(row_h.iloc[0])
                    if len(row_l) == 1:
                        L[si, window_idx - 1] = float(row_l.iloc[0])
            x_grid = np.arange(1, N_WINDOWS + 1)
            segs = _compute_fdr_significant_segments(H, L, x_grid)
            for w0, w1 in segs:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
                t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
                ax1.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
            rs_res = _compute_fdr_results(H, L, x_grid)

            # DMT arrays
            H = np.full((len(subjects), n_time), np.nan, dtype=float)
            L = np.full((len(subjects), n_time), np.nan, dtype=float)
            for si, subj in enumerate(subjects):
                sdf = df_raw[df_raw['subject'] == subj]
                for window_idx in range(1, N_WINDOWS + 1):
                    row_h = sdf[(sdf['State'] == 'DMT') & (sdf['Dose'] == 'High') & (sdf['window'] == window_idx)]['AUC']
                    row_l = sdf[(sdf['State'] == 'DMT') & (sdf['Dose'] == 'Low') & (sdf['window'] == window_idx)]['AUC']
                    if len(row_h) == 1:
                        H[si, window_idx - 1] = float(row_h.iloc[0])
                    if len(row_l) == 1:
                        L[si, window_idx - 1] = float(row_l.iloc[0])
            segs = _compute_fdr_significant_segments(H, L, x_grid, alternative='greater')
            for w0, w1 in segs:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
                t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
                ax2.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
            dmt_res = _compute_fdr_results(H, L, x_grid, alternative='greater')

            # Write FDR report
            out_dir = os.path.dirname(os.path.dirname(output_path))
            lines: List[str] = [
                'FDR COMPARISON: SMNA High vs Low (Interaction panels)',
                f"Alpha = {rs_res.get('alpha', 0.05)}",
                ''
            ]
            def _sec(name: str, res: Dict):
                lines.append(f'PANEL {name}:')
                segs2 = res.get('segments', [])
                lines.append(f"  Significant window ranges (count={len(segs2)}):")
                if len(segs2) == 0:
                    lines.append('    - None')
                for (w0, w1) in segs2:
                    t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
                    t1 = w1 * WINDOW_SIZE_SEC / 60.0
                    lines.append(f"    - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
                p_adj = [v for v in res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
                if p_adj:
                    lines.append(f"  Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
                lines.append('')
            _sec('RS', rs_res)
            _sec('DMT', dmt_res)
            with open(os.path.join(out_dir, 'fdr_segments_smna_interaction.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


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


def create_combined_summary_plot(out_dir: str) -> Optional[str]:
    """SMNA combined summary using per-30-second-window AUC (first 9 minutes = 18 windows).
    
    Uses z-scored data if USE_RS_ZSCORE=True, else absolute AUC values.
    If ZSCORE_BY_SUBJECT=True: uses subject-level normalization.
    If ZSCORE_BY_SUBJECT=False: uses session-level normalization.

    Saves: results/eda/smna/plots/all_subs_smna.png
    """
    # Build per-subject per-window AUC matrices for RS and DMT (High/Low)
    H_RS, L_RS, H_DMT, L_DMT = [], [], [], []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            # DMT paths
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_smna(p_high)
            d_low = load_cvx_smna(p_low)
            # RS paths
            p_rsh = build_rs_cvx_path(subject, high_session)
            p_rsl = build_rs_cvx_path(subject, low_session)
            r_high = load_cvx_smna(p_rsh)
            r_low = load_cvx_smna(p_rsl)
            if None in (d_high, d_low, r_high, r_low):
                continue
            th, yh_abs = d_high; tl, yl_abs = d_low
            trh, yrh_abs = r_high; trl, yrl_abs = r_low
            
            # Apply z-scoring if enabled
            if USE_RS_ZSCORE:
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    yrh_z, yh_z, yrl_z, yl_z, diag = zscore_with_subject_baseline(
                        trh, yrh_abs, th, yh_abs,
                        trl, yrl_abs, tl, yl_abs
                    )
                    if not diag['scalable']:
                        continue
                    yh, yl, yrh, yrl = yh_z, yl_z, yrh_z, yrl_z
                else:
                    # Z-score each session independently
                    yrh_z, yh_z, diag_h = zscore_with_session_baseline(trh, yrh_abs, th, yh_abs)
                    yrl_z, yl_z, diag_l = zscore_with_session_baseline(trl, yrl_abs, tl, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        continue
                    yh, yl, yrh, yrl = yh_z, yl_z, yrh_z, yrl_z
            else:
                # Use absolute values
                yh, yl, yrh, yrl = yh_abs, yl_abs, yrh_abs, yrl_abs
            
            # Compute per-30-second-window AUC 1..18
            auc_dmt_h = [compute_auc_window(th, yh, m) for m in range(N_WINDOWS)]
            auc_dmt_l = [compute_auc_window(tl, yl, m) for m in range(N_WINDOWS)]
            auc_rs_h = [compute_auc_window(trh, yrh, m) for m in range(N_WINDOWS)]
            auc_rs_l = [compute_auc_window(trl, yrl, m) for m in range(N_WINDOWS)]
            if None in auc_dmt_h or None in auc_dmt_l or None in auc_rs_h or None in auc_rs_l:
                continue
            H_RS.append(np.array(auc_rs_h, dtype=float))
            L_RS.append(np.array(auc_rs_l, dtype=float))
            H_DMT.append(np.array(auc_dmt_h, dtype=float))
            L_DMT.append(np.array(auc_dmt_l, dtype=float))
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
    # Use double column width with height matching coefficient plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35), sharex=True, sharey=True)
    # RS panel
    rs_segs = _compute_fdr_significant_segments(H_RS, L_RS, x)
    for w0, w1 in rs_segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax1.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    # Convert window indices to time in minutes for x-axis
    time_minutes = (x - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
    l1 = ax1.plot(time_minutes, rs_mean_h, color=COLOR_RS_HIGH, lw=LINE_WIDTH, label='High dose (40mg)')[0]
    ax1.fill_between(time_minutes, rs_mean_h - rs_sem_h, rs_mean_h + rs_sem_h, color=COLOR_RS_HIGH, alpha=0.25)
    l2 = ax1.plot(time_minutes, rs_mean_l, color=COLOR_RS_LOW, lw=LINE_WIDTH, label='Low dose (20mg)')[0]
    ax1.fill_between(time_minutes, rs_mean_l - rs_sem_l, rs_mean_l + rs_sem_l, color=COLOR_RS_LOW, alpha=0.25)
    leg1 = ax1.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg1.get_frame().set_facecolor('white'); leg1.get_frame().set_alpha(0.9)
    ax1.set_xlabel('Time (minutes)', fontsize=FONT_SIZE_AXIS_LABEL)
    # Use blue color from tab20c for Electrodermal Activity (EDA/SMNA modality) - only first line colored
    ax1.text(-0.20, 0.5, 'Electrodermal Activity', transform=ax1.transAxes, 
             fontsize=FONT_SIZE_AXIS_LABEL, fontweight='bold', color=tab20c_colors[4],
             rotation=90, va='center', ha='center')
    ax1.text(-0.12, 0.5, 'SMNA (Z-scored)', transform=ax1.transAxes, 
             fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal', color='black', 
             rotation=90, va='center', ha='center')
    ax1.set_title('Resting State (RS)', fontweight='bold', fontsize=FONT_SIZE_TITLE)
    ax1.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)
    ax1.grid(True, which='major', axis='y', alpha=0.25); ax1.grid(False, which='major', axis='x')
    # DMT panel
    dmt_segs = _compute_fdr_significant_segments(H_DMT, L_DMT, x, alternative='greater')
    for w0, w1 in dmt_segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax2.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    l3 = ax2.plot(time_minutes, dmt_mean_h, color=COLOR_DMT_HIGH, lw=LINE_WIDTH, label='High dose (40mg)')[0]
    ax2.fill_between(time_minutes, dmt_mean_h - dmt_sem_h, dmt_mean_h + dmt_sem_h, color=COLOR_DMT_HIGH, alpha=0.25)
    l4 = ax2.plot(time_minutes, dmt_mean_l, color=COLOR_DMT_LOW, lw=LINE_WIDTH, label='Low dose (20mg)')[0]
    ax2.fill_between(time_minutes, dmt_mean_l - dmt_sem_l, dmt_mean_l + dmt_sem_l, color=COLOR_DMT_LOW, alpha=0.25)
    leg2 = ax2.legend([l3, l4], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg2.get_frame().set_facecolor('white'); leg2.get_frame().set_alpha(0.9)
    ax2.set_xlabel('Time (minutes)', fontsize=FONT_SIZE_AXIS_LABEL)
    ax2.set_title('DMT', fontweight='bold', fontsize=FONT_SIZE_TITLE)
    ax2.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)
    ax2.grid(True, which='major', axis='y', alpha=0.25); ax2.grid(False, which='major', axis='x')

    for ax in (ax1, ax2):
        ticks = list(range(0, 10))  # 0-9 minutes
        ax.set_xticks(ticks)
        ax.set_xlim(-0.2, 9.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_smna.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    try:
        lines: List[str] = [
            'FDR COMPARISON: SMNA AUC High vs Low (All Subjects, RS and DMT)',
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
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_smna.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path


def create_dmt_only_20min_plot(out_dir: str) -> Optional[str]:
    """SMNA DMT-only extended plot using per-30-second-window AUC (~19 minutes)."""
    limit_sec = 1150.0
    total_windows = int(np.floor(limit_sec / WINDOW_SIZE_SEC))  # ~38 windows
    H_list, L_list = [] , []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_smna(p_high)
            d_low = load_cvx_smna(p_low)
            if None in (d_high, d_low):
                continue
            th, yh_abs = d_high; tl, yl_abs = d_low
            
            # Apply z-scoring if enabled (need RS data for baseline)
            if USE_RS_ZSCORE:
                # Load RS data for z-scoring baseline
                p_rsh = build_rs_cvx_path(subject, high_session)
                p_rsl = build_rs_cvx_path(subject, low_session)
                r_high = load_cvx_smna(p_rsh)
                r_low = load_cvx_smna(p_rsl)
                if None in (r_high, r_low):
                    continue
                trh, yrh_abs = r_high
                trl, yrl_abs = r_low
                
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    _, yh_z, _, yl_z, diag = zscore_with_subject_baseline(
                        trh, yrh_abs, th, yh_abs,
                        trl, yrl_abs, tl, yl_abs
                    )
                    if not diag['scalable']:
                        continue
                    yh, yl = yh_z, yl_z
                else:
                    # Z-score each session independently
                    _, yh_z, diag_h = zscore_with_session_baseline(trh, yrh_abs, th, yh_abs)
                    _, yl_z, diag_l = zscore_with_session_baseline(trl, yrl_abs, tl, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        continue
                    yh, yl = yh_z, yl_z
            else:
                # Use absolute values
                yh, yl = yh_abs, yl_abs
            
            auc_h = [compute_auc_window(th, yh, m) for m in range(total_windows)]
            auc_l = [compute_auc_window(tl, yl, m) for m in range(total_windows)]
            if None in auc_h or None in auc_l:
                continue
            H_list.append(np.array(auc_h, dtype=float))
            L_list.append(np.array(auc_l, dtype=float))
        except Exception:
            continue
    if not (H_list and L_list):
        return None
    H = np.vstack(H_list); L = np.vstack(L_list)
    mean_h = np.nanmean(H, axis=0); mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    x = np.arange(1, total_windows + 1)
    # Convert window indices to time in minutes for x-axis
    time_minutes = (x - 0.5) * WINDOW_SIZE_SEC / 60.0  # Center of each window
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    segs = _compute_fdr_significant_segments(H, L, x)
    for w0, w1 in segs:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    l1 = ax.plot(time_minutes, mean_h, color=COLOR_DMT_HIGH, lw=2.0, label='High dose (40mg)')[0]
    ax.fill_between(time_minutes, mean_h - sem_h, mean_h + sem_h, color=COLOR_DMT_HIGH, alpha=0.25)
    l2 = ax.plot(time_minutes, mean_l, color=COLOR_DMT_LOW, lw=2.0, label='Low dose (20mg)')[0]
    ax.fill_between(time_minutes, mean_l - sem_l, mean_l + sem_l, color=COLOR_DMT_LOW, alpha=0.25)
    leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg.get_frame().set_facecolor('white'); leg.get_frame().set_alpha(0.9)
    ax.set_xlabel('Time (minutes)')
    # Use blue color from tab20c for Electrodermal Activity (EDA/SMNA modality) - only first line colored
    ax.text(-0.20, 0.5, 'Electrodermal Activity', transform=ax.transAxes, 
            fontsize=28, fontweight='bold', color=tab20c_colors[4],
            rotation=90, va='center', ha='center')
    ylabel_text = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'
    ax.text(-0.12, 0.5, ylabel_text, transform=ax.transAxes, 
            fontsize=28, fontweight='normal', color='black', 
            rotation=90, va='center', ha='center')
    ax.set_title('DMT', fontweight='bold')
    ax.grid(True, which='major', axis='y', alpha=0.25); ax.grid(False, which='major', axis='x')
    ticks = list(range(0, 20))  # 0-19 minutes
    ax.set_xticks(ticks); ax.set_xlim(-0.2, 19.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_dmt_smna.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    try:
        lines: List[str] = [
            'FDR COMPARISON: SMNA AUC High vs Low (DMT only)',
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
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_dmt_smna.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path

def create_stacked_subjects_plot(out_dir: str) -> Optional[str]:
    """Create a stacked per-subject figure (RS left, DMT right) using per-30-second-window SMNA AUC (1..18).

    Saves results/eda/smna/plots/stacked_subs_smna.png
    """
    rows: List[Dict] = []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            # DMT High/Low by session mapping
            high_session, low_session = determine_sessions(subject)
            p_dmt_h, p_dmt_l = build_cvx_paths(subject, high_session, low_session)
            dmt_h = load_cvx_smna(p_dmt_h)
            dmt_l = load_cvx_smna(p_dmt_l)
            if None in (dmt_h, dmt_l):
                continue
            th, yh_abs = dmt_h; tl, yl_abs = dmt_l

            # RS session1/session2, map to High/Low using recorded dose per session
            p_rs1 = build_rs_cvx_path(subject, 'session1')
            p_rs2 = build_rs_cvx_path(subject, 'session2')
            rs1 = load_cvx_smna(p_rs1)
            rs2 = load_cvx_smna(p_rs2)
            if None in (rs1, rs2):
                continue
            t1, y1_abs = rs1; t2, y2_abs = rs2
            
            # Apply z-scoring if enabled
            if USE_RS_ZSCORE:
                # Determine which RS session is high/low
                try:
                    dose_s1 = get_dosis_sujeto(subject, 1)
                except Exception:
                    dose_s1 = 'Alta'
                cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
                
                if cond1 == 'High':
                    # session1=high, session2=low
                    trh_abs, yrh_abs = t1, y1_abs
                    trl_abs, yrl_abs = t2, y2_abs
                else:
                    # session1=low, session2=high
                    trh_abs, yrh_abs = t2, y2_abs
                    trl_abs, yrl_abs = t1, y1_abs
                
                if ZSCORE_BY_SUBJECT:
                    # Z-score using ALL sessions of subject
                    yrh_z, yh_z, yrl_z, yl_z, diag = zscore_with_subject_baseline(
                        trh_abs, yrh_abs, th, yh_abs,
                        trl_abs, yrl_abs, tl, yl_abs
                    )
                    if not diag['scalable']:
                        continue
                    yh, yl = yh_z, yl_z
                    if cond1 == 'High':
                        y1, y2 = yrh_z, yrl_z
                    else:
                        y1, y2 = yrl_z, yrh_z
                else:
                    # Z-score each session independently
                    yrh_z, yh_z, diag_h = zscore_with_session_baseline(trh_abs, yrh_abs, th, yh_abs)
                    yrl_z, yl_z, diag_l = zscore_with_session_baseline(trl_abs, yrl_abs, tl, yl_abs)
                    if not (diag_h['scalable'] and diag_l['scalable']):
                        continue
                    yh, yl = yh_z, yl_z
                    if cond1 == 'High':
                        y1, y2 = yrh_z, yrl_z
                    else:
                        y1, y2 = yrl_z, yrh_z
            else:
                # Use absolute values
                yh, yl, y1, y2 = yh_abs, yl_abs, y1_abs, y2_abs
            
            # Compute AUC from z-scored or absolute data
            auc_dmt_h = [compute_auc_window(th, yh, m) for m in range(N_WINDOWS)]
            auc_dmt_l = [compute_auc_window(tl, yl, m) for m in range(N_WINDOWS)]
            auc_rs1 = [compute_auc_window(t1, y1, m) for m in range(N_WINDOWS)]
            auc_rs2 = [compute_auc_window(t2, y2, m) for m in range(N_WINDOWS)]
            
            if None in auc_dmt_h or None in auc_dmt_l or None in auc_rs1 or None in auc_rs2:
                continue

            try:
                dose_s1 = get_dosis_sujeto(subject, 1)
            except Exception:
                dose_s1 = 'Alta'
            cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
            cond2 = 'Low' if cond1 == 'High' else 'High'
            if cond1 == 'High':
                auc_rs_h, auc_rs_l = auc_rs1, auc_rs2
            else:
                auc_rs_h, auc_rs_l = auc_rs2, auc_rs1

            rows.append({
                'subject': subject,
                'windows': list(range(1, N_WINDOWS + 1)),
                'rs_high': np.asarray(auc_rs_h, dtype=float),
                'rs_low': np.asarray(auc_rs_l, dtype=float),
                'dmt_high': np.asarray(auc_dmt_h, dtype=float),
                'dmt_low': np.asarray(auc_dmt_l, dtype=float),
            })
        except Exception:
            continue

    if not rows:
        return None

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

    time_ticks = list(range(0, 10))  # 0-9 minutes

    from matplotlib.lines import Line2D

    for i, row in enumerate(rows):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]

        # Convert window indices to time in minutes for x-axis
        time_minutes = (np.array(row['windows']) - 0.5) * WINDOW_SIZE_SEC / 60.0

        # RS panel
        ax_rs.plot(time_minutes, row['rs_high'], color=COLOR_RS_HIGH, lw=1.8, marker='o', markersize=4)
        ax_rs.plot(time_minutes, row['rs_low'], color=COLOR_RS_LOW, lw=1.8, marker='o', markersize=4)
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ylabel = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'
        ax_rs.set_ylabel(r'$\mathbf{Electrodermal\ Activity}$' + f'\n{ylabel}', fontsize=12)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(-0.2, 9.2)
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=COLOR_RS_HIGH, lw=1.8, marker='o', markersize=4, label='High dose (40mg)'),
            Line2D([0], [0], color=COLOR_RS_LOW, lw=1.8, marker='o', markersize=4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)

        # DMT panel
        ax_dmt.plot(time_minutes, row['dmt_high'], color=COLOR_DMT_HIGH, lw=1.8, marker='o', markersize=4)
        ax_dmt.plot(time_minutes, row['dmt_low'], color=COLOR_DMT_LOW, lw=1.8, marker='o', markersize=4)
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ylabel = 'SMNA (Z-scored)' if USE_RS_ZSCORE else 'SMNA AUC'
        ax_dmt.set_ylabel(r'$\mathbf{Electrodermal\ Activity}$' + f'\n{ylabel}', fontsize=12)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(-0.2, 9.2)
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=COLOR_DMT_HIGH, lw=1.8, marker='o', markersize=4, label='High dose (40mg)'),
            Line2D([0], [0], color=COLOR_DMT_LOW, lw=1.8, marker='o', markersize=4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)

        ax_rs.set_xticks(time_ticks)
        ax_dmt.set_xticks(time_ticks)

    fig.tight_layout(pad=2.0)

    # Subject labels centered between columns
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

    out_path = os.path.join(out_dir, 'plots', 'stacked_subs_smna.png')
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    return out_path

def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    table_data = coef_df[['label', 'beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'significance', 'family']].copy()
    numeric_cols = ['beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr']
    table_data[numeric_cols] = table_data[numeric_cols].round(4)
    table_data['interpretation'] = table_data.apply(lambda r: (
        f"{'Significant' if r['p_fdr'] < 0.05 else 'Non-significant'} "
        f"{'increase' if r['beta'] > 0 else 'decrease'} in SMNA AUC"), axis=1)
    table_data.to_csv(output_path, index=False)


def create_model_summary_txt(diagnostics: Dict, coef_df: pd.DataFrame, output_path: str) -> None:
    lines: List[str] = [
        "LME MODEL SUMMARY",
        "=" * 60,
        "",
        "Fixed Effects Formula:",
        "AUC ~ State*Dose + window_c + State:window_c + Dose:window_c",
        "",
        "Random Effects: ~ 1 | subject",
        "",
        "Model Fit Statistics:",
        f"AIC: {diagnostics.get('aic', np.nan):.2f}",
        f"BIC: {diagnostics.get('bic', np.nan):.2f}",
        f"Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
        f"N observations: {diagnostics.get('n_obs', 'N/A')}",
        f"N subjects: {diagnostics.get('n_groups', 'N/A')}",
        "",
        "Significant Fixed Effects (p_FDR < 0.05):",
    ]
    significant = coef_df[coef_df['p_fdr'] < 0.05]
    if len(significant) == 0:
        lines.append("• None")
    else:
        for _, row in significant.iterrows():
            lines.append(f"• {row['label']}: β = {row['beta']:.3f} {row['significance']}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_captions_file(output_dir: str, n_subjects: int) -> None:
    captions = [
        "Figure: LME Coefficients (SMNA AUC)\n\n"
        "Point estimates (β) and 95% CIs for fixed effects from the mixed model. "
        "Reference line at zero aids interpretation. Significant effects are visually emphasized.",
        "",
        "Figure: Marginal Means Over Time (RS vs DMT × High vs Low)\n\n"
        "Group-level mean ± 95% CI of SMNA AUC across the first 9 minutes for each condition (RS Low/High, DMT Low/High). "
        "Legends indicate dose levels; shading shows uncertainty.",
        "",
        "Figure: Main State Effect Over Time\n\n"
        "Mean ± 95% CI for RS and DMT (averaged across dose) across minutes 0–8. "
        "Illustrates overall state separation and temporal trend.",
        "",
        "Figure: State × Dose Interaction (Panels)\n\n"
        "Left: RS Low vs High; Right: DMT Low vs High. Lines show mean ± 95% CI across minutes 0–8. "
        "Highlights how dose effects differ between states.",
    ]
    path = os.path.join(output_dir, 'captions_smna.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def prepare_extended_long_data_smna() -> pd.DataFrame:
    """Build long-format per-30-second window SMNA AUC table for extended DMT (~19 min = 38 windows).
    
    This function exports DMT-only data for extended time range, used by composite analysis.
    Uses same z-scoring approach as prepare_long_data().
    """
    limit_sec = 1150.0
    total_windows = int(np.floor(limit_sec / WINDOW_SIZE_SEC))  # ~38 windows
    rows: List[Dict] = []
    qc_log: List[str] = []
    
    for subject in SUJETOS_VALIDADOS_EDA:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs_high_path = build_rs_cvx_path(subject, high_session)
        rs_low_path = build_rs_cvx_path(subject, low_session)

        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs_high = load_cvx_smna(rs_high_path)
        rs_low = load_cvx_smna(rs_low_path)
        
        if None in (dmt_high, dmt_low, rs_high, rs_low):
            continue

        t_dmt_high, smna_dmt_high_abs = dmt_high
        t_dmt_low, smna_dmt_low_abs = dmt_low
        t_rs_high, smna_rs_high_abs = rs_high
        t_rs_low, smna_rs_low_abs = rs_low

        if USE_RS_ZSCORE:
            if ZSCORE_BY_SUBJECT:
                smna_rs_high_z, smna_dmt_high_z, smna_rs_low_z, smna_dmt_low_z, diag = zscore_with_subject_baseline(
                    t_rs_high, smna_rs_high_abs, t_dmt_high, smna_dmt_high_abs,
                    t_rs_low, smna_rs_low_abs, t_dmt_low, smna_dmt_low_abs
                )
                if not diag['scalable']:
                    qc_log.append(f"{subject}: Not scalable: {diag['reason']}")
                    continue
                smna_dmt_high_z_use, smna_dmt_low_z_use = smna_dmt_high_z, smna_dmt_low_z
            else:
                smna_rs_high_z, smna_dmt_high_z, diag_high = zscore_with_session_baseline(
                    t_rs_high, smna_rs_high_abs, t_dmt_high, smna_dmt_high_abs
                )
                smna_rs_low_z, smna_dmt_low_z, diag_low = zscore_with_session_baseline(
                    t_rs_low, smna_rs_low_abs, t_dmt_low, smna_dmt_low_abs
                )
                if not (diag_high['scalable'] and diag_low['scalable']):
                    continue
                smna_dmt_high_z_use, smna_dmt_low_z_use = smna_dmt_high_z, smna_dmt_low_z
            
            # Process extended windows (DMT only)
            for window_idx in range(total_windows):
                window_label = window_idx + 1
                auc_dmt_h_z = compute_auc_window(t_dmt_high, smna_dmt_high_z_use, window_idx)
                auc_dmt_l_z = compute_auc_window(t_dmt_low, smna_dmt_low_z_use, window_idx)
                
                if auc_dmt_h_z is not None and auc_dmt_l_z is not None:
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_h_z, 'Scale': 'z'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_l_z, 'Scale': 'z'},
                    ])
                    if EXPORT_ABSOLUTE_SCALE:
                        auc_dmt_h_abs = compute_auc_window(t_dmt_high, smna_dmt_high_abs, window_idx)
                        auc_dmt_l_abs = compute_auc_window(t_dmt_low, smna_dmt_low_abs, window_idx)
                        if auc_dmt_h_abs is not None and auc_dmt_l_abs is not None:
                            rows.extend([
                                {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_h_abs, 'Scale': 'abs'},
                                {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_l_abs, 'Scale': 'abs'},
                            ])
        else:
            for window_idx in range(total_windows):
                window_label = window_idx + 1
                auc_dmt_h_abs = compute_auc_window(t_dmt_high, smna_dmt_high_abs, window_idx)
                auc_dmt_l_abs = compute_auc_window(t_dmt_low, smna_dmt_low_abs, window_idx)
                if auc_dmt_h_abs is not None and auc_dmt_l_abs is not None:
                    rows.extend([
                        {'subject': subject, 'session': high_session, 'window': window_label, 'State': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_h_abs, 'Scale': 'abs'},
                        {'subject': subject, 'session': low_session, 'window': window_label, 'State': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_l_abs, 'Scale': 'abs'},
                    ])

    if not rows:
        raise ValueError('No valid extended SMNA data found!')
    
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
    out_dir = os.path.join('results', 'eda', 'smna')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Data preparation
        scale_mode = "z-scoring (session baseline)" if USE_RS_ZSCORE else "absolute AUC values"
        print(f"Preparing long-format data with {scale_mode}...")
        df = prepare_long_data()
        
        # Save all scales
        df.to_csv(os.path.join(out_dir, 'smna_auc_long_data_all_scales.csv'), index=False)
        print(f"  ✓ Saved all scales: {len(df)} rows")
        
        # Save primary scale data separately
        scale_to_use = 'z' if USE_RS_ZSCORE else 'abs'
        df_primary = df[df['Scale'] == scale_to_use]
        scale_filename = 'smna_auc_long_data_z.csv' if USE_RS_ZSCORE else 'smna_auc_long_data_abs.csv'
        df_primary.to_csv(os.path.join(out_dir, scale_filename), index=False)
        scale_desc = "z-scored" if USE_RS_ZSCORE else "absolute"
        print(f"  ✓ Saved {scale_desc} data: {len(df_primary)} rows from {len(df_primary['subject'].unique())} subjects")
        
        # Export extended DMT data (~19 minutes)
        print("Preparing extended DMT data (~19 minutes)...")
        df_extended = prepare_extended_long_data_smna()
        df_extended.to_csv(os.path.join(out_dir, 'smna_extended_dmt_all_scales.csv'), index=False)
        df_ext_primary = df_extended[df_extended['Scale'] == scale_to_use]
        ext_filename = 'smna_extended_dmt_z.csv' if USE_RS_ZSCORE else 'smna_extended_dmt_abs.csv'
        df_ext_primary.to_csv(os.path.join(out_dir, ext_filename), index=False)
        print(f"  ✓ Saved extended DMT data: {len(df_ext_primary)} rows from {len(df_ext_primary['subject'].unique())} subjects")
        
        # QC check: RS by dose
        rs_qc = df_primary[df_primary['State'] == 'RS'].groupby('Dose', observed=False)['AUC'].agg(['count', 'mean', 'std']).round(4)
        with open(os.path.join(out_dir, 'qc_rs_by_dose.txt'), 'w') as f:
            f.write('QC CHECK: RS AUC by Dose\n')
            f.write('=' * 60 + '\n\n')
            f.write(f'Scale: {scale_desc}\n\n')
            f.write(str(rs_qc) + '\n\n')
            if USE_RS_ZSCORE:
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
                f.write('Note: Using absolute SMNA AUC values without normalization.\n')
        print(f"  ✓ QC check saved")

        # LME model (uses appropriate scale)
        print(f"Fitting LME model on {scale_desc} data...")
        fitted_model, diagnostics = fit_lme_model(df)
        plot_model_diagnostics(fitted_model, df_primary, plots_dir)

        # Hypotheses + report
        hypothesis_results = hypothesis_testing_with_fdr(fitted_model)
        report_path = generate_report(fitted_model, diagnostics, hypothesis_results, df, out_dir)

        # Coefficients (parse from report for consistency)
        coefficients = load_lme_results_from_report(report_path)
        coef_df = prepare_coefficient_data(coefficients)

        coef_plot_path = os.path.join(plots_dir, 'lme_coefficient_plot.png')
        create_coefficient_plot(coef_df, coef_plot_path)

        # Effect sizes table
        table_path = os.path.join(plots_dir, 'effect_sizes_table.csv')
        create_effect_sizes_table(coef_df, table_path)

        # Summary statistics
        stats_df = compute_empirical_means_and_ci(df)
        summary_path = os.path.join(plots_dir, 'summary_statistics.csv')
        overall_means = stats_df.groupby('condition').agg({
            'mean': 'mean',
            'se': lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1)),
            'n': 'mean'
        }).round(4)
        overall_means['ci_lower'] = overall_means['mean'] - 1.96 * overall_means['se']
        overall_means['ci_upper'] = overall_means['mean'] + 1.96 * overall_means['se']
        overall_means.to_csv(summary_path)

        # Text model summary (instead of PNG)
        model_summary_txt = os.path.join(out_dir, 'model_summary.txt')
        create_model_summary_txt(diagnostics, coef_df, model_summary_txt)

        # Combined summary (first 9 min) and DMT-only extended
        create_combined_summary_plot(out_dir)
        create_dmt_only_20min_plot(out_dir)

        # Stacked per-subject (per-minute AUC)
        create_stacked_subjects_plot(out_dir)

        # Captions
        generate_captions_file(out_dir, len(df['subject'].unique()))

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)


