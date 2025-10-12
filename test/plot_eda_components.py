# -*- coding: utf-8 -*-
"""
Plot per-subject EDA component analysis: SCR (phasic) and SCL (tonic) comparison.

This script analyzes both components from CVX decomposition:
- SCR (Skin Conductance Response): phasic component - transient responses
- SCL (Skin Conductance Level): tonic component - baseline conductance level

For each subject in SUJETOS_VALIDADOS_EDA, this script:
- Finds the correct session for high vs low using get_dosis_sujeto
- Loads CVX decomposition CSVs (…_cvx_decomposition.csv)
- Extracts the specified component (SCR or SCL) and time
- Trims both recordings to the first 10 minutes
- Computes mean value of the component
- Plots the component with mean levels and SEM shading
- Saves to test/eda/scr (SCR) or test/eda/scl (SCL)

Usage:
  python test/plot_eda_components.py --component SCR
  python test/plot_eda_components.py --component SCL
"""

import os
import sys
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
try:
    import seaborn as sns  # optional, for nicer boxplots
except Exception:
    sns = None

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_EDA,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)
# ANOVA analysis removed - no longer needed

#############################
# Plot aesthetics (paper-ready minimal style)
#############################
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.titlepad': 8.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Component configuration
COMPONENT_CONFIG = {
    'SCR': {
        'name': 'SCR',
        'long_name': 'Skin Conductance Response',
        'description': 'phasic component',
        'output_dir': 'scr',
        'units': 'SCR (μS)',
        'csv_column': 'EDR',  # Column name in CSV files
        'colors': {'high': '#555555', 'low': '#2A9FD6'}
    },
    'SCL': {
        'name': 'SCL', 
        'long_name': 'Skin Conductance Level',
        'description': 'tonic component (baseline-corrected)',
        'output_dir': 'scl',
        'units': 'Δ SCL (μS)',
        'csv_column': 'EDL',  # Column name in CSV files
        'colors': {'high': '#8B4513', 'low': '#32CD32'}  # Brown/Green for tonic
    }
}


def _fmt_mmss(x, pos):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


def _beautify_axes(ax, title=None, xlabel=None, ylabel=None, time_formatter=True):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if time_formatter:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')


# Boxplot 2x2 and ANOVA functions removed - analysis no longer needed


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if dose_s1 == 'Alta':
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


def load_cvx_component(csv_path: str, component: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load time and component arrays from CVX decomposition CSV.

    Args:
        csv_path: Path to CVX decomposition CSV
        component: 'SCR' or 'SCL'
        
    Returns (t, component_data) or None if missing/invalid.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  Missing CVX file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Failed to read CVX file {os.path.basename(csv_path)}: {e}")
        return None

    # Get the CSV column name for this component
    csv_column = COMPONENT_CONFIG[component]['csv_column']
    
    if csv_column not in df.columns:
        print(f"⚠️  {csv_column} column not found in {os.path.basename(csv_path)}")
        return None

    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)

    component_data = pd.to_numeric(df[csv_column], errors='coerce').to_numpy()
    return t, component_data


def compute_baseline_scl(t: np.ndarray, y: np.ndarray) -> float:
    """Compute baseline SCL value from the first second of recording.
    
    Args:
        t: Time array
        y: SCL signal array
        
    Returns:
        Mean SCL value in the first second (0-1s)
    """
    mask_first_sec = (t >= 0.0) & (t < 1.0)
    if not np.any(mask_first_sec):
        # If no data in first second, use first available non-NaN sample
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            return 0.0
        return float(y[valid_mask][0])
    
    y_first_sec = y[mask_first_sec]
    # Use nanmean to handle potential NaN values in baseline period
    baseline = np.nanmean(y_first_sec)
    return float(baseline) if not np.isnan(baseline) else 0.0


def apply_scl_baseline_correction(t: np.ndarray, y: np.ndarray, component: str) -> np.ndarray:
    """Apply baseline correction for SCL signals.
    
    Args:
        t: Time array
        y: Signal array
        component: 'SCR' or 'SCL'
        
    Returns:
        Original signal for SCR, baseline-corrected signal for SCL
    """
    if component == 'SCL':
        baseline = compute_baseline_scl(t, y)
        return y - baseline
    else:
        return y


def compute_mean_value(y: np.ndarray, component: str = 'SCR', baseline_value: Optional[float] = None) -> float:
    """Compute mean value of component signal.
    
    Args:
        y: Signal array (already baseline-corrected for SCL)
        component: 'SCR' or 'SCL' 
        baseline_value: Deprecated, kept for compatibility
    """
    # Use nanmean to handle NaN values correctly
    mean_val = np.nanmean(y)
    return float(mean_val) if not np.isnan(mean_val) else 0.0


def moving_average(x: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return x
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(x, kernel, mode='same')


def compute_envelope(y: np.ndarray, sr: float, window_sec: float = 1.0) -> np.ndarray:
    """Return a smoothed envelope for y using moving-average."""
    window_samples = max(1, int(round(sr * window_sec)))
    return moving_average(y, window_samples)


def validate_sufficient_data(t: np.ndarray, y: np.ndarray, duration_sec: float, min_valid_ratio: float = 0.5) -> bool:
    """Validate that there's sufficient valid (non-NaN) data for the specified duration.
    
    Args:
        t: Time array
        y: Signal array
        duration_sec: Required duration in seconds
        min_valid_ratio: Minimum ratio of valid samples required (0.5 = 50%)
        
    Returns:
        True if sufficient valid data exists
    """
    if len(t) == 0 or len(y) == 0:
        return False
    
    # Find data within the required duration
    mask_duration = (t >= 0.0) & (t <= duration_sec)
    if not np.any(mask_duration):
        return False
    
    y_duration = y[mask_duration]
    valid_samples = np.sum(~np.isnan(y_duration))
    total_samples = len(y_duration)
    
    if total_samples == 0:
        return False
    
    valid_ratio = valid_samples / total_samples
    return valid_ratio >= min_valid_ratio


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linearly resample y(t) to t_grid, preserving NaNs."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    
    # Find valid (non-NaN) data points
    valid_mask = ~np.isnan(y)
    if not np.any(valid_mask):
        return np.full_like(t_grid, np.nan, dtype=float)
    
    t_valid = t[valid_mask]
    y_valid = y[valid_mask]
    
    # Only interpolate within the valid data range
    yg = np.full_like(t_grid, np.nan, dtype=float)
    valid_range_mask = (t_grid >= t_valid[0]) & (t_grid <= t_valid[-1])
    
    if np.any(valid_range_mask):
        yg[valid_range_mask] = np.interp(t_grid[valid_range_mask], t_valid, y_valid)
    
    return yg


def _last_valid_timestamp_mmss(t: np.ndarray, y: np.ndarray) -> Optional[str]:
    """Return mm:ss string for the last non-NaN sample; None if no valid sample."""
    if t is None or y is None or len(t) == 0 or len(y) == 0:
        return None
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid_mask = (~np.isnan(y)) & np.isfinite(t)
    if not np.any(valid_mask):
        return None
    last_t = float(t[valid_mask][-1])
    if last_t < 0.0:
        last_t = 0.0
    minutes = int(last_t // 60)
    seconds = int(last_t % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _plot_dmt_only_20min(out_path: str, component: str) -> bool:
    """Create a DMT-only plot for 20 minutes (both SCR and SCL).
    Shows High vs Low dose for DMT task across the full 20-minute recording.
    """
        
    config = COMPONENT_CONFIG[component]
    
    # Extended grid: 0..1150s (19:10), 2 Hz for readability
    t_grid = np.arange(0.0, 1150.0, 0.5)
    
    high_curves = []
    low_curves = []
    
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_component(p_high, component)
            d_low = load_cvx_component(p_low, component)
            if None in (d_high, d_low):
                continue
            th, yh = d_high; tl, yl = d_low

            # Apply baseline correction for SCL
            yh = apply_scl_baseline_correction(th, yh, component)
            yl = apply_scl_baseline_correction(tl, yl, component)
            
            # Trim to 19 minutes 10 seconds
            mask_h = (th >= 0.0) & (th < 1150.0)
            mask_l = (tl >= 0.0) & (tl < 1150.0)
            if not (np.any(mask_h) and np.any(mask_l)):
                continue
            yh = yh[mask_h]; th = th[mask_h]
            yl = yl[mask_l]; tl = tl[mask_l]

            # Resample to common grid
            yhg = _resample_to_grid(th, yh, t_grid)
            ylg = _resample_to_grid(tl, yl, t_grid)
            high_curves.append(yhg)
            low_curves.append(ylg)
        except Exception:
            continue

    if not (high_curves and low_curves):
        return False

    H = np.vstack(high_curves)
    L = np.vstack(low_curves)
    # Mean ± SEM
    mean_h = np.nanmean(H, axis=0)
    mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    # Create single plot for DMT only
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Fixed colors: DMT High red, DMT Low blue
    c_dmt_high, c_dmt_low = 'tab:red', 'tab:blue'
    
    # Plot DMT data
    line_h = ax.plot(t_grid, mean_h, color=c_dmt_high, lw=2.0, marker=None, label='High')[0]
    ax.fill_between(t_grid, mean_h - sem_h, mean_h + sem_h, color=c_dmt_high, alpha=0.25)
    line_l = ax.plot(t_grid, mean_l, color=c_dmt_low, lw=2.0, marker=None, label='Low')[0]
    ax.fill_between(t_grid, mean_l - sem_l, mean_l + sem_l, color=c_dmt_low, alpha=0.25)
    
    # Legend with correct colors - High first, then Low
    legend = ax.legend([line_h, line_l], ['High', 'Low'], loc='upper right', 
                      frameon=True, fancybox=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax, title=None, xlabel='Time (minutes)', ylabel=config['units'])
    ax.set_title('DMT', fontweight='bold')
    
    # X ticks every minute from 0:00 to 19:00
    minute_ticks = np.arange(0.0, 1141.0, 60.0)
    ax.set_xticks(minute_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.set_xlim(0.0, 1150.0)
    
    # Component-specific Y-axis settings
    if component == 'SCR':
        ax.set_ylim(-0.05, 0.25)
        ax.set_yticks(np.arange(0.00, 0.26, 0.05))
    else:  # SCL
        ax.set_ylim(-2.0, 2.0)
        ax.set_yticks(np.arange(-1.5, 2.0, 0.5))
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return True


def _plot_combined_summary(out_path: str, component: str) -> bool:
    """Create a combined summary plot with DMT (left) and RS (right) subplots."""
    config = COMPONENT_CONFIG[component]
    
    # Common grid: 0..550s (9:10), 2 Hz for readability
    t_grid = np.arange(0.0, 550.0, 0.5)
    
    # Collect data for both tasks
    task_data = {}
    
    for kind in ['DMT', 'RS']:
        high_curves = []
        low_curves = []
        
        for subject in SUJETOS_VALIDADOS_EDA:
            try:
                if kind == 'DMT':
                    high_session, low_session = determine_sessions(subject)
                    p_high, p_low = build_cvx_paths(subject, high_session, low_session)
                    d_high = load_cvx_component(p_high, component)
                    d_low = load_cvx_component(p_low, component)
                    if None in (d_high, d_low):
                        continue
                    th, yh = d_high; tl, yl = d_low
                else:  # RS
                    p_r1 = build_rs_cvx_path(subject, 'session1')
                    p_r2 = build_rs_cvx_path(subject, 'session2')
                    r1 = load_cvx_component(p_r1, component)
                    r2 = load_cvx_component(p_r2, component)
                    if None in (r1, r2):
                        continue
                    tr1, yr1 = r1; tr2, yr2 = r2
                    dose_s1 = get_dosis_sujeto(subject, 1)
                    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
                        th, yh = tr1, yr1  # high
                        tl, yl = tr2, yr2  # low
                    else:
                        th, yh = tr2, yr2
                        tl, yl = tr1, yr1

                # Apply baseline correction for SCL
                yh = apply_scl_baseline_correction(th, yh, component)
                yl = apply_scl_baseline_correction(tl, yl, component)
                
                # Validate sufficient data for 9:10
                if not (validate_sufficient_data(th, yh, 550.0) and validate_sufficient_data(tl, yl, 550.0)):
                    continue
                
                # Trim to 9:10
                mask_h = (th >= 0.0) & (th < 550.0)
                mask_l = (tl >= 0.0) & (tl < 550.0)
                yh = yh[mask_h]; th = th[mask_h]
                yl = yl[mask_l]; tl = tl[mask_l]

                # Resample to common grid
                yhg = _resample_to_grid(th, yh, t_grid)
                ylg = _resample_to_grid(tl, yl, t_grid)
                high_curves.append(yhg)
                low_curves.append(ylg)
            except Exception:
                continue

        if high_curves and low_curves:
            H = np.vstack(high_curves)
            L = np.vstack(low_curves)
            # Mean ± SEM
            mean_h = np.nanmean(H, axis=0)
            mean_l = np.nanmean(L, axis=0)
            sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
            sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])
            
            task_data[kind] = {
                'mean_h': mean_h, 'mean_l': mean_l,
                'sem_h': sem_h, 'sem_l': sem_l
            }
        else:
            return False

    if len(task_data) != 2:
        return False

    # Create combined plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    # Fixed colors per task
    c_dmt_high, c_dmt_low = 'tab:red', 'tab:blue'
    c_rs_high, c_rs_low = 'tab:green', 'tab:purple'

    # RS subplot (left)
    rs_data = task_data['RS']
    line_h1 = ax1.plot(t_grid, rs_data['mean_h'], color=c_rs_high, lw=2.0, marker=None, label='High')[0]
    ax1.fill_between(t_grid, rs_data['mean_h'] - rs_data['sem_h'], 
                     rs_data['mean_h'] + rs_data['sem_h'], color=c_rs_high, alpha=0.25)
    line_l1 = ax1.plot(t_grid, rs_data['mean_l'], color=c_rs_low, lw=2.0, marker=None, label='Low')[0]
    ax1.fill_between(t_grid, rs_data['mean_l'] - rs_data['sem_l'], 
                     rs_data['mean_l'] + rs_data['sem_l'], color=c_rs_low, alpha=0.25)

    legend1 = ax1.legend([line_h1, line_l1], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=False)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)

    _beautify_axes(ax1, title=None, xlabel='Time (minutes)', ylabel=config['units'])
    ax1.set_title('Resting State (RS)', fontweight='bold')

    # DMT subplot (right)
    dmt_data = task_data['DMT']
    line_h2 = ax2.plot(t_grid, dmt_data['mean_h'], color=c_dmt_high, lw=2.0, marker=None, label='High')[0]
    ax2.fill_between(t_grid, dmt_data['mean_h'] - dmt_data['sem_h'], 
                     dmt_data['mean_h'] + dmt_data['sem_h'], color=c_dmt_high, alpha=0.25)
    line_l2 = ax2.plot(t_grid, dmt_data['mean_l'], color=c_dmt_low, lw=2.0, marker=None, label='Low')[0]
    ax2.fill_between(t_grid, dmt_data['mean_l'] - dmt_data['sem_l'], 
                     dmt_data['mean_l'] + dmt_data['sem_l'], color=c_dmt_low, alpha=0.25)

    legend2 = ax2.legend([line_h2, line_l2], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=False)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)

    _beautify_axes(ax2, title=None, xlabel='Time (minutes)', ylabel=config['units'])
    ax2.set_title('DMT', fontweight='bold')

    # X ticks at each minute (0:00..9:00)
    minute_ticks = np.arange(0.0, 541.0, 60.0)
    for ax in (ax1, ax2):
        ax.set_xticks(minute_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
        ax.set_xlim(0.0, 550.0)
        # Component-specific Y-axis settings
        if component == 'SCR':
            ax.set_ylim(-0.05, 0.25)
            ax.set_yticks(np.arange(0.00, 0.26, 0.05))
        else:  # SCL
            ax.set_ylim(-2.0, 2.0)
            ax.set_yticks(np.arange(-1.5, 2.0, 0.5))

    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return True


def _plot_all_subjects_stacked(out_path: str, component: str) -> bool:
    """Create a stacked figure (supplementary): one row per subject, RS left, DMT right.

    - X: 0:00–9:10 with ticks every minute (labels mm:ss), xlabel "Time (minutes)" on bottom row
    - Y: {units} with fixed range [-5, 5], ylabel on both columns
    - Colors fixed: RS High green, RS Low purple; DMT High red, DMT Low blue
    - Each row annotated with subject code in bold
    """
    config = COMPONENT_CONFIG[component]
    units = config['units']
    limit_sec = 550.0

    # Collect valid subjects and their trimmed data
    rows = []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            # DMT
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_component(p_high, component)
            d_low = load_cvx_component(p_low, component)
            if None in (d_high, d_low):
                continue
            th, yh = d_high; tl, yl = d_low
            yh = apply_scl_baseline_correction(th, yh, component)
            yl = apply_scl_baseline_correction(tl, yl, component)
            if not (validate_sufficient_data(th, yh, limit_sec) and validate_sufficient_data(tl, yl, limit_sec)):
                continue
            mask_h = (th >= 0.0) & (th <= limit_sec)
            mask_l = (tl >= 0.0) & (tl <= limit_sec)
            th = th[mask_h]; yh = yh[mask_h]
            tl = tl[mask_l]; yl = yl[mask_l]

            # RS
            p_r1 = build_rs_cvx_path(subject, 'session1')
            p_r2 = build_rs_cvx_path(subject, 'session2')
            r1 = load_cvx_component(p_r1, component)
            r2 = load_cvx_component(p_r2, component)
            if None in (r1, r2):
                continue
            tr1, yr1 = r1; tr2, yr2 = r2
            yr1 = apply_scl_baseline_correction(tr1, yr1, component)
            yr2 = apply_scl_baseline_correction(tr2, yr2, component)
            if not (validate_sufficient_data(tr1, yr1, limit_sec) and validate_sufficient_data(tr2, yr2, limit_sec)):
                continue
            m1 = (tr1 >= 0.0) & (tr1 <= limit_sec)
            m2 = (tr2 >= 0.0) & (tr2 <= limit_sec)
            tr1 = tr1[m1]; yr1 = yr1[m1]
            tr2 = tr2[m2]; yr2 = yr2[m2]

            # Determine RS dose mapping
            dose_s1 = get_dosis_sujeto(subject, 1)
            cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
            # session2 inferred opposite
            cond2 = 'Low' if cond1 == 'High' else 'High'

            rows.append({
                'subject': subject,
                't_rs1': tr1, 'y_rs1': yr1, 'cond1': cond1,
                't_rs2': tr2, 'y_rs2': yr2, 'cond2': cond2,
                't_dmt_h': th, 'y_dmt_h': yh,
                't_dmt_l': tl, 'y_dmt_l': yl,
            })
        except Exception:
            continue

    if not rows:
        return False

    n = len(rows)
    # Increase figure height to accommodate suptitles per row
    fig, axes = plt.subplots(n, 2, figsize=(14, max(4.0, 2.2 * n)), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([axes])

    # Colors
    c_dmt_high, c_dmt_low = 'tab:red', 'tab:blue'
    c_rs_high, c_rs_low = 'tab:green', 'tab:purple'

    minute_ticks = np.arange(0.0, 541.0, 60.0)

    from matplotlib.lines import Line2D

    for i, row in enumerate(rows):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]

        # RS left subplot
        if row['cond1'] == 'High':
            ax_rs.plot(row['t_rs1'], row['y_rs1'], color=c_rs_high, lw=1.4)
        else:
            ax_rs.plot(row['t_rs1'], row['y_rs1'], color=c_rs_low, lw=1.4)
        if row['cond2'] == 'High':
            ax_rs.plot(row['t_rs2'], row['y_rs2'], color=c_rs_high, lw=1.4)
        else:
            ax_rs.plot(row['t_rs2'], row['y_rs2'], color=c_rs_low, lw=1.4)
        
        _beautify_axes(ax_rs, title=None, xlabel='Time (minutes)', ylabel=units)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(0.0, limit_sec)

        # RS legend with fixed order
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=c_rs_high, lw=1.4, label='RS High'),
            Line2D([0], [0], color=c_rs_low, lw=1.4, label='RS Low'),
        ], loc='upper right', frameon=True, fancybox=False)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)

        # DMT right subplot
        ax_dmt.plot(row['t_dmt_h'], row['y_dmt_h'], color=c_dmt_high, lw=1.4)
        ax_dmt.plot(row['t_dmt_l'], row['y_dmt_l'], color=c_dmt_low, lw=1.4)
        
        _beautify_axes(ax_dmt, title=None, xlabel='Time (minutes)', ylabel=units)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(0.0, limit_sec)

        # DMT legend with fixed order
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=c_dmt_high, lw=1.4, label='DMT High'),
            Line2D([0], [0], color=c_dmt_low, lw=1.4, label='DMT Low'),
        ], loc='upper right', frameon=True, fancybox=False)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)

        # Y ticks on both columns
        ax_rs.tick_params(labelleft=True)
        ax_dmt.tick_params(labelleft=True)

        # X ticks and formatter
        ax_rs.set_xticks(minute_ticks)
        ax_rs.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
        ax_dmt.set_xticks(minute_ticks)
        ax_dmt.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))

        # Y limits
        ax_rs.set_ylim(-5.0, 5.0)
        ax_dmt.set_ylim(-5.0, 5.0)
        
        # Component-specific Y-axis settings for SCR
        if component == 'SCR':
            ax_rs.set_ylim(-0.25, 1.0)
            ax_dmt.set_ylim(-0.25, 1.0)
            ax_rs.set_yticks(np.arange(0.0, 1.01, 0.25))
            ax_dmt.set_yticks(np.arange(0.0, 1.01, 0.25))

        # Add subject suptitle above this row (centered between the two subplots)
        # Use fig.text to place text in figure coordinates
        # Calculate vertical position for this row's suptitle
        row_center_y = 1.0 - (i + 0.5) / n  # Approximate center of row in figure coords
        fig.text(0.5, row_center_y + 0.035, row['subject'], 
                 ha='center', va='bottom', fontweight='bold', fontsize=26,
                 transform=fig.transFigure)

    fig.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True

def _write_combined_captions(out_dir: str, component: str):
    """Write a single comprehensive captions file for all figures of this component."""
    config = COMPONENT_CONFIG[component]
    comp_name = component
    comp_long = config['long_name']
    comp_desc = config['description']
    units = config['units']
    output_subdir = config['output_dir']  # 'scr' or 'scl'
    
    caption_file = os.path.join(out_dir, f'captions_{output_subdir}.txt')
    
    captions = []
    
    # Caption 1: Combined summary (RS + DMT)
    captions.append(
        f"Figure: {comp_name} Comparison Across Tasks (Resting State and DMT)\n"
        f"Group-level mean ± SEM for {comp_long} ({comp_desc}) across 11 subjects during two tasks. "
        f"Left panel: Resting State (RS) with eyes closed, comparing High (green) versus Low (purple) dose conditions. "
        f"Right panel: DMT task, comparing High (red) versus Low (blue) dose conditions. "
        f"The X-axis represents time in minutes (0:00–9:00) with tick marks every minute. "
        f"The Y-axis displays {units} with a fixed range of [−5, 5] μS to enable direct comparison across tasks. "
        f"Shaded regions represent the standard error of the mean (SEM). "
        f"Each curve is the average of 11 individual subjects' {comp_name} time series, "
        f"resampled to a common 2 Hz grid and then averaged point-wise. "
        f"The plot highlights dose-dependent modulation in both resting and active psychedelic states."
    )
    
    # Caption 2: DMT-only summary (19:10 for SCL)
    if component == 'SCL':
        captions.append(
            f"\nFigure: DMT Task {comp_name} Extended Time Course (0–19 minutes)\n"
            f"Group-level mean ± SEM for {comp_long} ({comp_desc}) during the DMT task across the extended recording period (19 minutes 10 seconds). "
            f"The plot compares High (red) versus Low (blue) dose conditions. "
            f"The X-axis represents time in minutes (0:00–19:00) with tick marks every minute. "
            f"The Y-axis displays {units} with a fixed range of [−5, 5] μS. "
            f"Shaded regions represent the standard error of the mean (SEM). "
            f"This extended time series (N = 11 subjects) reveals the temporal dynamics of the tonic electrodermal component "
            f"throughout the full DMT experience, including onset, peak, and offset phases. "
            f"The longer recording window allows examination of sustained dose effects beyond the acute psychedelic state."
        )
    
    # Caption 3: Stacked per-subject
    captions.append(
        f"\nSupplementary Figure: Individual Subject {comp_name} Time Series (Stacked Layout)\n"
        f"Per-subject {comp_long} ({comp_desc}) time series for all 11 validated subjects. "
        f"Each row represents one subject, with the subject code displayed in bold on the left margin. "
        f"Left column: Resting State (RS) with eyes closed, showing High (green) and Low (purple) dose sessions overlaid. "
        f"Right column: DMT task, showing High (red) and Low (blue) dose sessions overlaid. "
        f"The X-axis represents time in minutes (0:00–9:00) with tick marks every minute. "
        f"The Y-axis displays {units} with a fixed range of [−5, 5] μS across all subjects to facilitate visual comparison. "
        f"Both axes are shared across all subplots to enable direct between-subject comparisons of signal amplitude and temporal dynamics. "
        f"This layout reveals individual variability in {comp_name} responses to dose and task conditions, "
        f"complementing the group-level averages shown in the main figures. "
        f"Notable inter-individual differences can be observed in baseline levels, response magnitudes, and temporal patterns."
    )
    
    combined_text = "\n\n".join(captions)
    
    try:
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        print(f"📝 Combined captions written: {caption_file}")
    except Exception as e:
        print(f"⚠️  Failed to write combined captions: {e}")


def plot_subject_component_combined(subject: str, component: str) -> bool:
    """Generate and save a combined figure with two subplots for one subject."""
    config = COMPONENT_CONFIG[component]
    
    # DMT data
    high_session, low_session = determine_sessions(subject)
    dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
    dmt_high = load_cvx_component(dmt_high_path, component)
    dmt_low = load_cvx_component(dmt_low_path, component)
    if dmt_high is None or dmt_low is None:
        print(f"⚠️  Skipping {subject}: missing DMT data")
        return False

    t_high, comp_high = dmt_high
    t_low, comp_low = dmt_low
    
    # Apply baseline correction for SCL
    comp_high = apply_scl_baseline_correction(t_high, comp_high, component)
    comp_low = apply_scl_baseline_correction(t_low, comp_low, component)
    
    limit_subj = 550.0  # 9 minutes 10 seconds
    # Validate sufficient valid data for 9:10
    if not (validate_sufficient_data(t_high, comp_high, limit_subj) and validate_sufficient_data(t_low, comp_low, limit_subj)):
        print(f"⚠️  Skipping {subject}: insufficient valid DMT data in first 9:10")
        return False
    
    idx_h = np.where(t_high <= limit_subj)[0]
    idx_l = np.where(t_low <= limit_subj)[0]
    if len(idx_h) < 10 or len(idx_l) < 10:
        print(f"⚠️  Skipping {subject}: insufficient DMT samples in first 9:10")
        return False
    t_high_10 = t_high[idx_h]
    t_low_10 = t_low[idx_l]
    y_comp_high = comp_high[idx_h]
    y_comp_low = comp_low[idx_l]
    n_dmt = min(len(t_high_10), len(y_comp_high), len(t_low_10), len(y_comp_low))
    if n_dmt < 10:
        print(f"⚠️  Skipping {subject}: insufficient overlapping DMT samples after trim")
        return False
    t_dmt = t_high_10[:n_dmt]
    y_dmt_high = y_comp_high[:n_dmt]
    y_dmt_low = y_comp_low[:n_dmt]
    mean_dmt_high = compute_mean_value(y_dmt_high)
    mean_dmt_low = compute_mean_value(y_dmt_low)

    # RS data
    rs1_path = build_rs_cvx_path(subject, 'session1')
    rs2_path = build_rs_cvx_path(subject, 'session2')
    rs1 = load_cvx_component(rs1_path, component)
    rs2 = load_cvx_component(rs2_path, component)
    if rs1 is None or rs2 is None:
        print(f"⚠️  Skipping {subject}: missing RS data (ses1 or ses2)")
        return False
    t1, y1 = rs1
    t2, y2 = rs2
    
    # Apply baseline correction for SCL
    y1 = apply_scl_baseline_correction(t1, y1, component)
    y2 = apply_scl_baseline_correction(t2, y2, component)
    
    # Validate sufficient valid data for 9:10
    if not (validate_sufficient_data(t1, y1, limit_subj) and validate_sufficient_data(t2, y2, limit_subj)):
        print(f"⚠️  Skipping {subject}: insufficient valid RS data in first 9:10")
        return False
    
    idx1 = np.where(t1 <= limit_subj)[0]
    idx2 = np.where(t2 <= limit_subj)[0]
    if len(idx1) < 10 or len(idx2) < 10:
        print(f"⚠️  Skipping {subject}: insufficient RS samples in first 9:10")
        return False
    t1_10 = t1[idx1]
    y1_10 = y1[idx1]
    t2_10 = t2[idx2]
    y2_10 = y2[idx2]
    n_rs = min(len(t1_10), len(y1_10), len(t2_10), len(y2_10))
    if n_rs < 10:
        print(f"⚠️  Skipping {subject}: insufficient overlapping RS samples after trim")
        return False
    t_rs = t1_10[:n_rs]
    y_rs1 = y1_10[:n_rs]
    y_rs2 = y2_10[:n_rs]
    mean_rs1 = compute_mean_value(y_rs1)
    mean_rs2 = compute_mean_value(y_rs2)

    # Determine dose labels for RS
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)
    except Exception:
        dose_s1 = 'Alta'
    try:
        dose_s2 = get_dosis_sujeto(subject, 2)
    except Exception:
        dose_s2 = 'Baja'
    cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
    cond2 = 'High' if str(dose_s2).lower().startswith('alta') or str(dose_s2).lower().startswith('a') else 'Low'

    # Combined figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True, sharey=True)

    # Colors fixed across subjects
    c_dmt_high = 'tab:red'
    c_dmt_low = 'tab:blue'
    c_rs_high = 'tab:green'
    c_rs_low = 'tab:purple'

    # Subplot 1: Resting State (RS) - LEFT
    # Map RS traces to High/Low colors based on cond1/cond2 computed above
    # cond1/cond2 already hold 'High'/'Low' strings
    if cond1 == 'High':
        ax1.plot(t_rs, y_rs1, color=c_rs_high, lw=1.4, label='RS High')
    else:
        ax1.plot(t_rs, y_rs1, color=c_rs_low, lw=1.4, label='RS Low')
    if cond2 == 'High':
        ax1.plot(t_rs, y_rs2, color=c_rs_high, lw=1.4, label='RS High')
    else:
        ax1.plot(t_rs, y_rs2, color=c_rs_low, lw=1.4, label='RS Low')
    _beautify_axes(ax1, title=None, xlabel='Time (minutes)', ylabel=config['units'])
    ax1.set_title('Resting State (RS)', fontweight='bold')

    # Subplot 2: DMT - RIGHT
    ax2.plot(t_dmt, y_dmt_high, color=c_dmt_high, lw=1.4, label='DMT High')
    ax2.plot(t_dmt, y_dmt_low, color=c_dmt_low, lw=1.4, label='DMT Low')
    _beautify_axes(ax2, title=None, xlabel='Time (minutes)', ylabel=config['units'])
    ax2.set_title('DMT', fontweight='bold')

    # Legends (no mean values)
    from matplotlib.lines import Line2D
    legend1 = ax1.legend(handles=[
        Line2D([0], [0], color=c_rs_high, lw=1.4, label='RS High'),
        Line2D([0], [0], color=c_rs_low, lw=1.4, label='RS Low'),
    ], loc='upper right', frameon=True, fancybox=True)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    legend2 = ax2.legend(handles=[
        Line2D([0], [0], color=c_dmt_high, lw=1.4, label='DMT High'),
        Line2D([0], [0], color=c_dmt_low, lw=1.4, label='DMT Low'),
    ], loc='upper right', frameon=True, fancybox=True)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)

    # Consistent axes
    ax1.set_xlim(0.0, 550.0)
    ax2.set_xlim(0.0, 550.0)
    ax1.set_ylim(-5.0, 5.0)
    ax2.set_ylim(-5.0, 5.0)
    
    # Component-specific Y-axis settings for SCR
    if component == 'SCR':
        ax1.set_ylim(-0.25, 1.0)
        ax2.set_ylim(-0.25, 1.0)
        ax1.set_yticks(np.arange(0.0, 1.01, 0.25))
        ax2.set_yticks(np.arange(0.0, 1.01, 0.25))

    # Y tick labels visible on both subplots
    ax1.tick_params(labelleft=True)
    ax2.tick_params(labelleft=True)

    # Major ticks every minute from 0:00 to 9:00; show formatter mm:ss, but label says minutes
    minute_ticks = np.arange(0.0, 541.0, 60.0)
    for ax in (ax1, ax2):
        ax.set_xticks(minute_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))

    fig.suptitle(f"{subject}", y=1.02, fontweight='bold', fontsize=26)
    fig.tight_layout(pad=1.2)

    out_dir = os.path.join('test', 'eda', config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject.lower()}_eda_{component.lower()}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved combined {component} plot: {out_path}")
    return True


def run_component_analysis(component: str):
    """Run complete analysis for specified component (SCR or SCL)."""
    config = COMPONENT_CONFIG[component]
    durations_log = [] if component == 'SCL' else None
    
    print(f"📊 Generating EDA {component} combined plots per validated subject…")
    successes_combined = 0
    
    for subject in SUJETOS_VALIDADOS_EDA:
        ok = plot_subject_component_combined(subject, component)
        if ok:
            successes_combined += 1
        
        # Collect durations (SCL only): last valid timestamp for each available file
        if durations_log is not None:
            # Build paths
            high_session, low_session = determine_sessions(subject)
            dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
            rs1_path = build_rs_cvx_path(subject, 'session1')
            rs2_path = build_rs_cvx_path(subject, 'session2')
            # Load
            dmt_high = load_cvx_component(dmt_high_path, component)
            dmt_low = load_cvx_component(dmt_low_path, component)
            rs1 = load_cvx_component(rs1_path, component)
            rs2 = load_cvx_component(rs2_path, component)
            if None in (dmt_high, dmt_low, rs1, rs2):
                continue
            # DMT High/Low
            th, yh = dmt_high; tl, yl = dmt_low
            dur_dmt_high = _last_valid_timestamp_mmss(th, yh)
            dur_dmt_low = _last_valid_timestamp_mmss(tl, yl)
            if dur_dmt_high is not None:
                durations_log.append({
                    'subject_number': subject,
                    'condition': 'DMT',
                    'dose': 'high',
                    'duration': dur_dmt_high,
                })
            if dur_dmt_low is not None:
                durations_log.append({
                    'subject_number': subject,
                    'condition': 'DMT',
                    'dose': 'low',
                    'duration': dur_dmt_low,
                })
            # RS session1/session2 mapped to doses
            try:
                dose_s1 = get_dosis_sujeto(subject, 1)
            except Exception:
                dose_s1 = 'Alta'
            try:
                dose_s2 = get_dosis_sujeto(subject, 2)
            except Exception:
                dose_s2 = 'Baja'
            d1 = 'high' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'low'
            d2 = 'high' if str(dose_s2).lower().startswith('alta') or str(dose_s2).lower().startswith('a') else 'low'
            t1, y1 = rs1; t2, y2 = rs2
            dur_rs1 = _last_valid_timestamp_mmss(t1, y1)
            dur_rs2 = _last_valid_timestamp_mmss(t2, y2)
            if dur_rs1 is not None:
                durations_log.append({
                    'subject_number': subject,
                    'condition': 'RS',
                    'dose': d1,
                    'duration': dur_rs1,
                })
            if dur_rs2 is not None:
                durations_log.append({
                    'subject_number': subject,
                    'condition': 'RS',
                    'dose': d2,
                    'duration': dur_rs2,
                })
    
    print(f"🎯 Completed combined plots: {successes_combined}/{len(SUJETOS_VALIDADOS_EDA)}")
    
    # Setup output directory
    out_dir = os.path.join('test', 'eda', config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    
    # Write durations JSON for SCL
    if durations_log is not None and len(durations_log) > 0:
        durations_path = os.path.join(out_dir, 'scl_valid_durations.json')
        try:
            with open(durations_path, 'w', encoding='utf-8') as f:
                json.dump(durations_log, f, ensure_ascii=False, indent=2)
            print(f"📝 SCL durations log saved: {durations_path}")
        except Exception as e:
            print(f"⚠️  Failed to write durations JSON: {e}")

    # Generate summary plots
    # Combined summary plot with DMT and RS subplots
    out_sum_combined = os.path.join(out_dir, f'all_subs_eda_{component.lower()}.png')
    ok_combined = _plot_combined_summary(out_sum_combined, component)
    if ok_combined:
        print(f"🖼️  Combined summary DMT+RS mean±SEM saved: {out_sum_combined}")

    # Additional DMT-only plot for 20 minutes (both SCR and SCL)
    out_dmt_20min = os.path.join(out_dir, f'all_subs_dmt_eda_{component.lower()}.png')
    ok_dmt_20min = _plot_dmt_only_20min(out_dmt_20min, component)
    if ok_dmt_20min:
        print(f"🖼️  DMT-only 20-min {component} summary saved: {out_dmt_20min}")

    # Stacked figure across subjects (supplementary)
    try:
        _plot_all_subjects_stacked(os.path.join(out_dir, f'stacked_subs_eda_{component.lower()}.png'), component)
    except Exception:
        pass
    
    # Write combined captions file for all figures
    _write_combined_captions(out_dir, component)


def main():
    parser = argparse.ArgumentParser(description='Analyze EDA components (SCR or SCL)')
    parser.add_argument('--component', choices=['SCR', 'SCL'], required=True,
                        help='EDA component to analyze: SCR (phasic) or SCL (tonic)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.component} analysis...")
    print(f"Component: {COMPONENT_CONFIG[args.component]['long_name']} ({COMPONENT_CONFIG[args.component]['description']})")
    
    run_component_analysis(args.component)
    
    print(f"\n✅ {args.component} analysis completed!")


if __name__ == '__main__':
    main()
