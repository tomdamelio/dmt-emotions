# -*- coding: utf-8 -*-
"""
Plot per-subject EDA component analysis: EDR (phasic) and EDL (tonic) comparison.

This script analyzes both components from CVX decomposition:
- EDR (Electrodermal Response): phasic component - transient responses
- EDL (Electrodermal Level): tonic component - baseline conductance level

For each subject in SUJETOS_VALIDADOS_EDA, this script:
- Finds the correct session for high vs low using get_dosis_sujeto
- Loads CVX decomposition CSVs (…_cvx_decomposition.csv)
- Extracts the specified component (EDR or EDL) and time
- Trims both recordings to the first 10 minutes
- Computes mean value of the component
- Plots the component with mean levels and SEM shading
- Saves to test/eda/scr (EDR) or test/eda/scl (EDL)

Usage:
  python test/plot_eda_components.py --component EDR
  python test/plot_eda_components.py --component EDL
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
from stats_anova_2x2 import run_anova_2x2_within, run_anova_2x2_per_minute, _render_anova_results

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
    'EDR': {
        'name': 'EDR',
        'long_name': 'Electrodermal Response',
        'description': 'phasic component',
        'output_dir': 'scr',
        'units': 'EDR (a.u.)',
        'colors': {'high': '#555555', 'low': '#2A9FD6'}
    },
    'EDL': {
        'name': 'EDL', 
        'long_name': 'Electrodermal Level',
        'description': 'tonic component (baseline-corrected)',
        'output_dir': 'scl',
        'units': 'Δ EDL (a.u.)',
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


def _records_to_long_df(records):
    rows = []
    for rec in records:
        sid = rec['subject']
        rows.append({'Subject': sid, 'Task': 'RS',  'Dose': 'Low',  'AUC': rec['RS_Low']})
        rows.append({'Subject': sid, 'Task': 'RS',  'Dose': 'High', 'AUC': rec['RS_High']})
        rows.append({'Subject': sid, 'Task': 'DMT', 'Dose': 'Low',  'AUC': rec['DMT_Low']})
        rows.append({'Subject': sid, 'Task': 'DMT', 'Dose': 'High', 'AUC': rec['DMT_High']})
    import pandas as pd
    df = pd.DataFrame(rows)
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    return df


def _plot_boxplot_2x2(records, out_path, component_name):
    """Create a comparative 2x2 plot (RS/DMT x Low/High) with paired lines per subject."""
    import numpy as np
    import pandas as pd
    df = _records_to_long_df(records)

    # Build arrays per cell in order: Low_RS, Low_DMT, High_RS, High_DMT
    low_rs = df[(df['Dose'] == 'Low') & (df['Task'] == 'RS')]
    low_dmt = df[(df['Dose'] == 'Low') & (df['Task'] == 'DMT')]
    high_rs = df[(df['Dose'] == 'High') & (df['Task'] == 'RS')]
    high_dmt = df[(df['Dose'] == 'High') & (df['Task'] == 'DMT')]

    # Fixed x-positions for four boxes within two dose columns
    pos = [0.88, 1.12, 1.88, 2.12]
    xticks = [1.0, 2.0]
    xticklabels = ['Low', 'High']

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    groups = [low_rs['AUC'].to_numpy(), low_dmt['AUC'].to_numpy(),
              high_rs['AUC'].to_numpy(), high_dmt['AUC'].to_numpy()]
    bp = ax.boxplot(groups, positions=pos, widths=0.18, patch_artist=True,
                    showfliers=False, medianprops=dict(color='k', linewidth=1.2))
    colors = ['#4C9F70', '#D65F5F', '#4C9F70', '#D65F5F']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)

    # Overlay paired individual lines (spaghetti) across all four positions
    subjects = sorted(df['Subject'].unique())
    for sid in subjects:
        s_low_rs = df[(df['Subject'] == sid) & (df['Dose'] == 'Low') & (df['Task'] == 'RS')]['AUC']
        s_low_dmt = df[(df['Subject'] == sid) & (df['Dose'] == 'Low') & (df['Task'] == 'DMT')]['AUC']
        s_high_rs = df[(df['Subject'] == sid) & (df['Dose'] == 'High') & (df['Task'] == 'RS')]['AUC']
        s_high_dmt = df[(df['Subject'] == sid) & (df['Dose'] == 'High') & (df['Task'] == 'DMT')]['AUC']
        if len(s_low_rs)==len(s_low_dmt)==len(s_high_rs)==len(s_high_dmt)==1:
            y = [float(s_low_rs.values[0]), float(s_low_dmt.values[0]), float(s_high_rs.values[0]), float(s_high_dmt.values[0])]
            ax.plot(pos, y, color='k', alpha=0.25, linewidth=1.0)
            ax.scatter(pos, y, color='k', s=12, alpha=0.5, zorder=3)

    # Legend boxes for RS vs DMT
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor='#4C9F70', edgecolor='none', alpha=0.55, label='RS'),
                    Patch(facecolor='#D65F5F', edgecolor='none', alpha=0.55, label='DMT')]
    ax.legend(handles=legend_elems, title='Task', frameon=False, loc='upper left')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    _beautify_axes(ax, title=f'{component_name} Mean – 2×2 design (paired lines)', 
                  xlabel='Dose', ylabel=f'Mean {component_name} (first 10 min)', time_formatter=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)


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
        component: 'EDR' or 'EDL'
        
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

    if component not in df.columns:
        print(f"⚠️  {component} column not found in {os.path.basename(csv_path)}")
        return None

    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)

    component_data = pd.to_numeric(df[component], errors='coerce').fillna(0.0).to_numpy()
    return t, component_data


def compute_baseline_edl(t: np.ndarray, y: np.ndarray) -> float:
    """Compute baseline EDL value from the first second of recording.
    
    Args:
        t: Time array
        y: EDL signal array
        
    Returns:
        Mean EDL value in the first second (0-1s)
    """
    mask_first_sec = (t >= 0.0) & (t < 1.0)
    if not np.any(mask_first_sec):
        # If no data in first second, use first available sample
        return float(y[0]) if len(y) > 0 else 0.0
    
    return float(np.mean(y[mask_first_sec]))


def apply_edl_baseline_correction(t: np.ndarray, y: np.ndarray, component: str) -> np.ndarray:
    """Apply baseline correction for EDL signals.
    
    Args:
        t: Time array
        y: Signal array
        component: 'EDR' or 'EDL'
        
    Returns:
        Original signal for EDR, baseline-corrected signal for EDL
    """
    if component == 'EDL':
        baseline = compute_baseline_edl(t, y)
        return y - baseline
    else:
        return y


def compute_mean_value(y: np.ndarray, component: str = 'EDR', baseline_value: Optional[float] = None) -> float:
    """Compute mean value of component signal.
    
    Args:
        y: Signal array (already baseline-corrected for EDL)
        component: 'EDR' or 'EDL' 
        baseline_value: Deprecated, kept for compatibility
    """
    return float(np.mean(y))


def moving_average(x: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return x
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(x, kernel, mode='same')


def compute_envelope(y: np.ndarray, sr: float, window_sec: float = 1.0) -> np.ndarray:
    """Return a smoothed envelope for y using moving-average."""
    window_samples = max(1, int(round(sr * window_sec)))
    return moving_average(y, window_samples)


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Linearly resample y(t) to t_grid, guarding NaNs."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    # Clip grid to data span; fill outside with edge values
    yg = np.interp(t_grid, t, y, left=y[0], right=y[-1])
    return yg


def _plot_dmt_only_20min(out_path: str, component: str) -> bool:
    """Create a DMT-only plot for 20 minutes (EDL analysis only).
    Shows High vs Low dose for DMT task across the full 20-minute recording.
    """
    if component != 'EDL':
        print(f"⚠️  DMT 20-min plot is only available for EDL component")
        return False
        
    config = COMPONENT_CONFIG[component]
    
    # Extended grid: 0..1200s (20 min), 2 Hz for readability
    t_grid = np.arange(0.0, 1200.0, 0.5)
    
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

            # Apply baseline correction for EDL
            yh = apply_edl_baseline_correction(th, yh, component)
            yl = apply_edl_baseline_correction(tl, yl, component)
            
            # Trim to 20 minutes
            mask_h = (th >= 0.0) & (th < 1200.0)
            mask_l = (tl >= 0.0) & (tl < 1200.0)
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
    
    # Colors from config
    c_high = config['colors']['high']
    c_low = config['colors']['low']
    
    # Plot DMT data
    line_h = ax.plot(t_grid, mean_h, color=c_high, lw=2.5, marker=None, label='High')[0]
    ax.fill_between(t_grid, mean_h - sem_h, mean_h + sem_h, color=c_high, alpha=0.25)
    line_l = ax.plot(t_grid, mean_l, color=c_low, lw=2.5, marker=None, label='Low')[0]
    ax.fill_between(t_grid, mean_l - sem_l, mean_l + sem_l, color=c_low, alpha=0.25)
    
    # Legend with correct colors - High first, then Low
    legend = ax.legend([line_h, line_l], ['High', 'Low'], loc='upper right', 
                      frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax, title=f'DMT – {component} (mean ± SEM, full 20 min)', 
                  xlabel='Time (mm:ss)', ylabel=config['units'])
    
    # Add overall figure title
    fig.suptitle(f'{config["long_name"]} ({component}) during DMT Task\n'
                f'Group Mean ± SEM across {len(high_curves)} Subjects ({config["description"]})', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return True


def _plot_combined_summary(out_path: str, component: str) -> bool:
    """Create a combined summary plot with DMT (left) and RS (right) subplots."""
    config = COMPONENT_CONFIG[component]
    
    # Common grid: 0..600s, 2 Hz for readability
    t_grid = np.arange(0.0, 600.0, 0.5)
    
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

                # Apply baseline correction for EDL
                yh = apply_edl_baseline_correction(th, yh, component)
                yl = apply_edl_baseline_correction(tl, yl, component)
                
                # Trim to 10 minutes
                mask_h = (th >= 0.0) & (th < 600.0)
                mask_l = (tl >= 0.0) & (tl < 600.0)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Colors from config
    c_high = config['colors']['high']
    c_low = config['colors']['low']
    
    # DMT subplot (left)
    dmt_data = task_data['DMT']
    line_h1 = ax1.plot(t_grid, dmt_data['mean_h'], color=c_high, lw=2.0, marker=None, label='High')[0]
    ax1.fill_between(t_grid, dmt_data['mean_h'] - dmt_data['sem_h'], 
                     dmt_data['mean_h'] + dmt_data['sem_h'], color=c_high, alpha=0.25)
    line_l1 = ax1.plot(t_grid, dmt_data['mean_l'], color=c_low, lw=2.0, marker=None, label='Low')[0]
    ax1.fill_between(t_grid, dmt_data['mean_l'] - dmt_data['sem_l'], 
                     dmt_data['mean_l'] + dmt_data['sem_l'], color=c_low, alpha=0.25)
    
    # Legend for DMT with correct colors - High first, then Low
    legend1 = ax1.legend([line_h1, line_l1], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=True, shadow=True)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax1, title=f'DMT – {component} (mean ± SEM, first 10 min)', 
                  xlabel='Time (mm:ss)', ylabel=config['units'])
    
    # RS subplot (right)
    rs_data = task_data['RS']
    line_h2 = ax2.plot(t_grid, rs_data['mean_h'], color=c_high, lw=2.0, marker=None, label='High')[0]
    ax2.fill_between(t_grid, rs_data['mean_h'] - rs_data['sem_h'], 
                     rs_data['mean_h'] + rs_data['sem_h'], color=c_high, alpha=0.25)
    line_l2 = ax2.plot(t_grid, rs_data['mean_l'], color=c_low, lw=2.0, marker=None, label='Low')[0]
    ax2.fill_between(t_grid, rs_data['mean_l'] - rs_data['sem_l'], 
                     rs_data['mean_l'] + rs_data['sem_l'], color=c_low, alpha=0.25)
    
    # Legend for RS with correct colors - High first, then Low
    legend2 = ax2.legend([line_h2, line_l2], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=True, shadow=True)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax2, title=f'RS – {component} (mean ± SEM, first 10 min)', 
                  xlabel='Time (mm:ss)')
    
    # Add overall figure title
    fig.suptitle(f'{config["long_name"]} ({component}) by Task and Dose\n'
                f'Group Mean ± SEM across 11 Subjects ({config["description"]})', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return True


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
    
    # Apply baseline correction for EDL
    comp_high = apply_edl_baseline_correction(t_high, comp_high, component)
    comp_low = apply_edl_baseline_correction(t_low, comp_low, component)
    
    ten_min = 600.0
    idx_h = np.where(t_high <= ten_min)[0]
    idx_l = np.where(t_low <= ten_min)[0]
    if len(idx_h) < 10 or len(idx_l) < 10:
        print(f"⚠️  Skipping {subject}: insufficient DMT samples in first 10 minutes")
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
    
    # Apply baseline correction for EDL
    y1 = apply_edl_baseline_correction(t1, y1, component)
    y2 = apply_edl_baseline_correction(t2, y2, component)
    
    idx1 = np.where(t1 <= ten_min)[0]
    idx2 = np.where(t2 <= ten_min)[0]
    if len(idx1) < 10 or len(idx2) < 10:
        print(f"⚠️  Skipping {subject}: insufficient RS samples in first 10 minutes")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)

    # Subplot 1: DMT high vs low - plot component directly without mean lines
    ax1.plot(t_dmt, y_dmt_high, color='tab:red', lw=1.4, label=f'DMT High (mean={mean_dmt_high:.3f})')
    ax1.plot(t_dmt, y_dmt_low, color='tab:blue', lw=1.4, label=f'DMT Low (mean={mean_dmt_low:.3f})')
    _beautify_axes(ax1, title='DMT High vs Low (first 10 min)', xlabel='Time', ylabel=config['units'])
    
    # Legend in upper right
    legend1 = ax1.legend(loc='upper right', frameon=True, fancybox=True)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)

    # Subplot 2: RS ses01 vs ses02 with dose tags - plot component directly without mean lines
    ax2.plot(t_rs, y_rs1, color='tab:green', lw=1.4, label=f'RS {cond1} (mean={mean_rs1:.3f})')
    ax2.plot(t_rs, y_rs2, color='tab:purple', lw=1.4, label=f'RS {cond2} (mean={mean_rs2:.3f})')
    _beautify_axes(ax2, title='RS ses01 vs ses02 (first 10 min)', xlabel='Time (mm:ss)')
    
    # Legend in upper right
    legend2 = ax2.legend(loc='upper right', frameon=True, fancybox=True)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)

    fig.suptitle(f"{subject} – EDA {component} (first 10 min)", y=1.02)
    fig.tight_layout(pad=1.2)

    out_dir = os.path.join('test', 'eda', config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject.lower()}_eda_{component.lower()}_combined_10min.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved combined {component} plot: {out_path}")
    return True


def run_component_analysis(component: str):
    """Run complete analysis for specified component (EDR or EDL)."""
    config = COMPONENT_CONFIG[component]
    
    print(f"📊 Generating EDA {component} combined plots per validated subject…")
    successes_combined = 0
    # Collect per-subject mean values for 2x2 ANOVA
    anova_records = []
    for subject in SUJETOS_VALIDADOS_EDA:
        ok = plot_subject_component_combined(subject, component)
        if ok:
            successes_combined += 1
        # Compute and store mean values for all four cells for ANOVA
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
        # Trim to 10 minutes and compute mean value consistently
        def trim_and_mean(pair):
            tt, yy = pair
            # Apply baseline correction for EDL
            yy = apply_edl_baseline_correction(tt, yy, component)
            idx = np.where(tt <= 600.0)[0]
            if len(idx) < 10:
                return None
            yy = yy[idx]
            return compute_mean_value(yy)
        mean_dmt_high = trim_and_mean(dmt_high)
        mean_dmt_low = trim_and_mean(dmt_low)
        mean_rs1 = trim_and_mean(rs1)
        mean_rs2 = trim_and_mean(rs2)
        if None in (mean_dmt_high, mean_dmt_low, mean_rs1, mean_rs2):
            continue
        # Map RS1/RS2 to Low/High using session dose
        dose_s1 = get_dosis_sujeto(subject, 1)
        dose_s2 = get_dosis_sujeto(subject, 2)
        if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
            rs_high = mean_rs1; rs_low = mean_rs2
        else:
            rs_high = mean_rs2; rs_low = mean_rs1
        # Map DMT High/Low already aligned by determine_sessions
        rec = {
            'subject': subject,
            'RS_Low': rs_low,
            'RS_High': rs_high,
            'DMT_Low': mean_dmt_low,
            'DMT_High': mean_dmt_high,
        }
        anova_records.append(rec)
    print(f"🎯 Completed combined plots: {successes_combined}/{len(SUJETOS_VALIDADOS_EDA)}")
    
    # Run 2x2 within-subject ANOVA (global) and per-minute ANOVAs; write report
    out_dir = os.path.join('test', 'eda', config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'anova_2x2_report.txt')
    if anova_records:
        _ = run_anova_2x2_within(anova_records, out_report_path=report_path)
        print(f"📄 ANOVA 2x2 report (global) saved: {report_path}")

    # Build per-minute records: for m in 0..9, compute per-subject means in that 1-min window
    records_by_minute = {m: [] for m in range(10)}
    for subject in SUJETOS_VALIDADOS_EDA:
        # Gather sources as before
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs1_path = build_rs_cvx_path(subject, 'session1')
        rs2_path = build_rs_cvx_path(subject, 'session2')
        dmt_high = load_cvx_component(dmt_high_path, component)
        dmt_low = load_cvx_component(dmt_low_path, component)
        rs1 = load_cvx_component(rs1_path, component)
        rs2 = load_cvx_component(rs2_path, component)
        if None in (dmt_high, dmt_low, rs1, rs2):
            continue

        # Minute windows
        def mean_window(pair, m):
            tt, yy = pair
            # Apply baseline correction for EDL
            yy = apply_edl_baseline_correction(tt, yy, component)
            mask = (tt >= 60.0 * m) & (tt < 60.0 * (m + 1))
            if not np.any(mask):
                return None
            return compute_mean_value(yy[mask])

        dose_s1 = get_dosis_sujeto(subject, 1)
        # For each minute, compute all four cells
        for m in range(10):
            mean_dh = mean_window(dmt_high, m)
            mean_dl = mean_window(dmt_low, m)
            mean_r1 = mean_window(rs1, m)
            mean_r2 = mean_window(rs2, m)
            if None in (mean_dh, mean_dl, mean_r1, mean_r2):
                continue
            # Map RS1/RS2 to Low/High via session dose
            if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
                rs_high = mean_r1; rs_low = mean_r2
            else:
                rs_high = mean_r2; rs_low = mean_r1
            rec = {
                'subject': subject,
                'RS_Low': rs_low,
                'RS_High': rs_high,
                'DMT_Low': mean_dl,
                'DMT_High': mean_dh,
            }
            records_by_minute[m].append(rec)

    # Write per-minute ANOVAs appended after global
    per_minute_text = run_anova_2x2_per_minute(records_by_minute, out_report_path=None)
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write(per_minute_text)
    print(f"📄 ANOVA 2x2 per-minute report appended to: {report_path}")

    # Add 0–5 minute ANOVA (first half) using aggregated means in 0..4 minutes
    records_first5 = []
    for subject in SUJETOS_VALIDADOS_EDA:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs1_path = build_rs_cvx_path(subject, 'session1')
        rs2_path = build_rs_cvx_path(subject, 'session2')
        dmt_high = load_cvx_component(dmt_high_path, component)
        dmt_low = load_cvx_component(dmt_low_path, component)
        rs1 = load_cvx_component(rs1_path, component)
        rs2 = load_cvx_component(rs2_path, component)
        if None in (dmt_high, dmt_low, rs1, rs2):
            continue
        def mean_0_5(pair):
            tt, yy = pair
            # Apply baseline correction for EDL
            yy = apply_edl_baseline_correction(tt, yy, component)
            mask = (tt >= 0.0) & (tt < 300.0)
            if not np.any(mask):
                return None
            return compute_mean_value(yy[mask])
        mean_dh = mean_0_5(dmt_high)
        mean_dl = mean_0_5(dmt_low)
        mean_r1 = mean_0_5(rs1)
        mean_r2 = mean_0_5(rs2)
        if None in (mean_dh, mean_dl, mean_r1, mean_r2):
            continue
        dose_s1 = get_dosis_sujeto(subject, 1)
        if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
            rs_high = mean_r1; rs_low = mean_r2
        else:
            rs_high = mean_r2; rs_low = mean_r1
        records_first5.append({
            'subject': subject,
            'RS_Low': rs_low,
            'RS_High': rs_high,
            'DMT_Low': mean_dl,
            'DMT_High': mean_dh,
        })
    if records_first5:
        res5_path = report_path
        res5 = run_anova_2x2_within(records_first5, out_report_path=None)
        # Render and append
        text5 = [
            '',
            '========================================',
            'Repeated-measures ANOVA 2x2 for first 5 minutes (0:00–4:59)',
            '========================================',
            _render_anova_results(res5)
        ]
        with open(res5_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(text5))
        print("📄 ANOVA 2x2 (first 5 minutes) appended to report")

    # Create 2x2 boxplot with paired data
    if anova_records:
        out_box = os.path.join(out_dir, f'{component.lower()}_mean_2x2_boxplot.png')
        _plot_boxplot_2x2(anova_records, out_box, component)
        print(f"🖼️  2x2 mean {component} boxplot saved: {out_box}")

        # Combined summary plot with DMT and RS subplots
        out_sum_combined = os.path.join(out_dir, f'summary_combined_dmt_rs_mean_sem_{component.lower()}.png')
        ok_combined = _plot_combined_summary(out_sum_combined, component)
        if ok_combined:
            print(f"🖼️  Combined summary DMT+RS mean±SEM saved: {out_sum_combined}")

        # For EDL only: additional DMT-only plot for 20 minutes
        if component == 'EDL':
            out_dmt_20min = os.path.join(out_dir, f'summary_dmt_only_20min_mean_sem_{component.lower()}.png')
            ok_dmt_20min = _plot_dmt_only_20min(out_dmt_20min, component)
            if ok_dmt_20min:
                print(f"🖼️  DMT-only 20-min EDL summary saved: {out_dmt_20min}")


def main():
    parser = argparse.ArgumentParser(description='Analyze EDA components (EDR or EDL)')
    parser.add_argument('--component', choices=['EDR', 'EDL'], required=True,
                        help='EDA component to analyze: EDR (phasic) or EDL (tonic)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting {args.component} analysis...")
    print(f"Component: {COMPONENT_CONFIG[args.component]['long_name']} ({COMPONENT_CONFIG[args.component]['description']})")
    
    run_component_analysis(args.component)
    
    print(f"\n✅ {args.component} analysis completed!")


if __name__ == '__main__':
    main()
