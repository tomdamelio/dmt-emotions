# -*- coding: utf-8 -*-
"""
Plot per-subject SMNA AUC comparison for EDA (first 10 minutes): DMT high vs DMT low.

For each subject in SUJETOS_VALIDADOS_EDA, this script:
- Finds the correct session for high vs low using get_dosis_sujeto
- Loads CVX decomposition CSVs (…_cvx_decomposition.csv)
- Extracts the SMNA series and time (or synthesizes time from sampling rate)
- Trims both recordings to the first 10 minutes
- Computes AUC (total) via trapezoidal rule
- Plots the SMNA envelope (smoothed magnitude) with filled AUC (semi-transparent)
- Saves to test/eda/smna_10min

Usage:
  python test/plot_eda_smna_auc.py
"""

import os
import sys
import json
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


def _plot_boxplot_2x2(records, out_path):
    """Create a comparative 2x2 plot (RS/DMT x Low/High) with paired lines per subject.
    Uses fixed positions so we can connect the four points of each participant.
    """
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
    _beautify_axes(ax, title='SMNA AUC – 2×2 design (paired lines)', xlabel='Dose', ylabel='AUC (first 10 min)', time_formatter=False)
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
    """Build RS CVX decomposition path for a given subject/session using session dose.
    session: 'session1' or 'session2'
    Note: RS files are stored under the dose-specific directory (dmt_high/dmt_low)
    matching that session's dose.
    """
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
        print(f"⚠️  Missing CVX file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Failed to read CVX file {os.path.basename(csv_path)}: {e}")
        return None

    if 'SMNA' not in df.columns:
        print(f"⚠️  SMNA column not found in {os.path.basename(csv_path)}")
        return None

    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)

    smna = pd.to_numeric(df['SMNA'], errors='coerce').fillna(0.0).to_numpy()
    return t, smna


def compute_auc(t: np.ndarray, y: np.ndarray) -> float:
    """Compute total AUC using trapezoidal integration."""
    return float(np.trapz(y, t))


def moving_average(x: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return x
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    return np.convolve(x, kernel, mode='same')


def compute_envelope(y: np.ndarray, sr: float, window_sec: float = 2.0) -> np.ndarray:
    """Return a smoothed magnitude envelope for y using moving-average of |y|."""
    window_samples = max(1, int(round(sr * window_sec)))
    y_abs = np.abs(y)
    return moving_average(y_abs, window_samples)


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


def _plot_summary_high_low(kind: str, out_path: str) -> bool:
    """Create a across-subject summary plot with High vs Low mean ± SEM over time.
    kind: 'DMT' or 'RS'
    """
    assert kind in ('DMT', 'RS')
    # Common grid: 0..600s, 2 Hz for readability
    t_grid = np.arange(0.0, 600.0, 0.5)
    high_curves = []
    low_curves = []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            if kind == 'DMT':
                high_session, low_session = determine_sessions(subject)
                p_high, p_low = build_cvx_paths(subject, high_session, low_session)
                d_high = load_cvx_smna(p_high)
                d_low = load_cvx_smna(p_low)
                if None in (d_high, d_low):
                    continue
                th, yh = d_high; tl, yl = d_low
            else:  # RS
                p_r1 = build_rs_cvx_path(subject, 'session1')
                p_r2 = build_rs_cvx_path(subject, 'session2')
                r1 = load_cvx_smna(p_r1)
                r2 = load_cvx_smna(p_r2)
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

    if not high_curves or not low_curves:
        return False

    H = np.vstack(high_curves)
    L = np.vstack(low_curves)
    # Mean ± SEM
    mean_h = np.nanmean(H, axis=0)
    mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # Colors: High = dark gray; Low = blue
    c_high = '#555555'
    c_low = '#2A9FD6'
    ax.plot(t_grid, mean_h, color=c_high, lw=1.8, marker=None, label=f'{kind} High')
    ax.fill_between(t_grid, mean_h - sem_h, mean_h + sem_h, color=c_high, alpha=0.20)
    ax.plot(t_grid, mean_l, color=c_low, lw=1.8, marker=None, label=f'{kind} Low')
    ax.fill_between(t_grid, mean_l - sem_l, mean_l + sem_l, color=c_low, alpha=0.20)
    _beautify_axes(ax, title=f'{kind} – SMNA activity (mean ± SEM, first 10 min)', xlabel='Time (mm:ss)', ylabel='SMNA (a.u.)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return True


def plot_subject_smna_combined(subject: str) -> bool:
    """Generate and save a combined figure with two subplots for one subject:
    - Left: DMT High vs Low (first 10 minutes)
    - Right: RS ses01 vs ses02 with High/Low labels (first 10 minutes)
    """
    # DMT data
    high_session, low_session = determine_sessions(subject)
    dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
    dmt_high = load_cvx_smna(dmt_high_path)
    dmt_low = load_cvx_smna(dmt_low_path)
    if dmt_high is None or dmt_low is None:
        print(f"⚠️  Skipping {subject}: missing DMT data")
        return False

    t_high, smna_high = dmt_high
    t_low, smna_low = dmt_low
    ten_min = 600.0
    idx_h = np.where(t_high <= ten_min)[0]
    idx_l = np.where(t_low <= ten_min)[0]
    if len(idx_h) < 10 or len(idx_l) < 10:
        print(f"⚠️  Skipping {subject}: insufficient DMT samples in first 10 minutes")
        return False
    t_high_10 = t_high[idx_h]
    t_low_10 = t_low[idx_l]
    y_high_10 = smna_high[idx_h]
    y_low_10 = smna_low[idx_l]
    n_dmt = min(len(t_high_10), len(y_high_10), len(t_low_10), len(y_low_10))
    if n_dmt < 10:
        print(f"⚠️  Skipping {subject}: insufficient overlapping DMT samples after trim")
        return False
    t_dmt = t_high_10[:n_dmt]
    y_dmt_high = y_high_10[:n_dmt]
    y_dmt_low = y_low_10[:n_dmt]
    auc_dmt_high = compute_auc(t_dmt, y_dmt_high)
    auc_dmt_low = compute_auc(t_dmt, y_dmt_low)

    # RS data
    rs1_path = build_rs_cvx_path(subject, 'session1')
    rs2_path = build_rs_cvx_path(subject, 'session2')
    rs1 = load_cvx_smna(rs1_path)
    rs2 = load_cvx_smna(rs2_path)
    if rs1 is None or rs2 is None:
        print(f"⚠️  Skipping {subject}: missing RS data (ses1 or ses2)")
        return False
    t1, y1 = rs1
    t2, y2 = rs2
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
    auc_rs1 = compute_auc(t_rs, y_rs1)
    auc_rs2 = compute_auc(t_rs, y_rs2)

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

    # Subplot 1: DMT high vs low
    ax1.plot(t_dmt, y_dmt_high, color='tab:red', lw=1.4, label=f'DMT High (AUC={auc_dmt_high:.1f})')
    ax1.fill_between(t_dmt, 0, y_dmt_high, color='tab:red', alpha=0.30)
    ax1.plot(t_dmt, y_dmt_low, color='tab:blue', lw=1.4, label=f'DMT Low (AUC={auc_dmt_low:.1f})')
    ax1.fill_between(t_dmt, 0, y_dmt_low, color='tab:blue', alpha=0.30)
    _beautify_axes(ax1, title='DMT High vs Low (first 10 min)', xlabel='Time', ylabel='SMNA (a.u.)')
    ax1.legend(loc='upper right')

    # Subplot 2: RS ses01 vs ses02 with dose tags
    ax2.plot(t_rs, y_rs1, color='tab:green', lw=1.4, label=f'RS ses01 ({cond1}) (AUC={auc_rs1:.1f})')
    ax2.fill_between(t_rs, 0, y_rs1, color='tab:green', alpha=0.30)
    ax2.plot(t_rs, y_rs2, color='tab:purple', lw=1.4, label=f'RS ses02 ({cond2}) (AUC={auc_rs2:.1f})')
    ax2.fill_between(t_rs, 0, y_rs2, color='tab:purple', alpha=0.30)
    _beautify_axes(ax2, title='RS ses01 vs ses02 (first 10 min)', xlabel='Time (mm:ss)')
    ax2.legend(loc='upper right')

    fig.suptitle(f"{subject} – EDA SMNA AUC (first 10 min)", y=1.02)
    fig.tight_layout(pad=1.2)

    out_dir = os.path.join('test', 'eda', 'smna_10min')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subject.lower()}_eda_smna_auc_combined_10min.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved combined SMNA AUC plot: {out_path}")
    return True


def main():
    print("📊 Generating EDA SMNA AUC combined plots per validated subject…")
    successes_combined = 0
    # Collect per-subject AUCs for 2x2 ANOVA
    anova_records = []
    for subject in SUJETOS_VALIDADOS_EDA:
        ok = plot_subject_smna_combined(subject)
        if ok:
            successes_combined += 1
        # Compute and store AUCs for all four cells for ANOVA
        # Build paths
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs1_path = build_rs_cvx_path(subject, 'session1')
        rs2_path = build_rs_cvx_path(subject, 'session2')
        # Load
        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs1 = load_cvx_smna(rs1_path)
        rs2 = load_cvx_smna(rs2_path)
        if None in (dmt_high, dmt_low, rs1, rs2):
            continue
        # Trim to 10 minutes and compute AUC consistently
        def trim_and_auc(pair):
            tt, yy = pair
            idx = np.where(tt <= 600.0)[0]
            if len(idx) < 10:
                return None
            tt = tt[idx]
            yy = yy[idx]
            return compute_auc(tt, yy)
        auc_dmt_high = trim_and_auc(dmt_high)
        auc_dmt_low = trim_and_auc(dmt_low)
        auc_rs1 = trim_and_auc(rs1)
        auc_rs2 = trim_and_auc(rs2)
        if None in (auc_dmt_high, auc_dmt_low, auc_rs1, auc_rs2):
            continue
        # Map RS1/RS2 to Low/High using session dose
        dose_s1 = get_dosis_sujeto(subject, 1)
        dose_s2 = get_dosis_sujeto(subject, 2)
        if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
            rs_high = auc_rs1; rs_low = auc_rs2
        else:
            rs_high = auc_rs2; rs_low = auc_rs1
        # Map DMT High/Low already aligned by determine_sessions
        rec = {
            'subject': subject,
            'RS_Low': rs_low,
            'RS_High': rs_high,
            'DMT_Low': auc_dmt_low,
            'DMT_High': auc_dmt_high,
        }
        anova_records.append(rec)
    print(f"🎯 Completed combined plots: {successes_combined}/{len(SUJETOS_VALIDADOS_EDA)}")
    # Run 2x2 within-subject ANOVA (global) and per-minute ANOVAs; write report
    report_path = os.path.join('test', 'eda', 'smna_10min', 'anova_2x2_report.txt')
    if anova_records:
        _ = run_anova_2x2_within(anova_records, out_report_path=report_path)
        print(f"📄 ANOVA 2x2 report (global) saved: {report_path}")

    # Build per-minute records: for m in 0..9, compute per-subject AUCs in that 1-min window
    records_by_minute = {m: [] for m in range(10)}
    for subject in SUJETOS_VALIDADOS_EDA:
        # Gather sources as before
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs1_path = build_rs_cvx_path(subject, 'session1')
        rs2_path = build_rs_cvx_path(subject, 'session2')
        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs1 = load_cvx_smna(rs1_path)
        rs2 = load_cvx_smna(rs2_path)
        if None in (dmt_high, dmt_low, rs1, rs2):
            continue

        # Minute windows
        def auc_window(pair, m):
            tt, yy = pair
            mask = (tt >= 60.0 * m) & (tt < 60.0 * (m + 1))
            if not np.any(mask):
                return None
            return compute_auc(tt[mask], yy[mask])

        dose_s1 = get_dosis_sujeto(subject, 1)
        # For each minute, compute all four cells
        for m in range(10):
            auc_dh = auc_window(dmt_high, m)
            auc_dl = auc_window(dmt_low, m)
            auc_r1 = auc_window(rs1, m)
            auc_r2 = auc_window(rs2, m)
            if None in (auc_dh, auc_dl, auc_r1, auc_r2):
                continue
            # Map RS1/RS2 to Low/High via session dose
            if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
                rs_high = auc_r1; rs_low = auc_r2
            else:
                rs_high = auc_r2; rs_low = auc_r1
            rec = {
                'subject': subject,
                'RS_Low': rs_low,
                'RS_High': rs_high,
                'DMT_Low': auc_dl,
                'DMT_High': auc_dh,
            }
            records_by_minute[m].append(rec)

    # Write per-minute ANOVAs appended after global
    per_minute_text = run_anova_2x2_per_minute(records_by_minute, out_report_path=None)
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write('\n\n')
        f.write(per_minute_text)
    print(f"📄 ANOVA 2x2 per-minute report appended to: {report_path}")

    # Add 0–5 minute ANOVA (first half) using aggregated AUCs in 0..4 minutes
    records_first5 = []
    for subject in SUJETOS_VALIDADOS_EDA:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs1_path = build_rs_cvx_path(subject, 'session1')
        rs2_path = build_rs_cvx_path(subject, 'session2')
        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs1 = load_cvx_smna(rs1_path)
        rs2 = load_cvx_smna(rs2_path)
        if None in (dmt_high, dmt_low, rs1, rs2):
            continue
        def auc_0_5(pair):
            tt, yy = pair
            mask = (tt >= 0.0) & (tt < 300.0)
            if not np.any(mask):
                return None
            return compute_auc(tt[mask], yy[mask])
        auc_dh = auc_0_5(dmt_high)
        auc_dl = auc_0_5(dmt_low)
        auc_r1 = auc_0_5(rs1)
        auc_r2 = auc_0_5(rs2)
        if None in (auc_dh, auc_dl, auc_r1, auc_r2):
            continue
        dose_s1 = get_dosis_sujeto(subject, 1)
        if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
            rs_high = auc_r1; rs_low = auc_r2
        else:
            rs_high = auc_r2; rs_low = auc_r1
        records_first5.append({
            'subject': subject,
            'RS_Low': rs_low,
            'RS_High': rs_high,
            'DMT_Low': auc_dl,
            'DMT_High': auc_dh,
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
        out_dir = os.path.join('test', 'eda', 'smna_10min')
        os.makedirs(out_dir, exist_ok=True)
        out_box = os.path.join(out_dir, 'smna_auc_2x2_boxplot.png')
        _plot_boxplot_2x2(anova_records, out_box)
        print(f"🖼️  2x2 AUC boxplot saved: {out_box}")

        # Summary mean±SEM curves across subjects for DMT and RS
        out_sum_dmt = os.path.join(out_dir, 'summary_dmt_high_low_mean_sem.png')
        out_sum_rs = os.path.join(out_dir, 'summary_rs_high_low_mean_sem.png')
        ok_dmt = _plot_summary_high_low('DMT', out_sum_dmt)
        ok_rs = _plot_summary_high_low('RS', out_sum_rs)
        if ok_dmt:
            print(f"🖼️  Summary DMT mean±SEM saved: {out_sum_dmt}")
        if ok_rs:
            print(f"🖼️  Summary RS mean±SEM saved: {out_sum_rs}")


if __name__ == '__main__':
    main()


