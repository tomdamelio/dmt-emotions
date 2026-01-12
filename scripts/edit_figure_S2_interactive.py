#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure S2 Editor using Pylustrator

This script allows you to interactively edit Figure S2 (DMT ECG HR extended) using pylustrator.
The plot is generated directly with matplotlib, allowing you to edit individual components.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_S2_interactive.py

Instructions:
    1. The figure will open in an interactive window
    2. Click on any element to select it
    3. Drag to move, use handles to resize
    4. Right-click for more options
    5. Close the window to save changes
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pylustrator
pylustrator.start()

try:
    from figure_config import (
        DOUBLE_COL_WIDTH, FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, 
        FONT_SIZE_TICK_LABEL, FONT_SIZE_LEGEND
    )
except ImportError:
    DOUBLE_COL_WIDTH = 7.2
    FONT_SIZE_TITLE = 12
    FONT_SIZE_AXIS_LABEL = 11
    FONT_SIZE_TICK_LABEL = 10
    FONT_SIZE_LEGEND = 10

# Import ECG analysis functions
try:
    from run_ecg_hr_analysis import (
        SUJETOS_VALIDADOS_ECG, determine_sessions, build_ecg_paths, load_ecg_csv,
        build_rs_ecg_path, USE_SESSION_ZSCORE, ZSCORE_BY_SUBJECT, WINDOW_SIZE_SEC,
        zscore_with_subject_baseline, zscore_with_session_baseline,
        compute_hr_mean_per_window, _compute_fdr_significant_segments,
        COLOR_DMT_HIGH, COLOR_DMT_LOW, tab20c_colors
    )
except ImportError as e:
    print(f"[ERROR] Could not import ECG functions: {e}")
    sys.exit(1)

print("Loading and processing ECG data...")

# Generate the plot data
limit_sec = 1150.0
total_windows = int(np.floor(limit_sec / WINDOW_SIZE_SEC))
H_list, L_list = [], []

for subject in SUJETOS_VALIDADOS_ECG:
    try:
        high_session, low_session = determine_sessions(subject)
        p_high, p_low = build_ecg_paths(subject, high_session, low_session)
        d_high = load_ecg_csv(p_high)
        d_low = load_ecg_csv(p_low)
        if d_high is None or d_low is None:
            continue
        th_abs = d_high['time'].to_numpy()
        yh_abs = pd.to_numeric(d_high['ECG_Rate'], errors='coerce').to_numpy()
        tl_abs = d_low['time'].to_numpy()
        yl_abs = pd.to_numeric(d_low['ECG_Rate'], errors='coerce').to_numpy()
        
        if USE_SESSION_ZSCORE:
            p_rsh = build_rs_ecg_path(subject, high_session)
            p_rsl = build_rs_ecg_path(subject, low_session)
            r_high = load_ecg_csv(p_rsh)
            r_low = load_ecg_csv(p_rsl)
            if r_high is None or r_low is None:
                continue
            trh_abs = r_high['time'].to_numpy()
            yrh_abs = pd.to_numeric(r_high['ECG_Rate'], errors='coerce').to_numpy()
            trl_abs = r_low['time'].to_numpy()
            yrl_abs = pd.to_numeric(r_low['ECG_Rate'], errors='coerce').to_numpy()
            
            if ZSCORE_BY_SUBJECT:
                _, yh_z, _, yl_z, diag = zscore_with_subject_baseline(
                    trh_abs, yrh_abs, th_abs, yh_abs,
                    trl_abs, yrl_abs, tl_abs, yl_abs
                )
                if not diag['scalable']:
                    continue
                yh, yl = yh_z, yl_z
            else:
                _, yh_z, diag_h = zscore_with_session_baseline(trh_abs, yrh_abs, th_abs, yh_abs)
                _, yl_z, diag_l = zscore_with_session_baseline(trl_abs, yrl_abs, tl_abs, yl_abs)
                if not (diag_h['scalable'] and diag_l['scalable']):
                    continue
                yh, yl = yh_z, yl_z
        else:
            yh, yl = yh_abs, yl_abs
        
        hr_h = [compute_hr_mean_per_window(th_abs, yh, m) for m in range(total_windows)]
        hr_l = [compute_hr_mean_per_window(tl_abs, yl, m) for m in range(total_windows)]
        if None in hr_h or None in hr_l:
            continue
        H_list.append(np.array(hr_h, dtype=float))
        L_list.append(np.array(hr_l, dtype=float))
    except Exception:
        continue

if not (H_list and L_list):
    print("[ERROR] No valid ECG data found")
    sys.exit(1)

print(f"  ✓ Loaded data from {len(H_list)} subjects")

H = np.vstack(H_list)
L = np.vstack(L_list)
mean_h = np.nanmean(H, axis=0)
mean_l = np.nanmean(L, axis=0)
sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

x = np.arange(1, total_windows + 1)
time_minutes = (x - 0.5) * WINDOW_SIZE_SEC / 60.0

print("Creating figure...")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))

# Compute FDR significant segments
segs = _compute_fdr_significant_segments(H, L, x, alternative='greater')
for w0, w1 in segs:
    t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
    t1 = w1 * WINDOW_SIZE_SEC / 60.0
    ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)

# Plot lines
l1 = ax.plot(time_minutes, mean_h, color=COLOR_DMT_HIGH, lw=1.5, label='High dose (40mg)')[0]
ax.fill_between(time_minutes, mean_h - sem_h, mean_h + sem_h, color=COLOR_DMT_HIGH, alpha=0.25)
l2 = ax.plot(time_minutes, mean_l, color=COLOR_DMT_LOW, lw=1.5, label='Low dose (20mg)')[0]
ax.fill_between(time_minutes, mean_l - sem_l, mean_l + sem_l, color=COLOR_DMT_LOW, alpha=0.25)

# Legend
leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], 
               loc='upper right', frameon=True, fancybox=False, 
               fontsize=FONT_SIZE_LEGEND, borderpad=0.4)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.9)

# Labels
ax.set_xlabel('Time (minutes)', fontsize=FONT_SIZE_AXIS_LABEL)
ylabel_text = 'HR (Z-scored)' if USE_SESSION_ZSCORE else 'HR (bpm)'
ax.text(-0.12, 0.5, 'Electrocardiography', transform=ax.transAxes, 
        fontsize=FONT_SIZE_AXIS_LABEL, fontweight='bold', color=tab20c_colors[0],
        rotation=90, va='center', ha='center')
ax.text(-0.06, 0.5, ylabel_text, transform=ax.transAxes, 
        fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal', color='black', 
        rotation=90, va='center', ha='center')
ax.set_title('DMT', fontweight='bold', fontsize=FONT_SIZE_TITLE)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)
ax.grid(True, which='major', axis='y', alpha=0.25)
ax.grid(False, which='major', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# X-axis
ticks = list(range(0, 20))
ax.set_xticks(ticks)
ax.set_xlim(-0.2, 19.2)

plt.tight_layout()

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED - Figure S2")
print("="*60)
print("Instructions:")
print("  - Click on any element to select it")
print("  - Drag to move, use handles to resize")
print("  - Right-click for more options")
print("  - Close the window to save changes")
print("="*60 + "\n")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).axes[0].legend(loc=(0.7116, 0.6871))
plt.figure(1).axes[0].texts[0].set(position=(-0.0986, 0.5))
#% end: automatic generated code from pylustrator
plt.show()

print("\n✓ Changes saved!")
