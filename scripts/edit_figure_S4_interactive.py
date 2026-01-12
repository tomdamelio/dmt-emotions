#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure S4 Editor using Pylustrator

This script allows you to interactively edit Figure S4 (DMT Composite extended) using pylustrator.
The plot is generated directly with matplotlib, allowing you to edit individual components.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_S4_interactive.py

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
import warnings

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

# Import composite analysis functions
try:
    from run_composite_arousal_index import (
        load_and_prepare, zscore_within_subject, compute_pca_and_index,
        _compute_fdr_results, WINDOW_SIZE_SEC,
        COLOR_DMT_HIGH, COLOR_DMT_LOW, tab20b_colors
    )
except ImportError as e:
    print(f"[ERROR] Could not import composite functions: {e}")
    sys.exit(1)

print("Loading and processing composite data...")

# Load and prepare data
df = load_and_prepare(limit_to_9min=False)
df = zscore_within_subject(df)
df, variance_explained, loadings, sign_flip_info = compute_pca_and_index(df)

# Extract DMT data only
dmt_df = df[df['State'] == 'DMT'].copy()

if len(dmt_df) == 0:
    print("[ERROR] No DMT data found")
    sys.exit(1)

subjects = sorted(dmt_df['subject'].unique())
max_window = int(dmt_df['window'].max())

print(f"  ✓ Loaded data from {len(subjects)} subjects, {max_window} windows")

# Build matrices
high_mat, low_mat = [], []
for subject in subjects:
    subj_df = dmt_df[dmt_df['subject'] == subject].sort_values('window')
    high_vals = np.full(max_window, np.nan)
    low_vals = np.full(max_window, np.nan)
    
    high_data = subj_df[subj_df['Dose'] == 'High']
    low_data = subj_df[subj_df['Dose'] == 'Low']
    
    for _, row in high_data.iterrows():
        window_idx = int(row['window']) - 1
        if 0 <= window_idx < max_window:
            high_vals[window_idx] = row['ArousalIndex']
    
    for _, row in low_data.iterrows():
        window_idx = int(row['window']) - 1
        if 0 <= window_idx < max_window:
            low_vals[window_idx] = row['ArousalIndex']
    
    if not np.all(np.isnan(high_vals)) and not np.all(np.isnan(low_vals)):
        high_mat.append(high_vals)
        low_mat.append(low_vals)

H = np.array(high_mat)
L = np.array(low_mat)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    mean_h = np.nanmean(H, axis=0)
    mean_l = np.nanmean(L, axis=0)
    n_valid_h = np.sum(~np.isnan(H), axis=0)
    n_valid_l = np.sum(~np.isnan(L), axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(n_valid_h)
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(n_valid_l)

window_grid = np.arange(1, max_window + 1)
time_grid = (window_grid - 0.5) * WINDOW_SIZE_SEC / 60.0

# Compute FDR
fdr_results = _compute_fdr_results(H, L, window_grid, alternative='greater')
segments = fdr_results.get('segments', [])

print("Creating figure...")

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))

# Shade significant segments
for w0, w1 in segments:
    t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
    t1 = w1 * WINDOW_SIZE_SEC / 60.0
    ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)

# Plot lines
l1 = ax.plot(time_grid, mean_h, color=COLOR_DMT_HIGH, lw=1.5, 
             marker='o', markersize=3, label='High dose (40mg)')[0]
ax.fill_between(time_grid, mean_h - sem_h, mean_h + sem_h, 
                color=COLOR_DMT_HIGH, alpha=0.25)

l2 = ax.plot(time_grid, mean_l, color=COLOR_DMT_LOW, lw=1.5, 
             marker='o', markersize=3, label='Low dose (20mg)')[0]
ax.fill_between(time_grid, mean_l - sem_l, mean_l + sem_l, 
                color=COLOR_DMT_LOW, alpha=0.25)

# Legend
leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], 
               loc='upper right', frameon=True, fancybox=False, 
               fontsize=FONT_SIZE_LEGEND, borderpad=0.4)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(0.9)

# Labels
ax.set_xlabel('Time (minutes)', fontsize=FONT_SIZE_AXIS_LABEL)
ax.text(-0.12, 0.5, 'Composite Arousal', transform=ax.transAxes, 
        fontsize=FONT_SIZE_AXIS_LABEL, fontweight='bold', color=tab20b_colors[8],
        rotation=90, va='center', ha='center')
ax.text(-0.06, 0.5, 'Index (PC1)', transform=ax.transAxes, 
        fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal', color='black', 
        rotation=90, va='center', ha='center')
ax.set_title('DMT', fontweight='bold', fontsize=FONT_SIZE_TITLE)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)
ax.grid(True, which='major', axis='y', alpha=0.25)
ax.grid(False, which='major', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# X-axis
max_time_min = max_window * WINDOW_SIZE_SEC / 60.0
max_tick = int(np.ceil(max_time_min))
ax.set_xticks(list(range(0, max_tick + 1)))
ax.set_xlim(-0.2, max_time_min + 0.2)

plt.tight_layout()

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED - Figure S4")
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
plt.figure(1).axes[0].legend(loc=(0.698, 0.7257))
plt.figure(1).axes[0].texts[0].set(position=(-0.0967, 0.5))
#% end: automatic generated code from pylustrator
plt.show()

print("\n✓ Changes saved!")
