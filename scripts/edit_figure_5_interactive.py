#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure 5 Editor using Pylustrator

This script allows you to interactively edit Figure 5 (CCA Analysis) using pylustrator.
You can drag, resize, and reposition elements, and the changes will be saved to a .py file.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_5_interactive.py

Instructions:
    1. The figure will open in an interactive window
    2. Click on any element to select it
    3. Drag to move, use handles to resize
    4. Right-click for more options
    5. Close the window to save changes
    6. The changes will be saved to 'scripts/figure_5_layout.py'
    7. You can then apply these changes to regenerate the figure

Controls:
    - Left click: Select element
    - Drag: Move element
    - Handles: Resize element
    - Right click: Context menu with more options
    - Ctrl+Z: Undo
    - Ctrl+Y: Redo
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import pylustrator FIRST (before creating any figures)
import pylustrator
pylustrator.start()

# Import figure configuration
try:
    from figure_config import (
        FONT_SIZE_LEGEND_SMALL, DOUBLE_COL_WIDTH, apply_rcparams
    )
    PANEL_LABEL_SIZE = 12
except ImportError:
    PANEL_LABEL_SIZE = 12
    FONT_SIZE_LEGEND_SMALL = 7
    DOUBLE_COL_WIDTH = 7.2

# Paths
CCA_ROOT = PROJECT_ROOT / 'results' / 'coupling'

# Load data
print("Loading CCA data...")
try:
    loadings_df = pd.read_csv(CCA_ROOT / 'cca_loadings.csv')
    cv_folds_df = pd.read_csv(CCA_ROOT / 'cca_cross_validation_folds.csv')
    cv_summary_df = pd.read_csv(CCA_ROOT / 'cca_cross_validation_summary.csv')
    merged_data_df = pd.read_csv(CCA_ROOT / 'merged_physio_tet_data.csv')
    print("  ✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"[ERROR] Missing CCA data file: {e}")
    sys.exit(1)

# Set up plotting style
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 7, 'axes.labelsize': 7,
    'axes.titlesize': 8, 'xtick.labelsize': 6, 'ytick.labelsize': 7,
    'legend.fontsize': 6, 'axes.linewidth': 0.5,
})

# Colors
tab20c = plt.cm.tab20c.colors
physio_colors = {'HR': tab20c[0], 'SMNA_AUC': tab20c[4], 'RVT': tab20c[8]}
gray_colors = [tab20c[12], tab20c[13], tab20c[14], tab20c[15]]
color_rs, color_rs_edge = tab20c[17], tab20c[16]
color_dmt, color_dmt_edge = tab20c[13], tab20c[12]

# Create figure
print("Creating figure...")
fig_width = 183 / 25.4
fig = plt.figure(figsize=(fig_width, fig_width * 0.78))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], wspace=0.35, hspace=0.4)

ax_A, ax_B = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_C, ax_D = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# Prepare data
state, cv = 'DMT', 1
data = loadings_df[(loadings_df['state'] == state) & (loadings_df['canonical_variate'] == cv)]
physio_data = data[data['variable_set'] == 'physio'].sort_values('loading', ascending=True)
tet_data = data[data['variable_set'] == 'tet'].sort_values('loading', ascending=True)

physio_labels = {'HR': ('ECG', 'HR'), 'SMNA_AUC': ('EDA', 'SMNA'), 'RVT': ('Resp', 'RVT')}
tet_labels = {'emotional_intensity': 'Emotional Intensity', 'interoception': 'Interoception',
              'unpleasantness': 'Unpleasantness', 'pleasantness': 'Pleasantness',
              'bliss': 'Bliss', 'anxiety': 'Anxiety'}

# Panel A: Physio loadings
print("  Creating Panel A (Physio loadings)...")
for i, (_, row) in enumerate(physio_data.iterrows()):
    ax_A.barh(i, row['loading'], height=0.6, 
             color=physio_colors.get(row['variable_name'], '#666'), alpha=0.85)
ax_A.set_yticks(range(len(physio_data)))
ax_A.set_yticklabels([physio_labels.get(v, (v, v))[0] for v in physio_data['variable_name']])
ax_A.set_xlabel('Canonical Loading')
ax_A.set_title('Physiological Variables', fontweight='bold')
ax_A.axvline(0, color='black', linewidth=0.5, alpha=0.3)
ax_A.spines['top'].set_visible(False)
ax_A.spines['right'].set_visible(False)

# Panel B: TET loadings
print("  Creating Panel B (TET loadings)...")
for i, (_, row) in enumerate(tet_data.iterrows()):
    ax_B.barh(i, row['loading'], height=0.6, color=gray_colors[i % 4], alpha=0.85)
ax_B.set_yticks(range(len(tet_data)))
ax_B.set_yticklabels([tet_labels.get(v, v) for v in tet_data['variable_name']])
ax_B.set_xlabel('Canonical Loading')
ax_B.set_title('Affective Dimensions', fontweight='bold')
ax_B.axvline(0, color='black', linewidth=0.5, alpha=0.3)
ax_B.spines['top'].set_visible(False)
ax_B.spines['right'].set_visible(False)

# Panel C: CV boxplots
print("  Creating Panel C (Cross-validation)...")
cv_cv1 = cv_folds_df[cv_folds_df['canonical_variate'] == 1]
rs_cv = cv_cv1[cv_cv1['state'] == 'RS']['r_oos'].values
dmt_cv = cv_cv1[cv_cv1['state'] == 'DMT']['r_oos'].values

bp = ax_C.boxplot([rs_cv, dmt_cv], positions=[0.8, 1.6], widths=0.3, patch_artist=True)
bp['boxes'][0].set_facecolor(color_rs)
bp['boxes'][1].set_facecolor(color_dmt)
ax_C.set_xticks([0.8, 1.6])
ax_C.set_xticklabels(['RS', 'DMT'])
ax_C.set_ylabel('Out-of-sample r')
ax_C.set_title('Cross-validation (CV1)', fontweight='bold')
ax_C.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax_C.spines['top'].set_visible(False)
ax_C.spines['right'].set_visible(False)

# Panel D: Scatterplot
print("  Creating Panel D (In-sample coupling)...")
from sklearn.cross_decomposition import CCA
dmt_merged = merged_data_df[merged_data_df['state'] == 'DMT'].copy()
X = dmt_merged[['HR', 'SMNA_AUC', 'RVT']].values
Y = dmt_merged[['pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
                'interoception_z', 'bliss_z', 'anxiety_z']].values
subjects = dmt_merged['subject'].values

valid = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
X, Y, subjects = X[valid], Y[valid], subjects[valid]
X = (X - X.mean(0)) / X.std(0)
Y = (Y - Y.mean(0)) / Y.std(0)

cca = CCA(n_components=2)
cca.fit(X, Y)
U, V = cca.transform(X, Y)

tab20 = plt.cm.tab20.colors
for i, subj in enumerate(np.unique(subjects)):
    mask = subjects == subj
    ax_D.scatter(U[mask, 0], V[mask, 0], s=30, color=tab20[(i % 10) * 2 + 1],
                alpha=0.7, edgecolors=tab20[(i % 10) * 2], linewidths=0.5, label=subj)

z = np.polyfit(U[:, 0], V[:, 0], 1)
x_line = np.linspace(U[:, 0].min(), U[:, 0].max(), 100)
ax_D.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=1.5, alpha=0.8)

in_r = cv_summary_df[(cv_summary_df['state'] == 'DMT') & 
                     (cv_summary_df['canonical_variate'] == 1)]['in_sample_r'].values[0]
ax_D.text(0.98, 0.95, f'r = {in_r:.2f}', transform=ax_D.transAxes,
         ha='right', va='top', fontsize=FONT_SIZE_LEGEND_SMALL, style='italic', fontweight='bold')
ax_D.set_xlabel('Physiological Score (U1)')
ax_D.set_ylabel('TET Score (V1)')
ax_D.set_title('In-sample coupling (DMT)', fontweight='bold')
ax_D.spines['top'].set_visible(False)
ax_D.spines['right'].set_visible(False)
ax_D.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND_SMALL, ncol=2)

# Add panel labels
for ax, label, x_off in [(ax_A, 'A', -0.35), (ax_B, 'B', -0.15), 
                          (ax_C, 'C', -0.35), (ax_D, 'D', -0.15)]:
    ax.text(x_off, 1.15, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight='bold')

plt.tight_layout()

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED")
print("="*60)
print("Instructions:")
print("  - Click on any element to select it")
print("  - Drag to move, use handles to resize")
print("  - Right-click for more options")
print("  - Close the window to save changes")
print("\nChanges will be saved to: scripts/figure_5_layout.py")
print("="*60 + "\n")

# Show the figure (pylustrator will make it interactive)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(17.370000/2.54, 14.240000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.125, 0.5926, 0.2284, 0.2874])
plt.figure(1).axes[0].texts[0].set(position=(-0.3429, 1.098))
plt.figure(1).axes[1].set(position=[0.5443, 0.5831, 0.3557, 0.2969])
plt.figure(1).axes[1].texts[0].set(position=(-0.257, 1.095))
plt.figure(1).axes[3].legend(loc=(0.7973, 0.06528), borderpad=0.2, markerscale=0.9, handlelength=0., ncols=2, fontsize=8.)
plt.figure(1).axes[3].set(position=[0.5136, 0.11, 0.3864, 0.3172])
plt.figure(1).axes[3].texts[0].set(position=(1.004, 1.025), fontsize=8.)
plt.figure(1).axes[3].texts[1].set(position=(-0.1571, 1.15))
#% end: automatic generated code from pylustrator
plt.show()

print("\n✓ Changes saved! You can now apply them to regenerate the figure.")
