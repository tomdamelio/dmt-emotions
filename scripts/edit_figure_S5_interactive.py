#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure S5 Editor using Pylustrator

This script allows you to interactively edit Figure S5 (Arousal Index vs TET) using pylustrator.
You can drag, resize, and reposition elements, and the changes will be saved.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_S5_interactive.py

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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import pylustrator FIRST (before creating any figures)
import pylustrator
pylustrator.start()

# Import figure configuration
try:
    from figure_config import (
        FONT_SIZE_PANEL_LABEL, FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL,
        FONT_SIZE_TICK_LABEL, FONT_SIZE_LEGEND, DOUBLE_COL_WIDTH
    )
    PANEL_LABEL_SIZE = FONT_SIZE_PANEL_LABEL
except ImportError:
    PANEL_LABEL_SIZE = 12
    FONT_SIZE_TITLE = 10
    FONT_SIZE_AXIS_LABEL = 9
    FONT_SIZE_TICK_LABEL = 8
    FONT_SIZE_LEGEND = 8
    DOUBLE_COL_WIDTH = 7.2

# Paths
COUPLING_ROOT = PROJECT_ROOT / 'results' / 'coupling'

# Load data
print("Loading coupling data...")
try:
    merged_df = pd.read_csv(COUPLING_ROOT / 'merged_physio_tet_data.csv')
    regression_df = pd.read_csv(COUPLING_ROOT / 'regression_tet_arousal_index.csv')
    print("  ✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"[ERROR] Missing data file: {e}")
    sys.exit(1)

if 'ArousalIndex' not in merged_df.columns and 'physio_PC1' in merged_df.columns:
    merged_df['ArousalIndex'] = merged_df['physio_PC1']

COLOR_RS, COLOR_DMT, COLOR_LINE = '#9E9AC8', '#5E4FA2', '#D62728'

# Create figure
print("Creating figure...")
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.7))
outcomes = [('emotional_intensity_z', 'Emotional Intensity (Z)'),
            ('valence_index_z', 'Valence Index (Z)')]
states = [('RS', COLOR_RS), ('DMT', COLOR_DMT)]

for row, (outcome, ylabel) in enumerate(outcomes):
    for col, (state, color) in enumerate(states):
        ax = axes[row, col]
        state_data = merged_df[merged_df['state'] == state]
        
        reg = regression_df[(regression_df['outcome_variable'] == outcome) & 
                           (regression_df['state'] == state)]
        beta = reg['beta'].values[0] if len(reg) > 0 else np.nan
        r2 = reg['r_squared'].values[0] if len(reg) > 0 else np.nan
        pval = reg['p_value'].values[0] if len(reg) > 0 else np.nan
        
        x_col = 'ArousalIndex' if 'ArousalIndex' in state_data.columns else 'physio_PC1'
        if x_col in state_data.columns and outcome in state_data.columns:
            valid = state_data[[x_col, outcome]].dropna()
            ax.scatter(valid[x_col], valid[outcome], alpha=0.5, color=color, s=30, 
                      edgecolors='white', linewidths=0.5)
            
            if not np.isnan(beta) and len(valid) > 2:
                x_line = np.linspace(valid[x_col].min(), valid[x_col].max(), 100)
                slope, intercept = np.polyfit(valid[x_col], valid[outcome], 1)
                ax.plot(x_line, slope * x_line + intercept, color=COLOR_LINE, 
                       linewidth=1.5, alpha=0.8)
        
        if not np.isnan(beta):
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            ax.text(0.05, 0.95, f'β={beta:.2f}{sig}\nR²={r2:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=FONT_SIZE_LEGEND,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                            edgecolor='gray', linewidth=0.5))
        
        ax.set_title(state, fontweight='bold', fontsize=FONT_SIZE_TITLE)
        ax.set_xlabel('Arousal Index (Physio PC1)', fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel(ylabel if col == 0 else '', fontsize=FONT_SIZE_AXIS_LABEL)
        ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(-0.15, 1.08, ['A', 'B', 'C', 'D'][row * 2 + col], 
               transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight='bold')

plt.tight_layout()

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED - Figure S5")
print("="*60)
print("Instructions:")
print("  - Click on any element to select it")
print("  - Drag to move, use handles to resize")
print("  - Right-click for more options")
print("  - Close the window to save changes")
print("="*60 + "\n")

# Show the figure (pylustrator will make it interactive)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).axes[0].texts[0].set(position=(0.7659, 0.3625))
plt.figure(1).axes[1].texts[0].set(position=(0.7404, 0.3625))
plt.figure(1).axes[2].texts[0].set(position=(0.7659, 0.3509))
plt.figure(1).axes[3].texts[0].set(position=(0.7404, 0.3509))
#% end: automatic generated code from pylustrator
plt.show()

print("\n✓ Changes saved! You can now apply them to regenerate the figure.")
