#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure S1 Editor using Pylustrator

This script allows you to interactively edit Figure S1 (Stacked subjects - ECG, EDA, Resp) using pylustrator.
You can drag, resize, and reposition elements, and the changes will be saved.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_S1_interactive.py

Instructions:
    1. The figure will open in an interactive window
    2. Click on any element to select it
    3. Drag to move, use handles to resize
    4. Right-click for more options
    5. Close the window to save changes

Controls:
    - Left click: Select element
    - Drag: Move element
    - Handles: Resize element
    - Right click: Context menu
    - Ctrl+Z: Undo
    - Ctrl+Y: Redo
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import pylustrator FIRST (before creating any figures)
import pylustrator
pylustrator.start()

# Import figure configuration
try:
    from figure_config import DOUBLE_COL_WIDTH, PANEL_LABEL_SIZE
except ImportError:
    DOUBLE_COL_WIDTH = 7.2
    PANEL_LABEL_SIZE = 12

# Paths
ECG_ROOT = PROJECT_ROOT / 'results' / 'ecg'
EDA_ROOT = PROJECT_ROOT / 'results' / 'eda'
RESP_ROOT = PROJECT_ROOT / 'results' / 'resp'

def _load_image(path):
    """Load image from path."""
    if not path.exists():
        print(f"[WARN] Missing image: {path}")
        return None
    return mpimg.imread(str(path))

# Load images
print("Loading stacked subject plots...")
imgs = [
    _load_image(ECG_ROOT / 'hr' / 'plots' / 'stacked_subs_ecg_hr.png'),
    _load_image(EDA_ROOT / 'smna' / 'plots' / 'stacked_subs_smna.png'),
    _load_image(RESP_ROOT / 'rvt' / 'plots' / 'stacked_subs_resp_rvt.png'),
]

if any(img is None for img in imgs):
    print("[ERROR] Some images are missing. Run the analysis scripts first.")
    sys.exit(1)

print("  ✓ Images loaded successfully")

# Create figure
print("Creating figure...")
fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))
gs = fig.add_gridspec(1, 3, wspace=0.05, hspace=0.05,
                      left=0.01, right=0.99, top=0.95, bottom=0.05)

for i, (img, label) in enumerate(zip(imgs, ['A', 'B', 'C'])):
    ax = fig.add_subplot(gs[0, i])
    ax.axis('off')
    ax.imshow(img)
    # Add panel label
    pos = ax.get_position()
    fig.text(pos.x0 - 0.01, pos.y1 + 0.02, label, 
            fontsize=PANEL_LABEL_SIZE, fontweight='bold', ha='left', va='top')

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED - Figure S1")
print("="*60)
print("Instructions:")
print("  - Click on any element to select it")
print("  - Drag to move, use handles to resize")
print("  - Right-click for more options")
print("  - Close the window to save changes")
print("="*60 + "\n")

# Show the figure (pylustrator will make it interactive)
plt.show()

print("\n✓ Changes saved! You can now apply them to regenerate the figure.")
