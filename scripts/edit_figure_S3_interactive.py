#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Figure S3 Editor using Pylustrator

This script allows you to interactively edit Figure S3 (Stacked subjects composite) using pylustrator.

Usage:
    micromamba run -n dmt-emotions python scripts/edit_figure_S3_interactive.py
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pylustrator
pylustrator.start()

try:
    from figure_config import DOUBLE_COL_WIDTH
except ImportError:
    DOUBLE_COL_WIDTH = 7.2

COMPOSITE_ROOT = PROJECT_ROOT / 'results' / 'composite'

print("Loading stacked subjects composite plot...")
img_path = COMPOSITE_ROOT / 'plots' / 'stacked_subs_composite.png'

if not img_path.exists():
    print(f"[ERROR] Image not found: {img_path}")
    print("Run: micromamba run -n dmt-emotions python src/run_composite_arousal_index.py")
    sys.exit(1)

img = mpimg.imread(str(img_path))
print("  ✓ Image loaded successfully")

print("Creating figure...")
fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 1.2))
ax = fig.add_subplot(1, 1, 1)
ax.axis('off')
ax.imshow(img)

print("\n" + "="*60)
print("INTERACTIVE EDITOR OPENED - Figure S3")
print("="*60)
print("Instructions:")
print("  - Click on any element to select it")
print("  - Drag to move, use handles to resize")
print("  - Close the window to save changes")
print("="*60 + "\n")

plt.show()

print("\n✓ Changes saved!")
