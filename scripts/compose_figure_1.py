#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compose Figure 1: the experimental timeline.

Since the revision requested by Reviewer 1 (comment 1.2), Figure 1 shows the
four-stage timeline only. The recording-setup drawing and the processing
schematic that used to be panels a and c moved to Supplementary Fig. 1, which is
built by scripts/compose_figure_S1.py. The timeline needs no composition, so this
script simply copies the panel to results/figures/figure_1.png at its native
resolution.

Usage:
    micromamba run -n dmt-emotions python scripts/compose_figure_1.py
"""

from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
PANELS_DIR = PROJECT_ROOT / 'results' / 'figures' / 'panels_figure_1'
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'figures' / 'figure_1.png'

# The timeline is stored as fig1_panel_b.png for historical reasons: it was
# panel b of the three-panel figure of the original submission.
TIMELINE_PANEL = 'fig1_panel_b.png'
# Native resolution: at 	extwidth in the manuscript this is above 300 dpi,
# which the previous 0.40 downscale was not.
SCALE = 1.0


def main() -> Path:
    img = Image.open(PANELS_DIR / TIMELINE_PANEL).convert('RGB')
    img = img.resize((round(img.width * SCALE), round(img.height * SCALE)), Image.LANCZOS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Figure 1 saved to: {OUTPUT_PATH}")
    print(f"  Size: {img.width} x {img.height} px")
    return OUTPUT_PATH


if __name__ == '__main__':
    main()
