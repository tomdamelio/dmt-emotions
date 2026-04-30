#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compose Figure 1 from three pre-made panels (A, B, C).

Layout:
  - Panel A: left column (~2/5 width, full height)
  - Panel B: top-right (~3/5 width), scaled to 40% of original size
  - Panel C: bottom-right (~3/5 width, below B), scaled to 35% of original size

Usage:
    micromamba run -n dmt-emotions python scripts/compose_figure_1.py
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PANELS_DIR = PROJECT_ROOT / 'results' / 'figures' / 'panels_figure_1'
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'figures' / 'figure_1.png'

# Panel label styling (matches figure_config.py: FONT_SIZE_PANEL_LABEL=14, bold)
# At 300 DPI, 14pt ≈ 58px. We use a slightly larger size for the composite.
LABEL_FONT_SIZE = 36
LABEL_FONT_WEIGHT = 'bold'

# Try to load Arial Bold; fall back to default
try:
    LABEL_FONT = ImageFont.truetype("arialbd.ttf", LABEL_FONT_SIZE)
except OSError:
    try:
        LABEL_FONT = ImageFont.truetype("Arial Bold.ttf", LABEL_FONT_SIZE)
    except OSError:
        try:
            LABEL_FONT = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", LABEL_FONT_SIZE)
        except OSError:
            LABEL_FONT = ImageFont.load_default()
            print("[WARN] Could not load Arial Bold. Using default font.")

# Margins and padding (pixels)
OUTER_MARGIN = 60
PANEL_GAP = 30
LABEL_OFFSET_X = 10   # from left edge of panel region
LABEL_OFFSET_Y = -45  # above top edge of panel image


def load_and_scale(path: Path, scale: float) -> Image.Image:
    """Load an image and scale it by the given factor."""
    img = Image.open(path).convert('RGBA')
    if scale != 1.0:
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def main():
    # Load panels
    panel_a = Image.open(PANELS_DIR / 'fig1_panel_a.png').convert('RGBA')
    panel_b = load_and_scale(PANELS_DIR / 'fig1_panel_b.png', 0.40)
    panel_c = load_and_scale(PANELS_DIR / 'fig1_panel_c.png', 0.35)

    # Right column width = max of B and C widths
    right_col_w = max(panel_b.width, panel_c.width)
    # Right column height = B + gap + C
    right_col_h = panel_b.height + PANEL_GAP + panel_c.height

    # Scale panel A to match right column height
    a_scale = right_col_h / panel_a.height
    panel_a = panel_a.resize(
        (int(panel_a.width * a_scale), right_col_h), Image.LANCZOS
    )

    # Canvas dimensions
    canvas_w = OUTER_MARGIN + panel_a.width + PANEL_GAP + right_col_w + OUTER_MARGIN
    canvas_h = OUTER_MARGIN + right_col_h + OUTER_MARGIN

    # Create white canvas
    canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))

    # Paste panels
    # Panel A: left
    a_x = OUTER_MARGIN
    a_y = OUTER_MARGIN
    canvas.paste(panel_a, (a_x, a_y), panel_a)

    # Panel B: top-right, horizontally centered in right column
    b_x = OUTER_MARGIN + panel_a.width + PANEL_GAP + (right_col_w - panel_b.width) // 2
    b_y = OUTER_MARGIN
    canvas.paste(panel_b, (b_x, b_y), panel_b)

    # Panel C: bottom-right, horizontally centered in right column
    c_x = OUTER_MARGIN + panel_a.width + PANEL_GAP + (right_col_w - panel_c.width) // 2
    c_y = OUTER_MARGIN + panel_b.height + PANEL_GAP
    canvas.paste(panel_c, (c_x, c_y), panel_c)

    # Draw panel labels
    draw = ImageDraw.Draw(canvas)
    label_color = (0, 0, 0)

    # Label A
    draw.text(
        (a_x + LABEL_OFFSET_X, a_y + LABEL_OFFSET_Y),
        'A', font=LABEL_FONT, fill=label_color
    )
    # Label B
    draw.text(
        (OUTER_MARGIN + panel_a.width + PANEL_GAP + LABEL_OFFSET_X, b_y + LABEL_OFFSET_Y),
        'B', font=LABEL_FONT, fill=label_color
    )
    # Label C
    draw.text(
        (OUTER_MARGIN + panel_a.width + PANEL_GAP + LABEL_OFFSET_X, c_y + LABEL_OFFSET_Y),
        'C', font=LABEL_FONT, fill=label_color
    )

    # Save as PNG (300 DPI)
    canvas_rgb = canvas.convert('RGB')
    canvas_rgb.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"✓ Figure 1 saved to: {OUTPUT_PATH}")
    print(f"  Canvas size: {canvas_w} x {canvas_h} px")
    print(f"  Panel A: {panel_a.width} x {panel_a.height} px")
    print(f"  Panel B: {panel_b.width} x {panel_b.height} px (40% scale)")
    print(f"  Panel C: {panel_c.width} x {panel_c.height} px (35% scale)")


if __name__ == '__main__':
    main()
