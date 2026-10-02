#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compose Supplementary Figure 1 (recording setup and processing pipeline)
from two pre-made panels.

Layout:
  - Panel A: recording-setup drawing, left, at its native size
  - Panel B: processing-pipeline schematic, right, scaled so that its height
    matches panel A (it is therefore wider than panel A)

Usage:
    micromamba run -n dmt-emotions python scripts/compose_figure_S1.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
PANELS_DIR = PROJECT_ROOT / 'results' / 'figures' / 'panels_figure_1'
OUTPUT_PATH = PROJECT_ROOT / 'results' / 'figures' / 'figure_S1.png'

LABEL_FONT_SIZE = 64
OUTER_MARGIN = 40
PANEL_GAP = 60
LABEL_BAND = 90          # white band above the panels that holds the labels


def _font():
    for name in ("arialbd.ttf", "Arial Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, LABEL_FONT_SIZE)
        except OSError:
            continue
    print("[WARN] Arial Bold not found; using default font.")
    return ImageFont.load_default()


# Label fixes on the pipeline schematic (a raster with no editable source),
# in its native pixel coordinates (2816 x 1536):
#   - the EDA row repeats "sudomotor nerve activity (SMNA)" above and below the
#     SMNA box; the copy below is removed;
#   - the PCA output reads "Composite Arousal Index (PC1)"; the paper calls it
#     the Physiological Arousal Index, so the label is redrawn with that name
#     (same font size, colour and line spacing as the original; the canvas is
#     widened on the right to fit the longer first line).
DUPLICATE_SMNA_LABEL = (1350, 893, 1742, 1017)    # x0, y0, x1, y1
INDEX_LABEL_BOX = (2561, 678, 2816, 918)
INDEX_LABEL_LINES = ("Physiological", "Arousal", "Index", "(PC1)")
INDEX_LABEL_TOPS = (687, 744, 802, 858)            # top of each original line
INDEX_LABEL_FONT_SIZE = 49                         # matches the original "Composite"
INDEX_LABEL_LEFT = 2564
TEXT_RGB = (21, 20, 19)


def _fix_pipeline_labels(panel: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(panel)
    white = (255, 255, 255, 255)
    draw.rectangle(DUPLICATE_SMNA_LABEL, fill=white)
    draw.rectangle(INDEX_LABEL_BOX, fill=white)
    font = None
    for name in ("arial.ttf", "Arial.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        try:
            font = ImageFont.truetype(name, INDEX_LABEL_FONT_SIZE)
            break
        except OSError:
            continue
    if font is None:
        print("[WARN] Arial not found; using default font.")
        font = ImageFont.load_default()
    widths = [draw.textbbox((0, 0), t, font=font)[2] for t in INDEX_LABEL_LINES]
    centre = INDEX_LABEL_LEFT + max(widths) / 2
    new_w = max(panel.width, int(INDEX_LABEL_LEFT + max(widths) + 20))
    if new_w > panel.width:
        wider = Image.new('RGBA', (new_w, panel.height), white)
        wider.paste(panel, (0, 0))
        panel = wider
        draw = ImageDraw.Draw(panel)
    for text, w, top in zip(INDEX_LABEL_LINES, widths, INDEX_LABEL_TOPS):
        y_off = draw.textbbox((0, 0), text, font=font)[1]
        draw.text((centre - w / 2, top - y_off), text, font=font, fill=TEXT_RGB + (255,))
    return panel


def main() -> Path:
    panel_a = Image.open(PANELS_DIR / 'fig1_panel_a.png').convert('RGBA')
    # the processing-pipeline schematic is stored as fig1_panel_c.png
    panel_b = _fix_pipeline_labels(
        Image.open(PANELS_DIR / 'fig1_panel_c.png').convert('RGBA'))

    # Scale B so its height equals A's height
    scale_b = panel_a.height / panel_b.height
    panel_b = panel_b.resize(
        (round(panel_b.width * scale_b), panel_a.height), Image.LANCZOS)

    canvas_w = OUTER_MARGIN + panel_a.width + PANEL_GAP + panel_b.width + OUTER_MARGIN
    canvas_h = OUTER_MARGIN + LABEL_BAND + panel_a.height + OUTER_MARGIN
    canvas = Image.new('RGBA', (canvas_w, canvas_h), (255, 255, 255, 255))

    y = OUTER_MARGIN + LABEL_BAND
    a_x = OUTER_MARGIN
    b_x = OUTER_MARGIN + panel_a.width + PANEL_GAP
    canvas.paste(panel_a, (a_x, y), panel_a)
    canvas.paste(panel_b, (b_x, y), panel_b)

    draw = ImageDraw.Draw(canvas)
    font = _font()
    draw.text((a_x, OUTER_MARGIN), 'A', font=font, fill=(0, 0, 0))
    draw.text((b_x, OUTER_MARGIN), 'B', font=font, fill=(0, 0, 0))

    canvas.convert('RGB').save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Supplementary Figure 1 saved to: {OUTPUT_PATH}")
    print(f"  Canvas: {canvas_w} x {canvas_h} px; A {panel_a.size}; B {panel_b.size}")
    return OUTPUT_PATH


if __name__ == '__main__':
    main()
