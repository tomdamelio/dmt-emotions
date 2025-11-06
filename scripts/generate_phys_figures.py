# -*- coding: utf-8 -*-
"""
Generate publication-ready composite panels for physiological figures (Nature Human Behaviour).

This script stitches already-generated PNG plots into final panels:

Figure 1 (3x2 grid - Combined HR, SCL, RVT):
  A (row1-left):  results/ecg/hr/plots/all_subs_ecg_hr.png
  B (row1-right): results/ecg/hr/plots/lme_coefficient_plot.png
  C (row2-left):  results/eda/scl/plots/all_subs_eda_scl.png
  D (row2-right): results/eda/scl/plots/lme_coefficient_plot.png
  E (row3-left):  results/resp/rvt/plots/all_subs_resp_rvt.png
  F (row3-right): results/resp/rvt/plots/lme_coefficient_plot.png

Figure S1 (single panel - DMT ECG HR extended):
  results/ecg/hr/plots/all_subs_dmt_ecg_hr.png

Figure S2 (1x2 horizontal - EDA SMNA):
  A (left):       results/eda/smna/plots/all_subs_smna.png
  B (right):      results/eda/smna/plots/lme_coefficient_plot.png

Figure S3 (1x3 horizontal - Stacked subjects for all modalities):
  A (left):       results/ecg/hr/plots/stacked_subs_ecg_hr.png
  B (center):     results/eda/scl/plots/stacked_subs_eda_scl.png
  C (right):      results/resp/rvt/plots/stacked_subs_resp_rvt.png

Outputs:
  results/figures/figure_1.png
  results/figures/figure_S1.png
  results/figures/figure_S2.png
  results/figures/figure_S3.png
"""

import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Get project root (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
EDA_ROOT = PROJECT_ROOT / 'results' / 'eda'
ECG_ROOT = PROJECT_ROOT / 'results' / 'ecg'
RESP_ROOT = PROJECT_ROOT / 'results' / 'resp'
OUT_DIR = PROJECT_ROOT / 'results' / 'figures'


def _load_image(path: str):
    if not os.path.exists(path):
        print(f"[WARN] Missing image: {path}")
        return None
    try:
        return mpimg.imread(path)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def _place(ax, img, label: str, label_xy: Tuple[float, float] = (-0.03, 1.08)):
    """Place image and label in subplot.
    
    Uses higher Y coordinate to ensure labels are well above plot area
    and aligned across rows.
    """
    ax.axis('off')
    if img is not None:
        # Preserve image aspect ratio
        ax.imshow(img)
    # Subplot label (A, B, C, ...)
    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=30,
        fontweight='bold',
        color='black',
    )


def create_figure_one() -> str:
    """Create Figure 1: Combined HR, SCL, and RVT analysis (3x2 grid).
    
    Row 1 (HR):  A = all_subs_ecg_hr.png,    B = lme_coefficient_plot.png
    Row 2 (SCL): C = all_subs_eda_scl.png,   D = lme_coefficient_plot.png
    Row 3 (RVT): E = all_subs_resp_rvt.png,  F = lme_coefficient_plot.png
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Row 1: HR
    A_path = str(ECG_ROOT / 'hr' / 'plots' / 'all_subs_ecg_hr.png')
    B_path = str(ECG_ROOT / 'hr' / 'plots' / 'lme_coefficient_plot.png')
    
    # Row 2: SCL
    C_path = str(EDA_ROOT / 'scl' / 'plots' / 'all_subs_eda_scl.png')
    D_path = str(EDA_ROOT / 'scl' / 'plots' / 'lme_coefficient_plot.png')
    
    # Row 3: RVT
    E_path = str(RESP_ROOT / 'rvt' / 'plots' / 'all_subs_resp_rvt.png')
    F_path = str(RESP_ROOT / 'rvt' / 'plots' / 'lme_coefficient_plot.png')

    imgs = [
        _load_image(A_path),  # A
        _load_image(B_path),  # B
        _load_image(C_path),  # C
        _load_image(D_path),  # D
        _load_image(E_path),  # E
        _load_image(F_path),  # F
    ]

    # GridSpec con márgenes mínimos y separación horizontal casi nula
    fig = plt.figure(figsize=(34, 27))
    gs = fig.add_gridspec(3, 2, 
                          wspace=-0.15,   # Espacio horizontal negativo para solapar levemente
                          hspace=0.08,    # Espacio vertical mínimo
                          left=0.01,      # Margen izquierdo
                          right=0.99,     # Margen derecho
                          top=0.99,       # Margen superior
                          bottom=0.01)    # Margen inferior
    
    axes = [
        fig.add_subplot(gs[0, 0]),  # A
        fig.add_subplot(gs[0, 1]),  # B
        fig.add_subplot(gs[1, 0]),  # C
        fig.add_subplot(gs[1, 1]),  # D
        fig.add_subplot(gs[2, 0]),  # E
        fig.add_subplot(gs[2, 1]),  # F
    ]
    
    # Place images
    for ax, img in zip(axes, imgs):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
    
    # Place labels using figure coordinates for perfect alignment
    # Get subplot positions
    positions = [ax.get_position() for ax in axes]
    
    # Place labels at same height for each row
    # Left column (A, C, E) needs extra offset because plots have titles
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    left_offset = 0.05  # Extra offset for left column to align with right
    right_offset = 0.02
    
    for i, (pos, label) in enumerate(zip(positions, labels)):
        # Even indices (0, 2, 4) are left column, odd indices (1, 3, 5) are right column
        offset = left_offset if i % 2 == 0 else right_offset
        fig.text(pos.x0 - 0.01, pos.y1 + offset, label, fontsize=30, fontweight='bold', ha='left', va='top')

    # Sanity check
    try:
        hA = axes[0].get_position().height
        hB = axes[1].get_position().height
        hC = axes[2].get_position().height
        hD = axes[3].get_position().height
        hE = axes[4].get_position().height
        hF = axes[5].get_position().height
        if abs(hA - hB) > 1e-3 or abs(hC - hD) > 1e-3 or abs(hE - hF) > 1e-3:
            print(f"[WARN] Row heights differ: A={hA:.4f}, B={hB:.4f}, C={hC:.4f}, D={hD:.4f}, E={hE:.4f}, F={hF:.4f}")
        else:
            print(f"[OK] Row heights equal: A=B={hA:.4f}, C=D={hC:.4f}, E=F={hE:.4f}")
    except Exception:
        pass
    
    out_path = str(OUT_DIR / 'figure_1.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_two() -> str:
    """Create Figure 2: Composite Autonomic Arousal Index (4 panels in 4x2 grid).
    
    Panel A (top-left, 1 row × 1 col): pca_scree.png
    Panel B (below A, 1 row × 1 col): pca_pc1_loadings.png
    Panel C (top-right, 2 rows × 1 col): lme_coefficient_plot.png
    Panel D (bottom, 2 rows × 2 cols): all_subs_composite.png
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_ROOT = PROJECT_ROOT / 'results' / 'composite' / 'plots'

    # New mapping: A=scree, B=loadings, C=lme_coef, D=all_subs
    A_path = str(COMPOSITE_ROOT / 'pca_scree.png')
    B_path = str(COMPOSITE_ROOT / 'pca_pc1_loadings.png')
    C_path = str(COMPOSITE_ROOT / 'lme_coefficient_plot.png')
    D_path = str(COMPOSITE_ROOT / 'all_subs_composite.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)
    C_img = _load_image(C_path)
    D_img = _load_image(D_path)

    # Create figure with 4x2 grid
    # Make first column narrower (60% of second column width) for A and B panels
    fig = plt.figure(figsize=(34, 27))
    gs = fig.add_gridspec(4, 2,
                          width_ratios=[0.6, 1.0],  # First column is 60% of second
                          wspace=0.02,    # Small positive spacing
                          hspace=0.08,    # Vertical spacing
                          left=0.01,
                          right=0.99,
                          top=0.99,
                          bottom=0.01)
    
    # Panel A: row 1 (index 0), left column
    ax_A = fig.add_subplot(gs[0, 0])
    
    # Panel B: row 2 (index 1), left column
    ax_B = fig.add_subplot(gs[1, 0])
    
    # Panel C: top 2 rows, right column
    ax_C = fig.add_subplot(gs[0:2, 1])
    
    # Panel D: bottom 2 rows, both columns
    ax_D = fig.add_subplot(gs[2:4, :])
    
    axes = [ax_A, ax_B, ax_C, ax_D]
    imgs = [A_img, B_img, C_img, D_img]
    
    # Place images
    for ax, img in zip(axes, imgs):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
    
    # Place labels using figure coordinates for perfect alignment
    positions = [ax.get_position() for ax in axes]
    labels = ['A', 'C', 'B', 'D']
    
    # Offsets for each panel
    # A: top panel, needs extra offset
    # B: left column bottom, needs extra offset
    # C, D: right column, standard offset
    offsets = [0.05, 0.05, 0.02, 0.02]
    
    for pos, label, offset in zip(positions, labels, offsets):
        fig.text(pos.x0 - 0.01, pos.y1 + offset, label, 
                fontsize=30, fontweight='bold', ha='left', va='top')

    out_path = str(OUT_DIR / 'figure_2.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S1() -> str:
    """Create Figure S1: DMT ECG HR extended timecourse (single panel)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Simply copy the DMT extended HR plot
    source_path = str(ECG_ROOT / 'hr' / 'plots' / 'all_subs_dmt_ecg_hr.png')
    img = _load_image(source_path)
    
    if img is None:
        print(f"[ERROR] Could not load {source_path}")
        return ""
    
    # Create a simple figure with the image
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(img)
    
    out_path = str(OUT_DIR / 'figure_S1.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S2() -> str:
    """Create Figure S2: EDA SMNA analysis (1x2 horizontal layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(EDA_ROOT / 'smna' / 'plots' / 'all_subs_smna.png')
    B_path = str(EDA_ROOT / 'smna' / 'plots' / 'lme_coefficient_plot.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)

    # GridSpec con separación mínima entre subplots
    fig = plt.figure(figsize=(34, 9))
    gs = fig.add_gridspec(1, 2,
                          wspace=-0.15,   # Espacio horizontal negativo para solapar levemente
                          hspace=0.0,
                          left=0.01,
                          right=0.99,
                          top=0.99,
                          bottom=0.01)
    
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]
    
    # Place images
    for ax, img in zip(axes, [A_img, B_img]):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
    
    # Place labels using figure coordinates for perfect alignment (same as figure_1)
    positions = [ax.get_position() for ax in axes]
    labels = ['A', 'B']
    left_offset = 0.135  # Extra offset for left column to align with right
    right_offset = 0.01
    
    for i, (pos, label) in enumerate(zip(positions, labels)):
        # i=0 is left column (A), i=1 is right column (B)
        offset = left_offset if i == 0 else right_offset
        fig.text(pos.x0 - 0.01, pos.y1 + offset, label, fontsize=30, fontweight='bold', ha='left', va='top')

    out_path = str(OUT_DIR / 'figure_S2.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S3() -> str:
    """Create Figure S3: Stacked subjects for all modalities (1x3 horizontal layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(ECG_ROOT / 'hr' / 'plots' / 'stacked_subs_ecg_hr.png')
    B_path = str(EDA_ROOT / 'scl' / 'plots' / 'stacked_subs_eda_scl.png')
    C_path = str(RESP_ROOT / 'rvt' / 'plots' / 'stacked_subs_resp_rvt.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)
    C_img = _load_image(C_path)

    # Create figure without GridSpec - use manual positioning
    fig = plt.figure(figsize=(30, 28))
    
    # Calculate aspect ratios for all images
    if A_img is not None:
        A_aspect = A_img.shape[0] / A_img.shape[1]
    if B_img is not None:
        B_aspect = B_img.shape[0] / B_img.shape[1]
    if C_img is not None:
        C_aspect = C_img.shape[0] / C_img.shape[1]
    
    # Define positions manually - all with same top (y1 = 0.99)
    # Each subplot gets equal width with almost no spacing
    subplot_width = 0.33
    spacing = -0.005
    top = 0.99
    
    # Calculate heights based on aspect ratios to maintain image proportions
    A_height = subplot_width * A_aspect if A_img is not None else 0.9
    B_height = subplot_width * B_aspect if B_img is not None else 0.9
    C_height = subplot_width * C_aspect if C_img is not None else 0.9
    
    # Create axes with manual positions [x0, y0, width, height]
    # All start from same top, but have different heights
    ax_A = fig.add_axes([0.01, top - A_height, subplot_width, A_height])
    ax_B = fig.add_axes([0.01 + subplot_width + spacing, top - B_height, subplot_width, B_height])
    ax_C = fig.add_axes([0.01 + 2*(subplot_width + spacing), top - C_height, subplot_width, C_height])
    
    axes = [ax_A, ax_B, ax_C]
    imgs = [A_img, B_img, C_img]
    
    # Place images
    for ax, img in zip(axes, imgs):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
    
    # Place labels using figure coordinates
    labels = ['A', 'B', 'C']
    offset = 0.02
    
    for ax, label in zip(axes, labels):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.005, pos.y1 + offset, label, fontsize=30, fontweight='bold', ha='left', va='top')

    out_path = str(OUT_DIR / 'figure_S3.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def main() -> None:
    fig1 = create_figure_one()
    print(f"Figure 1 saved to: {fig1}")
    fig2 = create_figure_two()
    print(f"Figure 2 saved to: {fig2}")
    figS1 = create_figure_S1()
    print(f"Figure S1 saved to: {figS1}")
    figS2 = create_figure_S2()
    print(f"Figure S2 saved to: {figS2}")
    figS3 = create_figure_S3()
    print(f"Figure S3 saved to: {figS3}")


if __name__ == '__main__':
    main()
