# -*- coding: utf-8 -*-
"""
Generate publication-ready composite panels for physiological figures (Nature Human Behaviour).

This script stitches already-generated PNG plots into final panels:

Panel 1 (2x2 grid - EDA):
  A (top-left):   results/eda/scl/plots/all_subs_eda_scl.png
  B (top-right):  results/eda/scl/plots/lme_coefficient_plot.png
  C (bottom-left):results/eda/smna/plots/all_subs_smna.png
  D (bottom-right):results/eda/smna/plots/lme_coefficient_plot.png

Panel 2 (1x2 vertical - EDA DMT):
  A (top):        results/eda/scl/plots/all_subs_dmt_eda_scl.png
  B (bottom):     results/eda/smna/plots/all_subs_dmt_smna.png

Panel 3 (1x2 horizontal - EDA stacked):
  A (left):       results/eda/scl/plots/stacked_subs_eda_scl.png
  B (right):      results/eda/smna/plots/stacked_subs_smna.png

Panel 4 (1x2 horizontal - ECG HR):
  A (left):       results/ecg/hr/plots/all_subs_ecg_hr.png
  B (right):      results/ecg/hr/plots/lme_coefficient_plot.png

Panel 5 (1x2 horizontal - Respiratory RVT):
  A (left):       results/resp/rvt/plots/all_subs_resp_rvt.png
  B (right):      results/resp/rvt/plots/lme_coefficient_plot.png

Panel 6 (3x2 grid - Combined HR, SCL, RVT):
  A (row1-left):  results/ecg/hr/plots/all_subs_ecg_hr.png
  B (row1-right): results/ecg/hr/plots/lme_coefficient_plot.png
  C (row2-left):  results/eda/scl/plots/all_subs_eda_scl.png
  D (row2-right): results/eda/scl/plots/lme_coefficient_plot.png
  E (row3-left):  results/resp/rvt/plots/all_subs_resp_rvt.png
  F (row3-right): results/resp/rvt/plots/lme_coefficient_plot.png

Outputs:
  results/figures/panel_1.png
  results/figures/panel_2.png
  results/figures/panel_3.png
  results/figures/panel_4.png
  results/figures/panel_5.png
  results/figures/panel_6.png
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


def create_panel_one() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(EDA_ROOT / 'scl' / 'plots' / 'all_subs_eda_scl.png')
    B_path = str(EDA_ROOT / 'scl' / 'plots' / 'lme_coefficient_plot.png')
    C_path = str(EDA_ROOT / 'smna' / 'plots' / 'all_subs_smna.png')
    D_path = str(EDA_ROOT / 'smna' / 'plots' / 'lme_coefficient_plot.png')

    imgs = [
        _load_image(A_path),
        _load_image(B_path),
        _load_image(C_path),
        _load_image(D_path),
    ]

    # GridSpec con márgenes mínimos y separación horizontal casi nula
    fig = plt.figure(figsize=(34, 18))
    gs = fig.add_gridspec(2, 2, 
                          wspace=-0.15,   # Espacio horizontal negativo para solapar levemente
                          hspace=0.08,    # Espacio vertical mínimo
                          left=0.01,      # Margen izquierdo
                          right=0.99,     # Margen derecho
                          top=0.99,       # Margen superior
                          bottom=0.01)    # Margen inferior
    
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    
    # Place images
    for ax, img in zip(axes, imgs):
        ax.axis('off')
        if img is not None:
            ax.imshow(img)
    
    # Place labels using figure coordinates for perfect alignment
    # Get subplot positions
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()
    pos3 = axes[3].get_position()
    
    # Place labels at same height for each row
    # Left column (A, C) needs extra offset because plots have titles
    left_offset = 0.075  # Extra offset for left column to align with right
    right_offset = 0.02
    
    fig.text(pos0.x0 - 0.01, pos0.y1 + left_offset, 'A', fontsize=30, fontweight='bold', ha='left', va='top')
    fig.text(pos1.x0 - 0.01, pos1.y1 + right_offset, 'B', fontsize=30, fontweight='bold', ha='left', va='top')
    fig.text(pos2.x0 - 0.01, pos2.y1 + left_offset, 'C', fontsize=30, fontweight='bold', ha='left', va='top')
    fig.text(pos3.x0 - 0.01, pos3.y1 + right_offset, 'D', fontsize=30, fontweight='bold', ha='left', va='top')

    # Sanity check
    try:
        hA = axes[0].get_position().height
        hB = axes[1].get_position().height
        hC = axes[2].get_position().height
        hD = axes[3].get_position().height
        if abs(hA - hB) > 1e-3 or abs(hC - hD) > 1e-3:
            print(f"[WARN] Row heights differ: A={hA:.4f}, B={hB:.4f}, C={hC:.4f}, D={hD:.4f}")
        else:
            print(f"[OK] Row heights equal: A=B={hA:.4f}, C=D={hC:.4f}")
    except Exception:
        pass
    
    out_path = str(OUT_DIR / 'panel_1.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')#, pad_inches=0.05)
    plt.close()
    return out_path


def create_panel_two() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(EDA_ROOT / 'scl' / 'plots' / 'all_subs_dmt_eda_scl.png')
    B_path = str(EDA_ROOT / 'smna' / 'plots' / 'all_subs_dmt_smna.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)

    # Vertical stack (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    _place(axes[0], A_img, 'A')
    _place(axes[1], B_img, 'B')

    plt.tight_layout(pad=0.8)
    out_path = str(OUT_DIR / 'panel_2.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_panel_three() -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(EDA_ROOT / 'scl' / 'plots' / 'stacked_subs_eda_scl.png')
    B_path = str(EDA_ROOT / 'smna' / 'plots' / 'stacked_subs_smna.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)

    # GridSpec con separación mínima entre subplots
    fig = plt.figure(figsize=(12, 28))
    gs = fig.add_gridspec(1, 2,
                          wspace=0.05,    # Separación horizontal mínima
                          hspace=0.0,
                          left=0.01,
                          right=0.99,
                          top=0.99,
                          bottom=0.01)
    
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]
    
    # Posicionar letras aún más arriba y a la izquierda para evitar superposición completa
    _place(axes[0], A_img, 'A', label_xy=(-0.03, 1.08))
    _place(axes[1], B_img, 'B', label_xy=(-0.03, 1.08))

    out_path = str(OUT_DIR / 'panel_3.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_panel_four() -> str:
    """Create Panel 4: ECG HR analysis (1x2 horizontal layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(ECG_ROOT / 'hr' / 'plots' / 'all_subs_ecg_hr.png')
    B_path = str(ECG_ROOT / 'hr' / 'plots' / 'lme_coefficient_plot.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)

    # GridSpec con separación mínima entre subplots (similar a panel_1 pero solo 1 fila)
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
    
    _place(axes[0], A_img, 'A')
    _place(axes[1], B_img, 'B')

    out_path = str(OUT_DIR / 'panel_4.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_panel_five() -> str:
    """Create Panel 5: Respiratory RVT analysis (1x2 horizontal layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(RESP_ROOT / 'rvt' / 'plots' / 'all_subs_resp_rvt.png')
    B_path = str(RESP_ROOT / 'rvt' / 'plots' / 'lme_coefficient_plot.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)

    # GridSpec con separación mínima entre subplots (idéntico a panel_4)
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
    
    _place(axes[0], A_img, 'A')
    _place(axes[1], B_img, 'B')

    out_path = str(OUT_DIR / 'panel_5.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_panel_six() -> str:
    """Create Panel 6: Combined HR, SCL, and RVT analysis (3x2 grid).
    
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

    # GridSpec con márgenes mínimos y separación horizontal casi nula (similar a panel_1)
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
    left_offset = 0.075  # Extra offset for left column to align with right
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
    
    out_path = str(OUT_DIR / 'panel_6.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def main() -> None:
    p1 = create_panel_one()
    print(f"Panel 1 saved to: {p1}")
    p2 = create_panel_two()
    print(f"Panel 2 saved to: {p2}")
    p3 = create_panel_three()
    print(f"Panel 3 saved to: {p3}")
    p4 = create_panel_four()
    print(f"Panel 4 saved to: {p4}")
    p5 = create_panel_five()
    print(f"Panel 5 saved to: {p5}")
    p6 = create_panel_six()
    print(f"Panel 6 saved to: {p6}")


if __name__ == '__main__':
    main()


