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

Outputs:
  results/figures/panel_1.png
  results/figures/panel_2.png
  results/figures/panel_3.png
  results/figures/panel_4.png
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


def _place(ax, img, label: str, label_xy: Tuple[float, float] = (0.01, 0.97)):
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
    fig = plt.figure(figsize=(34, 16))
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
    
    _place(axes[0], imgs[0], 'A')
    _place(axes[1], imgs[1], 'B')
    _place(axes[2], imgs[2], 'C')
    _place(axes[3], imgs[3], 'D')

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
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
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
    fig = plt.figure(figsize=(12, 26))
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
    _place(axes[0], A_img, 'A', label_xy=(-0.05, 1.05))
    _place(axes[1], B_img, 'B', label_xy=(-0.05, 1.05))

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
    fig = plt.figure(figsize=(34, 8))
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


def main() -> None:
    p1 = create_panel_one()
    print(f"Panel 1 saved to: {p1}")
    p2 = create_panel_two()
    print(f"Panel 2 saved to: {p2}")
    p3 = create_panel_three()
    print(f"Panel 3 saved to: {p3}")
    p4 = create_panel_four()
    print(f"Panel 4 saved to: {p4}")


if __name__ == '__main__':
    main()


