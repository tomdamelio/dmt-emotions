# -*- coding: utf-8 -*-
"""
Generate publication-ready composite panels for physiological figures (Nature Human Behaviour).

This script stitches already-generated PNG plots into final panels:

Figure 1 (3x2 grid - Combined HR, SMNA, RVT):
  A (row1-left):  results/ecg/hr/plots/all_subs_ecg_hr.png
  B (row1-right): results/ecg/hr/plots/lme_coefficient_plot.png
  C (row2-left):  results/eda/smna/plots/all_subs_smna.png
  D (row2-right): results/eda/smna/plots/lme_coefficient_plot.png
  E (row3-left):  results/resp/rvt/plots/all_subs_resp_rvt.png
  F (row3-right): results/resp/rvt/plots/lme_coefficient_plot.png

Figure S1 (single panel - DMT ECG HR extended):
  results/ecg/hr/plots/all_subs_dmt_ecg_hr.png

Figure S2 (1x3 horizontal - Stacked subjects for all modalities):
  A (left):       results/ecg/hr/plots/stacked_subs_ecg_hr.png
  B (center):     results/eda/scl/plots/stacked_subs_eda_scl.png
  C (right):      results/resp/rvt/plots/stacked_subs_resp_rvt.png

Figure 3 (TET Analysis - 5 panels):
  A (row1, full):  results/tet/figures/timeseries_all_dimensions.png
  B (row2, full):  results/tet/figures/lme_coefficients_forest.png
  C (row3, full):  results/tet/figures/pca_timeseries.png
  D (row4, left):  results/tet/figures/pca_scree_plot.png
  E (row4, right): results/tet/figures/pca_loadings_heatmap.png

Figure 4 (CCA Analysis - 4 panels):
  A (row1, left):   Physiological loadings (DMT CV1)
  B (row1, center): Affective dimension loadings (DMT CV1)
  C (row1, right):  Cross-validation boxplots
  D (row2, full):   In-sample canonical scores scatterplot

Outputs:
  results/figures/figure_1.png
  results/figures/figure_2.png
  results/figures/figure_3.png
  results/figures/figure_4.png
  results/figures/figure_S1.png
  results/figures/figure_S2.png
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


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
    """Create Figure 1: Combined HR, SMNA, and RVT analysis (3x2 grid).
    
    Row 1 (HR):   A = all_subs_ecg_hr.png,    B = lme_coefficient_plot.png
    Row 2 (SMNA): C = all_subs_smna.png,      D = lme_coefficient_plot.png
    Row 3 (RVT):  E = all_subs_resp_rvt.png,  F = lme_coefficient_plot.png
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Row 1: HR
    A_path = str(ECG_ROOT / 'hr' / 'plots' / 'all_subs_ecg_hr.png')
    B_path = str(ECG_ROOT / 'hr' / 'plots' / 'lme_coefficient_plot.png')
    
    # Row 2: SMNA
    C_path = str(EDA_ROOT / 'smna' / 'plots' / 'all_subs_smna.png')
    D_path = str(EDA_ROOT / 'smna' / 'plots' / 'lme_coefficient_plot.png')
    
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
    """Create Figure S2: Stacked subjects for composite arousal index (single panel)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_ROOT = PROJECT_ROOT / 'results' / 'composite' / 'plots'

    source_path = str(COMPOSITE_ROOT / 'stacked_subs_composite.png')
    img = _load_image(source_path)
    
    if img is None:
        print(f"[ERROR] Could not load {source_path}")
        return ""
    
    # Create a simple figure with the image
    fig = plt.figure(figsize=(12, 28))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(img)
    
    out_path = str(OUT_DIR / 'figure_S2.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_three() -> str:
    """Create Figure 3: TET Analysis composite (5 panels).
    
    Layout:
    Row 1: A = timeseries_all_dimensions.png (full width)
    Row 2: B = lme_coefficients_forest.png (full width)
    Row 3: C = pca_timeseries.png (full width)
    Row 4: D = pca_scree_plot.png (left, narrower), E = pca_loadings_heatmap.png (right, wider)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TET_ROOT = PROJECT_ROOT / 'results' / 'tet' / 'figures'

    A_path = str(TET_ROOT / 'timeseries_all_dimensions.png')
    B_path = str(TET_ROOT / 'lme_coefficients_forest.png')
    C_path = str(TET_ROOT / 'pca_timeseries.png')
    D_path = str(TET_ROOT / 'pca_scree_plot.png')
    E_path = str(TET_ROOT / 'pca_loadings_heatmap.png')

    A_img = _load_image(A_path)
    B_img = _load_image(B_path)
    C_img = _load_image(C_path)
    D_img = _load_image(D_path)
    E_img = _load_image(E_path)

    # Calculate aspect ratios to determine proper height ratios
    def get_aspect(img):
        if img is not None:
            return img.shape[0] / img.shape[1]  # height / width
        return 0.5
    
    A_aspect = get_aspect(A_img)
    B_aspect = get_aspect(B_img)
    C_aspect = get_aspect(C_img)
    
    # Normalize height ratios based on aspect ratios
    # A, B, C span full width, so their heights should reflect their aspect ratios
    height_ratios = [A_aspect, B_aspect, C_aspect, max(get_aspect(D_img), get_aspect(E_img)) * 0.6]

    # Create figure with GridSpec
    fig = plt.figure(figsize=(34, 45))
    gs = fig.add_gridspec(4, 2,
                          width_ratios=[0.35, 0.65],  # D narrower, E wider
                          height_ratios=height_ratios,
                          wspace=0.02,
                          hspace=0.06,
                          left=0.01,
                          right=0.99,
                          top=0.99,
                          bottom=0.01)
    
    # Panel A: row 0, full width
    ax_A = fig.add_subplot(gs[0, :])
    
    # Panel B: row 1, full width
    ax_B = fig.add_subplot(gs[1, :])
    
    # Panel C: row 2, full width
    ax_C = fig.add_subplot(gs[2, :])
    
    # Panel D: row 3, left column (narrower)
    ax_D = fig.add_subplot(gs[3, 0])
    
    # Panel E: row 3, right column (wider)
    ax_E = fig.add_subplot(gs[3, 1])
    
    # Place images with aspect='auto' for full-width panels to stretch properly
    ax_A.axis('off')
    if A_img is not None:
        ax_A.imshow(A_img, aspect='auto')
    
    ax_B.axis('off')
    if B_img is not None:
        ax_B.imshow(B_img, aspect='auto')
    
    ax_C.axis('off')
    if C_img is not None:
        ax_C.imshow(C_img, aspect='auto')
    
    # D and E keep their aspect ratio
    ax_D.axis('off')
    if D_img is not None:
        ax_D.imshow(D_img)
    
    ax_E.axis('off')
    if E_img is not None:
        ax_E.imshow(E_img)
    
    # Place labels using figure coordinates
    axes = [ax_A, ax_B, ax_C, ax_D, ax_E]
    labels = ['A', 'B', 'C', 'D', 'E']
    offsets = [0.01, 0.01, 0.01, 0.01, 0.01]
    
    for ax, label, offset in zip(axes, labels, offsets):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.01, pos.y1 + offset, label, 
                fontsize=30, fontweight='bold', ha='left', va='top')

    out_path = str(OUT_DIR / 'figure_3.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S3() -> str:
    """Create Figure S3: Stacked subjects for all modalities (1x3 horizontal layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    A_path = str(ECG_ROOT / 'hr' / 'plots' / 'stacked_subs_ecg_hr.png')
    B_path = str(EDA_ROOT / 'smna' / 'plots' / 'stacked_subs_smna.png')
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


def create_figure_four() -> str:
    """Create Figure 4: Combined CCA Analysis (4 panels).
    
    Panel A: Physiological loadings (DMT CV1)
    Panel B: Affective dimension loadings (DMT CV1)
    Panel C: Cross-validation boxplots
    Panel D: In-sample canonical scores scatterplot
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CCA_ROOT = PROJECT_ROOT / 'results' / 'tet' / 'physio_correlation'
    
    # Load data
    try:
        loadings_df = pd.read_csv(CCA_ROOT / 'cca_loadings.csv')
        cv_folds_df = pd.read_csv(CCA_ROOT / 'cca_cross_validation_folds.csv')
        cv_summary_df = pd.read_csv(CCA_ROOT / 'cca_cross_validation_summary.csv')
        merged_data_df = pd.read_csv(CCA_ROOT / 'merged_physio_tet_data.csv')
    except FileNotFoundError as e:
        print(f"[ERROR] Missing CCA data file: {e}")
        return ""
    
    # Nature style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
    })
    
    # Color scheme
    tab20c_colors = plt.cm.tab20c.colors
    physio_colors = {'HR': tab20c_colors[0], 'SMNA_AUC': tab20c_colors[4], 'RVT': tab20c_colors[8]}
    gray_colors = [tab20c_colors[12], tab20c_colors[13], tab20c_colors[14], tab20c_colors[15]]
    color_rs, color_rs_edge = tab20c_colors[17], tab20c_colors[16]
    color_dmt, color_dmt_edge = tab20c_colors[13], tab20c_colors[12]
    
    # Create figure
    fig_width_mm = 183
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.6
    
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch * 1.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], wspace=0.35, hspace=0.4)
    
    # Row 1: A (physio loadings) left, B (TET loadings) right
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    # Row 2: C (CV boxplots) left, D (scatterplot) right
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    
    state, canonical_variate = 'DMT', 1
    
    # Filter data
    data = loadings_df[(loadings_df['state'] == state) & (loadings_df['canonical_variate'] == canonical_variate)].copy()
    physio_data = data[data['variable_set'] == 'physio'].sort_values('loading', ascending=True)
    tet_data = data[data['variable_set'] == 'tet'].sort_values('loading', ascending=True)
    
    physio_labels = {
        'HR': ('Electrocardiography', 'HR (Z-scored)'),
        'SMNA_AUC': ('Electrodermal Activity', 'SMNA (Z-scored)'),
        'RVT': ('Respiration', 'RVT (Z-scored)')
    }
    tet_labels = {
        'emotional_intensity': 'Emotional Intensity', 'interoception': 'Interoception',
        'unpleasantness': 'Unpleasantness', 'pleasantness': 'Pleasantness',
        'bliss': 'Bliss', 'anxiety': 'Anxiety'
    }
    
    # Panel A: Physiological loadings
    for i, (idx, row) in enumerate(physio_data.iterrows()):
        ax_A.barh(i, row['loading'], height=0.6, color=physio_colors.get(row['variable_name'], '#666'), alpha=0.85)
    
    ax_A.set_yticks(np.arange(len(physio_data)))
    ax_A.set_yticklabels([])
    for i, var_name in enumerate(physio_data['variable_name']):
        if var_name in physio_labels:
            modality, metric = physio_labels[var_name]
            ax_A.text(-0.02, i + 0.15, modality, transform=ax_A.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7, fontweight='bold', color=physio_colors[var_name])
            ax_A.text(-0.02, i - 0.15, metric, transform=ax_A.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7, color='black')
    
    ax_A.set_xlabel('Canonical Loading')
    ax_A.set_title('Physiological Variables', fontweight='bold', pad=8)
    ax_A.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    ax_A.set_xlim(-0.1, 1.0)
    ax_A.spines['top'].set_visible(False)
    ax_A.spines['right'].set_visible(False)
    ax_A.grid(axis='x', alpha=0.2, linestyle='--')
    
    # Panel B: TET loadings
    for i, (idx, row) in enumerate(tet_data.iterrows()):
        ax_B.barh(i, row['loading'], height=0.6, color=gray_colors[i % len(gray_colors)], alpha=0.85)
    
    ax_B.set_yticks(np.arange(len(tet_data)))
    ax_B.set_yticklabels([tet_labels.get(v, v) for v in tet_data['variable_name']])
    ax_B.set_xlabel('Canonical Loading')
    ax_B.set_title('Affective Dimensions', fontweight='bold', pad=8)
    ax_B.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    ax_B.set_xlim(-0.1, 1.0)
    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.grid(axis='x', alpha=0.2, linestyle='--')
    
    # Panel C: Cross-validation boxplots
    cv_cv1 = cv_folds_df[cv_folds_df['canonical_variate'] == 1]
    rs_cv = cv_cv1[cv_cv1['state'] == 'RS']['r_oos'].values
    dmt_cv = cv_cv1[cv_cv1['state'] == 'DMT']['r_oos'].values
    
    positions = [0.8, 1.6]
    box_width = 0.15
    
    def draw_half_boxplot(ax, data, pos, color_fill, color_edge):
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_low = np.min(data[data >= q1 - 1.5 * iqr])
        whisker_high = np.max(data[data <= q3 + 1.5 * iqr])
        box = plt.Rectangle((pos - box_width, q1), box_width, q3 - q1,
                            facecolor=color_fill, edgecolor=color_edge, linewidth=0.5, alpha=0.7, zorder=2)
        ax.add_patch(box)
        ax.hlines(median, pos - box_width, pos, colors=color_edge, linewidth=1.5, zorder=3)
        ax.vlines(pos - box_width/2, whisker_low, q1, colors=color_edge, linewidth=0.5, zorder=2)
        ax.vlines(pos - box_width/2, q3, whisker_high, colors=color_edge, linewidth=0.5, zorder=2)
        ax.hlines([whisker_low, whisker_high], pos - box_width*0.75, pos - box_width*0.25, 
                 colors=color_edge, linewidth=0.5, zorder=2)
    
    draw_half_boxplot(ax_C, rs_cv, positions[0], color_rs, color_rs_edge)
    draw_half_boxplot(ax_C, dmt_cv, positions[1], color_dmt, color_dmt_edge)
    
    np.random.seed(42)
    jitter = 0.08
    ax_C.scatter(positions[0] + np.abs(np.random.normal(0, jitter, len(rs_cv))), rs_cv,
                s=20, color=color_rs, alpha=0.7, edgecolors=color_rs_edge, linewidths=0.5, zorder=3)
    ax_C.scatter(positions[1] + np.abs(np.random.normal(0, jitter, len(dmt_cv))), dmt_cv,
                s=20, color=color_dmt, alpha=0.7, edgecolors=color_dmt_edge, linewidths=0.5, zorder=3)
    
    ax_C.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_C.set_xticks(positions)
    ax_C.set_xticklabels(['RS', 'DMT'])
    ax_C.set_ylabel('Out-of-sample correlation (r$_{oos}$)')
    ax_C.set_title('Cross-validation (CV1)', fontweight='bold', pad=8)
    ax_C.spines['top'].set_visible(False)
    ax_C.spines['right'].set_visible(False)
    ax_C.set_ylim(-0.6, 1.0)
    ax_C.set_xlim(0.3, 2.1)
    ax_C.grid(axis='y', alpha=0.2, linestyle='--')
    
    rs_mean = cv_summary_df[(cv_summary_df['state'] == 'RS') & (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    dmt_mean = cv_summary_df[(cv_summary_df['state'] == 'DMT') & (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    ax_C.text(positions[0], rs_cv.max() + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {rs_mean:.2f}', ha='center', fontsize=6, style='italic')
    ax_C.text(positions[1], dmt_cv.max() + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {dmt_mean:.2f}**', ha='center', fontsize=6, style='italic', fontweight='bold')
    ax_C.text(0.98, 0.02, '**p < 0.01', transform=ax_C.transAxes, ha='right', fontsize=5, style='italic')
    
    # Panel D: In-sample scatterplot
    from sklearn.cross_decomposition import CCA
    
    dmt_merged = merged_data_df[merged_data_df['state'] == 'DMT'].copy()
    physio_measures = ['HR', 'SMNA_AUC', 'RVT']
    tet_affective = ['pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
                     'interoception_z', 'bliss_z', 'anxiety_z']
    
    X = dmt_merged[physio_measures].values
    Y = dmt_merged[tet_affective].values
    subjects = dmt_merged['subject'].values
    
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X, Y, subjects = X[valid_mask], Y[valid_mask], subjects[valid_mask]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    
    cca = CCA(n_components=2)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)
    x_scores, y_scores = U[:, 0], V[:, 0]
    
    in_sample_r = cv_summary_df[(cv_summary_df['state'] == 'DMT') & (cv_summary_df['canonical_variate'] == 1)]['in_sample_r'].values[0]
    
    tab20_colors = plt.cm.tab20.colors
    unique_subjects = np.unique(subjects)
    for i, subj in enumerate(unique_subjects):
        idx = (i % 10) * 2
        mask = subjects == subj
        ax_D.scatter(x_scores[mask], y_scores[mask], s=30, color=tab20_colors[idx + 1],
                    alpha=0.7, edgecolors=tab20_colors[idx], linewidths=0.5, label=subj)
    
    z = np.polyfit(x_scores, y_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_scores.min(), x_scores.max(), 100)
    ax_D.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Regression')
    
    ax_D.text(0.98, 0.95, f'$\\mathit{{r}}$ = {in_sample_r:.2f}', transform=ax_D.transAxes,
             ha='right', va='top', fontsize=7, style='italic', fontweight='bold')
    ax_D.set_xlabel('Physiological Canonical Score (U1)')
    ax_D.set_ylabel('TET Affective Canonical Score (V1)')
    ax_D.set_title('In-sample coupling (DMT CV1)', fontweight='bold', pad=8)
    ax_D.spines['top'].set_visible(False)
    ax_D.spines['right'].set_visible(False)
    ax_D.grid(alpha=0.2, linestyle='--')
    ax_D.legend(loc='lower right', fontsize=5, frameon=True, ncol=2)
    
    # Panel labels - 2x2 layout: A-B top row, C-D bottom row
    for ax, label, x_off in [(ax_A, 'A', -0.35), (ax_B, 'B', -0.15), (ax_C, 'C', -0.35), (ax_D, 'D', -0.15)]:
        ax.text(x_off, 1.15, label, transform=ax.transAxes, ha='left', va='top', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    out_path = str(OUT_DIR / 'figure_4.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def main() -> None:
    fig1 = create_figure_one()
    print(f"Figure 1 saved to: {fig1}")
    fig2 = create_figure_two()
    print(f"Figure 2 saved to: {fig2}")
    fig3 = create_figure_three()
    print(f"Figure 3 saved to: {fig3}")
    fig4 = create_figure_four()
    print(f"Figure 4 saved to: {fig4}")
    figS1 = create_figure_S1()
    print(f"Figure S1 saved to: {figS1}")
    figS2 = create_figure_S2()
    print(f"Figure S2 saved to: {figS2}")
    figS3 = create_figure_S3()
    print(f"Figure S3 saved to: {figS3}")


if __name__ == '__main__':
    main()