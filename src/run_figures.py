# -*- coding: utf-8 -*-
"""
Generate publication-ready composite panels for physiological figures (Nature Human Behaviour).

This script stitches already-generated PNG plots into final panels.

NOTE: Figure 1 is a methods figure created separately. This script generates Figures 2-5.

Figure 2 (3x2 grid - Combined HR, SMNA, RVT):
  A (row1-left):  results/ecg/hr/plots/all_subs_ecg_hr.png
  B (row1-right): results/ecg/hr/plots/lme_coefficient_plot.png
  C (row2-left):  results/eda/smna/plots/all_subs_smna.png
  D (row2-right): results/eda/smna/plots/lme_coefficient_plot.png
  E (row3-left):  results/resp/rvt/plots/all_subs_resp_rvt.png
  F (row3-right): results/resp/rvt/plots/lme_coefficient_plot.png

Figure 3 (Composite Autonomic Arousal Index - 4 panels):
  A: PCA scree plot
  B: PC1 loadings
  C: LME coefficients
  D: All subjects time course

Figure 4 (TET Analysis - 5 panels):
  A: Time courses of affective ratings
  B: LME coefficients forest plot
  C: PC1 and PC2 time courses
  D: Variance explained scree plot
  E: PCA loadings heatmap

Figure 5 (CCA Analysis - 4 panels):
  A: Physiological loadings (DMT CV1)
  B: Affective dimension loadings (DMT CV1)
  C: Cross-validation boxplots
  D: In-sample canonical scores scatterplot

Supplementary Figures:
  Figure S1: Stacked subjects for all modalities (1x3 horizontal)
  Figure S2: DMT ECG HR extended timecourse
  Figure S3: Stacked subjects composite arousal index
  Figure S4: DMT composite arousal index extended
  Figure S5: Physiological Arousal Index vs TET Affective Dimensions

Outputs:
  results/figures/figure_2.png
  results/figures/figure_3.png
  results/figures/figure_4.png
  results/figures/figure_5.png
  results/figures/figure_S1.png
  results/figures/figure_S2.png
  results/figures/figure_S3.png
  results/figures/figure_S4.png
  results/figures/figure_S5.png
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
    
    out_path = str(OUT_DIR / 'figure_2.png')
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

    out_path = str(OUT_DIR / 'figure_3.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S2() -> str:
    """Create Figure S2: DMT ECG HR extended timecourse (single panel)."""
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
    
    out_path = str(OUT_DIR / 'figure_S2.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S3() -> str:
    """Create Figure S3: Stacked subjects for composite arousal index (single panel)."""
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
    
    out_path = str(OUT_DIR / 'figure_S3.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_three() -> str:
    """Create Figure 4: TET Analysis (generated by run_tet_analysis.py).
    
    Simply copies the pre-generated figure4_tet_analysis.png to figure_4.png.
    The source figure is generated by src/run_tet_analysis.py and contains:
    - Panel A: Time courses of affective ratings (Arousal, Valence, individual dimensions)
    - Panel B: LME coefficients forest plot
    - Panel C: PC1 and PC2 time courses
    - Panel D: Variance explained scree plot
    - Panel E: PCA loadings heatmap
    """
    import shutil
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TET_ROOT = PROJECT_ROOT / 'results' / 'tet' / 'figures'
    
    source_path = TET_ROOT / 'figure4_tet_analysis.png'
    out_path = OUT_DIR / 'figure_4.png'
    
    if not source_path.exists():
        print(f"[ERROR] Source file not found: {source_path}")
        print("  Run 'python src/run_tet_analysis.py' first to generate the figure.")
        return ""
    
    # Copy the file
    shutil.copy2(source_path, out_path)
    print(f"  Copied {source_path.name} -> {out_path.name}")
    
    return str(out_path)


def create_figure_S1() -> str:
    """Create Figure S1: Stacked subjects for all modalities (1x3 horizontal layout)."""
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

    out_path = str(OUT_DIR / 'figure_S1.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S4() -> str:
    """Create Figure S4: DMT composite arousal index extended timecourse (single panel)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPOSITE_ROOT = PROJECT_ROOT / 'results' / 'composite' / 'plots'

    source_path = str(COMPOSITE_ROOT / 'all_subs_dmt_composite.png')
    img = _load_image(source_path)
    
    if img is None:
        print(f"[ERROR] Could not load {source_path}")
        return ""
    
    # Create a simple figure with the image
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    ax.imshow(img)
    
    out_path = str(OUT_DIR / 'figure_S4.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    return out_path


def create_figure_S5() -> str:
    """Create Figure S5: Physiological Arousal Index vs TET Affective Dimensions.
    
    2x2 panel showing regression of Arousal Index (physio PC1) against:
    - Row 1: Emotional Intensity (z-scored)
    - Row 2: Valence Index (z-scored)
    - Col 1: RS state
    - Col 2: DMT state
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COUPLING_ROOT = PROJECT_ROOT / 'results' / 'coupling'
    
    # Load data
    try:
        merged_df = pd.read_csv(COUPLING_ROOT / 'merged_physio_tet_data.csv')
        regression_df = pd.read_csv(COUPLING_ROOT / 'regression_tet_arousal_index.csv')
    except FileNotFoundError as e:
        print(f"[ERROR] Missing data file: {e}")
        # Fallback to copying existing image
        source_path = str(COUPLING_ROOT / 'figures' / 'pc1_composite_4panel.png')
        img = _load_image(source_path)
        if img is None:
            return ""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('off')
        ax.imshow(img)
        out_path = str(OUT_DIR / 'figure_S5.png')
        plt.savefig(out_path, dpi=400, bbox_inches='tight')
        plt.close()
        return out_path
    
    # Check if ArousalIndex column exists, if not create it from physio_PC1
    if 'ArousalIndex' not in merged_df.columns and 'physio_PC1' in merged_df.columns:
        merged_df['ArousalIndex'] = merged_df['physio_PC1']
    
    # Colors matching paper style
    COLOR_RS = '#9E9AC8'      # Light purple for RS
    COLOR_DMT = '#5E4FA2'     # Dark purple for DMT
    COLOR_LINE = '#D62728'    # Red for regression line
    
    # Set font to match paper style (sans-serif)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    })
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    outcomes = [
        ('emotional_intensity_z', 'Emotional Intensity (Z-scored)'),
        ('valence_index_z', 'Valence Index (Z-scored)')
    ]
    states = [('RS', COLOR_RS), ('DMT', COLOR_DMT)]
    panel_labels = ['A', 'B', 'C', 'D']
    
    panel_idx = 0
    for row_idx, (outcome_var, outcome_label) in enumerate(outcomes):
        for col_idx, (state, color) in enumerate(states):
            ax = axes[row_idx, col_idx]
            
            # Filter data
            state_data = merged_df[merged_df['state'] == state].copy()
            
            # Get regression stats
            reg_row = regression_df[
                (regression_df['outcome_variable'] == outcome_var) &
                (regression_df['state'] == state)
            ]
            
            if len(reg_row) > 0:
                beta = reg_row['beta'].values[0]
                r_squared = reg_row['r_squared'].values[0]
                p_value = reg_row['p_value'].values[0]
            else:
                beta, r_squared, p_value = np.nan, np.nan, np.nan
            
            # Scatter plot
            x_col = 'ArousalIndex' if 'ArousalIndex' in state_data.columns else 'physio_PC1'
            if x_col in state_data.columns and outcome_var in state_data.columns:
                # Align data
                valid_idx = state_data[[x_col, outcome_var]].dropna().index
                x_plot = state_data.loc[valid_idx, x_col]
                y_plot = state_data.loc[valid_idx, outcome_var]
                
                # Scatter
                ax.scatter(x_plot, y_plot, alpha=0.4, color=color, s=25, 
                          edgecolors='white', linewidths=0.3)
                
                # Regression line
                if not np.isnan(beta) and len(x_plot) > 2:
                    x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
                    slope, intercept = np.polyfit(x_plot, y_plot, 1)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=COLOR_LINE, linewidth=2, 
                           linestyle='-', zorder=5)
            
            # Annotation box (sans-serif font)
            if not np.isnan(beta):
                p_str = f'{p_value:.3f}' if p_value >= 0.001 else '<0.001'
                sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                annot_text = f'β = {beta:.2f}{sig_marker}\nR² = {r_squared:.3f}\np = {p_str}'
                
                ax.text(0.05, 0.95, annot_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.9),
                       fontsize=9)
            
            # Panel label
            ax.text(-0.12, 1.08, panel_labels[panel_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='top')
            
            # Labels and title - titles in BLACK
            ax.set_title(f'{state}', fontweight='bold', fontsize=12, color='black')
            ax.set_xlabel('Arousal Index (Physio PC1)', fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(outcome_label, fontsize=10, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
            
            panel_idx += 1
    
    plt.tight_layout()
    
    out_path = str(OUT_DIR / 'figure_S5.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Generated Figure S5 from data")
    return out_path


def create_figure_four() -> str:
    """Create Figure 4: Combined CCA Analysis (4 panels).
    
    Panel A: Physiological loadings (DMT CV1)
    Panel B: Affective dimension loadings (DMT CV1)
    Panel C: Cross-validation boxplots
    Panel D: In-sample canonical scores scatterplot
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CCA_ROOT = PROJECT_ROOT / 'results' / 'coupling'
    
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
    
    out_path = str(OUT_DIR / 'figure_5.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    return out_path


def main() -> None:
    """Generate all figures. Figure 1 is a methods figure created separately."""
    print("Note: Figure 1 is a methods figure created separately.\n")
    
    fig2 = create_figure_one()  # HR, SMNA, RVT → figure_2.png
    print(f"Figure 2 saved to: {fig2}")
    
    fig3 = create_figure_two()  # Composite Arousal Index → figure_3.png
    print(f"Figure 3 saved to: {fig3}")
    
    fig4 = create_figure_three()  # TET Analysis → figure_4.png
    print(f"Figure 4 saved to: {fig4}")
    
    fig5 = create_figure_four()  # CCA Analysis → figure_5.png
    print(f"Figure 5 saved to: {fig5}")
    
    # Supplementary figures
    figS1 = create_figure_S1()
    print(f"Figure S1 saved to: {figS1}")
    figS2 = create_figure_S2()
    print(f"Figure S2 saved to: {figS2}")
    figS3 = create_figure_S3()
    print(f"Figure S3 saved to: {figS3}")
    figS4 = create_figure_S4()
    print(f"Figure S4 saved to: {figS4}")
    figS5 = create_figure_S5()
    print(f"Figure S5 saved to: {figS5}")


if __name__ == '__main__':
    main()