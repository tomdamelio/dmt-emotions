#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create publication-quality CCA validation figure for Nature Human Behaviour.

Figure S4: Validation of physiological–affective coupling.
Panel A: Out-of-sample correlation distributions from LOOCV
Panel B: In-sample canonical correlation scatterplot for DMT CV1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def create_nature_style_validation_figure(
    cv_folds_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    merged_data_df: pd.DataFrame,
    output_path: str
):
    """
    Create Nature Human Behaviour style figure for CCA validation.
    
    Args:
        cv_folds_df: DataFrame with cross-validation fold results
        cv_summary_df: DataFrame with cross-validation summary statistics
        merged_data_df: DataFrame with merged physiological-TET data
        output_path: Path to save figure
    """
    # Create figure with Nature style
    # Nature recommends: 89 mm (single column) or 183 mm (double column)
    # Using double column for better readability
    fig_width_mm = 183
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.4  # Aspect ratio for two-panel figure
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(fig_width_inch, fig_height_inch),
        gridspec_kw={'width_ratios': [0.7, 1.3], 'wspace': 0.3}  # Panel A slightly wider, small spacing
    )
    
    # Nature style parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
    })
    
    # Color scheme using tab20c palette
    tab20c_colors = plt.cm.tab20c.colors
    
    # Use gray colors for RS and violet for DMT (consistent with other figures)
    # Lighter colors for fill, darker for edges
    color_rs = tab20c_colors[17]   # Light gray for RS fill
    color_rs_edge = tab20c_colors[16]  # Darker gray for RS edges
    color_dmt = tab20c_colors[13]  # Light violet for DMT fill
    color_dmt_edge = tab20c_colors[12]  # Darker violet for DMT edges
    
    # ========== PANEL A: Cross-validation boxplots ==========
    
    # Filter data for CV1 only
    cv_cv1 = cv_folds_df[cv_folds_df['canonical_variate'] == 1].copy()
    
    # Prepare data for half-boxplot + half-scatter plots
    rs_data = cv_cv1[cv_cv1['state'] == 'RS']['r_oos'].values
    dmt_data = cv_cv1[cv_cv1['state'] == 'DMT']['r_oos'].values
    
    positions = [0.8, 1.6]  # Closer together (was [1, 2])
    box_width = 0.15
    
    # Calculate quartiles for manual boxplot drawing
    def draw_half_boxplot(ax, data, pos, color_fill, color_edge):
        """Draw a half boxplot on the left side."""
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_low = np.min(data[data >= q1 - 1.5 * iqr])
        whisker_high = np.max(data[data <= q3 + 1.5 * iqr])
        
        # Draw box (left half only)
        box = plt.Rectangle(
            (pos - box_width, q1), box_width, q3 - q1,
            facecolor=color_fill, edgecolor=color_edge,
            linewidth=0.5, alpha=0.7, zorder=2
        )
        ax.add_patch(box)
        
        # Draw median line
        ax.hlines(median, pos - box_width, pos, 
                 colors=color_edge, linewidth=1.5, zorder=3)
        
        # Draw whiskers
        ax.vlines(pos - box_width/2, whisker_low, q1, 
                 colors=color_edge, linewidth=0.5, zorder=2)
        ax.vlines(pos - box_width/2, q3, whisker_high, 
                 colors=color_edge, linewidth=0.5, zorder=2)
        
        # Draw whisker caps
        ax.hlines(whisker_low, pos - box_width*0.75, pos - box_width*0.25, 
                 colors=color_edge, linewidth=0.5, zorder=2)
        ax.hlines(whisker_high, pos - box_width*0.75, pos - box_width*0.25, 
                 colors=color_edge, linewidth=0.5, zorder=2)
    
    # Draw half boxplots on the left
    draw_half_boxplot(ax1, rs_data, positions[0], color_rs, color_rs_edge)
    draw_half_boxplot(ax1, dmt_data, positions[1], color_dmt, color_dmt_edge)
    
    # Add individual data points on the right half with jitter
    np.random.seed(42)  # For reproducibility
    jitter_strength = 0.08
    
    # RS data points (right side)
    x_rs = np.ones(len(rs_data)) * positions[0] + np.abs(np.random.normal(0, jitter_strength, len(rs_data)))
    ax1.scatter(x_rs, rs_data, s=20, color=color_rs, alpha=0.7, 
                edgecolors=color_rs_edge, linewidths=0.5, zorder=3)
    
    # DMT data points (right side)
    x_dmt = np.ones(len(dmt_data)) * positions[1] + np.abs(np.random.normal(0, jitter_strength, len(dmt_data)))
    ax1.scatter(x_dmt, dmt_data, s=20, color=color_dmt, alpha=0.7, 
                edgecolors=color_dmt_edge, linewidths=0.5, zorder=3)
    
    # Add zero reference line
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Styling
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['RS', 'DMT'])
    ax1.set_ylabel('Out-of-sample correlation (r$_{oos}$)', fontweight='normal')
    ax1.set_title('Cross-validation performance (CV1)', fontweight='bold', pad=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(-0.6, 1.0)
    ax1.set_xlim(0.3, 2.1)  # Adjust x-axis limits to add space on left
    ax1.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='--')
    ax1.set_axisbelow(True)
    # Add padding to x-tick labels
    ax1.tick_params(axis='x', pad=5)
    
    # Add mean values and significance as elegant annotations
    rs_mean = cv_summary_df[(cv_summary_df['state'] == 'RS') & 
                            (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    dmt_mean = cv_summary_df[(cv_summary_df['state'] == 'DMT') & 
                             (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    
    # Add mean annotations above boxplots with subtle styling
    # Calculate y position above the maximum value
    rs_max = rs_data.max()
    dmt_max = dmt_data.max()
    
    ax1.text(positions[0], rs_max + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {rs_mean:.2f}', 
             ha='center', va='bottom', fontsize=7.5, style='italic')
    ax1.text(positions[1], dmt_max + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {dmt_mean:.2f}**', 
             ha='center', va='bottom', fontsize=7.5, style='italic', fontweight='bold')
    
    # Add significance note at bottom
    ax1.text(0.98, 0.02, '**p < 0.01', 
             transform=ax1.transAxes,
             ha='right', va='bottom',
             fontsize=6.5, style='italic')
    
    # ========== PANEL B: In-sample canonical scores scatterplot ==========
    
    # Prepare data for DMT state
    from sklearn.cross_decomposition import CCA
    
    # Filter DMT data
    dmt_data = merged_data_df[merged_data_df['state'] == 'DMT'].copy()
    
    # Define physiological and TET measures
    physio_measures = ['HR', 'SMNA_AUC', 'RVT']
    tet_affective = [
        'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
        'interoception_z', 'bliss_z', 'anxiety_z'
    ]
    
    # Extract matrices
    X = dmt_data[physio_measures].values
    Y = dmt_data[tet_affective].values
    subjects = dmt_data['subject'].values
    
    # Remove rows with missing values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    
    # Fit CCA
    cca = CCA(n_components=2)
    cca.fit(X, Y)
    
    # Transform to canonical variates
    U, V = cca.transform(X, Y)
    
    # Get first canonical variate (CV1)
    x = U[:, 0]
    y = V[:, 0]
    
    # Get in-sample correlation
    in_sample_r = cv_summary_df[
        (cv_summary_df['state'] == 'DMT') & 
        (cv_summary_df['canonical_variate'] == 1)
    ]['in_sample_r'].values[0]
    
    # Create color map for subjects with lighter fill and darker edges using tab20
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    subject_colors = {}
    subject_edge_colors = {}
    
    # Use tab20 colormap: even indices (0,2,4...) are darker, odd indices (1,3,5...) are lighter
    tab20_colors = plt.cm.tab20.colors
    
    for i, subj in enumerate(unique_subjects):
        # Use pairs from tab20: even=dark, odd=light
        idx = (i % 10) * 2  # 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
        subject_colors[subj] = tab20_colors[idx + 1]  # Odd index = lighter color for fill
        subject_edge_colors[subj] = tab20_colors[idx]  # Even index = darker color for edge
    
    # Scatter plot with colors by subject (with darker edge colors)
    for subj in unique_subjects:
        mask = subjects == subj
        ax2.scatter(
            x[mask], y[mask],
            s=30,
            color=subject_colors[subj],
            alpha=0.7,
            edgecolors=subject_edge_colors[subj],
            linewidths=0.5,
            label=subj
        )
    
    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax2.plot(x_line, p(x_line), 
             color='red', 
             linestyle='--', 
             linewidth=1.5, 
             alpha=0.8,
             label='Regression')
    
    # Add correlation text in upper right corner (bold, slightly left of edge)
    ax2.text(
        0.88, 0.95,
        f'$\\mathit{{r}}$ = {in_sample_r:.2f}',
        transform=ax2.transAxes,
        ha='right', va='top',
        fontsize=7.5,
        style='italic',
        fontweight='bold'
    )
    
    # Styling
    ax2.set_xlabel('Physiological Canonical Score (U1)', fontweight='normal')
    ax2.set_ylabel('TET Affective Canonical Score (V1)', fontweight='normal')
    ax2.set_title('In-sample coupling (DMT CV1)', fontweight='bold', pad=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.2, linewidth=0.5, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Zero reference lines removed - grid handles all lines uniformly
    
    # Add legend (inside plot area, lower right, small font)
    ax2.legend(
        loc='lower right',
        fontsize=5,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        ncol=1,
        borderpad=0.5,
        labelspacing=0.3,
        handletextpad=0.5
    )
    
    # Add panel labels (A and B) in upper left corner of each subplot
    ax1.text(
        -0.25, 1.20, 'A',
        transform=ax1.transAxes,
        ha='left', va='top',
        fontsize=12, fontweight='bold',
        color='black'
    )
    ax2.text(
        -0.20, 1.20, 'B',
        transform=ax2.transAxes,
        ha='left', va='top',
        fontsize=12, fontweight='bold',
        color='black'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    
    # Save as high-resolution PNG (Nature requires 300+ DPI)
    plt.savefig(
        output_path.with_suffix('.png'),
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    # Save as PDF (vector format for publication)
    plt.savefig(
        output_path.with_suffix('.pdf'),
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    
    # Save as TIFF (Nature accepts TIFF)
    plt.savefig(
        output_path.with_suffix('.tiff'),
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pil_kwargs={'compression': 'tiff_lzw'}
    )
    
    plt.close()
    
    print(f"✓ Saved publication figure:")
    print(f"  - PNG (600 DPI): {output_path.with_suffix('.png')}")
    print(f"  - PDF (vector): {output_path.with_suffix('.pdf')}")
    print(f"  - TIFF (600 DPI): {output_path.with_suffix('.tiff')}")


def main():
    """Main execution."""
    print("=" * 80)
    print("Creating Publication-Quality CCA Validation Figure")
    print("Style: Nature Human Behaviour")
    print("=" * 80)
    
    # Load data
    base_path = Path('results/tet/physio_correlation')
    
    print("\nLoading data...")
    cv_folds_df = pd.read_csv(base_path / 'cca_cross_validation_folds.csv')
    cv_summary_df = pd.read_csv(base_path / 'cca_cross_validation_summary.csv')
    merged_data_df = pd.read_csv(base_path / 'merged_physio_tet_data.csv')
    
    print(f"  - CV folds: {len(cv_folds_df)} observations")
    print(f"  - CV summary: {len(cv_summary_df)} conditions")
    print(f"  - Merged data: {len(merged_data_df)} observations")
    
    # Create output directory
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    print("\nCreating Figure S4: CCA Validation...")
    output_path = output_dir / 'FigureS4_CCA_Validation'
    
    create_nature_style_validation_figure(
        cv_folds_df=cv_folds_df,
        cv_summary_df=cv_summary_df,
        merged_data_df=merged_data_df,
        output_path=str(output_path)
    )
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    cv1_summary = cv_summary_df[cv_summary_df['canonical_variate'] == 1]
    
    print("\nCross-validation performance (CV1):")
    for _, row in cv1_summary.iterrows():
        print(f"  {row['state']:3s}: mean r_oos = {row['mean_r_oos']:6.3f} "
              f"(SD = {row['sd_r_oos']:.3f}, in-sample r = {row['in_sample_r']:.3f})")
    
    print("\n" + "=" * 80)
    print("Figure creation complete!")
    print("=" * 80)
    print("\nFiles ready for submission to Nature Human Behaviour:")
    print(f"  📁 {output_dir}")
    print("\nFigure specifications:")
    print("  - Format: PNG (600 DPI), PDF (vector), TIFF (600 DPI)")
    print("  - Width: 183 mm (double column)")
    print("  - Font: Arial, 7pt body, 8pt titles")
    print("  - Colors: Gray (RS), Violet (DMT)")
    print("  - Style: Nature Human Behaviour guidelines")


if __name__ == '__main__':
    main()
