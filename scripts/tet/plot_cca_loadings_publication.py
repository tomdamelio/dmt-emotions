#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create publication-quality CCA loadings figure for Nature Human Behaviour.

Figure 4: Canonical loadings for the significant DMT latent dimension (CV1).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def create_nature_style_loadings_figure(
    loadings_df: pd.DataFrame,
    output_path: str,
    state: str = 'DMT',
    canonical_variate: int = 1
):
    """
    Create Nature Human Behaviour style figure for CCA loadings.
    
    Args:
        loadings_df: DataFrame with canonical loadings
        output_path: Path to save figure
        state: State to plot ('DMT' or 'RS')
        canonical_variate: Which canonical variate to plot
    """
    # Filter data
    data = loadings_df[
        (loadings_df['state'] == state) & 
        (loadings_df['canonical_variate'] == canonical_variate)
    ].copy()
    
    # Separate physiological and TET variables
    physio_data = data[data['variable_set'] == 'physio'].copy()
    tet_data = data[data['variable_set'] == 'tet'].copy()
    
    # Sort by loading magnitude (descending)
    physio_data = physio_data.sort_values('loading', ascending=True)
    tet_data = tet_data.sort_values('loading', ascending=True)
    
    # Create figure with Nature style
    # Nature recommends: 89 mm (single column) or 183 mm (double column)
    # Using double column for better readability
    fig_width_mm = 183
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.3  # Reduced height for more compact figure
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(fig_width_inch, fig_height_inch),
        gridspec_kw={'width_ratios': [1, 1.5]}
    )
    
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
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2,
        'ytick.major.size': 2,
    })
    
    # Color scheme using tab20c palette (consistent with physiological pipeline plots)
    # tab20c has 20 colors in 5 groups of 4 gradients each
    # ECG (HR): Red group (indices 0-3)
    # EDA (SMNA): Blue group (indices 4-7)
    # Respiration (RVT): Green group (indices 8-11)
    # Gray for affective dimensions: index 19 (last color)
    tab20c_colors = plt.cm.tab20c.colors
    
    # Map physiological variables to their modality colors (using first gradient - darkest/most intense)
    physio_colors = {
        'HR': tab20c_colors[0],        # ECG: Red (first red gradient)
        'SMNA_AUC': tab20c_colors[4],  # EDA: Blue (first blue gradient)
        'RVT': tab20c_colors[8]        # Respiration: Green (first green gradient)
    }
    
    # Violet colors for affective dimensions from tab20c (violet group: indices 12-15)
    gray_colors = [tab20c_colors[12], tab20c_colors[13], tab20c_colors[14], tab20c_colors[15]]
    
    # Clean variable names with modality labels
    physio_labels = {
        'HR': ('Electrocardiography', 'HR (Z-scored)'),
        'SMNA_AUC': ('Electrodermal Activity', 'SMNA (Z-scored)'),
        'RVT': ('Respiration', 'RVT (Z-scored)')
    }
    
    tet_labels = {
        'emotional_intensity': 'Emotional Intensity',
        'interoception': 'Interoception',
        'unpleasantness': 'Unpleasantness',
        'pleasantness': 'Pleasantness',
        'bliss': 'Bliss',
        'anxiety': 'Anxiety'
    }
    
    # Plot physiological loadings (LEFT PANEL)
    y_pos_physio = np.arange(len(physio_data))
    
    # Create bars with individual colors per variable
    for i, (idx, row) in enumerate(physio_data.iterrows()):
        var_name = row['variable_name']
        loading = row['loading']
        color = physio_colors.get(var_name, '#666666')  # Default gray if not found
        
        ax1.barh(
            i,
            loading,
            height=0.6,
            color=color,
            edgecolor='none',
            alpha=0.85
        )
    
    # Hide default y-tick labels and add custom two-part labels
    ax1.set_yticks(y_pos_physio)
    ax1.set_yticklabels([])  # Hide default labels
    
    # Add custom y-axis labels with modality (bold, colored) above and metric (normal, black) below
    for i, var_name in enumerate(physio_data['variable_name']):
        if var_name in physio_labels:
            modality, metric = physio_labels[var_name]
            color = physio_colors[var_name]
            
            # Add modality label (bold, colored) - positioned above center with larger font
            ax1.text(
                -0.02, i + 0.15, modality,
                transform=ax1.get_yaxis_transform(),
                ha='right', va='center',
                fontsize=9, fontweight='bold', color=color
            )
            # Add metric label (normal, black) - positioned below center with larger font
            ax1.text(
                -0.02, i - 0.15, metric,
                transform=ax1.get_yaxis_transform(),
                ha='right', va='center',
                fontsize=9, fontweight='normal', color='black'
            )
        else:
            ax1.text(
                -0.02, i, var_name,
                transform=ax1.get_yaxis_transform(),
                ha='right', va='center',
                fontsize=9, fontweight='normal', color='black'
            )
    
    ax1.set_xlabel('Canonical Loading', fontweight='normal')
    ax1.set_title('Physiological Variables', fontweight='bold', pad=8)
    ax1.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax1.set_xlim(-0.1, 1.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Plot TET loadings (RIGHT PANEL) with different gray shades
    y_pos_tet = np.arange(len(tet_data))
    
    # Create bars with different gray colors
    for i, (idx, row) in enumerate(tet_data.iterrows()):
        loading = row['loading']
        # Cycle through gray colors
        color = gray_colors[i % len(gray_colors)]
        
        ax2.barh(
            i,
            loading,
            height=0.6,
            color=color,
            edgecolor='none',
            alpha=0.85
        )
    
    ax2.set_yticks(y_pos_tet)
    ax2.set_yticklabels([tet_labels.get(v, v) for v in tet_data['variable_name']])
    ax2.set_xlabel('Canonical Loading', fontweight='normal')
    ax2.set_title('Affective Dimensions', fontweight='bold', pad=8)
    ax2.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax2.set_xlim(-0.1, 1.0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add panel labels (A and B) in upper left corner of each subplot
    ax1.text(
        -0.35, 1.20, 'A',
        transform=ax1.transAxes,
        ha='left', va='top',
        fontsize=12, fontweight='bold',
        color='black'
    )
    ax2.text(
        -0.15, 1.20, 'B',
        transform=ax2.transAxes,
        ha='left', va='top',
        fontsize=12, fontweight='bold',
        color='black'
    )
    
    # Adjust layout (no overall title)
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
    print("Creating Publication-Quality CCA Loadings Figure")
    print("Style: Nature Human Behaviour")
    print("=" * 80)
    
    # Load loadings data
    loadings_file = 'results/tet/physio_correlation/cca_loadings.csv'
    print(f"\nLoading data from: {loadings_file}")
    
    loadings_df = pd.read_csv(loadings_file)
    print(f"Loaded {len(loadings_df)} loadings")
    
    # Create output directory
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure for DMT CV1
    print("\nCreating Figure 4: DMT Canonical Variate 1 Loadings...")
    output_path = output_dir / 'Figure4_CCA_Loadings_DMT_CV1'
    
    create_nature_style_loadings_figure(
        loadings_df=loadings_df,
        output_path=str(output_path),
        state='DMT',
        canonical_variate=1
    )
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics for DMT CV1")
    print("=" * 80)
    
    dmt_cv1 = loadings_df[
        (loadings_df['state'] == 'DMT') & 
        (loadings_df['canonical_variate'] == 1)
    ]
    
    print("\nPhysiological Loadings:")
    physio = dmt_cv1[dmt_cv1['variable_set'] == 'physio'].sort_values('loading', ascending=False)
    for _, row in physio.iterrows():
        print(f"  {row['variable_name']:15s}: {row['loading']:6.3f}")
    
    print("\nAffective Loadings:")
    tet = dmt_cv1[dmt_cv1['variable_set'] == 'tet'].sort_values('loading', ascending=False)
    for _, row in tet.iterrows():
        print(f"  {row['variable_name']:20s}: {row['loading']:6.3f}")
    
    print("\n" + "=" * 80)
    print("Figure creation complete!")
    print("=" * 80)
    print("\nFiles ready for submission to Nature Human Behaviour:")
    print(f"  📁 {output_dir}")
    print("\nFigure specifications:")
    print("  - Format: PNG (600 DPI), PDF (vector), TIFF (600 DPI)")
    print("  - Width: 183 mm (double column)")
    print("  - Font: Arial, 7pt body, 8pt titles")
    print("  - Colors: Colorblind-safe palette")
    print("  - Style: Nature Human Behaviour guidelines")


if __name__ == '__main__':
    main()
