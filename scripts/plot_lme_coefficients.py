# -*- coding: utf-8 -*-
"""
LME Coefficient Forest Plot

This script creates forest plots showing LME fixed effects with 95% confidence
intervals for TET dimensions. Plots are organized by effect type and ordered by
State effect strength.

Usage:
    python scripts/plot_lme_coefficients.py --input results/tet/lme/lme_results.csv --output results/tet/figures
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Nature Human Behaviour style configuration
AXES_TITLE_SIZE = 24
AXES_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
LEGEND_FONTSIZE = 18

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': LEGEND_FONTSIZE,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
})

# TET uses purple/violet color scheme from tab20c palette
# tab20c has 20 colors in 5 groups of 4 gradients each
# Purple group: indices 12-15 (darkest to lightest)
tab20c_colors = plt.cm.tab20c.colors
COLOR_STATE = tab20c_colors[12]      # Darkest purple for State effect
COLOR_DOSE = tab20c_colors[13]       # Medium purple for Dose effect
COLOR_INTERACTION = tab20c_colors[14]  # Lighter purple for Interaction


def load_lme_results(input_path: str) -> pd.DataFrame:
    """
    Load LME results from CSV file.
    
    Args:
        input_path (str): Path to LME results CSV
        
    Returns:
        pd.DataFrame: LME results with required columns
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"LME results file not found: {input_path}")
    
    logger.info(f"Loading LME results from: {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Validate required columns
    required_cols = ['dimension', 'effect', 'beta', 'ci_lower', 'ci_upper', 'p_fdr']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Loaded {len(df)} rows from LME results")
    
    return df


def prepare_plotting_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for forest plot visualization.
    
    Organizes results by effect type and orders dimensions by State effect strength.
    
    Args:
        df (pd.DataFrame): LME results
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping effect types to prepared data
    """
    logger.info("Preparing data for forest plots...")
    
    # Filter to only Arousal and Valence dimensions
    df_filtered = df[df['dimension'].isin(['emotional_intensity_z', 'valence_index_z'])].copy()
    
    # Filter out Intercept, Group Var, and triple interaction
    df_filtered = df_filtered[~df_filtered['effect'].isin(['Intercept', 'Group Var', 'state[T.DMT]:dose[T.Alta]:time_c'])].copy()
    
    # Add significance marker
    df_filtered['significant'] = df_filtered['p_fdr'] < 0.05
    
    # Fixed order: Arousal first (top), Valence second (bottom)
    # In matplotlib, y=0 is at bottom, so we reverse: Valence=0, Arousal=1
    # This way Arousal appears at top and Valence at bottom
    dimension_order = {
        'emotional_intensity_z': 1,  # Arousal at top (higher y position)
        'valence_index_z': 0         # Valence at bottom (lower y position)
    }
    df_filtered['dim_order'] = df_filtered['dimension'].map(dimension_order)
    
    # Clean dimension names and map to publication names
    dimension_name_map = {
        'emotional_intensity_z': 'Arousal',
        'valence_index_z': 'Valence'
    }
    df_filtered['dimension_clean'] = df_filtered['dimension'].map(dimension_name_map)
    
    # Define effect types and their display names
    effect_types = {
        'State': ['state[T.DMT]'],
        'Dose': ['dose[T.Alta]'],
        'State:Dose': ['state[T.DMT]:dose[T.Alta]'],
        'State:Time': ['state[T.DMT]:time_c'],
        'Dose:Time': ['dose[T.Alta]:time_c']
    }
    
    # Organize by effect type
    plot_data = {}
    
    for effect_name, effect_patterns in effect_types.items():
        effect_df = df_filtered[df_filtered['effect'].isin(effect_patterns)].copy()
        
        if len(effect_df) > 0:
            # Sort by dimension order
            effect_df = effect_df.sort_values('dim_order')
            plot_data[effect_name] = effect_df
            logger.info(f"  {effect_name}: {len(effect_df)} dimensions")
    
    return plot_data


def plot_coefficient_forest(
    plot_data: Dict[str, pd.DataFrame],
    output_path: str,
    figsize: tuple = (18, 3.5)
) -> None:
    """
    Create forest plot showing LME coefficients with 95% CIs.
    
    Args:
        plot_data (Dict[str, pd.DataFrame]): Prepared plotting data by effect type
        output_path (str): Path to save figure
        figsize (tuple): Figure size in inches (width, height)
            Width=18 for consistency with pca_timeseries.png and timeseries_all_dimensions.png
    """
    logger.info("Creating coefficient forest plot...")
    
    # Determine number of panels
    n_effects = len(plot_data)
    
    # Create figure with subplots
    # Width=18 for consistency with other TET figures
    STANDARD_WIDTH = 18
    fig, axes = plt.subplots(1, n_effects, figsize=(STANDARD_WIDTH, figsize[1]), sharey=True)
    
    # Handle single panel case
    if n_effects == 1:
        axes = [axes]
    
    # Color palette for effect types (using tab20c purple scheme)
    effect_colors = {
        'State': COLOR_STATE,           # Darkest purple
        'Dose': COLOR_DOSE,             # Medium purple
        'State:Dose': COLOR_INTERACTION,  # Lighter purple
        'State:Time': tab20c_colors[13],  # Medium purple
        'Dose:Time': tab20c_colors[14]    # Lighter purple
    }
    
    # Plot each effect type
    for ax, (effect_name, effect_df) in zip(axes, plot_data.items()):
        
        # Y positions
        y_positions = np.arange(len(effect_df))
        
        # Get color for this effect
        color = effect_colors.get(effect_name, '#666666')
        
        # Plot each dimension
        for i, (idx, row) in enumerate(effect_df.iterrows()):
            y_pos = y_positions[i]
            
            # Style based on significance - uniform size for all
            linewidth = 6.5
            alpha = 1.0
            marker_size = 200
            
            # Plot CI as thick horizontal line
            ax.plot(
                [row['ci_lower'], row['ci_upper']],
                [y_pos, y_pos],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=2
            )
            
            # Plot point estimate as large circle
            ax.scatter(
                row['beta'],
                y_pos,
                color=color,
                s=marker_size,
                alpha=alpha,
                edgecolors=color,
                linewidths=3.5,
                zorder=3
            )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=2.0, zorder=1)
        
        # Customize axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(effect_df['dimension_clean'], fontsize=TICK_LABEL_SIZE+2)
        ax.set_xlabel('β coefficient', fontweight='bold')
        ax.set_title(effect_name, fontweight='bold', color=color)
        ax.grid(True, axis='x', alpha=0.25, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Forest plot saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create LME coefficient forest plots for TET dimensions'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/lme/lme_results.csv',
        help='Path to LME results CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Load LME results
        lme_results = load_lme_results(args.input)
        
        # Prepare plotting data
        plot_data = prepare_plotting_data(lme_results)
        
        if not plot_data:
            logger.error("No data available for plotting")
            return 1
        
        # Create forest plot
        output_path = os.path.join(args.output, 'lme_coefficients_forest.png')
        plot_coefficient_forest(plot_data, output_path)
        
        logger.info("✓ LME coefficient forest plot generation complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating forest plot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
