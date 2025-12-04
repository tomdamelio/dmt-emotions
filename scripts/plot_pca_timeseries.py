#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot PCA Time Series - PC1 and PC2 trajectories during DMT

This script generates time series plots showing PC1 and PC2 scores over time
during DMT condition, separated by dose (Low 20mg vs High 40mg).

Usage:
    python scripts/plot_pca_timeseries.py
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

# Nature Human Behaviour style configuration
AXES_TITLE_SIZE = 24
AXES_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
LEGEND_FONTSIZE = 18
LEGEND_MARKERSCALE = 1.6
LEGEND_BORDERPAD = 0.6
LEGEND_LABELSPACING = 0.5
LEGEND_HANDLELENGTH = 2.0
LEGEND_BORDERAXESPAD = 0.5

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
    'legend.borderpad': LEGEND_BORDERPAD,
    'legend.handlelength': LEGEND_HANDLELENGTH,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
})

# TET uses purple/violet color scheme from tab20c palette
# tab20c has 20 colors in 5 groups of 4 gradients each
# Purple group: indices 12-15 (darkest to lightest)
tab20c_colors = plt.cm.tab20c.colors
COLOR_HIGH_DOSE = tab20c_colors[12]  # Darkest purple for High dose (40mg)
COLOR_LOW_DOSE = tab20c_colors[14]   # Lighter purple for Low dose (20mg)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_pc_time_courses(pc_scores_df):
    """
    Compute mean and SEM for PC scores over time, separated by state and dose.
    
    Args:
        pc_scores_df: DataFrame with PC scores (from pca_scores.csv)
    
    Returns:
        DataFrame with columns: component, state, dose, t_bin, t_sec, mean, sem, n
    """
    logger.info("Computing PC time courses...")
    
    # Filter to DMT state only
    dmt_data = pc_scores_df[pc_scores_df['state'] == 'DMT'].copy()
    
    # Melt PC columns to long format
    pc_cols = [col for col in dmt_data.columns if col.startswith('PC')]
    id_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
    
    dmt_long = dmt_data.melt(
        id_vars=id_cols,
        value_vars=pc_cols,
        var_name='component',
        value_name='score'
    )
    
    # Compute descriptive statistics
    time_courses = dmt_long.groupby(['component', 'dose', 't_bin', 't_sec'])['score'].agg([
        ('mean', 'mean'),
        ('sem', lambda x: x.sem()),
        ('n', 'count')
    ]).reset_index()
    
    logger.info(f"Computed time courses: {len(time_courses)} rows")
    
    return time_courses


def plot_pc_timeseries(time_courses_df, output_path, dpi=400):
    """
    Plot PC1 and PC2 time series during DMT, separated by dose.
    
    Args:
        time_courses_df: DataFrame with PC time courses
        output_path: Output file path
        dpi: Resolution in dots per inch
    """
    logger.info("Generating PC time series figure...")
    
    # Filter to PC1 and PC2 only
    pc_data = time_courses_df[time_courses_df['component'].isin(['PC1', 'PC2'])].copy()
    
    # Create figure with 2 panels (PC1 left, PC2 right)
    # Width=18 for consistency with timeseries_all_dimensions.png and lme_coefficients_forest.png
    STANDARD_WIDTH = 18
    fig = plt.figure(figsize=(STANDARD_WIDTH, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    
    # Plot each component
    for idx, component in enumerate(['PC1', 'PC2']):
        ax = fig.add_subplot(gs[0, idx])
        
        # Get data for this component
        comp_data = pc_data[pc_data['component'] == component]
        
        # Separate by dose
        low_dose = comp_data[comp_data['dose'] == 'Baja'].sort_values('t_bin')
        high_dose = comp_data[comp_data['dose'] == 'Alta'].sort_values('t_bin')
        
        # Convert time to minutes
        time_low = low_dose['t_sec'].values / 60
        time_high = high_dose['t_sec'].values / 60
        
        # Plot High dose (40mg) in darker purple FIRST (so it appears first in legend)
        ax.plot(time_high, high_dose['mean'], color=COLOR_HIGH_DOSE, linewidth=3, 
               label='High dose (40mg)', marker='o', markersize=4, zorder=2)
        ax.fill_between(
            time_high,
            high_dose['mean'] - high_dose['sem'],
            high_dose['mean'] + high_dose['sem'],
            color=COLOR_HIGH_DOSE,
            alpha=0.25,
            zorder=1
        )
        
        # Plot Low dose (20mg) in lighter purple SECOND (so it appears second in legend)
        ax.plot(time_low, low_dose['mean'], color=COLOR_LOW_DOSE, linewidth=3, 
               label='Low dose (20mg)', marker='o', markersize=4, zorder=2)
        ax.fill_between(
            time_low,
            low_dose['mean'] - low_dose['sem'],
            low_dose['mean'] + low_dose['sem'],
            color=COLOR_LOW_DOSE,
            alpha=0.25,
            zorder=1
        )
        
        # Add vertical line at DMT onset
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=0)
        
        # Formatting
        ax.set_xlabel('Time (minutes)', fontweight='bold')
        ax.set_ylabel(f'{component} Score', fontweight='bold')
        ax.set_title(f'{component}', fontweight='bold')
        ax.set_xlim(-1, 20)
        ax.grid(True, which='major', axis='y', alpha=0.25, linestyle='-', linewidth=0.5)
        ax.grid(False, which='major', axis='x')
        ax.set_axisbelow(True)
        
        # Add legend to BOTH panels
        # PC1: upper right, PC2: lower right
        legend_loc = 'upper right' if idx == 0 else 'lower right'
        legend = ax.legend(loc=legend_loc, frameon=True, fancybox=True,
                         fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE,
                         borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING,
                         borderaxespad=LEGEND_BORDERAXESPAD)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved: {output_path}")


def main():
    """Main workflow for PCA time series visualization."""
    
    parser = argparse.ArgumentParser(
        description='Generate PCA time series plots (PC1 and PC2 during DMT)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='results/tet/pca/pca_scores.csv',
        help='Path to PCA scores CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/tet/figures/pca_timeseries.png',
        help='Output file path'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution in DPI'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("PCA TIME SERIES VISUALIZATION")
        logger.info("=" * 80)
        
        # Check input file
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            logger.error("Run: python scripts/compute_pca_analysis.py first")
            return 1
        
        # Load PC scores
        logger.info(f"Loading PC scores from: {args.input}")
        pc_scores = pd.read_csv(args.input)
        logger.info(f"  Loaded {len(pc_scores)} rows")
        
        # Compute time courses
        time_courses = compute_pc_time_courses(pc_scores)
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Generate figure
        plot_pc_timeseries(time_courses, args.output, args.dpi)
        
        # Summary
        logger.info("=" * 80)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Output: {args.output}")
        logger.info(f"Figure shows PC1 and PC2 trajectories during DMT (0-20 min)")
        logger.info(f"Blue line: Low dose (20mg)")
        logger.info(f"Red line: High dose (40mg)")
        logger.info(f"Shading: SEM around mean trajectories")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
