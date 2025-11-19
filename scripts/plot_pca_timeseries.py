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


def plot_pc_timeseries(time_courses_df, output_path, dpi=300):
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
    fig = plt.figure(figsize=(16, 6), dpi=dpi)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    fig.suptitle('PCA Time Series: Dose Effects during DMT', 
                fontsize=16, fontweight='bold', y=0.98)
    
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
        
        # Plot Low dose (20mg) in blue
        ax.plot(time_low, low_dose['mean'], color='#4472C4', linewidth=2.5, 
               label='Low dose (20mg)', zorder=2)
        ax.fill_between(
            time_low,
            low_dose['mean'] - low_dose['sem'],
            low_dose['mean'] + low_dose['sem'],
            color='#4472C4',
            alpha=0.3,
            zorder=1
        )
        
        # Plot High dose (40mg) in red
        ax.plot(time_high, high_dose['mean'], color='#C44444', linewidth=2.5, 
               label='High dose (40mg)', zorder=2)
        ax.fill_between(
            time_high,
            high_dose['mean'] - high_dose['sem'],
            high_dose['mean'] + high_dose['sem'],
            color='#C44444',
            alpha=0.3,
            zorder=1
        )
        
        # Add vertical line at DMT onset
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
        
        # Formatting
        ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{component} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{component}', fontsize=14, fontweight='bold')
        ax.set_xlim(-1, 20)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.tick_params(labelsize=10)
        
        # Add legend only to first panel
        if idx == 0:
            ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
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
