#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot PCA Results - Generate PCA visualization figures
"""

import argparse
import logging
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_scree(variance_df, output_path, dpi=300):
    """Plot scree plot showing variance explained."""
    logger.info("Creating scree plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(variance_df['component'], variance_df['variance_explained'] * 100, 
            color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, var in enumerate(variance_df['variance_explained']):
        ax1.text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Cumulative variance
    ax2.plot(range(len(variance_df)), variance_df['cumulative_variance'] * 100, 
            marker='o', linewidth=2, markersize=8, color='darkred')
    ax2.axhline(y=75, color='gray', linestyle='--', linewidth=1.5, label='75% threshold')
    ax2.fill_between(range(len(variance_df)), 0, variance_df['cumulative_variance'] * 100, 
                     alpha=0.2, color='darkred')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(variance_df)))
    ax2.set_xticklabels(variance_df['component'])
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_loadings_heatmap(loadings_df, n_components=5, output_path=None, dpi=300):
    """Plot heatmap of PCA loadings."""
    logger.info(f"Creating loadings heatmap...")
    
    loadings_wide = loadings_df.pivot(index='dimension', columns='component', values='loading')
    components = [f'PC{i+1}' for i in range(min(n_components, len(loadings_wide.columns)))]
    loadings_wide = loadings_wide[components]
    loadings_wide.index = [d.replace('_z', '').replace('_', ' ').title() for d in loadings_wide.index]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(loadings_wide, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
                cbar_kws={'label': 'Loading'}, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('TET Dimension', fontsize=12, fontweight='bold')
    ax.set_title('PCA Loadings Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    try:
        parser = argparse.ArgumentParser(description='Generate PCA visualization figures')
        parser.add_argument('--input-dir', default='results/tet/pca', help='Input directory')
        parser.add_argument('--output-dir', default='results/tet/figures', help='Output directory')
        parser.add_argument('--dpi', type=int, default=300, help='Figure resolution')
        args = parser.parse_args()
        
        logger.info("="*80)
        logger.info("TET PCA VISUALIZATION")
        logger.info("="*80)
        
        # Load data
        variance_df = pd.read_csv(os.path.join(args.input_dir, 'pca_variance_explained.csv'))
        loadings_df = pd.read_csv(os.path.join(args.input_dir, 'pca_loadings.csv'))
        
        logger.info(f"Loaded variance: {len(variance_df)} components")
        logger.info(f"Loaded loadings: {len(loadings_df)} rows")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate figures
        plot_scree(variance_df, os.path.join(args.output_dir, 'pca_scree_plot.png'), args.dpi)
        plot_loadings_heatmap(loadings_df, 5, os.path.join(args.output_dir, 'pca_loadings_heatmap.png'), args.dpi)
        
        logger.info("="*80)
        logger.info("PCA VISUALIZATION COMPLETE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
