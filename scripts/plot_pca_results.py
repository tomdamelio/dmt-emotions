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
    """Plot scree plot showing variance explained (single panel with PC1-PC2 highlighted)."""
    logger.info("Creating scree plot...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create color array: first 2 components in dark blue, rest in light gray
    colors = ['#2E5090' if i < 2 else '#B0B0B0' for i in range(len(variance_df))]
    
    # Individual variance
    bars = ax.bar(variance_df['component'], variance_df['variance_explained'] * 100, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on top of bars
    for i, var in enumerate(variance_df['variance_explained']):
        ax.text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold' if i < 2 else 'normal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_loadings_heatmap(loadings_df, n_components=2, output_path=None, dpi=300):
    """Plot heatmap of PCA loadings (PC1 and PC2 only)."""
    logger.info(f"Creating loadings heatmap (PC1 and PC2)...")
    
    loadings_wide = loadings_df.pivot(index='dimension', columns='component', values='loading')
    # Only show PC1 and PC2
    components = ['PC1', 'PC2']
    loadings_wide = loadings_wide[components]
    loadings_wide.index = [d.replace('_z', '').replace('_', ' ').title() for d in loadings_wide.index]
    
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(loadings_wide, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
                cbar_kws={'label': 'Loading'}, linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel('TET Dimension', fontsize=12, fontweight='bold')
    ax.set_title('PCA Loadings Heatmap (Affective Dimensions)', fontsize=14, fontweight='bold')
    
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
        plot_loadings_heatmap(loadings_df, 2, os.path.join(args.output_dir, 'pca_loadings_heatmap.png'), args.dpi)
        
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
