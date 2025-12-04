#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCA Canonical Loadings (Structure Coefficients) using Theil BLUS Residuals.

This script computes and visualizes canonical loadings (correlations between
original variables and canonical variates) using Theil BLUS residualized data.

Structure coefficients are preferred over raw weights because:
- Weights are unstable under multicollinearity
- Loadings represent the correlation between each variable and the latent dimension
- Loadings are interpretable as "how much does this variable contribute to the dimension"

Interpretation threshold: |r| > 0.3 (Cohen's standard for medium effect)

Author: TET Analysis Pipeline
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cross_decomposition import CCA
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))

from scripts.tet.physio_data_loader import TETPhysioDataLoader
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Variable labels for display
PHYSIO_LABELS = {
    'HR': 'Heart Rate (HR)',
    'SMNA_AUC': 'Sympathetic EDA (SMNA)',
    'RVT': 'Resp. Volume/Time (RVT)'
}

TET_LABELS = {
    'pleasantness_z': 'Pleasantness',
    'unpleasantness_z': 'Unpleasantness',
    'emotional_intensity_z': 'Emotional Intensity',
    'interoception_z': 'Interoception',
    'bliss_z': 'Bliss',
    'anxiety_z': 'Anxiety'
}

# Colors
PHYSIO_COLOR = '#2E86AB'  # Blue
TET_COLOR = '#A23B72'     # Magenta
THRESHOLD_COLOR = '#E8E8E8'  # Light gray for non-significant


def load_data():
    """Load and prepare data."""
    loader = TETPhysioDataLoader(
        composite_physio_path='results/composite/arousal_index_long.csv',
        tet_path='results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec=30
    )
    loader.load_physiological_data()
    loader.load_tet_data()
    merged_df = loader.merge_datasets()
    
    analyzer = TETPhysioCCAAnalyzer(merged_df)
    
    return merged_df, analyzer


def compute_loadings_theil(analyzer, state):
    """
    Compute canonical loadings using Theil BLUS residuals.
    
    Loadings = Pearson correlation between original variable and canonical variate.
    
    Returns:
        dict with:
            - physio_loadings: dict {var_name: loading}
            - tet_loadings: dict {var_name: loading}
            - canonical_r: canonical correlation
            - U: physiological canonical variate
            - V: affective canonical variate
    """
    # Get original data
    X, Y = analyzer.prepare_matrices(state)
    
    # Get subject IDs
    state_data = analyzer.data[analyzer.data['state'] == state].copy()
    subject_ids_full = state_data['subject'].values
    X_full = state_data[analyzer.physio_measures].values
    Y_full = state_data[analyzer.tet_affective].values
    valid_mask = ~(np.isnan(X_full).any(axis=1) | np.isnan(Y_full).any(axis=1))
    subject_ids = subject_ids_full[valid_mask]
    
    # Compute Theil BLUS residuals
    Y_blus, X_blus, blus_subject_ids = analyzer._compute_theil_residuals_for_blocks(
        Y, X, subject_ids
    )
    
    # Fit CCA on BLUS residuals
    cca = CCA(n_components=1)
    cca.fit(X_blus, Y_blus)
    
    # Compute canonical variates
    U = (X_blus @ cca.x_weights_).flatten()
    V = (Y_blus @ cca.y_weights_).flatten()
    
    # Canonical correlation
    canonical_r = np.corrcoef(U, V)[0, 1]
    
    # Compute loadings (structure coefficients)
    # Loading = correlation between original variable and canonical variate
    physio_loadings = {}
    for j, var_name in enumerate(analyzer.physio_measures):
        loading = np.corrcoef(X_blus[:, j], U)[0, 1]
        physio_loadings[var_name] = loading
    
    tet_loadings = {}
    for k, var_name in enumerate(analyzer.tet_affective):
        loading = np.corrcoef(Y_blus[:, k], V)[0, 1]
        tet_loadings[var_name] = loading
    
    return {
        'physio_loadings': physio_loadings,
        'tet_loadings': tet_loadings,
        'canonical_r': canonical_r,
        'U': U,
        'V': V,
        'n_obs': len(U)
    }


def create_loadings_comparison_figure(analyzer, output_dir):
    """
    Create side-by-side comparison of loadings for RS and DMT.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    results = {}
    for state in ['RS', 'DMT']:
        results[state] = compute_loadings_theil(analyzer, state)
    
    # Threshold for significance
    threshold = 0.3
    
    for col, state in enumerate(['RS', 'DMT']):
        res = results[state]
        
        # Top row: Physiological loadings
        ax_physio = axes[0, col]
        physio_vars = list(res['physio_loadings'].keys())
        physio_vals = [res['physio_loadings'][v] for v in physio_vars]
        physio_labels = [PHYSIO_LABELS.get(v, v) for v in physio_vars]
        
        colors_physio = [PHYSIO_COLOR if abs(v) > threshold else THRESHOLD_COLOR 
                        for v in physio_vals]
        
        y_pos = np.arange(len(physio_vars))
        bars = ax_physio.barh(y_pos, physio_vals, color=colors_physio, 
                             edgecolor='black', linewidth=0.5, height=0.6)
        
        ax_physio.set_yticks(y_pos)
        ax_physio.set_yticklabels(physio_labels)
        ax_physio.axvline(0, color='black', linewidth=0.5)
        ax_physio.axvline(threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax_physio.axvline(-threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax_physio.set_xlim(-1, 1)
        ax_physio.set_xlabel('Canonical Loading (r)')
        ax_physio.set_title(f'{state}: Physiological Loadings\n(r = {res["canonical_r"]:.3f})',
                          fontweight='bold')
        ax_physio.grid(True, alpha=0.3, axis='x')
        
        # Add loading values as text
        for i, (val, bar) in enumerate(zip(physio_vals, bars)):
            x_pos = val + 0.05 if val >= 0 else val - 0.05
            ha = 'left' if val >= 0 else 'right'
            weight = 'bold' if abs(val) > threshold else 'normal'
            ax_physio.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, 
                          fontsize=8, fontweight=weight)
        
        # Bottom row: TET loadings
        ax_tet = axes[1, col]
        tet_vars = list(res['tet_loadings'].keys())
        tet_vals = [res['tet_loadings'][v] for v in tet_vars]
        tet_labels_display = [TET_LABELS.get(v, v) for v in tet_vars]
        
        colors_tet = [TET_COLOR if abs(v) > threshold else THRESHOLD_COLOR 
                     for v in tet_vals]
        
        y_pos = np.arange(len(tet_vars))
        bars = ax_tet.barh(y_pos, tet_vals, color=colors_tet,
                          edgecolor='black', linewidth=0.5, height=0.6)
        
        ax_tet.set_yticks(y_pos)
        ax_tet.set_yticklabels(tet_labels_display)
        ax_tet.axvline(0, color='black', linewidth=0.5)
        ax_tet.axvline(threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax_tet.axvline(-threshold, color='red', linewidth=1, linestyle='--', alpha=0.5)
        ax_tet.set_xlim(-1, 1)
        ax_tet.set_xlabel('Canonical Loading (r)')
        ax_tet.set_title(f'{state}: Affective Loadings', fontweight='bold')
        ax_tet.grid(True, alpha=0.3, axis='x')
        
        # Add loading values as text
        for i, (val, bar) in enumerate(zip(tet_vals, bars)):
            x_pos = val + 0.05 if val >= 0 else val - 0.05
            ha = 'left' if val >= 0 else 'right'
            weight = 'bold' if abs(val) > threshold else 'normal'
            ax_tet.text(x_pos, i, f'{val:.2f}', va='center', ha=ha,
                       fontsize=8, fontweight=weight)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=PHYSIO_COLOR, edgecolor='black', label='|r| > 0.3 (Physio)'),
        Patch(facecolor=TET_COLOR, edgecolor='black', label='|r| > 0.3 (Affect)'),
        Patch(facecolor=THRESHOLD_COLOR, edgecolor='black', label='|r| ≤ 0.3'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True)
    
    plt.suptitle('CCA Canonical Loadings (Structure Coefficients)\n'
                'Using Theil BLUS Residuals', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    
    # Save
    output_path = output_dir / 'Figure5_CCA_Loadings_Comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    return results


def create_heatmap_figure(analyzer, output_dir):
    """Create heatmap visualization of loadings."""
    
    results = {}
    for state in ['RS', 'DMT']:
        results[state] = compute_loadings_theil(analyzer, state)
    
    # Prepare data for heatmap
    all_vars = list(PHYSIO_LABELS.keys()) + list(TET_LABELS.keys())
    var_labels = [PHYSIO_LABELS.get(v, TET_LABELS.get(v, v)) for v in all_vars]
    
    data_matrix = np.zeros((len(all_vars), 2))
    
    for col, state in enumerate(['RS', 'DMT']):
        res = results[state]
        for row, var in enumerate(all_vars):
            if var in res['physio_loadings']:
                data_matrix[row, col] = res['physio_loadings'][var]
            elif var in res['tet_loadings']:
                data_matrix[row, col] = res['tet_loadings'][var]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Canonical Loading (r)', fontsize=10)
    
    # Set ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['RS', 'DMT'], fontsize=10, fontweight='bold')
    ax.set_yticks(np.arange(len(var_labels)))
    ax.set_yticklabels(var_labels, fontsize=9)
    
    # Add text annotations
    threshold = 0.3
    for i in range(len(var_labels)):
        for j in range(2):
            val = data_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            weight = 'bold' if abs(val) > threshold else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=8, fontweight=weight)
    
    # Add horizontal line separating physio and TET
    ax.axhline(2.5, color='black', linewidth=2)
    
    # Add labels for sections
    ax.text(-0.7, 1, 'PHYSIO', ha='center', va='center', fontsize=9,
           fontweight='bold', rotation=90)
    ax.text(-0.7, 5.5, 'AFFECT', ha='center', va='center', fontsize=9,
           fontweight='bold', rotation=90)
    
    ax.set_title('Canonical Loadings Heatmap\n(Theil BLUS Residuals)',
                fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'Figure5_CCA_Loadings_Heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    return data_matrix


def print_interpretation(results):
    """Print interpretation of the loadings."""
    
    print("\n" + "=" * 80)
    print("INTERPRETATION OF CANONICAL LOADINGS")
    print("=" * 80)
    
    threshold = 0.3
    
    for state in ['RS', 'DMT']:
        res = results[state]
        print(f"\n{'='*40}")
        print(f"{state} STATE (Canonical r = {res['canonical_r']:.3f})")
        print(f"{'='*40}")
        
        # Physiological interpretation
        print("\nPhysiological Variables (U):")
        sig_physio = []
        for var, loading in sorted(res['physio_loadings'].items(), 
                                   key=lambda x: abs(x[1]), reverse=True):
            marker = "**" if abs(loading) > threshold else "  "
            print(f"  {marker} {PHYSIO_LABELS.get(var, var):30s}: r = {loading:+.3f}")
            if abs(loading) > threshold:
                sig_physio.append((var, loading))
        
        # Affective interpretation
        print("\nAffective Variables (V):")
        sig_tet = []
        for var, loading in sorted(res['tet_loadings'].items(),
                                   key=lambda x: abs(x[1]), reverse=True):
            marker = "**" if abs(loading) > threshold else "  "
            print(f"  {marker} {TET_LABELS.get(var, var):30s}: r = {loading:+.3f}")
            if abs(loading) > threshold:
                sig_tet.append((var, loading))
        
        # Generate interpretation
        print(f"\n→ INTERPRETATION for {state}:")
        if sig_physio and sig_tet:
            physio_desc = ", ".join([f"{PHYSIO_LABELS.get(v, v)} ({l:+.2f})" 
                                    for v, l in sig_physio])
            tet_desc = ", ".join([f"{TET_LABELS.get(v, v)} ({l:+.2f})" 
                                 for v, l in sig_tet])
            print(f"  The latent dimension captures:")
            print(f"  - Physiological: {physio_desc}")
            print(f"  - Affective: {tet_desc}")
        else:
            print("  No clear pattern (no loadings > 0.3)")


def export_loadings_csv(results, output_dir):
    """Export loadings to CSV for further analysis."""
    
    rows = []
    for state in ['RS', 'DMT']:
        res = results[state]
        
        for var, loading in res['physio_loadings'].items():
            rows.append({
                'state': state,
                'variable_set': 'physio',
                'variable': var,
                'variable_label': PHYSIO_LABELS.get(var, var),
                'loading': loading,
                'significant': abs(loading) > 0.3
            })
        
        for var, loading in res['tet_loadings'].items():
            rows.append({
                'state': state,
                'variable_set': 'affect',
                'variable': var,
                'variable_label': TET_LABELS.get(var, var),
                'loading': loading,
                'significant': abs(loading) > 0.3
            })
    
    df = pd.DataFrame(rows)
    output_path = output_dir / 'cca_loadings_theil.csv'
    df.to_csv(output_path, index=False)
    print(f"\nExported loadings to: {output_path}")
    
    return df


def main():
    print("=" * 80)
    print("CCA Canonical Loadings Analysis (Theil BLUS Residuals)")
    print("=" * 80)
    
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    merged_df, analyzer = load_data()
    
    # Create comparison figure
    print("\nCreating loadings comparison figure...")
    results = create_loadings_comparison_figure(analyzer, output_dir)
    
    # Create heatmap
    print("\nCreating heatmap figure...")
    create_heatmap_figure(analyzer, output_dir)
    
    # Print interpretation
    print_interpretation(results)
    
    # Export to CSV
    export_loadings_csv(results, output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
