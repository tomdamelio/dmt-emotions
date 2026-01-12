# -*- coding: utf-8 -*-
"""
Composite Autonomic Arousal Index (PCA-PC1) + LME Analysis (first 9 minutes).

This script:
  1) Loads 30-second window HR, SMNA_AUC, and RVT data for subjects with all three signals
  2) Z-scores each signal within-subject to align scales
  3) Computes PCA on the three z-scored signals and extracts PC1 as ArousalIndex
  4) Fits LME: ArousalIndex ~ State * Dose + window_c + State:window_c + Dose:window_c + (1|subject)
  5) Saves outputs: CSVs, reports, plots (coefficients, marginal means, loadings, scree)

Outputs: results/composite/

Run:
  python scripts/run_composite_arousal_index.py
"""

import os
import sys
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
# Only set TkAgg if no backend is already set (allows Agg when imported from run_figures.py)
if matplotlib.get_backend() == 'agg' or not matplotlib.get_backend():
    pass  # Keep current backend
else:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass  # Ignore if backend already set
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
except Exception:
    mixedlm = None

try:
    from scipy import stats as scistats
except Exception:
    scistats = None


#############################
# Configuration
#############################

# Fixed list of subjects with all three signals valid
SUBJECTS_INTERSECTION = ['S04', 'S06', 'S07', 'S16', 'S18', 'S19', 'S20']

# Analysis window: first 9 minutes (18 windows of 30 seconds each)
N_WINDOWS = 18
WINDOW_SIZE_SEC = 30

# Output directory
OUT_DIR = './results/composite/'
PLOTS_DIR = os.path.join(OUT_DIR, 'plots')

# Input paths (using z-scored data)
SMNA_PATH = './results/eda/smna/smna_auc_long_data_z.csv'
HR_PATH = './results/ecg/hr/hr_minute_long_data_z.csv'
RVT_PATH = './results/resp/rvt/resp_rvt_minute_long_data_z.csv'

# Extended DMT data paths (~19 minutes)
SMNA_EXTENDED_PATH = './results/eda/smna/smna_extended_dmt_z.csv'
HR_EXTENDED_PATH = './results/ecg/hr/hr_extended_dmt_z.csv'
RVT_EXTENDED_PATH = './results/resp/rvt/resp_rvt_extended_dmt_z.csv'


#############################
# Data loading and preparation
#############################

def load_and_prepare(limit_to_9min: bool = True) -> pd.DataFrame:
    """Load HR, SMNA_AUC, and RVT data; merge on complete cases; filter subjects and windows.
    
    Parameters
    ----------
    limit_to_9min : bool
        If True, limit to first 9 minutes (18 windows) for LME analysis.
        If False, load all available windows.
    """
    print("Loading physiological data...")
    
    # Load SMNA (z-scored AUC per 30-second window)
    df_smna = pd.read_csv(SMNA_PATH)
    if 'AUC' in df_smna.columns:
        df_smna = df_smna.rename(columns={'AUC': 'SMNA_AUC'})
    else:
        raise ValueError(f"SMNA data must have 'AUC' column")
    
    # Load HR (z-scored)
    df_hr = pd.read_csv(HR_PATH)
    # Column should be 'HR' based on the z-scored data
    
    # Load RVT (z-scored)
    df_rvt = pd.read_csv(RVT_PATH)
    if 'RSP_RVT' in df_rvt.columns:
        df_rvt = df_rvt.rename(columns={'RSP_RVT': 'RVT'})
    
    print(f"  Loaded SMNA: {len(df_smna)} rows")
    print(f"  Loaded HR: {len(df_hr)} rows")
    print(f"  Loaded RVT: {len(df_rvt)} rows")
    
    # Filter subjects and windows
    def base_filter(df: pd.DataFrame, limit_windows: bool = True) -> pd.DataFrame:
        df = df[df['subject'].isin(SUBJECTS_INTERSECTION)].copy()
        if limit_windows:
            df = df[(df['window'] >= 1) & (df['window'] <= N_WINDOWS)].copy()
        else:
            df = df[df['window'] >= 1].copy()  # Only ensure window >= 1
        df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
        df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
        return df
    
    df_smna = base_filter(df_smna, limit_to_9min)[['subject', 'window', 'State', 'Dose', 'SMNA_AUC']]
    df_hr = base_filter(df_hr, limit_to_9min)[['subject', 'window', 'State', 'Dose', 'HR']]
    df_rvt = base_filter(df_rvt, limit_to_9min)[['subject', 'window', 'State', 'Dose', 'RVT']]
    
    print(f"  After filtering: SMNA={len(df_smna)}, HR={len(df_hr)}, RVT={len(df_rvt)}")
    
    # Merge on complete cases
    df = df_smna.merge(df_hr, on=['subject', 'window', 'State', 'Dose'], how='inner')
    df = df.merge(df_rvt, on=['subject', 'window', 'State', 'Dose'], how='inner')
    
    print(f"  After merge: {len(df)} complete observations")
    
    # Drop any remaining NAs
    df = df.dropna(subset=['SMNA_AUC', 'HR', 'RVT']).copy()
    
    print(f"  After dropping NAs: {len(df)} observations from {df['subject'].nunique()} subjects")
    
    # Recalculate window_c on merged data
    df['window_c'] = df['window'] - df['window'].mean()
    
    # Save merged data
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = '_9min' if limit_to_9min else '_all'
    df.to_csv(os.path.join(OUT_DIR, f'merged_minute_data_complete_cases{suffix}.csv'), index=False)
    print(f"  ✓ Saved: {os.path.join(OUT_DIR, f'merged_minute_data_complete_cases{suffix}.csv')}")
    
    return df


def load_extended_dmt_data() -> Optional[pd.DataFrame]:
    """Load extended DMT data (~19 minutes) from individual modality files.
    
    Returns merged DataFrame with HR, SMNA_AUC, RVT for DMT only, or None if files missing.
    """
    # Check if extended files exist
    if not all(os.path.exists(p) for p in [HR_EXTENDED_PATH, SMNA_EXTENDED_PATH, RVT_EXTENDED_PATH]):
        print("  Warning: Extended DMT data files not found. Run individual scripts first.")
        print(f"    Expected: {HR_EXTENDED_PATH}")
        print(f"    Expected: {SMNA_EXTENDED_PATH}")
        print(f"    Expected: {RVT_EXTENDED_PATH}")
        return None
    
    print("Loading extended DMT data (~19 minutes)...")
    
    # Load extended data
    df_hr = pd.read_csv(HR_EXTENDED_PATH)
    df_smna = pd.read_csv(SMNA_EXTENDED_PATH)
    df_rvt = pd.read_csv(RVT_EXTENDED_PATH)
    
    # Rename columns for consistency
    if 'AUC' in df_smna.columns:
        df_smna = df_smna.rename(columns={'AUC': 'SMNA_AUC'})
    if 'RSP_RVT' in df_rvt.columns:
        df_rvt = df_rvt.rename(columns={'RSP_RVT': 'RVT'})
    
    print(f"  Loaded extended HR: {len(df_hr)} rows, max window={df_hr['window'].max()}")
    print(f"  Loaded extended SMNA: {len(df_smna)} rows, max window={df_smna['window'].max()}")
    print(f"  Loaded extended RVT: {len(df_rvt)} rows, max window={df_rvt['window'].max()}")
    
    # Filter to intersection subjects
    df_hr = df_hr[df_hr['subject'].isin(SUBJECTS_INTERSECTION)].copy()
    df_smna = df_smna[df_smna['subject'].isin(SUBJECTS_INTERSECTION)].copy()
    df_rvt = df_rvt[df_rvt['subject'].isin(SUBJECTS_INTERSECTION)].copy()
    
    # Select columns for merge
    df_hr = df_hr[['subject', 'window', 'State', 'Dose', 'HR']]
    df_smna = df_smna[['subject', 'window', 'State', 'Dose', 'SMNA_AUC']]
    df_rvt = df_rvt[['subject', 'window', 'State', 'Dose', 'RVT']]
    
    # Merge on complete cases
    df = df_hr.merge(df_smna, on=['subject', 'window', 'State', 'Dose'], how='inner')
    df = df.merge(df_rvt, on=['subject', 'window', 'State', 'Dose'], how='inner')
    
    # Drop NAs
    df = df.dropna(subset=['HR', 'SMNA_AUC', 'RVT']).copy()
    
    # Set categorical types
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    
    # Add window_c
    df['window_c'] = df['window'] - df['window'].mean()
    
    print(f"  After merge: {len(df)} observations from {df['subject'].nunique()} subjects")
    print(f"  Window range: {df['window'].min()} to {df['window'].max()}")
    
    # Save extended merged data
    df.to_csv(os.path.join(OUT_DIR, 'merged_extended_dmt_complete_cases.csv'), index=False)
    print(f"  ✓ Saved: {os.path.join(OUT_DIR, 'merged_extended_dmt_complete_cases.csv')}")
    
    return df


def zscore_within_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each signal within subject to align scales."""
    print("Z-scoring signals within subject...")
    
    for col in ['SMNA_AUC', 'HR', 'RVT']:
        df[f'{col}_z'] = df.groupby('subject')[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    
    # Save z-scored data
    df.to_csv(os.path.join(OUT_DIR, 'merged_z_by_subject.csv'), index=False)
    print(f"  ✓ Saved: {os.path.join(OUT_DIR, 'merged_z_by_subject.csv')}")
    
    return df


#############################
# PCA and Arousal Index
#############################

def compute_pca_and_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, np.ndarray]:
    """Compute PCA on z-scored signals and extract PC1 as ArousalIndex."""
    print("Computing PCA...")
    
    X = df[['HR_z', 'SMNA_AUC_z', 'RVT_z']].to_numpy()
    
    # Fit PCA
    pca = PCA(n_components=3, random_state=22)
    pca.fit(X)
    
    # Extract PC1
    pc1_scores = pca.transform(X)[:, 0]
    loadings_pc1 = pca.components_[0]  # order: ['HR_z', 'SMNA_AUC_z', 'RVT_z']
    var_exp_pc1 = float(pca.explained_variance_ratio_[0])
    
    print(f"  PC1 variance explained: {var_exp_pc1:.4f}")
    print(f"  Raw loadings: HR_z={loadings_pc1[0]:.3f}, SMNA_AUC_z={loadings_pc1[1]:.3f}, RVT_z={loadings_pc1[2]:.3f}")
    
    # Check if all raw loadings are positive
    all_positive_raw = np.all(loadings_pc1 > 0)
    print(f"  Raw loadings all positive: {all_positive_raw}")
    
    # Sign convention: ensure PC1 ↑ = greater activation
    # If SMNA_AUC_z loading is negative, flip sign
    sign_flipped = False
    if loadings_pc1[1] < 0:
        print("  ⚠ Flipping PC1 sign to align with activation direction")
        pc1_scores = -pc1_scores
        loadings_pc1 = -loadings_pc1
        sign_flipped = True
    
    print(f"  Final loadings: HR_z={loadings_pc1[0]:.3f}, SMNA_AUC_z={loadings_pc1[1]:.3f}, RVT_z={loadings_pc1[2]:.3f}")
    
    # Verify all final loadings are positive
    all_positive_final = np.all(loadings_pc1 > 0)
    print(f"  Final loadings all positive: {all_positive_final}")
    
    if all_positive_final:
        print("  ✓ All loadings are positive - PC1 reflects coherent autonomic activation")
    else:
        print("  ⚠ Warning: Not all loadings are positive - mixed interpretation")
    
    # Store sign flip information for reporting
    sign_flip_info = {
        'sign_flipped': sign_flipped,
        'all_positive_raw': all_positive_raw,
        'all_positive_final': all_positive_final,
    }
    
    # Add ArousalIndex to dataframe
    df['ArousalIndex'] = pc1_scores
    
    # Save PCA artifacts
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Loadings
    pd.DataFrame({
        'signal': ['HR_z', 'SMNA_AUC_z', 'RVT_z'],
        'loading_pc1': loadings_pc1
    }).to_csv(os.path.join(OUT_DIR, 'pca_loadings_pc1.csv'), index=False)
    
    # Variance explained
    with open(os.path.join(OUT_DIR, 'pca_variance_explained.txt'), 'w', encoding='utf-8') as f:
        f.write(f'PC1_explained_variance_ratio = {var_exp_pc1:.4f}\n')
        f.write(f'PC2_explained_variance_ratio = {pca.explained_variance_ratio_[1]:.4f}\n')
        f.write(f'PC3_explained_variance_ratio = {pca.explained_variance_ratio_[2]:.4f}\n')
    
    # Scree plot - smaller size for better text readability when assembled
    var_ratio = pca.explained_variance_ratio_
    fig_scree, ax_scree = plt.subplots(figsize=(3.5, 2.5))
    # Use yellow/camel color from tab20b
    ax_scree.plot([1, 2, 3], var_ratio, 'o-', linewidth=2, markersize=8, color=tab20b_colors[8])
    ax_scree.set_xlabel('Principal Component', fontsize=AXES_LABEL_SIZE)
    ax_scree.set_ylabel('Explained Variance\nRatio', fontsize=AXES_LABEL_SIZE)
    ax_scree.set_xticks([1, 2, 3])
    ax_scree.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax_scree.grid(True, alpha=0.3)
    ax_scree.spines['top'].set_visible(False)
    ax_scree.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pca_scree.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(PLOTS_DIR, 'pca_scree.png')}")
    
    # Loadings bar plot - smaller size for better text readability when assembled
    fig_load, ax_load = plt.subplots(figsize=(3.5, 2.5))
    
    signal_names = ['ECG', 'EDA', 'Resp']
    x_pos = np.arange(len(signal_names))
    
    # Use yellow/beige colors from tab20b (indices 8-10)
    bar_colors = [tab20b_colors[8], tab20b_colors[9], tab20b_colors[10]]
    
    # Bar plot without edge color
    ax_load.bar(x_pos, loadings_pc1, color=bar_colors, width=0.6)
    
    # Styling
    ax_load.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    ax_load.set_ylabel('PC1 Loading', fontsize=AXES_LABEL_SIZE)
    ax_load.set_xlabel('Physiological signal', fontsize=AXES_LABEL_SIZE)
    ax_load.set_xticks(x_pos)
    ax_load.set_xticklabels(signal_names, fontsize=TICK_LABEL_SIZE)
    ax_load.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    ax_load.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax_load.set_axisbelow(True)
    ax_load.spines['top'].set_visible(False)
    ax_load.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pca_pc1_loadings.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.join(PLOTS_DIR, 'pca_pc1_loadings.png')}")
    
    # Create 3D PCA visualization
    create_pca_3d_plot(df, pca, loadings_pc1)
    
    # Save arousal index long data
    df.to_csv(os.path.join(OUT_DIR, 'arousal_index_long.csv'), index=False)
    print(f"  ✓ Saved: {os.path.join(OUT_DIR, 'arousal_index_long.csv')}")
    
    # Save sign flip information
    with open(os.path.join(OUT_DIR, 'pca_sign_convention.txt'), 'w', encoding='utf-8') as f:
        f.write('PCA SIGN CONVENTION REPORT\n')
        f.write('=' * 60 + '\n\n')
        f.write(f"Sign flipped: {sign_flip_info['sign_flipped']}\n")
        f.write(f"All raw loadings positive: {sign_flip_info['all_positive_raw']}\n")
        f.write(f"All final loadings positive: {sign_flip_info['all_positive_final']}\n\n")
        if sign_flip_info['sign_flipped']:
            f.write("PC1 sign was FLIPPED to ensure SMNA_AUC_z loading is positive.\n")
            f.write("This aligns PC1 with the activation direction.\n")
        else:
            f.write("PC1 sign was NOT flipped - raw loadings already aligned.\n")
        f.write("\nInterpretation:\n")
        if sign_flip_info['all_positive_final']:
            f.write("✓ All final loadings are POSITIVE.\n")
            f.write("  PC1 reflects coherent autonomic activation across all signals.\n")
            f.write("  Higher PC1 = greater HR, SMNA, and RVT simultaneously.\n")
        else:
            f.write("⚠ Not all final loadings are positive.\n")
            f.write("  PC1 reflects mixed autonomic patterns.\n")
    
    print(f"  ✓ Saved: {os.path.join(OUT_DIR, 'pca_sign_convention.txt')}")
    
    return df, var_exp_pc1, loadings_pc1, sign_flip_info


def create_pca_3d_plot(df: pd.DataFrame, pca, loadings_pc1: np.ndarray) -> None:
    """Create interactive 3D visualization of PCA space with PC1 loading vector using Plotly."""
    print("Creating interactive 3D PCA visualization...")
    
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  ⚠ Plotly not available. Install with: pip install plotly")
        print("  Skipping 3D visualization.")
        return
    
    # Subsample data for readability (max 1000 points)
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df.copy()
    
    # Extract z-scored data
    X_sample = df_sample[['HR_z', 'SMNA_AUC_z', 'RVT_z']].to_numpy()
    
    # Convert tab20b colors to RGB strings for Plotly
    def rgb_to_plotly(color_tuple):
        """Convert matplotlib color tuple to plotly RGB string."""
        r, g, b = [int(c * 255) for c in color_tuple[:3]]
        return f'rgb({r},{g},{b})'
    
    color_light = rgb_to_plotly(tab20b_colors[10])  # Light yellow/beige for points
    color_dark = rgb_to_plotly(tab20b_colors[8])    # Dark yellow/camel for PC1 vector
    
    # Create scatter plot of data points
    scatter = go.Scatter3d(
        x=X_sample[:, 0],
        y=X_sample[:, 1],
        z=X_sample[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=color_light,
            opacity=0.4,
            line=dict(width=0)
        ),
        name='Observations',
        hovertemplate='HR: %{x:.2f}<br>SMNA: %{y:.2f}<br>RVT: %{z:.2f}<extra></extra>'
    )
    
    # Scale factor for PC1 vector (make it visible)
    scale = 3.0
    
    # Create PC1 loading vector as a cone (arrow)
    pc1_arrow = go.Cone(
        x=[loadings_pc1[0] * scale],
        y=[loadings_pc1[1] * scale],
        z=[loadings_pc1[2] * scale],
        u=[loadings_pc1[0] * 0.3],
        v=[loadings_pc1[1] * 0.3],
        w=[loadings_pc1[2] * 0.3],
        colorscale=[[0, color_dark], [1, color_dark]],
        showscale=False,
        sizemode='absolute',
        sizeref=0.5,
        name='PC1 (Arousal Axis)',
        hovertemplate='PC1 Loading Vector<extra></extra>'
    )
    
    # Create line from origin to arrow base
    pc1_line = go.Scatter3d(
        x=[0, loadings_pc1[0] * scale],
        y=[0, loadings_pc1[1] * scale],
        z=[0, loadings_pc1[2] * scale],
        mode='lines',
        line=dict(
            color=color_dark,
            width=8
        ),
        name='PC1 Vector',
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Add text annotation at arrow tip
    annotation_text = go.Scatter3d(
        x=[loadings_pc1[0] * scale * 1.15],
        y=[loadings_pc1[1] * scale * 1.15],
        z=[loadings_pc1[2] * scale * 1.15],
        mode='text',
        text=['PC1<br>(Arousal Axis)'],
        textfont=dict(
            size=14,
            color=color_dark,
            family='Arial Black'
        ),
        showlegend=False,
        hoverinfo='skip'
    )
    
    # Create figure
    fig = go.Figure(data=[scatter, pc1_line, pc1_arrow, annotation_text])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text='HR (z-scored)', font=dict(size=16)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title=dict(text='SMNA (z-scored)', font=dict(size=16)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title=dict(text='RVT (z-scored)', font=dict(size=16)),
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)  # Isometric-like view
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=dict(
            text='PCA Space: Composite Autonomic Arousal Index',
            font=dict(size=18),
            x=0.5,
            xanchor='center'
        )
    )
    
    # Save as interactive HTML
    out_path_html = os.path.join(PLOTS_DIR, 'pca_3d_loadings_interactive.html')
    fig.write_html(out_path_html)
    print(f"  ✓ Saved interactive HTML: {out_path_html}")
    print(f"    → Open in browser to rotate and explore manually")


#############################
# Cross-correlations between signals
#############################

def compute_and_plot_cross_correlations(df: pd.DataFrame) -> str:
    """
    Compute within-subject correlations between HR, SMNA, and RVT for RS and DMT states.
    
    Returns path to the generated heatmap figure.
    """
    print("Computing cross-correlations between signals...")
    
    # Signal columns
    signals = ['HR_z', 'SMNA_AUC_z', 'RVT_z']
    signal_labels = ['HR', 'SMNA', 'RVT']
    
    # Pairs for correlation
    pairs = [
        ('HR_z', 'SMNA_AUC_z', 'HR–SMNA'),
        ('HR_z', 'RVT_z', 'HR–RVT'),
        ('SMNA_AUC_z', 'RVT_z', 'SMNA–RVT')
    ]
    
    # Storage for results
    pearson_results = []
    spearman_results = []
    
    # Compute correlations for each state and subject
    for state in ['RS', 'DMT']:
        state_df = df[df['State'] == state].copy()
        
        for subject in SUBJECTS_INTERSECTION:
            subj_df = state_df[state_df['subject'] == subject].copy()
            
            # Need at least 3 data points for correlation
            if len(subj_df) < 3:
                continue
            
            # Sort by window to ensure proper time series
            subj_df = subj_df.sort_values('window')
            
            # Extract signal arrays
            X = subj_df[signals].values
            
            # Check for NaNs
            if np.any(np.isnan(X)):
                continue
            
            # Compute Pearson correlations
            for sig1, sig2, pair_name in pairs:
                x = subj_df[sig1].values
                y = subj_df[sig2].values
                
                # Pearson
                r_pearson, _ = scistats.pearsonr(x, y)
                pearson_results.append({
                    'state': state,
                    'subject': subject,
                    'pair': pair_name,
                    'r': r_pearson
                })
                
                # Spearman
                r_spearman, _ = scistats.spearmanr(x, y)
                spearman_results.append({
                    'state': state,
                    'subject': subject,
                    'pair': pair_name,
                    'r': r_spearman
                })
    
    # Convert to DataFrames
    df_pearson = pd.DataFrame(pearson_results)
    df_spearman = pd.DataFrame(spearman_results)
    
    # Save CSVs
    pearson_path = os.path.join(OUT_DIR, 'corr_within_subject_pearson.csv')
    spearman_path = os.path.join(OUT_DIR, 'corr_within_subject_spearman.csv')
    
    df_pearson.to_csv(pearson_path, index=False)
    df_spearman.to_csv(spearman_path, index=False)
    
    print(f"  ✓ Saved: {pearson_path}")
    print(f"  ✓ Saved: {spearman_path}")
    
    # Compute average correlation matrices for each state
    corr_matrices = {}
    
    for state in ['RS', 'DMT']:
        state_df = df[df['State'] == state].copy()
        
        # Collect correlation matrices from all subjects
        subject_corr_matrices = []
        
        for subject in SUBJECTS_INTERSECTION:
            subj_df = state_df[state_df['subject'] == subject].copy()
            
            if len(subj_df) < 3:
                continue
            
            subj_df = subj_df.sort_values('window')
            X = subj_df[signals].values
            
            if np.any(np.isnan(X)):
                continue
            
            # Compute correlation matrix
            corr_mat = np.corrcoef(X.T)
            subject_corr_matrices.append(corr_mat)
        
        # Average across subjects
        if subject_corr_matrices:
            avg_corr = np.mean(subject_corr_matrices, axis=0)
            corr_matrices[state] = avg_corr
        else:
            corr_matrices[state] = np.eye(3)
    
    # Create heatmap figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color limits for symmetric colormap
    vmin, vmax = 0, 1
    
    # RS heatmap
    im1 = ax1.imshow(corr_matrices['RS'], cmap='RdPu', vmin=vmin, vmax=vmax, aspect='auto')
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(signal_labels, fontsize=TICK_LABEL_SIZE)
    ax1.set_yticklabels(signal_labels, fontsize=TICK_LABEL_SIZE)
    ax1.set_title('Resting State (RS)', fontsize=AXES_TITLE_SIZE, fontweight='bold')
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, f'{corr_matrices["RS"][i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=TICK_LABEL_SIZE)
    
    # DMT heatmap
    im2 = ax2.imshow(corr_matrices['DMT'], cmap='RdPu', vmin=vmin, vmax=vmax, aspect='auto')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(signal_labels, fontsize=TICK_LABEL_SIZE)
    ax2.set_yticklabels(signal_labels, fontsize=TICK_LABEL_SIZE)
    ax2.set_title('DMT', fontsize=AXES_TITLE_SIZE, fontweight='bold')
    
    # Add correlation values as text
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{corr_matrices["DMT"][i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=TICK_LABEL_SIZE)
    
    # Add colorbar
    cbar = fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r', fontsize=AXES_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    
    # Save figure
    out_path = os.path.join(PLOTS_DIR, 'corr_heatmap_RS_vs_DMT.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {out_path}")
    
    # Print summary statistics
    print("\n  Summary statistics (Pearson r):")
    for state in ['RS', 'DMT']:
        print(f"\n  {state}:")
        state_data = df_pearson[df_pearson['state'] == state]
        for pair_name in ['HR–SMNA', 'HR–RVT', 'SMNA–RVT']:
            pair_data = state_data[state_data['pair'] == pair_name]['r']
            if len(pair_data) > 0:
                mean_r = pair_data.mean()
                std_r = pair_data.std()
                print(f"    {pair_name}: r̄ = {mean_r:.3f} ± {std_r:.3f} (N={len(pair_data)})")
    
    return out_path


#############################
# Dynamic autonomic coherence
#############################

def compute_and_plot_dynamic_coherence(df: pd.DataFrame, window: int = 2) -> str:
    """
    Compute sliding-window correlations between signals over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with HR_z, SMNA_AUC_z, RVT_z, State, Dose, subject, minute
    window : int
        Window size in minutes for sliding correlation (default=2)
    
    Returns
    -------
    str
        Path to the generated figure
    """
    print(f"Computing dynamic autonomic coherence (window={window} min)...")
    
    # Signal pairs
    pairs = [
        ('HR_z', 'SMNA_AUC_z', 'HR–SMNA'),
        ('HR_z', 'RVT_z', 'HR–RVT'),
        ('SMNA_AUC_z', 'RVT_z', 'SMNA–RVT')
    ]
    
    # Storage for results
    results = []
    
    # Compute sliding correlations for each subject, state, dose, pair
    for state in ['RS', 'DMT']:
        state_df = df[df['State'] == state].copy()
        
        for subject in SUBJECTS_INTERSECTION:
            subj_df = state_df[state_df['subject'] == subject].copy()
            
            for dose in ['Low', 'High']:
                dose_df = subj_df[subj_df['Dose'] == dose].copy()
                
                if len(dose_df) < 2:
                    continue
                
                # Sort by window
                dose_df = dose_df.sort_values('window')
                
                # For each window, compute correlation in sliding window
                for window_idx in range(1, N_WINDOWS + 1):
                    # Define window: retrospective [window_idx-window+1, window_idx]
                    window_start = max(1, window_idx - window + 1)
                    window_end = window_idx
                    
                    window_df = dose_df[
                        (dose_df['window'] >= window_start) & 
                        (dose_df['window'] <= window_end)
                    ]
                    
                    # Need at least 2 points for correlation
                    if len(window_df) < 2:
                        for sig1, sig2, pair_name in pairs:
                            results.append({
                                'state': state,
                                'subject': subject,
                                'dose': dose,
                                'pair': pair_name,
                                'window': window_idx,
                                'r': np.nan
                            })
                        continue
                    
                    # Compute correlations for each pair
                    for sig1, sig2, pair_name in pairs:
                        x = window_df[sig1].values
                        y = window_df[sig2].values
                        
                        # Check for NaNs
                        valid = (~np.isnan(x)) & (~np.isnan(y))
                        if np.sum(valid) < 2:
                            r = np.nan
                        else:
                            try:
                                r, _ = scistats.pearsonr(x[valid], y[valid])
                            except Exception:
                                r = np.nan
                        
                        results.append({
                            'state': state,
                            'subject': subject,
                            'dose': dose,
                            'pair': pair_name,
                            'window': window_idx,
                            'r': r
                        })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Aggregate by state, dose, pair, minute
    agg_results = []
    for state in ['RS', 'DMT']:
        for dose in ['Low', 'High']:
            for _, _, pair_name in pairs:
                for window_idx in range(1, N_WINDOWS + 1):
                    subset = df_results[
                        (df_results['state'] == state) &
                        (df_results['dose'] == dose) &
                        (df_results['pair'] == pair_name) &
                        (df_results['window'] == window_idx)
                    ]
                    
                    r_values = subset['r'].dropna()
                    
                    if len(r_values) > 0:
                        mean_r = r_values.mean()
                        sem_r = r_values.sem() if len(r_values) > 1 else np.nan
                        n = len(r_values)
                    else:
                        mean_r = np.nan
                        sem_r = np.nan
                        n = 0
                    
                    agg_results.append({
                        'state': state,
                        'dose': dose,
                        'pair': pair_name,
                        'window': window_idx,
                        'n': n,
                        'mean_r': mean_r,
                        'sem_r': sem_r
                    })
    
    df_agg = pd.DataFrame(agg_results)
    
    # Save CSV
    csv_path = os.path.join(OUT_DIR, f'dynamic_coherence_window{window}.csv')
    df_agg.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    # Create figure: 2 columns (RS, DMT) × 3 rows (one per pair)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
    
    pair_names = ['HR–SMNA', 'HR–RVT', 'SMNA–RVT']
    
    for col_idx, state in enumerate(['RS', 'DMT']):
        for row_idx, pair_name in enumerate(pair_names):
            ax = axes[row_idx, col_idx]
            
            # Plot Low and High
            for dose in ['Low', 'High']:
                subset = df_agg[
                    (df_agg['state'] == state) &
                    (df_agg['dose'] == dose) &
                    (df_agg['pair'] == pair_name)
                ].sort_values('window')
                
                if len(subset) == 0:
                    continue
                
                windows = subset['window'].values
                # Convert windows to time in minutes for x-axis
                time_minutes = (windows - 0.5) * WINDOW_SIZE_SEC / 60.0
                mean_r = subset['mean_r'].values
                sem_r = subset['sem_r'].values
                
                # Color based on dose
                if dose == 'High':
                    color = tab20b_colors[8]   # Dark yellow/camel
                    linestyle = '-'
                else:
                    color = tab20b_colors[10]  # Light yellow/beige
                    linestyle = '--'
                
                # Plot line
                ax.plot(time_minutes, mean_r, color=color, linestyle=linestyle, 
                       linewidth=2.5, marker='o', markersize=4, label=dose)
                
                # Plot SEM band
                ax.fill_between(time_minutes, mean_r - sem_r, mean_r + sem_r, 
                               color=color, alpha=0.2)
            
            # Styling
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.25, axis='y')
            ax.set_xlim(-0.2, 9.2)
            ax.set_ylim(-1, 1)
            
            # Labels
            if row_idx == 2:
                ax.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
            if col_idx == 0:
                ax.set_ylabel(f'{pair_name}\nCorrelation (r)', fontsize=AXES_LABEL_SIZE)
            
            # Title only on top row
            if row_idx == 0:
                ax.set_title(state, fontsize=AXES_TITLE_SIZE, fontweight='bold')
            
            # Legend only on top-right
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE, frameon=True, 
                         fancybox=False, framealpha=0.9)
            
            ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(PLOTS_DIR, f'dynamic_autonomic_coherence_window{window}.png')
    plt.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {fig_path}")
    
    # Print summary: max correlation per state and pair
    print("\n  Peak correlations:")
    for state in ['RS', 'DMT']:
        print(f"\n  {state}:")
        for pair_name in pair_names:
            subset = df_agg[
                (df_agg['state'] == state) &
                (df_agg['pair'] == pair_name)
            ]
            
            if len(subset) > 0:
                # Find max mean_r across all doses and minutes
                max_row = subset.loc[subset['mean_r'].idxmax()]
                time_min = (max_row['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
                print(f"    {pair_name}: max r̄ = {max_row['mean_r']:.3f} "
                      f"at {time_min:.1f} min (window {int(max_row['window'])}, {max_row['dose']})")
    
    return fig_path


#############################
# LME Model
#############################

def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    """Fit LME model on ArousalIndex."""
    print("Fitting LME model...")
    
    if mixedlm is None:
        return None, {'error': 'statsmodels not available'}
    
    # Ensure categorical ordering
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    
    try:
        formula = 'ArousalIndex ~ State * Dose + window_c + State:window_c + Dose:window_c'
        model = mixedlm(formula, df, groups=df['subject'])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fitted = model.fit()
            convergence_warnings = [str(warning.message) for warning in w]
    
    except Exception as e:
        return None, {'error': str(e)}
    
    diagnostics = {
        'aic': getattr(fitted, 'aic', np.nan),
        'bic': getattr(fitted, 'bic', np.nan),
        'loglik': getattr(fitted, 'llf', np.nan),
        'n_obs': getattr(fitted, 'nobs', len(df)),
        'n_groups': len(df['subject'].unique()),
        'convergence_warnings': convergence_warnings,
        'random_effects_var': getattr(fitted, 'cov_re', None),
        'residual_var': getattr(fitted, 'scale', np.nan),
    }
    
    print(f"  Model fit: AIC={diagnostics['aic']:.2f}, BIC={diagnostics['bic']:.2f}")
    if convergence_warnings:
        print(f"  ⚠ Convergence warnings: {len(convergence_warnings)}")
    
    return fitted, diagnostics


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    order = np.argsort(p_array)
    sorted_p = p_array[order]
    adjusted = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[order[i]] = sorted_p[i]
        else:
            adjusted[order[i]] = min(sorted_p[i] * n / (i + 1), adjusted[order[i + 1]])
    
    return np.minimum(adjusted, 1.0).tolist()


def hypothesis_testing_with_fdr(fitted_model) -> Dict:
    """Extract coefficients and apply FDR correction by family."""
    if fitted_model is None:
        return {}
    
    print("Performing hypothesis testing with FDR correction...")
    
    params = fitted_model.params
    pvalues = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    stderr = fitted_model.bse
    
    results: Dict[str, Dict] = {
        'all_params': params.to_dict(),
        'all_pvalues': pvalues.to_dict(),
        'all_stderr': stderr.to_dict(),
        'conf_int': conf_int.to_dict(),
    }
    
    # Define families for FDR correction
    families: Dict[str, List[str]] = {
        'State': [],
        'Dose': [],
        'Interaction': []
    }
    
    for p in ['State[T.DMT]', 'State[T.DMT]:window_c']:
        if p in pvalues.index:
            families['State'].append(p)
    
    for p in ['Dose[T.High]', 'Dose[T.High]:window_c']:
        if p in pvalues.index:
            families['Dose'].append(p)
    
    for p in ['State[T.DMT]:Dose[T.High]']:
        if p in pvalues.index:
            families['Interaction'].append(p)
    
    # Apply FDR within each family
    fdr_results: Dict[str, Dict] = {}
    for fam, plist in families.items():
        if not plist:
            continue
        
        raw = [pvalues[p] for p in plist]
        adj = benjamini_hochberg_correction(raw)
        
        fam_dict: Dict[str, Dict] = {}
        for i, p in enumerate(plist):
            fam_dict[p] = {
                'beta': float(params[p]),
                'se': float(stderr[p]),
                'p_raw': float(pvalues[p]),
                'p_fdr': float(adj[i]),
                'ci_lower': float(conf_int.loc[p, 0]),
                'ci_upper': float(conf_int.loc[p, 1]),
            }
        fdr_results[fam] = fam_dict
    
    # Conditional contrasts
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(stderr['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'description': 'High - Low within RS',
        }
    
    if all(k in params.index for k in ['Dose[T.High]', 'State[T.DMT]:Dose[T.High]']):
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': float(params['State[T.DMT]:Dose[T.High]']),
            'se': float(stderr['State[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['State[T.DMT]:Dose[T.High]']),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }
    
    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    
    print(f"  ✓ FDR correction applied to {len(fdr_results)} families")
    
    return results


#############################
# Reporting
#############################

def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, 
                   df: pd.DataFrame, var_exp: float, loadings: np.ndarray, 
                   sign_flip_info: Dict) -> None:
    """Generate comprehensive text report."""
    print("Generating analysis report...")
    
    lines: List[str] = [
        '=' * 80,
        'COMPOSITE AUTONOMIC AROUSAL INDEX (PCA-PC1) + LME ANALYSIS',
        '=' * 80,
        '',
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        f"Subjects: {', '.join(sorted(df['subject'].unique()))}",
        '',
        'DESIGN:',
        '  Within-subjects 2×2: State (RS vs DMT) × Dose (Low vs High)',
        '  Time windows: 18 thirty-second windows (0-540 seconds = 9 minutes)',
        '  Dependent variable: ArousalIndex (PC1 from PCA on HR_z, SMNA_AUC_z, RVT_z)',
        '',
        'PCA RESULTS:',
        f"  PC1 explained variance: {var_exp:.4f} ({var_exp*100:.2f}%)",
        f"  PC1 loadings:",
        f"    HR_z:      {loadings[0]:7.3f}",
        f"    SMNA_AUC_z: {loadings[1]:7.3f}",
        f"    RVT_z:     {loadings[2]:7.3f}",
        '',
        '  Sign convention:',
        f"    - Sign flipped: {sign_flip_info['sign_flipped']}",
        f"    - All raw loadings positive: {sign_flip_info['all_positive_raw']}",
        f"    - All final loadings positive: {sign_flip_info['all_positive_final']}",
        '',
    ]
    
    if sign_flip_info['all_positive_final']:
        lines.extend([
            '  ✓ All loadings are POSITIVE - PC1 reflects coherent autonomic activation',
            '    Higher ArousalIndex = greater HR, SMNA, and RVT simultaneously',
        ])
    else:
        lines.extend([
            '  ⚠ Not all loadings are positive - PC1 reflects mixed autonomic patterns',
        ])
    
    lines.extend(['', ''])
    
    # Add cross-correlation summary if available
    pearson_path = os.path.join(OUT_DIR, 'corr_within_subject_pearson.csv')
    if os.path.exists(pearson_path):
        df_corr = pd.read_csv(pearson_path)
        lines.extend([
            'CROSS-CORRELATIONS BETWEEN SIGNALS:',
            '  Within-subject Pearson correlations (window-by-window):',
            ''
        ])
        
        for state in ['RS', 'DMT']:
            lines.append(f'  {state}:')
            state_data = df_corr[df_corr['state'] == state]
            for pair_name in ['HR–SMNA', 'HR–RVT', 'SMNA–RVT']:
                pair_data = state_data[state_data['pair'] == pair_name]['r']
                if len(pair_data) > 0:
                    mean_r = pair_data.mean()
                    std_r = pair_data.std()
                    lines.append(f"    {pair_name}: r̄ = {mean_r:.3f} ± {std_r:.3f} (N={len(pair_data)})")
            lines.append('')
        
        lines.extend([
            '  Interpretation: Positive correlations across all signal pairs indicate',
            '  coherent autonomic responses. Higher correlations under DMT suggest',
            '  increased physiological coupling during the psychedelic state.',
            ''
        ])
    
    # Add dynamic coherence summary if available
    dynamic_path = os.path.join(OUT_DIR, 'dynamic_coherence_window2.csv')
    if os.path.exists(dynamic_path):
        df_dynamic = pd.read_csv(dynamic_path)
        lines.extend([
            'DYNAMIC AUTONOMIC COHERENCE:',
            '  Sliding-window correlations (2-minute retrospective window):',
            ''
        ])
        
        for state in ['RS', 'DMT']:
            lines.append(f'  {state} - Peak correlations:')
            state_data = df_dynamic[df_dynamic['state'] == state]
            for pair_name in ['HR–SMNA', 'HR–RVT', 'SMNA–RVT']:
                pair_data = state_data[state_data['pair'] == pair_name]
                if len(pair_data) > 0:
                    max_row = pair_data.loc[pair_data['mean_r'].idxmax()]
                    time_min = (max_row['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
                    lines.append(f"    {pair_name}: max r̄ = {max_row['mean_r']:.3f} "
                               f"at {time_min:.1f} min (window {int(max_row['window'])}, {max_row['dose']})")
            lines.append('')
        
        lines.extend([
            '  Interpretation: Correlations increase in the first minutes post-t0',
            '  and are generally higher in High dose, suggesting coordinated',
            '  upregulation of autonomic activity.',
            ''
        ])
    
    lines.extend([
        'MODEL SPECIFICATION:',
        '  Fixed effects: ArousalIndex ~ State*Dose + window_c + State:window_c + Dose:window_c',
        '  Random effects: ~ 1 | subject',
        '  Where window_c = window - mean(window) [centered time]',
        '',
    ])
    
    if fitted_model is not None:
        lines.extend([
            'MODEL FIT STATISTICS:',
            f"  AIC: {diagnostics.get('aic', np.nan):.2f}",
            f"  BIC: {diagnostics.get('bic', np.nan):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', np.nan):.6f}",
            '',
        ])
        
        warns = diagnostics.get('convergence_warnings', [])
        if warns:
            lines.extend(['CONVERGENCE WARNINGS:', *[f'  - {w}' for w in warns], ''])
        else:
            lines.append('Model converged without warnings\n')
    
    if 'fdr_families' in hypothesis_results:
        lines.extend([
            'HYPOTHESIS TESTING RESULTS (with BH-FDR correction):',
            '=' * 60,
            ''
        ])
        
        for fam, famres in hypothesis_results['fdr_families'].items():
            lines.extend([f'FAMILY {fam.upper()}:', '-' * 30])
            for param, res in famres.items():
                sig = '***' if res['p_fdr'] < 0.001 else '**' if res['p_fdr'] < 0.01 else '*' if res['p_fdr'] < 0.05 else ''
                lines.extend([
                    f'  {param}:',
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {sig}",
                    '',
                ])
            lines.append('')
    
    if 'conditional_contrasts' in hypothesis_results:
        lines.extend(['CONDITIONAL CONTRASTS:', '-' * 30])
        for _, res in hypothesis_results['conditional_contrasts'].items():
            sig = '***' if res['p_raw'] < 0.001 else '**' if res['p_raw'] < 0.01 else '*' if res['p_raw'] < 0.05 else ''
            lines.extend([
                f"  {res['description']}:",
                f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {sig}",
                ''
            ])
    
    lines.extend(['', '=' * 80])
    
    out_path = os.path.join(OUT_DIR, 'lme_analysis_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Saved: {out_path}")


def create_model_summary_txt(diagnostics: Dict, hypothesis_results: Dict, 
                             var_exp: float, loadings: np.ndarray, 
                             sign_flip_info: Dict) -> None:
    """Create compact model summary."""
    lines = [
        'COMPOSITE AROUSAL INDEX - MODEL SUMMARY',
        '=' * 60,
        '',
        'PCA Results:',
        f"  PC1 explained variance: {var_exp:.4f} ({var_exp*100:.2f}%)",
        f"  Loadings: HR_z={loadings[0]:.3f}, SMNA_AUC_z={loadings[1]:.3f}, RVT_z={loadings[2]:.3f}",
    ]
    
    if sign_flip_info['all_positive_final']:
        lines.append("  ✓ All loadings POSITIVE - coherent autonomic activation")
    else:
        lines.append("  ⚠ Mixed loading signs")
    
    lines.append('')
    lines.append('LME Formula:')
    lines.append('  ArousalIndex ~ State*Dose + window_c + State:window_c + Dose:window_c')
    lines.append('  Random: ~ 1 | subject')
    lines.append('')
    lines.append('Model Fit:')
    lines.append(f"  AIC: {diagnostics.get('aic', np.nan):.2f}")
    lines.append(f"  BIC: {diagnostics.get('bic', np.nan):.2f}")
    lines.append(f"  N obs: {diagnostics.get('n_obs', 'N/A')}")
    lines.append(f"  N subjects: {diagnostics.get('n_groups', 'N/A')}")
    lines.append('')
    lines.append('Significant Effects (p_FDR < 0.05):')
    
    sig_found = False
    if 'fdr_families' in hypothesis_results:
        for fam, famres in hypothesis_results['fdr_families'].items():
            for param, res in famres.items():
                if res['p_fdr'] < 0.05:
                    sig = '***' if res['p_fdr'] < 0.001 else '**' if res['p_fdr'] < 0.01 else '*'
                    lines.append(f"  • {param}: β = {res['beta']:.3f} {sig}")
                    sig_found = True
    
    if not sig_found:
        lines.append('  • None')
    
    out_path = os.path.join(OUT_DIR, 'model_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Saved: {out_path}")


#############################
# Plot aesthetics (aligned with centralized figure_config)
#############################

# Import centralized figure configuration
try:
    from figure_config import (
        FONT_SIZE_TITLE, FONT_SIZE_AXIS_LABEL, FONT_SIZE_TICK_LABEL,
        FONT_SIZE_LEGEND, FONT_SIZE_PANEL_LABEL, FONT_SIZE_ANNOTATION,
        FONT_SIZE_TITLE_SMALL, FONT_SIZE_AXIS_LABEL_SMALL,
        FONT_SIZE_TICK_LABEL_SMALL, FONT_SIZE_LEGEND_SMALL,
        LINE_WIDTH, MARKER_SIZE, LEGEND_MARKERSCALE, LEGEND_BORDERPAD,
        LEGEND_HANDLELENGTH, LEGEND_LABELSPACING, LEGEND_BORDERAXESPAD,
        COLOR_COMPOSITE_HIGH, COLOR_COMPOSITE_LOW, DOUBLE_COL_WIDTH,
        apply_rcparams, add_panel_label, style_legend
    )
    # Use centralized config
    AXES_TITLE_SIZE = FONT_SIZE_TITLE
    AXES_LABEL_SIZE = FONT_SIZE_AXIS_LABEL
    TICK_LABEL_SIZE = FONT_SIZE_TICK_LABEL
    AXES_TITLE_SIZE_SMALL = FONT_SIZE_TITLE_SMALL
    AXES_LABEL_SIZE_SMALL = FONT_SIZE_AXIS_LABEL_SMALL
    TICK_LABEL_SIZE_SMALL = FONT_SIZE_TICK_LABEL_SMALL
    LEGEND_FONTSIZE = FONT_SIZE_LEGEND
    LEGEND_FONTSIZE_SMALL = FONT_SIZE_LEGEND_SMALL
    # Apply standardized rcParams
    apply_rcparams()
except ImportError:
    # Fallback to Nature-compliant defaults if config not available
    AXES_TITLE_SIZE = 10
    AXES_LABEL_SIZE = 9
    TICK_LABEL_SIZE = 8
    AXES_TITLE_SIZE_SMALL = 9
    AXES_LABEL_SIZE_SMALL = 8
    TICK_LABEL_SIZE_SMALL = 7
    LEGEND_FONTSIZE = 8
    LEGEND_FONTSIZE_SMALL = 7
    LEGEND_MARKERSCALE = 1.2
    LEGEND_BORDERPAD = 0.4
    LEGEND_HANDLELENGTH = 2.0
    LEGEND_LABELSPACING = 0.5
    LEGEND_BORDERAXESPAD = 0.5

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.titlepad': 6.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.fontsize': LEGEND_FONTSIZE,
    'legend.borderpad': LEGEND_BORDERPAD,
    'legend.handlelength': LEGEND_HANDLELENGTH,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
})

# Composite index uses yellow/beige/camel color scheme from tab20b
tab20b_colors = plt.cm.tab20b.colors
tab20c_colors = plt.cm.tab20c.colors  # Keep for other modalities compatibility
# Yellow/beige group from tab20b: indices 8-11 (darkest to lightest)
try:
    COLOR_RS_HIGH = COLOR_COMPOSITE_HIGH
    COLOR_RS_LOW = COLOR_COMPOSITE_LOW
    COLOR_DMT_HIGH = COLOR_COMPOSITE_HIGH
    COLOR_DMT_LOW = COLOR_COMPOSITE_LOW
except NameError:
    COLOR_RS_HIGH = tab20b_colors[8]    # Dark yellow/camel for High
    COLOR_RS_LOW = tab20b_colors[10]    # Light yellow/beige for Low
    COLOR_DMT_HIGH = tab20b_colors[8]   # Same dark yellow/camel for High
    COLOR_DMT_LOW = tab20b_colors[10]   # Same light yellow/beige for Low


#############################
# Visualization
#############################

def load_lme_results_from_report(report_path: str) -> Dict:
    """Parse LME report to extract coefficient information."""
    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)
    
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    coefficients: Dict[str, Dict] = {}
    current_family: Optional[str] = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('FAMILY STATE:'):
            current_family = 'State'
        elif line.startswith('FAMILY DOSE:'):
            current_family = 'Dose'
        elif line.startswith('FAMILY INTERACTION:'):
            current_family = 'Interaction'
        elif line.startswith('CONDITIONAL CONTRASTS:'):
            current_family = 'Contrasts'
        
        if current_family and line.endswith(':') and not line.startswith('FAMILY') and not line.startswith('CONDITIONAL'):
            name = line.rstrip(':').strip()
            if i + 3 < len(lines):
                beta_line = lines[i + 1].strip()
                ci_line = lines[i + 2].strip()
                p_line = lines[i + 3].strip()
                
                try:
                    if beta_line.startswith('β ='):
                        parts = beta_line.split(',')
                        beta = float(parts[0].split('=')[1].strip())
                        se = float(parts[1].split('=')[1].strip())
                    
                    if ci_line.startswith('95% CI:'):
                        txt = ci_line.replace('95% CI:', '').replace('[', '').replace(']', '').strip()
                        ci_lower = float(txt.split(',')[0])
                        ci_upper = float(txt.split(',')[1])
                    
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw = float([p for p in p_parts if 'p_raw' in p][0].split('=')[1].strip().split()[0])
                        p_fdr_text = [p for p in p_parts if 'p_FDR' in p][0].split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        significance = '***' if '***' in p_fdr_text else '**' if '**' in p_fdr_text else '*' if '*' in p_fdr_text else ''
                    
                    coefficients[name] = {
                        'family': current_family,
                        'beta': beta,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_raw': p_raw,
                        'p_fdr': p_fdr,
                        'significance': significance,
                    }
                except Exception:
                    continue
    
    return coefficients


def prepare_coefficient_data(coefficients: Dict) -> pd.DataFrame:
    """Prepare coefficient data for plotting."""
    order = [
        'State[T.DMT]',
        'Dose[T.High]',
        'State[T.DMT]:window_c',
        'Dose[T.High]:window_c',
        'State[T.DMT]:Dose[T.High]'
    ]
    
    labels = {
        'State[T.DMT]': 'State (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'State[T.DMT]:window_c': 'State × Time',
        'Dose[T.High]:window_c': 'Dose × Time',
        'State[T.DMT]:Dose[T.High]': 'State × Dose'
    }
    
    # Use yellow/beige color scheme for composite index (aligned with tab20b)
    # Yellow/beige group: indices 8-10
    fam_colors = {
        'State': tab20b_colors[8],       # Dark yellow/camel
        'Dose': tab20b_colors[9],        # Medium yellow/beige
        'Interaction': tab20b_colors[10], # Light yellow/beige
    }
    
    rows: List[Dict] = []
    for i, p in enumerate(order):
        if p in coefficients:
            c = coefficients[p]
            rows.append({
                'parameter': p,
                'label': labels.get(p, p),
                'beta': c['beta'],
                'se': c['se'],
                'ci_lower': c['ci_lower'],
                'ci_upper': c['ci_upper'],
                'p_raw': c['p_raw'],
                'p_fdr': c['p_fdr'],
                'significance': c['significance'],
                'family': c['family'],
                'order': i,
                'significant': c['p_fdr'] < 0.05,
                'color': fam_colors.get(c['family'], '#666666'),
            })
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No coefficient data to plot')
    
    return df


def create_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    """Create coefficient plot with CIs and significance asterisks."""
    print("Creating coefficient plot...")
    
    # Smaller dimensions for better text readability when assembled
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    coef_df = coef_df.sort_values('order')
    y_positions = np.arange(len(coef_df))
    
    for _, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
        linewidth = 4.0
        alpha = 1.0
        marker_size = 30  # Match Figure 2
        
        # CI line
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], 
                color=row['color'], linewidth=linewidth, alpha=alpha)
        
        # Point estimate with edge color matching fill (like Figure 2)
        ax.scatter(row['beta'], y_pos, color=row['color'], s=marker_size, 
                  alpha=alpha, edgecolors=row['color'], linewidths=1.5, zorder=3)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.0)
    ax.set_yticks(y_positions)
    # Much smaller font sizes to match other panels when assembled
    ax.set_yticklabels(coef_df['label'], fontsize=7)
    ax.set_xlabel('Coefficient Estimate (β)\nwith 95% CI', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add significance asterisks based on FDR-corrected p-values
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for _, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
        p_fdr = row.get('p_fdr', 1.0)
        if p_fdr < 0.001:
            sig_marker = '***'
        elif p_fdr < 0.01:
            sig_marker = '**'
        elif p_fdr < 0.05:
            sig_marker = '*'
        else:
            sig_marker = ''
        if sig_marker:
            # Position asterisks to the right of the CI
            x_pos = row['ci_upper'] + x_range * 0.02
            ax.text(x_pos, y_pos, sig_marker, fontsize=9, fontweight='bold',
                   va='center', ha='left', color=row['color'])
    
    plt.subplots_adjust(left=0.40, right=0.92)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Compute empirical means and confidence intervals."""
    grouped = df.groupby(['window', 'State', 'Dose'], observed=False)['ArousalIndex']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['window', 'State', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['State'].astype(str) + '_' + stats_df['Dose'].astype(str)
    
    alpha = 1 - confidence
    t_critical = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1) if scistats is not None else 1.96
    stats_df['ci_lower'] = stats_df['mean'] - t_critical * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_critical * stats_df['se']
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    
    return stats_df


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Resample time series to common grid via linear interpolation."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(t_grid, np.nan, dtype=float)
    t_valid = t[valid]
    y_valid = y[valid]
    yg = np.full_like(t_grid, np.nan, dtype=float)
    mask = (t_grid >= t_valid[0]) & (t_grid <= t_valid[-1])
    if np.any(mask):
        yg[mask] = np.interp(t_grid[mask], t_valid, y_valid)
    return yg


def _compute_fdr_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05, alternative: str = 'two-sided') -> Dict:
    """Compute paired t-test across time, apply BH-FDR, and summarize results."""
    result: Dict[str, object] = {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    if scistats is None:
        print("Warning: scipy.stats not available for FDR computation")
        return result
    
    n_time = A.shape[1]
    pvals = np.full(n_time, np.nan, dtype=float)
    
    # Compute t-tests at each time point
    for t in range(n_time):
        a = A[:, t]
        b = B[:, t]
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) >= 2:
            try:
                _, p = scistats.ttest_rel(a[mask], b[mask], alternative=alternative)
                pvals[t] = float(p)
            except Exception:
                pvals[t] = np.nan
    
    valid_idx = np.where(~np.isnan(pvals))[0]
    if len(valid_idx) == 0:
        print("Warning: No valid p-values computed for FDR")
        return result
    
    # Apply BH-FDR correction
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid_idx].tolist())
    adj[valid_idx] = np.array(adj_vals, dtype=float)
    
    # Find significant time points
    sig = adj < alpha
    n_sig = np.sum(sig)
    print(f"FDR analysis: {n_sig}/{len(sig)} time points significant (alpha={alpha}, alternative={alternative})")
    
    # Find contiguous segments of significance
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segments.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    
    print(f"Found {len(segments)} significant segments")
    if segments:
        for j, (x0, x1) in enumerate(segments):
            print(f"  Segment {j+1}: {x0:.1f}s - {x1:.1f}s ({x0/60:.2f} - {x1/60:.2f} min)")
    
    result['pvals'] = pvals.tolist()
    result['pvals_adj'] = adj.tolist()
    result['sig_mask'] = sig.tolist()
    result['segments'] = segments
    return result
    
    result['pvals'] = pvals.tolist()
    result['pvals_adj'] = adj.tolist()
    result['sig_mask'] = sig.tolist()
    result['segments'] = segments
    return result


def create_combined_summary_plot(df: pd.DataFrame) -> Optional[str]:
    """Create combined RS+DMT summary plot (all available windows) for composite arousal index.
    
    Saves results/composite/plots/all_subs_composite.png
    """
    print("Creating combined summary plot...")
    
    # Determine max window across all data
    max_window = int(df['window'].max())
    print(f"  Max window in data: {max_window} ({max_window * WINDOW_SIZE_SEC / 60:.1f} minutes)")
    
    # Create window grid for all available windows
    window_grid = np.arange(1, max_window + 1)
    
    state_data: Dict[str, Dict[str, np.ndarray]] = {}
    
    for kind in ['RS', 'DMT']:
        # Extract data for this state
        state_df = df[df['State'] == kind].copy()
        
        # Get unique subjects
        subjects = sorted(state_df['subject'].unique())
        
        # Build matrices: rows = subjects, cols = windows
        high_mat = []
        low_mat = []
        
        for subject in subjects:
            subj_df = state_df[state_df['subject'] == subject].sort_values('window')
            
            # Create arrays for all windows (fill with NaN if missing)
            high_vals = np.full(max_window, np.nan)
            low_vals = np.full(max_window, np.nan)
            
            # Fill in available data
            high_data = subj_df[subj_df['Dose'] == 'High']
            low_data = subj_df[subj_df['Dose'] == 'Low']
            
            for _, row in high_data.iterrows():
                window_idx = int(row['window']) - 1  # Convert to 0-based index
                if 0 <= window_idx < max_window:
                    high_vals[window_idx] = row['ArousalIndex']
            
            for _, row in low_data.iterrows():
                window_idx = int(row['window']) - 1  # Convert to 0-based index
                if 0 <= window_idx < max_window:
                    low_vals[window_idx] = row['ArousalIndex']
            
            # Only include if we have some data
            if not np.all(np.isnan(high_vals)) and not np.all(np.isnan(low_vals)):
                high_mat.append(high_vals)
                low_mat.append(low_vals)
        
        if not high_mat or not low_mat:
            print(f"Warning: No valid data for {kind} state")
            continue
        
        H = np.array(high_mat)
        L = np.array(low_mat)
        
        # Compute means and SEMs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_h = np.nanmean(H, axis=0)
            mean_l = np.nanmean(L, axis=0)
            
            if H.shape[0] > 1:
                sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
            else:
                sem_h = np.full_like(mean_h, np.nan)
            
            if L.shape[0] > 1:
                sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])
            else:
                sem_l = np.full_like(mean_l, np.nan)
        
        state_data[kind] = {
            'mean_h': mean_h,
            'mean_l': mean_l,
            'sem_h': sem_h,
            'sem_l': sem_l,
            'H_mat': H,
            'L_mat': L,
        }
    
    if len(state_data) != 2:
        print("Warning: Could not create combined plot - missing state data")
        return None
    
    # Create plot - more compact for better assembly
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)
    
    # RS (left)
    rs = state_data['RS']
    print(f"Computing FDR for RS with {rs['H_mat'].shape[0]} subjects, {rs['H_mat'].shape[1]} time points")
    rs_fdr = _compute_fdr_results(rs['H_mat'], rs['L_mat'], window_grid)
    rs_segments = rs_fdr.get('segments', [])
    print(f"Adding {len(rs_segments)} shaded regions to RS panel")
    
    # Convert window grid to time in minutes for plotting
    time_grid = (window_grid - 0.5) * WINDOW_SIZE_SEC / 60.0
    
    # Shade significant window ranges (convert to time)
    for w0, w1 in rs_segments:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax1.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    
    line_h1 = ax1.plot(time_grid, rs['mean_h'], color=COLOR_RS_HIGH, lw=2.5, 
                       marker='o', markersize=5, label='High dose (40mg)')[0]
    ax1.fill_between(time_grid, rs['mean_h'] - rs['sem_h'], rs['mean_h'] + rs['sem_h'], 
                     color=COLOR_RS_HIGH, alpha=0.25)
    line_l1 = ax1.plot(time_grid, rs['mean_l'], color=COLOR_RS_LOW, lw=2.5, 
                       marker='o', markersize=5, label='Low dose (20mg)')[0]
    ax1.fill_between(time_grid, rs['mean_l'] - rs['sem_l'], rs['mean_l'] + rs['sem_l'], 
                     color=COLOR_RS_LOW, alpha=0.25)
    
    legend1 = ax1.legend([line_h1, line_l1], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', 
                        frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, 
                        markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, 
                        labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    
    ax1.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
    # Use yellow/camel color from tab20b for Composite Arousal Index
    ax1.text(-0.20, 0.5, 'Composite Arousal', transform=ax1.transAxes, 
             fontsize=AXES_LABEL_SIZE, fontweight='bold', color=tab20b_colors[8],
             rotation=90, va='center', ha='center')
    ax1.text(-0.12, 0.5, 'Index (PC1)', transform=ax1.transAxes, 
             fontsize=AXES_LABEL_SIZE, fontweight='normal', color='black', 
             rotation=90, va='center', ha='center')
    ax1.set_title('Resting State (RS)', fontweight='bold')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    
    # DMT (right)
    dmt = state_data['DMT']
    print(f"Computing FDR for DMT with {dmt['H_mat'].shape[0]} subjects, {dmt['H_mat'].shape[1]} time points")
    dmt_fdr = _compute_fdr_results(dmt['H_mat'], dmt['L_mat'], window_grid, alternative='greater')
    dmt_segments = dmt_fdr.get('segments', [])
    print(f"Adding {len(dmt_segments)} shaded regions to DMT panel")
    
    # Shade significant window ranges (convert to time)
    for w0, w1 in dmt_segments:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax2.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    
    line_h2 = ax2.plot(time_grid, dmt['mean_h'], color=COLOR_DMT_HIGH, lw=2.5, 
                       marker='o', markersize=5, label='High dose (40mg)')[0]
    ax2.fill_between(time_grid, dmt['mean_h'] - dmt['sem_h'], dmt['mean_h'] + dmt['sem_h'], 
                     color=COLOR_DMT_HIGH, alpha=0.25)
    line_l2 = ax2.plot(time_grid, dmt['mean_l'], color=COLOR_DMT_LOW, lw=2.5, 
                       marker='o', markersize=5, label='Low dose (20mg)')[0]
    ax2.fill_between(time_grid, dmt['mean_l'] - dmt['sem_l'], dmt['mean_l'] + dmt['sem_l'], 
                     color=COLOR_DMT_LOW, alpha=0.25)
    
    legend2 = ax2.legend([line_h2, line_l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', 
                        frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, 
                        markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, 
                        labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)
    
    ax2.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
    ax2.set_title('DMT', fontweight='bold')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    
    # X ticks: dynamic based on max window
    max_time_min = max_window * WINDOW_SIZE_SEC / 60.0
    max_tick = int(np.ceil(max_time_min))
    for ax in (ax1, ax2):
        time_ticks = list(range(0, max_tick + 1))
        ax.set_xticks(time_ticks)
        ax.set_xlim(-0.2, max_time_min + 0.2)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, 'all_subs_composite.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    # Write FDR report
    try:
        report_lines: List[str] = [
            'FDR COMPARISON: High vs Low (Composite Arousal Index, RS and DMT)',
            f"Alpha = {rs_fdr.get('alpha', 0.05)}",
            f"Time resolution: 30-second windows (windows 1-{max_window}, 0-{max_time_min:.1f} minutes)",
            '',
        ]
        
        def _panel_section(name: str, res: Dict):
            report_lines.append(f'PANEL {name}:')
            segs = res.get('segments', [])
            report_lines.append(f"  Significant window ranges (count={len(segs)}):")
            if len(segs) == 0:
                report_lines.append('    - None')
            for (w0, w1) in segs:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
                t1 = w1 * WINDOW_SIZE_SEC / 60.0
                report_lines.append(f"    - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
            # Summary of p-values
            p_adj = [v for v in res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
            if p_adj:
                report_lines.append(f"  Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
            report_lines.append('')
        
        _panel_section('RS', rs_fdr)
        _panel_section('DMT', dmt_fdr)
        
        with open(os.path.join(OUT_DIR, 'fdr_segments_all_subs_composite.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    except Exception as e:
        print(f"Warning: Could not write FDR report: {e}")
    
    print(f"  ✓ Saved: {out_path}")
    return out_path


def create_dmt_only_extended_plot_from_saved() -> Optional[str]:
    """Create DMT-only extended plot from pre-saved extended data."""
    # Load pre-saved extended DMT data
    hr_path = os.path.join(OUT_DIR, 'extended_dmt_hr_z.csv')
    smna_path = os.path.join(OUT_DIR, 'extended_dmt_smna_z.csv')
    rvt_path = os.path.join(OUT_DIR, 'extended_dmt_rvt_z.csv')
    
    df_hr = pd.read_csv(hr_path)
    df_smna = pd.read_csv(smna_path)
    df_rvt = pd.read_csv(rvt_path)
    
    # Merge on complete cases
    df = df_hr.merge(df_smna, on=['subject', 'window', 'Dose'], how='inner')
    df = df.merge(df_rvt, on=['subject', 'window', 'Dose'], how='inner')
    
    # Filter to subjects in intersection
    df = df[df['subject'].isin(SUBJECTS_INTERSECTION)].copy()
    
    if len(df) == 0:
        print("  No data after filtering to intersection subjects")
        return None
    
    # Z-score within subject
    for col in ['SMNA_AUC', 'HR', 'RVT']:
        df[f'{col}_z'] = df.groupby('subject')[col].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    
    # Load PCA loadings
    loadings_path = os.path.join(OUT_DIR, 'pca_loadings_pc1.csv')
    if os.path.exists(loadings_path):
        loadings_df = pd.read_csv(loadings_path)
        loadings = loadings_df['loading_pc1'].values
        print(f"  Using saved PCA loadings: HR_z={loadings[0]:.3f}, SMNA_AUC_z={loadings[1]:.3f}, RVT_z={loadings[2]:.3f}")
    else:
        print("  Warning: PCA loadings not found, computing new PCA")
        from sklearn.decomposition import PCA
        X = df[['HR_z', 'SMNA_AUC_z', 'RVT_z']].to_numpy()
        pca = PCA(n_components=1, random_state=22)
        pca.fit(X)
        loadings = pca.components_[0]
        if loadings[1] < 0:
            loadings = -loadings
    
    # Project onto PC1
    X = df[['HR_z', 'SMNA_AUC_z', 'RVT_z']].to_numpy()
    df['ArousalIndex'] = np.dot(X, loadings)
    
    # Get unique subjects and max window
    subjects = sorted(df['subject'].unique())
    max_window = int(df['window'].max())
    print(f"  Max window: {max_window} ({max_window * WINDOW_SIZE_SEC / 60:.1f} minutes)")
    
    # Build matrices
    high_mat = []
    low_mat = []
    
    for subject in subjects:
        subj_df = df[df['subject'] == subject].sort_values('window')
        
        high_vals = np.full(max_window, np.nan)
        low_vals = np.full(max_window, np.nan)
        
        high_data = subj_df[subj_df['Dose'] == 'High']
        low_data = subj_df[subj_df['Dose'] == 'Low']
        
        for _, row in high_data.iterrows():
            window_idx = int(row['window']) - 1
            if 0 <= window_idx < max_window:
                high_vals[window_idx] = row['ArousalIndex']
        
        for _, row in low_data.iterrows():
            window_idx = int(row['window']) - 1
            if 0 <= window_idx < max_window:
                low_vals[window_idx] = row['ArousalIndex']
        
        if not np.all(np.isnan(high_vals)) and not np.all(np.isnan(low_vals)):
            high_mat.append(high_vals)
            low_mat.append(low_vals)
    
    if not high_mat or not low_mat:
        print("  No valid data for extended plot")
        return None
    
    H = np.array(high_mat)
    L = np.array(low_mat)
    
    print(f"  Data shape: {H.shape[0]} subjects × {H.shape[1]} windows")
    
    # Compute means and SEMs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_h = np.nanmean(H, axis=0)
        mean_l = np.nanmean(L, axis=0)
        
        n_valid_h = np.sum(~np.isnan(H), axis=0)
        n_valid_l = np.sum(~np.isnan(L), axis=0)
        
        sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(n_valid_h)
        sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(n_valid_l)
    
    # Create window grid
    window_grid = np.arange(1, max_window + 1)
    
    # Compute FDR
    print(f"  Computing FDR for DMT High vs Low...")
    fdr_results = _compute_fdr_results(H, L, window_grid, alternative='greater')
    segments = fdr_results.get('segments', [])
    print(f"  Found {len(segments)} significant segments")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    time_grid = (window_grid - 0.5) * WINDOW_SIZE_SEC / 60.0
    
    # Shade significant segments
    for w0, w1 in segments:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
        t1 = w1 * WINDOW_SIZE_SEC / 60.0
        ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    
    # Plot
    l1 = ax.plot(time_grid, mean_h, color=COLOR_DMT_HIGH, lw=2.5, 
                 marker='o', markersize=4, label='High dose (40mg)')[0]
    ax.fill_between(time_grid, mean_h - sem_h, mean_h + sem_h, 
                    color=COLOR_DMT_HIGH, alpha=0.25)
    
    l2 = ax.plot(time_grid, mean_l, color=COLOR_DMT_LOW, lw=2.5, 
                 marker='o', markersize=4, label='Low dose (20mg)')[0]
    ax.fill_between(time_grid, mean_l - sem_l, mean_l + sem_l, 
                    color=COLOR_DMT_LOW, alpha=0.25)
    
    # Legend
    leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', 
                   frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, 
                   markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, 
                   labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    
    # Labels
    ax.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
    ax.text(-0.20, 0.5, 'Composite Arousal', transform=ax.transAxes, 
            fontsize=AXES_LABEL_SIZE, fontweight='bold', color=tab20b_colors[8],
            rotation=90, va='center', ha='center')
    ax.text(-0.12, 0.5, 'Index (PC1)', transform=ax.transAxes, 
            fontsize=AXES_LABEL_SIZE, fontweight='normal', color='black', 
            rotation=90, va='center', ha='center')
    ax.set_title('DMT', fontweight='bold')
    
    # Grid
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    
    # X-axis
    max_time_min = max_window * WINDOW_SIZE_SEC / 60.0
    max_tick = int(np.ceil(max_time_min))
    time_ticks = list(range(0, max_tick + 1))
    ax.set_xticks(time_ticks)
    ax.set_xlim(-0.2, max_time_min + 0.2)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(PLOTS_DIR, 'all_subs_dmt_composite.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    # Write FDR report
    try:
        report_lines: List[str] = [
            'FDR COMPARISON: High vs Low (Composite Arousal Index, DMT only)',
            f"Alpha = {fdr_results.get('alpha', 0.05)}",
            f"Time resolution: 30-second windows (windows 1-{max_window}, 0-{max_time_min:.1f} minutes)",
            '',
            f"Significant window ranges (count={len(segments)}):",
        ]
        
        if len(segments) == 0:
            report_lines.append('  - None')
        else:
            for w0, w1 in segments:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
                t1 = w1 * WINDOW_SIZE_SEC / 60.0
                report_lines.append(f"  - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
        
        p_adj = [v for v in fdr_results.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
        if p_adj:
            report_lines.append('')
            report_lines.append(f"Min p_FDR: {np.nanmin(p_adj):.6f}")
            report_lines.append(f"Median p_FDR: {np.nanmedian(p_adj):.6f}")
        
        with open(os.path.join(OUT_DIR, 'fdr_segments_all_subs_dmt_composite.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ Saved FDR report")
    
    except Exception as e:
        print(f"  Warning: Could not write FDR report: {e}")
    
    print(f"  ✓ Saved: {out_path}")
    return out_path


def create_dmt_only_extended_plot(df: pd.DataFrame) -> Optional[str]:
    """Create DMT-only extended plot using all available windows for composite arousal index.
    
    Similar to all_subs_dmt_ecg_hr.png but for PC1 (ArousalIndex).
    Uses the provided dataframe which should already have ArousalIndex computed.
    
    Saves results/composite/plots/all_subs_dmt_composite.png
    """
    print("Creating DMT-only extended plot...")
    
    # Extract DMT data only
    dmt_df = df[df['State'] == 'DMT'].copy()
    
    if len(dmt_df) == 0:
        print("Warning: No DMT data found")
        return None
    
    # Get unique subjects
    subjects = sorted(dmt_df['subject'].unique())
    
    # Determine max window across all subjects
    max_window = int(dmt_df['window'].max())
    print(f"  Max window in DMT data: {max_window} ({max_window * WINDOW_SIZE_SEC / 60:.1f} minutes)")
    
    # Build matrices: rows = subjects, cols = windows
    high_mat = []
    low_mat = []
    
    for subject in subjects:
        subj_df = dmt_df[dmt_df['subject'] == subject].sort_values('window')
        
        # Create arrays for all windows (fill with NaN if missing)
        high_vals = np.full(max_window, np.nan)
        low_vals = np.full(max_window, np.nan)
        
        # Fill in available data
        high_data = subj_df[subj_df['Dose'] == 'High']
        low_data = subj_df[subj_df['Dose'] == 'Low']
        
        for _, row in high_data.iterrows():
            window_idx = int(row['window']) - 1  # Convert to 0-based index
            if 0 <= window_idx < max_window:
                high_vals[window_idx] = row['ArousalIndex']
        
        for _, row in low_data.iterrows():
            window_idx = int(row['window']) - 1  # Convert to 0-based index
            if 0 <= window_idx < max_window:
                low_vals[window_idx] = row['ArousalIndex']
        
        # Only include if we have some data
        if not np.all(np.isnan(high_vals)) and not np.all(np.isnan(low_vals)):
            high_mat.append(high_vals)
            low_mat.append(low_vals)
    
    if not high_mat or not low_mat:
        print("Warning: No valid DMT data for extended plot")
        return None
    
    H = np.array(high_mat)
    L = np.array(low_mat)
    
    print(f"  Data shape: {H.shape[0]} subjects × {H.shape[1]} windows")
    
    # Compute means and SEMs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_h = np.nanmean(H, axis=0)
        mean_l = np.nanmean(L, axis=0)
        
        # Count valid subjects per window for proper SEM
        n_valid_h = np.sum(~np.isnan(H), axis=0)
        n_valid_l = np.sum(~np.isnan(L), axis=0)
        
        sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(n_valid_h)
        sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(n_valid_l)
    
    # Create window grid
    window_grid = np.arange(1, max_window + 1)
    
    # Compute FDR for High vs Low
    print(f"  Computing FDR for DMT High vs Low...")
    fdr_results = _compute_fdr_results(H, L, window_grid, alternative='greater')
    segments = fdr_results.get('segments', [])
    print(f"  Found {len(segments)} significant segments")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Convert window grid to time in minutes for plotting
    time_grid = (window_grid - 0.5) * WINDOW_SIZE_SEC / 60.0
    
    # Shade significant window ranges
    for w0, w1 in segments:
        t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0  # Start of first window
        t1 = w1 * WINDOW_SIZE_SEC / 60.0  # End of last window
        ax.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
    
    # Plot High and Low
    l1 = ax.plot(time_grid, mean_h, color=COLOR_DMT_HIGH, lw=2.5, 
                 marker='o', markersize=4, label='High dose (40mg)')[0]
    ax.fill_between(time_grid, mean_h - sem_h, mean_h + sem_h, 
                    color=COLOR_DMT_HIGH, alpha=0.25)
    
    l2 = ax.plot(time_grid, mean_l, color=COLOR_DMT_LOW, lw=2.5, 
                 marker='o', markersize=4, label='Low dose (20mg)')[0]
    ax.fill_between(time_grid, mean_l - sem_l, mean_l + sem_l, 
                    color=COLOR_DMT_LOW, alpha=0.25)
    
    # Legend
    leg = ax.legend([l1, l2], ['High dose (40mg)', 'Low dose (20mg)'], loc='upper right', 
                   frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, 
                   markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, 
                   labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    
    # Labels
    ax.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
    # Use yellow/camel color from tab20b for Composite Arousal Index
    ax.text(-0.20, 0.5, 'Composite Arousal', transform=ax.transAxes, 
            fontsize=AXES_LABEL_SIZE, fontweight='bold', color=tab20b_colors[8],
            rotation=90, va='center', ha='center')
    ax.text(-0.12, 0.5, 'Index (PC1)', transform=ax.transAxes, 
            fontsize=AXES_LABEL_SIZE, fontweight='normal', color='black', 
            rotation=90, va='center', ha='center')
    ax.set_title('DMT', fontweight='bold')
    
    # Grid
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    
    # X-axis: set ticks based on max time
    max_time_min = max_window * WINDOW_SIZE_SEC / 60.0
    # Round up to nearest minute
    max_tick = int(np.ceil(max_time_min))
    time_ticks = list(range(0, max_tick + 1))
    ax.set_xticks(time_ticks)
    ax.set_xlim(-0.2, max_time_min + 0.2)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    
    # Save figure
    out_path = os.path.join(PLOTS_DIR, 'all_subs_dmt_composite.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    # Write FDR report
    try:
        report_lines: List[str] = [
            'FDR COMPARISON: High vs Low (Composite Arousal Index, DMT only)',
            f"Alpha = {fdr_results.get('alpha', 0.05)}",
            f"Time resolution: 30-second windows (windows 1-{max_window}, 0-{max_time_min:.1f} minutes)",
            '',
            f"Significant window ranges (count={len(segments)}):",
        ]
        
        if len(segments) == 0:
            report_lines.append('  - None')
        else:
            for w0, w1 in segments:
                t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
                t1 = w1 * WINDOW_SIZE_SEC / 60.0
                report_lines.append(f"  - Window {int(w0)} to {int(w1)} ({t0:.1f}-{t1:.1f} min)")
        
        # Summary of p-values
        p_adj = [v for v in fdr_results.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
        if p_adj:
            report_lines.append('')
            report_lines.append(f"Min p_FDR: {np.nanmin(p_adj):.6f}")
            report_lines.append(f"Median p_FDR: {np.nanmedian(p_adj):.6f}")
        
        with open(os.path.join(OUT_DIR, 'fdr_segments_all_subs_dmt_composite.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ Saved FDR report: {os.path.join(OUT_DIR, 'fdr_segments_all_subs_dmt_composite.txt')}")
    
    except Exception as e:
        print(f"  Warning: Could not write FDR report: {e}")
    
    print(f"  ✓ Saved: {out_path}")
    return out_path


def create_stacked_subjects_plot(df: pd.DataFrame) -> Optional[str]:
    """Create stacked per-subject figure (RS left, DMT right) for composite arousal index.
    
    Saves results/composite/plots/stacked_subs_composite.png
    """
    print("Creating stacked subjects plot...")
    
    # Stacked per-subject figure specific sizes (use centralized config)
    STACKED_AXES_LABEL_SIZE = AXES_LABEL_SIZE
    STACKED_TICK_LABEL_SIZE = TICK_LABEL_SIZE
    STACKED_SUBJECT_FONTSIZE = AXES_TITLE_SIZE
    
    # Get unique subjects
    subjects = sorted(df['subject'].unique())
    
    if not subjects:
        print("Warning: No subjects found for stacked plot")
        return None
    
    n = len(subjects)
    
    # Larger figure to keep typography proportional
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(18, max(6.0, 3.2 * n)),
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.8, 'wspace': 0.35}
    )
    if n == 1:
        axes = np.array([axes])
    
    time_ticks = np.arange(0, 10)  # 0-9 minutes for x-axis
    
    from matplotlib.lines import Line2D
    
    for i, subject in enumerate(subjects):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]
        
        subj_df = df[df['subject'] == subject].copy()
        
        # RS data
        rs_df = subj_df[subj_df['State'] == 'RS'].sort_values('window')
        rs_high = rs_df[rs_df['Dose'] == 'High']
        rs_low = rs_df[rs_df['Dose'] == 'Low']
        
        if len(rs_high) > 0:
            # Convert window to time in minutes for x-axis
            time_minutes = (rs_high['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
            ax_rs.plot(time_minutes, rs_high['ArousalIndex'], 
                      color=COLOR_RS_HIGH, lw=1.4, marker='o', markersize=3)
        if len(rs_low) > 0:
            time_minutes = (rs_low['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
            ax_rs.plot(time_minutes, rs_low['ArousalIndex'], 
                      color=COLOR_RS_LOW, lw=1.4, marker='o', markersize=3)
        
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.set_ylabel(r'$\mathbf{Composite\ Arousal}$' + '\nIndex (PC1)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(-0.2, 9.2)
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=COLOR_RS_HIGH, lw=1.4, label='High dose (40mg)'),
            Line2D([0], [0], color=COLOR_RS_LOW, lw=1.4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, 
           fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, 
           borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)
        
        # DMT data
        dmt_df = subj_df[subj_df['State'] == 'DMT'].sort_values('window')
        dmt_high = dmt_df[dmt_df['Dose'] == 'High']
        dmt_low = dmt_df[dmt_df['Dose'] == 'Low']
        
        if len(dmt_high) > 0:
            # Convert window to time in minutes for x-axis
            time_minutes = (dmt_high['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
            ax_dmt.plot(time_minutes, dmt_high['ArousalIndex'], 
                       color=COLOR_DMT_HIGH, lw=1.4, marker='o', markersize=3)
        if len(dmt_low) > 0:
            time_minutes = (dmt_low['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
            ax_dmt.plot(time_minutes, dmt_low['ArousalIndex'], 
                       color=COLOR_DMT_LOW, lw=1.4, marker='o', markersize=3)
        
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.set_ylabel(r'$\mathbf{Composite\ Arousal}$' + '\nIndex (PC1)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(-0.2, 9.2)
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=COLOR_DMT_HIGH, lw=1.4, label='High dose (40mg)'),
            Line2D([0], [0], color=COLOR_DMT_LOW, lw=1.4, label='Low dose (20mg)'),
        ], loc='upper right', frameon=True, fancybox=False, 
           fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, 
           borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)
        
        # Ticks
        ax_rs.set_xticks(time_ticks)
        ax_dmt.set_xticks(time_ticks)
    
    fig.tight_layout(pad=2.0)
    
    # Add subject codes centered between columns, aligned to final layout
    for i, subject in enumerate(subjects):
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 1].get_position()
        y_center = (pos_left.y0 + pos_left.y1) / 2.0
        x_center = (pos_left.x1 + pos_right.x0) / 2.0
        fig.text(
            x_center,
            y_center + 0.02,
            subject,
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=STACKED_SUBJECT_FONTSIZE,
            transform=fig.transFigure,
        )
    
    out_path = os.path.join(PLOTS_DIR, 'stacked_subs_composite.png')
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {out_path}")
    return out_path


def create_marginal_means_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    """Create marginal means plot over time."""
    print("Creating marginal means plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    conditions = stats_df['condition'].unique()
    for condition in sorted(conditions):
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('window')
        if len(cond_data) == 0:
            continue
        
        if condition == 'RS_Low':
            color = COLOR_RS_LOW
        elif condition == 'RS_High':
            color = COLOR_RS_HIGH
        elif condition == 'DMT_Low':
            color = COLOR_DMT_LOW
        elif condition == 'DMT_High':
            color = COLOR_DMT_HIGH
        else:
            color = '#666666'
        
        # Convert window to time in minutes for x-axis
        time_minutes = (cond_data['window'] - 0.5) * WINDOW_SIZE_SEC / 60.0
        ax.plot(time_minutes, cond_data['mean'], color=color, linewidth=2.5, 
                label=condition.replace('_', ' '), marker='o', markersize=5)
        ax.fill_between(time_minutes, cond_data['ci_lower'], cond_data['ci_upper'], 
                       color=color, alpha=0.2)
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Arousal Index (PC1)')
    ticks = list(range(0, 10))  # 0-9 minutes
    ax.set_xticks(ticks)
    ax.set_xlim(-0.2, 9.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                      fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, 
                      borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, 
                      borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


#############################
# Main execution
#############################

def main() -> bool:
    """Main execution function."""
    print("\n" + "=" * 80)
    print("COMPOSITE AUTONOMIC AROUSAL INDEX ANALYSIS")
    print("=" * 80 + "\n")
    
    try:
        # 1. Load and prepare data
        df = load_and_prepare()
        
        # 2. Z-score within subject
        df = zscore_within_subject(df)
        
        # 3. Compute PCA and arousal index
        df, var_exp, loadings, sign_flip_info = compute_pca_and_index(df)
        
        # 4. Compute cross-correlations between signals (DISABLED)
        # compute_and_plot_cross_correlations(df)
        
        # 5. Compute dynamic autonomic coherence (DISABLED)
        # compute_and_plot_dynamic_coherence(df, window=2)
        
        # 6. Fit LME model
        fitted, diagnostics = fit_lme_model(df)
        
        if fitted is None:
            print("\n⚠ LME model fitting failed!")
            return False
        
        # 7. Hypothesis testing with FDR
        hypothesis_results = hypothesis_testing_with_fdr(fitted)
        
        # 8. Generate reports
        generate_report(fitted, diagnostics, hypothesis_results, df, var_exp, loadings, sign_flip_info)
        create_model_summary_txt(diagnostics, hypothesis_results, var_exp, loadings, sign_flip_info)
        
        # 9. Create plots
        # Coefficient plot
        report_path = os.path.join(OUT_DIR, 'lme_analysis_report.txt')
        coefficients = load_lme_results_from_report(report_path)
        coef_df = prepare_coefficient_data(coefficients)
        create_coefficient_plot(coef_df, os.path.join(PLOTS_DIR, 'lme_coefficient_plot.png'))
        
        # Load extended DMT data for plots (~19 minutes)
        print("\nLoading extended DMT data for visualization...")
        df_extended_dmt = load_extended_dmt_data()
        
        if df_extended_dmt is not None and len(df_extended_dmt) > 0:
            print(f"  Extended DMT data loaded: {len(df_extended_dmt)} observations, max window: {int(df_extended_dmt['window'].max())}")
            
            # Z-score within subject for extended DMT data
            print("  Z-scoring extended DMT data within subject...")
            for col in ['SMNA_AUC', 'HR', 'RVT']:
                df_extended_dmt[f'{col}_z'] = df_extended_dmt.groupby('subject')[col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
            
            # Use saved PCA loadings to compute ArousalIndex on extended DMT data
            loadings_path = os.path.join(OUT_DIR, 'pca_loadings_pc1.csv')
            if os.path.exists(loadings_path):
                loadings_df = pd.read_csv(loadings_path)
                loadings_extended = loadings_df['loading_pc1'].values
                X_extended = df_extended_dmt[['HR_z', 'SMNA_AUC_z', 'RVT_z']].to_numpy()
                df_extended_dmt['ArousalIndex'] = np.dot(X_extended, loadings_extended)
                print(f"  ✓ Applied saved PCA loadings to extended DMT data")
            else:
                print("  Warning: Could not find saved PCA loadings, using raw z-scores")
                df_extended_dmt['ArousalIndex'] = df_extended_dmt[['HR_z', 'SMNA_AUC_z', 'RVT_z']].mean(axis=1)
            
            # For combined plot (RS + DMT), use 9-min data since RS doesn't have extended data
            df_extended = df.copy()
        else:
            print("  Warning: Extended DMT data not available")
            print("  Run individual scripts first to generate extended data:")
            print("    python pipelines/run_ecg_hr_analysis.py")
            print("    python pipelines/run_eda_smna_analysis.py")
            print("    python pipelines/run_resp_rvt_analysis.py")
            print("  Using 9-min data for all plots")
            df_extended = df.copy()
            df_extended_dmt = None
        
        # Combined summary plot (RS + DMT panels with FDR) - using 9-min data (RS doesn't have extended)
        create_combined_summary_plot(df_extended)
        
        # DMT-only extended plot (~19 minutes) - using extended DMT data if available
        if df_extended_dmt is not None and len(df_extended_dmt) > 0:
            create_dmt_only_extended_plot(df_extended_dmt)
        else:
            print("  Skipping extended DMT plot (no extended data available)")
        
        # Stacked per-subject plot - using 9-min data
        create_stacked_subjects_plot(df_extended)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to: {OUT_DIR}")
        print(f"  - arousal_index_long.csv: Long-format data with ArousalIndex")
        print(f"  - pca_loadings_pc1.csv: PC1 loadings for each signal")
        print(f"  - pca_variance_explained.txt: Variance explained by each PC")
        print(f"  - lme_analysis_report.txt: Full LME analysis report")
        print(f"  - model_summary.txt: Compact model summary")
        print(f"  - fdr_segments_all_subs_composite.txt: FDR analysis report")
        print(f"  - plots/pca_scree.png: Scree plot")
        print(f"  - plots/pca_pc1_loadings.png: PC1 loadings bar plot")
        print(f"  - plots/pca_3d_loadings_interactive.html: 3D PCA space (interactive, open in browser)")
        print(f"  - plots/lme_coefficient_plot.png: LME coefficients with CIs")
        print(f"  - plots/all_subs_composite.png: Combined RS+DMT summary with FDR (all windows)")
        print(f"  - plots/all_subs_dmt_composite.png: DMT-only extended timecourse with FDR (all windows)")
        print(f"  - plots/stacked_subs_composite.png: Stacked per-subject timecourses")
        print()
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

