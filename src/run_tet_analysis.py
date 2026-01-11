#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TET Analysis Pipeline - Self-contained script for Figure 3

Generates publication-ready analysis of Temporal Experience Tracking (TET) data:
- PCA on 6 affective dimensions (PC1=Arousal, PC2=Valence)
- LME models for State × Dose interactions
- Figure 3 with 5 panels (A-E)

Expected Results (from paper):
- PC1 (Arousal): 41.0% variance - loads on Emotional Intensity, Interoception, Anxiety, Unpleasantness
- PC2 (Valence): 31.8% variance - bipolar: Pleasantness/Bliss (+) vs Unpleasantness/Anxiety (-)
- Cumulative: 72.8%
- State × Dose interaction for Arousal: β = 0.47, 95% CI [0.32, 0.61], p < .001
- State × Dose interaction for Valence: β = −0.50, 95% CI [−0.70, −0.30], p < .001

Outputs:
    results/tet/figures/figure3_tet_analysis.png
    results/tet/pca/ - PCA results
    results/tet/lme/ - LME results

Usage:
    python src/run_tet_analysis.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tet'
FIGURES_DIR = RESULTS_DIR / 'figures'
DATA_PATH = RESULTS_DIR / 'preprocessed' / 'tet_preprocessed.csv'

# Affective dimensions for PCA (order matters for loadings interpretation)
# Order: Anxiety, Bliss, Emotional Intensity, Interoception, Pleasantness, Unpleasantness
AFFECTIVE_COLS = [
    'anxiety', 'bliss', 'emotional_intensity', 
    'interoception', 'pleasantness', 'unpleasantness'
]

# Z-scored versions
AFFECTIVE_COLS_Z = [f'{col}_z' for col in AFFECTIVE_COLS]

# Display labels for figures (matching paper Figure 3e)
AFFECTIVE_LABELS = {
    'anxiety_z': 'Anxiety',
    'bliss_z': 'Bliss', 
    'emotional_intensity_z': 'Emotional Intensity',
    'interoception_z': 'Interoception',
    'pleasantness_z': 'Pleasantness',
    'unpleasantness_z': 'Unpleasantness'
}

# Colors
COLOR_HIGH = '#5E4FA2'  # Purple for High dose (40mg)
COLOR_LOW = '#9E9AC8'   # Light purple for Low dose (20mg)
COLOR_AROUSAL = '#5E4FA2'
COLOR_VALENCE = '#9E9AC8'

# Analysis time window (0-20 min for DMT, 0-10 min for RS)
DMT_MAX_TIME_MIN = 20
RS_MAX_TIME_MIN = 10


def load_data():
    """Load preprocessed TET data."""
    print("Loading TET data...")
    df = pd.read_csv(DATA_PATH)
    df['time_min'] = df['t_sec'] / 60
    
    # Filter to analysis windows
    # DMT: 0-20 min, RS: 0-10 min
    df_filtered = df[
        ((df['state'] == 'DMT') & (df['time_min'] <= DMT_MAX_TIME_MIN)) |
        ((df['state'] == 'RS') & (df['time_min'] <= RS_MAX_TIME_MIN))
    ].copy()
    
    print(f"  Loaded {len(df_filtered)} observations from {df_filtered['subject'].nunique()} subjects")
    print(f"  DMT sessions: {len(df_filtered[df_filtered['state'] == 'DMT'])} obs")
    print(f"  RS sessions: {len(df_filtered[df_filtered['state'] == 'RS'])} obs")
    return df_filtered


def compute_pca(df):
    """
    Compute PCA on affective dimensions.
    
    PCA is fit on ALL data (DMT + RS) to capture the full variance structure.
    This matches the paper methodology where PCA identifies latent dimensions
    across all experimental conditions.
    
    Returns:
        df with PC scores added
        loadings DataFrame  
        variance_explained array
    """
    print("\nComputing PCA on affective dimensions...")
    
    # Use ALL data for PCA (both DMT and RS)
    # This captures the full variance structure across conditions
    X = df[AFFECTIVE_COLS_Z].values
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        print(f"  Warning: Found {np.sum(np.isnan(X))} NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Standardize at group level (across all subjects/conditions)
    # This ensures each dimension contributes equally to PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components first to get full variance structure
    pca = PCA(n_components=min(6, X_scaled.shape[1]))
    pca.fit(X_scaled)
    
    # Transform data
    scores = pca.transform(X_scaled)
    
    # Add scores to dataframe
    df = df.copy()
    for i in range(pca.n_components_):
        df[f'PC{i+1}'] = scores[:, i]
    
    # Create loadings DataFrame with proper dimension labels
    loadings = pd.DataFrame(
        pca.components_.T,
        index=AFFECTIVE_COLS_Z,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    variance_explained = pca.explained_variance_ratio_
    
    print(f"  PC1 variance: {variance_explained[0]*100:.1f}%")
    print(f"  PC2 variance: {variance_explained[1]*100:.1f}%")
    print(f"  Cumulative (PC1+PC2): {sum(variance_explained[:2])*100:.1f}%")
    
    # Print loadings for interpretation
    print("\n  PC1 loadings (Arousal):")
    for dim in AFFECTIVE_COLS_Z:
        print(f"    {AFFECTIVE_LABELS[dim]}: {loadings.loc[dim, 'PC1']:.2f}")
    
    print("\n  PC2 loadings (Valence):")
    for dim in AFFECTIVE_COLS_Z:
        print(f"    {AFFECTIVE_LABELS[dim]}: {loadings.loc[dim, 'PC2']:.2f}")
    
    return df, loadings, variance_explained


def fit_lme_models(df):
    """
    Fit LME models for Arousal and Valence indices.
    
    Models include State × Dose interaction to test differential dose effects
    during DMT vs resting state.
    
    Model formula:
        Y ~ State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c + (1|Subject)
    
    Reference levels: State='RS', Dose='Baja' (Low)
    
    The State × Dose interaction tests whether the dose effect differs between
    DMT and RS conditions. A positive coefficient indicates greater dose 
    separation during DMT than during RS.
    
    Returns:
        Dictionary with model results
    """
    print("\nFitting LME models...")
    
    # Use all data (DMT + RS) for State × Dose interaction
    df_all = df.copy()
    
    # Center time variable
    df_all['time_c'] = df_all['time_min'] - df_all['time_min'].mean()
    
    # Set reference levels explicitly
    # Reference: State='RS', Dose='Baja' (Low)
    # This means coefficients show effect of DMT vs RS and Alta vs Baja
    df_all['state'] = pd.Categorical(df_all['state'], categories=['RS', 'DMT'], ordered=False)
    df_all['dose'] = pd.Categorical(df_all['dose'], categories=['Baja', 'Alta'], ordered=False)
    
    # Create valence index (pleasantness - unpleasantness)
    df_all['valence_index'] = df_all['pleasantness_z'] - df_all['unpleasantness_z']
    
    results = {}
    
    # Arousal Index = emotional_intensity_z
    # Paper: β = 0.47, 95% CI [0.32, 0.61], p < .001 for State × Dose
    print("  Fitting Arousal (Emotional Intensity) model...")
    model_arousal = smf.mixedlm(
        "emotional_intensity_z ~ state * dose + state * time_c + dose * time_c",
        df_all,
        groups=df_all["subject"],
        re_formula="1"  # Random intercept only
    )
    fit_arousal = model_arousal.fit(reml=True, method='lbfgs')
    results['arousal'] = {'model': fit_arousal}
    
    # Valence Index
    # Paper: β = −0.50, 95% CI [−0.70, −0.30], p < .001 for State × Dose
    print("  Fitting Valence Index model...")
    model_valence = smf.mixedlm(
        "valence_index ~ state * dose + state * time_c + dose * time_c",
        df_all,
        groups=df_all["subject"],
        re_formula="1"
    )
    fit_valence = model_valence.fit(reml=True, method='lbfgs')
    results['valence'] = {'model': fit_valence}
    
    # Individual dimensions
    for dim in AFFECTIVE_COLS:
        dim_z = f'{dim}_z'
        if dim_z in df_all.columns:
            print(f"  Fitting {dim} model...")
            model = smf.mixedlm(
                f"{dim_z} ~ state * dose + state * time_c + dose * time_c",
                df_all,
                groups=df_all["subject"],
                re_formula="1"
            )
            fit = model.fit(reml=True, method='lbfgs')
            results[dim] = {'model': fit}
    
    # Print key results
    _print_lme_summary(results)
    
    return results, df_all


def _print_lme_summary(results):
    """Print summary of key LME results."""
    print("\n  --- LME Summary (State × Dose interaction) ---")
    
    # The interaction term is state[T.DMT]:dose[T.Alta]
    # This represents the additional effect of High dose during DMT
    # compared to the additive effects of DMT and High dose separately
    
    interaction_key = 'state[T.DMT]:dose[T.Alta]'
    
    for name in ['arousal', 'valence']:
        if name in results:
            model = results[name]['model']
            if interaction_key in model.params:
                beta = model.params[interaction_key]
                ci = model.conf_int().loc[interaction_key]
                p = model.pvalues[interaction_key]
                print(f"  {name.capitalize()}: β = {beta:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], p = {p:.2e}")


def compute_time_courses(df):
    """Compute mean ± SEM time courses by dose for DMT sessions."""
    print("\nComputing time courses...")
    
    df_dmt = df[df['state'] == 'DMT'].copy()
    
    # Create valence index
    df_dmt['valence_index'] = df_dmt['pleasantness_z'] - df_dmt['unpleasantness_z']
    
    # Group by dose and time
    time_courses = {}
    
    for dose in ['Alta', 'Baja']:
        df_dose = df_dmt[df_dmt['dose'] == dose]
        
        # Define aggregation columns
        agg_cols = {
            'emotional_intensity_z': ['mean', 'sem'],
            'pleasantness_z': ['mean', 'sem'],
            'unpleasantness_z': ['mean', 'sem'],
            'interoception_z': ['mean', 'sem'],
            'anxiety_z': ['mean', 'sem'],
            'bliss_z': ['mean', 'sem'],
            'valence_index': ['mean', 'sem'],
        }
        
        # Add PC columns if they exist
        for pc in ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6']:
            if pc in df_dose.columns:
                agg_cols[pc] = ['mean', 'sem']
        
        # Aggregate by time bin
        grouped = df_dose.groupby('time_min').agg(agg_cols)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        time_courses[dose] = grouped
        
        print(f"  {dose} dose: {len(grouped)} time points, "
              f"time range: {grouped['time_min'].min():.1f}-{grouped['time_min'].max():.1f} min")
    
    return time_courses


def compute_significance_masks(df):
    """
    Compute significance masks for time series plots.
    
    Computes two types of significance:
    1. state_sig: DMT vs RS at each time bin (for gray shading)
    2. dose_sig: High vs Low dose within DMT (for black bars)
    
    Returns dict with:
    - time_bins: list of time points
    - state_sig: boolean mask where DMT differs from RS (p_FDR < 0.05)
    - dose_sig: boolean mask where High differs from Low dose (p_FDR < 0.05)
    """
    from scipy.stats import ttest_rel
    
    print("\nComputing significance masks...")
    
    # Create valence index for both states
    df = df.copy()
    df['valence_index'] = df['pleasantness_z'] - df['unpleasantness_z']
    
    df_dmt = df[df['state'] == 'DMT'].copy()
    df_rs = df[df['state'] == 'RS'].copy()
    
    # Variables to test
    variables = ['emotional_intensity_z', 'valence_index', 'interoception_z', 
                 'anxiety_z', 'unpleasantness_z', 'pleasantness_z', 'bliss_z']
    
    # Add PC columns if they exist
    for pc in ['PC1', 'PC2']:
        if pc in df_dmt.columns:
            variables.append(pc)
    
    significance = {}
    
    for var in variables:
        if var not in df_dmt.columns:
            continue
            
        # Get unique time bins (use RS time range: 0-10 min for state comparison)
        dmt_time_bins = sorted(df_dmt['time_min'].unique())
        rs_time_bins = sorted(df_rs['time_min'].unique())
        common_time_bins = sorted(set(dmt_time_bins) & set(rs_time_bins))
        
        # =====================================================================
        # 1. State effect: DMT vs RS (for gray shading)
        # =====================================================================
        state_p_values = []
        for t in common_time_bins:
            dmt_data = df_dmt[df_dmt['time_min'] == t]
            rs_data = df_rs[df_rs['time_min'] == t]
            
            # Match by subject for paired test
            subjects = set(dmt_data['subject']) & set(rs_data['subject'])
            if len(subjects) >= 3:
                dmt_vals = dmt_data[dmt_data['subject'].isin(subjects)].groupby('subject')[var].mean()
                rs_vals = rs_data[rs_data['subject'].isin(subjects)].groupby('subject')[var].mean()
                
                common_subjects = sorted(set(dmt_vals.index) & set(rs_vals.index))
                if len(common_subjects) >= 3:
                    _, p = ttest_rel(dmt_vals[common_subjects], rs_vals[common_subjects])
                    state_p_values.append(p if not np.isnan(p) else 1.0)
                else:
                    state_p_values.append(1.0)
            else:
                state_p_values.append(1.0)
        
        # FDR correction for state effect
        if len(state_p_values) > 0 and not all(p == 1.0 for p in state_p_values):
            _, state_p_fdr, _, _ = multipletests(state_p_values, method='fdr_bh')
            state_sig_mask = state_p_fdr < 0.05
        else:
            state_sig_mask = np.zeros(len(common_time_bins), dtype=bool)
        
        # =====================================================================
        # 2. Dose effect: High vs Low within DMT (for black bars)
        # =====================================================================
        dose_p_values = []
        for t in dmt_time_bins:
            high_data = df_dmt[(df_dmt['time_min'] == t) & (df_dmt['dose'] == 'Alta')]
            low_data = df_dmt[(df_dmt['time_min'] == t) & (df_dmt['dose'] == 'Baja')]
            
            subjects = set(high_data['subject']) & set(low_data['subject'])
            if len(subjects) >= 3:
                high_vals = high_data[high_data['subject'].isin(subjects)].groupby('subject')[var].mean()
                low_vals = low_data[low_data['subject'].isin(subjects)].groupby('subject')[var].mean()
                
                common_subjects = sorted(set(high_vals.index) & set(low_vals.index))
                if len(common_subjects) >= 3:
                    _, p = ttest_rel(high_vals[common_subjects], low_vals[common_subjects])
                    dose_p_values.append(p if not np.isnan(p) else 1.0)
                else:
                    dose_p_values.append(1.0)
            else:
                dose_p_values.append(1.0)
        
        # FDR correction for dose effect
        if len(dose_p_values) > 0 and not all(p == 1.0 for p in dose_p_values):
            _, dose_p_fdr, _, _ = multipletests(dose_p_values, method='fdr_bh')
            dose_sig_mask = dose_p_fdr < 0.05
        else:
            dose_sig_mask = np.zeros(len(dmt_time_bins), dtype=bool)
        
        significance[var] = {
            'time_bins': dmt_time_bins,
            'dose_sig': dose_sig_mask,
            'state_time_bins': common_time_bins,
            'state_sig': state_sig_mask,
        }
        
        n_state_sig = np.sum(state_sig_mask)
        n_dose_sig = np.sum(dose_sig_mask)
        if n_state_sig > 0 or n_dose_sig > 0:
            print(f"  {var}: State effect {n_state_sig}/{len(common_time_bins)}, "
                  f"Dose effect {n_dose_sig}/{len(dmt_time_bins)} time bins (p_FDR < 0.05)")
    
    return significance


def save_individual_subplots(df, time_courses, loadings, variance_explained, 
                             lme_results, significance=None):
    """
    Save individual subplot PNGs for Figure 4 assembly.
    
    Generates separate PNG files for each panel component:
    - panel_A_timeseries.png: Arousal, Valence, and 5 individual dimensions
    - panel_B_lme_forest.png: LME coefficients forest plot
    - panel_C_pc_timeseries.png: PC1 and PC2 time courses
    - panel_D_variance.png: Variance explained bar plot
    - panel_E_loadings.png: PCA loadings heatmap
    
    These are assembled by run_figures.py into the final figure_4.png.
    """
    print("\nSaving individual subplots for Figure 4...")
    
    PLOTS_DIR = FIGURES_DIR.parent / 'plots'
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # =========================================================================
    # Panel A: Time series (Arousal, Valence, 5 dimensions)
    # =========================================================================
    fig_a = plt.figure(figsize=(14, 5))
    gs_a = fig_a.add_gridspec(2, 11, height_ratios=[1.2, 0.9], hspace=0.35, wspace=0.4)
    
    # Top row: Arousal and Valence
    ax_arousal = fig_a.add_subplot(gs_a[0, :5])
    ax_valence = fig_a.add_subplot(gs_a[0, 6:])
    
    _plot_timeseries_panel(ax_arousal, time_courses, 'emotional_intensity_z', 
                           'Arousal', subtitle='(Emotional Intensity)', show_legend=True,
                           significance=significance, show_gray_shading=True,
                           show_ylabel=True, is_first_col=True, legend_loc='upper right')
    _plot_timeseries_panel(ax_valence, time_courses, 'valence_index',
                           'Valence', subtitle='(Pleasantness−Unpleasantness)', show_legend=True,
                           significance=significance, show_gray_shading=True,
                           show_ylabel=True, is_first_col=False, legend_loc='lower right')
    
    # Bottom row: 5 individual dimensions
    dims = ['interoception_z', 'anxiety_z', 'unpleasantness_z', 'pleasantness_z', 'bliss_z']
    dim_labels = ['Interoception', 'Anxiety', 'Unpleasantness', 'Pleasantness', 'Bliss']
    
    row1_bbox = gs_a[1, :].get_position(fig_a)
    n_dims = 5
    total_width = row1_bbox.width
    gap_frac = 0.035
    subplot_width = (total_width - (n_dims - 1) * gap_frac * total_width) / n_dims
    
    for i, (dim, label) in enumerate(zip(dims, dim_labels)):
        left = row1_bbox.x0 + i * (subplot_width + gap_frac * total_width)
        ax = fig_a.add_axes([left, row1_bbox.y0, subplot_width, row1_bbox.height])
        _plot_timeseries_panel(ax, time_courses, dim, label, 
                               show_legend=False, small=True, significance=significance,
                               show_gray_shading=True, show_ylabel=(i == 0), is_first_col=(i == 0))
    
    plt.tight_layout()
    path_a = PLOTS_DIR / 'panel_A_timeseries.png'
    fig_a.savefig(path_a, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_a)
    saved_paths['A'] = path_a
    print(f"  Saved: {path_a}")
    
    # =========================================================================
    # Panel B: LME Forest Plot (without label - added during assembly)
    # =========================================================================
    fig_b = plt.figure(figsize=(14, 2.5))
    ax_lme = fig_b.add_subplot(1, 1, 1)
    _plot_lme_forest_horizontal(ax_lme, lme_results, add_label=False)
    plt.tight_layout()
    path_b = PLOTS_DIR / 'panel_B_lme_forest.png'
    fig_b.savefig(path_b, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_b)
    saved_paths['B'] = path_b
    print(f"  Saved: {path_b}")
    
    # =========================================================================
    # Panel C: PC Time Courses
    # =========================================================================
    fig_c = plt.figure(figsize=(14, 3.5))
    gs_c = fig_c.add_gridspec(1, 11, wspace=0.4)
    
    ax_pc1 = fig_c.add_subplot(gs_c[0, :5])
    ax_pc2 = fig_c.add_subplot(gs_c[0, 6:])
    
    _plot_pc_timeseries(ax_pc1, time_courses, 'PC1', 'PC1', show_legend=True,
                        significance=significance, legend_loc='upper right',
                        show_gray_shading=True, show_ylabel=True)
    _plot_pc_timeseries(ax_pc2, time_courses, 'PC2', 'PC2', show_legend=True,
                        significance=significance, legend_loc='lower right',
                        show_gray_shading=True, show_ylabel=True)
    
    plt.tight_layout()
    path_c = PLOTS_DIR / 'panel_C_pc_timeseries.png'
    fig_c.savefig(path_c, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_c)
    saved_paths['C'] = path_c
    print(f"  Saved: {path_c}")
    
    # =========================================================================
    # Panel D: Variance Explained
    # =========================================================================
    fig_d = plt.figure(figsize=(5, 3.5))
    ax_var = fig_d.add_subplot(1, 1, 1)
    _plot_variance_explained(ax_var, variance_explained)
    plt.tight_layout()
    path_d = PLOTS_DIR / 'panel_D_variance.png'
    fig_d.savefig(path_d, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_d)
    saved_paths['D'] = path_d
    print(f"  Saved: {path_d}")
    
    # =========================================================================
    # Panel E: PCA Loadings Heatmap
    # =========================================================================
    fig_e = plt.figure(figsize=(8, 3.5))
    ax_load = fig_e.add_subplot(1, 1, 1)
    _plot_loadings_heatmap_paper_style(ax_load, loadings)
    plt.tight_layout()
    path_e = PLOTS_DIR / 'panel_E_loadings.png'
    fig_e.savefig(path_e, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_e)
    saved_paths['E'] = path_e
    print(f"  Saved: {path_e}")
    
    print(f"  All subplots saved to: {PLOTS_DIR}")
    return saved_paths


def plot_figure3(df, time_courses, loadings, variance_explained, lme_results, 
                 significance=None):
    """
    Generate Figure 3 with 5 panels matching the reference image.
    
    NOTE: This function generates the composite figure directly. For modular
    subplot generation (used by run_figures.py), use save_individual_subplots().
    
    Panel A: Time series - Arousal & Valence (top) + 5 individual dimensions (bottom)
             Gray shading = State effect (DMT vs RS), Black bars = State × Dose interaction
    Panel B: LME coefficients forest plot (horizontal layout)
    Panel C: PC1 and PC2 time courses (full width)
    Panel D: Variance explained bar plot (matching paper style)
    Panel E: PCA loadings heatmap (horizontal orientation, matching paper style)
    """
    print("\nGenerating Figure 3 (composite)...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 16))
    
    # Define grid: 5 rows
    # Row 0: Panel A top (Arousal + Valence) - larger
    # Row 1: Panel A bottom (5 individual dimensions) - FULL WIDTH
    # Row 2: Panel B (LME coefficients) - horizontal
    # Row 3: Panel C (PC1 + PC2 time courses) - full width
    # Row 4: Panel D (variance) + Panel E (loadings heatmap horizontal)
    
    gs = gridspec.GridSpec(5, 11, figure=fig, 
                           height_ratios=[1.2, 0.9, 0.7, 1.0, 0.9],
                           hspace=0.4, wspace=0.4)
    
    # =========================================================================
    # Panel A - Top row: Arousal and Valence time series (small gap in middle)
    # =========================================================================
    ax_arousal = fig.add_subplot(gs[0, :5])
    ax_valence = fig.add_subplot(gs[0, 6:])  # Skip column 5 for small gap (~5mm)
    
    _plot_timeseries_panel(ax_arousal, time_courses, 'emotional_intensity_z', 
                           'Arousal', subtitle='(Emotional Intensity)', show_legend=True,
                           significance=significance, show_gray_shading=True,
                           show_ylabel=True, is_first_col=True, legend_loc='upper right')
    _plot_timeseries_panel(ax_valence, time_courses, 'valence_index',
                           'Valence', subtitle='(Pleasantness−Unpleasantness)', show_legend=True,
                           significance=significance, show_gray_shading=True,
                           show_ylabel=True, is_first_col=False, legend_loc='lower right')
    
    ax_arousal.text(-0.10, 1.12, 'A', transform=ax_arousal.transAxes, 
                    fontsize=18, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel A - Bottom row: Individual dimensions (5 equal-width subplots)
    # Use separate GridSpec for this row to ensure equal widths
    # =========================================================================
    dims = ['interoception_z', 'anxiety_z', 'unpleasantness_z', 'pleasantness_z', 'bliss_z']
    dim_labels = ['Interoception', 'Anxiety', 'Unpleasantness', 'Pleasantness', 'Bliss']
    
    # Get the position of row 1 from main gridspec
    row1_bbox = gs[1, :].get_position(fig)
    
    # Create 5 equal-width axes manually
    n_dims = 5
    total_width = row1_bbox.width
    gap_frac = 0.035  # Slightly larger gap to avoid y-tick overlap
    subplot_width = (total_width - (n_dims - 1) * gap_frac * total_width) / n_dims
    
    for i, (dim, label) in enumerate(zip(dims, dim_labels)):
        left = row1_bbox.x0 + i * (subplot_width + gap_frac * total_width)
        ax = fig.add_axes([left, row1_bbox.y0, subplot_width, row1_bbox.height])
        _plot_timeseries_panel(ax, time_courses, dim, label, 
                               show_legend=False, small=True, significance=significance,
                               show_gray_shading=True, show_ylabel=(i == 0), is_first_col=(i == 0))
    
    # =========================================================================
    # Panel B - LME Coefficients Forest Plot (horizontal)
    # =========================================================================
    ax_lme = fig.add_subplot(gs[2, :])
    _plot_lme_forest_horizontal(ax_lme, lme_results)
    # Note: "B" label is added inside _plot_lme_forest_horizontal using fig.text
    
    # =========================================================================
    # Panel C - PC Time Courses (full width, side by side with small gap)
    # =========================================================================
    ax_pc1 = fig.add_subplot(gs[3, :5])
    ax_pc2 = fig.add_subplot(gs[3, 6:])  # Skip column 5 for small gap (~5mm)
    
    _plot_pc_timeseries(ax_pc1, time_courses, 'PC1', 'PC1', show_legend=True,
                        significance=significance, legend_loc='upper right',
                        show_gray_shading=True, show_ylabel=True)
    _plot_pc_timeseries(ax_pc2, time_courses, 'PC2', 'PC2', show_legend=True,
                        significance=significance, legend_loc='lower right',
                        show_gray_shading=True, show_ylabel=True)
    
    ax_pc1.text(-0.10, 1.12, 'C', transform=ax_pc1.transAxes,
                fontsize=18, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel D - Variance Explained (left side of bottom row)
    # =========================================================================
    ax_var = fig.add_subplot(gs[4, :3])
    _plot_variance_explained(ax_var, variance_explained)
    ax_var.text(-0.17, 1.12, 'D', transform=ax_var.transAxes,
                fontsize=18, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel E - PCA Loadings Heatmap (horizontal, right side - matching paper)
    # =========================================================================
    ax_load = fig.add_subplot(gs[4, 3:])
    _plot_loadings_heatmap_paper_style(ax_load, loadings)
    ax_load.text(-0.07, 1.12, 'E', transform=ax_load.transAxes,
                 fontsize=18, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / 'figure4_tet_analysis.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    
    plt.close(fig)
    return output_path


def _plot_timeseries_panel(ax, time_courses, var_name, title, show_legend=True, 
                          small=False, significance=None, show_gray_shading=False,
                          subtitle=None, show_ylabel=True, is_first_col=True,
                          legend_loc='upper right'):
    """Plot a single time series panel with significance indicators.
    
    Args:
        ax: matplotlib axis
        time_courses: dict with time course data by dose
        var_name: variable name to plot
        title: plot title (bold)
        subtitle: subtitle (not bold, e.g., "(Emotional Intensity)")
        show_legend: whether to show legend
        small: whether this is a small subplot
        significance: dict with significance masks
        show_gray_shading: whether to show gray background for significant State effect
        show_ylabel: whether to show Y axis label
        is_first_col: whether this is the first column (for Y label)
        legend_loc: location of legend (e.g., 'upper right', 'lower right')
    
    Font sizes (homogenized across all panels):
        - Title: 12 (small: 11)
        - Subtitle: 11 (small: 10)
        - Axis labels: 11 (small: 10)
        - Tick labels: 10 (small: 9)
        - Legend: 10 (small: 9)
    """
    
    # Get time values
    tc_high = time_courses['Alta']
    time = tc_high['time_min'].values
    
    # Add vertical dashed line at t=0 (injection time)
    ax.axvline(0, color='#AAAAAA', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Add gray background shading ONLY where State effect is significant (DMT vs RS)
    if show_gray_shading and significance and var_name in significance:
        sig_data = significance[var_name]
        if 'state_sig' in sig_data and 'state_time_bins' in sig_data:
            state_sig = sig_data['state_sig']
            state_time_bins = sig_data['state_time_bins']
            
            # Draw gray shading for significant State effect regions
            sig_regions = _get_contiguous_regions(state_sig, state_time_bins)
            for start_t, end_t in sig_regions:
                ax.axvspan(start_t, end_t, alpha=0.2, color='gray', zorder=0)
    
    # Plot time series for each dose
    for dose, color, label in [('Alta', COLOR_HIGH, 'High dose (40mg)'), 
                                ('Baja', COLOR_LOW, 'Low dose (20mg)')]:
        tc = time_courses[dose]
        
        mean_col = f'{var_name}_mean'
        sem_col = f'{var_name}_sem'
        
        if mean_col in tc.columns:
            mean = tc[mean_col].values
            sem = tc[sem_col].values
            
            ax.fill_between(time, mean - sem, mean + sem, alpha=0.3, color=color, zorder=2)
            ax.plot(time, mean, color=color, linewidth=2, label=label, zorder=3)
    
    # Add significance bars (black horizontal bars for State × Dose interaction)
    if significance and var_name in significance:
        sig_data = significance[var_name]
        dose_sig = sig_data['dose_sig']
        time_bins = sig_data['time_bins']
        
        # Find y position for significance bar (top of plot)
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        bar_y = y_lim[1] - y_range * 0.05
        
        # Draw black bars where significant
        sig_regions = _get_contiguous_regions(dose_sig, time_bins)
        for start_t, end_t in sig_regions:
            ax.hlines(bar_y, start_t, end_t, colors='black', linewidth=3, zorder=4)
    
    # Axis labels - homogenized font sizes
    ax.set_xlabel('Time (minutes)', fontsize=10 if small else 11, fontweight='bold')
    
    # Y label - show when show_ylabel is True
    if show_ylabel:
        ax.set_ylabel('Intensity (Z-scored)', fontsize=10 if small else 11, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    # Title: main title bold, subtitle not bold - homogenized font sizes
    if subtitle:
        ax.set_title('')
        y_title = 1.02
        ax.text(0.5, y_title + 0.08, title, transform=ax.transAxes, 
                fontsize=11 if small else 12, fontweight='bold', ha='center', va='bottom')
        ax.text(0.5, y_title, subtitle, transform=ax.transAxes,
                fontsize=10 if small else 11, fontweight='normal', ha='center', va='bottom')
    else:
        ax.set_title(title, fontsize=11 if small else 12, fontweight='bold')
    
    ax.set_xlim(0, 20)
    ax.axhline(0, color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.5)
    
    if show_legend and not small:
        ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9)
    
    # Style: gray spines like figure 4
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    # Homogenize tick label sizes and colors
    ax.tick_params(axis='both', labelsize=9 if small else 10, colors='#555555')


def _get_contiguous_regions(mask, time_bins):
    """Get contiguous regions where mask is True."""
    regions = []
    in_region = False
    start_t = None
    
    for i, (is_sig, t) in enumerate(zip(mask, time_bins)):
        if is_sig and not in_region:
            start_t = t
            in_region = True
        elif not is_sig and in_region:
            end_t = time_bins[i-1] if i > 0 else t
            regions.append((start_t, end_t))
            in_region = False
    
    # Close last region if still open
    if in_region:
        regions.append((start_t, time_bins[-1]))
    
    return regions


def _plot_pc_timeseries(ax, time_courses, pc_name, title, show_legend=True,
                        significance=None, legend_loc='upper right',
                        show_gray_shading=False, show_ylabel=True):
    """Plot PC score time series with gray shading and significance bars."""
    
    # Add vertical dashed line at t=0 (injection time)
    ax.axvline(0, color='#AAAAAA', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # Add gray background shading ONLY where State effect is significant
    if show_gray_shading and significance and pc_name in significance:
        sig_data = significance[pc_name]
        if 'state_sig' in sig_data and 'state_time_bins' in sig_data:
            state_sig = sig_data['state_sig']
            state_time_bins = sig_data['state_time_bins']
            
            sig_regions = _get_contiguous_regions(state_sig, state_time_bins)
            for start_t, end_t in sig_regions:
                ax.axvspan(start_t, end_t, alpha=0.2, color='gray', zorder=0)
    
    for dose, color, label in [('Alta', COLOR_HIGH, 'High dose (40mg)'), 
                                ('Baja', COLOR_LOW, 'Low dose (20mg)')]:
        tc = time_courses[dose]
        time = tc['time_min'].values
        
        mean = tc[f'{pc_name}_mean'].values
        sem = tc[f'{pc_name}_sem'].values
        
        ax.fill_between(time, mean - sem, mean + sem, alpha=0.3, color=color, zorder=2)
        ax.plot(time, mean, color=color, linewidth=2, label=label, zorder=3)
    
    # Add significance bars (black horizontal bars for State × Dose interaction)
    if significance and pc_name in significance:
        sig_data = significance[pc_name]
        dose_sig = sig_data['dose_sig']
        time_bins = sig_data['time_bins']
        
        # Find y position for significance bar (top of plot)
        y_lim = ax.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        bar_y = y_lim[1] - y_range * 0.05
        
        # Draw black bars where significant
        sig_regions = _get_contiguous_regions(dose_sig, time_bins)
        for start_t, end_t in sig_regions:
            ax.hlines(bar_y, start_t, end_t, colors='black', linewidth=3, zorder=4)
    
    # Homogenized font sizes matching other panels
    ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    if show_ylabel:
        ax.set_ylabel(f'{pc_name} Score', fontsize=11, fontweight='bold')
    else:
        ax.set_ylabel('')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 20)
    ax.axhline(0, color='#AAAAAA', linestyle='-', linewidth=0.5, alpha=0.5)
    
    if show_legend:
        ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9)
    
    # Style: gray spines like figure 4
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    # Homogenize tick label sizes and colors
    ax.tick_params(axis='both', labelsize=10, colors='#555555')


def _plot_lme_forest_horizontal(ax, lme_results, add_label: bool = True):
    """Plot LME coefficients as horizontal forest plot matching paper style.
    
    Style: horizontal layout with Arousal on top, Valence on bottom,
    each effect in its own mini-panel with shared y-axis labels on left.
    Significance markers (*, **, ***) are added based on p-values.
    
    Args:
        ax: matplotlib axis (will be hidden, used for positioning)
        lme_results: dict with LME model results
        add_label: whether to add "B" panel label (default True)
    """
    
    # Effects to plot - using DMT and Alta as treatment levels
    # Reference: State='RS', Dose='Baja'
    effect_names = ['State', 'Dose', 'State:Dose', 'State:Time', 'Dose:Time']
    effect_keys = ['state[T.DMT]', 'dose[T.Alta]', 'state[T.DMT]:dose[T.Alta]', 
                   'state[T.DMT]:time_c', 'dose[T.Alta]:time_c']
    
    arousal_model = lme_results['arousal']['model']
    valence_model = lme_results['valence']['model']
    
    n_effects = len(effect_names)
    
    # Hide main axis but keep it for the "B" label positioning
    ax.set_visible(False)
    fig = ax.figure
    bbox = ax.get_position()
    
    # Add "B" label using figure coordinates (since ax is invisible)
    # Only add if add_label is True (for composite figure, not individual subplots)
    if add_label:
        fig.text(bbox.x0 - 0.035, bbox.y1 + 0.015, 'B', fontsize=18, fontweight='bold', va='bottom')
    
    # Create mini axes for each effect with better spacing
    total_width = bbox.width
    gap = 0.015
    usable_width = total_width - (n_effects - 1) * gap
    width_per_effect = usable_width / n_effects
    
    def _get_significance_marker(p_value: float) -> str:
        """Return significance marker based on p-value."""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        return ''
    
    for idx, (effect_name, effect_key) in enumerate(zip(effect_names, effect_keys)):
        left = bbox.x0 + idx * (width_per_effect + gap)
        mini_ax = fig.add_axes([left, bbox.y0, width_per_effect, bbox.height])
        
        data = []
        
        # Arousal (top, y=1)
        if effect_key in arousal_model.params:
            beta = arousal_model.params[effect_key]
            ci = arousal_model.conf_int().loc[effect_key]
            p_val = arousal_model.pvalues[effect_key]
            sig_marker = _get_significance_marker(p_val)
            data.append(('Arousal', 1, beta, ci[0], ci[1], COLOR_AROUSAL, sig_marker))
        
        # Valence (bottom, y=0)
        if effect_key in valence_model.params:
            beta = valence_model.params[effect_key]
            ci = valence_model.conf_int().loc[effect_key]
            p_val = valence_model.pvalues[effect_key]
            sig_marker = _get_significance_marker(p_val)
            data.append(('Valence', 0, beta, ci[0], ci[1], COLOR_VALENCE, sig_marker))
        
        # Plot points with error bars (horizontal style like paper)
        for name, y, beta, ci_l, ci_h, color, sig_marker in data:
            mini_ax.errorbar(beta, y, xerr=[[beta - ci_l], [ci_h - beta]],
                           fmt='o', color=color, markersize=7, capsize=3,
                           capthick=1.5, elinewidth=1.5, zorder=3)
            # Add significance marker to the right of the CI
            if sig_marker:
                mini_ax.text(ci_h + 0.02, y, sig_marker, fontsize=20, fontweight='bold',
                           va='center', ha='left', color=color)
        
        # Reference line at 0
        mini_ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        
        # Formatting
        mini_ax.set_ylim(-0.5, 1.5)
        mini_ax.set_yticks([0, 1])
        
        # Y-axis labels only on first panel - homogenized font sizes
        if idx == 0:
            mini_ax.set_yticklabels(['Valence', 'Arousal'], fontsize=10)
        else:
            mini_ax.set_yticklabels([])
        
        # X-axis label - homogenized font sizes
        mini_ax.set_xlabel('β coefficient', fontsize=10, fontweight='bold')
        
        # Title in purple color (matching paper) - homogenized font sizes
        mini_ax.set_title(effect_name, fontsize=11, color=COLOR_AROUSAL, fontweight='bold')
        
        # Homogenize tick label sizes
        mini_ax.tick_params(axis='both', labelsize=9)
        
        # Remove top and right spines
        mini_ax.spines['top'].set_visible(False)
        mini_ax.spines['right'].set_visible(False)
        mini_ax.spines['left'].set_linewidth(0.5)
        mini_ax.spines['bottom'].set_linewidth(0.5)
        
        # Auto-scale x-axis with padding (extra margin for significance markers)
        all_vals = [d[2] for d in data] + [d[3] for d in data] + [d[4] for d in data]
        if all_vals:
            range_val = max(all_vals) - min(all_vals)
            margin = max(0.15, range_val * 0.35)
            mini_ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)
        
        # Add light horizontal grid lines
        mini_ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')


def _plot_variance_explained(ax, variance_explained):
    """Plot variance explained bar chart matching paper style.
    
    Font sizes homogenized with other panels:
        - Axis labels: 11
        - Tick labels: 10
        - Annotations: 10
    """
    n_components = 5
    x = np.arange(n_components)
    var_pct = variance_explained[:n_components] * 100
    
    # Colors: PC1 and PC2 in purple, rest in gray (matching paper)
    colors = [COLOR_HIGH, COLOR_HIGH, '#A0A0A0', '#A0A0A0', '#A0A0A0']
    
    bars = ax.bar(x, var_pct, color=colors, edgecolor='none', width=0.7)
    
    # Add percentage labels on top of bars - homogenized font size
    for i, (bar, pct) in enumerate(zip(bars, var_pct)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, 
                color='#555555', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)], fontsize=10)
    ax.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 50)
    
    # Style: gray spines like figure 4
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    # Homogenize tick label sizes and colors
    ax.tick_params(axis='both', labelsize=10, colors='#555555')


def _plot_loadings_heatmap(ax, loadings):
    """Plot PCA loadings heatmap for PC1 and PC2 (vertical orientation).
    
    Font sizes homogenized with other panels:
        - Axis labels: 11
        - Tick labels: 10
        - Cell annotations: 10
    """
    load_matrix = loadings[['PC1', 'PC2']].T.values
    dim_labels = ['Anxiety', 'Bliss', 'Emot.\nIntens.', 
                  'Interocep.', 'Pleasant.', 'Unpleasant.']
    
    im = ax.imshow(load_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    for i in range(2):
        for j in range(6):
            val = load_matrix[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=10, color=color)
    
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(dim_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['PC1', 'PC2'], fontsize=10)
    ax.set_xlabel('Affective Dimension', fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Loading', fontsize=11)
    cbar.ax.tick_params(labelsize=10)


def _plot_loadings_heatmap_horizontal(ax, loadings):
    """Plot PCA loadings heatmap in horizontal orientation (dimensions as rows).
    
    Font sizes homogenized with other panels:
        - Axis labels: 11
        - Tick labels: 10
        - Cell annotations: 10
    """
    # Transpose: dimensions as rows, PCs as columns
    load_matrix = loadings[['PC1', 'PC2']].values  # (6 dims, 2 PCs)
    
    dim_labels = ['Anxiety', 'Bliss', 'Emotional Intensity', 
                  'Interoception', 'Pleasantness', 'Unpleasantness']
    
    im = ax.imshow(load_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add text annotations - homogenized font size
    for i in range(6):  # dimensions
        for j in range(2):  # PCs
            val = load_matrix[i, j]
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=10, color=color, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PC1', 'PC2'], fontsize=10)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(dim_labels, fontsize=10)
    ax.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax.set_ylabel('Affective Dimension', fontsize=11, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Loading', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)


def _plot_loadings_heatmap_paper_style(ax, loadings):
    """Plot PCA loadings heatmap matching paper Figure 3e style.
    
    Paper style: PCs as rows, dimensions as columns, with rotated x-labels.
    Colorbar extends to full height of the heatmap. No cell borders.
    
    Font sizes homogenized with other panels:
        - Axis labels: 11
        - Tick labels: 10
        - Cell annotations: 10
        - Colorbar label: 11
    """
    # PCs as rows, dimensions as columns (matching paper Figure 3e)
    load_matrix = loadings[['PC1', 'PC2']].T.values  # (2 PCs, 6 dims)
    
    # Dimension labels matching paper order
    dim_labels = ['Anxiety', 'Bliss', 'Emotional\nIntensity', 
                  'Interoception', 'Pleasantness', 'Unpleasantness']
    
    # Create custom colormap similar to paper (gray for negative, purple for positive)
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ['#A0A0A0', '#C8C8C8', '#E8E8E8', '#FFFFFF', '#D8D0E8', '#B8A8D8', '#5E4FA2']
    cmap = LinearSegmentedColormap.from_list('paper_cmap', colors_cmap, N=256)
    
    im = ax.imshow(load_matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Add text annotations - homogenized font size
    for i in range(2):  # PCs (rows)
        for j in range(6):  # dimensions (columns)
            val = load_matrix[i, j]
            # Use white text for dark cells, black for light cells
            color = 'white' if abs(val) > 0.45 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=10, color=color, fontweight='bold')
    
    # X-axis: dimensions (rotated labels) - homogenized font size
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(dim_labels, fontsize=10, rotation=45, ha='right')
    
    # Y-axis: PCs - homogenized font size
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['PC1', 'PC2'], fontsize=10, fontweight='bold')
    
    # Labels - homogenized font size
    ax.set_xlabel('Affective Dimension', fontsize=11, fontweight='bold')
    
    # Remove all spines (no border around heatmap)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', length=0)
    
    # Colorbar on the right side - FULL HEIGHT (shrink=1.0)
    cbar = plt.colorbar(im, ax=ax, shrink=1.0, pad=0.02, aspect=10)
    cbar.set_label('Loading', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    # Gray outline for colorbar
    cbar.outline.set_edgecolor('#888888')
    cbar.outline.set_linewidth(0.5)


def save_results(df, loadings, variance_explained, lme_results):
    """Save analysis results to CSV files."""
    print("\nSaving results...")
    
    # Create directories
    (RESULTS_DIR / 'pca').mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'lme').mkdir(parents=True, exist_ok=True)
    
    # Save PCA loadings (long format)
    loadings_long = loadings.reset_index()
    loadings_long.columns = ['dimension'] + list(loadings.columns)
    loadings_melted = loadings_long.melt(id_vars='dimension', 
                                          var_name='component', 
                                          value_name='loading')
    loadings_melted.to_csv(RESULTS_DIR / 'pca' / 'pca_loadings.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'pca' / 'pca_loadings.csv'}")
    
    # Save PCA loadings (wide format for easy viewing)
    loadings_wide = loadings.copy()
    loadings_wide.index.name = 'dimension'
    loadings_wide.to_csv(RESULTS_DIR / 'pca' / 'pca_loadings_wide.csv')
    print(f"  Saved: {RESULTS_DIR / 'pca' / 'pca_loadings_wide.csv'}")
    
    # Save variance explained
    var_df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(variance_explained))],
        'variance_explained': variance_explained,
        'cumulative_variance': np.cumsum(variance_explained)
    })
    var_df.to_csv(RESULTS_DIR / 'pca' / 'pca_variance_explained.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'pca' / 'pca_variance_explained.csv'}")
    
    # Save PCA scores
    score_cols = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec']
    # Add PC columns that exist
    for i in range(1, 7):
        pc_col = f'PC{i}'
        if pc_col in df.columns:
            score_cols.append(pc_col)
    
    df[score_cols].to_csv(RESULTS_DIR / 'pca' / 'pca_scores.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'pca' / 'pca_scores.csv'}")
    
    # Save LME results
    lme_rows = []
    for dim_name, result in lme_results.items():
        model = result['model']
        for effect in model.params.index:
            if effect != 'Group Var':
                ci = model.conf_int().loc[effect]
                p_val = model.pvalues[effect] if effect in model.pvalues else np.nan
                lme_rows.append({
                    'dimension': dim_name,
                    'effect': effect,
                    'beta': model.params[effect],
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'p_value': p_val,
                })
    
    lme_df = pd.DataFrame(lme_rows)
    lme_df.to_csv(RESULTS_DIR / 'lme' / 'lme_results.csv', index=False)
    print(f"  Saved: {RESULTS_DIR / 'lme' / 'lme_results.csv'}")
    
    # Save summary of State × Dose interactions
    interaction_key = 'state[T.DMT]:dose[T.Alta]'
    interaction_rows = []
    for dim_name, result in lme_results.items():
        model = result['model']
        if interaction_key in model.params:
            ci = model.conf_int().loc[interaction_key]
            p_val = model.pvalues[interaction_key]
            interaction_rows.append({
                'dimension': dim_name,
                'beta': model.params[interaction_key],
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'p_value': p_val,
                'significant': p_val < 0.05
            })
    
    if interaction_rows:
        interaction_df = pd.DataFrame(interaction_rows)
        interaction_df.to_csv(RESULTS_DIR / 'lme' / 'lme_state_dose_interaction.csv', index=False)
        print(f"  Saved: {RESULTS_DIR / 'lme' / 'lme_state_dose_interaction.csv'}")


def print_paper_results(variance_explained, lme_results, loadings):
    """Print results formatted for paper text."""
    print("\n" + "="*80)
    print("RESULTS FOR PAPER")
    print("="*80)
    
    # PCA results
    print("\n--- PCA Results ---")
    print(f"PC1 (Arousal): {variance_explained[0]*100:.1f}% variance")
    print(f"PC2 (Valence): {variance_explained[1]*100:.1f}% variance")
    print(f"Cumulative (PC1+PC2): {sum(variance_explained[:2])*100:.1f}% variance")
    
    # Expected from paper: PC1=41.0%, PC2=31.8%, Total=72.8%
    print("\n  Expected (paper): PC1=41.0%, PC2=31.8%, Total=72.8%")
    
    # Print loadings summary
    print("\n--- PCA Loadings ---")
    print("PC1 (Arousal) - should load on Intensity, Anxiety, Interoception, Unpleasantness:")
    for dim in AFFECTIVE_COLS_Z:
        loading = loadings.loc[dim, 'PC1']
        label = AFFECTIVE_LABELS[dim]
        print(f"  {label}: {loading:.2f}")
    
    print("\nPC2 (Valence) - bipolar: Pleasantness/Bliss (+) vs Unpleasantness/Anxiety (-):")
    for dim in AFFECTIVE_COLS_Z:
        loading = loadings.loc[dim, 'PC2']
        label = AFFECTIVE_LABELS[dim]
        print(f"  {label}: {loading:.2f}")
    
    # LME results - State × Dose interaction
    print("\n--- LME: State × Dose Interaction ---")
    print("(Effect of High dose during DMT vs additive effects)")
    
    interaction_key = 'state[T.DMT]:dose[T.Alta]'
    
    for name in ['arousal', 'valence']:
        if name in lme_results:
            model = lme_results[name]['model']
            if interaction_key in model.params:
                beta = model.params[interaction_key]
                ci = model.conf_int().loc[interaction_key]
                p = model.pvalues[interaction_key]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {name.capitalize()}: β = {beta:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], p = {p:.2e} {sig}")
    
    # Expected from paper
    print("\n  Expected (paper):")
    print("    Arousal: β = 0.47, 95% CI [0.32, 0.61], p < .001")
    print("    Valence: β = −0.50, 95% CI [−0.70, −0.30], p < .001")
    
    # Individual dimensions
    print("\n--- Individual Dimensions (State × Dose) ---")
    for dim in AFFECTIVE_COLS:
        if dim in lme_results:
            model = lme_results[dim]['model']
            if interaction_key in model.params:
                beta = model.params[interaction_key]
                ci = model.conf_int().loc[interaction_key]
                p = model.pvalues[interaction_key]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {dim}: β = {beta:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}], p = {p:.2e} {sig}")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("TET ANALYSIS PIPELINE")
    print("="*80)
    
    # Check data exists
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Run preprocessing first or check data path.")
        return 1
    
    # Load data
    df = load_data()
    
    # Compute PCA
    df, loadings, variance_explained = compute_pca(df)
    
    # Fit LME models
    lme_results, df_all = fit_lme_models(df)
    
    # Compute time courses
    time_courses = compute_time_courses(df)
    
    # Compute significance masks for plotting
    significance = compute_significance_masks(df)
    
    # Save individual subplots (for run_figures.py assembly)
    subplot_paths = save_individual_subplots(df, time_courses, loadings, 
                                              variance_explained, lme_results, 
                                              significance)
    
    # Generate composite Figure 3 (legacy, for direct viewing)
    fig_path = plot_figure3(df, time_courses, loadings, variance_explained, 
                            lme_results, significance)
    
    # Save results
    save_results(df, loadings, variance_explained, lme_results)
    
    # Print paper results
    print_paper_results(variance_explained, lme_results, loadings)
    
    print("\n✓ TET analysis complete!")
    print(f"  Composite figure: {fig_path}")
    print(f"  Individual subplots: {len(subplot_paths)} panels saved")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
