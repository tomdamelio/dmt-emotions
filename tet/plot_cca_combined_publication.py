#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create publication-quality combined CCA figure for Nature Human Behaviour.

Figure 4: Canonical Correlation Analysis of Physiological-Affective Coupling.
  Panel A: Physiological loadings for DMT CV1
  Panel B: Affective dimension loadings for DMT CV1
  Panel C: Cross-validation performance (boxplots)
  Panel D: In-sample canonical scores scatterplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def create_combined_cca_figure(
    loadings_df: pd.DataFrame,
    cv_folds_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    merged_data_df: pd.DataFrame,
    output_path: str,
    state: str = 'DMT',
    canonical_variate: int = 1
):
    """
    Create Nature Human Behaviour style combined CCA figure.
    
    Args:
        loadings_df: DataFrame with canonical loadings
        cv_folds_df: DataFrame with cross-validation fold results
        cv_summary_df: DataFrame with cross-validation summary statistics
        merged_data_df: DataFrame with merged physiological-TET data
        output_path: Path to save figure
        state: State to plot ('DMT' or 'RS')
        canonical_variate: Which canonical variate to plot
    """
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
    
    # Color scheme using tab20c palette
    tab20c_colors = plt.cm.tab20c.colors
    
    # Physiological colors
    physio_colors = {
        'HR': tab20c_colors[0],        # ECG: Red
        'SMNA_AUC': tab20c_colors[4],  # EDA: Blue
        'RVT': tab20c_colors[8]        # Respiration: Green
    }
    
    # Violet colors for affective dimensions
    gray_colors = [tab20c_colors[12], tab20c_colors[13], tab20c_colors[14], tab20c_colors[15]]
    
    # Colors for RS/DMT
    color_rs = tab20c_colors[17]
    color_rs_edge = tab20c_colors[16]
    color_dmt = tab20c_colors[13]
    color_dmt_edge = tab20c_colors[12]
    
    # Create figure: 2x2 grid
    fig_width_mm = 183
    fig_width_inch = fig_width_mm / 25.4
    fig_height_inch = fig_width_inch * 0.6
    
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    gs = fig.add_gridspec(2, 3, 
                          width_ratios=[1, 1.5, 1.5],
                          height_ratios=[1, 1],
                          wspace=0.35,
                          hspace=0.4)
    
    ax_A = fig.add_subplot(gs[0, 0])  # Physio loadings
    ax_B = fig.add_subplot(gs[0, 1])  # TET loadings
    ax_C = fig.add_subplot(gs[0, 2])  # CV boxplots
    ax_D = fig.add_subplot(gs[1, :])  # Scatterplot (full width)
    
    # ========== PANEL A: Physiological Loadings ==========
    data = loadings_df[
        (loadings_df['state'] == state) & 
        (loadings_df['canonical_variate'] == canonical_variate)
    ].copy()
    
    physio_data = data[data['variable_set'] == 'physio'].copy()
    physio_data = physio_data.sort_values('loading', ascending=True)
    
    physio_labels = {
        'HR': ('Electrocardiography', 'HR (Z-scored)'),
        'SMNA_AUC': ('Electrodermal Activity', 'SMNA (Z-scored)'),
        'RVT': ('Respiration', 'RVT (Z-scored)')
    }
    
    y_pos_physio = np.arange(len(physio_data))
    
    for i, (idx, row) in enumerate(physio_data.iterrows()):
        var_name = row['variable_name']
        loading = row['loading']
        color = physio_colors.get(var_name, '#666666')
        ax_A.barh(i, loading, height=0.6, color=color, edgecolor='none', alpha=0.85)
    
    ax_A.set_yticks(y_pos_physio)
    ax_A.set_yticklabels([])
    
    for i, var_name in enumerate(physio_data['variable_name']):
        if var_name in physio_labels:
            modality, metric = physio_labels[var_name]
            color = physio_colors[var_name]
            ax_A.text(-0.02, i + 0.15, modality, transform=ax_A.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7, fontweight='bold', color=color)
            ax_A.text(-0.02, i - 0.15, metric, transform=ax_A.get_yaxis_transform(),
                     ha='right', va='center', fontsize=7, fontweight='normal', color='black')
    
    ax_A.set_xlabel('Canonical Loading', fontweight='normal')
    ax_A.set_title('Physiological Variables', fontweight='bold', pad=8)
    ax_A.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax_A.set_xlim(-0.1, 1.0)
    ax_A.spines['top'].set_visible(False)
    ax_A.spines['right'].set_visible(False)
    ax_A.grid(axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
    ax_A.set_axisbelow(True)
    
    # ========== PANEL B: TET Loadings ==========
    tet_data = data[data['variable_set'] == 'tet'].copy()
    tet_data = tet_data.sort_values('loading', ascending=True)
    
    tet_labels = {
        'emotional_intensity': 'Emotional Intensity',
        'interoception': 'Interoception',
        'unpleasantness': 'Unpleasantness',
        'pleasantness': 'Pleasantness',
        'bliss': 'Bliss',
        'anxiety': 'Anxiety'
    }
    
    y_pos_tet = np.arange(len(tet_data))
    
    for i, (idx, row) in enumerate(tet_data.iterrows()):
        loading = row['loading']
        color = gray_colors[i % len(gray_colors)]
        ax_B.barh(i, loading, height=0.6, color=color, edgecolor='none', alpha=0.85)
    
    ax_B.set_yticks(y_pos_tet)
    ax_B.set_yticklabels([tet_labels.get(v, v) for v in tet_data['variable_name']])
    ax_B.set_xlabel('Canonical Loading', fontweight='normal')
    ax_B.set_title('Affective Dimensions', fontweight='bold', pad=8)
    ax_B.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax_B.set_xlim(-0.1, 1.0)
    ax_B.spines['top'].set_visible(False)
    ax_B.spines['right'].set_visible(False)
    ax_B.grid(axis='x', alpha=0.2, linewidth=0.5, linestyle='--')
    ax_B.set_axisbelow(True)
    
    # ========== PANEL C: Cross-validation boxplots ==========
    cv_cv1 = cv_folds_df[cv_folds_df['canonical_variate'] == 1].copy()
    rs_data = cv_cv1[cv_cv1['state'] == 'RS']['r_oos'].values
    dmt_data_cv = cv_cv1[cv_cv1['state'] == 'DMT']['r_oos'].values
    
    positions = [0.8, 1.6]
    box_width = 0.15
    
    def draw_half_boxplot(ax, data, pos, color_fill, color_edge):
        q1, median, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        whisker_low = np.min(data[data >= q1 - 1.5 * iqr])
        whisker_high = np.max(data[data <= q3 + 1.5 * iqr])
        
        box = plt.Rectangle((pos - box_width, q1), box_width, q3 - q1,
                            facecolor=color_fill, edgecolor=color_edge,
                            linewidth=0.5, alpha=0.7, zorder=2)
        ax.add_patch(box)
        ax.hlines(median, pos - box_width, pos, colors=color_edge, linewidth=1.5, zorder=3)
        ax.vlines(pos - box_width/2, whisker_low, q1, colors=color_edge, linewidth=0.5, zorder=2)
        ax.vlines(pos - box_width/2, q3, whisker_high, colors=color_edge, linewidth=0.5, zorder=2)
        ax.hlines(whisker_low, pos - box_width*0.75, pos - box_width*0.25, 
                 colors=color_edge, linewidth=0.5, zorder=2)
        ax.hlines(whisker_high, pos - box_width*0.75, pos - box_width*0.25, 
                 colors=color_edge, linewidth=0.5, zorder=2)
    
    draw_half_boxplot(ax_C, rs_data, positions[0], color_rs, color_rs_edge)
    draw_half_boxplot(ax_C, dmt_data_cv, positions[1], color_dmt, color_dmt_edge)
    
    np.random.seed(42)
    jitter_strength = 0.08
    
    x_rs = np.ones(len(rs_data)) * positions[0] + np.abs(np.random.normal(0, jitter_strength, len(rs_data)))
    ax_C.scatter(x_rs, rs_data, s=20, color=color_rs, alpha=0.7, 
                edgecolors=color_rs_edge, linewidths=0.5, zorder=3)
    
    x_dmt = np.ones(len(dmt_data_cv)) * positions[1] + np.abs(np.random.normal(0, jitter_strength, len(dmt_data_cv)))
    ax_C.scatter(x_dmt, dmt_data_cv, s=20, color=color_dmt, alpha=0.7, 
                edgecolors=color_dmt_edge, linewidths=0.5, zorder=3)
    
    ax_C.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_C.set_xticks(positions)
    ax_C.set_xticklabels(['RS', 'DMT'])
    ax_C.set_ylabel('Out-of-sample correlation (r$_{oos}$)', fontweight='normal')
    ax_C.set_title('Cross-validation (CV1)', fontweight='bold', pad=8)
    ax_C.spines['top'].set_visible(False)
    ax_C.spines['right'].set_visible(False)
    ax_C.set_ylim(-0.6, 1.0)
    ax_C.set_xlim(0.3, 2.1)
    ax_C.grid(axis='y', alpha=0.2, linewidth=0.5, linestyle='--')
    ax_C.set_axisbelow(True)
    ax_C.tick_params(axis='x', pad=5)
    
    rs_mean = cv_summary_df[(cv_summary_df['state'] == 'RS') & 
                            (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    dmt_mean = cv_summary_df[(cv_summary_df['state'] == 'DMT') & 
                             (cv_summary_df['canonical_variate'] == 1)]['mean_r_oos'].values[0]
    
    rs_max = rs_data.max()
    dmt_max = dmt_data_cv.max()
    
    ax_C.text(positions[0], rs_max + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {rs_mean:.2f}', 
             ha='center', va='bottom', fontsize=6, style='italic')
    ax_C.text(positions[1], dmt_max + 0.08, f'$\\mathit{{r}}_{{oos}}$ = {dmt_mean:.2f}**', 
             ha='center', va='bottom', fontsize=6, style='italic', fontweight='bold')
    ax_C.text(0.98, 0.02, '**p < 0.01', transform=ax_C.transAxes,
             ha='right', va='bottom', fontsize=5, style='italic')
    
    # ========== PANEL D: In-sample scatterplot ==========
    from sklearn.cross_decomposition import CCA
    
    dmt_merged = merged_data_df[merged_data_df['state'] == 'DMT'].copy()
    physio_measures = ['HR', 'SMNA_AUC', 'RVT']
    tet_affective = ['pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
                     'interoception_z', 'bliss_z', 'anxiety_z']
    
    X = dmt_merged[physio_measures].values
    Y = dmt_merged[tet_affective].values
    subjects = dmt_merged['subject'].values
    
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    
    cca = CCA(n_components=2)
    cca.fit(X, Y)
    U, V = cca.transform(X, Y)
    
    x_scores = U[:, 0]
    y_scores = V[:, 0]
    
    in_sample_r = cv_summary_df[
        (cv_summary_df['state'] == 'DMT') & 
        (cv_summary_df['canonical_variate'] == 1)
    ]['in_sample_r'].values[0]
    
    unique_subjects = np.unique(subjects)
    tab20_colors = plt.cm.tab20.colors
    subject_colors = {}
    subject_edge_colors = {}
    
    for i, subj in enumerate(unique_subjects):
        idx = (i % 10) * 2
        subject_colors[subj] = tab20_colors[idx + 1]
        subject_edge_colors[subj] = tab20_colors[idx]
    
    for subj in unique_subjects:
        mask = subjects == subj
        ax_D.scatter(x_scores[mask], y_scores[mask], s=30, color=subject_colors[subj],
                    alpha=0.7, edgecolors=subject_edge_colors[subj], linewidths=0.5, label=subj)
    
    z = np.polyfit(x_scores, y_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_scores.min(), x_scores.max(), 100)
    ax_D.plot(x_line, p(x_line), color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Regression')
    
    ax_D.text(0.98, 0.95, f'$\\mathit{{r}}$ = {in_sample_r:.2f}', transform=ax_D.transAxes,
             ha='right', va='top', fontsize=7, style='italic', fontweight='bold')
    
    ax_D.set_xlabel('Physiological Canonical Score (U1)', fontweight='normal')
    ax_D.set_ylabel('TET Affective Canonical Score (V1)', fontweight='normal')
    ax_D.set_title('In-sample coupling (DMT CV1)', fontweight='bold', pad=8)
    ax_D.spines['top'].set_visible(False)
    ax_D.spines['right'].set_visible(False)
    ax_D.grid(alpha=0.2, linewidth=0.5, linestyle='--')
    ax_D.set_axisbelow(True)
    
    ax_D.legend(loc='lower right', fontsize=5, frameon=True, framealpha=0.95,
               edgecolor='gray', ncol=2, borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
    
    # Panel labels
    for ax, label, x_off in [(ax_A, 'A', -0.35), (ax_B, 'B', -0.15), (ax_C, 'C', -0.25), (ax_D, 'D', -0.08)]:
        ax.text(x_off, 1.15, label, transform=ax.transAxes, ha='left', va='top',
               fontsize=12, fontweight='bold', color='black')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    
    plt.savefig(output_path.with_suffix('.png'), dpi=600, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(output_path.with_suffix('.tiff'), dpi=600, bbox_inches='tight',
               facecolor='white', edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
    
    plt.close()
    
    print(f"✓ Saved combined CCA figure:")
    print(f"  - PNG (600 DPI): {output_path.with_suffix('.png')}")
    print(f"  - PDF (vector): {output_path.with_suffix('.pdf')}")
    print(f"  - TIFF (600 DPI): {output_path.with_suffix('.tiff')}")


def main():
    """Main execution."""
    print("=" * 80)
    print("Creating Combined CCA Figure (Figure 4)")
    print("Style: Nature Human Behaviour")
    print("=" * 80)
    
    base_path = Path('results/tet/physio_correlation')
    
    print("\nLoading data...")
    loadings_df = pd.read_csv(base_path / 'cca_loadings.csv')
    cv_folds_df = pd.read_csv(base_path / 'cca_cross_validation_folds.csv')
    cv_summary_df = pd.read_csv(base_path / 'cca_cross_validation_summary.csv')
    merged_data_df = pd.read_csv(base_path / 'merged_physio_tet_data.csv')
    
    print(f"  - Loadings: {len(loadings_df)} entries")
    print(f"  - CV folds: {len(cv_folds_df)} observations")
    print(f"  - CV summary: {len(cv_summary_df)} conditions")
    print(f"  - Merged data: {len(merged_data_df)} observations")
    
    output_dir = Path('results/tet/physio_correlation/figures_publication')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating Figure 4: Combined CCA Analysis...")
    output_path = output_dir / 'Figure4_CCA_Combined'
    
    create_combined_cca_figure(
        loadings_df=loadings_df,
        cv_folds_df=cv_folds_df,
        cv_summary_df=cv_summary_df,
        merged_data_df=merged_data_df,
        output_path=str(output_path),
        state='DMT',
        canonical_variate=1
    )
    
    print("\n" + "=" * 80)
    print("Figure 4 creation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
