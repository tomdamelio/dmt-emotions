"""
Visualization module for physiological-TET integration analysis.

This module provides the TETPhysioVisualizer class for generating publication-ready
figures showing relationships between physiological signals (HR, SMNA AUC, RVT) and
affective TET dimensions.

Classes:
    TETPhysioVisualizer: Generate correlation heatmaps, regression scatter plots,
                         and CCA loading visualizations.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch


class TETPhysioVisualizer:
    """
    Visualizer for physiological-TET integration analysis results.
    
    This class generates publication-ready figures showing:
    - Correlation heatmaps between TET affective dimensions and physiological measures
    - Regression scatter plots (TET vs physiological PC1)
    - CCA canonical loading biplots and bar charts
    
    Attributes:
        figure_paths (List[str]): List of generated figure file paths
        logger (logging.Logger): Logger instance for status messages
    
    Example:
        >>> visualizer = TETPhysioVisualizer()
        >>> visualizer.plot_correlation_heatmaps(correlation_df, 'results/tet/physio_correlation')
        >>> visualizer.plot_regression_scatter(merged_data, pc1_scores, regression_df, 'results/tet/physio_correlation')
        >>> visualizer.plot_cca_loadings(loadings_df, correlations_df, 'results/tet/physio_correlation')
        >>> figure_paths = visualizer.export_figures('results/tet/physio_correlation')
    """
    
    def __init__(self):
        """Initialize the physiological-TET visualizer."""
        self.figure_paths = []
        self.logger = logging.getLogger(__name__)
        
        # Set publication-quality defaults
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 13
    
    def plot_correlation_heatmaps(
        self,
        correlation_df: pd.DataFrame,
        output_dir: str
    ) -> List[str]:
        """
        Generate correlation heatmaps for TET-physio relationships.
        
        Creates separate heatmaps for RS and DMT states showing Pearson correlations
        between TET affective dimensions and physiological measures. Cells are annotated
        with correlation values and significance markers.
        
        Args:
            correlation_df: DataFrame with columns:
                - tet_dimension: TET dimension name
                - physio_measure: Physiological measure name (HR, SMNA_AUC, RVT)
                - state: RS or DMT
                - r: Pearson correlation coefficient
                - p_fdr: FDR-corrected p-value
            output_dir: Directory to save figures
        
        Returns:
            List of generated figure paths
        
        Figure specifications:
            - 2 panels side-by-side (RS | DMT)
            - Rows: TET affective dimensions (6)
            - Columns: Physiological measures (3)
            - Cell values: Pearson r
            - Cell annotations: r value + significance markers
              (* p_fdr < 0.05, ** p_fdr < 0.01, *** p_fdr < 0.001)
            - Color scale: Blue-white-red diverging (-1 to +1)
            - Figure size: 10×6 inches, 300 DPI
        """
        output_path = Path(output_dir) / 'figures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get unique states
        states = sorted(correlation_df['state'].unique())
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, len(states), figsize=(10, 6))
        if len(states) == 1:
            axes = [axes]
        
        for idx, state in enumerate(states):
            # Filter data for this state
            state_data = correlation_df[correlation_df['state'] == state].copy()
            
            # Pivot to create heatmap matrix
            heatmap_data = state_data.pivot(
                index='tet_dimension',
                columns='physio_measure',
                values='r'
            )
            
            # Create annotation matrix with r values and significance markers
            annot_matrix = []
            for tet_dim in heatmap_data.index:
                row = []
                for physio_measure in heatmap_data.columns:
                    r_val = heatmap_data.loc[tet_dim, physio_measure]
                    
                    # Get p_fdr value
                    p_fdr = state_data[
                        (state_data['tet_dimension'] == tet_dim) &
                        (state_data['physio_measure'] == physio_measure)
                    ]['p_fdr'].values[0]
                    
                    # Add significance markers
                    if p_fdr < 0.001:
                        sig_marker = '***'
                    elif p_fdr < 0.01:
                        sig_marker = '**'
                    elif p_fdr < 0.05:
                        sig_marker = '*'
                    else:
                        sig_marker = ''
                    
                    row.append(f'{r_val:.2f}{sig_marker}')
                annot_matrix.append(row)
            
            # Order TET dimensions by average absolute correlation (strongest first)
            avg_abs_corr = heatmap_data.abs().mean(axis=1).sort_values(ascending=False)
            heatmap_data = heatmap_data.loc[avg_abs_corr.index]
            annot_matrix = [annot_matrix[list(heatmap_data.index).index(dim)] 
                           for dim in heatmap_data.index]
            
            # Create heatmap
            sns.heatmap(
                heatmap_data,
                annot=np.array(annot_matrix),
                fmt='',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={'label': 'Pearson r'},
                ax=axes[idx],
                linewidths=0.5,
                linecolor='gray'
            )
            
            axes[idx].set_title(f'{state} State', fontweight='bold')
            axes[idx].set_xlabel('Physiological Measure')
            axes[idx].set_ylabel('TET Affective Dimension' if idx == 0 else '')
            
            # Rotate labels for better readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        # Save figure
        figure_path = output_path / 'correlation_heatmaps.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figure_paths.append(str(figure_path))
        self.logger.info(f"Generated correlation heatmap: {figure_path}")
        
        return [str(figure_path)]
    
    def plot_regression_scatter(
        self,
        merged_data: pd.DataFrame,
        pc1_scores: pd.DataFrame,
        regression_df: pd.DataFrame,
        output_dir: str,
        pc2_scores: Optional[pd.DataFrame] = None
    ) -> List[str]:
        """
        Generate scatter plots for TET vs physiological PC1 and PC2.
        
        Creates figures showing relationships between TET dimensions
        (arousal and valence) and the principal components of physiological signals.
        
        Args:
            merged_data: Merged physio-TET dataset with columns:
                - subject, session_id, state, dose, t_bin
                - emotional_intensity_z, valence_index_z
            pc1_scores: DataFrame with physiological PC1 scores
            regression_df: DataFrame with regression results:
                - outcome_variable: TET dimension name
                - state: RS or DMT
                - beta: Standardized regression coefficient
                - r_squared: R² value
                - p_value: Significance of beta
            output_dir: Directory to save figures
            pc2_scores: Optional DataFrame with physiological PC2 scores
        
        Returns:
            List of generated figure paths
        
        Figure 1: Arousal vs Physio PC1
            - 2 panels (RS | DMT)
            - X-axis: Physiological PC1
            - Y-axis: Emotional intensity (z-scored)
            - Points: Semi-transparent (alpha=0.3)
            - Regression line with 95% CI band
            - Annotations: β, R², p-value
        
        Figure 2: Valence vs Physio PC1
            - Same layout as Figure 1
            - Y-axis: Valence index (z-scored)
        
        Figure 3: Arousal vs Physio PC2 (if pc2_scores provided)
            - Same layout as Figure 1
            - X-axis: Physiological PC2
        
        Figure 4: Valence vs Physio PC2 (if pc2_scores provided)
            - Same layout as Figure 2
            - X-axis: Physiological PC2
        
        Figure specifications:
            - Figure size: 12×5 inches (2 panels)
            - Point limit: 5000 (random sample if N > 5000)
            - 300 DPI
        """
        output_path = Path(output_dir) / 'figures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_figures = []
        
        # Merge PC1 scores with merged data
        plot_data = merged_data.merge(
            pc1_scores,
            on=['subject', 'session_id', 't_bin'],
            how='inner'
        )
        
        # Merge PC2 scores if provided
        if pc2_scores is not None:
            plot_data = plot_data.merge(
                pc2_scores,
                on=['subject', 'session_id', 't_bin'],
                how='inner'
            )
        
        # Define outcome variables to plot
        outcomes = [
            ('emotional_intensity_z', 'Arousal (Emotional Intensity)'),
            ('valence_index_z', 'Valence Index')
        ]
        
        # Define PC components to plot
        pc_components = [('physio_PC1', 'PC1')]
        if pc2_scores is not None:
            pc_components.append(('physio_PC2', 'PC2'))
        
        for outcome_var, outcome_label in outcomes:
            for pc_var, pc_label in pc_components:
                # Create figure with subplots for each state
                states = sorted(plot_data['state'].unique())
                fig, axes = plt.subplots(1, len(states), figsize=(12, 5))
                if len(states) == 1:
                    axes = [axes]
                
                for idx, state in enumerate(states):
                    # Filter data for this state
                    state_data = plot_data[plot_data['state'] == state].copy()
                    
                    # Limit points if too many (for clarity)
                    if len(state_data) > 5000:
                        state_data = state_data.sample(n=5000, random_state=42)
                    
                    # Get regression statistics (only available for PC1)
                    if pc_var == 'physio_PC1':
                        reg_stats = regression_df[
                            (regression_df['outcome_variable'] == outcome_var) &
                            (regression_df['state'] == state)
                        ]
                        
                        if len(reg_stats) > 0:
                            beta = reg_stats['beta'].values[0]
                            r_squared = reg_stats['r_squared'].values[0]
                            p_value = reg_stats['p_value'].values[0]
                        else:
                            beta = np.nan
                            r_squared = np.nan
                            p_value = np.nan
                    else:
                        # For PC2, compute correlation on the fly
                        from scipy.stats import pearsonr
                        valid_data = state_data[[pc_var, outcome_var]].dropna()
                        if len(valid_data) > 0:
                            r, p_value = pearsonr(valid_data[pc_var], valid_data[outcome_var])
                            beta = r  # Use correlation as proxy for beta
                            r_squared = r ** 2
                        else:
                            beta = np.nan
                            r_squared = np.nan
                            p_value = np.nan
                    
                    # Create scatter plot with regression line
                    sns.regplot(
                        data=state_data,
                        x=pc_var,
                        y=outcome_var,
                        ax=axes[idx],
                        scatter_kws={'alpha': 0.3, 'color': 'gray', 's': 20},
                        line_kws={'color': 'red', 'linewidth': 2},
                        ci=95
                    )
                    
                    # Add annotations
                    if not np.isnan(beta):
                        p_str = f'{p_value:.3f}' if p_value >= 0.001 else '<0.001'
                        if pc_var == 'physio_PC1':
                            annot_text = f'β = {beta:.2f}\nR² = {r_squared:.3f}\np = {p_str}'
                        else:
                            annot_text = f'r = {beta:.2f}\nR² = {r_squared:.3f}\np = {p_str}'
                        
                        # Position annotation in upper left or upper right depending on correlation
                        x_pos = 0.05 if beta > 0 else 0.95
                        axes[idx].text(
                            x_pos, 0.95,
                            annot_text,
                            transform=axes[idx].transAxes,
                            verticalalignment='top',
                            horizontalalignment='left' if beta > 0 else 'right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=9
                        )
                    
                    axes[idx].set_title(f'{state} State', fontweight='bold')
                    axes[idx].set_xlabel(f'Physiological {pc_label}')
                    axes[idx].set_ylabel(f'{outcome_label} (z-scored)' if idx == 0 else '')
                    axes[idx].grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                
                # Save figure
                figure_name = outcome_var.replace('_z', '').replace('_', '_')
                pc_suffix = pc_label.lower()
                figure_path = output_path / f'{figure_name}_vs_{pc_suffix}_scatter.png'
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.figure_paths.append(str(figure_path))
                generated_figures.append(str(figure_path))
                self.logger.info(f"Generated regression scatter plot: {figure_path}")
        
        return generated_figures
    
    def plot_pc1_composite_figure(
        self,
        merged_data: pd.DataFrame,
        pc1_scores: pd.DataFrame,
        regression_df: pd.DataFrame,
        output_dir: str
    ) -> str:
        """
        Generate composite 4-panel figure with PC1 scatter plots.
        
        Creates a 2x2 grid showing:
        - Panel A (top-left): Emotional Intensity vs PC1 - RS
        - Panel B (top-right): Emotional Intensity vs PC1 - DMT
        - Panel C (bottom-left): Valence Index vs PC1 - RS
        - Panel D (bottom-right): Valence Index vs PC1 - DMT
        
        Args:
            merged_data: Merged physio-TET dataset
            pc1_scores: DataFrame with physiological PC1 scores
            regression_df: DataFrame with regression results
            output_dir: Directory to save figure
        
        Returns:
            Path to generated composite figure
        """
        output_path = Path(output_dir) / 'figures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Merge PC1 scores with merged data
        plot_data = merged_data.merge(
            pc1_scores,
            on=['subject', 'session_id', 't_bin'],
            how='inner'
        )
        
        # Define outcome variables
        outcomes = [
            ('emotional_intensity_z', 'Emotional Intensity (z)'),
            ('valence_index_z', 'Valence Index (z)')
        ]
        
        # Create 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        panel_labels = ['A', 'B', 'C', 'D']
        states = ['RS', 'DMT']
        
        panel_idx = 0
        for row_idx, (outcome_var, outcome_label) in enumerate(outcomes):
            for col_idx, state in enumerate(states):
                ax = axes[row_idx, col_idx]
                
                # Filter data for this state
                state_data = plot_data[plot_data['state'] == state].copy()
                
                # Limit points if too many
                if len(state_data) > 5000:
                    state_data = state_data.sample(n=5000, random_state=42)
                
                # Get regression statistics
                reg_stats = regression_df[
                    (regression_df['outcome_variable'] == outcome_var) &
                    (regression_df['state'] == state)
                ]
                
                if len(reg_stats) > 0:
                    beta = reg_stats['beta'].values[0]
                    r_squared = reg_stats['r_squared'].values[0]
                    p_value = reg_stats['p_value'].values[0]
                else:
                    beta = np.nan
                    r_squared = np.nan
                    p_value = np.nan
                
                # Create scatter plot with regression line
                sns.regplot(
                    data=state_data,
                    x='physio_PC1',
                    y=outcome_var,
                    ax=ax,
                    scatter_kws={'alpha': 0.3, 'color': 'gray', 's': 20},
                    line_kws={'color': 'red', 'linewidth': 2},
                    ci=95
                )
                
                # Add annotations
                if not np.isnan(beta):
                    p_str = f'{p_value:.3f}' if p_value >= 0.001 else '<0.001'
                    annot_text = f'β = {beta:.2f}\nR² = {r_squared:.3f}\np = {p_str}'
                    
                    # Position annotation
                    x_pos = 0.05 if beta > 0 else 0.95
                    ax.text(
                        x_pos, 0.95,
                        annot_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left' if beta > 0 else 'right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10
                    )
                
                # Add panel label
                ax.text(
                    -0.1, 1.05,
                    panel_labels[panel_idx],
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight='bold',
                    va='top'
                )
                
                # Set labels and title
                ax.set_title(f'{state} State', fontweight='bold', fontsize=13)
                ax.set_xlabel('Physiological PC1 (Arousal Index)', fontsize=11)
                ax.set_ylabel(outcome_label if col_idx == 0 else '', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                panel_idx += 1
        
        plt.tight_layout()
        
        # Save figure
        figure_path = output_path / 'pc1_composite_4panel.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.figure_paths.append(str(figure_path))
        self.logger.info(f"Generated PC1 composite figure: {figure_path}")
        
        return str(figure_path)
    
    def plot_cca_loadings(
        self,
        canonical_loadings_df: pd.DataFrame,
        canonical_correlations_df: pd.DataFrame,
        output_dir: str
    ) -> List[str]:
        """
        Generate CCA canonical loading visualizations.
        
        Creates biplots and bar charts showing how original variables contribute
        to canonical variates from Canonical Correlation Analysis.
        
        Args:
            canonical_loadings_df: DataFrame with columns:
                - state: RS or DMT
                - canonical_variate: 1 or 2
                - variable_set: 'physio' or 'tet'
                - variable_name: Original variable name
                - loading: Canonical loading (correlation)
            canonical_correlations_df: DataFrame with columns:
                - state: RS or DMT
                - canonical_variate: 1 or 2
                - canonical_correlation: r_i
                - p_value: Significance
            output_dir: Directory to save figures
        
        Returns:
            List of generated figure paths
        
        Creates:
            1. Biplot for each state (RS, DMT):
               - X-axis: Canonical Variate 1
               - Y-axis: Canonical Variate 2
               - Arrows: Canonical loadings for each variable
               - Arrow color: Blue (physio), Red (TET)
               - Labels: Variable names at arrow tips
            
            2. Bar chart for each canonical variate:
               - Separate bars for physio (blue) and TET (red)
               - Horizontal line at ±0.3 (meaningful threshold)
        
        Figure specifications:
            - Biplot: 10×8 inches
            - Bar charts: 12×5 inches
            - 300 DPI
            - Include canonical correlation r_i in title
            - Mark significant variates (p < 0.05) with asterisk
        """
        output_path = Path(output_dir) / 'figures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_figures = []
        states = sorted(canonical_loadings_df['state'].unique())
        
        # 1. Generate biplots for each state
        for state in states:
            state_loadings = canonical_loadings_df[
                canonical_loadings_df['state'] == state
            ].copy()
            
            # Pivot to get CV1 and CV2 loadings
            cv1_loadings = state_loadings[
                state_loadings['canonical_variate'] == 1
            ].set_index('variable_name')
            cv2_loadings = state_loadings[
                state_loadings['canonical_variate'] == 2
            ].set_index('variable_name')
            
            # Create biplot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot arrows for each variable
            for var_name in cv1_loadings.index:
                cv1_load = cv1_loadings.loc[var_name, 'loading']
                cv2_load = cv2_loadings.loc[var_name, 'loading']
                var_set = cv1_loadings.loc[var_name, 'variable_set']
                
                # Color by variable set
                color = 'blue' if var_set == 'physio' else 'red'
                
                # Draw arrow
                arrow = FancyArrowPatch(
                    (0, 0), (cv1_load, cv2_load),
                    arrowstyle='->', mutation_scale=20,
                    color=color, linewidth=2, alpha=0.7
                )
                ax.add_patch(arrow)
                
                # Add label at arrow tip
                ax.text(
                    cv1_load * 1.1, cv2_load * 1.1,
                    var_name.replace('_z', '').replace('_', ' '),
                    fontsize=9,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
            
            # Get canonical correlations for title
            state_corrs = canonical_correlations_df[
                canonical_correlations_df['state'] == state
            ]
            cv1_corr = state_corrs[state_corrs['canonical_variate'] == 1]['canonical_correlation'].values[0]
            cv2_corr = state_corrs[state_corrs['canonical_variate'] == 2]['canonical_correlation'].values[0]
            cv1_pval = state_corrs[state_corrs['canonical_variate'] == 1]['p_value'].values[0]
            cv2_pval = state_corrs[state_corrs['canonical_variate'] == 2]['p_value'].values[0]
            
            cv1_sig = '*' if cv1_pval < 0.05 else ''
            cv2_sig = '*' if cv2_pval < 0.05 else ''
            
            ax.set_xlabel(f'Canonical Variate 1 (r = {cv1_corr:.2f}{cv1_sig})', fontweight='bold')
            ax.set_ylabel(f'Canonical Variate 2 (r = {cv2_corr:.2f}{cv2_sig})', fontweight='bold')
            ax.set_title(f'CCA Loading Biplot - {state} State', fontweight='bold', fontsize=13)
            
            # Set axis limits
            max_load = max(
                abs(cv1_loadings['loading'].max()),
                abs(cv1_loadings['loading'].min()),
                abs(cv2_loadings['loading'].max()),
                abs(cv2_loadings['loading'].min())
            )
            ax.set_xlim(-max_load * 1.2, max_load * 1.2)
            ax.set_ylim(-max_load * 1.2, max_load * 1.2)
            
            # Add grid and reference lines
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='blue', linewidth=2, label='Physiological'),
                Line2D([0], [0], color='red', linewidth=2, label='TET Affective')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save biplot
            figure_path = output_path / f'cca_biplot_{state.lower()}.png'
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.figure_paths.append(str(figure_path))
            generated_figures.append(str(figure_path))
            self.logger.info(f"Generated CCA biplot: {figure_path}")
        
        # 2. Generate bar charts for each canonical variate
        for cv_num in [1, 2]:
            fig, axes = plt.subplots(1, len(states), figsize=(12, 5))
            if len(states) == 1:
                axes = [axes]
            
            for idx, state in enumerate(states):
                cv_loadings = canonical_loadings_df[
                    (canonical_loadings_df['state'] == state) &
                    (canonical_loadings_df['canonical_variate'] == cv_num)
                ].copy()
                
                # Separate physio and TET
                physio_loadings = cv_loadings[cv_loadings['variable_set'] == 'physio']
                tet_loadings = cv_loadings[cv_loadings['variable_set'] == 'tet']
                
                # Create bar positions
                n_physio = len(physio_loadings)
                n_tet = len(tet_loadings)
                x_physio = np.arange(n_physio)
                x_tet = np.arange(n_physio + 1, n_physio + 1 + n_tet)
                
                # Plot bars
                axes[idx].bar(
                    x_physio,
                    physio_loadings['loading'],
                    color='blue',
                    alpha=0.7,
                    label='Physiological'
                )
                axes[idx].bar(
                    x_tet,
                    tet_loadings['loading'],
                    color='red',
                    alpha=0.7,
                    label='TET Affective'
                )
                
                # Add threshold lines
                axes[idx].axhline(0.3, color='gray', linewidth=1, linestyle='--', alpha=0.5)
                axes[idx].axhline(-0.3, color='gray', linewidth=1, linestyle='--', alpha=0.5)
                axes[idx].axhline(0, color='black', linewidth=0.5)
                
                # Set labels
                all_vars = list(physio_loadings['variable_name']) + [''] + list(tet_loadings['variable_name'])
                all_x = list(x_physio) + [n_physio] + list(x_tet)
                axes[idx].set_xticks(all_x)
                axes[idx].set_xticklabels(
                    [v.replace('_z', '').replace('_', ' ') for v in all_vars],
                    rotation=45,
                    ha='right'
                )
                
                # Get canonical correlation for title
                state_corrs = canonical_correlations_df[
                    (canonical_correlations_df['state'] == state) &
                    (canonical_correlations_df['canonical_variate'] == cv_num)
                ]
                cv_corr = state_corrs['canonical_correlation'].values[0]
                cv_pval = state_corrs['p_value'].values[0]
                cv_sig = '*' if cv_pval < 0.05 else ''
                
                axes[idx].set_title(
                    f'{state} State\n(r = {cv_corr:.2f}{cv_sig})',
                    fontweight='bold'
                )
                axes[idx].set_ylabel('Canonical Loading' if idx == 0 else '')
                axes[idx].grid(True, alpha=0.3, linestyle='--', axis='y')
                
                if idx == 0:
                    axes[idx].legend(loc='upper left')
            
            plt.suptitle(f'Canonical Variate {cv_num} Loadings', fontweight='bold', fontsize=13)
            plt.tight_layout()
            
            # Save bar chart
            figure_path = output_path / f'cca_loadings_cv{cv_num}.png'
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.figure_paths.append(str(figure_path))
            generated_figures.append(str(figure_path))
            self.logger.info(f"Generated CCA loading bar chart: {figure_path}")
        
        return generated_figures
    
    def export_figures(self, output_dir: str) -> List[str]:
        """
        Export all generated figures to output directory.
        
        This method provides a summary of all figures generated during the visualization
        process. Individual plotting methods save figures automatically, so this method
        primarily serves as a reporting function.
        
        Args:
            output_dir: Directory where figures are saved
        
        Returns:
            List of all generated figure paths
        
        Note:
            This method is called automatically by individual plotting methods.
            It ensures all figures are saved with consistent naming and format.
            All figures are saved as PNG with 300 DPI for publication quality.
        """
        output_path = Path(output_dir) / 'figures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        if len(self.figure_paths) == 0:
            self.logger.warning("No figures have been generated yet.")
        else:
            self.logger.info(f"Total figures generated: {len(self.figure_paths)}")
            self.logger.info(f"Figures saved to: {output_path}")
            
            # Log each figure
            for fig_path in self.figure_paths:
                self.logger.info(f"  - {Path(fig_path).name}")
        
        return self.figure_paths
