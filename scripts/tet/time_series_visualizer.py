# -*- coding: utf-8 -*-
"""
TET Time Series Visualization Module

This module provides functionality for generating time series plots with
statistical annotations showing dose effects over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import logging
import sys
import os
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
logger = logging.getLogger(__name__)


class TETTimeSeriesVisualizer:
    """
    Generates time series plots with statistical annotations.
    
    This class creates publication-ready figures showing:
    - Mean trajectories with SEM shading for each dose
    - RS baseline as first time point
    - Grey background for significant DMT vs RS effects
    - Black bars for significant State:Dose interactions
    - Dimensions ordered by effect strength
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data
        lme_results (pd.DataFrame): LME model results
        lme_contrasts (pd.DataFrame): LME dose contrasts
        time_courses (pd.DataFrame): Time course data with mean ± SEM
        dimensions (List[str]): List of dimensions to plot
    
    Example:
        >>> from tet.time_series_visualizer import TETTimeSeriesVisualizer
        >>> import pandas as pd
        >>> 
        >>> # Load data
        >>> data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
        >>> lme_results = pd.read_csv('results/tet/lme/lme_results.csv')
        >>> lme_contrasts = pd.read_csv('results/tet/lme/lme_contrasts.csv')
        >>> time_courses = pd.read_csv('results/tet/descriptive/time_course_all_dimensions.csv')
        >>> 
        >>> # Create visualizer
        >>> viz = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)
        >>> 
        >>> # Generate and export figure
        >>> fig = viz.generate_figure()
        >>> viz.export_figure('results/tet/figures/time_series_with_annotations.png')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        lme_results: pd.DataFrame,
        lme_contrasts: pd.DataFrame,
        time_courses: pd.DataFrame,
        dimensions: Optional[List[str]] = None
    ):
        """
        Initialize time series visualizer.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data
            lme_results (pd.DataFrame): LME model results
            lme_contrasts (pd.DataFrame): LME dose contrasts
            time_courses (pd.DataFrame): Time course data
            dimensions (Optional[List[str]]): List of dimensions to plot.
                If None, uses affective dimensions + valence_index_z.
        """
        self.data = data
        self.lme_results = lme_results
        self.lme_contrasts = lme_contrasts
        self.time_courses = time_courses
        
        # Default: z-scored affective dimensions + valence index
        if dimensions is None:
            affective_dims = [f"{dim}_z" for dim in config.TET_AFFECTIVE_COLUMNS]
            self.dimensions = affective_dims + ['valence_index_z']
        else:
            self.dimensions = dimensions
        
        # Compute RS baselines
        self.rs_baselines = self._compute_rs_baselines()
        
        # Order dimensions by effect strength
        self.ordered_dimensions = self._order_dimensions_by_effect()
        
        # Identify significant time points
        self.significance_annotations = self._identify_significant_timepoints()
        
        # Identify time bins with significant dose differences
        self.dose_interaction_bins = self._identify_dose_interactions()
        
        logger.info(f"Initialized TETTimeSeriesVisualizer with {len(self.dimensions)} dimensions")
    
    def _compute_rs_baselines(self) -> pd.DataFrame:
        """
        Compute RS baseline (mean across entire RS condition).
        
        Returns:
            pd.DataFrame: RS baselines with columns: dimension, dose, baseline_mean
        """
        logger.info("Computing RS baselines...")
        
        rs_data = self.data[self.data['state'] == 'RS']
        
        baselines = []
        for dimension in self.dimensions:
            for dose in ['Baja', 'Alta']:
                dose_data = rs_data[rs_data['dose'] == dose]
                baseline_mean = dose_data[dimension].mean()
                
                baselines.append({
                    'dimension': dimension,
                    'dose': dose,
                    'baseline_mean': baseline_mean
                })
        
        baselines_df = pd.DataFrame(baselines)
        logger.info(f"Computed {len(baselines_df)} RS baselines")
        
        return baselines_df
    
    def _order_dimensions_by_effect(self) -> List[str]:
        """
        Order dimensions by strength of State effect from LME results.
        
        Returns:
            List[str]: Ordered list of dimension names
        """
        logger.info("Ordering dimensions by State effect strength...")
        
        # Extract State effect coefficients
        state_effects = self.lme_results[
            self.lme_results['effect'] == 'state[T.DMT]'
        ][['dimension', 'beta']].copy()
        
        # Sort by absolute value of beta (descending)
        state_effects['abs_beta'] = state_effects['beta'].abs()
        state_effects = state_effects.sort_values('abs_beta', ascending=False)
        
        ordered = state_effects['dimension'].tolist()
        
        logger.info(f"Ordered {len(ordered)} dimensions")
        logger.info(f"  Strongest: {ordered[0]} (β={state_effects.iloc[0]['beta']:.3f})")
        logger.info(f"  Weakest: {ordered[-1]} (β={state_effects.iloc[-1]['beta']:.3f})")
        
        return ordered
    
    def _identify_significant_timepoints(self) -> pd.DataFrame:
        """
        Identify time points with significant effects.
        
        Returns:
            pd.DataFrame: Significance annotations with columns:
                - dimension, t_bin, main_effect_sig, interaction_sig
        """
        logger.info("Identifying significant time points...")
        
        annotations = []
        
        for dimension in self.dimensions:
            # Get interaction significance from LME results
            interaction_row = self.lme_results[
                (self.lme_results['dimension'] == dimension) &
                (self.lme_results['effect'] == 'state[T.DMT]:dose[T.Alta]')
            ]
            
            interaction_sig = False
            if len(interaction_row) > 0:
                interaction_sig = interaction_row.iloc[0]['significant']
            
            # Test main effect at each time bin
            # Compare DMT values at each time point vs RS baseline (mean across entire RS)
            dmt_data = self.data[self.data['state'] == 'DMT']
            
            # Get RS baseline for this dimension (mean across entire RS condition)
            rs_baseline = self.data[self.data['state'] == 'RS'][dimension].mean()
            
            for t_bin in dmt_data['t_bin'].unique():
                bin_data = dmt_data[dmt_data['t_bin'] == t_bin]
                
                # One-sample t-test: DMT values at this time bin vs RS baseline
                # H0: mean(DMT_at_t) = RS_baseline
                try:
                    t_stat, p_value = stats.ttest_1samp(bin_data[dimension], rs_baseline)
                    main_effect_sig = p_value < 0.05
                except:
                    main_effect_sig = False
                
                annotations.append({
                    'dimension': dimension,
                    't_bin': t_bin,
                    'main_effect_sig': main_effect_sig,
                    'interaction_sig': interaction_sig
                })
        
        annotations_df = pd.DataFrame(annotations)
        
        n_main = annotations_df['main_effect_sig'].sum()
        n_interaction = annotations_df['interaction_sig'].sum()
        
        logger.info(f"Identified {n_main} time points with main effects")
        logger.info(f"Identified {n_interaction} time points with interactions")
        
        return annotations_df
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Hochberg FDR correction to p-values.
        
        Args:
            p_values (List[float]): List of raw p-values
            
        Returns:
            List[float]: FDR-corrected p-values
        """
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
    
    def _identify_dose_interactions(self) -> pd.DataFrame:
        """
        Identify time bins with significant dose differences (High vs Low) during DMT.
        Uses Benjamini-Hochberg FDR correction across all time bins per dimension.
        
        Returns:
            pd.DataFrame: Dose interaction annotations with columns:
                - dimension, t_bin, p_raw, p_fdr, dose_effect_sig
        """
        logger.info("Identifying time bins with dose differences (with FDR correction)...")
        
        annotations = []
        
        # Filter DMT data only
        dmt_data = self.data[self.data['state'] == 'DMT']
        
        for dimension in self.dimensions:
            # Collect all p-values for this dimension across time bins
            time_bins = sorted(dmt_data['t_bin'].unique())
            p_values = []
            t_stats = []
            
            for t_bin in time_bins:
                # Get data for this time bin
                bin_data = dmt_data[dmt_data['t_bin'] == t_bin]
                
                # Separate by dose
                low_dose = bin_data[bin_data['dose'] == 'Baja'][dimension].values
                high_dose = bin_data[bin_data['dose'] == 'Alta'][dimension].values
                
                # Independent samples t-test: High vs Low dose at this time bin
                # H0: mean(High) = mean(Low)
                try:
                    t_stat, p_value = stats.ttest_ind(high_dose, low_dose)
                    p_values.append(p_value)
                    t_stats.append(t_stat)
                except:
                    p_values.append(1.0)  # Non-significant if test fails
                    t_stats.append(0.0)
            
            # Apply BH-FDR correction across all time bins for this dimension
            p_fdr = self._benjamini_hochberg_correction(p_values)
            
            # Store results
            for t_bin, p_raw, p_adj, t_stat in zip(time_bins, p_values, p_fdr, t_stats):
                annotations.append({
                    'dimension': dimension,
                    't_bin': t_bin,
                    'p_raw': p_raw,
                    'p_fdr': p_adj,
                    't_stat': t_stat,
                    'dose_effect_sig': p_adj < 0.05
                })
        
        annotations_df = pd.DataFrame(annotations)
        
        n_sig_raw = (annotations_df['p_raw'] < 0.05).sum()
        n_sig_fdr = annotations_df['dose_effect_sig'].sum()
        logger.info(f"Found {n_sig_raw} time bins with p<0.05 (uncorrected)")
        logger.info(f"Found {n_sig_fdr} time bins with p_FDR<0.05 (BH-corrected)")
        
        return annotations_df
    
    def _plot_dimension(
        self,
        ax: plt.Axes,
        dimension: str,
        show_ylabel: bool = True
    ):
        """
        Plot a single dimension panel.
        
        Args:
            ax (plt.Axes): Matplotlib axes to plot on
            dimension (str): Dimension name
            show_ylabel (bool): Whether to show y-axis label (default: True)
        """
        # Get time course data for this dimension
        tc_dim = self.time_courses[
            (self.time_courses['dimension'] == dimension) &
            (self.time_courses['state'] == 'DMT')
        ]
        
        # Separate by dose
        tc_low = tc_dim[tc_dim['dose'] == 'Baja'].sort_values('t_bin')
        tc_high = tc_dim[tc_dim['dose'] == 'Alta'].sort_values('t_bin')
        
        # Convert time to minutes
        time_low = tc_low['t_sec'].values / 60
        time_high = tc_high['t_sec'].values / 60
        
        # Plot DMT time series with SEM shading (without RS baseline points)
        # Plot High dose FIRST so it appears first in legend (inverted order)
        ax.plot(time_high, tc_high['mean'], color=COLOR_HIGH_DOSE, linewidth=3, 
                label='High dose (40mg)', marker='o', markersize=3, zorder=2)
        ax.fill_between(
            time_high,
            tc_high['mean'] - tc_high['sem'],
            tc_high['mean'] + tc_high['sem'],
            color=COLOR_HIGH_DOSE,
            alpha=0.25,
            zorder=1
        )
        
        # Plot Low dose SECOND so it appears second in legend
        ax.plot(time_low, tc_low['mean'], color=COLOR_LOW_DOSE, linewidth=3, 
                label='Low dose (20mg)', marker='o', markersize=3, zorder=2)
        ax.fill_between(
            time_low,
            tc_low['mean'] - tc_low['sem'],
            tc_low['mean'] + tc_low['sem'],
            color=COLOR_LOW_DOSE,
            alpha=0.25,
            zorder=1
        )
        
        # Add grey background shading for significant main effects (DMT vs RS)
        # Group consecutive bins to create continuous shaded regions
        sig_bins = self.significance_annotations[
            (self.significance_annotations['dimension'] == dimension) &
            (self.significance_annotations['main_effect_sig'] == True)
        ]['t_bin'].values
        
        if len(sig_bins) > 0:
            # Group consecutive bins
            main_effect_groups = []
            current_group = [sig_bins[0]]
            
            for i in range(1, len(sig_bins)):
                if sig_bins[i] == current_group[-1] + 1:
                    current_group.append(sig_bins[i])
                else:
                    main_effect_groups.append(current_group)
                    current_group = [sig_bins[i]]
            main_effect_groups.append(current_group)
            
            # Draw grey shaded regions for main effects (DMT vs RS)
            for group in main_effect_groups:
                t_start = group[0] * 4 / 60  # Convert to minutes
                t_end = (group[-1] + 1) * 4 / 60
                # Use grey shading for main effects
                ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=0)
        
        # Add black horizontal bars at top for significant dose differences (High vs Low, FDR-corrected)
        # Similar to ECG analysis style
        dose_sig_bins = self.dose_interaction_bins[
            (self.dose_interaction_bins['dimension'] == dimension) &
            (self.dose_interaction_bins['dose_effect_sig'] == True)
        ]['t_bin'].values
        
        if len(dose_sig_bins) > 0:
            # Group consecutive bins to create continuous bars
            dose_effect_groups = []
            current_group = [dose_sig_bins[0]]
            
            for i in range(1, len(dose_sig_bins)):
                if dose_sig_bins[i] == current_group[-1] + 1:
                    current_group.append(dose_sig_bins[i])
                else:
                    dose_effect_groups.append(current_group)
                    current_group = [dose_sig_bins[i]]
            dose_effect_groups.append(current_group)
            
            # Draw black horizontal bars at top for dose effects (FDR-corrected)
            y_max = 3  # Fixed y-axis max
            y_bar = y_max * 0.95  # Position at 95% of y-axis height
            
            for group in dose_effect_groups:
                t_start = group[0] * 4 / 60  # Convert to minutes
                t_end = (group[-1] + 1) * 4 / 60
                ax.plot([t_start, t_end], [y_bar, y_bar], 
                       color='black', linewidth=4, solid_capstyle='butt', zorder=3)
        
        # Add dashed line at DMT onset
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=2, alpha=0.6, zorder=0)
        
        # Formatting
        dim_name = dimension.replace('_z', '').replace('_', ' ').title()
        ax.set_title(dim_name, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontweight='bold')
        
        # Only show ylabel for first two dimensions (Arousal and Interoception)
        if show_ylabel:
            ax.set_ylabel('Intensity (Z-scored)', fontweight='bold')
        else:
            ax.set_ylabel('')
        
        ax.set_xlim(-1, 20)
        ax.set_ylim(-3, 3)  # Fixed y-axis range for all subplots
        ax.grid(True, which='major', axis='y', alpha=0.25, linestyle='-', linewidth=0.5)
        ax.grid(False, which='major', axis='x')
        ax.set_axisbelow(True)
    
    def generate_figure(self) -> plt.Figure:
        """
        Generate complete multi-panel figure with custom layout.
        
        Layout:
        - Row 1: 2 large panels (Arousal, Valence) - each spanning 5 columns
        - Row 2: 5 small panels (Interoception, Anxiety, Unpleasantness, Pleasantness, Bliss) - each spanning 2 columns
        
        Returns:
            plt.Figure: Matplotlib figure object
        """
        logger.info("Generating time series figure with custom layout...")
        
        # Define dimensions to plot in specific order
        # Row 1: Main dimensions (large panels)
        main_dimensions = ['emotional_intensity_z', 'valence_index_z']
        
        # Row 2: Secondary dimensions (small panels)
        secondary_dimensions = [
            'interoception_z',
            'anxiety_z', 
            'unpleasantness_z',
            'pleasantness_z',
            'bliss_z'
        ]
        
        # Create figure with GridSpec for custom layout
        # Increased hspace to separate rows more
        # Width=18 for consistency with pca_timeseries.png and lme_coefficients_forest.png
        STANDARD_WIDTH = 18
        fig = plt.figure(figsize=(STANDARD_WIDTH, 8), dpi=300)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.50, wspace=0.4)
        
        # Row 1: Large panels (Arousal and Valence)
        # Arousal: columns 0-4 (show ylabel)
        ax_arousal = fig.add_subplot(gs[0, 0:5])
        if 'emotional_intensity_z' in self.dimensions:
            self._plot_dimension(ax_arousal, 'emotional_intensity_z', show_ylabel=True)
            # Two-line title: bold main term, regular subtitle
            ax_arousal.set_title('Arousal\n(Emotional Intensity)', 
                                fontsize=16, fontweight='bold', 
                                linespacing=1.2)
            # Make subtitle (second line) not bold by using text annotation
            # Get title position and replace with custom formatting
            title_obj = ax_arousal.title
            title_obj.set_text('')  # Clear default title
            ax_arousal.text(0.5, 1.08, 'Arousal', 
                          transform=ax_arousal.transAxes,
                          fontsize=18, fontweight='bold', 
                          ha='center', va='bottom')
            ax_arousal.text(0.5, 1.02, '(Emotional Intensity)', 
                          transform=ax_arousal.transAxes,
                          fontsize=14, fontweight='normal', 
                          ha='center', va='bottom')
            # Larger legend with bigger font and frame (like pca_timeseries)
            legend_arousal = ax_arousal.legend(loc='upper right', fontsize=14, frameon=True, 
                                              fancybox=True, framealpha=0.9, 
                                              markerscale=1.5, handlelength=2.5)
            legend_arousal.get_frame().set_facecolor('white')
            legend_arousal.get_frame().set_alpha(0.9)
        
        # Valence: columns 5-9 (no ylabel)
        ax_valence = fig.add_subplot(gs[0, 5:10])
        if 'valence_index_z' in self.dimensions:
            self._plot_dimension(ax_valence, 'valence_index_z', show_ylabel=False)
            # Clear the default title generated by _plot_dimension
            ax_valence.set_title('')
            # Two-line title: bold main term, regular subtitle
            # Note: Title says "Valence" not "Valence Index"
            ax_valence.text(0.5, 1.08, 'Valence', 
                          transform=ax_valence.transAxes,
                          fontsize=18, fontweight='bold', 
                          ha='center', va='bottom')
            ax_valence.text(0.5, 1.02, '(Pleasantness-Unpleasantness)', 
                          transform=ax_valence.transAxes,
                          fontsize=14, fontweight='normal', 
                          ha='center', va='bottom')
            # Add legend to Valence panel in lower right position with frame (like pca_timeseries)
            legend_valence = ax_valence.legend(loc='lower right', fontsize=14, frameon=True, 
                                              fancybox=True, framealpha=0.9, 
                                              markerscale=1.5, handlelength=2.5)
            legend_valence.get_frame().set_facecolor('white')
            legend_valence.get_frame().set_alpha(0.9)
        
        # Row 2: Small panels (5 dimensions, 2 columns each)
        secondary_positions = [
            (1, 0, 2),   # Interoception: columns 0-1 (show ylabel)
            (1, 2, 4),   # Anxiety: columns 2-3 (no ylabel)
            (1, 4, 6),   # Unpleasantness: columns 4-5 (no ylabel)
            (1, 6, 8),   # Pleasantness: columns 6-7 (no ylabel)
            (1, 8, 10),  # Bliss: columns 8-9 (no ylabel)
        ]
        
        for idx, (dimension, (row, col_start, col_end)) in enumerate(zip(secondary_dimensions, secondary_positions)):
            if dimension in self.dimensions:
                ax = fig.add_subplot(gs[row, col_start:col_end])
                # Only show ylabel for first dimension (Interoception, idx=0)
                show_ylabel = (idx == 0)
                self._plot_dimension(ax, dimension, show_ylabel=show_ylabel)
                
                # Format title - larger and bold for secondary panels
                dim_name = dimension.replace('_z', '').replace('_', ' ').title()
                ax.set_title(dim_name, fontsize=14, fontweight='bold')
        
        logger.info("Figure generation complete")
        
        return fig
    
    def export_fdr_report(self, output_path: str) -> str:
        """
        Export FDR analysis report showing significant dose difference segments.
        
        Args:
            output_path (str): Output file path for report
            
        Returns:
            str: Path to exported report
        """
        lines = [
            'FDR COMPARISON: High (40mg) vs Low (20mg) Dose Effects Over Time',
            'Benjamini-Hochberg FDR correction applied per dimension across all time bins',
            'Alpha = 0.05',
            '',
        ]
        
        for dimension in self.ordered_dimensions:
            # Get significant bins for this dimension
            dose_sig = self.dose_interaction_bins[
                (self.dose_interaction_bins['dimension'] == dimension) &
                (self.dose_interaction_bins['dose_effect_sig'] == True)
            ].sort_values('t_bin')
            
            dim_name = dimension.replace('_z', '').replace('_', ' ').title()
            lines.append(f'DIMENSION: {dim_name}')
            lines.append('-' * 60)
            
            if len(dose_sig) == 0:
                lines.append('  No significant dose differences (p_FDR < 0.05)')
            else:
                # Group consecutive bins
                sig_bins = dose_sig['t_bin'].values
                bin_groups = []
                current_group = [sig_bins[0]]
                
                for i in range(1, len(sig_bins)):
                    if sig_bins[i] == current_group[-1] + 1:
                        current_group.append(sig_bins[i])
                    else:
                        bin_groups.append(current_group)
                        current_group = [sig_bins[i]]
                bin_groups.append(current_group)
                
                lines.append(f'  Significant segments (count={len(bin_groups)}):')
                for group in bin_groups:
                    t_start_sec = group[0] * 4
                    t_end_sec = (group[-1] + 1) * 4
                    t_start_min = t_start_sec / 60
                    t_end_min = t_end_sec / 60
                    
                    # Get min p_FDR in this segment
                    segment_data = dose_sig[dose_sig['t_bin'].isin(group)]
                    min_p_fdr = segment_data['p_fdr'].min()
                    
                    lines.append(
                        f'    - Bins {group[0]}-{group[-1]}: '
                        f'{t_start_min:.2f}-{t_end_min:.2f} min '
                        f'({t_start_sec}-{t_end_sec}s), '
                        f'min p_FDR={min_p_fdr:.4f}'
                    )
                
                # Summary statistics
                lines.append(f'  Total significant bins: {len(dose_sig)}')
                lines.append(f'  Min p_FDR: {dose_sig["p_fdr"].min():.6f}')
                lines.append(f'  Median p_FDR: {dose_sig["p_fdr"].median():.6f}')
            
            lines.append('')
        
        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported FDR report to: {output_path}")
        
        return output_path
    
    def export_figure(
        self,
        output_path: str,
        dpi: int = 300,
        export_fdr_report: bool = True
    ) -> str:
        """
        Generate and export figure to file.
        
        Args:
            output_path (str): Output file path
            dpi (int): Resolution in dots per inch (default: 300)
            export_fdr_report (bool): Whether to export FDR report (default: True)
            
        Returns:
            str: Path to exported file
        """
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate figure
        fig = self.generate_figure()
        
        # Save figure
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"Exported figure to: {output_path}")
        
        # Export FDR report if requested
        if export_fdr_report:
            report_path = output_path.replace('.png', '_fdr_report.txt')
            self.export_fdr_report(report_path)
        
        return output_path
