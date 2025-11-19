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
    
    def _identify_dose_interactions(self) -> pd.DataFrame:
        """
        Identify time bins with significant dose differences (High vs Low) during DMT.
        
        Returns:
            pd.DataFrame: Dose interaction annotations with columns:
                - dimension, t_bin, dose_effect_sig
        """
        logger.info("Identifying time bins with dose differences...")
        
        annotations = []
        
        # Filter DMT data only
        dmt_data = self.data[self.data['state'] == 'DMT']
        
        for dimension in self.dimensions:
            for t_bin in sorted(dmt_data['t_bin'].unique()):
                # Get data for this time bin
                bin_data = dmt_data[dmt_data['t_bin'] == t_bin]
                
                # Separate by dose
                low_dose = bin_data[bin_data['dose'] == 'Baja'][dimension].values
                high_dose = bin_data[bin_data['dose'] == 'Alta'][dimension].values
                
                # Independent samples t-test: High vs Low dose at this time bin
                # H0: mean(High) = mean(Low)
                try:
                    t_stat, p_value = stats.ttest_ind(high_dose, low_dose)
                    dose_effect_sig = p_value < 0.05
                except:
                    dose_effect_sig = False
                
                annotations.append({
                    'dimension': dimension,
                    't_bin': t_bin,
                    'dose_effect_sig': dose_effect_sig
                })
        
        annotations_df = pd.DataFrame(annotations)
        
        n_sig = annotations_df['dose_effect_sig'].sum()
        logger.info(f"Found {n_sig} time bins with significant dose differences (p<0.05)")
        
        return annotations_df
    
    def _plot_dimension(
        self,
        ax: plt.Axes,
        dimension: str
    ):
        """
        Plot a single dimension panel.
        
        Args:
            ax (plt.Axes): Matplotlib axes to plot on
            dimension (str): Dimension name
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
        ax.plot(time_low, tc_low['mean'], color='#4472C4', linewidth=2, label='20mg', zorder=2)
        ax.fill_between(
            time_low,
            tc_low['mean'] - tc_low['sem'],
            tc_low['mean'] + tc_low['sem'],
            color='#4472C4',
            alpha=0.3,
            zorder=1
        )
        
        ax.plot(time_high, tc_high['mean'], color='#C44444', linewidth=2, label='40mg', zorder=2)
        ax.fill_between(
            time_high,
            tc_high['mean'] - tc_high['sem'],
            tc_high['mean'] + tc_high['sem'],
            color='#C44444',
            alpha=0.3,
            zorder=1
        )
        
        # Add grey background for significant main effects
        sig_bins = self.significance_annotations[
            (self.significance_annotations['dimension'] == dimension) &
            (self.significance_annotations['main_effect_sig'] == True)
        ]['t_bin'].values
        
        for t_bin in sig_bins:
            t_start = t_bin * 4 / 60  # Convert to minutes
            t_end = (t_bin + 1) * 4 / 60
            ax.axvspan(t_start, t_end, color='grey', alpha=0.2, zorder=0)
        
        # Add black bars at top for significant dose differences at specific time bins
        dose_sig_bins = self.dose_interaction_bins[
            (self.dose_interaction_bins['dimension'] == dimension) &
            (self.dose_interaction_bins['dose_effect_sig'] == True)
        ]['t_bin'].values
        
        if len(dose_sig_bins) > 0:
            y_max = 3  # Fixed y-axis max
            y_bar = y_max * 0.95
            
            # Group consecutive bins to create continuous bars
            bin_groups = []
            current_group = [dose_sig_bins[0]]
            
            for i in range(1, len(dose_sig_bins)):
                if dose_sig_bins[i] == current_group[-1] + 1:
                    current_group.append(dose_sig_bins[i])
                else:
                    bin_groups.append(current_group)
                    current_group = [dose_sig_bins[i]]
            bin_groups.append(current_group)
            
            # Draw bars for each group
            for group in bin_groups:
                t_start = group[0] * 4 / 60  # Convert to minutes
                t_end = (group[-1] + 1) * 4 / 60
                ax.plot([t_start, t_end], [y_bar, y_bar], 
                       color='black', linewidth=3, solid_capstyle='butt', zorder=3)
        
        # Add dashed line at DMT onset
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
        
        # Formatting
        dim_name = dimension.replace('_z', '').replace('_', ' ').title()
        ax.set_title(dim_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (minutes)', fontsize=8)
        ax.set_ylabel('Z-scored Intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-1, 20)
        ax.set_ylim(-3, 3)  # Fixed y-axis range for all subplots
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
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
        fig = plt.figure(figsize=(20, 8), dpi=300)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.35, wspace=0.4)
        
        fig.suptitle('TET Time Series: Dose Effects with Statistical Annotations', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Row 1: Large panels (Arousal and Valence)
        # Arousal: columns 0-4
        ax_arousal = fig.add_subplot(gs[0, 0:5])
        if 'emotional_intensity_z' in self.dimensions:
            self._plot_dimension(ax_arousal, 'emotional_intensity_z')
            ax_arousal.set_title('Arousal (Emotional Intensity)', fontsize=14, fontweight='bold')
            ax_arousal.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Valence: columns 5-9
        ax_valence = fig.add_subplot(gs[0, 5:10])
        if 'valence_index_z' in self.dimensions:
            self._plot_dimension(ax_valence, 'valence_index_z')
            ax_valence.set_title('Valence (Pleasantness-Unpleasantness)', fontsize=14, fontweight='bold')
        
        # Row 2: Small panels (5 dimensions, 2 columns each)
        secondary_positions = [
            (1, 0, 2),   # Interoception: columns 0-1
            (1, 2, 4),   # Anxiety: columns 2-3
            (1, 4, 6),   # Unpleasantness: columns 4-5
            (1, 6, 8),   # Pleasantness: columns 6-7
            (1, 8, 10),  # Bliss: columns 8-9
        ]
        
        for idx, (dimension, (row, col_start, col_end)) in enumerate(zip(secondary_dimensions, secondary_positions)):
            if dimension in self.dimensions:
                ax = fig.add_subplot(gs[row, col_start:col_end])
                self._plot_dimension(ax, dimension)
                
                # Format title
                dim_name = dimension.replace('_z', '').replace('_', ' ').title()
                ax.set_title(dim_name, fontsize=11, fontweight='bold')
        
        logger.info("Figure generation complete")
        
        return fig
    
    def export_figure(
        self,
        output_path: str,
        dpi: int = 300
    ) -> str:
        """
        Generate and export figure to file.
        
        Args:
            output_path (str): Output file path
            dpi (int): Resolution in dots per inch (default: 300)
            
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
        
        return output_path
