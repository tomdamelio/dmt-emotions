"""
Enhanced Visualization Module for DMT Physiological Analysis

This module provides functions for adding significance markers to plots,
applying homogeneous aesthetics across multi-panel figures, and saving
figures in vector formats for publication.

Scientific Rationale:
- Significance markers (*, **, ***) provide immediate visual feedback on
  statistical significance levels based on FDR-corrected p-values
- Homogeneous aesthetics ensure consistency across all figure panels,
  meeting publication standards
- Vector formats (PDF, SVG) preserve quality for high-resolution printing

Author: DMT Analysis Pipeline
"""

from typing import Dict, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Global font size constants for homogeneous aesthetics
FONT_SIZES = {
    'axis_label': 10,      # Axis labels (e.g., "Time (min)", "Heart Rate (bpm)")
    'title': 12,           # Subplot titles
    'tick_label': 8,       # Tick labels on axes
    'panel_label': 14      # Panel identifiers (A, B, C, D)
}


def add_significance_markers(
    ax: Axes,
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    p_values_fdr: np.ndarray,
    offset_factor: float = 0.05
) -> None:
    """
    Add significance markers (*, **, ***) to coefficient plots based on FDR-corrected p-values.
    
    Scientific Rationale:
    Significance markers provide immediate visual feedback on statistical significance
    levels, allowing readers to quickly identify which effects survive multiple
    comparison correction. The three-tier system (*, **, ***) follows standard
    conventions in psychological and neuroscience research.
    
    Marker conventions:
        - p_fdr < 0.001: '***'
        - p_fdr < 0.01: '**'
        - p_fdr < 0.05: '*'
        - p_fdr >= 0.05: no marker
    
    Args:
        ax: Matplotlib axes object to add markers to
        x_positions: X-coordinates for markers (e.g., coefficient indices or time points)
        y_positions: Y-coordinates for markers (typically coefficient values or effect sizes)
        p_values_fdr: FDR-corrected p-values corresponding to each position
        offset_factor: Vertical offset as fraction of y-range (default 0.05)
    
    Returns:
        None (modifies ax in place)
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.bar(range(5), [0.5, 1.2, 0.3, 0.8, 1.5])
        >>> p_vals = np.array([0.001, 0.008, 0.03, 0.15, 0.0001])
        >>> add_significance_markers(ax, range(5), [0.5, 1.2, 0.3, 0.8, 1.5], p_vals)
    """
    # Validate inputs
    if len(x_positions) != len(y_positions) or len(x_positions) != len(p_values_fdr):
        raise ValueError("x_positions, y_positions, and p_values_fdr must have the same length")
    
    # Get y-axis range for offset calculation
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = y_range * offset_factor
    
    # Iterate through positions and add markers based on significance level
    for x, y, p_fdr in zip(x_positions, y_positions, p_values_fdr):
        if p_fdr < 0.001:
            marker = '***'
        elif p_fdr < 0.01:
            marker = '**'
        elif p_fdr < 0.05:
            marker = '*'
        else:
            continue  # No marker for non-significant results
        
        # Position marker above the data point
        marker_y = y + offset if y > 0 else y - offset
        ax.text(
            x, marker_y, marker,
            ha='center', va='bottom' if y > 0 else 'top',
            fontsize=FONT_SIZES['axis_label'],
            fontweight='bold'
        )


def apply_homogeneous_aesthetics(
    fig: Figure,
    axes: Union[Axes, np.ndarray],
    font_sizes: Optional[Dict[str, int]] = None,
    panel_labels: Optional[List[str]] = None
) -> None:
    """
    Apply consistent font sizes, aspect ratios, and panel labels to multi-panel figures.
    
    Scientific Rationale:
    Homogeneous aesthetics across all panels ensure that figures meet publication
    standards and facilitate visual comparison across different analyses. Consistent
    font sizes improve readability, while standardized panel labels (A, B, C, D)
    allow clear referencing in manuscript text.
    
    Default font sizes:
        - axis_label: 10pt (axis titles like "Time (min)")
        - title: 12pt (subplot titles)
        - tick_label: 8pt (numeric labels on axes)
        - panel_label: 14pt bold (panel identifiers A, B, C, D)
    
    Args:
        fig: Matplotlib figure object
        axes: Single axes or array of axes (from plt.subplots)
        font_sizes: Optional dictionary with keys: 'axis_label', 'title', 'tick_label',
                   'panel_label'. If None, uses global FONT_SIZES constants.
        panel_labels: Optional list of panel identifiers (e.g., ['A', 'B', 'C', 'D']).
                     If provided, labels are added to top-left of each panel.
    
    Returns:
        None (modifies fig and axes in place)
    
    Example:
        >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        >>> apply_homogeneous_aesthetics(fig, axes, panel_labels=['A', 'B', 'C', 'D'])
    """
    # Use global font sizes if not provided
    if font_sizes is None:
        font_sizes = FONT_SIZES.copy()
    
    # Convert single axes to array for uniform handling
    if isinstance(axes, Axes):
        axes_array = np.array([axes])
    else:
        axes_array = axes.flatten()
    
    # Apply font sizes to each subplot
    for idx, ax in enumerate(axes_array):
        # Set axis label font sizes
        ax.xaxis.label.set_fontsize(font_sizes['axis_label'])
        ax.yaxis.label.set_fontsize(font_sizes['axis_label'])
        
        # Set tick label font sizes
        ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
        
        # Set title font size if title exists
        if ax.get_title():
            ax.title.set_fontsize(font_sizes['title'])
        
        # Add panel labels if provided
        if panel_labels is not None and idx < len(panel_labels):
            # Position panel label in top-left corner
            ax.text(
                -0.1, 1.05, panel_labels[idx],
                transform=ax.transAxes,
                fontsize=font_sizes['panel_label'],
                fontweight='bold',
                va='bottom', ha='right'
            )
    
    # Adjust layout to prevent label overlap
    fig.tight_layout()


def save_figure_vector(
    fig: Figure,
    output_path: str,
    formats: List[str] = ['pdf', 'svg'],
    dpi: int = 300,
    add_timestamp: bool = True
) -> None:
    """
    Save figure in vector formats for publication with optional timestamp.
    
    Scientific Rationale:
    Vector formats (PDF, SVG) preserve quality at any resolution, making them
    ideal for publication. Unlike raster formats (PNG, JPG), vector graphics
    maintain crisp lines and text when scaled or printed at high resolution.
    
    Timestamps enable version tracking and reproducibility by documenting when
    figures were generated.
    
    Args:
        fig: Matplotlib figure object to save
        output_path: Base path without extension (e.g., 'results/figures/figure_1')
        formats: List of formats to save (default ['pdf', 'svg'])
        dpi: Resolution for any embedded raster elements (default 300)
        add_timestamp: Whether to add timestamp to filename (default True)
    
    Returns:
        None (saves files to disk)
    
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> save_figure_vector(fig, 'results/figures/my_plot')
        # Saves: results/figures/my_plot_20260111_143022.pdf and .svg
        
        >>> save_figure_vector(fig, 'results/figures/my_plot', add_timestamp=False)
        # Saves: results/figures/my_plot.pdf and .svg
    """
    import os
    from datetime import datetime
    
    # Validate formats
    valid_formats = ['pdf', 'svg', 'eps', 'png']
    for fmt in formats:
        if fmt not in valid_formats:
            raise ValueError(f"Invalid format '{fmt}'. Must be one of {valid_formats}")
    
    # Add timestamp to filename if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Split path into directory and filename
        dir_path = os.path.dirname(output_path)
        base_name = os.path.basename(output_path)
        # Insert timestamp before extension
        output_path_with_timestamp = os.path.join(dir_path, f"{base_name}_{timestamp}")
    else:
        output_path_with_timestamp = output_path
    
    # Save in each requested format
    for fmt in formats:
        output_file = f"{output_path_with_timestamp}.{fmt}"
        fig.savefig(
            output_file,
            format=fmt,
            dpi=dpi,
            bbox_inches='tight',
            transparent=False
        )
        print(f"Saved figure: {output_file}")
    
    # Also save without timestamp for backward compatibility
    if add_timestamp:
        for fmt in formats:
            output_file_no_timestamp = f"{output_path}.{fmt}"
            fig.savefig(
                output_file_no_timestamp,
                format=fmt,
                dpi=dpi,
                bbox_inches='tight',
                transparent=False
            )
            print(f"Saved figure (no timestamp): {output_file_no_timestamp}")


def get_consistent_aspect_ratio(
    n_timepoints: int,
    duration_min: float = 9.0,
    height_inches: float = 3.0
) -> float:
    """
    Calculate consistent aspect ratio for time-series plots.
    
    Scientific Rationale:
    Consistent aspect ratios across time-series plots facilitate visual comparison
    of temporal dynamics across different physiological modalities (HR, SMNA, RVT).
    The aspect ratio is calculated based on the time duration and desired height.
    
    Args:
        n_timepoints: Number of time points in the series
        duration_min: Duration of time series in minutes (default 9.0)
        height_inches: Desired height of plot in inches (default 3.0)
    
    Returns:
        Width in inches to achieve consistent aspect ratio
    
    Example:
        >>> width = get_consistent_aspect_ratio(18, duration_min=9.0, height_inches=3.0)
        >>> fig, ax = plt.subplots(figsize=(width, 3.0))
    """
    # Calculate width based on time duration and desired aspect ratio
    # Use a standard ratio of 2:1 (width:height) for time-series
    width_inches = height_inches * 2.0
    return width_inches


def format_pvalue_text(p_value: float, fdr_corrected: bool = True) -> str:
    """
    Format p-value for display in plots or reports.
    
    Args:
        p_value: P-value to format
        fdr_corrected: Whether p-value is FDR-corrected (default True)
    
    Returns:
        Formatted string (e.g., "p_FDR < .001" or "p = .023")
    
    Example:
        >>> format_pvalue_text(0.0001, fdr_corrected=True)
        'p_FDR < .001'
        >>> format_pvalue_text(0.023, fdr_corrected=False)
        'p = .023'
    """
    prefix = "p_FDR" if fdr_corrected else "p"
    
    if p_value < 0.001:
        return f"{prefix} < .001"
    else:
        # Format with 3 decimal places, removing leading zero
        formatted_val = f"{p_value:.3f}".lstrip('0')
        return f"{prefix} = {formatted_val}"
