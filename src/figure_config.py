# -*- coding: utf-8 -*-
"""
Centralized figure configuration for Nature Human Behaviour publication.

This module defines standardized font sizes, dimensions, and styling parameters
to ensure consistency across all figures in the manuscript.

Nature Human Behaviour specifications:
- Single column width: 89mm (~3.5 inches)
- Double column width: 183mm (~7.2 inches)
- Maximum height: 247mm (~9.7 inches)
- Minimum font size after reduction: 5pt (recommend 7-8pt in source)
- Panel labels: Bold uppercase letters (A, B, C, etc.)

Reference: https://www.nature.com/nathumbehav/submission-guidelines
"""

from typing import Dict, Any
import matplotlib.pyplot as plt

# =============================================================================
# FIGURE DIMENSIONS (in inches, converted from mm)
# =============================================================================

# Nature column widths
SINGLE_COL_WIDTH_MM = 89
DOUBLE_COL_WIDTH_MM = 183
MAX_HEIGHT_MM = 247

# Convert to inches (1 inch = 25.4 mm)
MM_TO_INCH = 1 / 25.4
SINGLE_COL_WIDTH = SINGLE_COL_WIDTH_MM * MM_TO_INCH  # ~3.5 inches
DOUBLE_COL_WIDTH = DOUBLE_COL_WIDTH_MM * MM_TO_INCH  # ~7.2 inches
MAX_HEIGHT = MAX_HEIGHT_MM * MM_TO_INCH  # ~9.7 inches

# Standard figure sizes (width, height) in inches
FIG_SIZE_SINGLE = (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.75)  # 3.5 x 2.6
FIG_SIZE_DOUBLE = (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5)   # 7.2 x 3.6
FIG_SIZE_DOUBLE_TALL = (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.75)  # 7.2 x 5.4
FIG_SIZE_FULL_PAGE = (DOUBLE_COL_WIDTH, MAX_HEIGHT * 0.8)  # 7.2 x 7.8

# =============================================================================
# FONT SIZES (in points)
# =============================================================================

# These sizes are designed to be readable after typical journal reduction
# Nature typically reduces figures, so we use slightly larger sizes

# Main text elements
FONT_SIZE_TITLE = 12        # Figure/panel titles
FONT_SIZE_SUBTITLE = 11     # Subtitles (e.g., "(Emotional Intensity)")
FONT_SIZE_AXIS_LABEL = 11   # X and Y axis labels
FONT_SIZE_TICK_LABEL = 10   # Tick labels on axes
FONT_SIZE_LEGEND = 10       # Legend text
FONT_SIZE_ANNOTATION = 10   # Annotations, statistics text

# Panel labels (A, B, C, etc.)
FONT_SIZE_PANEL_LABEL = 14  # Bold uppercase letters
PANEL_LABEL_WEIGHT = 'bold'

# Small subplots (when many panels in one figure)
FONT_SIZE_TITLE_SMALL = 11
FONT_SIZE_AXIS_LABEL_SMALL = 10
FONT_SIZE_TICK_LABEL_SMALL = 9
FONT_SIZE_LEGEND_SMALL = 9

# =============================================================================
# LINE AND MARKER SIZES
# =============================================================================

LINE_WIDTH = 1.5            # Main data lines
LINE_WIDTH_THIN = 1.0       # Secondary lines, grid
LINE_WIDTH_THICK = 2.0      # Emphasis lines
MARKER_SIZE = 4             # Data point markers
MARKER_SIZE_SMALL = 3       # Small markers for dense plots

# =============================================================================
# COLORS (using tab20c palette for consistency)
# =============================================================================

# Get tab20c colors
TAB20C = plt.cm.tab20c.colors
TAB20B = plt.cm.tab20b.colors

# ECG/HR colors (blue family - indices 0-3)
COLOR_ECG_HIGH = TAB20C[0]   # Dark blue for High dose
COLOR_ECG_LOW = TAB20C[2]    # Light blue for Low dose

# EDA/SMNA colors (orange family - indices 4-7)
COLOR_EDA_HIGH = TAB20C[4]   # Dark orange for High dose
COLOR_EDA_LOW = TAB20C[6]    # Light orange for Low dose

# Resp/RVT colors (green family - indices 8-11)
COLOR_RESP_HIGH = TAB20C[8]  # Dark green for High dose
COLOR_RESP_LOW = TAB20C[10]  # Light green for Low dose

# Composite/PCA colors (yellow/beige family from tab20b - indices 8-11)
COLOR_COMPOSITE_HIGH = TAB20B[8]   # Dark yellow for High dose
COLOR_COMPOSITE_LOW = TAB20B[10]   # Light yellow for Low dose
COLOR_PCA = TAB20B[8]              # PCA loadings color

# TET colors (purple family)
COLOR_TET_HIGH = '#5E4FA2'   # Dark purple for High dose
COLOR_TET_LOW = '#9E9AC8'    # Light purple for Low dose

# State colors
COLOR_RS = '#9E9AC8'         # Resting State
COLOR_DMT = '#5E4FA2'        # DMT

# Significance shading
COLOR_SIG_SHADE = '0.85'     # Gray for significant regions
ALPHA_SIG_SHADE = 0.35

# =============================================================================
# LEGEND CONFIGURATION
# =============================================================================

LEGEND_FONTSIZE = FONT_SIZE_LEGEND
LEGEND_MARKERSCALE = 1.2
LEGEND_BORDERPAD = 0.4
LEGEND_HANDLELENGTH = 2.0
LEGEND_LABELSPACING = 0.5
LEGEND_BORDERAXESPAD = 0.5
LEGEND_FRAMEON = True
LEGEND_FANCYBOX = False
LEGEND_FRAMEALPHA = 0.9

# =============================================================================
# AXIS CONFIGURATION
# =============================================================================

SPINE_LINEWIDTH = 0.8
TICK_MAJOR_WIDTH = 0.8
TICK_MAJOR_LENGTH = 4
TICK_MINOR_WIDTH = 0.5
TICK_MINOR_LENGTH = 2
GRID_ALPHA = 0.3
GRID_LINEWIDTH = 0.5

# =============================================================================
# MATPLOTLIB RCPARAMS
# =============================================================================

def get_rcparams() -> Dict[str, Any]:
    """Get matplotlib rcParams for consistent figure styling.
    
    Returns:
        Dictionary of rcParams to update matplotlib settings.
    """
    return {
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT_SIZE_TICK_LABEL,
        
        # Axes settings
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_AXIS_LABEL,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.linewidth': SPINE_LINEWIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.titlepad': 6,
        'axes.labelpad': 4,
        
        # Tick settings
        'xtick.labelsize': FONT_SIZE_TICK_LABEL,
        'ytick.labelsize': FONT_SIZE_TICK_LABEL,
        'xtick.major.width': TICK_MAJOR_WIDTH,
        'ytick.major.width': TICK_MAJOR_WIDTH,
        'xtick.major.size': TICK_MAJOR_LENGTH,
        'ytick.major.size': TICK_MAJOR_LENGTH,
        
        # Legend settings
        'legend.fontsize': LEGEND_FONTSIZE,
        'legend.frameon': LEGEND_FRAMEON,
        'legend.framealpha': LEGEND_FRAMEALPHA,
        'legend.borderpad': LEGEND_BORDERPAD,
        'legend.handlelength': LEGEND_HANDLELENGTH,
        'legend.labelspacing': LEGEND_LABELSPACING,
        
        # Line settings
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        
        # Grid settings
        'grid.alpha': GRID_ALPHA,
        'grid.linewidth': GRID_LINEWIDTH,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    }


def apply_rcparams() -> None:
    """Apply standardized rcParams to matplotlib."""
    plt.rcParams.update(get_rcparams())


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add a panel label (A, B, C, etc.) to an axis.
    
    Args:
        ax: Matplotlib axis object.
        label: Panel label string (e.g., 'A', 'B').
        x: X position in axis coordinates (default: -0.12).
        y: Y position in axis coordinates (default: 1.08).
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL, fontweight=PANEL_LABEL_WEIGHT,
            va='top', ha='left')


def add_panel_label_fig(fig, ax, label: str, x_offset: float = -0.02, 
                        y_offset: float = 0.02) -> None:
    """Add a panel label using figure coordinates (for assembled figures).
    
    Args:
        fig: Matplotlib figure object.
        ax: Matplotlib axis object (used to get position).
        label: Panel label string.
        x_offset: X offset from axis left edge.
        y_offset: Y offset from axis top edge.
    """
    pos = ax.get_position()
    fig.text(pos.x0 + x_offset, pos.y1 + y_offset, label,
             fontsize=FONT_SIZE_PANEL_LABEL, fontweight=PANEL_LABEL_WEIGHT,
             va='top', ha='left')


def style_legend(legend, facecolor: str = 'white', alpha: float = 0.9) -> None:
    """Apply consistent styling to a legend.
    
    Args:
        legend: Matplotlib legend object.
        facecolor: Background color.
        alpha: Background transparency.
    """
    if legend is not None:
        frame = legend.get_frame()
        frame.set_facecolor(facecolor)
        frame.set_alpha(alpha)
        frame.set_linewidth(0.5)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_figure_single(height_ratio: float = 0.75):
    """Create a single-column figure.
    
    Args:
        height_ratio: Height as fraction of width.
    
    Returns:
        Matplotlib figure and axis.
    """
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * height_ratio))
    return fig, ax


def create_figure_double(height_ratio: float = 0.5):
    """Create a double-column figure.
    
    Args:
        height_ratio: Height as fraction of width.
    
    Returns:
        Matplotlib figure and axis.
    """
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * height_ratio))
    return fig, ax


def create_figure_double_panels(nrows: int, ncols: int, height_ratio: float = 0.5,
                                **kwargs):
    """Create a double-column figure with multiple panels.
    
    Args:
        nrows: Number of rows.
        ncols: Number of columns.
        height_ratio: Height as fraction of width.
        **kwargs: Additional arguments for plt.subplots.
    
    Returns:
        Matplotlib figure and axes array.
    """
    apply_rcparams()
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * height_ratio),
                             **kwargs)
    return fig, axes
