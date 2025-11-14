"""
TET Results Formatter

This module provides utilities for formatting statistical results with consistent notation.
"""

import numpy as np


class StatisticalFormatter:
    """
    Format statistical results with consistent notation.
    
    This class provides static methods for formatting statistical results
    (coefficients, effect sizes, p-values, figure references) with standardized
    notation for use in markdown reports.
    
    Examples
    --------
    >>> formatter = StatisticalFormatter()
    >>> formatter.format_lme_result(2.34, 1.98, 2.70, 0.0001)
    'β = 2.34, 95% CI [1.98, 2.70], p_fdr < 0.001'
    
    >>> formatter.format_effect_size(0.52, 0.28, 0.71, 0.003)
    'r = 0.52, 95% CI [0.28, 0.71], p_fdr = 0.003'
    """
    
    @staticmethod
    def format_lme_result(beta: float, ci_lower: float, ci_upper: float, p_fdr: float) -> str:
        """
        Format LME coefficient result.
        
        Parameters
        ----------
        beta : float
            Coefficient estimate
        ci_lower : float
            Lower bound of 95% CI
        ci_upper : float
            Upper bound of 95% CI
        p_fdr : float
            FDR-corrected p-value
        
        Returns
        -------
        str
            Formatted string: β = X.XX, 95% CI [X.XX, X.XX], p_fdr = X.XXX
        """
        p_str = StatisticalFormatter.format_p_value(p_fdr)
        return f"β = {beta:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], {p_str}"
    
    @staticmethod
    def format_effect_size(r: float, ci_lower: float, ci_upper: float, p_fdr: float) -> str:
        """
        Format effect size result.
        
        Parameters
        ----------
        r : float
            Effect size (correlation coefficient)
        ci_lower : float
            Lower bound of 95% CI
        ci_upper : float
            Upper bound of 95% CI
        p_fdr : float
            FDR-corrected p-value
        
        Returns
        -------
        str
            Formatted string: r = X.XX, 95% CI [X.XX, X.XX], p_fdr = X.XXX
        """
        p_str = StatisticalFormatter.format_p_value(p_fdr)
        return f"r = {r:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], {p_str}"
    
    @staticmethod
    def format_p_value(p: float, prefix: str = 'p_fdr') -> str:
        """
        Format p-value with appropriate precision.
        
        Parameters
        ----------
        p : float
            P-value
        prefix : str, optional
            Prefix for p-value (e.g., 'p', 'p_fdr'), by default 'p_fdr'
        
        Returns
        -------
        str
            Formatted p-value: p_fdr < 0.001 or p_fdr = X.XXX
        """
        if p < 0.001:
            return f"{prefix} < 0.001"
        else:
            return f"{prefix} = {p:.3f}"
    
    @staticmethod
    def format_figure_reference(figure_name: str, relative_path: str = '../results/tet/figures') -> str:
        """
        Format figure reference with relative path.
        
        Parameters
        ----------
        figure_name : str
            Name of figure file (e.g., 'timeseries_pleasantness.png')
        relative_path : str, optional
            Relative path to figures directory, by default '../results/tet/figures'
        
        Returns
        -------
        str
            Formatted figure reference: [See Figure: path/to/figure.png]
        """
        return f"[See Figure: {relative_path}/{figure_name}]"
    
    @staticmethod
    def format_dimension_name(dimension: str) -> str:
        """
        Clean dimension name for display.
        
        Removes '_z' suffix, converts underscores to spaces, and applies title case.
        
        Parameters
        ----------
        dimension : str
            Raw dimension name (e.g., 'complex_imagery_z')
        
        Returns
        -------
        str
            Cleaned dimension name (e.g., 'Complex Imagery')
        
        Examples
        --------
        >>> StatisticalFormatter.format_dimension_name('complex_imagery_z')
        'Complex Imagery'
        >>> StatisticalFormatter.format_dimension_name('pleasantness')
        'Pleasantness'
        """
        # Remove _z suffix
        if dimension.endswith('_z'):
            dimension = dimension[:-2]
        
        # Convert underscores to spaces
        dimension = dimension.replace('_', ' ')
        
        # Title case
        dimension = dimension.title()
        
        return dimension
    
    @staticmethod
    def format_mean_sem(mean: float, sem: float) -> str:
        """
        Format mean ± SEM.
        
        Parameters
        ----------
        mean : float
            Mean value
        sem : float
            Standard error of the mean
        
        Returns
        -------
        str
            Formatted string: X.XX ± X.XX
        """
        return f"{mean:.2f} ± {sem:.2f}"
    
    @staticmethod
    def format_ci(ci_lower: float, ci_upper: float) -> str:
        """
        Format confidence interval.
        
        Parameters
        ----------
        ci_lower : float
            Lower bound
        ci_upper : float
            Upper bound
        
        Returns
        -------
        str
            Formatted CI: [X.XX, X.XX]
        """
        return f"[{ci_lower:.2f}, {ci_upper:.2f}]"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """
        Format percentage.
        
        Parameters
        ----------
        value : float
            Value as proportion (0-1) or percentage (0-100)
        
        Returns
        -------
        str
            Formatted percentage: XX.X%
        """
        # Convert to percentage if needed
        if value <= 1.0:
            value = value * 100
        
        return f"{value:.1f}%"
