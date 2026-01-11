"""
Statistical Reporting Module for DMT Physiological Analysis

This module provides APA-style formatting functions for statistical results,
ensuring complete and standardized reporting across all analyses.

Scientific Rationale:
- APA style is the standard for psychological and neuroscience publications
- Complete reporting (test statistic, df, p-value, effect size) enables meta-analyses
- Standardized formatting reduces manual errors and ensures consistency

Author: DMT Analysis Pipeline
"""

from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


def format_ttest_result(
    t_stat: float,
    df: int,
    p_value: float,
    cohens_d: float,
    mean_diff: Optional[float] = None,
) -> str:
    """
    Format t-test result in APA style.

    APA format for t-tests: t(df) = X.XX, p = .XXX, d = X.XX

    Args:
        t_stat: t-statistic
        df: Degrees of freedom
        p_value: p-value
        cohens_d: Cohen's d effect size
        mean_diff: Optional mean difference between groups

    Returns:
        Formatted string in APA style

    Examples:
        >>> format_ttest_result(2.45, 17, 0.025, 0.58)
        't(17) = 2.45, p = .025, d = 0.58'

        >>> format_ttest_result(3.21, 17, 0.0001, 0.76, mean_diff=1.23)
        't(17) = 3.21, p < .001, d = 0.76, M_diff = 1.23'
    """
    # Format p-value
    if p_value < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p_value:.3f}".replace("0.", ".")

    # Build base string
    result = f"t({df}) = {t_stat:.2f}, {p_str}, d = {cohens_d:.2f}"

    # Add mean difference if provided
    if mean_diff is not None:
        result += f", M_diff = {mean_diff:.2f}"

    return result


def format_lme_result(
    beta: float,
    ci_lower: float,
    ci_upper: float,
    p_fdr: float,
    parameter_name: str,
) -> str:
    """
    Format LME (Linear Mixed-Effects) result in APA style.

    APA format for LME: β = X.XX, 95% CI [X.XX, X.XX], p_FDR = .XXX

    Args:
        beta: Coefficient estimate
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        p_fdr: FDR-corrected p-value
        parameter_name: Name of parameter (e.g., 'State[T.DMT]', 'Dose[T.High]')

    Returns:
        Formatted string in APA style

    Examples:
        >>> format_lme_result(0.45, 0.12, 0.78, 0.008, 'State[T.DMT]')
        'State[T.DMT]: β = 0.45, 95% CI [0.12, 0.78], p_FDR = .008'

        >>> format_lme_result(-0.23, -0.56, 0.10, 0.15, 'Dose[T.High]')
        'Dose[T.High]: β = -0.23, 95% CI [-0.56, 0.10], p_FDR = .150'
    """
    # Format p-value
    if p_fdr < 0.001:
        p_str = "p_FDR < .001"
    else:
        p_str = f"p_FDR = {p_fdr:.3f}".replace("0.", ".")

    # Build result string
    result = (
        f"{parameter_name}: β = {beta:.2f}, "
        f"95% CI [{ci_lower:.2f}, {ci_upper:.2f}], {p_str}"
    )

    return result


def format_correlation_result(
    r: float,
    n: int,
    p_fdr: float,
) -> str:
    """
    Format correlation result in APA style.

    APA format for correlations: r(n-2) = .XX, p_FDR = .XXX

    Args:
        r: Pearson correlation coefficient
        n: Sample size
        p_fdr: FDR-corrected p-value

    Returns:
        Formatted string in APA style

    Examples:
        >>> format_correlation_result(0.67, 18, 0.002)
        'r(16) = .67, p_FDR = .002'

        >>> format_correlation_result(-0.45, 20, 0.045)
        'r(18) = -.45, p_FDR = .045'
    """
    # Degrees of freedom for correlation is n-2
    df = n - 2

    # Format p-value
    if p_fdr < 0.001:
        p_str = "p_FDR < .001"
    else:
        p_str = f"p_FDR = {p_fdr:.3f}".replace("0.", ".")

    # Format r (remove leading zero)
    r_str = f"{r:.2f}".replace("0.", ".")
    if r < 0:
        r_str = f"-{r_str.replace('-', '')}"

    result = f"r({df}) = {r_str}, {p_str}"

    return result


def save_results_table(
    results_dict: Dict[str, Any],
    output_path: str,
    format: str = "csv",
) -> None:
    """
    Save statistical results to CSV or text file with all relevant statistics.

    This function converts a dictionary of statistical results into a structured
    table format suitable for publication or further analysis.

    Args:
        results_dict: Dictionary containing statistical results.
                     Can be nested (e.g., {'phase1': {'t_stat': 2.5, ...}})
                     or flat (e.g., {'t_stat': 2.5, 'p_value': 0.02, ...})
        output_path: Path to save results (with or without extension)
        format: Output format ('csv' or 'txt')

    Examples:
        >>> results = {
        ...     'phase1': {'t_stat': 2.45, 'p_value': 0.025, 'cohens_d': 0.58},
        ...     'phase2': {'t_stat': 1.89, 'p_value': 0.075, 'cohens_d': 0.45}
        ... }
        >>> save_results_table(results, 'results/phase_analysis.csv')

        >>> results = {'t_stat': 2.45, 'df': 17, 'p_value': 0.025}
        >>> save_results_table(results, 'results/ttest_result.txt', format='txt')
    """
    # Ensure output path has correct extension
    if not output_path.endswith(f".{format}"):
        output_path = f"{output_path}.{format}"

    # Convert results to DataFrame
    if isinstance(results_dict, dict):
        # Check if nested dictionary (multiple results)
        first_value = next(iter(results_dict.values()))
        if isinstance(first_value, dict):
            # Nested: convert to long format
            df = pd.DataFrame.from_dict(results_dict, orient="index")
            df.index.name = "analysis"
            df = df.reset_index()
        else:
            # Flat: single row
            df = pd.DataFrame([results_dict])
    elif isinstance(results_dict, pd.DataFrame):
        # Already a DataFrame
        df = results_dict
    else:
        raise ValueError(
            f"results_dict must be dict or DataFrame, got {type(results_dict)}"
        )

    # Save based on format
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "txt":
        # For text format, create a more readable output
        with open(output_path, "w") as f:
            f.write("Statistical Results\n")
            f.write("=" * 80 + "\n\n")

            if len(df) == 1:
                # Single result: write as key-value pairs
                for col in df.columns:
                    value = df[col].iloc[0]
                    f.write(f"{col}: {value}\n")
            else:
                # Multiple results: write as table
                f.write(df.to_string(index=False))
                f.write("\n")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'txt'")

    print(f"Results saved to: {output_path}")


def format_phase_comparison_results(
    phase_results: pd.DataFrame,
) -> str:
    """
    Format phase comparison results for reporting.

    This is a convenience function that formats multiple phase comparisons
    into a readable text report.

    Args:
        phase_results: DataFrame from phase_analyzer.compare_doses_within_phases()
                      with columns: phase, State, t_stat, p_value, cohens_d,
                      mean_high, mean_low, sem_high, sem_low

    Returns:
        Formatted text report with all phase comparisons

    Example:
        >>> phase_df = pd.DataFrame({
        ...     'phase': [0, 1],
        ...     'phase_label': ['Onset', 'Recovery'],
        ...     'State': ['DMT', 'DMT'],
        ...     't_stat': [2.45, 1.89],
        ...     'p_value': [0.025, 0.075],
        ...     'cohens_d': [0.58, 0.45],
        ...     'mean_high': [1.23, 0.98],
        ...     'mean_low': [0.67, 0.54]
        ... })
        >>> print(format_phase_comparison_results(phase_df))
    """
    report_lines = ["Phase-Based Dose Comparisons (High vs Low)", "=" * 80, ""]

    for _, row in phase_results.iterrows():
        phase_label = row.get("phase_label", f"Phase {row['phase']}")
        state = row["State"]

        # Format t-test result
        ttest_str = format_ttest_result(
            t_stat=row["t_stat"],
            df=row.get("df", 17),  # Default to 17 for 18 subjects
            p_value=row["p_value"],
            cohens_d=row["cohens_d"],
        )

        # Add descriptive statistics
        mean_high = row.get("mean_high", np.nan)
        mean_low = row.get("mean_low", np.nan)
        sem_high = row.get("sem_high", np.nan)
        sem_low = row.get("sem_low", np.nan)

        report_lines.append(f"{phase_label} ({state}):")
        report_lines.append(f"  {ttest_str}")
        if not np.isnan(mean_high):
            report_lines.append(
                f"  High: M = {mean_high:.2f}, SEM = {sem_high:.2f}"
            )
            report_lines.append(
                f"  Low:  M = {mean_low:.2f}, SEM = {sem_low:.2f}"
            )
        report_lines.append("")

    return "\n".join(report_lines)


def format_feature_comparison_results(
    feature_results: pd.DataFrame,
) -> str:
    """
    Format feature comparison results for reporting.

    This is a convenience function that formats multiple feature comparisons
    (e.g., peak amplitude, time-to-peak) into a readable text report.

    Args:
        feature_results: DataFrame with columns: feature, t_stat, df, p_value,
                        cohens_d, mean_high, mean_low, std_high, std_low

    Returns:
        Formatted text report with all feature comparisons

    Example:
        >>> feature_df = pd.DataFrame({
        ...     'feature': ['peak_amplitude', 'time_to_peak'],
        ...     't_stat': [3.21, -1.45],
        ...     'df': [17, 17],
        ...     'p_value': [0.005, 0.165],
        ...     'cohens_d': [0.76, -0.34],
        ...     'mean_high': [2.45, 3.2],
        ...     'mean_low': [1.67, 3.8]
        ... })
        >>> print(format_feature_comparison_results(feature_df))
    """
    report_lines = ["Feature-Based Dose Comparisons (High vs Low)", "=" * 80, ""]

    for _, row in feature_results.iterrows():
        feature = row["feature"]

        # Format t-test result
        ttest_str = format_ttest_result(
            t_stat=row["t_stat"],
            df=row["df"],
            p_value=row["p_value"],
            cohens_d=row["cohens_d"],
        )

        # Add descriptive statistics
        mean_high = row.get("mean_high", np.nan)
        mean_low = row.get("mean_low", np.nan)
        std_high = row.get("std_high", np.nan)
        std_low = row.get("std_low", np.nan)

        report_lines.append(f"{feature}:")
        report_lines.append(f"  {ttest_str}")
        if not np.isnan(mean_high):
            report_lines.append(f"  High: M = {mean_high:.2f}, SD = {std_high:.2f}")
            report_lines.append(f"  Low:  M = {mean_low:.2f}, SD = {std_low:.2f}")
        report_lines.append("")

    return "\n".join(report_lines)
