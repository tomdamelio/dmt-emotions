"""
Generate boxplots comparing High vs Low dose for peak, time_to_peak, and AUC metrics.

This script creates publication-ready boxplots showing dose effects on DMT experience
intensity metrics with statistical annotations.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Plot aesthetics - matching LME coefficient plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})


def load_data(metrics_path: Path, tests_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load peak/AUC metrics and test results.

    Args:
        metrics_path: Path to peak_auc_metrics.csv
        tests_path: Path to peak_auc_tests.csv

    Returns:
        Tuple of (metrics_df, tests_df)
    """
    logger.info(f"Loading metrics from {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)

    logger.info(f"Loading test results from {tests_path}")
    tests_df = pd.read_csv(tests_path)

    return metrics_df, tests_df


def prepare_plot_data(metrics_df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """
    Prepare data for plotting a specific metric type.

    Args:
        metrics_df: DataFrame with peak/AUC metrics
        metric_type: One of 'peak', 'time_to_peak_min', 'auc_0_9'

    Returns:
        DataFrame with columns: dimension, dose, subject, value
    """
    # Select relevant columns
    plot_data = metrics_df[["subject", "dose", "dimension", metric_type]].copy()
    plot_data = plot_data.rename(columns={metric_type: "value"})

    return plot_data


def get_dimension_order(tests_df: pd.DataFrame, metric_type: str) -> list[str]:
    """
    Get dimensions ordered by effect size for a specific metric.

    Args:
        tests_df: DataFrame with test results
        metric_type: One of 'peak', 'time_to_peak_min', 'auc_0_9'

    Returns:
        List of dimension names ordered by absolute effect size (descending)
    """
    metric_tests = tests_df[tests_df["metric"] == metric_type].copy()
    metric_tests["abs_effect"] = metric_tests["effect_r"].abs()
    metric_tests = metric_tests.sort_values("abs_effect", ascending=False)

    return metric_tests["dimension"].tolist()


def format_pvalue_stars(p_value: float) -> str:
    """
    Convert p-value to significance stars.

    Args:
        p_value: P-value from statistical test

    Returns:
        String with stars: '***' p<0.001, '**' p<0.01, '*' p<0.05, '' otherwise
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def format_dimension_label(dimension: str) -> str:
    """
    Format dimension name for display.

    Args:
        dimension: Dimension name with _z suffix

    Returns:
        Formatted dimension name
    """
    # Remove _z suffix
    label = dimension.replace("_z", "")
    # Replace underscores with spaces
    label = label.replace("_", " ")
    # Title case
    label = label.title()

    return label


def plot_single_metric_boxplot(
    metrics_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    metric_type: str,
    metric_label: str,
    output_path: Path,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> None:
    """
    Generate boxplot for a single metric comparing High vs Low dose.

    Args:
        metrics_df: DataFrame with peak/AUC metrics per subject
        tests_df: DataFrame with Wilcoxon test results
        metric_type: One of 'peak', 'time_to_peak_min', 'auc_0_9'
        metric_label: Y-axis label for the metric
        output_path: Path to save figure
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch
    """
    # Get dimension order by effect size
    dim_order = get_dimension_order(tests_df, metric_type)

    # Prepare plot data
    plot_data = prepare_plot_data(metrics_df, metric_type)

    # Filter to dimensions in order
    plot_data = plot_data[plot_data["dimension"].isin(dim_order)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Color scheme - matching LME plot style
    colors = {"Baja": "#4169E1", "Alta": "#DC143C"}  # Royal blue for Low, Crimson for High

    # Create boxplots
    sns.boxplot(
        data=plot_data,
        x="dimension",
        y="value",
        hue="dose",
        order=dim_order,
        palette=colors,
        ax=ax,
        showfliers=False,
        linewidth=1.5,
    )

    # Add paired lines connecting subjects
    for dimension in dim_order:
        dim_data = plot_data[plot_data["dimension"] == dimension]

        # Pivot to get High and Low for each subject
        pivot_data = dim_data.pivot(index="subject", columns="dose", values="value")

        # Only plot if both doses present
        if "Alta" in pivot_data.columns and "Baja" in pivot_data.columns:
            # Remove subjects with missing data
            pivot_data = pivot_data.dropna()

            # Get x positions for this dimension
            dim_idx = dim_order.index(dimension)
            x_low = dim_idx - 0.2  # Offset for Low dose box
            x_high = dim_idx + 0.2  # Offset for High dose box

            # Plot lines connecting paired observations
            for _, row in pivot_data.iterrows():
                ax.plot(
                    [x_low, x_high],
                    [row["Baja"], row["Alta"]],
                    color="gray",
                    alpha=0.3,
                    linewidth=0.5,
                    zorder=1,
                )

    # Add statistical annotations - only stars for significant results
    metric_tests = tests_df[tests_df["metric"] == metric_type].copy()

    for dimension in dim_order:
        dim_test = metric_tests[metric_tests["dimension"] == dimension]

        if len(dim_test) == 0:
            continue

        row = dim_test.iloc[0]

        # Only annotate if significant after FDR correction
        if row["p_fdr"] < 0.05:
            # Get y position for annotation (above the boxes)
            dim_data = plot_data[plot_data["dimension"] == dimension]
            y_max = dim_data["value"].max()
            y_range = dim_data["value"].max() - dim_data["value"].min()
            y_pos = y_max + 0.15 * y_range

            # Format annotation text - only stars
            stars = format_pvalue_stars(row["p_fdr"])

            # Add annotation
            dim_idx = dim_order.index(dimension)
            ax.text(
                dim_idx,
                y_pos,
                stars,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
            )

    # Format x-axis labels
    formatted_labels = [format_dimension_label(d) for d in dim_order]
    ax.set_xticklabels(formatted_labels, rotation=45, ha="right", fontsize=9)

    # Set labels
    ax.set_xlabel("Dimension", fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(
        f"Dose Comparison: {metric_label.split('(')[0].strip()}",
        fontsize=12,
        fontweight="bold",
    )

    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    # Replace dose labels
    labels = ["Low (20mg)" if l == "Baja" else "High (40mg)" for l in labels]
    ax.legend(
        handles,
        labels,
        title="Dose",
        fontsize=9,
        title_fontsize=9,
        loc="upper right",
    )

    # Add grid for readability
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved {metric_type} plot to {output_path}")

    plt.close()


def plot_dose_comparison_boxplots(
    metrics_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    output_dir: Path,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 300,
) -> dict[str, Path]:
    """
    Generate separate boxplots for each metric comparing High vs Low dose.

    Creates three separate figures:
    - Peak values
    - Time to peak
    - AUC 0-9 minutes

    Each figure shows boxplots for all dimensions with paired lines connecting
    individual subjects and statistical significance markers.

    Args:
        metrics_df: DataFrame with peak/AUC metrics per subject
        tests_df: DataFrame with Wilcoxon test results
        output_dir: Directory to save figures
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch

    Returns:
        Dictionary mapping metric types to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define metrics to plot
    metrics = [
        ("peak", "Peak Value (z-score)", "peak_dose_comparison.png"),
        (
            "time_to_peak_min",
            "Time to Peak (minutes)",
            "time_to_peak_dose_comparison.png",
        ),
        ("auc_0_9", "AUC 0-9 min (z-score × min)", "auc_dose_comparison.png"),
    ]

    output_paths = {}

    # Create separate plot for each metric
    for metric_type, ylabel, filename in metrics:
        output_path = output_dir / filename

        plot_single_metric_boxplot(
            metrics_df=metrics_df,
            tests_df=tests_df,
            metric_type=metric_type,
            metric_label=ylabel,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )

        output_paths[metric_type] = output_path

    return output_paths


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate peak/AUC dose comparison boxplots"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/tet/peak_auc/peak_auc_metrics.csv"),
        help="Path to peak_auc_metrics.csv",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=Path("results/tet/peak_auc/peak_auc_tests.csv"),
        help="Path to peak_auc_tests.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tet/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure resolution (default: 300)"
    )

    args = parser.parse_args()

    # Validate input files
    if not args.metrics.exists():
        logger.error(f"Metrics file not found: {args.metrics}")
        return

    if not args.tests.exists():
        logger.error(f"Tests file not found: {args.tests}")
        return

    # Load data
    metrics_df, tests_df = load_data(args.metrics, args.tests)

    logger.info(f"Loaded {len(metrics_df)} metric observations")
    logger.info(f"Loaded {len(tests_df)} test results")

    # Generate plots
    output_paths = plot_dose_comparison_boxplots(
        metrics_df=metrics_df,
        tests_df=tests_df,
        output_dir=args.output,
        dpi=args.dpi,
    )

    # Generate figure caption
    caption_path = args.output / "peak_auc_figures_caption.txt"
    caption_text = """Figure: Dose Effects on Peak, Time-to-Peak, and AUC Metrics

This set of three figures shows dose-dependent effects (High 40mg vs Low 20mg) on 
temporal dynamics of subjective experience dimensions during DMT sessions.

Panel A (peak_dose_comparison.png): Peak intensity values (z-scores) reached during 
the first 9 minutes of DMT sessions. Higher peak values indicate stronger subjective 
experiences on that dimension.

Panel B (time_to_peak_dose_comparison.png): Time in minutes when peak intensity was 
reached. Earlier peaks indicate faster onset of subjective effects.

Panel C (auc_dose_comparison.png): Area under the curve (AUC) from 0-9 minutes, 
representing the cumulative intensity of experience over time. Higher AUC indicates 
sustained elevated experiences.

Visual elements:
- Blue boxes: Low dose (20mg)
- Red boxes: High dose (40mg)
- Gray lines: Paired observations connecting the same subject across doses
- Stars above dimensions: Significant dose effects after FDR correction
  * p < 0.05, ** p < 0.01, *** p < 0.001

Dimensions are ordered by effect size (absolute value of Wilcoxon r statistic) from 
left to right, with strongest dose effects shown first.

Statistical test: Wilcoxon signed-rank test (paired, non-parametric) with 
Benjamini-Hochberg FDR correction across 15 dimensions.
"""

    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text)

    logger.info(f"Saved figure caption to {caption_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Peak/AUC Dose Comparison Boxplots Generated")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Generated {len(output_paths)} figure(s)")
    for metric_type, path in output_paths.items():
        logger.info(f"  - {metric_type}: {path.name}")
    logger.info(f"Caption: {caption_path.name}")


if __name__ == "__main__":
    main()
