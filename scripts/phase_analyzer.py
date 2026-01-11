"""
Temporal Phase Analysis Module

This module provides functions for analyzing physiological signals averaged within
distinct temporal phases. Phase-based analysis can detect dose differences that may
not appear in time-to-time comparisons due to temporal misalignment.

Scientific Rationale:
- Temporal misalignment may obscure dose differences in pointwise comparisons
- Phase averaging reduces noise and captures overall trajectory
- Flexible boundaries allow sensitivity analyses
- Aligns with pharmacokinetic profile of inhaled DMT (rapid onset, gradual recovery)
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def define_temporal_phases(
    total_duration_sec: int = 540, phase_boundaries: List[int] = None
) -> List[Tuple[int, int]]:
    """
    Define temporal phases for analysis.

    Args:
        total_duration_sec: Total duration in seconds (default 540 = 9 minutes)
        phase_boundaries: List of phase boundaries in seconds
                         (default [0, 180, 540] = onset and recovery phases)

    Returns:
        List of (start_sec, end_sec) tuples defining each phase

    Example:
        >>> phases = define_temporal_phases()
        >>> phases
        [(0, 180), (180, 540)]
    """
    if phase_boundaries is None:
        phase_boundaries = [0, 180, 540]

    # Validate boundaries
    if not phase_boundaries:
        raise ValueError("phase_boundaries cannot be empty")

    if phase_boundaries[0] != 0:
        raise ValueError("First phase boundary must be 0")

    if phase_boundaries[-1] != total_duration_sec:
        raise ValueError(
            f"Last phase boundary must equal total_duration_sec ({total_duration_sec})"
        )

    if not all(
        phase_boundaries[i] < phase_boundaries[i + 1]
        for i in range(len(phase_boundaries) - 1)
    ):
        raise ValueError("Phase boundaries must be strictly increasing")

    # Create phase tuples
    phases = [
        (phase_boundaries[i], phase_boundaries[i + 1])
        for i in range(len(phase_boundaries) - 1)
    ]

    return phases


def compute_phase_averages(
    df: pd.DataFrame,
    phases: List[Tuple[int, int]],
    value_column: str = "value",
    window_size_sec: int = 30,
) -> pd.DataFrame:
    """
    Compute phase-averaged signals for each participant.

    Args:
        df: Long-format dataframe with columns: subject, window, State, Dose, value_column
        phases: List of (start_sec, end_sec) tuples from define_temporal_phases()
        value_column: Name of column containing signal values
        window_size_sec: Size of each window in seconds (default 30)

    Returns:
        DataFrame with columns: subject, phase, phase_label, State, Dose,
                               mean_value, sem_value, start_sec, end_sec

    Example:
        >>> phases = define_temporal_phases()
        >>> phase_df = compute_phase_averages(df, phases, value_column='hr_z')
    """
    # Validate input
    required_columns = ["subject", "window", "State", "Dose", value_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert window index to time in seconds
    # Assuming window 1 = 0-30s, window 2 = 30-60s, etc.
    df = df.copy()
    df["time_sec"] = (df["window"] - 1) * window_size_sec

    phase_results = []

    for phase_idx, (start_sec, end_sec) in enumerate(phases):
        # Filter data within this phase
        phase_mask = (df["time_sec"] >= start_sec) & (df["time_sec"] < end_sec)
        phase_data = df[phase_mask].copy()

        if len(phase_data) == 0:
            continue

        # Compute mean and SEM for each subject/State/Dose combination
        grouped = (
            phase_data.groupby(["subject", "State", "Dose"])[value_column]
            .agg(["mean", "sem"])
            .reset_index()
        )

        grouped["phase"] = phase_idx
        grouped["phase_label"] = f"Phase_{phase_idx}_({start_sec}-{end_sec}s)"
        grouped["start_sec"] = start_sec
        grouped["end_sec"] = end_sec
        grouped.rename(
            columns={"mean": "mean_value", "sem": "sem_value"}, inplace=True
        )

        phase_results.append(grouped)

    if not phase_results:
        raise ValueError("No data found in any phase")

    result_df = pd.concat(phase_results, ignore_index=True)

    return result_df


def compare_doses_within_phases(
    phase_df: pd.DataFrame, value_column: str = "mean_value"
) -> pd.DataFrame:
    """
    Perform paired t-tests comparing High vs Low dose within each phase.

    Args:
        phase_df: DataFrame from compute_phase_averages()
        value_column: Column containing phase-averaged values

    Returns:
        DataFrame with columns: phase, phase_label, State, t_stat, df, p_value,
                               cohens_d, mean_high, mean_low, sem_high, sem_low,
                               start_sec, end_sec

    Example:
        >>> comparison_df = compare_doses_within_phases(phase_df)
        >>> significant = comparison_df[comparison_df['p_value'] < 0.05]
    """
    # Validate input
    required_columns = [
        "subject",
        "phase",
        "phase_label",
        "State",
        "Dose",
        value_column,
        "start_sec",
        "end_sec",
    ]
    missing_columns = [col for col in required_columns if col not in phase_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Filter to DMT state only (paired comparison requires same state)
    dmt_data = phase_df[phase_df["State"] == "DMT"].copy()

    if len(dmt_data) == 0:
        raise ValueError("No DMT data found for comparison")

    results = []

    # Group by phase
    for phase_idx in dmt_data["phase"].unique():
        phase_data = dmt_data[dmt_data["phase"] == phase_idx]

        # Get phase metadata
        phase_label = phase_data["phase_label"].iloc[0]
        start_sec = phase_data["start_sec"].iloc[0]
        end_sec = phase_data["end_sec"].iloc[0]

        # Pivot to get High and Low doses for each subject
        pivot_data = phase_data.pivot_table(
            index="subject", columns="Dose", values=value_column
        )

        # Check if both High and Low doses exist
        if "High" not in pivot_data.columns or "Low" not in pivot_data.columns:
            continue

        # Remove subjects with missing data
        complete_data = pivot_data.dropna()

        if len(complete_data) < 2:
            # Need at least 2 subjects for paired t-test
            continue

        high_values = complete_data["High"].values
        low_values = complete_data["Low"].values

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(high_values, low_values)

        # Compute Cohen's d for paired samples
        # d = mean_diff / std_diff
        diff = high_values - low_values
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)

        # Compute descriptive statistics
        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)
        sem_high = stats.sem(high_values)
        sem_low = stats.sem(low_values)

        results.append(
            {
                "phase": phase_idx,
                "phase_label": phase_label,
                "State": "DMT",
                "t_stat": t_stat,
                "df": len(complete_data) - 1,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "mean_high": mean_high,
                "mean_low": mean_low,
                "sem_high": sem_high,
                "sem_low": sem_low,
                "n_subjects": len(complete_data),
                "start_sec": start_sec,
                "end_sec": end_sec,
            }
        )

    if not results:
        raise ValueError("No valid phase comparisons could be computed")

    result_df = pd.DataFrame(results)

    return result_df
