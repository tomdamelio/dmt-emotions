"""
Feature Extraction Module for DMT Physiological Analysis

This module provides functions for extracting temporal features from physiological
time series data, including peak amplitude, time-to-peak, and threshold crossing times.

Scientific Rationale:
- Peak amplitude reflects the magnitude of physiological response
- Time-to-peak reflects the onset speed of the response
- Threshold crossings (33%, 50%) characterize the dose-response relationship
- These features capture aspects of temporal dynamics not visible in time-aligned comparisons

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def extract_peak_amplitude(
    t: np.ndarray,
    signal: np.ndarray,
    window_start_sec: float = 0.0,
    window_end_sec: float = 540.0,
) -> Dict[str, float]:
    """
    Extract peak amplitude within specified window.

    This function identifies the maximum value in the signal and the time at which
    it occurs. This captures the magnitude of the physiological response.

    Args:
        t: Time array in seconds (monotonically increasing)
        signal: Signal values corresponding to time points
        window_start_sec: Start of analysis window in seconds (default 0)
        window_end_sec: End of analysis window in seconds (default 540 = 9 minutes)

    Returns:
        Dictionary with keys:
            - 'peak_amplitude': Maximum value in window
            - 'peak_time_sec': Time at which maximum occurs (in seconds)

    Raises:
        ValueError: If time array is not monotonically increasing
        ValueError: If window boundaries are invalid
        ValueError: If no data points exist within window

    Scientific Reference:
        Peak amplitude is a standard feature in pharmacokinetic analysis,
        reflecting the maximum effect magnitude (Cmax analog).

    Requirements: 3.1
    """
    # Validate inputs
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time array must be monotonically increasing")

    if window_start_sec >= window_end_sec:
        raise ValueError(
            f"Invalid window: start ({window_start_sec}) must be < end ({window_end_sec})"
        )

    # Filter to window
    mask = (t >= window_start_sec) & (t <= window_end_sec)
    if not np.any(mask):
        raise ValueError(
            f"No data points in window [{window_start_sec}, {window_end_sec}]"
        )

    t_window = t[mask]
    signal_window = signal[mask]

    # Handle NaN values
    valid_mask = ~np.isnan(signal_window)
    if not np.any(valid_mask):
        return {"peak_amplitude": np.nan, "peak_time_sec": np.nan}

    t_valid = t_window[valid_mask]
    signal_valid = signal_window[valid_mask]

    # Find peak
    peak_idx = np.argmax(signal_valid)
    peak_amplitude = signal_valid[peak_idx]
    peak_time_sec = t_valid[peak_idx]

    return {"peak_amplitude": float(peak_amplitude), "peak_time_sec": float(peak_time_sec)}


def extract_time_to_peak(
    t: np.ndarray,
    signal: np.ndarray,
    window_start_sec: float = 0.0,
    window_end_sec: float = 540.0,
) -> float:
    """
    Extract time-to-peak (in minutes post-t₀).

    This function identifies when the signal reaches its maximum value,
    reflecting the onset speed of the physiological response.

    Args:
        t: Time array in seconds (monotonically increasing)
        signal: Signal values corresponding to time points
        window_start_sec: Start of analysis window in seconds (default 0)
        window_end_sec: End of analysis window in seconds (default 540 = 9 minutes)

    Returns:
        Time to peak in minutes from window start

    Raises:
        ValueError: If time array is not monotonically increasing
        ValueError: If window boundaries are invalid
        ValueError: If no data points exist within window

    Scientific Reference:
        Time-to-peak (Tmax) is a standard pharmacokinetic parameter reflecting
        the speed of drug absorption and onset of effects.

    Requirements: 3.2
    """
    result = extract_peak_amplitude(t, signal, window_start_sec, window_end_sec)

    if np.isnan(result["peak_time_sec"]):
        return np.nan

    # Convert to minutes from window start
    time_to_peak_min = (result["peak_time_sec"] - window_start_sec) / 60.0

    return float(time_to_peak_min)


def extract_threshold_crossings(
    t: np.ndarray,
    signal: np.ndarray,
    thresholds: List[float] = [0.33, 0.50],
    window_start_sec: float = 0.0,
    window_end_sec: float = 540.0,
) -> Dict[str, float]:
    """
    Extract times when signal crosses specified thresholds (as fraction of max).

    This function identifies when the signal first crosses threshold values
    (e.g., 33% and 50% of maximum), characterizing the dose-response relationship
    and temporal dynamics of the response.

    Args:
        t: Time array in seconds (monotonically increasing)
        signal: Signal values corresponding to time points
        thresholds: List of threshold fractions (e.g., [0.33, 0.50] for 33% and 50%)
        window_start_sec: Start of analysis window in seconds (default 0)
        window_end_sec: End of analysis window in seconds (default 540 = 9 minutes)

    Returns:
        Dictionary mapping threshold to crossing time in minutes from window start
        (e.g., {'t_33': 1.2, 't_50': 2.5}). Returns NaN if threshold not crossed.

    Raises:
        ValueError: If time array is not monotonically increasing
        ValueError: If window boundaries are invalid
        ValueError: If threshold values are not in (0, 1)
        ValueError: If no data points exist within window

    Scientific Reference:
        Threshold crossing times characterize the temporal profile of drug effects
        and are used in dose-response modeling.

    Requirements: 3.3
    """
    # Validate thresholds
    if not all(0 < th < 1 for th in thresholds):
        raise ValueError("All threshold values must be in range (0, 1)")

    # Get peak amplitude
    peak_result = extract_peak_amplitude(t, signal, window_start_sec, window_end_sec)

    if np.isnan(peak_result["peak_amplitude"]):
        # Return NaN for all thresholds
        return {f"t_{int(th*100)}": np.nan for th in thresholds}

    peak_amplitude = peak_result["peak_amplitude"]

    # Filter to window
    mask = (t >= window_start_sec) & (t <= window_end_sec)
    t_window = t[mask]
    signal_window = signal[mask]

    # Handle NaN values
    valid_mask = ~np.isnan(signal_window)
    t_valid = t_window[valid_mask]
    signal_valid = signal_window[valid_mask]

    # Find threshold crossings
    crossing_times = {}

    for threshold in thresholds:
        threshold_value = threshold * peak_amplitude
        threshold_key = f"t_{int(threshold*100)}"

        # Find first crossing
        crossing_mask = signal_valid >= threshold_value

        if np.any(crossing_mask):
            # Get first index where threshold is crossed
            first_crossing_idx = np.argmax(crossing_mask)
            crossing_time_sec = t_valid[first_crossing_idx]

            # Convert to minutes from window start
            crossing_time_min = (crossing_time_sec - window_start_sec) / 60.0
            crossing_times[threshold_key] = float(crossing_time_min)
        else:
            # Threshold never crossed
            crossing_times[threshold_key] = np.nan

    return crossing_times


def extract_all_features(
    df: pd.DataFrame,
    time_column: str = "time",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Extract all temporal features for each subject and session.

    This function processes a long-format dataframe and extracts peak amplitude,
    time-to-peak, and threshold crossing times for each unique combination of
    subject, session, State, and Dose.

    Args:
        df: Long-format dataframe with columns:
            - subject: Subject identifier
            - session: Session identifier
            - State: Condition ('RS' or 'DMT')
            - Dose: Dose level ('Low' or 'High')
            - time_column: Time values in seconds
            - value_column: Signal values
        time_column: Name of time column (default 'time')
        value_column: Name of signal value column (default 'value')

    Returns:
        DataFrame with columns:
            - subject: Subject identifier
            - session: Session identifier
            - State: Condition
            - Dose: Dose level
            - peak_amplitude: Maximum value in window
            - time_to_peak: Time to peak in minutes
            - t_33: Time to 33% threshold in minutes
            - t_50: Time to 50% threshold in minutes

    Raises:
        ValueError: If required columns are missing
        ValueError: If dataframe is empty

    Scientific Rationale:
        Batch feature extraction enables systematic comparison of temporal dynamics
        across conditions and doses using paired statistical tests.

    Requirements: 3.4, 3.5
    """
    # Validate inputs
    required_cols = ["subject", "session", "State", "Dose", time_column, value_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Input dataframe is empty")

    # Group by subject, session, State, Dose
    grouping_cols = ["subject", "session", "State", "Dose"]

    features_list = []

    for group_keys, group_df in df.groupby(grouping_cols):
        # Extract time and signal arrays
        t = group_df[time_column].values
        signal = group_df[value_column].values

        # Sort by time (in case not already sorted)
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        signal = signal[sort_idx]

        try:
            # Extract peak amplitude and time
            peak_result = extract_peak_amplitude(t, signal)

            # Extract time-to-peak
            ttp = extract_time_to_peak(t, signal)

            # Extract threshold crossings
            threshold_result = extract_threshold_crossings(t, signal)

            # Combine results
            feature_dict = {
                "subject": group_keys[0],
                "session": group_keys[1],
                "State": group_keys[2],
                "Dose": group_keys[3],
                "peak_amplitude": peak_result["peak_amplitude"],
                "time_to_peak": ttp,
                "t_33": threshold_result["t_33"],
                "t_50": threshold_result["t_50"],
            }

            features_list.append(feature_dict)

        except Exception as e:
            # Log warning and continue with NaN values
            print(
                f"Warning: Feature extraction failed for {group_keys}: {str(e)}"
            )
            feature_dict = {
                "subject": group_keys[0],
                "session": group_keys[1],
                "State": group_keys[2],
                "Dose": group_keys[3],
                "peak_amplitude": np.nan,
                "time_to_peak": np.nan,
                "t_33": np.nan,
                "t_50": np.nan,
            }
            features_list.append(feature_dict)

    # Convert to dataframe
    features_df = pd.DataFrame(features_list)

    return features_df
