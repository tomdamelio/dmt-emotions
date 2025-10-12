# -*- coding: utf-8 -*-
"""
Plot SCR frequency (rate) time course from EmotiPhai SCR detection.

This script analyzes discrete SCR events detected by EmotiPhai algorithm:
- Loads SCR onset times and amplitudes from *_emotiphai_scr.csv files
- Computes SCR count per minute (frequency/rate)
- Plots time course comparing DMT vs RS and High vs Low dose conditions

Usage:
  python test/plot_emotiphai_scr_rate.py
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_EDA,
    get_dosis_sujeto,
)

#############################
# Plot aesthetics (paper-ready minimal style)
#############################
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.titlepad': 8.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Sampling rate for EDA data (Hz)
SAMPLING_RATE = 250.0

# Time window for analysis (seconds)
MAX_TIME = 540.0  # 9 minutes

# Number of minutes
N_MINUTES = 9


def _fmt_mmss(x, pos):
    """Format time axis as mm:ss."""
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


def _beautify_axes(ax, title=None, xlabel=None, ylabel=None, time_formatter=True):
    """Apply consistent styling to axes."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if time_formatter:
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')


def build_emotiphai_scr_path(subject: str, task: str, session: str, dose: str) -> str:
    """
    Build path to EmotiPhai SCR CSV file.
    
    Args:
        subject: Subject ID (e.g., 'S02')
        task: 'dmt' or 'rs'
        session: 'session1' or 'session2'
        dose: 'high' or 'low'
    
    Returns:
        Full path to CSV file
    """
    # All files are in dmt_high and dmt_low directories regardless of task
    dose_dir = f'dmt_{dose}'
    filename = f"{subject}_{task}_{session}_{dose}_emotiphai_scr.csv"
    path = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', dose_dir, filename)
    return path


def load_emotiphai_scr(path: str, fs: float = SAMPLING_RATE) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load EmotiPhai SCR events from CSV.
    
    Args:
        path: Path to CSV file
        fs: Sampling rate in Hz (default 250.0)
    
    Returns:
        (onsets_sec, amplitudes) or None if file empty/missing
    """
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        
        # Check if dataframe is empty or has no valid data
        if df.empty or len(df) == 0:
            print(f"    DEBUG: Empty dataframe from {path}")
            return None
        
        # Check if columns exist
        if 'SCR_Onsets_Emotiphai' not in df.columns or 'SCR_Amplitudes_Emotiphai' not in df.columns:
            print(f"    DEBUG: Missing columns in {path}. Available: {list(df.columns)}")
            return None
        
        onsets = df['SCR_Onsets_Emotiphai'].values
        amplitudes = df['SCR_Amplitudes_Emotiphai'].values
        
        # Filter out NaN values and empty rows
        valid_mask = ~(np.isnan(onsets) | np.isnan(amplitudes))
        onsets = onsets[valid_mask]
        amplitudes = amplitudes[valid_mask]
        
        # Debug: print info about loaded data
        print(f"    DEBUG: Loaded {len(df)} rows, {len(onsets)} valid SCRs from {path}")
        if len(onsets) > 0:
            print(f"    DEBUG: First onset: {onsets[0]}, First amplitude: {amplitudes[0]}")
        
        if len(onsets) == 0:
            print(f"    DEBUG: No valid SCRs after filtering from {path}")
            return None
        
        # Convert timepoints to seconds
        onsets_sec = onsets / fs
        
        return onsets_sec, amplitudes
        
    except Exception as e:
        print(f"⚠️  Error loading {path}: {e}")
        return None


def compute_scr_rate_per_minute(onsets_sec: np.ndarray, max_time: float = MAX_TIME) -> np.ndarray:
    """
    Count SCRs in 1-minute bins.
    
    Args:
        onsets_sec: SCR onset times in seconds
        max_time: Maximum time in seconds (default 540s = 9 min)
    
    Returns:
        Array of length 9 with SCR counts per minute
    """
    n_minutes = int(np.ceil(max_time / 60.0))
    counts = np.zeros(n_minutes)
    
    for onset in onsets_sec:
        if onset < max_time:
            minute_idx = int(onset // 60.0)
            if minute_idx < n_minutes:
                counts[minute_idx] += 1
    
    return counts


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)
    except Exception:
        dose_s1 = 'Alta'
    
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    else:
        return 'session2', 'session1'


def plot_scr_rate_timecourse_combined():
    """
    Create combined summary plot with DMT and RS subplots showing SCR rate over time.
    
    Layout: 2 subplots side-by-side (RS left, DMT right)
    - Mean ± SEM across all validated subjects
    - High vs Low dose comparison
    """
    print("📊 Generating SCR rate time course summary plot...")
    
    # Data collection structure
    task_data = {}
    
    for kind in ['RS', 'DMT']:
        high_rates = []  # List of arrays (one per subject)
        low_rates = []
        
        for subject in SUJETOS_VALIDADOS_EDA:
            try:
                if kind == 'DMT':
                    # DMT task
                    high_session, low_session = determine_sessions(subject)
                    path_high = build_emotiphai_scr_path(subject, 'dmt', high_session, 'high')
                    path_low = build_emotiphai_scr_path(subject, 'dmt', low_session, 'low')
                    print(f"    DEBUG: {subject} DMT paths: {path_high}, {path_low}")
                    
                    data_high = load_emotiphai_scr(path_high)
                    data_low = load_emotiphai_scr(path_low)
                    
                    # Handle empty files as 0 SCRs (valid result)
                    if data_high is None:
                        onsets_high = np.array([])
                    else:
                        onsets_high, _ = data_high
                    
                    if data_low is None:
                        onsets_low = np.array([])
                    else:
                        onsets_low, _ = data_low
                    
                else:  # RS
                    # Resting state - map sessions to doses
                    high_session, low_session = determine_sessions(subject)
                    
                    # Build paths for both sessions
                    path_high = build_emotiphai_scr_path(subject, 'rs', high_session, 'high')
                    path_low = build_emotiphai_scr_path(subject, 'rs', low_session, 'low')
                    print(f"    DEBUG: {subject} RS paths: {path_high}, {path_low}")
                    
                    data_high = load_emotiphai_scr(path_high)
                    data_low = load_emotiphai_scr(path_low)
                    
                    # Handle empty files as 0 SCRs (valid result)
                    if data_high is None:
                        onsets_high = np.array([])
                    else:
                        onsets_high, _ = data_high
                    
                    if data_low is None:
                        onsets_low = np.array([])
                    else:
                        onsets_low, _ = data_low
                
                # Compute SCR rate per minute
                rate_high = compute_scr_rate_per_minute(onsets_high)
                rate_low = compute_scr_rate_per_minute(onsets_low)
                
                # Log SCR counts for debugging
                n_high = len(onsets_high)
                n_low = len(onsets_low)
                print(f"  {subject} {kind}: High={n_high} SCRs, Low={n_low} SCRs")
                
                high_rates.append(rate_high)
                low_rates.append(rate_low)
                
            except Exception as e:
                print(f"⚠️  Error processing {subject} {kind}: {e}")
                continue
        
        if len(high_rates) > 0 and len(low_rates) > 0:
            # Stack into arrays (subjects x minutes)
            H = np.vstack(high_rates)
            L = np.vstack(low_rates)
            
            # Compute mean ± SEM
            mean_h = np.mean(H, axis=0)
            mean_l = np.mean(L, axis=0)
            sem_h = np.std(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
            sem_l = np.std(L, axis=0, ddof=1) / np.sqrt(L.shape[0])
            
            task_data[kind] = {
                'mean_h': mean_h,
                'mean_l': mean_l,
                'sem_h': sem_h,
                'sem_l': sem_l,
                'n': H.shape[0]
            }
            print(f"✅ {kind}: N={H.shape[0]} subjects")
        else:
            print(f"⚠️  No valid data for {kind}")
            return False
    
    if len(task_data) != 2:
        print("❌ Insufficient data for plotting")
        return False
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # Fixed colors per task
    c_dmt_high, c_dmt_low = 'tab:red', 'tab:blue'
    c_rs_high, c_rs_low = 'tab:green', 'tab:purple'
    
    # X-axis: minute centers (0.5, 1.5, ..., 8.5 minutes in seconds)
    x_centers = np.arange(0.5, 9.0, 1.0) * 60.0  # Convert to seconds for plotting
    
    # RS subplot (left)
    rs_data = task_data['RS']
    line_h1 = ax1.plot(x_centers, rs_data['mean_h'], color=c_rs_high, lw=2.0, marker='o', markersize=5, label='High')[0]
    ax1.fill_between(x_centers, rs_data['mean_h'] - rs_data['sem_h'], 
                     rs_data['mean_h'] + rs_data['sem_h'], color=c_rs_high, alpha=0.25)
    line_l1 = ax1.plot(x_centers, rs_data['mean_l'], color=c_rs_low, lw=2.0, marker='o', markersize=5, label='Low')[0]
    ax1.fill_between(x_centers, rs_data['mean_l'] - rs_data['sem_l'], 
                     rs_data['mean_l'] + rs_data['sem_l'], color=c_rs_low, alpha=0.25)
    
    legend1 = ax1.legend([line_h1, line_l1], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=False)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax1, title=None, xlabel='Time (minutes)', ylabel='SCR count per minute')
    ax1.set_title('Resting State (RS)', fontweight='bold')
    
    # DMT subplot (right)
    dmt_data = task_data['DMT']
    line_h2 = ax2.plot(x_centers, dmt_data['mean_h'], color=c_dmt_high, lw=2.0, marker='o', markersize=5, label='High')[0]
    ax2.fill_between(x_centers, dmt_data['mean_h'] - dmt_data['sem_h'], 
                     dmt_data['mean_h'] + dmt_data['sem_h'], color=c_dmt_high, alpha=0.25)
    line_l2 = ax2.plot(x_centers, dmt_data['mean_l'], color=c_dmt_low, lw=2.0, marker='o', markersize=5, label='Low')[0]
    ax2.fill_between(x_centers, dmt_data['mean_l'] - dmt_data['sem_l'], 
                     dmt_data['mean_l'] + dmt_data['sem_l'], color=c_dmt_low, alpha=0.25)
    
    legend2 = ax2.legend([line_h2, line_l2], ['High', 'Low'], loc='upper right', 
                        frameon=True, fancybox=False)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)
    
    _beautify_axes(ax2, title=None, xlabel='Time (minutes)', ylabel='SCR count per minute')
    ax2.set_title('DMT', fontweight='bold')
    
    # X ticks at each minute boundary (0:00, 1:00, ..., 9:00)
    minute_ticks = np.arange(0.0, 541.0, 60.0)
    for ax in (ax1, ax2):
        ax.set_xticks(minute_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
        ax.set_xlim(0.0, 540.0)
        # Y-axis will auto-scale, but ensure it starts above 0
        ax.set_ylim(bottom=0.1)
    
    plt.tight_layout()
    
    # Save figure
    out_dir = os.path.join('test', 'eda', 'emotiphai_scr')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'all_subs_scr_rate_timecourse.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved SCR rate time course: {out_path}")
    
    # Write caption
    write_caption(out_dir, rs_data['n'], dmt_data['n'])
    
    return True


def write_caption(out_dir: str, n_rs: int, n_dmt: int):
    """Write figure caption to text file."""
    caption_file = os.path.join(out_dir, 'captions_emotiphai_scr.txt')
    
    caption = f"""Figure: SCR Frequency Time Course (EmotiPhai Detection)

Group-level mean ± SEM for skin conductance response (SCR) frequency during two tasks.
SCRs were detected using the EmotiPhai algorithm. The plot shows the number of SCRs per minute over 9 minutes.

Left panel: Resting State (RS) with eyes closed, comparing High (green) versus Low (purple) dose conditions (N={n_rs} subjects).
Right panel: DMT task, comparing High (red) versus Low (blue) dose conditions (N={n_dmt} subjects).

The X-axis represents time in minutes (0:00–9:00) with tick marks every minute. Data points are plotted at the center of each 1-minute bin.
The Y-axis displays SCR count per minute (frequency of discrete SCR events).
Shaded regions represent the standard error of the mean (SEM) across subjects.

This analysis quantifies the temporal dynamics of autonomic arousal by counting discrete phasic electrodermal responses, 
providing a complementary measure to the continuous SCR amplitude time series. Higher SCR frequency indicates 
increased sympathetic nervous system activity and arousal.
"""
    
    try:
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
        print(f"📝 Caption written: {caption_file}")
    except Exception as e:
        print(f"⚠️  Failed to write caption: {e}")


def plot_dmt_scr_rate_19min():
    """
    Create DMT-only plot showing SCR rate over 19 minutes.
    
    Shows High vs Low dose comparison for DMT task across the full 19-minute recording.
    """
    print("📊 Generating DMT SCR rate time course (19 minutes)...")
    
    # Extended time window for 19 minutes
    MAX_TIME_19MIN = 1140.0  # 19 minutes in seconds
    N_MINUTES_19MIN = 19
    
    # Data collection for DMT only
    high_rates = []  # List of arrays (one per subject)
    low_rates = []
    
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            # DMT task
            high_session, low_session = determine_sessions(subject)
            path_high = build_emotiphai_scr_path(subject, 'dmt', high_session, 'high')
            path_low = build_emotiphai_scr_path(subject, 'dmt', low_session, 'low')
            
            data_high = load_emotiphai_scr(path_high)
            data_low = load_emotiphai_scr(path_low)
            
            # Handle empty files as 0 SCRs (valid result)
            if data_high is None:
                onsets_high = np.array([])
            else:
                onsets_high, _ = data_high
            
            if data_low is None:
                onsets_low = np.array([])
            else:
                onsets_low, _ = data_low
            
            # Compute SCR rate per minute for 19 minutes
            rate_high = compute_scr_rate_per_minute(onsets_high, MAX_TIME_19MIN)
            rate_low = compute_scr_rate_per_minute(onsets_low, MAX_TIME_19MIN)
            
            # Log SCR counts for debugging
            n_high = len(onsets_high)
            n_low = len(onsets_low)
            print(f"  {subject} DMT: High={n_high} SCRs, Low={n_low} SCRs")
            
            high_rates.append(rate_high)
            low_rates.append(rate_low)
            
        except Exception as e:
            print(f"⚠️  Error processing {subject} DMT: {e}")
            continue
    
    if len(high_rates) == 0 or len(low_rates) == 0:
        print("⚠️  No valid DMT data for 19-minute plot")
        return False
    
    # Stack into arrays (subjects x minutes)
    H = np.vstack(high_rates)
    L = np.vstack(low_rates)
    
    # Compute mean ± SEM
    mean_h = np.mean(H, axis=0)
    mean_l = np.mean(L, axis=0)
    sem_h = np.std(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.std(L, axis=0, ddof=1) / np.sqrt(L.shape[0])
    
    print(f"✅ DMT 19-min: N={H.shape[0]} subjects")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Fixed colors for DMT
    c_dmt_high, c_dmt_low = 'tab:red', 'tab:blue'
    
    # X-axis: minute centers (0.5, 1.5, ..., 18.5 minutes in seconds)
    x_centers = np.arange(0.5, 19.0, 1.0) * 60.0  # Convert to seconds for plotting
    
    # Plot DMT data
    line_h = ax.plot(x_centers, mean_h, color=c_dmt_high, lw=2.0, marker='o', markersize=4, label='High')[0]
    ax.fill_between(x_centers, mean_h - sem_h, mean_h + sem_h, color=c_dmt_high, alpha=0.25)
    line_l = ax.plot(x_centers, mean_l, color=c_dmt_low, lw=2.0, marker='o', markersize=4, label='Low')[0]
    ax.fill_between(x_centers, mean_l - sem_l, mean_l + sem_l, color=c_dmt_low, alpha=0.25)
    
    # Legend
    legend = ax.legend([line_h, line_l], ['High', 'Low'], loc='upper right', 
                      frameon=True, fancybox=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Beautify axes
    _beautify_axes(ax, title=None, 
                   xlabel='Time (minutes)', ylabel='SCR count per minute')
    
    # X ticks at each minute boundary (0:00, 1:00, ..., 19:00)
    minute_ticks = np.arange(0.0, 1141.0, 60.0)
    ax.set_xticks(minute_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.set_xlim(0.0, 1140.0)
    # Y-axis will auto-scale, but ensure it starts above 0
    ax.set_ylim(bottom=0.1)
    
    plt.tight_layout()
    
    # Save figure
    out_dir = os.path.join('test', 'eda', 'emotiphai_scr')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dmt_scr_rate_19min_timecourse.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved DMT SCR rate 19-min time course: {out_path}")
    
    # Write caption
    write_dmt_19min_caption(out_dir, H.shape[0])
    
    return True


def write_dmt_19min_caption(out_dir: str, n_dmt: int):
    """Write figure caption to text file for DMT 19-min plot."""
    caption_file = os.path.join(out_dir, 'captions_dmt_scr_19min.txt')
    
    caption = f"""Figure: DMT SCR Frequency Time Course (19 Minutes)

Group-level mean ± SEM for skin conductance response (SCR) frequency during DMT task.
SCRs were detected using the EmotiPhai algorithm. The plot shows the number of SCRs per minute over 19 minutes.

The plot compares High (red) versus Low (blue) dose conditions for the DMT task (N={n_dmt} subjects).
The X-axis represents time in minutes (0:00–19:00) with tick marks every minute. Data points are plotted at the center of each 1-minute bin.
The Y-axis displays SCR count per minute (frequency of discrete SCR events).
Shaded regions represent the standard error of the mean (SEM) across subjects.

This extended time course analysis reveals the temporal dynamics of autonomic arousal throughout the full DMT session, 
providing insight into how SCR frequency evolves over the complete 19-minute recording period. Higher SCR frequency 
indicates increased sympathetic nervous system activity and emotional arousal during the DMT experience.
"""
    
    try:
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
        print(f"📝 DMT 19-min caption written: {caption_file}")
    except Exception as e:
        print(f"⚠️  Failed to write DMT 19-min caption: {e}")


def collect_scr_counts():
    """
    Collect SCR counts for all subjects and conditions.
    
    Returns:
        Dictionary with structure: {task: {dose: [counts]}}
    """
    print("📊 Collecting SCR counts for all subjects...")
    
    data = {
        'RS': {'high': [], 'low': []},
        'DMT': {'high': [], 'low': []}
    }
    
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            
            # DMT data
            dmt_high_path = build_emotiphai_scr_path(subject, 'dmt', high_session, 'high')
            dmt_low_path = build_emotiphai_scr_path(subject, 'dmt', low_session, 'low')
            
            dmt_high_count = load_emotiphai_scr_count(dmt_high_path)
            dmt_low_count = load_emotiphai_scr_count(dmt_low_path)
            
            if dmt_high_count is not None and dmt_low_count is not None:
                data['DMT']['high'].append(dmt_high_count)
                data['DMT']['low'].append(dmt_low_count)
                print(f"  {subject} DMT: High={dmt_high_count}, Low={dmt_low_count}")
            else:
                print(f"⚠️  Skipping {subject} DMT: missing data")
            
            # RS data
            rs_high_path = build_emotiphai_scr_path(subject, 'rs', high_session, 'high')
            rs_low_path = build_emotiphai_scr_path(subject, 'rs', low_session, 'low')
            
            rs_high_count = load_emotiphai_scr_count(rs_high_path)
            rs_low_count = load_emotiphai_scr_count(rs_low_path)
            
            if rs_high_count is not None and rs_low_count is not None:
                data['RS']['high'].append(rs_high_count)
                data['RS']['low'].append(rs_low_count)
                print(f"  {subject} RS: High={rs_high_count}, Low={rs_low_count}")
            else:
                print(f"⚠️  Skipping {subject} RS: missing data")
                
        except Exception as e:
            print(f"⚠️  Error processing {subject}: {e}")
            continue
    
    # Convert to numpy arrays
    for task in data:
        for dose in data[task]:
            data[task][dose] = np.array(data[task][dose])
    
    return data




def load_emotiphai_scr_count(path: str, fs: float = SAMPLING_RATE) -> Optional[int]:
    """
    Load EmotiPhai SCR events from CSV and return count.
    
    Args:
        path: Path to CSV file
        fs: Sampling rate in Hz (default 250.0)
    
    Returns:
        Number of SCRs or None if file empty/missing
    """
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        
        # Check if dataframe is empty or has no valid data
        if df.empty or len(df) == 0:
            return 0  # Empty file = 0 SCRs
        
        # Check if columns exist
        if 'SCR_Onsets_Emotiphai' not in df.columns or 'SCR_Amplitudes_Emotiphai' not in df.columns:
            return None
        
        onsets = df['SCR_Onsets_Emotiphai'].values
        amplitudes = df['SCR_Amplitudes_Emotiphai'].values
        
        # Filter out NaN values and empty rows
        valid_mask = ~(np.isnan(onsets) | np.isnan(amplitudes))
        onsets = onsets[valid_mask]
        amplitudes = amplitudes[valid_mask]
        
        # Count valid SCRs
        scr_count = len(onsets)
        
        return scr_count
        
    except Exception as e:
        print(f"⚠️  Error loading {path}: {e}")
        return None




def plot_scr_count_boxplot(data):
    """
    Create boxplot with individual points and jitter for SCR counts.
    
    Args:
        data: Dictionary with SCR counts by task and dose
    """
    print("📊 Creating SCR count boxplot with individual points...")
    
    # Prepare data for plotting
    rs_high = data['RS']['high']
    rs_low = data['RS']['low']
    dmt_high = data['DMT']['high']
    dmt_low = data['DMT']['low']
    
    print(f"RS: High={np.mean(rs_high):.1f}±{np.std(rs_high, ddof=1)/np.sqrt(len(rs_high)):.1f} (N={len(rs_high)}), Low={np.mean(rs_low):.1f}±{np.std(rs_low, ddof=1)/np.sqrt(len(rs_low)):.1f} (N={len(rs_low)})")
    print(f"DMT: High={np.mean(dmt_high):.1f}±{np.std(dmt_high, ddof=1)/np.sqrt(len(dmt_high)):.1f} (N={len(dmt_high)}), Low={np.mean(dmt_low):.1f}±{np.std(dmt_low, ddof=1)/np.sqrt(len(dmt_low)):.1f} (N={len(dmt_low)})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define positions and data
    x_pos = np.array([0.8, 1.2, 1.8, 2.2])  # Positions for 4 boxes
    data_arrays = [rs_low, rs_high, dmt_low, dmt_high]  # Inverted order: low first, then high
    
    # Colors (same as time course plots) - ORDER MUST MATCH data_arrays
    # data_arrays = [rs_low, rs_high, dmt_low, dmt_high]
    # colors =     [purple, green,   blue,    red]
    colors = ['tab:purple', 'tab:green', 'tab:blue', 'tab:red']
    
    # Create boxplots
    bp = ax.boxplot(data_arrays, positions=x_pos, widths=0.3, patch_artist=True,
                    showfliers=False)
    
    # Color the boxes, edges, medians, and whiskers
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)  # Match edge color to fill color
        patch.set_linewidth(1.5)
    
    # Color medians
    for median, color in zip(bp['medians'], colors):
        median.set_color(color)
        median.set_linewidth(1.5)
    
    # Color whiskers - each box has 2 whiskers (upper and lower)
    whisker_colors = []
    for color in colors:
        whisker_colors.extend([color, color])  # Add same color twice for upper and lower whisker
    
    for whisker, color in zip(bp['whiskers'], whisker_colors):
        whisker.set_color(color)
        whisker.set_linewidth(1.5)
    
    # Color caps (whisker ends) - each box has 2 caps (upper and lower)
    cap_colors = []
    for color in colors:
        cap_colors.extend([color, color])  # Add same color twice for upper and lower cap
    
    for cap, color in zip(bp['caps'], cap_colors):
        cap.set_color(color)
        cap.set_linewidth(1.5)
    
    # Add individual points with jitter and connecting lines
    np.random.seed(42)  # For reproducible jitter
    jitter_amount = 0.05
    
    # Store jittered positions for connecting lines
    jittered_positions = []
    
    for i, (data_array, pos, color) in enumerate(zip(data_arrays, x_pos, colors)):
        # Add jitter to x-coordinates
        jitter = np.random.normal(0, jitter_amount, len(data_array))
        x_jittered = pos + jitter
        jittered_positions.append(x_jittered)
        
        # Plot individual points
        ax.scatter(x_jittered, data_array, color=color, alpha=0.7, s=30, 
                  edgecolors=color, linewidth=0.5, zorder=3)
    
    # Connect points for same subjects within each condition
    # RS: connect low and high dose points for each subject (new order: low first, then high)
    for i in range(len(rs_high)):
        ax.plot([jittered_positions[0][i], jittered_positions[1][i]], 
                [rs_low[i], rs_high[i]], 
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # DMT: connect low and high dose points for each subject (new order: low first, then high)
    for i in range(len(dmt_high)):
        ax.plot([jittered_positions[2][i], jittered_positions[3][i]], 
                [dmt_low[i], dmt_high[i]], 
                color='gray', alpha=0.3, linewidth=0.8, zorder=1)
    
    # Customize plot
    ax.set_xlim(0.4, 2.6)
    ax.set_xticks([0.8, 1.2, 1.8, 2.2])  # Align with individual boxplots
    # Labels must match data order: [rs_low, rs_high, dmt_low, dmt_high]
    ax.set_xticklabels(['Low', 'High', 'Low', 'High'])
    ax.set_ylabel('SCR count per minute')
    
    # Add sub-labels for conditions (Resting State and DMT)
    ax.text(1.0, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
            'Resting State (RS)', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(2.0, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
            'DMT', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Grid - only horizontal lines
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')  # Remove vertical grid lines
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    out_dir = os.path.join('test', 'eda', 'emotiphai_scr')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'scr_count_boxplot_comparison.png')
    fig.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Saved SCR count boxplot: {out_path}")
    
    # Write caption
    write_boxplot_caption(out_dir, len(rs_high), len(dmt_high))
    
    return True


def write_boxplot_caption(out_dir: str, n_rs: int, n_dmt: int):
    """Write figure caption to text file."""
    caption_file = os.path.join(out_dir, 'captions_scr_count_boxplot.txt')
    
    caption = f"""Figure: SCR Count Distribution (Boxplot with Individual Points)

Boxplot showing the distribution of skin conductance response (SCR) counts across tasks and doses.
SCRs were detected using the EmotiPhai algorithm and counted over 9 minutes.

The plot shows four conditions:
- Resting State (RS) High Dose (green, N={n_rs})
- Resting State (RS) Low Dose (purple, N={n_rs})  
- DMT High Dose (red, N={n_dmt})
- DMT Low Dose (blue, N={n_dmt})

Boxplots show median (line), quartiles (box), and range (whiskers). Individual points represent 
each subject's SCR count with horizontal jitter for visibility. Gray lines connect the same 
subject's high and low dose measurements within each task condition, showing individual 
within-subject dose effects. The Y-axis displays the total number of SCRs detected during 
the 9-minute recording period.

This analysis reveals the distribution and variability of autonomic arousal levels by counting 
discrete phasic electrodermal responses. Higher SCR counts indicate increased sympathetic nervous 
system activity and emotional arousal.
"""
    
    try:
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
        print(f"📝 Boxplot caption written: {caption_file}")
    except Exception as e:
        print(f"⚠️  Failed to write boxplot caption: {e}")




def main():
    """Main execution function."""
    print("🚀 Starting EmotiPhai SCR analysis...")
    print(f"Validated subjects: {len(SUJETOS_VALIDADOS_EDA)}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print(f"Time window: {MAX_TIME}s ({N_MINUTES} minutes)")
    print()
    
    # Generate time course plot (9 minutes, RS + DMT)
    print("=" * 60)
    print("1. SCR Rate Time Course (9 min, RS + DMT)")
    print("=" * 60)
    success1 = plot_scr_rate_timecourse_combined()
    
    # Generate DMT-only 19-minute plot
    print("\n" + "=" * 60)
    print("2. DMT SCR Rate Time Course (19 min)")
    print("=" * 60)
    success2 = plot_dmt_scr_rate_19min()
    
    # Generate boxplot
    print("\n" + "=" * 60)
    print("3. SCR Count Boxplot")
    print("=" * 60)
    
    # Collect data for boxplot
    count_data = collect_scr_counts()
    
    # Check if we have enough data for boxplot
    min_samples = min(len(count_data['RS']['high']), len(count_data['RS']['low']), 
                     len(count_data['DMT']['high']), len(count_data['DMT']['low']))
    
    if min_samples < 3:
        print(f"❌ Insufficient data for boxplot (min samples: {min_samples})")
        success3 = False
    else:
        success3 = plot_scr_count_boxplot(count_data)
    
    if success1 and success2 and success3:
        print("\n✅ SCR analysis completed successfully!")
    else:
        print("\n⚠️  SCR analysis completed with some issues")
    
    return success1 and success2 and success3


if __name__ == '__main__':
    main()

