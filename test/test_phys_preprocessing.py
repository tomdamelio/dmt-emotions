# -*- coding: utf-8 -*-
"""
Exploratory testing of physiological preprocessing results (EDA, ECG, RESP)

This script tests the physiological signals extracted by preprocess_phys.py for the test subject (S04).
It loads the processed CSV files, removes the time column, and generates signal plots 
to verify signal quality and processing correctness for all three modalities.

FEATURES:
- Tests EDA, ECG, and RESP signals from preprocess_phys.py output
- Duration and sampling rate validation for all signal types
- Detection of sampling rate mismatches between config and actual data
- Analysis of time series consistency
- Detailed reporting of duration discrepancies per signal type
- Signal-specific plotting using appropriate NeuroKit functions

This helps identify issues like:
- Incorrect sampling rate assumptions in preprocessing
- Duration truncation/padding problems
- Time axis discrepancies in plots
- Signal processing failures per modality

Usage:
1. First run preprocess_phys.py in TEST_MODE to generate S04 data
2. Then run this script to visualize and validate the results
3. Check the DURATION & SAMPLING RATE ANALYSIS section for any issues
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from pathlib import Path

# Add parent directory to path to import from scripts and config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import DERIVATIVES_DATA, SUJETOS_TEST, TEST_MODE, NEUROKIT_PARAMS, DURACIONES_ESPERADAS
except ImportError:
    print("❌ Could not import config. Make sure config.py exists in parent directory.")
    sys.exit(1)

def test_duration_and_sampling_rate(signal_df, time_series, expected_duration_sec, file_description):
    """
    Test if the actual duration and sampling rate match expectations
    
    Args:
        signal_df: DataFrame with physiological signal data (EDA, ECG, or RESP)
        time_series: Time series data
        expected_duration_sec: Expected duration in seconds
        file_description: Description of the file being tested
        
    Returns:
        dict: Test results with sampling rate analysis
    """
    print(f"\n🔍 Testing duration and sampling rate for: {file_description}")
    
    # Calculate actual duration from samples using config sampling rate
    config_sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
    actual_samples = len(signal_df)
    calculated_duration_from_samples = actual_samples / config_sampling_rate
    
    # Calculate actual duration from time series (if available)
    if time_series is not None and len(time_series) > 0:
        time_duration = time_series.iloc[-1] - time_series.iloc[0]
        # Calculate implied sampling rate from time series
        implied_sampling_rate = (len(time_series) - 1) / time_duration if time_duration > 0 else 0
    else:
        time_duration = None
        implied_sampling_rate = None
    
    # Expected values
    expected_duration_min = expected_duration_sec / 60
    expected_samples = expected_duration_sec * config_sampling_rate
    
    # Test results
    results = {
        'file': file_description,
        'expected_duration_sec': expected_duration_sec,
        'expected_duration_min': expected_duration_min,
        'expected_samples': expected_samples,
        'actual_samples': actual_samples,
        'config_sampling_rate': config_sampling_rate,
        'calculated_duration_from_samples_sec': calculated_duration_from_samples,
        'calculated_duration_from_samples_min': calculated_duration_from_samples / 60,
        'time_series_duration_sec': time_duration,
        'time_series_duration_min': time_duration / 60 if time_duration else None,
        'implied_sampling_rate': implied_sampling_rate,
        'samples_match_expected': abs(actual_samples - expected_samples) < 10,  # Allow small tolerance
        'duration_discrepancy_sec': abs(calculated_duration_from_samples - expected_duration_sec),
        'sampling_rate_discrepancy': abs(implied_sampling_rate - config_sampling_rate) if implied_sampling_rate else None
    }
    
    # Print detailed analysis
    print(f"   📊 Expected: {expected_duration_min:.2f} min ({expected_duration_sec} sec, {expected_samples} samples)")
    print(f"   📊 Actual samples: {actual_samples}")
    print(f"   📊 Config sampling rate: {config_sampling_rate} Hz")
    print(f"   📊 Duration from samples (using config SR): {calculated_duration_from_samples:.2f} sec ({calculated_duration_from_samples/60:.2f} min)")
    
    if time_duration is not None:
        print(f"   📊 Duration from time series: {time_duration:.2f} sec ({time_duration/60:.2f} min)")
        if implied_sampling_rate:
            print(f"   📊 Implied sampling rate from time: {implied_sampling_rate:.2f} Hz")
            
            # Check for sampling rate mismatch
            if results['sampling_rate_discrepancy'] and results['sampling_rate_discrepancy'] > 1:
                print(f"   ⚠️  SAMPLING RATE MISMATCH! Config: {config_sampling_rate} Hz, Implied: {implied_sampling_rate:.2f} Hz")
                print(f"   ⚠️  Discrepancy: {results['sampling_rate_discrepancy']:.2f} Hz")
            else:
                print(f"   ✅ Sampling rate consistent with config")
    
    # Check duration match
    if results['samples_match_expected']:
        print(f"   ✅ Sample count matches expected (within tolerance)")
    else:
        print(f"   ❌ Sample count mismatch! Expected: {expected_samples}, Got: {actual_samples}")
        print(f"   ❌ Duration discrepancy: {results['duration_discrepancy_sec']:.2f} seconds")
    
    return results

def load_physiological_data(file_path, signal_type):
    """
    Load physiological data from CSV file and separate time from signal variables
    
    Args:
        file_path (str): Path to the CSV file
        signal_type (str): Type of signal ('eda', 'ecg', 'resp')
    
    Returns:
        tuple: (signal_df, time_series) where signal_df contains only physiological variables
    """
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None, None
    
    # Load complete data
    df_complete = pd.read_csv(file_path)
    
    # Separate time from signal variables
    time_series = df_complete['time'].copy()
    signal_df = df_complete.drop('time', axis=1).copy()
    
    print(f"✅ Loaded {os.path.basename(file_path)}: {len(signal_df)} samples, {len(signal_df.columns)} {signal_type.upper()} variables")
    print(f"   {signal_type.upper()} columns: {list(signal_df.columns)}")
    
    return signal_df, time_series

def plot_physiological_signals(signal_df, time_series, title, signal_type, sampling_rate=None):
    """
    Plot physiological signals using appropriate NeuroKit functions
    
    Args:
        signal_df: DataFrame with signal variables
        time_series: Time series data
        title: Plot title
        signal_type: Type of signal ('eda', 'ecg', 'resp')
        sampling_rate: Sampling rate (Hz), defaults to config value if None
    """
    if sampling_rate is None:
        sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
    if signal_df is None or time_series is None:
        print(f"⚠️  Cannot plot {title} - data not available")
        return
    
    try:
        # Create a figure with custom size
        fig, axes = plt.subplots(figsize=(15, 10))
        
        # Use appropriate NeuroKit plot function based on signal type
        if signal_type == 'eda':
            nk.eda_plot(signal_df)
        elif signal_type == 'ecg':
            nk.ecg_plot(signal_df)
        elif signal_type == 'resp':
            nk.rsp_plot(signal_df)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Customize the plot
        plt.suptitle(f'{signal_type.upper()} Analysis - {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join('test', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"{signal_type}_test_{title.lower().replace(' ', '_').replace(':', '')}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 {signal_type.upper()} plot saved: {plot_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error plotting {signal_type.upper()} {title}: {str(e)}")
        # Fallback: create a simple plot
        create_simple_signal_plot(signal_df, time_series, title, signal_type)

def create_simple_signal_plot(signal_df, time_series, title, signal_type):
    """
    Create a simple signal plot as fallback if specific plot function fails
    
    Args:
        signal_df (pd.DataFrame): DataFrame with signal variables
        time_series (pd.Series): Time series
        title (str): Title for the plot
        signal_type (str): Type of signal ('eda', 'ecg', 'resp')
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{signal_type.upper()} Signals - {title}', fontsize=16, fontweight='bold')
    
    # Define signal-specific components to plot
    signal_components = {
        'eda': [
            ('EDA_Clean', 'EDA Clean Signal', 'EDA (µS)'),
            ('EDA_Tonic', 'EDA Tonic (SCL)', 'EDA (µS)'),
            ('EDA_Phasic', 'EDA Phasic (SCR)', 'EDA (µS)'),
            ('SCR_Peaks', 'SCR Peaks', 'Peak Detection')
        ],
        'ecg': [
            ('ECG_Clean', 'ECG Clean Signal', 'ECG (mV)'),
            ('ECG_R_Peaks', 'R Peaks', 'Peak Detection'),
            ('HRV_RMSSD', 'HRV RMSSD', 'HRV (ms)'),
            ('ECG_Rate', 'Heart Rate', 'BPM')
        ],
        'resp': [
            ('RSP_Clean', 'RESP Clean Signal', 'Respiration'),
            ('RSP_Rate', 'Respiratory Rate', 'BPM'),
            ('RSP_Amplitude', 'Respiratory Amplitude', 'Amplitude'),
            ('RSP_Peaks', 'Respiratory Peaks', 'Peak Detection')
        ]
    }
    
    components = signal_components.get(signal_type, [])
    axes_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for i, (col_name, plot_title, ylabel) in enumerate(components[:4]):
        if i < len(axes_positions) and col_name in signal_df.columns:
            ax = axes[axes_positions[i]]
            ax.plot(time_series, signal_df[col_name])
            ax.set_title(plot_title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        elif i < len(axes_positions):
            # Hide empty subplot
            axes[axes_positions[i]].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join('test', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"{signal_type}_simple_{title.lower().replace(' ', '_').replace(':', '')}.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"📊 Simple {signal_type.upper()} plot saved: {plot_path}")
    plt.show()

def test_physiological_preprocessing():
    """
    Main testing function to validate physiological preprocessing results (EDA, ECG, RESP)
    """
    print("🧪 Starting physiological preprocessing validation test")
    print("=" * 60)
    
    # Check if we're in test mode
    if not TEST_MODE:
        print("⚠️  TEST_MODE is False in config.py")
        print("   Set TEST_MODE = True and run preprocess_phys.py first")
        return
    
    # Get test subject
    test_subject = SUJETOS_TEST[0] if SUJETOS_TEST else 'S04'
    print(f"🎯 Testing physiological data for subject: {test_subject}")
    
    # Define expected file paths for all signal types
    signal_types = ['eda', 'ecg', 'resp']
    signal_dirs = {}
    for signal in signal_types:
        signal_dirs[signal] = {
            'dmt_high': os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_high'),
            'dmt_low': os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_low')
        }
    
    # Generate expected files for all signal types
    expected_files = {}
    for signal in signal_types:
        expected_files[signal] = {
            f'{signal.upper()} DMT High Dose': os.path.join(signal_dirs[signal]['dmt_high'], f'{test_subject}_dmt_session1_high.csv'),
            f'{signal.upper()} Resting High Dose': os.path.join(signal_dirs[signal]['dmt_high'], f'{test_subject}_rs_session1_high.csv'),
            f'{signal.upper()} DMT Low Dose': os.path.join(signal_dirs[signal]['dmt_low'], f'{test_subject}_dmt_session2_low.csv'),
            f'{signal.upper()} Resting Low Dose': os.path.join(signal_dirs[signal]['dmt_low'], f'{test_subject}_rs_session2_low.csv')
        }
    
    print(f"\n📁 Expected output directories:")
    for signal in signal_types:
        print(f"   📂 {signal.upper()}:")
        for condition, path in signal_dirs[signal].items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"      {exists} {condition}: {path}")
    
    print(f"\n📄 Testing physiological files:")
    
    # Load and test each signal type
    all_results = {}
    for signal_type in signal_types:
        print(f"\n🧬 Testing {signal_type.upper()} signals:")
        signal_results = {}
        
        for description, file_path in expected_files[signal_type].items():
            print(f"\n🔍 Testing: {description}")
            print(f"   Path: {file_path}")
            
            # Load data
            signal_df, time_series = load_physiological_data(file_path, signal_type)
            
            if signal_df is not None:
                # Determine expected duration based on file type
                if 'DMT' in description:
                    expected_duration_sec = DURACIONES_ESPERADAS['DMT']  # Both DMT sessions have same duration
                else:  # Resting
                    expected_duration_sec = DURACIONES_ESPERADAS['Reposo']  # Both Resting sessions have same duration
                
                # Test duration and sampling rate
                duration_test_results = test_duration_and_sampling_rate(
                    signal_df, time_series, expected_duration_sec, description
                )
                
                # Store results for summary
                signal_results[description] = {
                    'samples': len(signal_df),
                    'duration_min': len(signal_df) / NEUROKIT_PARAMS['sampling_rate_default'] / 60,
                    'variables': len(signal_df.columns),
                    'success': True,
                    'duration_test': duration_test_results
                }
                
                # Generate plot
                print(f"   📊 Generating {signal_type.upper()} plot for {description}...")
                plot_physiological_signals(signal_df, time_series, description, signal_type)
                
            else:
                signal_results[description] = {'success': False}
        
        all_results[signal_type] = signal_results
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    # Calculate total statistics across all signal types
    total_successful = 0
    total_files = 0
    signal_success_counts = {}
    
    for signal_type, signal_results in all_results.items():
        successful_in_signal = sum(1 for r in signal_results.values() if r.get('success', False))
        total_in_signal = len(signal_results)
        signal_success_counts[signal_type] = {'success': successful_in_signal, 'total': total_in_signal}
        total_successful += successful_in_signal
        total_files += total_in_signal
    
    print(f"✅ Successfully loaded files: {total_successful}/{total_files}")
    print(f"\n📊 Success by signal type:")
    for signal_type, counts in signal_success_counts.items():
        print(f"   {signal_type.upper()}: {counts['success']}/{counts['total']} files")
    
    if total_successful > 0:
        print(f"\n📊 Data characteristics by signal type:")
        for signal_type, signal_results in all_results.items():
            print(f"\n   📂 {signal_type.upper()}:")
            for desc, result in signal_results.items():
                if result.get('success', False):
                    print(f"      {desc}:")
                    print(f"         Samples: {result['samples']:,}")
                    print(f"         Duration: {result['duration_min']:.2f} minutes")
                    print(f"         Variables: {result['variables']}")
        
        # Duration and sampling rate test summary
        print(f"\n🔍 DURATION & SAMPLING RATE ANALYSIS")
        print("=" * 60)
        
        duration_issues = []
        sampling_rate_issues = []
        
        for signal_type, signal_results in all_results.items():
            for desc, result in signal_results.items():
                if result.get('success', False) and 'duration_test' in result:
                    dt = result['duration_test']
                    
                    print(f"\n📁 {desc}:")
                    print(f"   Expected: {dt['expected_duration_min']:.2f} min")
                    print(f"   From samples: {dt['calculated_duration_from_samples_min']:.2f} min")
                    
                    if dt['time_series_duration_min']:
                        print(f"   From time series: {dt['time_series_duration_min']:.2f} min")
                        print(f"   Implied sampling rate: {dt['implied_sampling_rate']:.2f} Hz")
                        print(f"   Config sampling rate: {dt['config_sampling_rate']} Hz")
                        
                        if dt['sampling_rate_discrepancy'] and dt['sampling_rate_discrepancy'] > 1:
                            sampling_rate_issues.append({
                                'file': desc,
                                'expected_sr': dt['config_sampling_rate'],
                                'actual_sr': dt['implied_sampling_rate'],
                                'discrepancy': dt['sampling_rate_discrepancy']
                            })
                    
                    if not dt['samples_match_expected']:
                        duration_issues.append({
                            'file': desc,
                            'expected_duration': dt['expected_duration_min'],
                            'actual_duration': dt['calculated_duration_from_samples_min'],
                            'discrepancy_sec': dt['duration_discrepancy_sec']
                        })
        
        # Report issues
        if duration_issues:
            print(f"\n⚠️  DURATION ISSUES FOUND:")
            for issue in duration_issues:
                print(f"   ❌ {issue['file']}: Expected {issue['expected_duration']:.2f} min, got {issue['actual_duration']:.2f} min")
                print(f"      Discrepancy: {issue['discrepancy_sec']:.2f} seconds")
        else:
            print(f"\n✅ All durations match expected values")
        
        if sampling_rate_issues:
            print(f"\n⚠️  SAMPLING RATE ISSUES FOUND:")
            for issue in sampling_rate_issues:
                print(f"   ❌ {issue['file']}: Config {issue['expected_sr']} Hz, actual {issue['actual_sr']:.2f} Hz")
                print(f"      Discrepancy: {issue['discrepancy']:.2f} Hz")
            print(f"\n💡 This explains why plots show shorter durations than expected!")
            print(f"💡 The time axis in plots uses the actual sampling rate from data,")
            print(f"💡 but preprocessing used the config sampling rate for truncation/padding.")
        else:
            print(f"\n✅ All sampling rates consistent with config")
    
    if total_successful < total_files:
        print(f"\n⚠️  Missing files:")
        for signal_type, signal_results in all_results.items():
            missing_in_signal = [desc for desc, result in signal_results.items() if not result.get('success', False)]
            if missing_in_signal:
                print(f"   📂 {signal_type.upper()}:")
                for desc in missing_in_signal:
                    print(f"      ❌ {desc}")
        print(f"\n💡 Run preprocess_phys.py in TEST_MODE first to generate the data")
    
    print(f"\n🎯 Test completed!")
    if total_successful > 0:
        plot_dir = os.path.join('test', 'plots')
        print(f"📊 Plots saved in: {plot_dir}")
        print(f"📊 Generated plots for {len(signal_types)} signal types: {', '.join([s.upper() for s in signal_types])}")

if __name__ == "__main__":
    test_physiological_preprocessing()
