# -*- coding: utf-8 -*-
"""
Exploratory testing of EDA preprocessing results

This script tests the EDA signals extracted by preprocess_eda.py for the test subject (S04).
It loads the processed CSV files, removes the time column, and generates EDA plots 
to verify signal quality and processing correctness.

NEW FEATURES:
- Duration and sampling rate validation
- Detection of sampling rate mismatches between config and actual data
- Analysis of time series consistency
- Detailed reporting of duration discrepancies

This helps identify issues like:
- Incorrect sampling rate assumptions in preprocessing
- Duration truncation/padding problems
- Time axis discrepancies in plots

Usage:
1. First run preprocess_eda.py in TEST_MODE to generate S04 data
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

def test_duration_and_sampling_rate(eda_df, time_series, expected_duration_sec, file_description):
    """
    Test if the actual duration and sampling rate match expectations
    
    Args:
        eda_df: DataFrame with EDA data
        time_series: Time series data
        expected_duration_sec: Expected duration in seconds
        file_description: Description of the file being tested
        
    Returns:
        dict: Test results with sampling rate analysis
    """
    print(f"\n🔍 Testing duration and sampling rate for: {file_description}")
    
    # Calculate actual duration from samples using config sampling rate
    config_sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
    actual_samples = len(eda_df)
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

def load_eda_data(file_path):
    """
    Load EDA data from CSV file and separate time from EDA variables
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        tuple: (eda_df, time_series) where eda_df contains only EDA variables
    """
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None, None
    
    # Load complete data
    df_complete = pd.read_csv(file_path)
    
    # Separate time from EDA variables
    time_series = df_complete['time'].copy()
    eda_df = df_complete.drop('time', axis=1).copy()
    
    print(f"✅ Loaded {os.path.basename(file_path)}: {len(eda_df)} samples, {len(eda_df.columns)} EDA variables")
    print(f"   EDA columns: {list(eda_df.columns)}")
    
    return eda_df, time_series

def plot_eda_signals(eda_df, time_series, title, sampling_rate=None):
    """
    Plot EDA signals using NeuroKit
    
    Args:
        eda_df: DataFrame with EDA variables
        time_series: Time series data
        title: Plot title
        sampling_rate: Sampling rate (Hz), defaults to config value if None
    """
    if sampling_rate is None:
        sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
    if eda_df is None or time_series is None:
        print(f"⚠️  Cannot plot {title} - data not available")
        return
    
    try:
        # Create a figure with custom size
        fig, axes = plt.subplots(figsize=(15, 10))
        
        # Use NeuroKit's eda_plot function
        # Note: eda_plot expects the full dataframe with EDA processing results
        nk.eda_plot(eda_df)
        
        # Customize the plot
        plt.suptitle(f'EDA Analysis - {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join('test', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"eda_test_{title.lower().replace(' ', '_').replace(':', '')}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 Plot saved: {plot_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error plotting {title}: {str(e)}")
        # Fallback: create a simple plot
        create_simple_eda_plot(eda_df, time_series, title)

def create_simple_eda_plot(eda_df, time_series, title):
    """
    Create a simple EDA plot as fallback if eda_plot fails
    
    Args:
        eda_df (pd.DataFrame): DataFrame with EDA variables
        time_series (pd.Series): Time series
        title (str): Title for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'EDA Signals - {title}', fontsize=16, fontweight='bold')
    
    # Plot main EDA components if available
    if 'EDA_Clean' in eda_df.columns:
        axes[0, 0].plot(time_series, eda_df['EDA_Clean'])
        axes[0, 0].set_title('EDA Clean Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('EDA (µS)')
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'EDA_Tonic' in eda_df.columns:
        axes[0, 1].plot(time_series, eda_df['EDA_Tonic'])
        axes[0, 1].set_title('EDA Tonic (SCL)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('EDA (µS)')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'EDA_Phasic' in eda_df.columns:
        axes[1, 0].plot(time_series, eda_df['EDA_Phasic'])
        axes[1, 0].set_title('EDA Phasic (SCR)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('EDA (µS)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot SCR peaks if available
    if 'SCR_Peaks' in eda_df.columns:
        axes[1, 1].plot(time_series, eda_df['SCR_Peaks'])
        axes[1, 1].set_title('SCR Peaks')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Peak Detection')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = os.path.join('test', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f"eda_simple_{title.lower().replace(' ', '_').replace(':', '')}.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"📊 Simple plot saved: {plot_path}")
    plt.show()

def test_eda_preprocessing():
    """
    Main testing function to validate EDA preprocessing results
    """
    print("🧪 Starting EDA preprocessing validation test")
    print("=" * 60)
    
    # Check if we're in test mode
    if not TEST_MODE:
        print("⚠️  TEST_MODE is False in config.py")
        print("   Set TEST_MODE = True and run preprocess_eda.py first")
        return
    
    # Get test subject
    test_subject = SUJETOS_TEST[0] if SUJETOS_TEST else 'S04'
    print(f"🎯 Testing EDA data for subject: {test_subject}")
    
    # Define expected file paths
    eda_dirs = {
        'dmt_high': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high'),
        'dmt_low': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
    }
    
    # Expected files for the test subject
    expected_files = {
        'DMT High Dose': os.path.join(eda_dirs['dmt_high'], f'{test_subject}_dmt_session1_high.csv'),
        'Resting High Dose': os.path.join(eda_dirs['dmt_high'], f'{test_subject}_rs_session1_high.csv'),
        'DMT Low Dose': os.path.join(eda_dirs['dmt_low'], f'{test_subject}_dmt_session2_low.csv'),
        'Resting Low Dose': os.path.join(eda_dirs['dmt_low'], f'{test_subject}_rs_session2_low.csv')
    }
    
    print(f"\n📁 Expected output directories:")
    for condition, path in eda_dirs.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {condition}: {path}")
    
    print(f"\n📄 Testing EDA files:")
    
    # Load and test each file
    results = {}
    for description, file_path in expected_files.items():
        print(f"\n🔍 Testing: {description}")
        print(f"   Path: {file_path}")
        
        # Load data
        eda_df, time_series = load_eda_data(file_path)
        
        if eda_df is not None:
            # Determine expected duration based on file type
            if 'DMT' in description:
                expected_duration_sec = DURACIONES_ESPERADAS['DMT']  # Both DMT sessions have same duration
            else:  # Resting
                expected_duration_sec = DURACIONES_ESPERADAS['Reposo']  # Both Resting sessions have same duration
            
            # Test duration and sampling rate
            duration_test_results = test_duration_and_sampling_rate(
                eda_df, time_series, expected_duration_sec, description
            )
            
            # Store results for summary
            results[description] = {
                'samples': len(eda_df),
                'duration_min': len(eda_df) / NEUROKIT_PARAMS['sampling_rate_default'] / 60,
                'variables': len(eda_df.columns),
                'success': True,
                'duration_test': duration_test_results
            }
            
            # Generate plot
            print(f"   📊 Generating plot for {description}...")
            plot_eda_signals(eda_df, time_series, description)
            
        else:
            results[description] = {'success': False}
    
    # Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    successful_files = sum(1 for r in results.values() if r.get('success', False))
    total_files = len(expected_files)
    
    print(f"✅ Successfully loaded files: {successful_files}/{total_files}")
    
    if successful_files > 0:
        print(f"\n📊 Data characteristics:")
        for desc, result in results.items():
            if result.get('success', False):
                print(f"   {desc}:")
                print(f"      Samples: {result['samples']:,}")
                print(f"      Duration: {result['duration_min']:.2f} minutes")
                print(f"      EDA variables: {result['variables']}")
        
        # Duration and sampling rate test summary
        print(f"\n🔍 DURATION & SAMPLING RATE ANALYSIS")
        print("=" * 60)
        
        duration_issues = []
        sampling_rate_issues = []
        
        for desc, result in results.items():
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
    
    if successful_files < total_files:
        print(f"\n⚠️  Missing files:")
        for desc, result in results.items():
            if not result.get('success', False):
                print(f"   ❌ {desc}")
        print(f"\n💡 Run preprocess_eda.py in TEST_MODE first to generate the data")
    
    print(f"\n🎯 Test completed!")
    if successful_files > 0:
        plot_dir = os.path.join('test', 'plots')
        print(f"📊 Plots saved in: {plot_dir}")

if __name__ == "__main__":
    test_eda_preprocessing()
