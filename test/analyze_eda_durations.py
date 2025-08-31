# -*- coding: utf-8 -*-
"""
EDA Data Duration Analysis - Simplified

This script analyzes EDA recording durations and generates a simple visualization.
Only shows recordings below minimum duration and creates a basic plot.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("MNE not available. Install with: pip install mne")
    exit(1)

# Configuration
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
PHYSIOLOGY_DATA = os.path.join(ROOT_DATA, 'original', 'physiology')

# Expected parameters
EXPECTED_SRATE = 250  # Hz
EXPECTED_DURATION_DMT = 20 * 60  # 20 minutes in seconds for DMT
EXPECTED_DURATION_REPOSO = 10 * 60  # 10 minutes in seconds for Reposo
EXPECTED_SAMPLES_DMT = EXPECTED_SRATE * EXPECTED_DURATION_DMT  # 300,000
EXPECTED_SAMPLES_REPOSO = EXPECTED_SRATE * EXPECTED_DURATION_REPOSO  # 150,000

# All subjects and experiments
SUBJECTS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 
           'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

EXPERIMENTS = ['DMT_1', 'DMT_2', 'Reposo_1', 'Reposo_2']

def analyze_single_recording(experiment, subject, verbose=False):
    """
    Analyze a single EDA recording
    
    Returns dict with recording info or None if file not found/corrupted
    """
    # Construct filename pattern based on experiment type
    if 'DMT' in experiment:
        session = experiment.split('_')[1]
        filename = f'{subject}_DMT_Session{session}_DMT.vhdr'
    else:  # Reposo
        session = experiment.split('_')[1] 
        filename = f'{subject}_RS_Session{session}_EC.vhdr'
    
    filepath = os.path.join(PHYSIOLOGY_DATA, experiment, subject, filename)
    
    if not os.path.exists(filepath):
        if verbose:
            print(f"❌ File not found: {filepath}")
        return None
    
    try:
        # Load with MNE
        raw = mne.io.read_raw_brainvision(filepath, preload=False, verbose=False)
        
        # Get basic info
        n_samples = raw.n_times
        srate = raw.info['sfreq']
        duration_sec = n_samples / srate
        duration_min = duration_sec / 60
        
        # Check for EDA channel
        has_eda = 'GSR' in raw.ch_names
        
        # Determine expected values based on experiment type
        if 'DMT' in experiment:
            expected_duration = EXPECTED_DURATION_DMT
            expected_samples = EXPECTED_SAMPLES_DMT
        else:  # Reposo
            expected_duration = EXPECTED_DURATION_REPOSO
            expected_samples = EXPECTED_SAMPLES_REPOSO
        
        # Calculate deviations from expected
        sample_deviation = n_samples - expected_samples
        duration_deviation = duration_sec - expected_duration
        srate_deviation = srate - EXPECTED_SRATE
        
        # Check if recording meets minimum duration
        meets_minimum = duration_sec >= expected_duration
        
        result = {
            'subject': subject,
            'experiment': experiment,
            'filename': filename,
            'n_samples': n_samples,
            'sampling_rate': srate,
            'duration_sec': duration_sec,
            'duration_min': duration_min,
            'has_eda': has_eda,
            'expected_duration': expected_duration,
            'expected_samples': expected_samples,
            'sample_deviation': sample_deviation,
            'duration_deviation': duration_deviation,
            'srate_deviation': srate_deviation,
            'meets_minimum': meets_minimum,
            'file_exists': True
        }
        
        if verbose:
            print(f"✅ {subject} {experiment}: {n_samples:,} samples, {duration_min:.1f} min")
            
        return result
        
    except Exception as e:
        if verbose:
            print(f"❌ Error reading {filepath}: {str(e)}")
        return {
            'subject': subject,
            'experiment': experiment,
            'filename': filename,
            'error': str(e),
            'file_exists': True,
            'corrupted': True
        }

def analyze_all_recordings(verbose=True):
    """Analyze all EDA recordings and return comprehensive DataFrame"""
    results = []
    
    print("🔍 Analyzing EDA recording durations...")
    print(f"Expected DMT: {EXPECTED_SAMPLES_DMT:,} samples ({EXPECTED_DURATION_DMT/60:.1f} min @ {EXPECTED_SRATE} Hz)")
    print(f"Expected Reposo: {EXPECTED_SAMPLES_REPOSO:,} samples ({EXPECTED_DURATION_REPOSO/60:.1f} min @ {EXPECTED_SRATE} Hz)")
    print("=" * 70)
    
    for experiment in EXPERIMENTS:
        if verbose:
            print(f"\n📁 Experiment: {experiment}")
        
        for subject in SUBJECTS:
            result = analyze_single_recording(experiment, subject, verbose=False)
            if result:
                results.append(result)
            elif verbose:
                print(f"   ❌ {subject}: File not found")
    
    return pd.DataFrame(results)

def report_short_recordings(df):
    """Report recordings that don't meet minimum duration requirements"""
    print("\n" + "="*70)
    print("⚠️  RECORDINGS BELOW MINIMUM DURATION")
    print("="*70)
    
    if 'corrupted' in df.columns:
        valid_df = df[~df['corrupted']].copy()
    else:
        valid_df = df[df['file_exists'] == True].copy()
    
    if len(valid_df) == 0:
        print("❌ No valid recordings to analyze")
        return
    
    # Filter recordings that don't meet minimum duration
    short_recordings = valid_df[~valid_df['meets_minimum']].copy()
    
    if len(short_recordings) == 0:
        print("✅ All recordings meet minimum duration requirements!")
        return
    
    print(f"\n📊 Found {len(short_recordings)} recordings below minimum duration:")
    print("-" * 90)
    print(f"{'Subject':<8} {'Experiment':<10} {'Filename':<35} {'Duration':<12} {'Expected':<12} {'Deficit':<10}")
    print("-" * 90)
    
    for _, row in short_recordings.iterrows():
        deficit_min = (row['expected_duration'] - row['duration_sec']) / 60
        print(f"{row['subject']:<8} {row['experiment']:<10} {row['filename']:<35} "
              f"{row['duration_min']:>8.1f} min {row['expected_duration']/60:>8.1f} min "
              f"{deficit_min:>7.1f} min")
    
    # Summary by experiment type
    print(f"\n📈 Summary by experiment type:")
    for exp in EXPERIMENTS:
        exp_short = short_recordings[short_recordings['experiment'] == exp]
        if len(exp_short) > 0:
            expected_dur = EXPECTED_DURATION_DMT/60 if 'DMT' in exp else EXPECTED_DURATION_REPOSO/60
            avg_duration = exp_short['duration_min'].mean()
            print(f"   {exp}: {len(exp_short)} short recordings (avg: {avg_duration:.1f} min, expected: {expected_dur:.1f} min)")
    
    # Most problematic subjects
    print(f"\n👤 Subjects with most short recordings:")
    subject_counts = short_recordings['subject'].value_counts().head(5)
    for subject, count in subject_counts.items():
        print(f"   {subject}: {count} short recordings")
    
    return short_recordings



def create_simple_plot(df, save_plots=True):
    """Create a simple duration distribution plot"""
    if 'corrupted' in df.columns:
        valid_df = df[~df['corrupted']].copy()
    else:
        valid_df = df[df['file_exists'] == True].copy()
    
    if len(valid_df) == 0:
        print("❌ No valid data for visualization")
        return
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    plt.hist(valid_df['duration_min'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(20, color='red', linestyle='--', linewidth=2, label='Expected DMT (20 min)')
    plt.axvline(10, color='orange', linestyle='--', linewidth=2, label='Expected Reposo (10 min)')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of EDA Recording Durations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plots:
        plot_path = os.path.join('test', 'eda_duration_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {plot_path}")
    
    plt.show()



def main():
    """Main analysis function - simplified"""
    print("🧠 EDA Duration Analysis Tool")
    print("="*50)
    
    # Check if data directory exists
    if not os.path.exists(PHYSIOLOGY_DATA):
        print(f"❌ Data directory not found: {PHYSIOLOGY_DATA}")
        print("Please check your data path configuration.")
        return
    
    # Analyze all recordings
    df = analyze_all_recordings()
    
    if df.empty:
        print("❌ No recordings found!")
        return
    
    # Report short recordings (most important)
    short_recordings_df = report_short_recordings(df)
    
    # Create simple plot
    create_simple_plot(df)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
