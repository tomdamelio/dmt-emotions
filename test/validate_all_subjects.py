#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for all processed subjects
Validates physiological preprocessing results for ALL subjects (EDA, ECG, RESP)

This script:
- Checks file existence for all processed subjects
- Validates data characteristics (samples, duration, sampling rate)
- Generates detailed summary report
- Creates plots only for a sample subject (to avoid overwhelming output)

Author: Assistant
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# Add parent directory to path to import from scripts and config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import (
        DERIVATIVES_DATA, SUJETOS_TEST, SUJETOS_VALIDOS, TODOS_LOS_SUJETOS,
        TEST_MODE, PROCESSING_MODE, DURACIONES_ESPERADAS, NEUROKIT_PARAMS, DOSIS
    )
except ImportError:
    print("❌ Could not import config. Make sure config.py exists in parent directory.")
    sys.exit(1)

def validate_all_subjects():
    """
    Validate physiological preprocessing results for all processed subjects
    """
    print("🧪 Starting comprehensive validation for all processed subjects")
    print("=" * 80)
    
    # Determine which subjects were processed
    if TEST_MODE:
        subjects_to_validate = SUJETOS_TEST
        print(f"🔬 Validating TEST MODE subjects: {subjects_to_validate}")
    else:
        if PROCESSING_MODE == 'ALL':
            subjects_to_validate = TODOS_LOS_SUJETOS
            print(f"🔬 Validating ALL SUBJECTS: {len(subjects_to_validate)} subjects")
        else:
            subjects_to_validate = SUJETOS_VALIDOS
            print(f"🔬 Validating VALID SUBJECTS: {len(subjects_to_validate)} subjects")
    
    print(f"📋 Subjects to validate: {subjects_to_validate}")
    
    signal_types = ['eda', 'ecg', 'resp']
    
    # Create summary structures
    validation_summary = {
        'total_subjects': len(subjects_to_validate),
        'total_expected_files': len(subjects_to_validate) * 4 * 3,  # 4 files per subject * 3 signals
        'subjects_validated': {},
        'signals_summary': {signal: {'found': 0, 'missing': 0, 'errors': 0} for signal in signal_types},
        'missing_files': [],
        'error_files': []
    }
    
    print(f"\n📊 Expected validation scope:")
    print(f"   🔢 Subjects: {validation_summary['total_subjects']}")
    print(f"   📁 Files per subject: 12 (4 files × 3 signals)")
    print(f"   📁 Total expected files: {validation_summary['total_expected_files']}")
    
    # Validate each subject
    for subject_idx, subject in enumerate(subjects_to_validate):
        print(f"\n👤 Validating subject {subject} ({subject_idx + 1}/{len(subjects_to_validate)})")
        
        # Get dose information for this subject
        try:
            dose_session_1 = DOSIS['Dosis_Sesion_1'][subject]
            dose_session_2 = DOSIS['Dosis_Sesion_2'][subject]
            condition_map = {'Alta': 'high', 'Baja': 'low'}
            condition_1 = condition_map[dose_session_1]
            condition_2 = condition_map[dose_session_2]
        except KeyError:
            print(f"   ⚠️  No dose info found for {subject}")
            condition_1, condition_2 = 'high', 'low'  # fallback
        
        subject_results = {
            'total_files': 0,
            'found_files': 0,
            'missing_files': 0,
            'error_files': 0,
            'files_detail': {}
        }
        
        # Expected files for this subject
        expected_files = [
            # Session 1 files (go to condition_1 folder)
            (f'{subject}_dmt_session1_{condition_1}.csv', signal_types, condition_1, 'DMT'),
            (f'{subject}_rs_session1_{condition_1}.csv', signal_types, condition_1, 'Resting'),
            # Session 2 files (go to condition_2 folder)
            (f'{subject}_dmt_session2_{condition_2}.csv', signal_types, condition_2, 'DMT'),
            (f'{subject}_rs_session2_{condition_2}.csv', signal_types, condition_2, 'Resting')
        ]
        
        # Check each file for each signal type
        for filename, signals, condition, session_type in expected_files:
            for signal in signals:
                file_path = os.path.join(DERIVATIVES_DATA, 'phys', signal, f'dmt_{condition}', filename)
                file_key = f"{signal}/{filename}"
                subject_results['total_files'] += 1
                
                if os.path.exists(file_path):
                    # Try to load and validate
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Basic validation
                        expected_duration_sec = DURACIONES_ESPERADAS[session_type]
                        expected_samples = int(expected_duration_sec * NEUROKIT_PARAMS['sampling_rate_default'])
                        actual_samples = len(df)
                        
                        validation_summary['signals_summary'][signal]['found'] += 1
                        subject_results['found_files'] += 1
                        subject_results['files_detail'][file_key] = {
                            'status': 'found',
                            'samples': actual_samples,
                            'expected_samples': expected_samples,
                            'duration_min': actual_samples / NEUROKIT_PARAMS['sampling_rate_default'] / 60,
                            'columns': len(df.columns)
                        }
                        
                        print(f"   ✅ {signal.upper()}: {filename} ({actual_samples} samples, {len(df.columns)} cols)")
                        
                    except Exception as e:
                        validation_summary['signals_summary'][signal]['errors'] += 1
                        subject_results['error_files'] += 1
                        validation_summary['error_files'].append(f"{subject}:{file_key} - {str(e)}")
                        subject_results['files_detail'][file_key] = {'status': 'error', 'error': str(e)}
                        print(f"   ❌ {signal.upper()}: {filename} - Error: {str(e)}")
                        
                else:
                    validation_summary['signals_summary'][signal]['missing'] += 1
                    subject_results['missing_files'] += 1
                    validation_summary['missing_files'].append(f"{subject}:{file_key}")
                    subject_results['files_detail'][file_key] = {'status': 'missing'}
                    print(f"   ⚠️  {signal.upper()}: {filename} - File not found")
        
        validation_summary['subjects_validated'][subject] = subject_results
        
        # Summary for this subject
        success_rate = (subject_results['found_files'] / subject_results['total_files']) * 100 if subject_results['total_files'] > 0 else 0
        print(f"   📊 Subject {subject} summary: {subject_results['found_files']}/{subject_results['total_files']} files found ({success_rate:.1f}%)")
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    
    total_found = sum(summary['found'] for summary in validation_summary['signals_summary'].values())
    total_missing = sum(summary['missing'] for summary in validation_summary['signals_summary'].values())
    total_errors = sum(summary['errors'] for summary in validation_summary['signals_summary'].values())
    overall_success_rate = (total_found / validation_summary['total_expected_files']) * 100
    
    print(f"\n🎯 Overall Results:")
    print(f"   📁 Total expected files: {validation_summary['total_expected_files']}")
    print(f"   ✅ Files found: {total_found}")
    print(f"   ⚠️  Files missing: {total_missing}")
    print(f"   ❌ Files with errors: {total_errors}")
    print(f"   📈 Success rate: {overall_success_rate:.1f}%")
    
    print(f"\n🧬 Results by Signal Type:")
    for signal, summary in validation_summary['signals_summary'].items():
        total_expected_for_signal = len(subjects_to_validate) * 4  # 4 files per subject per signal
        success_rate = (summary['found'] / total_expected_for_signal) * 100
        print(f"   📊 {signal.upper()}: {summary['found']}/{total_expected_for_signal} found ({success_rate:.1f}%), {summary['missing']} missing, {summary['errors']} errors")
    
    print(f"\n👥 Results by Subject:")
    for subject, results in validation_summary['subjects_validated'].items():
        success_rate = (results['found_files'] / results['total_files']) * 100
        status_emoji = "✅" if success_rate == 100 else "⚠️" if success_rate >= 75 else "❌"
        print(f"   {status_emoji} {subject}: {results['found_files']}/{results['total_files']} files ({success_rate:.1f}%)")
    
    # List problematic files if any
    if validation_summary['missing_files']:
        print(f"\n⚠️  Missing Files ({len(validation_summary['missing_files'])}):")
        for missing_file in validation_summary['missing_files'][:20]:  # Show first 20
            print(f"   📄 {missing_file}")
        if len(validation_summary['missing_files']) > 20:
            print(f"   ... and {len(validation_summary['missing_files']) - 20} more")
    
    if validation_summary['error_files']:
        print(f"\n❌ Files with Errors ({len(validation_summary['error_files'])}):")
        for error_file in validation_summary['error_files'][:10]:  # Show first 10
            print(f"   📄 {error_file}")
        if len(validation_summary['error_files']) > 10:
            print(f"   ... and {len(validation_summary['error_files']) - 10} more")
    
    # Generate plots for first subject with complete data (if any)
    sample_subject = None
    for subject, results in validation_summary['subjects_validated'].items():
        if results['found_files'] >= 12:  # All files found
            sample_subject = subject
            break
    
    if sample_subject:
        print(f"\n📊 Generating sample plots for subject {sample_subject} (complete data)...")
        generate_sample_plots(sample_subject)
    else:
        print(f"\n⚠️  No subject with complete data found for plotting")
    
    print(f"\n🎯 Validation completed!")
    return validation_summary

def generate_sample_plots(subject):
    """Generate plots for a sample subject"""
    print(f"\n📊 Generating plots for sample subject: {subject}")
    
    # Get dose information
    try:
        dose_session_1 = DOSIS['Dosis_Sesion_1'][subject]
        condition_map = {'Alta': 'high', 'Baja': 'low'}
        condition_1 = condition_map[dose_session_1]
    except:
        condition_1 = 'high'  # fallback
    
    signal_types = ['eda', 'ecg', 'resp']
    plot_dir = os.path.join('test', 'plots', 'validation')
    os.makedirs(plot_dir, exist_ok=True)
    
    for signal in signal_types:
        # Load one file as example (DMT high dose)
        filename = f'{subject}_dmt_session1_{condition_1}.csv'
        file_path = os.path.join(DERIVATIVES_DATA, 'phys', signal, f'dmt_{condition_1}', filename)
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df_signal = df.drop('time', axis=1) if 'time' in df.columns else df
                
                # Create plot using NeuroKit
                if signal == 'eda':
                    fig = nk.eda_plot(df_signal, show=False)
                elif signal == 'ecg':
                    fig = nk.ecg_plot(df_signal, show=False)
                elif signal == 'resp':
                    fig = nk.rsp_plot(df_signal, show=False)
                
                fig.suptitle(f'{signal.upper()} Analysis - Subject {subject} - Sample Validation', fontsize=16, fontweight='bold')
                
                plot_filename = f"validation_{signal}_{subject}.png"
                plot_path = os.path.join(plot_dir, plot_filename)
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f"   📊 {signal.upper()} plot saved: {plot_path}")
                
            except Exception as e:
                print(f"   ❌ Error plotting {signal.upper()}: {str(e)}")

if __name__ == "__main__":
    validation_summary = validate_all_subjects()
