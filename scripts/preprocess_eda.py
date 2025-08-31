#%%
# -*- coding: utf-8 -*-
"""
EDA (Electrodermal Activity) data preprocessing

This script processes EDA data from BrainVision files (.vhdr) and generates 
preprocessed CSV files organized by experimental condition for subsequent analysis.

Features:
- Processes all NeuroKit EDA variables (tonic, phasic, SCR metrics)
- Applies adaptive duration strategy (truncation/zero-padding)
- No baseline correction applied (preserves raw processed data)
- Organizes output by dose condition (high/low)
- JSON logging of all processing steps

Input: .vhdr files in ../data/original/physiology/
Output: Individual CSV files organized in:
- ../data/derivatives/phys/eda/dmt_high/ (high dose session files)
- ../data/derivatives/phys/eda/dmt_low/ (low dose session files)

Each CSV contains: time, EDA_Raw, EDA_Clean, EDA_Tonic, EDA_Phasic, SCR_* variables
File format: {subject}_{session_type}_{experiment}_{condition}.csv
"""
import os

import mne
import numpy as np
import pandas as pd
import neurokit2 as nk

# Import project configuration
import sys
import json
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    PHYSIOLOGY_DATA, DERIVATIVES_DATA, DOSIS, SUJETOS_VALIDOS, SUJETOS_TEST, TEST_MODE,
    PATRONES_ARCHIVOS, DURACIONES_ESPERADAS, NEUROKIT_PARAMS, CANALES,
    TOLERANCIA_DURACION, get_nombre_archivo, get_duracion_esperada
)

#%%

# Global variable for logging
processing_log = {}

def log_file_info(subject, filename, info_dict):
    """
    Register processing information in the global log
    
    Args:
        subject (str): Subject code (e.g., 'S01')
        filename (str): Name of the processed file
        info_dict (dict): Dictionary with file information
    """
    if subject not in processing_log:
        processing_log[subject] = {}
    
    processing_log[subject][filename] = info_dict

def save_processing_log():
    """Save processing log to a JSON file"""
    log_dir = os.path.join(DERIVATIVES_DATA, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Add metadata to log
    log_with_metadata = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'script': 'preprocess_eda.py',
            'total_subjects': len(processing_log),
            'config_file': 'config.py'
        },
        'subjects': processing_log
    }
    
    log_file = os.path.join(log_dir, 'eda_preprocessing_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Processing log saved to: {log_file}")
    return log_file

def eda(experiment, subject, fname):
    
    file_path = os.path.join(PHYSIOLOGY_DATA, experiment, subject, fname)
    
    # Initialize file information for logging
    file_info = {
        'experiment': experiment,
        'subject': subject,
        'filename': fname,
        'file_path': file_path,
        'success': False,
        'has_gsr_channel': False,
        'original_samples': 0,
        'original_duration_sec': 0.0,
        'original_duration_min': 0.0,
        'expected_samples': 0,
        'expected_duration_sec': 0.0,
        'final_samples': 0,
        'final_duration_sec': 0.0,
        'final_duration_min': 0.0,
        'strategy_applied': None,  # 'truncated' or 'zero_padded'
        'missing_samples_padded': 0,
        'excess_samples_truncated': 0,
        'sampling_rate': 0.0,
        'processing_method': NEUROKIT_PARAMS['method'],
        'errors': []
    }
    
    try:
        raw_data = mne.io.read_raw_brainvision(file_path)
        data = raw_data.load_data()
        df_data = data.to_data_frame()
        
        # Calculate sampling rate using MNE
        sampling_rate = raw_data.info['sfreq']
        dt = 1/sampling_rate
        
        # Register basic information
        file_info['original_samples'] = len(df_data)
        file_info['original_duration_sec'] = len(df_data) / sampling_rate
        file_info['original_duration_min'] = file_info['original_duration_sec'] / 60
        file_info['sampling_rate'] = sampling_rate
        
        print(f"   📊 {subject} {experiment}: {len(df_data)} samples original ({file_info['original_duration_min']:.2f} min)")
    
        eda_channel = CANALES['EDA']
        if eda_channel not in df_data.columns:
            file_info['has_gsr_channel'] = False
            file_info['errors'].append(f"{eda_channel} channel not found")
            raise KeyError(f"{eda_channel} channel not found in {subject} {experiment}")
        
        file_info['has_gsr_channel'] = True
        
        # First process EDA with NeuroKit (on complete data)
        eda_signal = df_data[eda_channel]
        df_eda, info_eda = nk.eda_process(eda_signal,
                                            sampling_rate=sampling_rate,
                                            method=NEUROKIT_PARAMS['method'])
        # Low-pass filter with a 3 Hz cutoff frequency and a 4th order Butterworth filter
        
        print(f"   ✅ EDA processing completed: {len(df_eda)} samples")
        
        # NOW apply truncation/padding to the already processed EDA data
        # Get expected duration from configuration
        expected_duration_sec = get_duracion_esperada(experiment)
        
        expected_samples = int(expected_duration_sec * sampling_rate)
        current_samples = len(df_eda)
        
        # Register expected duration information
        file_info['expected_samples'] = expected_samples
        file_info['expected_duration_sec'] = expected_duration_sec
        
        print(f"   📏 Applying duration strategy: {current_samples} → {expected_samples} samples")
        
        # Apply adaptive strategy to processed EDA data
        if current_samples >= expected_samples:
            # Truncate to exact duration
            df_eda_final = df_eda[:expected_samples].copy()
            time_final = df_data['time'][:expected_samples].copy()
            
            # Register truncation information
            file_info['strategy_applied'] = 'truncated'
            file_info['excess_samples_truncated'] = current_samples - expected_samples
            
            print(f"   ✂️  Truncated to {expected_duration_sec/60:.1f} min ({len(df_eda_final)} samples)")
        else:
            # Zero padding until expected duration
            df_eda_final = df_eda.copy()
            time_original = df_data['time'][:len(df_eda)].copy()  # Align with processed EDA
            
            # Create additional samples with zero padding
            missing_samples = expected_samples - current_samples
            last_time = time_original.iloc[-1]
            
            # Register padding information
            file_info['strategy_applied'] = 'zero_padded'
            file_info['missing_samples_padded'] = missing_samples
            
            # Generate times for missing samples
            new_times = np.linspace(last_time + dt, last_time + missing_samples * dt, missing_samples)
            
            # Create DataFrame with padding for EDA (all columns as 0)
            padding_eda = pd.DataFrame(
                {col: 0.0 for col in df_eda.columns}, 
                index=range(len(df_eda), len(df_eda) + missing_samples)
            )
            
            # Create Series with padding for time
            padding_time = pd.Series(
                new_times, 
                index=range(len(time_original), len(time_original) + missing_samples)
            )
            
            df_eda_final = pd.concat([df_eda, padding_eda], ignore_index=True)
            time_final = pd.concat([time_original, padding_time], ignore_index=True)
            
            print(f"   📈 Zero-padded with {missing_samples} samples to reach {expected_duration_sec/60:.1f} min")
        
        # Verify that trimming/padding worked correctly
        final_samples = len(df_eda_final)
        final_duration_sec = final_samples / sampling_rate
        
        # Register final information
        file_info['final_samples'] = final_samples
        file_info['final_duration_sec'] = final_duration_sec
        file_info['final_duration_min'] = final_duration_sec / 60
        
        assert final_samples == expected_samples, (
            f"Trimming failed: expected {expected_samples} samples, got {final_samples} "
            f"for {subject} {experiment}"
        )
        
        assert abs(final_duration_sec - expected_duration_sec) < TOLERANCIA_DURACION, (
            f"Duration mismatch: expected {expected_duration_sec:.1f}s, got {final_duration_sec:.1f}s "
            f"for {subject} {experiment}"
        )
        
        print(f"   ✅ Final validation: {final_samples} samples, {final_duration_sec/60:.2f} min")
        
        # Mark as successful
        file_info['success'] = True
        
        # Register in log
        log_file_info(subject, fname, file_info)
        
        time = time_final
        
        return df_eda_final, info_eda, time
        
    except Exception as e:
        # Register error in file_info
        error_msg = str(e)
        file_info['errors'].append(error_msg)
        file_info['success'] = False
        
        # Register in log even if there's an error
        log_file_info(subject, fname, file_info)
        
        print(f"   ❌ ERROR: {subject} {experiment} - {error_msg}")
        
        return None, None, None
       
#%% Use project configuration

# EXECUTION MODE: Configured in config.py
if TEST_MODE:
    subjects = SUJETOS_TEST.copy()
    print(f"🧪 TEST MODE: Processing {len(subjects)} test subjects:")
    print(f"   {subjects}")
else:
    subjects = SUJETOS_VALIDOS.copy()
    print(f"📋 COMPLETE MODE: Processing {len(subjects)} valid subjects:")
    print(f"   {subjects}")

print(f"📊 Dose configuration loaded from config.py")

#### Note: Subjects with problematic EDA signal are documented in config.py

#%% Processing and saving by condition

# Create output directories organized by condition
output_dirs = {
    'dmt_high': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high'),
    'dmt_low': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)
    print(f"📁 Created directory: {dir_path}")

print(f"\n🔄 Starting EDA processing ...")
print(f"📋 Saving preprocessed data organized by condition")

def save_eda_data(df_eda, time, subject, experiment, condition, session_type):
    """
    Save preprocessed EDA data in CSV format organized by condition
    
    Args:
        df_eda: DataFrame with EDA data processed by NeuroKit
        time: Corresponding time series
        subject: Subject code (e.g., 'S04')
        experiment: Experiment type (e.g., 'DMT_1')
        condition: 'high' or 'low'
        session_type: 'dmt' or 'rs' (resting)
    """
    if df_eda is None:
        print(f"   ⚠️  No data to save for {subject} {experiment}")
        return
    
    # Create complete DataFrame with time and all EDA variables
    df_complete = df_eda.copy()
    df_complete.insert(0, 'time', time)
    
    # Generate descriptive filename
    # Format: {subject}_{session_type}_{experiment}_{condition}.csv
    filename = f"{subject}_{session_type}_{experiment}_{condition}.csv"
    
    # Determine output directory
    output_dir = output_dirs[f'dmt_{condition}']
    output_path = os.path.join(output_dir, filename)
    
    # Save file
    df_complete.to_csv(output_path, index=False)
    print(f"   💾 Saved: {filename} ({len(df_complete)} samples)")

# Process each subject
for subject in subjects:
    print(f"\n👤 Processing subject: {subject}")
    
    # Determine which session has which dose for this subject
    dose_session_1 = DOSIS['Dosis_Sesion_1'][subject]  # 'Alta' or 'Baja'
    dose_session_2 = DOSIS['Dosis_Sesion_2'][subject]  # 'Alta' or 'Baja'
    
    print(f"   📊 Session 1: {dose_session_1}, Session 2: {dose_session_2}")
    
    # Map dose to condition
    condition_map = {'Alta': 'high', 'Baja': 'low'}
    condition_1 = condition_map[dose_session_1]
    condition_2 = condition_map[dose_session_2]
    
    # Process and save Session 1 in the folder corresponding to its condition
    print(f"   🔄 Processing Session 1 (condition: {condition_1}) → folder dmt_{condition_1}")
    
    # DMT Session 1
    filename_dmt1 = get_nombre_archivo('DMT_1', subject)
    df_eda_dmt1, info_eda_dmt1, time_dmt1 = eda('DMT_1', subject, filename_dmt1)
    save_eda_data(df_eda_dmt1, time_dmt1, subject, 'session1', condition_1, 'dmt')
    
    # Resting Session 1  
    filename_rs1 = get_nombre_archivo('Reposo_1', subject)
    df_eda_rs1, info_eda_rs1, time_rs1 = eda('Reposo_1', subject, filename_rs1)
    save_eda_data(df_eda_rs1, time_rs1, subject, 'session1', condition_1, 'rs')
    
    # Process and save Session 2 in the folder corresponding to its condition  
    print(f"   🔄 Processing Session 2 (condition: {condition_2}) → folder dmt_{condition_2}")
    
    # DMT Session 2
    filename_dmt2 = get_nombre_archivo('DMT_2', subject)
    df_eda_dmt2, info_eda_dmt2, time_dmt2 = eda('DMT_2', subject, filename_dmt2)
    save_eda_data(df_eda_dmt2, time_dmt2, subject, 'session2', condition_2, 'dmt')
    
    # Resting Session 2
    filename_rs2 = get_nombre_archivo('Reposo_2', subject)
    df_eda_rs2, info_eda_rs2, time_rs2 = eda('Reposo_2', subject, filename_rs2)
    save_eda_data(df_eda_rs2, time_rs2, subject, 'session2', condition_2, 'rs')

#%% Final summary and logging

print(f"\n✅ EDA preprocessing completed successfully!")
print(f"📁 Generated file structure:")
print(f"   📂 {output_dirs['dmt_high']}")
print(f"   📂 {output_dirs['dmt_low']}")

# Save detailed processing log
log_path = save_processing_log()

# Log summary
successful_files = 0
total_files = 0
files_by_condition = {'dmt_high': 0, 'dmt_low': 0}

for subject in processing_log:
    for filename in processing_log[subject]:
        total_files += 1
        if processing_log[subject][filename]['success']:
            successful_files += 1

# Count files by condition
for condition in output_dirs:
    files_in_dir = len([f for f in os.listdir(output_dirs[condition]) if f.endswith('.csv')])
    files_by_condition[condition] = files_in_dir

print(f"\n📊 Summary of generated files:")
print(f"   Total files processed: {total_files}")
print(f"   Successful files: {successful_files}")
print(f"   Files with errors: {total_files - successful_files}")
print(f"   📈 Files by condition:")
print(f"      DMT High: {files_by_condition['dmt_high']} files")
print(f"      DMT Low: {files_by_condition['dmt_low']} files")

print(f"\n🔧 Data characteristics:")
print(f"   - No baseline correction applied")
print(f"   - All NeuroKit EDA variables included")
print(f"   - DMT: duration 20:15 min, Resting: duration 10:15 min")
print(f"   - Data organized by condition (high/low)")

print(f"\n📋 Detailed log saved to: {log_path}")

print(f"\n📝 Generated file format (example for {subjects[0]}):")
print(f"   📂 dmt_high/")
print(f"      {subjects[0]}_dmt_session1_high.csv   # DMT from high dose session")  
print(f"      {subjects[0]}_rs_session1_high.csv    # Resting from high dose session")
print(f"   📂 dmt_low/")
print(f"      {subjects[0]}_dmt_session2_low.csv    # DMT from low dose session")
print(f"      {subjects[0]}_rs_session2_low.csv     # Resting from low dose session")
print(f"   📌 Each folder contains only files from the session corresponding to that condition")

print(f"\n🎯 Data is ready for subsequent analysis")