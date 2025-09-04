#%%
# -*- coding: utf-8 -*-
"""
Physiological data preprocessing (EDA, ECG, RESP)

This script processes physiological data from BrainVision files (.vhdr) and generates 
preprocessed CSV files organized by experimental condition for subsequent analysis.

Features:
- Processes EDA, ECG, and RESP signals with all NeuroKit variables
- Applies adaptive duration strategy (truncation/zero-padding)
- No baseline correction applied (preserves raw processed data)
- Organizes output by dose condition (high/low) and signal type
- JSON logging of all processing steps (hierarchical by signal)

Input: .vhdr files in ../data/original/physiology/
Output: Individual CSV files organized in:
- ../data/derivatives/phys/eda/dmt_high/ (EDA high dose session files)
- ../data/derivatives/phys/eda/dmt_low/ (EDA low dose session files)
- ../data/derivatives/phys/ecg/dmt_high/ (ECG high dose session files)
- ../data/derivatives/phys/ecg/dmt_low/ (ECG low dose session files)
- ../data/derivatives/phys/resp/dmt_high/ (RESP high dose session files)
- ../data/derivatives/phys/resp/dmt_low/ (RESP low dose session files)

Each CSV contains signal-specific variables from NeuroKit processing
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
    PHYSIOLOGY_DATA, DERIVATIVES_DATA, DOSIS, SUJETOS_VALIDOS, SUJETOS_TEST, TODOS_LOS_SUJETOS,
    TEST_MODE, PROCESSING_MODE, PATRONES_ARCHIVOS, DURACIONES_ESPERADAS, NEUROKIT_PARAMS, CANALES,
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
            'script': 'preprocess_phys.py',
            'total_subjects': len(processing_log),
            'config_file': 'config.py'
        },
        'subjects': processing_log
    }
    
    log_file = os.path.join(log_dir, 'physiology_preprocessing_log.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Processing log saved to: {log_file}")
    return log_file

def process_physiology(experiment, subject, fname):
    
    file_path = os.path.join(PHYSIOLOGY_DATA, experiment, subject, fname)
    
    # Initialize file information for logging (hierarchical by signal)
    file_info = {
        'experiment': experiment,
        'subject': subject,
        'filename': fname,
        'file_path': file_path,
        'original_samples': 0,
        'original_duration_sec': 0.0,
        'original_duration_min': 0.0,
        'sampling_rate': 0.0,
        'signals': {
            'eda': {
                'success': False,
                'has_channel': False,
                'processing_method': 'default (neurokit)',
                'final_samples': 0,
                'strategy_applied': None,
                'errors': []
            },
            'ecg': {
                'success': False,
                'has_channel': False,
                'processing_method': 'default (neurokit)',
                'final_samples': 0,
                'strategy_applied': None,
                'errors': []
            },
            'resp': {
                'success': False,
                'has_channel': False,
                'processing_method': 'default (khodadad2018)',
                'final_samples': 0,
                'strategy_applied': None,
                'errors': []
            }
        }
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
        
        # Get expected duration from configuration
        expected_duration_sec = get_duracion_esperada(experiment)
        expected_samples = int(expected_duration_sec * sampling_rate)
        
        # Process each signal type
        signal_configs = {
            'eda': {'channel': CANALES['EDA'], 'process_func': nk.eda_process},
            'ecg': {'channel': CANALES['ECG'], 'process_func': nk.ecg_process},
            'resp': {'channel': CANALES['RESP'], 'process_func': nk.rsp_process}
        }
        
        processed_data = {}
        
        for signal_name, config in signal_configs.items():
            signal_info = file_info['signals'][signal_name]
            
            try:
                # Check if channel exists
                channel_name = config['channel']
                if channel_name not in df_data.columns:
                    signal_info['has_channel'] = False
                    signal_info['errors'].append(f"{channel_name} channel not found")
                    print(f"   ⚠️  {signal_name.upper()}: {channel_name} channel not found")
                    continue
                
                signal_info['has_channel'] = True
                
                # Process signal with NeuroKit using default methods
                signal_data = df_data[channel_name]
                print(f"   🔄 Processing {signal_name.upper()}...")
                
                # Use default method for each signal type (don't specify method parameter)
                df_processed, info_processed = config['process_func'](
                    signal_data,
                    sampling_rate=sampling_rate
                )
                
                print(f"   ✅ {signal_name.upper()} processing completed: {len(df_processed)} samples")
                
                # Apply duration strategy to processed data
                current_samples = len(df_processed)
                
                if current_samples >= expected_samples:
                    # Truncate to exact duration
                    df_final = df_processed[:expected_samples].copy()
                    signal_info['strategy_applied'] = 'truncated'
                    print(f"   ✂️  {signal_name.upper()}: Truncated to {expected_duration_sec/60:.1f} min")
                else:
                    # Zero padding until expected duration
                    missing_samples = expected_samples - current_samples
                    
                    # Create DataFrame with padding (all columns as 0)
                    padding_data = pd.DataFrame(
                        {col: 0.0 for col in df_processed.columns}, 
                        index=range(len(df_processed), len(df_processed) + missing_samples)
                    )
                    
                    df_final = pd.concat([df_processed, padding_data], ignore_index=True)
                    signal_info['strategy_applied'] = 'zero_padded'
                    print(f"   📈 {signal_name.upper()}: Zero-padded with {missing_samples} samples")
                
                # Verify processing
                final_samples = len(df_final)
                assert final_samples == expected_samples, (
                    f"{signal_name.upper()} processing failed: expected {expected_samples}, got {final_samples}"
                )
                
                signal_info['final_samples'] = final_samples
                signal_info['success'] = True
                
                processed_data[signal_name] = df_final
                
            except Exception as e:
                error_msg = str(e)
                signal_info['errors'].append(error_msg)
                signal_info['success'] = False
                print(f"   ❌ {signal_name.upper()} ERROR: {error_msg}")
        
        # Create time series for final duration
        time_final = pd.Series(np.arange(expected_samples) / sampling_rate)
        
        print(f"   ✅ Final validation: {expected_samples} samples, {expected_duration_sec/60:.2f} min")
        
        # Register in log
        log_file_info(subject, fname, file_info)
        
        return processed_data, time_final
        
    except Exception as e:
        # Register global error in file_info  
        error_msg = str(e)
        print(f"   ❌ GLOBAL ERROR: {subject} {experiment} - {error_msg}")
        
        # Mark all signals as failed
        for signal_name in file_info['signals']:
            file_info['signals'][signal_name]['errors'].append(f"Global error: {error_msg}")
            file_info['signals'][signal_name]['success'] = False
        
        # Register in log even if there's an error
        log_file_info(subject, fname, file_info)
        
        return {}, None
       
#%% Use project configuration

# EXECUTION MODE: Configured in config.py
if TEST_MODE:
    subjects = SUJETOS_TEST.copy()
    print(f"🧪 TEST MODE: Processing {len(subjects)} test subjects:")
    print(f"   {subjects}")
else:
    if PROCESSING_MODE == 'ALL':
        subjects = TODOS_LOS_SUJETOS.copy()
        print(f"📋 ALL SUBJECTS MODE: Processing {len(subjects)} subjects (S01-S20, except S14):")
        print(f"   {subjects}")
    else:  # PROCESSING_MODE == 'VALID'
        subjects = SUJETOS_VALIDOS.copy()
        print(f"📋 VALID SUBJECTS MODE: Processing {len(subjects)} valid subjects:")
        print(f"   {subjects}")

print(f"📊 Dose configuration loaded from config.py")

#### Note: Subjects with problematic EDA signal are documented in config.py

#%% Processing and saving by condition

# Create output directories organized by signal type and condition
output_dirs = {}
for signal in ['eda', 'ecg', 'resp']:
    for condition in ['high', 'low']:
        key = f'{signal}_{condition}'
        output_dirs[key] = os.path.join(DERIVATIVES_DATA, 'phys', signal, f'dmt_{condition}')
        os.makedirs(output_dirs[key], exist_ok=True)
        print(f"📁 Created directory: {output_dirs[key]}")

print(f"\n🔄 Starting physiological data processing ...")
print(f"📋 Saving preprocessed data organized by signal type and condition")

def save_physiology_data(processed_data, time, subject, experiment, condition, session_type):
    """
    Save preprocessed physiological data in CSV format organized by signal type and condition
    
    Args:
        processed_data: Dict with signal data (keys: 'eda', 'ecg', 'resp')
        time: Corresponding time series
        subject: Subject code (e.g., 'S04')
        experiment: Experiment type (e.g., 'DMT_1')
        condition: 'high' or 'low'
        session_type: 'dmt' or 'rs' (resting)
    """
    if not processed_data:
        print(f"   ⚠️  No data to save for {subject} {experiment}")
        return
    
    saved_files = []
    
    # Save each signal type separately
    for signal_name, df_signal in processed_data.items():
        if df_signal is None:
            print(f"   ⚠️  No {signal_name.upper()} data to save for {subject} {experiment}")
            continue
        
        # Create complete DataFrame with time and all signal variables
        df_complete = df_signal.copy()
        df_complete.insert(0, 'time', time)
        
        # Generate descriptive filename
        # Format: {subject}_{session_type}_{experiment}_{condition}.csv
        filename = f"{subject}_{session_type}_{experiment}_{condition}.csv"
        
        # Determine output directory for this signal type
        output_dir = output_dirs[f'{signal_name}_{condition}']
        output_path = os.path.join(output_dir, filename)
        
        # Save file
        df_complete.to_csv(output_path, index=False)
        print(f"   💾 Saved {signal_name.upper()}: {filename} ({len(df_complete)} samples)")
        saved_files.append(f"{signal_name}:{filename}")
    
    return saved_files

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
    processed_dmt1, time_dmt1 = process_physiology('DMT_1', subject, filename_dmt1)
    save_physiology_data(processed_dmt1, time_dmt1, subject, 'session1', condition_1, 'dmt')
    
    # Resting Session 1  
    filename_rs1 = get_nombre_archivo('Reposo_1', subject)
    processed_rs1, time_rs1 = process_physiology('Reposo_1', subject, filename_rs1)
    save_physiology_data(processed_rs1, time_rs1, subject, 'session1', condition_1, 'rs')
    
    # Process and save Session 2 in the folder corresponding to its condition  
    print(f"   🔄 Processing Session 2 (condition: {condition_2}) → folder dmt_{condition_2}")
    
    # DMT Session 2
    filename_dmt2 = get_nombre_archivo('DMT_2', subject)
    processed_dmt2, time_dmt2 = process_physiology('DMT_2', subject, filename_dmt2)
    save_physiology_data(processed_dmt2, time_dmt2, subject, 'session2', condition_2, 'dmt')
    
    # Resting Session 2
    filename_rs2 = get_nombre_archivo('Reposo_2', subject)
    processed_rs2, time_rs2 = process_physiology('Reposo_2', subject, filename_rs2)
    save_physiology_data(processed_rs2, time_rs2, subject, 'session2', condition_2, 'rs')

#%% Final summary and logging

print(f"\n✅ Physiological preprocessing completed successfully!")
print(f"📁 Generated file structure:")
for signal in ['eda', 'ecg', 'resp']:
    print(f"   📂 {signal.upper()}:")
    print(f"      📂 {output_dirs[f'{signal}_high']}")
    print(f"      📂 {output_dirs[f'{signal}_low']}")

# Save detailed processing log
log_path = save_processing_log()

# Log summary
successful_signals = {'eda': 0, 'ecg': 0, 'resp': 0}
total_files = 0

for subject in processing_log:
    for filename in processing_log[subject]:
        total_files += 1
        for signal in ['eda', 'ecg', 'resp']:
            if processing_log[subject][filename]['signals'][signal]['success']:
                successful_signals[signal] += 1

# Count files by signal and condition
files_by_signal = {}
for signal in ['eda', 'ecg', 'resp']:
    files_by_signal[signal] = {'high': 0, 'low': 0}
    for condition in ['high', 'low']:
        dir_key = f'{signal}_{condition}'
        if dir_key in output_dirs and os.path.exists(output_dirs[dir_key]):
            files_in_dir = len([f for f in os.listdir(output_dirs[dir_key]) if f.endswith('.csv')])
            files_by_signal[signal][condition] = files_in_dir

print(f"\n📊 Summary of generated files:")
print(f"   Total files processed: {total_files}")
print(f"   📈 Successful signals per type:")
for signal, count in successful_signals.items():
    print(f"      {signal.upper()}: {count} successful processings")
print(f"   📈 Files generated by signal and condition:")
for signal in ['eda', 'ecg', 'resp']:
    print(f"      {signal.upper()}: High={files_by_signal[signal]['high']}, Low={files_by_signal[signal]['low']}")

print(f"\n🔧 Data characteristics:")
print(f"   - No baseline correction applied")
print(f"   - All NeuroKit variables included per signal type")
print(f"   - DMT: duration 20:15 min, Resting: duration 10:15 min")
print(f"   - Data organized by signal type and condition (high/low)")

print(f"\n📋 Detailed log saved to: {log_path}")

print(f"\n📝 Generated file format (example for {subjects[0]}):")
for signal in ['eda', 'ecg', 'resp']:
    print(f"   📂 {signal}/")
    print(f"      📂 dmt_high/")
    print(f"         {subjects[0]}_dmt_session1_high.csv   # {signal.upper()} DMT from high dose session")  
    print(f"         {subjects[0]}_rs_session1_high.csv    # {signal.upper()} Resting from high dose session")
    print(f"      📂 dmt_low/")
    print(f"         {subjects[0]}_dmt_session2_low.csv    # {signal.upper()} DMT from low dose session")
    print(f"         {subjects[0]}_rs_session2_low.csv     # {signal.upper()} Resting from low dose session")

print(f"\n🎯 Data is ready for subsequent analysis")