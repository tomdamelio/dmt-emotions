#%%
# -*- coding: utf-8 -*-
"""
Physiological data preprocessing (EDA, ECG, RESP)

This script processes physiological data from BrainVision files (.vhdr) and generates 
preprocessed CSV files organized by experimental condition for subsequent analysis.

Features:
- Processes EDA, ECG, and RESP signals with all NeuroKit variables
- Applies adaptive duration strategy (truncation/NaN-padding)
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
import biosppy

# Import project configuration
import sys
import json
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    PHYSIOLOGY_DATA, DERIVATIVES_DATA, DOSIS, SUJETOS_VALIDOS, SUJETOS_TEST, TODOS_LOS_SUJETOS,
    TEST_MODE, PROCESSING_MODE, PATRONES_ARCHIVOS, DURACIONES_ESPERADAS, NEUROKIT_PARAMS, CANALES,
    TOLERANCIA_DURACION, EDA_ANALYSIS_CONFIG, get_nombre_archivo, get_duracion_esperada
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

def process_eda_emotiphai(eda_signal, sampling_rate=250.0, min_amplitude=0.05):
    """
    Process EDA signal using BioSPPy emotiphai_eda method to extract SCR features
    
    Args:
        eda_signal: Raw EDA signal
        sampling_rate: Sampling rate in Hz
        min_amplitude: Minimum threshold for SCR amplitude
        
    Returns:
        DataFrame: Contains SCR_Onsets, SCR_Peaks, SCR_Amplitudes as binary/continuous variables
    """
    try:
        # Use BioSPPy EDA processing to get cleaned signal
        eda_result = biosppy.signals.eda.eda(
            signal=np.array(eda_signal), 
            sampling_rate=sampling_rate, 
            show=False
        )
        
        # Extract filtered signal for emotiphai processing
        filtered_signal = eda_result['filtered']
        
        # Apply BioSPPy emotiphai_eda method directly
        emotiphai_result = biosppy.signals.eda.emotiphai_eda(
            signal=filtered_signal,
            sampling_rate=sampling_rate,
            min_amplitude=min_amplitude
        )
        
        # Extract results from BioSPPy emotiphai_eda
        # Debug: Print the structure of emotiphai_result to understand its format
        print(f"   🔍 Emotiphai result type: {type(emotiphai_result)}")
        print(f"   🔍 Emotiphai result attributes: {dir(emotiphai_result) if hasattr(emotiphai_result, '__dict__') else 'No __dict__'}")
        
        # Try different ways to access the results
        try:
            # Method 1: Direct attribute access
            onsets = emotiphai_result.onsets
            peaks = emotiphai_result.peaks
            amplitudes = emotiphai_result.amplitudes
        except AttributeError:
            try:
                # Method 2: Index access (if it's a tuple/list)
                onsets = emotiphai_result[0] if len(emotiphai_result) > 0 else []
                peaks = emotiphai_result[1] if len(emotiphai_result) > 1 else []
                amplitudes = emotiphai_result[2] if len(emotiphai_result) > 2 else []
                print(f"   🔧 Using index access for Emotiphai results")
            except (TypeError, IndexError):
                # Method 3: Check if it's a named tuple or similar
                if hasattr(emotiphai_result, '_fields'):
                    print(f"   🔍 Emotiphai result fields: {emotiphai_result._fields}")
                    onsets = getattr(emotiphai_result, emotiphai_result._fields[0], [])
                    peaks = getattr(emotiphai_result, emotiphai_result._fields[1], [])
                    amplitudes = getattr(emotiphai_result, emotiphai_result._fields[2], [])
                else:
                    # Fallback: empty arrays
                    onsets = []
                    peaks = []
                    amplitudes = []
                    print(f"   ⚠️  Could not extract Emotiphai results, using empty arrays")
        
        # Create per-event DataFrame (one row per detected SCR)
        num_events = min(len(onsets), len(peaks), len(amplitudes))
        if num_events > 0:
            emotiphai_df = pd.DataFrame({
                'SCR_Onsets_Emotiphai': np.asarray(onsets[:num_events], dtype=int),
                'SCR_Peaks_Emotiphai': np.asarray(peaks[:num_events], dtype=int),
                'SCR_Amplitudes_Emotiphai': np.asarray(amplitudes[:num_events], dtype=float)
            })
        else:
            emotiphai_df = pd.DataFrame({
                'SCR_Onsets_Emotiphai': np.array([], dtype=int),
                'SCR_Peaks_Emotiphai': np.array([], dtype=int),
                'SCR_Amplitudes_Emotiphai': np.array([], dtype=float)
            })
        
        print(f"   ✅ Emotiphai EDA detection completed: {len(onsets)} SCRs detected")
        return emotiphai_df
        
    except Exception as e:
        print(f"   ❌ Emotiphai EDA processing failed: {e}")
        # Return empty per-event DataFrame with expected columns
        empty_df = pd.DataFrame({
            'SCR_Onsets_Emotiphai': np.array([], dtype=int),
            'SCR_Peaks_Emotiphai': np.array([], dtype=int),
            'SCR_Amplitudes_Emotiphai': np.array([], dtype=float)
        })
        return empty_df

def process_eda_cvx_decomposition(eda_signal, sampling_rate=250.0):
    """
    Process EDA signal using BioSPPy cvx_decomposition method to extract EDA components
    
    Args:
        eda_signal: Raw EDA signal
        sampling_rate: Sampling rate in Hz
        
    Returns:
        DataFrame: Contains edr, smna, edl components from cvx_decomposition
    """
    try:
        # Apply BioSPPy cvx_decomposition method directly
        cvx_result = biosppy.signals.eda.cvx_decomposition(
            signal=np.array(eda_signal),
            sampling_rate=sampling_rate
        )
        
        # Extract the three arrays we want: edr, smna, edl
        # Debug: Print the structure of cvx_result to understand its format
        print(f"   🔍 CVX result type: {type(cvx_result)}")
        print(f"   🔍 CVX result attributes: {dir(cvx_result) if hasattr(cvx_result, '__dict__') else 'No __dict__'}")
        
        # Try different ways to access the results
        try:
            # Method 1: Direct attribute access
            edr = cvx_result.edr
            smna = cvx_result.smna  
            edl = cvx_result.edl
        except AttributeError:
            try:
                # Method 2: Index access (if it's a tuple/list)
                edr = cvx_result[0] if len(cvx_result) > 0 else np.zeros(len(eda_signal))
                smna = cvx_result[1] if len(cvx_result) > 1 else np.zeros(len(eda_signal))
                edl = cvx_result[2] if len(cvx_result) > 2 else np.zeros(len(eda_signal))
                print(f"   🔧 Using index access for CVX results")
            except (TypeError, IndexError):
                # Method 3: Check if it's a named tuple or similar
                if hasattr(cvx_result, '_fields'):
                    print(f"   🔍 CVX result fields: {cvx_result._fields}")
                    edr = getattr(cvx_result, cvx_result._fields[0], np.zeros(len(eda_signal)))
                    smna = getattr(cvx_result, cvx_result._fields[1], np.zeros(len(eda_signal)))
                    edl = getattr(cvx_result, cvx_result._fields[2], np.zeros(len(eda_signal)))
                else:
                    # Fallback: zero arrays
                    edr = np.zeros(len(eda_signal))
                    smna = np.zeros(len(eda_signal))
                    edl = np.zeros(len(eda_signal))
                    print(f"   ⚠️  Could not extract CVX results, using zero arrays")
        
        # Create DataFrame with cvx decomposition results (only the 3 arrays requested)
        cvx_df = pd.DataFrame({
            'EDR': edr,
            'SMNA': smna,
            'EDL': edl
        })
        
        print(f"   ✅ CVX decomposition completed: {len(cvx_df)} samples processed")
        return cvx_df
        
    except Exception as e:
        print(f"   ❌ CVX decomposition processing failed: {e}")
        # Return empty DataFrame with expected structure
        empty_df = pd.DataFrame({
            'EDR': np.zeros(len(eda_signal)),
            'SMNA': np.zeros(len(eda_signal)),
            'EDL': np.zeros(len(eda_signal))
        })
        return empty_df

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
        
        # Process each signal type with default methods
        signal_configs = {
            'eda': {'channel': CANALES['EDA'], 'process_func': nk.eda_process},  # Default method (neurokit)
            'ecg': {'channel': CANALES['ECG'], 'process_func': nk.ecg_process},  # Default method (neurokit)
            'resp': {'channel': CANALES['RESP'], 'process_func': nk.rsp_process}  # Default method (khodadad2018)
        }
        
        processed_data = {}
        processed_info = {}  # Store NeuroKit info dictionaries
        
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
                
                # Use default method for each signal type
                df_processed, info_processed = config['process_func'](
                    signal_data,
                    sampling_rate=sampling_rate
                )
                
                print(f"   ✅ {signal_name.upper()} processing completed: {len(df_processed)} samples")
                print(f"   📋 {signal_name.upper()} info dict keys: {list(info_processed.keys()) if info_processed else 'None'}")
                
                # Apply duration strategy to processed data
                current_samples = len(df_processed)
                
                if current_samples >= expected_samples:
                    # Truncate to exact duration
                    df_final = df_processed[:expected_samples].copy()
                    signal_info['strategy_applied'] = 'truncated'
                    print(f"   ✂️  {signal_name.upper()}: Truncated to {expected_duration_sec/60:.1f} min")
                else:
                    # NaN padding until expected duration
                    missing_samples = expected_samples - current_samples
                    
                    # Create DataFrame with padding (all columns as NaN)
                    padding_data = pd.DataFrame(
                        {col: np.nan for col in df_processed.columns}, 
                        index=range(len(df_processed), len(df_processed) + missing_samples)
                    )
                    
                    df_final = pd.concat([df_processed, padding_data], ignore_index=True)
                    signal_info['strategy_applied'] = 'nan_padded'
                    print(f"   📈 {signal_name.upper()}: NaN-padded with {missing_samples} samples")
                
                # Verify processing
                final_samples = len(df_final)
                assert final_samples == expected_samples, (
                    f"{signal_name.upper()} processing failed: expected {expected_samples}, got {final_samples}"
                )
                
                signal_info['final_samples'] = final_samples
                signal_info['success'] = True
                
                processed_data[signal_name] = df_final
                processed_info[signal_name] = info_processed  # Store NeuroKit info dict
                
            except Exception as e:
                error_msg = str(e)
                signal_info['errors'].append(error_msg)
                signal_info['success'] = False
                print(f"   ❌ {signal_name.upper()} ERROR: {error_msg}")
        
        # Create time series for final duration
        time_final = pd.Series(np.arange(expected_samples) / sampling_rate)
        
        print(f"   ✅ Final validation: {expected_samples} samples, {expected_duration_sec/60:.2f} min")
        
        # KBK analysis has been removed from the pipeline
        kbk_data = None
        
        # Process emotiphai EDA analysis for EDA if enabled and available
        emotiphai_data = None
        if EDA_ANALYSIS_CONFIG['emotiphai'] and 'eda' in processed_data and processed_data['eda'] is not None:
            print(f"   📊 Processing emotiphai EDA analysis for EDA...")
            # Get original EDA signal for emotiphai processing
            eda_channel = signal_configs['eda']['channel']
            if eda_channel in df_data.columns:
                original_eda = df_data[eda_channel]
                emotiphai_data = process_eda_emotiphai(original_eda, sampling_rate)
                # Enforce session duration on per-event Emotiphai results (threshold in samples)
                try:
                    threshold_samples = int(expected_duration_sec * sampling_rate)
                    if emotiphai_data is not None and not emotiphai_data.empty and 'SCR_Onsets_Emotiphai' in emotiphai_data.columns:
                        before_rows = len(emotiphai_data)
                        emotiphai_data = emotiphai_data[emotiphai_data['SCR_Onsets_Emotiphai'] <= threshold_samples].copy()
                        removed_rows = before_rows - len(emotiphai_data)
                        if removed_rows > 0:
                            print(f"   ✂️  Emotiphai: removed {removed_rows} events beyond session limit ({threshold_samples} samples)")
                except Exception as _:
                    # Never block pipeline on cleaning step
                    pass
                # Note: Emotiphai is per-event; do NOT truncate/pad to time length
        elif not EDA_ANALYSIS_CONFIG['emotiphai']:
            print(f"   ⏭️  Emotiphai EDA analysis disabled in config")
        
        # Process CVX decomposition analysis for EDA if enabled and available
        cvx_data = None
        if EDA_ANALYSIS_CONFIG['cvx'] and 'eda' in processed_data and processed_data['eda'] is not None:
            print(f"   📊 Processing CVX decomposition analysis for EDA...")
            # Get original EDA signal for CVX processing
            eda_channel = signal_configs['eda']['channel']
            if eda_channel in df_data.columns:
                original_eda = df_data[eda_channel]
                cvx_data = process_eda_cvx_decomposition(original_eda, sampling_rate)
                
                # Apply same duration strategy to CVX data
                if len(cvx_data) >= expected_samples:
                    cvx_data = cvx_data[:expected_samples].copy()
                    print(f"   ✂️  CVX: Truncated to {expected_duration_sec/60:.1f} min")
                elif len(cvx_data) < expected_samples:
                    missing_samples = expected_samples - len(cvx_data)
                    padding_data = pd.DataFrame(
                        {col: np.nan for col in cvx_data.columns}, 
                        index=range(len(cvx_data), len(cvx_data) + missing_samples)
                    )
                    cvx_data = pd.concat([cvx_data, padding_data], ignore_index=True)
                    print(f"   📈 CVX: NaN-padded with {missing_samples} samples")
        elif not EDA_ANALYSIS_CONFIG['cvx']:
            print(f"   ⏭️  CVX decomposition analysis disabled in config")
        
        # Validate EDA analysis dimensionality consistency (CVX only against time length)
        if cvx_data is not None:
            print(f"   🔍 EDA analysis dimensionality validation (CVX vs time series):")
            print(f"      CVX rows: {len(cvx_data)} | expected samples: {expected_samples}")
            try:
                assert len(cvx_data) == expected_samples, (
                    f"CVX dimensionality mismatch: expected {expected_samples}, got {len(cvx_data)}"
                )
            except AssertionError as e:
                print(f"   ❌ DIMENSIONALITY ERROR: {e}")
                raise e
        
        # Register in log
        log_file_info(subject, fname, file_info)
        
        return processed_data, time_final, processed_info, emotiphai_data, cvx_data
        
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
        
        return {}, None, {}, None, None
       
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

# Display EDA analysis configuration
print(f"\n📊 EDA Analysis Configuration:")
for analysis_type, enabled in EDA_ANALYSIS_CONFIG.items():
    status = "✅ ENABLED" if enabled else "❌ DISABLED"
    print(f"   {analysis_type.upper()}: {status}")

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

def save_physiology_data(processed_data, time, subject, experiment, condition, session_type, processed_info=None, emotiphai_data=None, cvx_data=None):
    """
    Save preprocessed physiological data in CSV format organized by signal type and condition
    
    Args:
        processed_data: Dict with signal data (keys: 'eda', 'ecg', 'resp')
        time: Corresponding time series
        subject: Subject code (e.g., 'S04')
        experiment: Experiment type (e.g., 'DMT_1')
        condition: 'high' or 'low'
        session_type: 'dmt' or 'rs' (resting)
        processed_info: Dict with NeuroKit info dictionaries for each signal type
        emotiphai_data: DataFrame with emotiphai EDA analysis results (only for EDA)
        cvx_data: DataFrame with CVX decomposition analysis results (only for EDA)
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
        
        # Save CSV file
        df_complete.to_csv(output_path, index=False)
        print(f"   💾 Saved {signal_name.upper()} data: {filename} ({len(df_complete)} samples)")
        
        # Save corresponding info dict as JSON if available
        if processed_info and signal_name in processed_info and processed_info[signal_name]:
            info_filename = f"{subject}_{session_type}_{experiment}_{condition}_info.json"
            info_path = os.path.join(output_dir, info_filename)
            
            # Prepare info dict for JSON serialization (handle numpy types)
            def convert_numpy_types(obj):
                """Recursively convert numpy types to JSON-serializable types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                else:
                    return obj
            
            info_to_save = convert_numpy_types(processed_info[signal_name])
            
            # Add metadata
            info_to_save['_metadata'] = {
                'signal_type': signal_name,
                'subject': subject,
                'experiment': experiment,
                'condition': condition,
                'session_type': session_type,
                'saved_timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info_to_save, f, indent=2, ensure_ascii=False)
            print(f"   📋 Saved {signal_name.upper()} info: {info_filename}")
            
        saved_files.append(f"{signal_name}:{filename}")
    
    # Save emotiphai EDA analysis for EDA signal (additional file)
    if emotiphai_data is not None and 'eda' in processed_data:
        print(f"   📊 Saving emotiphai EDA analysis for EDA...")
        
        # Emotiphai data contains SCR event information, no need for time column
        emotiphai_complete = emotiphai_data.copy()
        
        # Generate emotiphai filename with _kbk_scr_emotiphai_eda suffix
        emotiphai_filename = f"{subject}_{session_type}_{experiment}_{condition}_emotiphai_scr.csv"
        emotiphai_output_dir = output_dirs[f'eda_{condition}']
        emotiphai_output_path = os.path.join(emotiphai_output_dir, emotiphai_filename)
        
        # Save emotiphai CSV file
        emotiphai_complete.to_csv(emotiphai_output_path, index=False)
        print(f"   💾 Saved emotiphai EDA data: {emotiphai_filename} ({len(emotiphai_complete)} samples)")
        
        saved_files.append(f"emotiphai_eda:{emotiphai_filename}")
    
    # Save CVX decomposition analysis for EDA signal (additional file)
    if cvx_data is not None and 'eda' in processed_data:
        print(f"   📊 Saving CVX decomposition analysis for EDA...")
        
        # Create CVX DataFrame with time
        cvx_complete = cvx_data.copy()
        cvx_complete.insert(0, 'time', time)
        
        # Generate CVX filename with _cvx_decomposition suffix
        cvx_filename = f"{subject}_{session_type}_{experiment}_{condition}_cvx_decomposition.csv"
        cvx_output_dir = output_dirs[f'eda_{condition}']
        cvx_output_path = os.path.join(cvx_output_dir, cvx_filename)
        
        # Save CVX CSV file
        cvx_complete.to_csv(cvx_output_path, index=False)
        print(f"   💾 Saved CVX decomposition data: {cvx_filename} ({len(cvx_complete)} samples)")
        
        saved_files.append(f"cvx_decomposition:{cvx_filename}")
    
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
    processed_dmt1, time_dmt1, info_dmt1, emotiphai_dmt1, cvx_dmt1 = process_physiology('DMT_1', subject, filename_dmt1)
    save_physiology_data(processed_dmt1, time_dmt1, subject, 'session1', condition_1, 'dmt', info_dmt1, emotiphai_dmt1, cvx_dmt1)
    
    # Resting Session 1  
    filename_rs1 = get_nombre_archivo('Reposo_1', subject)
    processed_rs1, time_rs1, info_rs1, emotiphai_rs1, cvx_rs1 = process_physiology('Reposo_1', subject, filename_rs1)
    save_physiology_data(processed_rs1, time_rs1, subject, 'session1', condition_1, 'rs', info_rs1, emotiphai_rs1, cvx_rs1)
    
    # Process and save Session 2 in the folder corresponding to its condition  
    print(f"   🔄 Processing Session 2 (condition: {condition_2}) → folder dmt_{condition_2}")
    
    # DMT Session 2
    filename_dmt2 = get_nombre_archivo('DMT_2', subject)
    processed_dmt2, time_dmt2, info_dmt2, emotiphai_dmt2, cvx_dmt2 = process_physiology('DMT_2', subject, filename_dmt2)
    save_physiology_data(processed_dmt2, time_dmt2, subject, 'session2', condition_2, 'dmt', info_dmt2, emotiphai_dmt2, cvx_dmt2)
    
    # Resting Session 2
    filename_rs2 = get_nombre_archivo('Reposo_2', subject)
    processed_rs2, time_rs2, info_rs2, emotiphai_rs2, cvx_rs2 = process_physiology('Reposo_2', subject, filename_rs2)
    save_physiology_data(processed_rs2, time_rs2, subject, 'session2', condition_2, 'rs', info_rs2, emotiphai_rs2, cvx_rs2)

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