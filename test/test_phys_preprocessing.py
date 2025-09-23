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
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import time

# Configure matplotlib for better plot display
plt.ion()  # Turn on interactive mode

# Add parent directory to path to import from scripts and config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import (
        DERIVATIVES_DATA, SUJETOS_TEST, SUJETOS_VALIDOS, TODOS_LOS_SUJETOS,
        TEST_MODE, PROCESSING_MODE, DURACIONES_ESPERADAS, NEUROKIT_PARAMS
    )
except ImportError:
    print("❌ Could not import config. Make sure config.py exists in parent directory.")
    sys.exit(1)

def load_validation_log():
    """Carga el archivo de validación manual"""
    validation_path = 'validation_log.json'
    
    if not os.path.exists(validation_path):
        print(f"⚠️  Archivo de validación no encontrado: {validation_path}")
        return None
    
    try:
        with open(validation_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error cargando archivo de validación: {e}")
        return None

def save_validation_log(validation_data):
    """Guarda el archivo de validación manual actualizado"""
    validation_path = 'validation_log.json'
    
    try:
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Archivo de validación actualizado: {validation_path}")
    except Exception as e:
        print(f"❌ Error guardando archivo de validación: {e}")

def update_file_existence_info(validation_data, subject, signal_type, file_key, file_path, file_exists, extra_fields=None):
    """
    Actualiza la información de existencia de archivo en el log de validación
    
    Args:
        validation_data: Datos del archivo de validación
        subject: Código del sujeto
        signal_type: Tipo de señal
        file_key: Clave del archivo
        file_path: Ruta del archivo
        file_exists: Si el archivo existe
    """
    if validation_data is None:
        return
    
    try:
        # Asegurar que existe la estructura (nueva jerarquía: subject -> file_key -> signal_type)
        if 'subjects' not in validation_data:
            validation_data['subjects'] = {}
        if subject not in validation_data['subjects']:
            validation_data['subjects'][subject] = {}
        if file_key not in validation_data['subjects'][subject]:
            validation_data['subjects'][subject][file_key] = {}
        if signal_type not in validation_data['subjects'][subject][file_key]:
            validation_data['subjects'][subject][file_key][signal_type] = {
                'category': '',
                'notes': '',
                'file_exists': None,
                'file_path': ''
            }
        
        # Actualizar información de existencia
        file_entry = validation_data['subjects'][subject][file_key][signal_type]
        file_entry['file_exists'] = file_exists
        file_entry['file_path'] = file_path
        
        # Agregar campos extra opcionales (p. ej., rutas derivadas)
        if isinstance(extra_fields, dict):
            for k, v in extra_fields.items():
                file_entry[k] = v
            
        # Agregar nota automática si el archivo no existe
        if not file_exists:
            existing_notes = file_entry.get('notes', '').strip()
            auto_note = "⚠️ Archivo no encontrado"
            if auto_note not in existing_notes:
                if existing_notes:
                    file_entry['notes'] = f"{existing_notes} | {auto_note}"
                else:
                    file_entry['notes'] = auto_note
                    
    except Exception as e:
        print(f"⚠️  Error actualizando información de archivo {subject}/{file_key}/{signal_type}: {e}")

def collect_manual_validation_input(validation_data, subject, signal_type, file_key, file_exists, plot_title):
    """
    Recolecta input manual del usuario para validación de señales
    
    Args:
        subject: Código del sujeto
        signal_type: Tipo de señal
        file_key: Clave del archivo
        file_exists: Si el archivo existe
        plot_title: Título del plot para contexto
        
    Returns:
        tuple: (category, notes) - input del usuario
    """
    print(f"\n{'='*60}")
    print(f"📋 VALIDACIÓN MANUAL - {plot_title}")
    print(f"{'='*60}")
    
    # Cargar valores previos como defaults si existen
    default_category = ""
    default_notes = ""
    if validation_data is not None:
        try:
            prev_entry = (
                validation_data.get('subjects', {})
                .get(subject, {})
                .get(file_key, {})
                .get(signal_type, {})
            )
            default_category = (prev_entry.get('category') or "").strip()
            default_notes = (prev_entry.get('notes') or "").strip()
        except Exception:
            pass

    if not file_exists and not default_category and not default_notes:
        print(f"⚠️  ARCHIVO NO ENCONTRADO - No se puede validar la calidad de la señal")
        default_category = "bad"
        default_notes = "Archivo no encontrado"
    
    print(f"📊 Archivo: {subject}/{file_key}/{signal_type}")
    print(f"📈 Revisa el plot interactivo (ventana abierta) y evalúa la calidad de la señal")
    print(f"   💡 Puedes interactuar con el plot (zoom, pan) mientras anotas")
    print(f"\n📝 Opciones de categoría:")
    print(f"   good       - Señal de buena calidad, lista para análisis")
    print(f"   acceptable  - Señal usable con algunas limitaciones") 
    print(f"   maybe      - Señal dudosa; revisar más adelante")
    print(f"   bad        - Señal no usable para análisis")
    
    # Mostrar defaults si existen
    if default_category or default_notes:
        print(f"\n🗂️  Valores previos detectados (se usarán si respondes 'ok'):")
        if default_category:
            print(f"   Categoría previa: {default_category}")
        if default_notes:
            print(f"   Notas previas: {default_notes}")

    # Solicitar categoría (permitiendo 'ok' para mantener)
    while True:
        extra = "/ok=mantener" if default_category else ""
        cat_in = input(f"\n🔍 Categoría para {signal_type.upper()} [{subject}/{file_key}] (good/acceptable/maybe/bad{extra}): ").strip().lower()
        if cat_in == 'ok' and default_category:
            category = default_category
            category_changed = False
            break
        if cat_in in ['good', 'acceptable', 'maybe', 'bad']:
            category = cat_in
            category_changed = (category != default_category)
            break
        print("❌ Por favor ingresa: good, acceptable, maybe, bad" + (" o ok" if default_category else ""))
    
    # Solicitar notas (Q=omitir, ok=mantener)
    print(f"\n📝 Notas cualitativas para {signal_type.upper()} (Q=omitir, ok=mantener):")
    print(f"   Ejemplo: 'Artefactos al inicio', 'Señal estable', 'Ruido en minuto 5-8', etc.")
    while True:
        raw_notes = input(f"💬 Notas (Q para omitir, ok para mantener): ").strip()
        if raw_notes.lower() == 'ok' and (default_notes or default_notes == ""):
            notes = default_notes
            notes_changed = False
            break
        if raw_notes.lower() == 'q':
            notes = ""
            notes_changed = (notes != default_notes)
            break
        if raw_notes == "":
            print("❌ Por favor escribe alguna nota, o usa Q para omitir, u ok para mantener")
            continue
        notes = raw_notes
        notes_changed = (notes != default_notes)
        break
    
    print(f"✅ Registrado - {signal_type.upper()}: {category}" + (f" | {notes}" if notes else ""))
    
    # Ask if user wants to close the plot
    while True:
        close_plot = input(f"\n🔍 ¿Cerrar la ventana del plot? (si/no): ").strip().lower()
        if close_plot in ['si', 'no']:
            break
        print("❌ Por favor responde: si o no")
    
    if close_plot == 'si':
        plt.close('all')
        print(f"   🔒 Plot cerrado")
    else:
        print(f"   📊 Plot permanece abierto")
    
    changed = category_changed or notes_changed
    return category, notes, changed

def update_manual_validation_info(validation_data, subject, signal_type, file_key, category, notes):
    """
    Actualiza los campos manuales (category y notes) en el log de validación
    
    Args:
        validation_data: Datos del archivo de validación
        subject: Código del sujeto
        signal_type: Tipo de señal
        file_key: Clave del archivo
        category: Categoría de validación (good/acceptable/maybe/bad)
        notes: Notas cualitativas del usuario
    """
    if validation_data is None:
        return
    
    try:
        # Asegurar que existe la estructura
        if 'subjects' not in validation_data:
            validation_data['subjects'] = {}
        if subject not in validation_data['subjects']:
            validation_data['subjects'][subject] = {}
        if file_key not in validation_data['subjects'][subject]:
            validation_data['subjects'][subject][file_key] = {}
        if signal_type not in validation_data['subjects'][subject][file_key]:
            validation_data['subjects'][subject][file_key][signal_type] = {
                'category': '',
                'notes': '',
                'file_exists': None,
                'file_path': ''
            }
        
        # Actualizar campos manuales
        file_entry = validation_data['subjects'][subject][file_key][signal_type]
        file_entry['category'] = category
        
        # Manejar notas: preservar notas automáticas existentes y agregar las manuales
        existing_notes = file_entry.get('notes', '').strip()
        if notes:
            if existing_notes and not existing_notes.startswith("⚠️"):
                # Si hay notas existentes no automáticas, combinar
                file_entry['notes'] = f"{existing_notes} | {notes}"
            elif existing_notes.startswith("⚠️"):
                # Si hay notas automáticas, agregar las manuales
                file_entry['notes'] = f"{existing_notes} | {notes}"
            else:
                # Solo notas manuales
                file_entry['notes'] = notes
        # Si no hay notas manuales, mantener las existentes (automáticas)
                    
    except Exception as e:
        print(f"⚠️  Error actualizando validación manual {subject}/{file_key}/{signal_type}: {e}")

def parse_file_description_to_validation_key(description, subject, file_path):
    """
    Convierte la descripción de archivo del test a la clave usada en validation_log
    
    Args:
        description: Descripción como "EDA DMT High Dose" o "RESP Resting Low Dose"
        subject: Código del sujeto
        file_path: Ruta del archivo para extraer información adicional
        
    Returns:
        str: file_key como "dmt_session1_high" o None si no se puede parsear
    """
    try:
        # Las descripciones reales son:
        # "EDA DMT High Dose" -> necesitamos extraer session del filename
        # "EDA Resting Low Dose" -> necesitamos extraer session del filename
        
        # Extraer información del filename (ej: "S01_dmt_session2_low.csv")
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        parts = filename_no_ext.split('_')
        
        if len(parts) >= 4:
            # parts = ["S01", "dmt", "session2", "low"]
            session_type = parts[1]  # "dmt" o "rs"
            session_full = parts[2]  # "session1" o "session2"
            condition = parts[3]     # "high" o "low"
            
            file_key = f"{session_type}_{session_full}_{condition}"
            return file_key
        else:
            print(f"⚠️  No se pudo parsear el filename: {filename}")
            return None
        
    except Exception as e:
        print(f"⚠️  Error parseando descripción '{description}' con archivo '{file_path}': {e}")
        return None

def create_plot_title_from_filename(file_path, signal_type, subject):
    """
    Crea un título estructurado para los plots basado en el nombre del archivo
    
    Args:
        file_path: Ruta del archivo como "path/S01_dmt_session2_low.csv"
        signal_type: Tipo de señal ('eda', 'ecg', 'resp')
        subject: Código del sujeto
        
    Returns:
        str: Título formateado como "S01 - EDA - DMT - ses02 - low"
    """
    try:
        # Extraer el nombre del archivo sin la extensión
        filename = os.path.basename(file_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Ejemplo: "S01_dmt_session2_low" -> ["S01", "dmt", "session2", "low"]
        parts = filename_no_ext.split('_')
        
        if len(parts) >= 4:
            # parts[0] = subject (ej: "S01")
            # parts[1] = session_type (ej: "dmt" o "rs")
            # parts[2] = session (ej: "session1" o "session2") 
            # parts[3] = condition (ej: "high" o "low")
            
            file_subject = parts[0]
            session_type = parts[1]
            session_full = parts[2]
            condition = parts[3]
            
            # Convertir session_type a nombre legible
            if session_type == "dmt":
                condition_type = "DMT"
            elif session_type == "rs":
                condition_type = "Rest"
            else:
                condition_type = session_type.upper()
            
            # Convertir session a formato ses##
            if session_full == "session1":
                session = "ses01"
            elif session_full == "session2":
                session = "ses02"
            else:
                session = session_full
            
            # Crear título estructurado
            title = f"{file_subject} - {signal_type.upper()} - {condition_type} - {session} - {condition}"
            return title
            
        else:
            # Fallback si no se puede parsear
            print(f"⚠️  No se pudo parsear el nombre del archivo: {filename}")
            return f"{subject} - {signal_type.upper()} - {filename_no_ext}"
        
    except Exception as e:
        print(f"⚠️  Error creando título desde filename '{file_path}': {e}")
        # Fallback al título básico
        return f"{subject} - {signal_type.upper()} - {os.path.basename(file_path)}"

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

def load_physiological_info(file_path, signal_type):
    """
    Load NeuroKit2 info dictionary from JSON file corresponding to CSV data
    
    Args:
        file_path (str): Path to the CSV file (will derive JSON path from this)
        signal_type (str): Type of signal ('eda', 'ecg', 'resp')
    
    Returns:
        dict: NeuroKit info dictionary or None if not found
    """
    try:
        # Derive info JSON path from CSV path
        # Example: S01_dmt_session2_low.csv -> S01_dmt_session2_low_info.json
        csv_dir = os.path.dirname(file_path)
        csv_filename = os.path.basename(file_path)
        csv_name_no_ext = os.path.splitext(csv_filename)[0]
        info_filename = f"{csv_name_no_ext}_info.json"
        info_path = os.path.join(csv_dir, info_filename)
        
        if not os.path.exists(info_path):
            print(f"   ⚠️  Info file not found: {info_filename}")
            return None
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info_dict = json.load(f)
        
        print(f"   📋 Loaded {signal_type.upper()} info: {info_filename}")
        return info_dict
        
    except Exception as e:
        print(f"   ❌ Error loading info file: {e}")
        return None

def load_physiological_data(file_path, signal_type):
    """
    Load physiological data from CSV file and separate time from signal variables
    
    Args:
        file_path (str): Path to the CSV file
        signal_type (str): Type of signal ('eda', 'ecg', 'resp')
    
    Returns:
        tuple: (signal_df, time_series, info_dict) where signal_df contains only physiological variables
    """
    # Normalize path to avoid accidental artifacts
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        # Try to auto-correct common path typos observed in runs
        alt_path = (
            file_path
            .replace(os.path.join('dmt_highh'), os.path.join('dmt_high'))
            .replace(os.path.join('dmt_higgh'), os.path.join('dmt_high'))
            .replace(os.path.join('dmt_loww'), os.path.join('dmt_low'))
        )
        if alt_path != file_path and os.path.exists(alt_path):
            print(f"⚠️  Path normalized from '{file_path}' to '{alt_path}'")
            file_path = alt_path
        else:
            print(f"⚠️  File not found: {file_path}")
            return None, None, None
    
    # Load complete data
    df_complete = pd.read_csv(file_path)
    
    # Separate time from signal variables
    time_series = df_complete['time'].copy()
    signal_df = df_complete.drop('time', axis=1).copy()
    
    print(f"✅ Loaded {os.path.basename(file_path)}: {len(signal_df)} samples, {len(signal_df.columns)} {signal_type.upper()} variables")
    print(f"   {signal_type.upper()} columns: {list(signal_df.columns)}")
    
    # Load corresponding info dictionary
    info_dict = load_physiological_info(file_path, signal_type)
    
    return signal_df, time_series, info_dict

def plot_physiological_signals(signal_df, time_series, file_path, signal_type, subject, sampling_rate=None, info_dict=None):
    """
    Plot physiological signals using appropriate NeuroKit functions with proper info dict
    Returns True if plot was displayed successfully, False otherwise
    
    Args:
        signal_df: DataFrame with signal variables
        time_series: Time series data
        file_path: Path to the file being plotted
        signal_type: Type of signal ('eda', 'ecg', 'resp')
        subject: Subject code
        sampling_rate: Sampling rate (Hz), defaults to config value if None
        info_dict: NeuroKit info dictionary for proper plotting
        
    Returns:
        bool: True if plot was successfully displayed, False otherwise
    """
    # Create structured title from filename
    plot_title = create_plot_title_from_filename(file_path, signal_type, subject)
    
    if sampling_rate is None:
        sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
    
    print(f"   📊 Using sampling rate: {sampling_rate} Hz for plotting")
    if signal_df is None or time_series is None:
        print(f"⚠️  Cannot plot {plot_title} - data not available")
        return False
    
    try:
        # Try using NeuroKit2 plotting functions with info dict if available
        if info_dict:
            print(f"   📋 Using NeuroKit2 plotting with info dict (sampling_rate: {sampling_rate} Hz)")
            try:
                # Close any existing plots first
                plt.close('all')
                
                # Create the plot
                if signal_type == 'eda':
                    nk.eda_plot(signal_df, info_dict)
                elif signal_type == 'ecg':
                    nk.ecg_plot(signal_df, info_dict)
                elif signal_type == 'resp':
                    nk.rsp_plot(signal_df, info_dict)
                else:
                    raise ValueError(f"Unknown signal type: {signal_type}")
                
                # Customize the plot with structured title
                plt.suptitle(plot_title, fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save plot with structured filename
                plot_dir = os.path.join('test', 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                plot_filename = f"{plot_title.lower().replace(' ', '_').replace('-', '_').replace(':', '')}.png"
                plot_path = os.path.join(plot_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                
                print(f"📊 {signal_type.upper()} plot saved: {plot_path}")
                
                # Display the interactive plot without blocking
                print(f"🖥️  Mostrando plot interactivo de NeuroKit2...")
                print(f"   💡 El plot permanecerá abierto mientras haces las anotaciones")
                
                # Ensure the plot is rendered and displayed
                plt.draw()
                plt.pause(0.1)
                
                # Check if we have figures to display
                current_figs = plt.get_fignums()
                print(f"   🔍 Figuras creadas: {len(current_figs)}")
                
                if current_figs:
                    # Try to bring window to front on Windows
                    try:
                        fig = plt.figure(current_figs[0])
                        # Windows-specific: bring to front
                        if hasattr(fig.canvas.manager, 'window'):
                            fig.canvas.manager.window.wm_attributes('-topmost', 1)
                            fig.canvas.manager.window.wm_attributes('-topmost', 0)
                    except Exception as e:
                        print(f"   ⚠️  No se pudo traer la ventana al frente: {e}")
                    
                    # Show plot without blocking - allows annotation while viewing
                    plt.show(block=False)
                    plt.pause(0.5)  # Give time for window to appear
                    
                    print(f"   📊 NeuroKit2 plot mostrado - puedes revisar el plot mientras anotas")
                else:
                    print(f"   ❌ No se crearon figuras - error en NeuroKit2 plotting")
                    return False
                
                # Don't close plots here - let them stay open during annotation
                return True  # Success with NeuroKit2
                
            except Exception as e:
                print(f"   ⚠️  NeuroKit2 plotting failed: {e}")
                plt.close('all')  # Clean up any partial plots
                return False
        else:
            print(f"   ⚠️  No info dict available - cannot create NeuroKit2 plots")
            return False
        
    except Exception as e:
        print(f"❌ Error plotting {signal_type.upper()} {plot_title}: {str(e)}")
        plt.close('all')  # Clean up any partial plots
        return False


# Emotiphai per-event plot removed by request; events are shown over CVX EDR only


def plot_cvx_exploratory(eda_csv_path, subject):
    """Create exploratory plot for CVX decomposition (EDR/SMNA/EDL) and overlay Emotiphai SCR events."""
    try:
        cvx_path = eda_csv_path.replace('.csv', '_cvx_decomposition.csv')
        if not os.path.exists(cvx_path):
            print(f"   ⚠️  CVX file not found: {os.path.basename(cvx_path)}")
            return False

        df_cvx = pd.read_csv(cvx_path)
        if 'time' in df_cvx.columns:
            t = df_cvx['time'].values
        else:
            t = np.arange(len(df_cvx)) / NEUROKIT_PARAMS['sampling_rate_default']
        sr = NEUROKIT_PARAMS['sampling_rate_default']

        # Try load Emotiphai events
        emotiphai_path = eda_csv_path.replace('.csv', '_emotiphai_scr.csv')
        df_emo = None
        if os.path.exists(emotiphai_path):
            df_emo = pd.read_csv(emotiphai_path)
        else:
            print(f"   ⚠️  Emotiphai file not found for overlay: {os.path.basename(emotiphai_path)}")

        plt.figure(figsize=(10, 6))
        plt.suptitle(create_plot_title_from_filename(cvx_path, 'eda', subject) + " - CVX (EDR/SMNA/EDL) + Emotiphai SCR", fontsize=12)

        # EDR
        ax1 = plt.subplot(3, 1, 1)
        edr = df_cvx.get('EDR', np.zeros(len(t))).values
        ax1.plot(t, edr, lw=0.8)
        ax1.set_ylabel('EDR')

        # Overlay Emotiphai events on EDR
        if df_emo is not None and len(df_emo) > 0:
            onset_idx = df_emo.get('SCR_Onsets_Emotiphai', pd.Series([], dtype=int)).astype(int).to_numpy()
            peak_idx = df_emo.get('SCR_Peaks_Emotiphai', pd.Series([], dtype=int)).astype(int).to_numpy()
            amplitudes = df_emo.get('SCR_Amplitudes_Emotiphai', pd.Series([], dtype=float)).astype(float).to_numpy()

            # Convert to time (seconds)
            onset_t = onset_idx / sr
            peak_t = peak_idx / sr

            # Safe clip indices to EDR length
            onset_idx_clip = np.clip(onset_idx, 0, len(edr) - 1)
            peak_idx_clip = np.clip(peak_idx, 0, len(edr) - 1)
            edr_on = edr[onset_idx_clip]
            edr_pk = edr[peak_idx_clip]

            ax1.scatter(onset_t, edr_on, s=12, c='tab:blue', label='Onset (Emotiphai)', zorder=3)
            ax1.scatter(peak_t, edr_pk, s=12, c='tab:red', label='Peak (Emotiphai)', zorder=3)

            # Amplitude as vertical bar descending from the peak
            for x_pt, y_pk, amp in zip(peak_t, edr_pk, amplitudes):
                ax1.vlines(x_pt, y_pk - amp, y_pk, colors='tab:purple', alpha=0.7, linewidth=1)

            ax1.legend(loc='upper right', fontsize=8)

        # SMNA
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(t, df_cvx.get('SMNA', np.zeros(len(t))), color='orange', lw=0.8)
        ax2.set_ylabel('SMNA')

        # EDL
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(t, df_cvx.get('EDL', np.zeros(len(t))), color='green', lw=0.8)
        ax3.set_ylabel('EDL')
        ax3.set_xlabel('Time (s)')

        plt.tight_layout()

        # Save exploratory plot
        plot_dir = os.path.join('test', 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = os.path.basename(cvx_path).replace('.csv', '.png')
        plt.savefig(os.path.join(plot_dir, plot_filename), dpi=200, bbox_inches='tight')

        plt.show(block=False)
        plt.pause(0.3)
        print(f"   📊 CVX exploratory plot shown and saved: {plot_filename}")
        return True
    except Exception as e:
        print(f"   ⚠️  CVX exploratory plotting failed: {e}")
        return False

def test_physiological_preprocessing():
    """
    Main testing function to validate physiological preprocessing results (EDA, ECG, RESP)
    """
    print("🧪 Starting physiological preprocessing validation test")
    print("=" * 60)
    
    # Load validation log for updating
    print("📋 Loading validation log for updating...")
    validation_data = load_validation_log()
    
    # Determine which subjects to test (per requested behavior)
    if TEST_MODE:
        subjects_to_test = SUJETOS_TEST
        print(f"🔬 TEST_MODE=True: validating TEST subjects: {subjects_to_test}")
    else:
        subjects_to_test = TODOS_LOS_SUJETOS
        print(f"🔬 TEST_MODE=False: validating ALL subjects in order ({len(subjects_to_test)})")

    # Signal types and aggregator for results
    signal_types = ['eda', 'ecg', 'resp']
    all_results = {}

    print(f"\n📄 Testing physiological files:")

    # Iterate over selected subjects
    for test_subject in subjects_to_test:
        print(f"\n🎯 Testing physiological data for subject: {test_subject}")

        # Define expected output directories per signal
        signal_dirs = {}
        for signal in signal_types:
            signal_dirs[signal] = {
                'dmt_high': os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_high'),
                'dmt_low': os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_low')
            }

        print(f"\n📁 Expected output directories for {test_subject}:")
        for signal in signal_types:
            print(f"   📂 {signal.upper()}:")
            for condition, path in signal_dirs[signal].items():
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"      {exists} {condition}: {path}")

        # Generate expected files for this subject (force clean subfolder names)
        expected_files = {}
        for signal in signal_types:
            dmt_high_dir = os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_high')
            dmt_low_dir = os.path.join(DERIVATIVES_DATA, 'phys', signal, 'dmt_low')
            expected_files[signal] = {
                f'{signal.upper()} DMT High Dose': os.path.join(dmt_high_dir, f'{test_subject}_dmt_session1_high.csv'),
                f'{signal.upper()} Resting High Dose': os.path.join(dmt_high_dir, f'{test_subject}_rs_session1_high.csv'),
                f'{signal.upper()} DMT Low Dose': os.path.join(dmt_low_dir, f'{test_subject}_dmt_session2_low.csv'),
                f'{signal.upper()} Resting Low Dose': os.path.join(dmt_low_dir, f'{test_subject}_rs_session2_low.csv')
            }

        # Load and test each signal type for this subject
        for signal_type in signal_types:
            if signal_type not in all_results:
                all_results[signal_type] = {}
            print(f"\n🧬 Testing {signal_type.upper()} signals:")

            for description, file_path in expected_files[signal_type].items():
                print(f"\n🔍 Testing: {description}")
                print(f"   Path: {file_path}")
                
                # Load data
                signal_df, time_series, info_dict = load_physiological_data(file_path, signal_type)
                
                # Update validation log with file existence info
                if validation_data is not None:
                    # Parse description to get validation key
                    file_key = parse_file_description_to_validation_key(description, test_subject, file_path)
                    print(f"   🔍 Parsed file_key: '{file_key}' from description: '{description}'")
                    if file_key:
                        # Compose derived paths for auxiliary files (EDA only)
                        extra = None
                        if signal_type == 'eda':
                            emotiphai_path = file_path.replace('.csv', '_emotiphai_scr.csv')
                            cvx_path = file_path.replace('.csv', '_cvx_decomposition.csv')
                            extra = {
                                'file_path_emotiphai_scr': emotiphai_path,
                                'file_path_cvx_decomposition': cvx_path,
                            }
                        update_file_existence_info(
                            validation_data, test_subject, signal_type, file_key,
                            file_path, signal_df is not None, extra_fields=extra
                        )
                        print(f"   ✅ Updated validation log for {test_subject}/{file_key}/{signal_type}")
                    else:
                        print(f"   ❌ Could not parse file_key from description: '{description}'")
                else:
                    print(f"   ⚠️  Validation data is None - cannot update validation log")
                
                if signal_df is not None:
                    # Determine expected duration based on file type
                    if 'DMT' in description:
                        expected_duration_sec = DURACIONES_ESPERADAS['DMT']  # Both DMT sessions have same duration
                    else:  # Resting
                        expected_duration_sec = DURACIONES_ESPERADAS['Reposo']  # Both Resting sessions have same duration
                    
                    # Test duration and sampling rate
                    duration_test_results = test_duration_and_sampling_rate(
                        signal_df, time_series, expected_duration_sec, f"{test_subject} - {description}"
                    )
                    
                    # Store results for summary
                    result_key = f"{test_subject} - {description}"
                    all_results[signal_type][result_key] = {
                        'samples': len(signal_df),
                        'duration_min': len(signal_df) / NEUROKIT_PARAMS['sampling_rate_default'] / 60,
                        'variables': len(signal_df.columns),
                        'success': True,
                        'duration_test': duration_test_results
                    }
                    
                    # Generate plot
                    print(f"   📊 Generating {signal_type.upper()} plot for {description}...")
                    plot_success = plot_physiological_signals(signal_df, time_series, file_path, signal_type, test_subject, None, info_dict)

                    # For EDA, also show CVX exploratory plot with Emotiphai overlays (after NeuroKit)
                    if signal_type == 'eda':
                        try:
                            cvx_ok = plot_cvx_exploratory(file_path, test_subject)
                            if cvx_ok:
                                print("   📊 CVX + Emotiphai overlay displayed")
                        except Exception as e:
                            print(f"   ⚠️  Failed to display CVX/Emotiphai overlay plot: {e}")
                    
                    # Collect manual validation input ONLY after plot was displayed and closed
                    if validation_data is not None and file_key:
                        plot_title = create_plot_title_from_filename(file_path, signal_type, test_subject)
                        if plot_success:
                            print(f"📈 Se generó y mostró el plot para revisar la calidad de la señal")
                        else:
                            print(f"⚠️  No se pudo generar el plot - evalúa basándote en la información disponible")
                        
                        category, notes, changed = collect_manual_validation_input(
                            validation_data, test_subject, signal_type, file_key, True, plot_title
                        )
                        if changed:
                            update_manual_validation_info(
                                validation_data, test_subject, signal_type, file_key, category, notes
                            )
                    
                else:
                    result_key = f"{test_subject} - {description}"
                    all_results[signal_type][result_key] = {'success': False}
                    # Skip manual validation for missing files; existence already logged
    
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
    
    # Save updated validation log
    if validation_data is not None:
        save_validation_log(validation_data)
    
    print(f"\n🎯 Test completed!")
    if total_successful > 0:
        plot_dir = os.path.join('test', 'plots')
        print(f"📊 Plots saved in: {plot_dir}")
        print(f"📊 Generated plots for {len(signal_types)} signal types: {', '.join([s.upper() for s in signal_types])}")
    
    if validation_data is not None:
        print(f"📝 Validation log updated with file existence information")
    
    # Final cleanup of any remaining plots
    remaining_figs = plt.get_fignums()
    if remaining_figs:
        print(f"\n🧹 Cerrando {len(remaining_figs)} ventanas de plots restantes...")
        plt.close('all')
        print(f"   ✅ Todas las ventanas cerradas")

if __name__ == "__main__":
    test_physiological_preprocessing()