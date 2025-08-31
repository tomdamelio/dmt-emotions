#%%
# -*- coding: utf-8 -*-
"""
EDA preprocessed data analysis and visualization

This script performs statistical analysis and generates visualizations of 
previously processed EDA (SCL - Skin Conductance Level) data.

UPDATED FEATURES:
- Loads individual subject files from new preprocess_eda.py output structure
- Respects TEST_MODE from config.py (processes only S04 in test mode)
- Uses config-based sampling rate and subject lists
- Combines individual EDA_Tonic signals into high/low dose groups for analysis

Input: Individual CSV files in derivatives/phys/eda/dmt_high/ and dmt_low/
       - Format: {subject}_{dmt/rs}_{session}_{high/low}.csv
       - Uses EDA_Tonic column for statistical analysis
Output: Time series plots, statistical tests and FDR analysis

Prerequisite: Run preprocess_eda.py first to generate the CSV files
Test mode: If TEST_MODE=True in config.py, only processes subject S04
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as st
from joblib import Parallel, delayed

# Add parent directory to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import (DERIVATIVES_DATA, SUJETOS_VALIDOS, SUJETOS_TEST, 
                       TEST_MODE, NEUROKIT_PARAMS, DOSIS)
except ImportError:
    print("❌ Could not import config. Make sure config.py exists in parent directory.")
    sys.exit(1)

#%% 
print("🔄 Loading EDA preprocessed data...")

# Determine which subjects to process
subjects_to_process = SUJETOS_TEST if TEST_MODE else SUJETOS_VALIDOS
print(f"📊 Processing subjects: {subjects_to_process} (TEST_MODE: {TEST_MODE})")

# New data paths following preprocess_eda.py structure
eda_dirs = {
    'dmt_high': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high'),
    'dmt_low': os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
}

def load_subject_data(subject, condition_dir, condition_name):
    """
    Load EDA data for a specific subject and condition
    
    Args:
        subject: Subject ID (e.g., 'S04')
        condition_dir: Directory path for the condition
        condition_name: Name for logging ('dmt_high' or 'dmt_low')
    
    Returns:
        tuple: (dmt_data, rs_data) or (None, None) if files not found
    """
    # Get session info from DOSIS DataFrame
    subject_dosis = DOSIS[DOSIS['Subject'] == subject]
    
    if subject_dosis.empty:
        print(f"⚠️  Subject {subject} not found in DOSIS config")
        return None, None
    
    # Determine session based on condition
    if 'high' in condition_name:
        # Find which session has 'Alta' dose
        if subject_dosis['Dosis_Sesion_1'].iloc[0] == 'Alta':
            session = 'session1'
        else:
            session = 'session2'
    else:  # 'low' condition
        # Find which session has 'Baja' dose  
        if subject_dosis['Dosis_Sesion_1'].iloc[0] == 'Baja':
            session = 'session1'
        else:
            session = 'session2'
    
    # File paths
    dmt_file = os.path.join(condition_dir, f"{subject}_dmt_{session}_{condition_name.split('_')[1]}.csv")
    rs_file = os.path.join(condition_dir, f"{subject}_rs_{session}_{condition_name.split('_')[1]}.csv")
    
    dmt_data = rs_data = None
    
    # Load DMT file
    if os.path.exists(dmt_file):
        dmt_data = pd.read_csv(dmt_file)
        print(f"   ✅ Loaded DMT {condition_name}: {subject} ({len(dmt_data)} samples)")
    else:
        print(f"   ❌ Missing DMT file: {dmt_file}")
    
    # Load RS file  
    if os.path.exists(rs_file):
        rs_data = pd.read_csv(rs_file)
        print(f"   ✅ Loaded RS {condition_name}: {subject} ({len(rs_data)} samples)")
    else:
        print(f"   ❌ Missing RS file: {rs_file}")
    
    return dmt_data, rs_data

# Load data for all subjects
print(f"\n📁 Loading data from: {eda_dirs}")

all_data = {
    'dmt_high': {'dmt': [], 'rs': [], 'subjects': []},
    'dmt_low': {'dmt': [], 'rs': [], 'subjects': []}
}

files_loaded = 0
total_expected = len(subjects_to_process) * 4  # 2 conditions × 2 session types

for subject in subjects_to_process:
    print(f"\n🔍 Loading subject: {subject}")
    
    for condition_name, condition_dir in eda_dirs.items():
        dmt_data, rs_data = load_subject_data(subject, condition_dir, condition_name)
        
        if dmt_data is not None and rs_data is not None:
            # Extract EDA_Tonic for analysis (main signal of interest)
            if 'EDA_Tonic' in dmt_data.columns and 'EDA_Tonic' in rs_data.columns:
                all_data[condition_name]['dmt'].append(dmt_data['EDA_Tonic'])
                all_data[condition_name]['rs'].append(rs_data['EDA_Tonic'])
                all_data[condition_name]['subjects'].append(subject)
                files_loaded += 2
            else:
                print(f"   ⚠️  EDA_Tonic column not found in {subject} {condition_name}")

print(f"\n📊 Data loading summary:")
print(f"   Files loaded: {files_loaded}/{total_expected}")
print(f"   High dose subjects: {len(all_data['dmt_high']['subjects'])}")
print(f"   Low dose subjects: {len(all_data['dmt_low']['subjects'])}")

# Check if we have enough data to proceed
files_exist = files_loaded > 0

if not files_exist:
    print("\n❌ No preprocessed files found. Run preprocess_eda.py first.")
    print("Expected structure:")
    for condition_name, condition_dir in eda_dirs.items():
        print(f"- {condition_dir}")
    scl_alta_combined = scl_baja_combined = None
    tiempo_alta = tiempo_baja = None
else:
    print("\n✅ Converting to analysis format...")
    
    # Convert to DataFrame format expected by rest of script
    # Combine DMT data for each condition
    if len(all_data['dmt_high']['dmt']) > 0:
        scl_alta_combined = pd.concat(all_data['dmt_high']['dmt'], axis=1, ignore_index=True)
        scl_alta_combined.columns = [f"Subject_{subj}" for subj in all_data['dmt_high']['subjects']]
        # Add time column (assuming consistent sampling rate)
        sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
        tiempo_alta = pd.Series(np.arange(len(scl_alta_combined)) / sampling_rate)
        scl_alta_combined.insert(0, 'time', tiempo_alta)
        
        # Separate time and signal for compatibility
        scl_alta = scl_alta_combined.drop(columns=['time'])
        print(f"   High dose combined: {scl_alta.shape[1]} subjects, {scl_alta.shape[0]} time points")
    else:
        scl_alta_combined = scl_alta = tiempo_alta = None
        
    if len(all_data['dmt_low']['dmt']) > 0:
        scl_baja_combined = pd.concat(all_data['dmt_low']['dmt'], axis=1, ignore_index=True)
        scl_baja_combined.columns = [f"Subject_{subj}" for subj in all_data['dmt_low']['subjects']]
        # Add time column (assuming consistent sampling rate)
        sampling_rate = NEUROKIT_PARAMS['sampling_rate_default']
        tiempo_baja = pd.Series(np.arange(len(scl_baja_combined)) / sampling_rate)
        scl_baja_combined.insert(0, 'time', tiempo_baja)
        
        # Separate time and signal for compatibility
        scl_baja = scl_baja_combined.drop(columns=['time'])
        print(f"   Low dose combined: {scl_baja.shape[1]} subjects, {scl_baja.shape[0]} time points")
    else:
        scl_baja_combined = scl_baja = tiempo_baja = None

#%% Ploteo de promedio con sombreado de error

def calculate_means_and_stdevs(data):
    # Concatenate the Series into a DataFrame
    # df = pd.concat(data, axis=1)
    df = data
    # Calculate mean across columns (axis=1)
    means = df.mean(axis=1)

    # Calculate standard deviation across columns (axis=1)
    errors = df.std(axis=1)/np.sqrt(df.shape[1]) #chequear si ahi me divide por la raiz de la cantidad de columnas
    # errors.fillna(0)
    
    return means, errors

# Calcular promedios y errores solo si los archivos existen
if files_exist:
    # Usar archivos CSV preprocesados
    alta_promedio_listas, alta_errores = calculate_means_and_stdevs(scl_alta)
    baja_promedio_listas, baja_errores = calculate_means_and_stdevs(scl_baja)
    
    # Los tiempos ahora vienen directamente de los CSVs
    tiempo_alto_listas = tiempo_alta
    tiempo_bajo_listas = tiempo_baja
else:
    print("No data available for analysis. Run preprocess_eda.py first.")
    alta_promedio_listas = baja_promedio_listas = None
    tiempo_alto_listas = tiempo_bajo_listas = None



#%% Ploteo con fill between y grafico el promedio

if alta_promedio_listas is not None:
    # plt.close('all')

    relleno_pos_alta = alta_promedio_listas + alta_errores
    relleno_neg_alta = alta_promedio_listas - alta_errores

    plt.plot(tiempo_alto_listas, alta_promedio_listas.values, label = 'Dosis Alta', color = "#9C27B0")
    plt.fill_between(tiempo_alto_listas, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")

    relleno_pos_baja = baja_promedio_listas + baja_errores
    relleno_neg_baja = baja_promedio_listas - baja_errores

    plt.plot(tiempo_bajo_listas, baja_promedio_listas.values, label = 'Dosis Baja', color = "#FFA726")
    plt.fill_between(tiempo_bajo_listas, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")

    plt.ylabel('EDA Tonica restando baseline ($\mu$S)')
    plt.xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.legend()
    plt.show()
else:
    print("No hay datos para plotear. Ejecute preprocess_eda.py primero.")

#%% Test de Wilcoxon entre los dosis alta y dosis baja

def test(baja, alta):
    
    pvalor = []
    
    df_alta = alta
    df_baja = baja

    # Paso a arrays para poder hacer las stats
    arr_baja = df_baja.values
    arr_alta = df_alta.values
    
    # Preallocate the result array
    pvalor = np.empty(arr_baja.shape[0])
    
    def compute_pvalue(j):
       return stats.wilcoxon(arr_baja[j], arr_alta[j]).pvalue
   
    pvalor = Parallel(n_jobs=-1)(delayed(compute_pvalue)(j) for j in range(arr_baja.shape[0]))
    
    return pvalor
    
# Ejecutar test estadístico solo si hay datos
if files_exist:
    pvalores = test(scl_baja, scl_alta)
else:
    print("No hay datos para test estadístico.")
    pvalores = None


#%% Chequeo de pvalores mayores y menores

if pvalores is not None:
    neg = 0
    pos = 0

    for j in range(len(pvalores)):
        if pvalores[j] > 0.05:
            # print('Acepta Hipotesis Nula')
            neg += 1
        else:
            # print('Rechaza hipotesis nula')
            pos += 1
           
    print()
    print("Cantidad de rechazos de Ho:", pos)
    print("Cantidad de aceptadas de Ho:", neg)
else:
    print("No hay p-valores para analizar.")
    
#%% Ploteo de pvalores con respecto al umbral

if pvalores is not None and alta_promedio_listas is not None:
    plt.figure(2)
    plt.plot(tiempo_bajo_listas,pvalores,'o', label = 'P-valor', color = '#A366A6')
    plt.plot(tiempo_bajo_listas, 0.05*np.ones(len(tiempo_bajo_listas)),'--', label = 'Umbral', color = '#CCA60D')
    plt.xlabel('Tiempo(s)')
    plt.legend()
    plt.tight_layout()
else:
    print("No hay datos para plotear p-valores.")

#%% Calculo de pvalores que pasan el umbral para después plotear, ya con fdr

if pvalores is not None:
    #####################################################################################################################################
    # Esta celda funciona asi, para calcular pvalores > 0.05, hay que comentar lo de fdr y los indices modificados por fdr
    # Dsps descomentar indices menores al 0.05 y graficar en la siguiente celda. Con fdr es al reves y graficar con la siguiente celda.
    ####################################################################################################################################

    arr_pvalores = np.array(pvalores)
    # Calculo con fdr los pvalores nuevos
    rtas, pval_fdr = st.fdrcorrection(arr_pvalores)

    # Indices de los pvalores que son menores que el 0.05
    indices_p_pos = (np.where(arr_pvalores < 0.05))[0]

    # Indices de los pvalores ya modificados por fdr que dan menores a 0.05
    # indices_p_pos = (np.where(rtas == True))[0]

    # Divido en regiones de los valores que cumplen para poder dibujar eso en el grafico
    regiones = np.split(indices_p_pos, np.where(np.diff(indices_p_pos) != 1)[0] + 1)

    # Use sampling rate from config
    frec_sampleo = NEUROKIT_PARAMS['sampling_rate_default']
else:
    print("No hay p-valores para análisis FDR.")
    regiones = []
    pval_fdr = None

#%% AGREGADO: Ploteo con fill between Y Pvalores

if alta_promedio_listas is not None:
    size = 14

    plt.close('all')

    relleno_pos_alta = alta_promedio_listas + alta_errores
    relleno_neg_alta = alta_promedio_listas - alta_errores

    plt.plot(tiempo_alto_listas, alta_promedio_listas.values, label = 'High Dose', color = "#9C27B0")
    plt.fill_between(tiempo_alto_listas, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.4, color = "#9C27B0")

    relleno_pos_baja = baja_promedio_listas + baja_errores
    relleno_neg_baja = baja_promedio_listas - baja_errores

    plt.plot(tiempo_bajo_listas, baja_promedio_listas.values, label = 'Low Dose', color = "#FFA726")
    plt.fill_between(tiempo_bajo_listas, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.4, color = "#FDD835")

    # Acá agrego el background de color gris
    if 'regiones' in locals() and len(regiones) > 0 and len(regiones[0]) != 0: # en este caso el fdr me saca todos los pvalores
        for i, region in enumerate(regiones):
            start = region[0] + 1
            end = region[-1] + 1
            plt.axvspan(start/frec_sampleo, end/frec_sampleo, color='#C6C6C6', alpha=0.3)
            # mid_point = (start + end) / 2
            # plt.text(mid_point, plt.ylim()[1], f'Region {i+1}', horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

    plt.ylabel('SCL substracting baseline (uS)', fontsize = size)
    plt.xlabel('Time (s)', fontsize = size)
    plt.xticks(fontsize = size)
    plt.yticks(fontsize = size)
    plt.tight_layout()
    plt.legend(fontsize = size)
    plt.show()
else:
    print("No hay datos para plotear.")

#%% Ploteo de pvalores con fdr para mostrar qué valores dan

if pval_fdr is not None and alta_promedio_listas is not None:
    plt.figure(2)
    plt.plot(tiempo_bajo_listas,pval_fdr,'o', label = 'P-valor', color = '#A366A6')
    plt.plot(tiempo_bajo_listas, 0.05*np.ones(len(tiempo_bajo_listas)),'--', label = 'Umbral', color = '#CCA60D')
    plt.xlabel('Tiempo(s)')
    plt.legend()
    plt.tight_layout()
else:
    print("No hay datos FDR para plotear.")









