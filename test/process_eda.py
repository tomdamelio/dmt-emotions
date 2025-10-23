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

# New data paths following preprocess_phys.py output structure
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
    # Get session info from DOSIS DataFrame (indexed by subject)
    try:
        dose_s1 = DOSIS.loc[subject, 'Dosis_Sesion_1']  # 'Alta' or 'Baja'
        dose_s2 = DOSIS.loc[subject, 'Dosis_Sesion_2']
    except Exception:
        print(f"⚠️  Subject {subject} not found in DOSIS config index")
        return None, None

    # Determine session based on condition name (dmt_high / dmt_low)
    if 'high' in condition_name:
        session = 'session1' if dose_s1 == 'Alta' else 'session2'
    else:  # 'low' condition
        session = 'session1' if dose_s1 == 'Baja' else 'session2'
    
    # File paths
    dmt_file = os.path.join(condition_dir, f"{subject}_dmt_{session}_{condition_name.split('_')[1]}.csv")
    rs_file = os.path.join(condition_dir, f"{subject}_rs_{session}_{condition_name.split('_')[1]}.csv")
    
    dmt_data = rs_data = None
    cvx_dmt = None
    scr_count_dmt = None
    cvx_rs = None
    scr_count_rs = None
    
    # Load DMT file
    if os.path.exists(dmt_file):
        dmt_data = pd.read_csv(dmt_file)
        print(f"   ✅ Loaded DMT {condition_name}: {subject} ({len(dmt_data)} samples)")
        # Try matching CVX and Emotiphai files from DMT path
        cvx_path = dmt_file.replace('.csv', '_cvx_decomposition.csv')
        if os.path.exists(cvx_path):
            try:
                cvx_dmt = pd.read_csv(cvx_path)
                print(f"      ➕ Loaded CVX: {os.path.basename(cvx_path)}")
            except Exception as e:
                print(f"      ⚠️  Could not load CVX for {subject}: {e}")
        emo_path = dmt_file.replace('.csv', '_emotiphai_scr.csv')
        if os.path.exists(emo_path):
            try:
                scr_count_dmt = len(pd.read_csv(emo_path))
                print(f"      ➕ Emotiphai SCR count: {scr_count_dmt}")
            except Exception as e:
                print(f"      ⚠️  Could not count Emotiphai SCR for {subject}: {e}")
    else:
        print(f"   ❌ Missing DMT file: {dmt_file}")
    
    # Load RS file  
    if os.path.exists(rs_file):
        rs_data = pd.read_csv(rs_file)
        print(f"   ✅ Loaded RS {condition_name}: {subject} ({len(rs_data)} samples)")
        rs_cvx_path = rs_file.replace('.csv', '_cvx_decomposition.csv')
        if os.path.exists(rs_cvx_path):
            try:
                cvx_rs = pd.read_csv(rs_cvx_path)
                print(f"      ➕ Loaded RS CVX: {os.path.basename(rs_cvx_path)}")
            except Exception as e:
                print(f"      ⚠️  Could not load RS CVX for {subject}: {e}")
        rs_emo_path = rs_file.replace('.csv', '_emotiphai_scr.csv')
        if os.path.exists(rs_emo_path):
            try:
                scr_count_rs = len(pd.read_csv(rs_emo_path))
                print(f"      ➕ RS Emotiphai SCR count: {scr_count_rs}")
            except Exception as e:
                print(f"      ⚠️  Could not count RS Emotiphai SCR for {subject}: {e}")
    else:
        print(f"   ❌ Missing RS file: {rs_file}")
    
    return dmt_data, rs_data, cvx_dmt, scr_count_dmt, cvx_rs, scr_count_rs

# Load data for all subjects
print(f"\n📁 Loading data from: {eda_dirs}")

all_data = {
    'dmt_high': {
        'dmt': [], 'rs': [], 'subjects': [],
        'cvx_dmt': [], 'cvx_rs': [],
        'scr_counts_dmt': [], 'scr_counts_rs': []
    },
    'dmt_low': {
        'dmt': [], 'rs': [], 'subjects': [],
        'cvx_dmt': [], 'cvx_rs': [],
        'scr_counts_dmt': [], 'scr_counts_rs': []
    }
}

files_loaded = 0
total_expected = len(subjects_to_process) * 4  # 2 conditions × 2 session types

for subject in subjects_to_process:
    print(f"\n🔍 Loading subject: {subject}")
    
    for condition_name, condition_dir in eda_dirs.items():
        dmt_data, rs_data, cvx_dmt, scr_count_dmt, cvx_rs, scr_count_rs = load_subject_data(subject, condition_dir, condition_name)
        
        if dmt_data is not None and rs_data is not None:
            # Extract EDA_Tonic for analysis (main signal of interest)
            if 'EDA_Tonic' in dmt_data.columns and 'EDA_Tonic' in rs_data.columns:
                all_data[condition_name]['dmt'].append(dmt_data['EDA_Tonic'])
                all_data[condition_name]['rs'].append(rs_data['EDA_Tonic'])
                all_data[condition_name]['subjects'].append(subject)
                files_loaded += 2
                # Save CVX if present
                if cvx_dmt is not None and {'EDR','SMNA','EDL'}.issubset(set(cvx_dmt.columns)):
                    all_data[condition_name]['cvx_dmt'].append(cvx_dmt[['EDR','SMNA','EDL']])
                if cvx_rs is not None and {'EDR','SMNA','EDL'}.issubset(set(cvx_rs.columns)):
                    all_data[condition_name]['cvx_rs'].append(cvx_rs[['EDR','SMNA','EDL']])
                # SCR counts
                if scr_count_dmt is not None:
                    all_data[condition_name]['scr_counts_dmt'].append(scr_count_dmt)
                if scr_count_rs is not None:
                    all_data[condition_name]['scr_counts_rs'].append(scr_count_rs)
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
    # Usar archivos CSV preprocesados (EDA_Tonic)
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


#%%
# ==========================
# New analyses: CVX EDR/SMNA
# ==========================

def align_by_subject(subjects_a, dataframes_a, subjects_b, dataframes_b):
    """Return two lists of DataFrames aligned by intersecting subjects order.
    Each dataframe is expected to have identical row count.
    """
    map_a = {s: df for s, df in zip(subjects_a, dataframes_a)}
    map_b = {s: df for s, df in zip(subjects_b, dataframes_b)}
    common = [s for s in subjects_a if s in map_b]
    a_aligned = [map_a[s] for s in common]
    b_aligned = [map_b[s] for s in common]
    return common, a_aligned, b_aligned

def build_matrix_from_component(dfs, component):
    """Concat a single component (e.g., 'EDR' or 'SMNA') across subjects."""
    cols = []
    for df in dfs:
        if component in df.columns:
            cols.append(df[component].reset_index(drop=True))
    if len(cols) == 0:
        return None
    return pd.concat(cols, axis=1, ignore_index=True)

def plot_cvx_component_comparison(component_name, label_y, color_high="#9C27B0", color_low="#FFA726"):
    # Fetch lists
    subj_high = all_data['dmt_high']['subjects']
    subj_low = all_data['dmt_low']['subjects']
    cvx_high = all_data['dmt_high']['cvx_dmt']
    cvx_low = all_data['dmt_low']['cvx_dmt']

    if len(cvx_high) == 0 or len(cvx_low) == 0:
        print(f"⚠️  No CVX data available for {component_name} comparison.")
        return None, None, None, None

    commons, high_aligned, low_aligned = align_by_subject(subj_high, cvx_high, subj_low, cvx_low)
    if len(commons) == 0:
        print(f"⚠️  No common subjects for {component_name} comparison.")
        return None, None, None, None

    mat_high = build_matrix_from_component(high_aligned, component_name)
    mat_low = build_matrix_from_component(low_aligned, component_name)
    if mat_high is None or mat_low is None:
        print(f"⚠️  Missing component {component_name} in CVX data.")
        return None, None, None, None

    # Match column counts if needed
    ncols = min(mat_high.shape[1], mat_low.shape[1])
    mat_high = mat_high.iloc[:, :ncols]
    mat_low = mat_low.iloc[:, :ncols]

    # Time vector
    sr = NEUROKIT_PARAMS['sampling_rate_default']
    t_vec = pd.Series(np.arange(len(mat_high)) / sr)

    # Means/errors
    mean_high, sem_high = calculate_means_and_stdevs(mat_high)
    mean_low, sem_low = calculate_means_and_stdevs(mat_low)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_vec, mean_high.values, label='High Dose', color=color_high)
    plt.fill_between(t_vec, (mean_high - sem_high).values, (mean_high + sem_high).values, alpha=0.3, color=color_high)
    plt.plot(t_vec, mean_low.values, label='Low Dose', color=color_low)
    plt.fill_between(t_vec, (mean_low - sem_low).values, (mean_low + sem_low).values, alpha=0.3, color=color_low)
    plt.xlabel('Time (s)')
    plt.ylabel(label_y)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Point-by-point Wilcoxon
    pvals = test(mat_low, mat_high)
    return mat_low, mat_high, pvals, t_vec

# EDR comparison
mat_low_edr, mat_high_edr, pvals_edr, t_edr = plot_cvx_component_comparison('EDR', 'EDR (a.u.)')

#%%
# SMNA comparison
mat_low_smna, mat_high_smna, pvals_smna, t_smna = plot_cvx_component_comparison('SMNA', 'SMNA (a.u.)')

#%%
# Boxplot intra-subject: SCR counts (Emotiphai)
def boxplot_scr_counts_intrasubject():
    subj_high = all_data['dmt_high']['subjects']
    subj_low = all_data['dmt_low']['subjects']
    cnt_high = all_data['dmt_high']['scr_counts_dmt']
    cnt_low = all_data['dmt_low']['scr_counts_dmt']
    # Align by subject
    map_h = {s: c for s, c in zip(subj_high, cnt_high)}
    map_l = {s: c for s, c in zip(subj_low, cnt_low)}
    commons = [s for s in subj_high if s in map_l and s in map_h]
    if len(commons) == 0:
        print("⚠️  No common subjects for SCR count boxplot.")
        return
    arr_h = np.array([map_h[s] for s in commons])
    arr_l = np.array([map_l[s] for s in commons])

    # Paired Wilcoxon
    try:
        stat_paired = stats.wilcoxon(arr_h, arr_l)
        pval = stat_paired.pvalue
    except Exception:
        pval = None

    plt.figure(figsize=(6, 5))
    data = [arr_h, arr_l]
    plt.boxplot(data, labels=['High Dose', 'Low Dose'])
    # plot paired points
    for i in range(len(commons)):
        plt.plot([1, 2], [arr_h[i], arr_l[i]], color='gray', alpha=0.6)
        plt.scatter([1, 2], [arr_h[i], arr_l[i]], color=['#9C27B0', '#FFA726'])
    plt.ylabel('SCR count (Emotiphai)')
    if pval is not None:
        plt.title(f'Paired Wilcoxon p={pval:.4f} (n={len(commons)})')
    plt.tight_layout()
    plt.show()

boxplot_scr_counts_intrasubject()

#%%







