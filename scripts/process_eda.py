#%%
# -*- coding: utf-8 -*-
"""
EDA preprocessed data analysis and visualization

This script performs statistical analysis and generates visualizations of 
previously processed EDA (SCL - Skin Conductance Level) data.

Input: CSV files in ../data/derivatives/preprocessing/SCL/
Output: Time series plots, statistical tests and FDR analysis

Prerequisite: Run preprocess_eda.py first to generate the CSV files
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as st
from joblib import Parallel, delayed

# Configuración de rutas siguiendo estándares BIDS
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
DERIVATIVES_DATA = os.path.join(ROOT_DATA, 'derivatives', 'preprocessing')

#%% 
medida = 'SCL'
preproc_dir = os.path.join(DERIVATIVES_DATA, medida)
os.makedirs(preproc_dir, exist_ok=True)

#%% Leer los archivos preprocesados (o crearlos si no existen)

# Verificar si los archivos preprocesados existen
csv_files = [
    f'{medida}_dmt_alta.csv',
    f'{medida}_dmt_baja.csv'
]

files_exist = all(os.path.exists(os.path.join(preproc_dir, csv_file)) for csv_file in csv_files)

if not files_exist:
    print("Preprocessed files do not exist. Run preprocess_eda.py first.")
    print("Or verify that the data structure is correct:")
    print(f"- Processed files will be saved in: {preproc_dir}")
    scl_alta_combined = scl_baja_combined = None
else:
    # Leer archivos CSV combinados (tiempo + señal tónica)
    scl_alta_combined = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
    scl_baja_combined = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
    
    # Separar columnas de tiempo y señal
    tiempo_alta = scl_alta_combined['time']
    tiempo_baja = scl_baja_combined['time']
    
    # Extraer solo las columnas de señal tónica (todas excepto 'time')
    scl_alta = scl_alta_combined.drop(columns=['time'])
    scl_baja = scl_baja_combined.drop(columns=['time'])
    
    print("Preprocessed files loaded successfully.")
    print(f"- High dose: {scl_alta.shape[1]} subjects, {scl_alta.shape[0]} time points")
    print(f"- Low dose: {scl_baja.shape[1]} subjects, {scl_baja.shape[0]} time points")

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

    #frec_sampleo = int(info_eda_baja_dmt['sampling_rate'])
    frec_sampleo = 250.0
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









