# -*- coding: utf-8 -*-
from warnings import warn
import os

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
import statsmodels.stats.multitest as st
from scipy import signal
import scipy.io

import seaborn as sns

#%% Agarro los datos de los reportes

import os

# Define el directorio que quieres leer
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
directorio = os.path.join(ROOT_DATA, 'resampled')

# Obtiene la lista de archivos en el directorio y los ordena alfabeticamente
archivos = sorted(os.listdir(directorio))

# Filtra solo los archivos
archivos_ordenados = [archivo for archivo in archivos if os.path.isfile(os.path.join(directorio, archivo))]


#%% Leer los archivos preprocesados

medida = 'SCR' #Opciones: HR, SCL, SCR. Depende de tus carpetas y archivos
preproc_dir = os.path.join(ROOT_DATA, 'Preprocesado', medida)

df_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
df_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
df_alta_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_alta.csv'))
df_baja_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_baja.csv'))

carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']

#%% Tabla de dosis

dosis = [['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta']]

columnas = ['DMT_1', 'DMT_2'] # definimos los nombres de las columnas
indices = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s15','s16','s17','s18', 's19','s20']

dosis = pd.DataFrame(dosis, columns=columnas, index = indices)


#%% Downsampleo la data de SCR va por otro lado y correlaciono

plt.close('all')

# Creo un dataframe vacio para poder meterle la info
df_pearson = pd.DataFrame(columns=["Alta", "Baja"])
df_spearman = pd.DataFrame(columns=["Alta", "Baja"])

print('CHEQUEAR DATOS DE PEARSON y SPEARMAN CON TODOS LOS SUJETOS')
# OJO: esta idea funciona pq no cambio de sujeto antes de pasar por su dosis alta y baja
    # Creo un diccionario temporal para ir guardando la data
temp_storage_pearson = {}
temp_storage_spearman = {}


for fname in archivos_ordenados[:]:
    
    mat = scipy.io.loadmat(os.path.join(directorio, fname))

    dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

    df_dibujo = pd.DataFrame(mat['dimensions'], columns = dimensiones)
    
    carpeta = fname[0:3]
    experimento = fname[4:6]
    
    if experimento == 'DM':
        experimento = experimento + fname[6:7] + '_' + fname[15:16]
        
    else:
        experimento = experimento + '_' + fname[14:15]
    
    if (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Baja': 

        if carpeta.upper() in carpetas:
            
            downsample_rate = 15
            
            # Downsampleo los reportes
            reporte_baja = signal.decimate(df_dibujo.values.T, downsample_rate)
            reporte_baja = pd.DataFrame(reporte_baja.T)
            reporte_baja.columns = dimensiones

            print("Largo del reporte:", len(reporte_baja))
            print("Largo de SCR:", len(df_baja))
            
            reporte_baja = reporte_baja[:-abs(len(df_baja)-len(reporte_baja))]
            
            for columna in df_dibujo.columns.values:
                
                df_reporte = pd.DataFrame(reporte_baja[columna])
                df_reporte.columns = [carpeta+' '+columna] * len(df_reporte.columns)
                
                df_senal = pd.DataFrame(df_baja[carpeta.upper()])
                df_senal.columns = [carpeta+' '+columna] * len(df_senal.columns)
                
                # Hecho eso hago la correlacion entre los df_dibujo[columna] y df_alta[]
                df_pearson_baja = pd.DataFrame(df_reporte).corrwith(df_senal)                
                df_spearman_baja = pd.DataFrame(df_reporte).corrwith(df_senal, method = 'spearman')

                print(df_spearman_baja)
                key_pearson = df_pearson_baja.index[0]
                key_spearman = df_spearman_baja.index[0]
                
                # Pearson
                if key_pearson in temp_storage_pearson and 'Alta' in temp_storage_pearson[key_pearson]:
                    # Add Baja data and append row to DataFrame
                    temp_storage_pearson[key_pearson]['Baja'] = df_pearson_baja.values[0]
                    
                    row_to_append_pearson = pd.DataFrame({
                        "Alta": [temp_storage_pearson[key_pearson]['Alta']],
                        "Baja": [temp_storage_pearson[key_pearson]['Baja']]
                    }, index=[key_pearson])
                    
                    df_pearson = pd.concat([df_pearson, row_to_append_pearson])
                    del temp_storage_pearson[key_pearson]
                    
                else:
                    # Store the series_alta data
                    temp_storage_pearson[key_pearson] = {'Baja': df_pearson_baja.values[0]}
                
                # Spearman    
                if key_spearman in temp_storage_spearman and 'Alta' in temp_storage_spearman[key_spearman]:
                    # Add Baja data and append row to DataFrame
                    temp_storage_spearman[key_spearman]['Baja'] = df_spearman_baja.values[0]
                    
                    row_to_append_spearman = pd.DataFrame({
                        "Alta": [temp_storage_spearman[key_spearman]['Alta']],
                        "Baja": [temp_storage_spearman[key_spearman]['Baja']]
                    }, index=[key_spearman])
                    
                    df_spearman = pd.concat([df_spearman, row_to_append_spearman])
                    del temp_storage_spearman[key_spearman]
                    
                else:
                    # Store the series_alta data
                    temp_storage_spearman[key_spearman] = {'Baja': df_spearman_baja.values[0]}

            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta':
          
        if carpeta.upper() in carpetas:
            print('Separo')
            
            downsample_rate = 15
            # Downsampleo los reportes
            reporte_alta = signal.decimate(df_dibujo.values.T, downsample_rate)
            reporte_alta = pd.DataFrame(reporte_alta.T)
            reporte_alta.columns = dimensiones

            print("Largo del reporte:", len(reporte_alta))
            print("Largo de SCR:", len(df_alta))
            
            reporte_alta = reporte_alta[:-abs(len(df_alta)-len(reporte_alta))]
            
            for columna in df_dibujo.columns.values:
                
                df_reporte = pd.DataFrame(reporte_alta[columna])
                df_reporte.columns = [carpeta+' '+columna] * len(df_reporte.columns)
                
                df_senal = pd.DataFrame(df_alta[carpeta.upper()])
                df_senal.columns = [carpeta+' '+columna] * len(df_senal.columns)
                
                # Hecho eso hago la correlacion entre los df_dibujo[columna] y df_alta[]
                df_pearson_alta = pd.DataFrame(df_reporte).corrwith(df_senal)
                df_spearman_alta = pd.DataFrame(df_reporte).corrwith(df_senal, method = 'spearman')
                
                print(df_spearman_alta)
                key_pearson = df_pearson_alta.index[0]
                key_spearman = df_spearman_alta.index[0]
                
                if key_pearson in temp_storage_pearson and 'Baja' in temp_storage_pearson[key_pearson]:
                    
                    # Add Alta data and append row to DataFrame
                    temp_storage_pearson[key_pearson]['Alta'] = df_pearson_alta.values[0]
                    
                    row_to_append_pearson = pd.DataFrame({
                        "Alta": [temp_storage_pearson[key_pearson]['Alta']],
                        "Baja": [temp_storage_pearson[key_pearson]['Baja']]
                    }, index=[key_pearson])
                    
                    df_pearson = pd.concat([df_pearson, row_to_append_pearson])
                    del temp_storage_pearson[key_pearson]
                    
                else:
                    # Store the series_alta data
                    temp_storage_pearson[key_pearson] = {'Alta': df_pearson_alta.values[0]}
                    
                if key_spearman in temp_storage_spearman and 'Baja' in temp_storage_spearman[key_spearman]:
                    
                    # Add Alta data and append row to DataFrame
                    temp_storage_spearman[key_spearman]['Alta'] = df_spearman_alta.values[0]
                    
                    row_to_append_spearman = pd.DataFrame({
                        "Alta": [temp_storage_spearman[key_spearman]['Alta']],
                        "Baja": [temp_storage_spearman[key_spearman]['Baja']]
                    }, index=[key_spearman])
                    
                    df_spearman = pd.concat([df_spearman, row_to_append_spearman])
                    del temp_storage_spearman[key_spearman]
                    
                else:
                    # Store the series_alta data
                    temp_storage_spearman[key_spearman] = {'Alta': df_spearman_alta.values[0]}
    

#%% Reorganizacion de Pearson y Spearman dataframe para el boxplot

# Separate the index into carpet and dimension
df_pearson.index = df_pearson.index.str.split(' ', expand=True)
df_pearson.index.names = ['Carpet', 'Dimension']

# Create the Alta DataFrame with carpets as index and dimensions as columns
df_pearson_reorg_high = df_pearson['Alta'].unstack(level='Dimension')
df_pearson_reorg_high.index.name = None  # Remove index name

# Create the Baja DataFrame with carpets as index and dimensions as columns
df_pearson_reorg_low = df_pearson['Baja'].unstack(level='Dimension')
df_pearson_reorg_low.index.name = None  # Remove index name

# Separate the index into carpet and dimension
df_spearman.index = df_spearman.index.str.split(' ', expand=True)
df_spearman.index.names = ['Carpet', 'Dimension']

# Create the Alta DataFrame with carpets as index and dimensions as columns
df_spearman_reorg_high = df_spearman['Alta'].unstack(level='Dimension')
df_spearman_reorg_high.index.name = None  # Remove index name

# Create the Baja DataFrame with carpets as index and dimensions as columns
df_spearman_reorg_low = df_spearman['Baja'].unstack(level='Dimension')
df_spearman_reorg_low.index.name = None  # Remove index name


#%% Armo los Dataframes para el Box Plot
df_pearson_reorg_high['Source'] = 'High Dose'
df_pearson_reorg_low['Source'] = 'Low Dose'

df_spearman_reorg_high['Source'] = 'High Dose'
df_spearman_reorg_low['Source'] = 'Low Dose'

df_boxplot_pearson = pd.concat([df_pearson_reorg_high, df_pearson_reorg_low])
df_boxplot_spearman = pd.concat([df_spearman_reorg_high, df_spearman_reorg_low])

# Melt the DataFrame to have a long-form DataFrame suitable for seaborn
df_melted_pearson = pd.melt(df_boxplot_pearson, id_vars=['Source'], var_name='Dimensiones', value_name='Valor de Correlación')
df_melted_spearman = pd.melt(df_boxplot_spearman, id_vars=['Source'], var_name='Dimensiones', value_name='Valor de Correlación')


#%% Ploteo el box plot

plt.close('all')
# Plot using seaborn
plt.figure(figsize=(10, 6))
plt.title('Pearson - '+ medida)
sns.boxplot(x='Dimensiones', y='Valor de Correlación', hue='Source', data = df_melted_pearson)
plt.xlabel('')
plt.ylabel('Valor de Correlación',fontsize = 14)
plt.tick_params(axis = 'y', labelsize = 14)
# Rotar y ajustar las etiquetas del eje x
plt.xticks(rotation = 45, ha='right', fontsize=14)
# Eliminar el nombre de la leyenda
plt.legend(title='', loc = 'lower right', fontsize = 16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.title('Spearman - '+ medida)
sns.boxplot(x='Dimensiones', y='Valor de Correlación', hue='Source', data = df_melted_spearman)
plt.xlabel('')
plt.ylabel('Valor de Correlación',fontsize = 14)
plt.tick_params(axis = 'y', labelsize = 14)
# Rotar y ajustar las etiquetas del eje x
plt.xticks(rotation = 45, ha='right', fontsize=14)
# Eliminar el nombre de la leyenda
plt.legend(title='', loc = 'lower right', fontsize = 16)
plt.tight_layout()
plt.show()

#%% Hago el wilcoxon test con cada dimension
    
from joblib import Parallel, delayed

def test(baja, alta):
    """
    Calculates the p-values of the Wilcoxon test to compare corresponding columns in two DataFrames.

    Parameters:
    baja: DataFrame with values (low group)
    alta: DataFrame with values (high group)

    Returns:
    pvalores: Dictionary of p-values
    """
    
    columnas_sin_source = baja.columns[:-1]
    
    def compute_pvalue(col_name):
        arr_baja = baja[col_name].values
        arr_alta = alta[col_name].values
        return col_name, stats.wilcoxon(arr_baja, arr_alta).pvalue
    
    # Parallelize the p-value computation for each column
    resultados = Parallel(n_jobs=-1)(delayed(compute_pvalue)(col) for col in columnas_sin_source)
    
    # Convert the list of results to a dictionary
    pvalores = {col: round(pvalue, 8) for col, pvalue in resultados}
    
    return pvalores

# Mismo codigo de PCA todos pero con Dataframes
pvalores = test(df_pearson_reorg_low, df_pearson_reorg_high)

print(pd.DataFrame.from_dict(pvalores, orient='index', columns=['p-value']))

              
#%%