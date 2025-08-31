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

medida = 'SCL' #Opciones: HR, SCL, SCR. Depende de tus carpetas y archivos
preproc_dir = os.path.join(ROOT_DATA, 'Preprocesado', medida)

df_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
df_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
df_alta_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_alta.csv'))
df_baja_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_baja.csv'))


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


#%% Downsampleo la data de SCL y HR que son más, SCR va por otro lado

if medida == 'HR':
    carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
    
elif medida == 'SCL':
    carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']

downsample_rate = 245 #int(len(df_alta)/len(df_interpolated))
# Porque el dowsample_rate es 245 (cuenta de arriba)

xdem_alta = signal.decimate(df_alta_tiempo.values.T, downsample_rate)
xdem_alta = pd.DataFrame(xdem_alta.T)
xdem_alta.columns = carpetas

xdem_baja = signal.decimate(df_baja_tiempo.values.T, downsample_rate)
xdem_baja = pd.DataFrame(xdem_baja.T)
xdem_baja.columns = carpetas

ydem_alta = signal.decimate(df_alta.values.T, downsample_rate)
ydem_alta = pd.DataFrame(ydem_alta.T)
ydem_alta.columns = carpetas

ydem_baja = signal.decimate(df_baja.values.T, downsample_rate)
ydem_baja = pd.DataFrame(ydem_baja.T)
ydem_baja.columns = carpetas
          
# Chequeo de que downsamplee bien
sujeto_a_mirar = 'S04'

plt.figure()
plt.title('Dosis Alta')
plt.plot(df_alta_tiempo[sujeto_a_mirar].values,df_alta[sujeto_a_mirar].values,label='Normal')
plt.plot(xdem_alta[sujeto_a_mirar],ydem_alta[sujeto_a_mirar],'.', label = 'Downsampleada')
plt.legend()
plt.tight_layout()
plt.show()


#%% Upsampleo mi data y correlaciono

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
            # Upsampleo df_dibujo
                # Creating a new index for upsampling
                # Create a new DataFrame to hold 1200 rows
            df_upsampled = pd.DataFrame(index=np.arange(1200), columns=df_dibujo.columns)
                # Fill every 4th row with original data and leave NaNs in between
            df_upsampled.iloc[::4, :] = df_dibujo.values
                # Control de error al hacer algun chiche
            df_upsampled = df_upsampled.apply(pd.to_numeric, errors='coerce')
                # Interpolate to fill in-between NaN values linearly
            df_interpolated = df_upsampled.interpolate(method='linear')
            
            print('Chequeo de que tenga sentido la intepolacion que hago')
                # plt.figure()
                # plt.plot(np.array(df_dibujo.index), df_dibujo['Pleasantness'].values)
                # plt.plot(np.array(df_upsampled.index), df_upsampled['Pleasantness'].values)
                # plt.plot(np.array(df_interpolated.index), df_interpolated['Pleasantness'].values)
                # plt.show()
                
            if medida == 'HR' or medida == 'SCL':
               df_interpolated = df_interpolated[:-abs(len(df_interpolated)-len(ydem_baja))]
            
            for columna in df_dibujo.columns.values:
                
                df_reporte = pd.DataFrame(df_interpolated[columna])
                df_reporte.columns = [carpeta+' '+columna] * len(df_reporte.columns)
                
                df_senal = pd.DataFrame(ydem_baja[carpeta.upper()])
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
            print('hola')
            # Upsampleo df_dibujo
                # Creating a new index for upsampling
                # Create a new DataFrame to hold 1200 rows
            df_upsampled = pd.DataFrame(index=np.arange(1200), columns=df_dibujo.columns)
                # Fill every 4th row with original data and leave NaNs in between
            df_upsampled.iloc[::4, :] = df_dibujo.values
                # Control de error al hacer algun chiche
            df_upsampled = df_upsampled.apply(pd.to_numeric, errors='coerce')
                # Interpolate to fill in-between NaN values linearly
            df_interpolated = df_upsampled.interpolate(method='linear')
            
            if medida == 'HR' or medida == 'SCL':
                df_interpolated = df_interpolated[:-abs(len(df_interpolated)-len(ydem_alta))]
            
            
            for columna in df_dibujo.columns.values:
                
                df_reporte = pd.DataFrame(df_interpolated[columna])
                df_reporte.columns = [carpeta+' '+columna] * len(df_reporte.columns)
                
                df_senal = pd.DataFrame(ydem_alta[carpeta.upper()])
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
pvalores = test(df_spearman_reorg_low, df_spearman_reorg_high)

print(pd.DataFrame.from_dict(pvalores, orient='index', columns=['p-value']))

#%%
plt.figure()
plt.plot(df_spearman_reorg_low['Disembodiment'].values,df_spearman_reorg_high['Disembodiment'].values, 'o')

#%% POSPUESTO DESPUES DE LA CHARLA CON EVAN: Leer los archivos preprocesados de PCA

# medida_2 = 'PCA'

# pc1_alta = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc1_alta.csv')
# pc1_baja = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc1_baja.csv')
# pc2_alta = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc2_alta.csv')
# pc2_baja = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc2_baja.csv')

# #%% Downsampleo con PCA

# print(len(df_alta),'\n',len(df_baja),'\n',len(df_alta_tiempo),'\n',len(df_baja_tiempo))
# print('OJO: esta df_alta acá, asi que si las length de antes son distintas, cagamos')

# sujeto_a_mirar = 'S04'

# if medida == 'HR' or medida == 'SCL':
    
#     if medida == 'HR':
#         carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
    
#     elif medida == 'SCL':
#         carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']
    
#     longitud_pca = 60000 #dividido por 60000 pq es el sample de mis datos de autorreporte, dsps debería poner len(pc1_alta)
    
#     downsample_rate = int(len(df_alta)/longitud_pca) 

#     xdem_alta = signal.decimate(df_alta_tiempo.values.T, downsample_rate)
#     xdem_alta = pd.DataFrame(xdem_alta.T)
#     xdem_alta.columns = carpetas
    
#     xdem_baja = signal.decimate(df_baja_tiempo.values.T, downsample_rate)
#     xdem_baja = pd.DataFrame(xdem_baja.T)
#     xdem_baja.columns = carpetas

#     ydem_alta = signal.decimate(df_alta.values.T, downsample_rate)
#     ydem_alta = pd.DataFrame(ydem_alta.T)
#     ydem_alta.columns = carpetas

#     ydem_baja = signal.decimate(df_baja.values.T, downsample_rate)
#     ydem_baja = pd.DataFrame(ydem_baja.T)
#     ydem_baja.columns = carpetas
    
#     plt.figure()
#     plt.title('Dosis Alta')
#     plt.plot(df_alta_tiempo[sujeto_a_mirar].values,df_alta[sujeto_a_mirar].values,label='Normal')
#     plt.plot(xdem_alta[sujeto_a_mirar],ydem_alta[sujeto_a_mirar],'.', label = 'Downsampleada')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     plt.title('Dosis Baja')
#     plt.plot(df_baja_tiempo[sujeto_a_mirar].values,df_baja[sujeto_a_mirar].values,label='Normal')
#     plt.plot(xdem_baja[sujeto_a_mirar],ydem_baja[sujeto_a_mirar],'.', label = 'Downsampleada')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# elif medida == 'SCR':
    
#     #SCR
#     # carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']
#     #PCA
#     carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S13','S15','S16','S17','S18', 'S19','S20']
    
#     longitud_pca = 60000
    
#     print('ATENCIÓN: SCR hay que upsamplearlo')
#     # downsample_rate = int(longitud_pca/len(df_alta)) 
#     downsample_rate = 13 #este vale mientras pca sea el de 300

#     pc1_alta_dem = signal.decimate(pc1_alta.values.T, downsample_rate)
#     pc1_alta_dem = pd.DataFrame(pc1_alta_dem.T)
#     pc1_alta_dem.columns = carpetas

#     pc1_baja_dem = signal.decimate(pc1_baja.values.T, downsample_rate)
#     pc1_baja_dem = pd.DataFrame(pc1_baja_dem.T)
#     pc1_baja_dem.columns = carpetas

#     pc2_alta_dem = signal.decimate(pc2_alta.values.T, downsample_rate)
#     pc2_alta_dem = pd.DataFrame(pc2_alta_dem.T)
#     pc2_alta_dem.columns = carpetas

#     pc2_baja_dem = signal.decimate(pc2_baja.values.T, downsample_rate)
#     pc2_baja_dem = pd.DataFrame(pc2_baja_dem.T)
#     pc2_baja_dem.columns = carpetas
    
#     plt.figure()
#     plt.title('Dosis Alta')
#     plt.plot(pc1_alta[sujeto_a_mirar].values,pc2_alta[sujeto_a_mirar].values,label='Normal')
#     plt.plot(pc1_alta_dem[sujeto_a_mirar],pc2_alta_dem[sujeto_a_mirar],'.', label = 'Downsampleada')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     plt.figure()
#     plt.title('Dosis Baja')
#     plt.plot(pc1_baja[sujeto_a_mirar].values,pc2_baja[sujeto_a_mirar].values,label='Normal')
#     plt.plot(pc1_baja_dem[sujeto_a_mirar],pc2_baja_dem[sujeto_a_mirar],'.', label = 'Downsampleada')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    



