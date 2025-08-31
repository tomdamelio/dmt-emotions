# -*- coding: utf-8 -*-

import os

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

# Statistics
from scipy import stats
import statsmodels.stats.multitest as st

#%% Esto me agarra la carpeta donde estan los archivos, más facil para no tener que ordenar todo

import os

ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

# Define el directorio que quieres leer
directorio = os.path.join(ROOT_DATA, "resampled")

# Obtiene la lista de archivos en el directorio y los ordena alfabeticamente
archivos = sorted(os.listdir(directorio))

# Filtra solo los archivos
archivos_ordenados = [archivo for archivo in archivos if os.path.isfile(os.path.join(directorio, archivo))]

# Imprime los archivos en orden
# for archivo in archivos_ordenados:
#     print(archivo)

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
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Alta', 'Baja'],
         ['Alta', 'Baja'],
         ['Baja', 'Alta'],
         ['Baja', 'Alta']]

columnas = ['DMT_1', 'DMT_2'] # definimos los nombres de las columnas
indices = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s13','s15','s16','s17','s18', 's19','s20']

dosis = pd.DataFrame(dosis, columns=columnas, index = indices)

#%% Agarro todos los mats, los meto en listas de dataframes separadas
# Dsps voy a juntarlas todas y concatenar, pero prefiero asi, asi esta todo ordenado

alta = []
baja = []
rs_alta = []
rs_baja = [] 

def concatenacion(fname):
    
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
    
        # elijo calcular el promedio de toda la señal de cada uno pq no sé cuando pasan los picos en el tiempo
        # este experimento no tiene bien asociado el autoreporte al tiempo, se dibujo uniformemente distribudo para comparar entre ambos, que uno sucede despues del otro
        baja.append(df_dibujo)
            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta':
            
        # elijo calcular el promedio de toda la señal de cada uno pq no sé cuando pasan los picos en el tiempo
        # este experimento no tiene bien asociado el autoreporte al tiempo, se dibujo uniformemente distribudo para comparar entre ambos, que uno sucede despues del otro
        alta.append(df_dibujo)

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre acá
    elif (experimento == 'RS_1' or experimento == 'RS_2') and dosis['DMT_' + experimento[3]][carpeta] == 'Alta':
        
        rs_alta.append(df_dibujo)
        
    else:
        
        rs_baja.append(df_dibujo)
        
#%% Corro la funcion anterior

for archivo in archivos:
    concatenacion(archivo)
    
# Paso las listas de dataframes a dataframes concatenados
alta = pd.concat(alta, ignore_index = True)
baja = pd.concat(baja, ignore_index = True)
# rs_alta = pd.concat(rs_alta, ignore_index = True)
# rs_baja = pd.concat(rs_baja, ignore_index = True)
# Esta bien si tengo 300*18 en alta, 300*19 en baja, 150*19 y 150*19
    
todos_dfs = [alta, baja]#, rs_alta, rs_baja]

df_concatenados = pd.concat(todos_dfs, ignore_index = True)

#%%

csv = 'Datos_reportes_para_clusterizar_sin_reposo.csv'
cluster_dir = os.path.join(ROOT_DATA, 'Data Cluster')
os.makedirs(cluster_dir, exist_ok=True)
df_concatenados.to_csv(os.path.join(cluster_dir, csv), index=False)

#%%

df_concat = pd.read_csv(os.path.join(cluster_dir, csv))
#%%