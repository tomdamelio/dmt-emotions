# -*- coding: utf-8 -*-

import os

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Preprocesado y modelado
# ==============================================================================
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

#%% Esto me agarra la carpeta donde estan los archivos, más facil para no tener que ordenar todo

import os

# Define el directorio que quieres leer
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
directorio = os.path.join(ROOT_DATA, 'resampled')

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


#%% Hago la funciona que plotea todos los sujetos por cada dibujo, si puedo divido tambien por dosis

def dibujos_divididos(fname):
    
    size = 14
    
    mat = scipy.io.loadmat(os.path.join(directorio, fname))

    dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

    df_dibujo = pd.DataFrame(mat['dimensions'], columns = dimensiones)
    print(df_dibujo.columns)
    
    carpeta = fname[0:3]
    experimento = fname[4:6]
    print(experimento)
    
    i = 0
    
    if experimento == 'DM':
        experimento = experimento + fname[6:7] + '_' + fname[15:16]
        print(carpeta + '-' + experimento)
        
    else:
        experimento = experimento + '_' + fname[14:15]
        print(carpeta + '-' + experimento)
    
    if (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Baja': 
    
        tiempo = np.arange(0, 1200, 4) #tiempo de sampleo de Evan
    
        for columna in df_dibujo.columns.values:
            plt.figure(i)
            plt.title(columna + ' - Low Dose')
            plt.plot(tiempo, df_dibujo[columna].values, label = carpeta)
            plt.ylabel('Reported Value', fontsize = size)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.legend(ncol = 2)
            plt.show()
        
            i += 1
            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta':
            
        tiempo = np.arange(0, 1200, 4) #tiempo de sampleo de Evan
    
        for columna in df_dibujo.columns.values:
            plt.figure(i + 15*1)
            plt.title(columna + ' - High Dose')
            plt.plot(tiempo, df_dibujo[columna].values, label = carpeta)
            plt.ylabel('Reported Value', fontsize = size)
            plt.xlabel('Time (s)', fontsize = size)
            plt.xticks(fontsize = size)
            plt.yticks(fontsize = size)
            plt.tight_layout()
            plt.legend(ncol = 2)
            plt.show()
        
            i += 1

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre acá   
    else:
        
        tiempo = np.arange(0, 600, 4) #tiempo de sampleo de Evan
    
        for columna in df_dibujo.columns.values:
            plt.figure(i + 15*2)
            plt.title(columna)
            plt.plot(tiempo, df_dibujo[columna].values, label = carpeta)
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.legend()
            plt.show()
        
            i += 1


#%% Grafico todos los sujetos para cada dibujo (son 44, ojo)

plt.close('all')

for archivo in archivos_ordenados[:]:
    
    dibujos_divididos(archivo)
    
#%% Agarro todos los mats y le ploteo a cada uno todos sus dibujos

def dibujos_por_sujeto(fname, numero):
    
    mat = scipy.io.loadmat(os.path.join(directorio, fname))

    dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

    df_dibujo = pd.DataFrame(mat['dimensions'], columns = dimensiones)
    print(df_dibujo.columns)
    
    carpeta = fname[0:3]
    experimento = fname[4:6]
    print(experimento)
    
    if experimento == 'DM':
        experimento = experimento + fname[6:7] + '_' + fname[15:16]
        print(carpeta + '-' + experimento)
        
    else:
        experimento = experimento + '_' + fname[14:15]
        print(carpeta + '-' + experimento)
    
    if experimento == 'DMT_1' or experimento == 'DMT_2':
    
        tiempo = np.arange(0, 1200, 4) #tiempo de sampleo de Evan
    
        plt.title(carpeta + '-' + experimento)
        plt.figure(numero)
        for columna in df_dibujo.columns.values:
            plt.plot(tiempo, df_dibujo[columna].values, label = columna)
            plt.xlabel('Tiempo (s)')
            plt.tight_layout()
            plt.show()
        plt.legend()

        
    else:
        
        tiempo = np.arange(0, 600, 4) #tiempo de sampleo de Evan
    
        plt.title(carpeta + '-' + experimento)
        plt.figure(numero)
        for columna in df_dibujo.columns.values:
            plt.plot(tiempo, df_dibujo[columna].values, label = columna)
            plt.xlabel('Tiempo (s)')
            plt.tight_layout()
            plt.show()
        plt.legend()

        
    
#%% Grafico todos los dibujos de todos los sujetos (son 44, ojo)

i = 0

for archivo in archivos_ordenados[:-10]:
    
    dibujos_por_sujeto(archivo, i)
    i += 1

#%% Calculo los promedios de los sujetos

# Este primer codigo me deja diferenciar mis columnas dentro de los diccionarios sin tener una lista de listas, que me era desprolijo
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

promedio_alta = {}
promedio_baja = {}
error_alta = {}
error_baja = {}

for nombre in dimensiones:
    promedio_alta[nombre] = []
    promedio_baja[nombre] = []
    error_alta[nombre] = []
    error_baja[nombre] = []
    

def promedio(fname):

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
        for columna in df_dibujo.columns.values:
            prom = df_dibujo[columna].mean()
            promedio_baja[columna].append(prom)
            
            error = df_dibujo[columna].std()/np.sqrt(df_dibujo[columna].shape[0])
            error_baja[columna].append(error)
            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta':
            
        # elijo calcular el promedio de toda la señal de cada uno pq no sé cuando pasan los picos en el tiempo
        # este experimento no tiene bien asociado el autoreporte al tiempo, se dibujo uniformemente distribudo para comparar entre ambos, que uno sucede despues del otro
        for columna in df_dibujo.columns.values:
            prom = df_dibujo[columna].mean()
            promedio_alta[columna].append(prom)
            
            error = df_dibujo[columna].std()/np.sqrt(df_dibujo[columna].shape[0])
            error_alta[columna].append(error)

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre acá


#%% Corro sobre todos los archivos el promediado

for archivo in archivos_ordenados[:]:
    
    promedio(archivo)

promedio_final_alta = {}
promedio_final_baja = {}
error_final_alta = {}
error_final_baja = {}
# 
# for nombre in dimensiones:
#     promedio_final_alta[nombre] = []
#     promedio_final_baja[nombre] = []
#     error_final_alta[nombre] = []
#     error_final_baja[nombre] = []

# Esto lo voy a usar para comparar con lo que me de el box plot
for columna in dimensiones:
    promedio_final_alta[columna] = np.mean(promedio_alta[columna])
    
    # prom_final_baja = np.mean(promedio_baja[columna])
    promedio_final_baja[columna] = np.mean(promedio_baja[columna])
    
    # err_final_alta = np.mean(error_alta[columna])
    error_final_alta[columna] = np.mean(error_alta[columna])
    
    # err_final_baja = np.mean(error_baja[columna])
    error_final_baja[columna] = np.mean(error_baja[columna])

#%% Cambio los dics a dataframes para hacer el box plot

df_promedio_alta = pd.DataFrame(promedio_alta) # uso esto y no promedio_final_alta pq el box plot ya va a calcular la mediana de estos datos
df_promedio_baja = pd.DataFrame(promedio_baja) # no lo hice antes porque me sirve comparar entre promedios

#%%
df_promedio_alta['Source'] = 'Dosis Alta'
df_promedio_baja['Source'] = 'Dosis Baja'

df_boxplot = pd.concat([df_promedio_alta, df_promedio_baja])

# Melt the DataFrame to have a long-form DataFrame suitable for seaborn
df_melted = pd.melt(df_boxplot, id_vars=['Source'], var_name='Dimensiones', value_name='Valor Promedio')

#%% Ploteo el box plot

plt.close('all')
# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Dimensiones', y='Valor Promedio', hue='Source', data = df_melted)
plt.xlabel('')
plt.tick_params(axis = 'y', labelsize = 14)
# Rotar y ajustar las etiquetas del eje x
plt.xticks(rotation = 45, ha='right', fontsize=12)
# Eliminar el nombre de la leyenda
plt.legend(title='', fontsize = 14)
plt.tight_layout()
plt.show()


#%% Calculo los maximos de los sujetos

# Este primer codigo me deja diferenciar mis columnas dentro de los diccionarios sin tener una lista de listas, que me era desprolijo
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']

maximo_alta = {}
maximo_baja = {}
error_alta = {}
error_baja = {}

for nombre in dimensiones:
    maximo_alta[nombre] = []
    maximo_baja[nombre] = []
    error_alta[nombre] = []
    error_baja[nombre] = []
    

def maximo(fname):
    
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
    
    if (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Baja' and carpeta != 's12': 
    
        # elijo calcular el maximo de toda la señal de cada uno pq no sé cuando pasan los picos en el tiempo
        # este experimento no tiene bien asociado el autoreporte al tiempo, se dibujo uniformemente distribudo para comparar entre ambos, que uno sucede despues del otro
        for columna in df_dibujo.columns.values:
            prom = df_dibujo[columna].max()
            maximo_baja[columna].append(prom)
            
            error = df_dibujo[columna].std()/np.sqrt(df_dibujo[columna].shape[0])
            error_baja[columna].append(error)
            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta' and carpeta != 's12':
            
        # elijo calcular el maximo de toda la señal de cada uno pq no sé cuando pasan los picos en el tiempo
        # este experimento no tiene bien asociado el autoreporte al tiempo, se dibujo uniformemente distribudo para comparar entre ambos, que uno sucede despues del otro
        for columna in df_dibujo.columns.values:
            prom = df_dibujo[columna].max()
            maximo_alta[columna].append(prom)
            
            error = df_dibujo[columna].std()/np.sqrt(df_dibujo[columna].shape[0])
            error_alta[columna].append(error)

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre acá


#%% Corro sobre todos los archivos el promediado

for archivo in archivos_ordenados[:]:
    
    maximo(archivo)

maximo_final_alta = {}
maximo_final_baja = {}
error_final_alta = {}
error_final_baja = {}
# 
# for nombre in dimensiones:
#     maximo_final_alta[nombre] = []
#     maximo_final_baja[nombre] = []
#     error_final_alta[nombre] = []
#     error_final_baja[nombre] = []

# Esto lo voy a usar para comparar con lo que me de el box plot
for columna in dimensiones:
    maximo_final_alta[columna] = np.max(maximo_alta[columna])
    
    # prom_final_baja = np.mean(maximo_baja[columna])
    maximo_final_baja[columna] = np.max(maximo_baja[columna])
    
    # err_final_alta = np.mean(error_alta[columna])
    error_final_alta[columna] = np.max(error_alta[columna])
    
    # err_final_baja = np.mean(error_baja[columna])
    error_final_baja[columna] = np.max(error_baja[columna])

#%% Cambio los dics a dataframes para hacer el box plot

df_maximo_alta = pd.DataFrame(maximo_alta) # uso esto y no maximo_final_alta pq el box plot ya va a calcular la mediana de estos datos
df_maximo_baja = pd.DataFrame(maximo_baja) # no lo hice antes porque me sirve comparar entre maximos

#%%
df_maximo_alta['Source'] = 'High Dose'
df_maximo_baja['Source'] = 'Low Dose'

df_boxplot = pd.concat([df_maximo_alta, df_maximo_baja])

# Melt the DataFrame to have a long-form DataFrame suitable for seaborn
df_melted = pd.melt(df_boxplot, id_vars=['Source'], var_name='Dimensiones', value_name='Maximum Value')

#%% Ploteo el box plot

plt.close('all')
# Plot using seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Dimensiones', y='Maximum Value', hue='Source', data = df_melted)
plt.xlabel('')
plt.ylabel('Maximum Value',fontsize = 14)
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
    Calcula los valores p de la prueba de Wilcoxon para comparar columnas correspondientes en dos diccionarios de listas.

    Parámetros:
    baja: diccionario de listas de valores (grupo bajo)
    alta: diccionario de listas de valores (grupo alto)

    Retorna:
    pvalores: diccionario de valores p
    """
    def compute_pvalue(col_name):
        arr_baja = np.array(baja[col_name])
        arr_alta = np.array(alta[col_name])
        return col_name, stats.wilcoxon(arr_baja, arr_alta).pvalue
    
    # Paralelizo el cálculo de valores p para cada columna
    resultados = Parallel(n_jobs=-1)(delayed(compute_pvalue)(col) for col in baja.keys())
    
    # Convertir la lista de resultados a un diccionario
    pvalores = {col: round(pvalue,3) for col, pvalue in resultados}
    
    return pvalores

pvalores = test(maximo_baja, maximo_alta)

print(pd.DataFrame.from_dict(pvalores, orient = 'index'))


#%%

    
    


    
    
    
    
