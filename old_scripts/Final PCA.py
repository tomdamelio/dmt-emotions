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

ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

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
rs_alta = pd.concat(rs_alta, ignore_index = True)
rs_baja = pd.concat(rs_baja, ignore_index = True)
# Esta bien si tengo 300*18 en alta, 300*19 en baja, 150*19 y 150*19
    
todos_dfs = [alta, baja, rs_alta, rs_baja]

    
#%% Concateno ahora y corro el pca con fit transform

df_concatenados = pd.concat(todos_dfs, ignore_index = True)

indices = ['PC1','PC2']
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
        'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
        'Temporality', 'General_Intensity']


# Componentes a quedarme, importante aclararlo
cantidad_pc = 2 # son 15, si resto por 12 me quedo con las primeras 3 componentes

# Creo un pipeline donde le hace el standard Scaler (que pone cada columna independiente a un z score)
# y después hace el pca, le aclaro que me voy a quedar con la cantidad de PC aclarado
X = scale(df_concatenados)
pca = PCA(n_components = cantidad_pc)
X = pca.fit_transform(X)

df_pca = pd.DataFrame(data = X, columns = indices)


loadings = pd.DataFrame(pca.components_.T, columns = indices, index = dimensiones)
top_3_pc1 = loadings['PC1'].abs().nlargest(3)
top_3_pc2 = loadings['PC2'].abs().nlargest(3)

print('PC1:\n', top_3_pc1)
print()
print('PC2:\n', top_3_pc2)
print()
print('PC1 explained variance ratio:', round(pca.explained_variance_ratio_[0],3)*100, '%')
print('PC2 explained variance ratio:', round(pca.explained_variance_ratio_[1],3)*100, '%')


#%% Armo un heat map para mostrar los loadings

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
plt.imshow(abs(loadings), cmap='viridis', aspect='auto')
plt.yticks(range(len(dimensiones)), dimensiones)
plt.xticks(range(cantidad_pc), np.arange(cantidad_pc) + 1)
plt.grid(False)
plt.colorbar()

#%% Separo los dataframes en condiciones

sujetos = 18 # el s12 no tiene un archivo asi que di de baja todos los suyos

df_alta = df_pca[:300*sujetos].reset_index(drop = True)
df_baja = df_pca[300*sujetos:300*sujetos + 300*sujetos].reset_index(drop = True)
df_rs_alta = df_pca[300*sujetos + 300*sujetos : 300*sujetos + 300*sujetos + 150*sujetos].reset_index(drop = True)
df_rs_baja = df_pca[300*sujetos + 300*sujetos + 150*sujetos : 300*sujetos + 300*sujetos + 150*sujetos + 150*sujetos].reset_index(drop = True)

dfs = [df_alta,df_baja,df_rs_alta,df_rs_baja]

#%% Separo sujetos y ploteo sobre cada condicion

size = 14

plt.close('all')

xlim = -5
xlimfin = 9
ylim = -5
ylimfin = 5
    
for i in range(len(dfs)):
    
    if i == 0:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/300)): #pq tienen distinta cantidad de sujetos
            x = dataframe['PC1'].loc[i*300:(i+1)*300-1]
            y = dataframe['PC2'].loc[i*300:(i+1)*300-1]
            
            plt.figure(1)
            plt.title('High Dose', fontsize = size)
            plt.plot(x, y)
            plt.ylabel('Principal Component 2', fontsize = size)
            plt.xlabel('Principal Component 1', fontsize = size)
            plt.xticks(fontsize = size)
            plt.yticks(fontsize = size)
            plt.xlim(xlim,xlimfin)
            plt.ylim(ylim,ylimfin)
            plt.tight_layout()
            plt.show()
            
        
    elif i == 1: 
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/300)):
            x = dataframe['PC1'].loc[i*300:(i+1)*300-1]
            y = dataframe['PC2'].loc[i*300:(i+1)*300-1]
            
            plt.figure(2)
            plt.title('Low Dose', fontsize = size)
            plt.plot(x, y)
            plt.ylabel('Principal Component 2', fontsize = size)
            plt.xlabel('Principal Component 1', fontsize = size)
            plt.xticks(fontsize = size)
            plt.yticks(fontsize = size)
            plt.xlim(xlim,xlimfin)
            plt.ylim(ylim,ylimfin)
            plt.tight_layout()
            plt.show()
            
    elif i == 2:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/150)):
            x = dataframe['PC1'].loc[i*150:(i+1)*150-1]
            y = dataframe['PC2'].loc[i*150:(i+1)*150-1]
            
            plt.figure(3)
            plt.title('Resting High Dose', fontsize = size)
            plt.plot(x, y)
            plt.ylabel('Principal Component 2', fontsize = size)
            plt.xlabel('Principal Component 1', fontsize = size)
            plt.xticks(fontsize = size)
            plt.yticks(fontsize = size)
            plt.xlim(xlim,xlimfin)
            plt.ylim(ylim,ylimfin)
            plt.tight_layout()
            plt.show()
            
    else:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/150)):
            x = dataframe['PC1'].loc[i*150:(i+1)*150-1]
            y = dataframe['PC2'].loc[i*150:(i+1)*150-1]
            
            plt.figure(4)
            plt.title('Resting Low Dose', fontsize = size)
            plt.plot(x, y)
            plt.ylabel('Principal Component 2', fontsize = size)
            plt.xlabel('Principal Component 1', fontsize = size)
            plt.xticks(fontsize = size)
            plt.yticks(fontsize = size)
            plt.xlim(xlim,xlimfin)
            plt.ylim(ylim,ylimfin)
            plt.tight_layout()
            plt.show()
            
#%% Mismo codigo que antes pero guardo en listas para promediar 

promedio_pc1_alta = []
promedio_pc2_alta = []
promedio_pc1_baja = []
promedio_pc2_baja = []


for i in range(len(dfs)):
    
    if i == 0:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/300)): #pq tienen distinta cantidad de sujetos
            x = dataframe['PC1'].loc[i*300:(i+1)*300-1].reset_index(drop = True)
            y = dataframe['PC2'].loc[i*300:(i+1)*300-1].reset_index(drop = True)
            
            promedio_pc1_alta.append(x)
            promedio_pc2_alta.append(y)
        
    elif i == 1: 
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/300)):
            x = dataframe['PC1'].loc[i*300:(i+1)*300-1].reset_index(drop = True)
            y = dataframe['PC2'].loc[i*300:(i+1)*300-1].reset_index(drop = True)
            
            promedio_pc1_baja.append(x)
            promedio_pc2_baja.append(y)
            
            
    elif i == 2:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/150)):
            x = dataframe['PC1'].loc[i*150:(i+1)*150-1].reset_index(drop = True)
            y = dataframe['PC2'].loc[i*150:(i+1)*150-1].reset_index(drop = True)
            
            
    else:
        
        dataframe = dfs[i]
        for i in range(int(len(dataframe)/150)):
            x = dataframe['PC1'].loc[i*150:(i+1)*150-1].reset_index(drop = True)
            y = dataframe['PC2'].loc[i*150:(i+1)*150-1].reset_index(drop = True)


#%% Guardado de archivos de HR para después promediar o comparar con PCA más facil

carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S13','S15','S16','S17','S18', 'S19','S20']

medida = 'PCA'

#%%
# Lista de datos procesados pasada a dataframe y guardada en csv
df_alta_guardar = pd.DataFrame(promedio_pc1_alta).T
df_alta_guardar.columns = carpetas
csv = f'{medida}_pc1_alta.csv'
preproc_dir = os.path.join(ROOT_DATA, 'Preprocesado', medida)
os.makedirs(preproc_dir, exist_ok=True)
df_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_baja_guardar = pd.DataFrame(promedio_pc1_baja).T
df_baja_guardar.columns = carpetas
csv = f'{medida}_pc1_baja.csv'
df_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_tiempo_alta_guardar = pd.DataFrame(promedio_pc2_alta).T
df_tiempo_alta_guardar.columns = carpetas
csv = f'{medida}_pc2_alta.csv'
df_tiempo_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_tiempo_baja_guardar = pd.DataFrame(promedio_pc2_baja).T
df_tiempo_baja_guardar.columns = carpetas
csv = f'{medida}_pc2_baja.csv'
df_tiempo_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)


#%% Leer los archivos preprocesados

pc1_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_pc1_alta.csv'))
pc1_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_pc1_baja.csv'))
pc2_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_pc2_alta.csv'))
pc2_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_pc2_baja.csv'))



#%% Promedio y agarro el error estandar

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

alta_promedio_listas_pc1, alta_errores_pc1 = calculate_means_and_stdevs(pc1_alta)
alta_promedio_listas_pc2, alta_errores_pc2 = calculate_means_and_stdevs(pc2_alta)
baja_promedio_listas_pc1, baja_errores_pc1 = calculate_means_and_stdevs(pc1_baja)
baja_promedio_listas_pc2, baja_errores_pc2 = calculate_means_and_stdevs(pc2_baja)

#%% Ploteo con fill between y grafico el promedio

# plt.close('all')

plt.figure()
tiempo = np.arange(0, 1200, 4)

relleno_pos_alta = alta_promedio_listas_pc1 + alta_errores_pc1
relleno_neg_alta = alta_promedio_listas_pc1 - alta_errores_pc1


plt.plot(tiempo, alta_promedio_listas_pc1.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas_pc1 + baja_errores_pc1
relleno_neg_baja = baja_promedio_listas_pc1 - baja_errores_pc1

plt.plot(tiempo, baja_promedio_listas_pc1.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")


plt.ylabel('Principal Component 1')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.legend()
plt.show()

relleno_pos_alta = alta_promedio_listas_pc2 + alta_errores_pc2
relleno_neg_alta = alta_promedio_listas_pc2 - alta_errores_pc2

plt.figure()
plt.plot(tiempo, alta_promedio_listas_pc2.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas_pc2 + baja_errores_pc2
relleno_neg_baja = baja_promedio_listas_pc2 - baja_errores_pc2

plt.plot(tiempo, baja_promedio_listas_pc2.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")


plt.ylabel('Principal Component 2')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.legend()
plt.show()


#%% Test de Wilcoxon entre los dosis alta y dosis baja

from joblib import Parallel, delayed

def test(baja, alta):
    
    pvalor = []
    
    # Concateno los sujetos en un dataframe
    # df_baja = pd.concat(baja, axis=1)
    # df_alta = pd.concat(alta, axis=1)
    df_baja = baja
    df_alta = alta
    # Paso a arrays para poder hacer las stats
    arr_baja = df_baja.values
    arr_alta = df_alta.values
    
    # Preallocate the result array
    pvalor = np.empty(arr_baja.shape[0])
    
    def compute_pvalue(j):
       return stats.wilcoxon(arr_baja[j], arr_alta[j]).pvalue
   
    pvalor = Parallel(n_jobs=-1)(delayed(compute_pvalue)(j) for j in range(arr_baja.shape[0]))
    
    return pvalor
    
# pvalores_pc1 = test(promedio_pc1_baja, promedio_pc1_alta)
# pvalores_pc2 = test(promedio_pc2_baja, promedio_pc2_alta)

pvalores_pc1 = test(pc1_baja, pc1_alta)
pvalores_pc2 = test(pc2_baja, pc2_alta)

#%% Calculo de pvalores que pasan el umbral para después plotear, ya con fdr

#####################################################################################################################################
# Esta celda funciona asi, para calcular pvalores > 0.05, hay que comentar lo de fdr y los indices modificados por fdr
# Dsps descomentar indices menores al 0.05 y graficar en la siguiente celda. Con fdr es al reves y graficar con la siguiente celda.
####################################################################################################################################

arr_pvalores_pc1 = np.array(pvalores_pc1)
arr_pvalores_pc2 = np.array(pvalores_pc2)


# Calculo con fdr los pvalores nuevos
rtas_pc1, pval_fdr = st.fdrcorrection(arr_pvalores_pc1)

# Indices de los pvalores que son menores que el 0.05
# indices_p_pos_pc1 = (np.where(arr_pvalores_pc1 < 0.05))[0]
indices_p_pos_pc2 = (np.where(arr_pvalores_pc2 < 0.05))[0]

# Indices de los pvalores ya modificados por fdr que dan menores a 0.05
indices_p_pos_pc1 = (np.where(rtas_pc1 == True))[0]

# Divido en regiones de los valores que cumplen para poder dibujar eso en el grafico
regiones_pc1 = np.split(indices_p_pos_pc1, np.where(np.diff(indices_p_pos_pc1) != 1)[0] + 1)
regiones_pc2 = np.split(indices_p_pos_pc2, np.where(np.diff(indices_p_pos_pc2) != 1)[0] + 1)

#%% AGREGADO: Ploteo con fill between Y Pvalores

frec_sampleo = 1/4 # Espacio de 4 segundos para que dure 1200 segundos, pero no tiene un sentido fisico

size = 14

# plt.close('all')

plt.figure()
tiempo = np.arange(0, 1200, 4)

relleno_pos_alta = alta_promedio_listas_pc1 + alta_errores_pc1
relleno_neg_alta = alta_promedio_listas_pc1 - alta_errores_pc1


plt.plot(tiempo, alta_promedio_listas_pc1.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas_pc1 + baja_errores_pc1
relleno_neg_baja = baja_promedio_listas_pc1 - baja_errores_pc1

plt.plot(tiempo, baja_promedio_listas_pc1.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")

# Acá agrego el background de color gris
if len(regiones_pc1[0]) != 0: # en este caso el fdr me saca todos los pvalores
    for i, region in enumerate(regiones_pc1):
        start = region[0] + 1
        end = region[-1] + 1
        plt.axvspan(start/frec_sampleo, end/frec_sampleo, color='#C6C6C6', alpha=0.3)
        # mid_point = (start + end) / 2
        # plt.text(mid_point, plt.ylim()[1], f'Region {i+1}', horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

plt.ylabel('Principal Component 1', fontsize = size)
plt.xlabel('Time (s)', fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.tight_layout()
plt.legend(fontsize = size, loc = 'upper right')
plt.show()


plt.figure()

relleno_pos_alta = alta_promedio_listas_pc2 + alta_errores_pc2
relleno_neg_alta = alta_promedio_listas_pc2 - alta_errores_pc2


plt.plot(tiempo, alta_promedio_listas_pc2.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas_pc2 + baja_errores_pc2
relleno_neg_baja = baja_promedio_listas_pc2 - baja_errores_pc2

plt.plot(tiempo, baja_promedio_listas_pc2.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")

# Acá agrego el background de color gris
if len(regiones_pc2[0]) != 0: # en este caso el fdr me saca todos los pvalores
    for i, region in enumerate(regiones_pc2):
        start = region[0] + 1
        end = region[-1] + 1
        plt.axvspan(start/frec_sampleo, end/frec_sampleo, color='#C6C6C6', alpha=0.3)
        # mid_point = (start + end) / 2
        # plt.text(mid_point, plt.ylim()[1], f'Region {i+1}', horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

plt.ylabel('Principal Component 2', fontsize = size)
plt.xlabel('Time (s)', fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.tight_layout()
plt.legend(fontsize = size)
plt.show()

#%%



            
