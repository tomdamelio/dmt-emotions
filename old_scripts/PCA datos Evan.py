# -*- coding: utf-8 -*-

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


#%% Lee el archivo mat y lo organiza en un dataframe

mat = scipy.io.loadmat("C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/resampled/s01_DMT_Session2_DMT.mat")
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 'Elementary_Imagery', 'Complex_Imagery',
    'Auditory', 'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 'Disembodiment', 'Salience',
    'Temporality', 'General_Intensity']

df_dibujo = pd.DataFrame(mat['dimensions'], columns = dimensiones)

#%% Creo el tiempo con el sampleo de Evan

tiempo = np.arange(0, 1200, 4)

#%% Ploteo cada uno de los dibujos

plt.close('all')

for columna in df_dibujo.columns.values:
    plt.plot(tiempo, df_dibujo[columna].values, label = columna)
    plt.xlabel('Tiempo (s)')
    plt.tight_layout()
    plt.show()
plt.legend()

#%% Les calculo el PCA o lo entreno como dicen

indices = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15']

# Componentes a quedarme, importante aclararlo
cantidad_pc = len(indices) - 12 # son 15, si resto por 12 me quedo con las primeras 3 componentes

# Creo un pipeline donde le hace el standard Scaler (que pone cada columna independiente a un z score)
# y después hace el pca, le aclaro que me voy a quedar con la cantidad de PC aclarado
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components = cantidad_pc))
pca_pipe.fit(df_dibujo)

# Se extrae el modelo entrenado del pipeline
modelo_pca = pca_pipe.named_steps['pca']

#%% Convierto el arrar de modelo_pca a dataframe y le miro las componentes

componentes = modelo_pca.components_

df_pca = pd.DataFrame(data = componentes, columns = df_dibujo.columns)

#%% Printeo las filas para ver cuales tienen mas peso y armo un heat map para comparar

print(df_pca.loc[0])
print(df_pca.loc[1])

#################################################################################################################################
#  Nota importante:                                                                                                             # 
#       Le puse modulo a las componentes para que en el gráfico sea más visible cuáles son las dimensiones más influyentes      #
#################################################################################################################################

# Heatmap componentes, limitado por la cantidad de componentes que quise calcular
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
plt.imshow(abs(componentes.T), cmap='viridis', aspect='auto')
plt.yticks(range(len(df_dibujo.columns)), df_dibujo.columns)
plt.xticks(range(cantidad_pc), np.arange(cantidad_pc) + 1)
plt.grid(False)
plt.colorbar()


#%%################################################################

# Los proximos 3 bloques plotean las dimensiones normales.        #

# El primero plotea todas contra todas para ver por arriba.       #

# El segundo te hace acordar los titulos que tenías para que      #
# en el tercero elijas el gráfico particular que querés.          #
###################################################################


#%% Ploteo la correlación de todas las dimensiones entre sí

# Crear la figura y los ejes
fig, axes = plt.subplots(len(dimensiones), len(dimensiones), figsize=(12, 12))

# Dibujar la matriz de dispersión
for i, dimension1 in enumerate(dimensiones):
    for j, dimension2 in enumerate(dimensiones):
        if i != j:
            # Dibujar gráficos de dispersión fuera de la diagonal
            sns.scatterplot(data = df_dibujo, x = dimension2, y=dimension1, ax=axes[i, j], legend=False)
        else:
            # Dejar los espacios en blanco en la diagonal
            axes[i, j].set_visible(False)
            
         # Mostrar solo un nombre de eje Y por fila
        if j == 0 and i != j:
            axes[i, j].set_ylabel(dimension1)
        else:
            axes[i, j].set_ylabel('')

        # Mostrar solo un nombre de eje X por columna
        if i == len(dimensiones) - 1:
            axes[i, j].set_xlabel(dimension2)
        else:
            axes[i, j].set_xlabel('')



#%% Este crea cada una por separado
# plt.close('all')
df_dibujo.columns

#%%
dimension1 = 'Auditory'
dimension2 = 'Elementary_Imagery'

for nombre in df_dibujo.columns:
    plt.figure()
    # Dibujar gráfico de dispersión
    sns.scatterplot(df_dibujo, x = dimension2, y=nombre)
    plt.xlabel(dimension1)
    plt.ylabel(nombre)
    plt.show()


#%%################################################################################

# Ploteo de Componentes ahora, todos contra todos y después alguna contra otra    #

###################################################################################
#%% Ploteo la correlación de todas las componentes entre sí

# Crear la figura y los ejes
fig, axes = plt.subplots(cantidad_pc, cantidad_pc, figsize=(12, 12))

transpuesto = df_pca.transpose()

# Dibujar la matriz de dispersión
for i in range(cantidad_pc):
    for j in range(cantidad_pc):
        if i != j:
            # Dibujar gráficos de dispersión fuera de la diagonal
            sns.scatterplot(data = transpuesto, x = j, y = i, ax=axes[i, j], legend=False)
        else:
            # Dejar los espacios en blanco en la diagonal
            axes[i, j].set_visible(False)

# Ajustar la presentación
plt.tight_layout()
plt.show()

#%% Este crea cada una por separado
plt.close('all')

# Principal Component que querés
componente1 = 1
componente2 = 2

#Acá las defino para el dataframe que está corrido 1
componente1 = componente1 - 1
componente2 = componente2 - 1

# Dibujar gráfico de dispersión
plt.plot(df_pca.loc[componente1], df_pca.loc[componente2], 'o')
plt.xlabel(f'PC{componente1 + 1}')
plt.ylabel(f'PC{componente2 + 1}')
plt.show()







