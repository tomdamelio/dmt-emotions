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
directorio = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/resampled"

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

#%% Grafico todos los sujetos para cada dibujo (son 44, ojo)
## ESTA FUNCIONANDO MAL, DESPUES LO VOY A VER SI ES IMPORTANTE


plt.close('all')

promedio_alta_please = {}
promedio_alta_unplease = {}
promedio_alta_emi = {}
promedio_alta_eli = {}
promedio_alta_ci = {}
promedio_alta_aud = {}
promedio_alta_int = {}
promedio_alta_bli = {}
promedio_alta_anx = {}
promedio_alta_ent = {}
promedio_alta_self = {}
promedio_alta_disemb = {}
promedio_alta_sal = {}
promedio_alta_temp = {}
promedio_alta_gi = {}

promedio_baja_please = {}
promedio_baja_unplease = {}
promedio_baja_emi = {}
promedio_baja_eli = {}
promedio_baja_ci = {}
promedio_baja_aud = {}
promedio_baja_int = {}
promedio_baja_bli = {}
promedio_baja_anx = {}
promedio_baja_ent = {}
promedio_baja_self = {}
promedio_baja_disemb = {}
promedio_baja_sal = {}
promedio_baja_temp = {}
promedio_baja_gi = {}

sujetos = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's13', 's15', 's16', 's17', 's18', 's19', 's20']

promedio_alta = [promedio_alta_anx, promedio_alta_aud, promedio_alta_bli, promedio_alta_ci, promedio_alta_disemb, promedio_alta_eli, promedio_alta_emi, promedio_alta_ent, promedio_alta_gi, promedio_alta_int, promedio_alta_please, promedio_alta_unplease]
promedio_baja = [promedio_baja_anx, promedio_baja_aud, promedio_baja_bli, promedio_baja_ci, promedio_baja_disemb, promedio_baja_eli, promedio_baja_emi, promedio_baja_ent, promedio_baja_gi, promedio_baja_int, promedio_baja_please, promedio_baja_unplease]

promedios = [promedio_alta, promedio_baja]

for promedio in promedios:
    for dim in promedio:
        for sujeto in sujetos:
            dim[sujeto] = []

for fname in archivos_ordenados[:]:
    
    mat = scipy.io.loadmat(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/resampled/{fname}')

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
    
    if (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Baja': 
        
    
        for columna in df_dibujo.columns.values:
            
            if columna == 'Pleasantness':
                
                data = promedio_baja_please
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                for key, value in data.items():
                    print(f"Key: {key}, Value: {value}")
                
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_please = pd.DataFrame(output_dict)
                
                print(promedio_alta_please)
            elif columna == 'Unpleasantness':
                
                data = promedio_baja_unplease
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_unplease = pd.DataFrame(output_dict)
             
            elif columna == 'Emotional_Intensity':
                
                data = promedio_baja_emi
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_emi = pd.DataFrame(output_dict)
            
            elif columna == 'Elementary_Imagery':
                
                data = promedio_baja_eli
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_eli = pd.DataFrame(output_dict)
            
            elif columna == 'Complex_Imagery':
                
                data = promedio_baja_ci
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_ci = pd.DataFrame(output_dict)
            
            elif columna == 'Auditory':
                
                data = promedio_baja_aud
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_aud = pd.DataFrame(output_dict)
            
            elif columna == 'Interoception':
                
                data = promedio_baja_int
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_int = pd.DataFrame(output_dict)
            
            elif columna == 'Bliss':
                
                data = promedio_baja_bli
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_bli = pd.DataFrame(output_dict)
            
            elif columna == 'Anxiety':
                
                data = promedio_baja_anx
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_anx = pd.DataFrame(output_dict)
            
            elif columna == 'Entity':
                
                data = promedio_baja_ent
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_ent = pd.DataFrame(output_dict)
            
            elif columna == 'Selfhood':
                
                data = promedio_baja_self
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_self = pd.DataFrame(output_dict)
            
            elif columna == 'Disembodiment':
                
                data = promedio_baja_disemb
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_disemb = pd.DataFrame(output_dict)
            
            elif columna == 'Salience':
                
                data = promedio_baja_sal
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_sal = pd.DataFrame(output_dict)
            
            elif columna == 'Temporality':
                
                data = promedio_baja_temp
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_temp = pd.DataFrame(output_dict)
            
            else:
                
                data = promedio_baja_gi
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis baja
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_baja_gi = pd.DataFrame(output_dict)

            
    elif (experimento == 'DMT_1' or experimento == 'DMT_2') and dosis[experimento][carpeta] == 'Alta':
                
        for columna in df_dibujo.columns.values:
        
            
            if columna == 'Pleasantness':
                
                data = promedio_alta_please
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                
                # promedio_alta_please = pd.DataFrame(output_dict)
                print(output_dict)

            elif columna == 'Unpleasantness':
                
                data = promedio_alta_unplease
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_unplease = pd.DataFrame(output_dict)
             
            elif columna == 'Emotional_Intensity':
                
                data = promedio_alta_emi
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_emi = pd.DataFrame(output_dict)
            
            elif columna == 'Elementary_Imagery':
                
                data = promedio_alta_eli
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_eli = pd.DataFrame(output_dict)
            
            elif columna == 'Complex_Imagery':
                
                data = promedio_alta_ci
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_ci = pd.DataFrame(output_dict)
            
            elif columna == 'Auditory':
                
                data = promedio_alta_aud
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_aud = pd.DataFrame(output_dict)
            
            elif columna == 'Interoception':
                
                data = promedio_alta_int
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_int = pd.DataFrame(output_dict)
            
            elif columna == 'Bliss':
                
                data = promedio_alta_bli
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_bli = pd.DataFrame(output_dict)
            
            elif columna == 'Anxiety':
                
                data = promedio_alta_anx
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_anx = pd.DataFrame(output_dict)
            
            elif columna == 'Entity':
                
                data = promedio_alta_ent
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_ent = pd.DataFrame(output_dict)
            
            elif columna == 'Selfhood':
                
                data = promedio_alta_self
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_self = pd.DataFrame(output_dict)
            
            elif columna == 'Disembodiment':
                
                data = promedio_alta_disemb
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_disemb = pd.DataFrame(output_dict)
            
            elif columna == 'Salience':
                
                data = promedio_alta_sal
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_sal = pd.DataFrame(output_dict)
            
            elif columna == 'Temporality':
                
                data = promedio_alta_temp
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_temp = pd.DataFrame(output_dict)
            
            else:
                
                data = promedio_alta_gi
                
                #Esto me hace un DataFrame con todos los sujetos metidos en pleasantness que tienen dosis alta
                data[carpeta].append(list(df_dibujo[columna].values))
                #Chiche pq tengo el diccionario raro
                output_dict = {}  # Initialize an empty dictionary

                for key, value in data.items():
                    if value:  # Check if value is not empty to avoid index error
                        output_dict[key] = value[0]
                promedio_alta_gi = pd.DataFrame(output_dict)
            

    # Si son de reposo van a entrar acá, es necesario el doble if para evitar que uno de DMT entre aca
# Reposo no le pido nada    
