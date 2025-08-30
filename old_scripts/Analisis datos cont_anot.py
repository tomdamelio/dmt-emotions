# -*- coding: utf-8 -*-

import pandas as pd 
import ast
import numpy as np
import matplotlib.pyplot as plt


#%%

filename = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Datos/sub-24_ses-A_task-Experiment_VR_non_immersive_beh - sub-24_ses-A_task-Experiment_VR_non_immersive_beh.csv"

info = pd.read_csv(filename, delimiter = ",", header = 0)

#%%
def distincion_datos(numero, datos):

    cont_anot = ast.literal_eval(datos['continuous_annotation'][numero])
    lumin = ast.literal_eval(datos['continuous_annotation_luminance'][numero])
    
    intensidad = []
    tiempo_int = []
    
    luminance = []
    tiempo_lum = []
    
    for i in range(len(cont_anot)):
        inten = cont_anot[i][0]
        intensidad.append(inten)
        
        tiempo = cont_anot[i][1]
        tiempo_int.append(tiempo)
        
    for j in range(len(lumin)):
        lum = lumin[j][0]
        luminance.append(lum)
        
        tiempo = lumin[j][1]
        tiempo_lum.append(tiempo)
        
    # plt.figure(numero)
    # plt.plot(tiempo_lum,luminance, 'o')
    
    # plt.figure(numero + 1)
    # plt.plot(tiempo_int,intensidad, 'o')    
    
    resta_ini = tiempo_lum[0] - tiempo_int[0]
    print(resta_ini)
    
    resta_fin = tiempo_lum[len(tiempo_lum)-1] - tiempo_int[len(tiempo_int)-2]
    print(resta_fin)
    
    largo = len(tiempo_lum) - len(tiempo_int)
    
    
    if largo > 0:
        tiempo_lum = tiempo_lum[:-largo]
        print("Lista de luminancia tiene", largo, "puntos más")
        
    elif largo < 0:
        tiempo_int = tiempo_int[:largo]
        print("Lista de afectivo tiene", largo, "puntos más")
    
    else:
        print("Tienen la misma longitud las listas")
    
    
    diferencia = [e1 - e2 for e1, e2 in zip(tiempo_lum,tiempo_int)]
    diferencia_3 = [round(e1 - e2,3) for e1, e2 in zip(tiempo_lum,tiempo_int)]
    
    x = np.arange(1, len(tiempo_lum) + 1,1)
    
    plt.figure(numero + 2)
    plt.plot(x, diferencia, label = 'Exacto')
    plt.plot(x, diferencia_3, label = '3 decimales')
    plt.ylim(min(diferencia)-0.002,max(diferencia)+0.002)
    plt.legend()

# return cont_anot, lumin

#%%
plt.close('all')

for h in range(0,3):
    distincion_datos(h, info)
    
    
    
