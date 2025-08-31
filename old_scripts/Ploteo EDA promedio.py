#%%
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

ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

#%%

def eda(experimento, carpeta, fname, numero = None, df_rs = None, malo = None):
    
    fname = os.path.join(ROOT_DATA, experimento, carpeta, fname)
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    ################  OJO ACA:  ################################################### 
    # Estas son las longitudes de todos mis datos, me quedé con la mínima, lo hice así pq lo necesitaba hacer una vez
    df = df_data[:293434] #min(326005, 324840, 320150, 313305, 327750, 330335, 295405, 340695, 293435, 329035)
    # Nota buena y consistente: EDA y HR tienen la misma longitud minima
    ###############################################################################
    
    
    
    if 'GSR' in df.columns:
        
        edap = df['GSR']
        
        ######### Tengo que usar la frec de sampleo de info ##########
        
        interv = []
        for i in range(0, len(df['time'])-1):
            t = df['time'][i+1] - df['time'][i]
            interv.append(t)
            
        dt = np.mean(interv)
        
        ##############################################################
        
        if df_rs is None:
            df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt) #neurokit method
            
            tiempo = None
            # print('\n Entre \n')
            # plt.figure(numero)
            # plt.plot(df['time'], df_eda['EDA_Tonic'], label = f'{carpeta}')
            # plt.xlabel('Tiempo (s)')
            # plt.legend()

            
        else:
            df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt) #neurokit method
            df_eda['EDA_Tonic'] = df_eda['EDA_Tonic'] - np.mean(df_rs['EDA_Tonic'])
            
            plt.figure(numero)
            plt.plot(df['time'], df_eda['EDA_Tonic'], label = f'{carpeta}')
            plt.xlabel('Tiempo (s)')
            plt.legend()
            plt.tight_layout()
            
            tiempo = df['time']
            
        return df_eda, info_eda, tiempo
        
    else:
        malo = [1]
        
        return malo, 0
       
#%%

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

columnas = ['Dosis_Sesion_1', 'Dosis_Sesion_2'] # definimos los nombres de las columnas
indices = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']

dosis = pd.DataFrame(dosis, columns=columnas, index = indices)

#%% Carpetas y experimentos

# exps = ['DMT_1', 'Reposo_1', 'DMT_2', 'Reposo_2']
# carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']

## Ploteo por ahora solo las que tengo info
carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']

#### Sujetos que tiene señal de EDA muerta:
#### S08 (DMT_2), S10 (DMT_2), S11 (DMT_2), S12 (DMT_2, DMT_1 esta bien igual), S15 (DMT_2)

#%% Iterando sobre todos los sujetos de las carpetas

plt.close('all')  

promedio_dmt_alta = []
promedio_dmt_baja = []
promedio_tiempo_alto = []
promedio_tiempo_bajo = []

for carpet in carpetas:
        
        # Si la dosis es baja en la Sesion 1
    if dosis['Dosis_Sesion_1'][carpet] == 'Baja':
        
        j = 2
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs, none = eda(exp, carpet, nombre, j + 1)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs, none = eda(exp, carpet, nombre, j)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo = eda(exp, carpet, nombre, i + 1, df_eda_baja_rs)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto = eda(exp, carpet, nombre, i, df_eda_alta_rs)
        
        
    # La dosis es alta en la sesion 1
    else:
        
        j = 2
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs, none = eda(exp, carpet, nombre, j)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs, none = eda(exp, carpet, nombre , j + 1)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto = eda(exp, carpet, nombre, i, df_eda_alta_rs)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo = eda(exp, carpet, nombre, i+1, df_eda_baja_rs)
        
     
    ## Acá agrego el hecho de que guarde los datos de cada uno en listas para después promediar cada punto
    if len(df_eda_alta_dmt) != 1 and len(df_eda_alta_rs) != 1:
        
        prom_dmt = df_eda_alta_dmt['EDA_Tonic']
        prom_time = tiempo_alto
        promedio_dmt_alta.append(prom_dmt)
        promedio_tiempo_alto.append(prom_time)
        
        
    else:
        print(f'Sujeto {carpet} tiene datos (con dosis DMT Alta) de EDA indescifrables')
        
    if len(df_eda_baja_dmt) != 1 and len(df_eda_baja_rs) != 1:
        
        prom_dmt = df_eda_baja_dmt['EDA_Tonic']
        prom_time = tiempo_bajo
        promedio_dmt_baja.append(prom_dmt)
        promedio_tiempo_bajo.append(prom_time)
        

    # if df_eda_alta_dmt is not None:
    #     del df_eda_alta_dmt, df_eda_alta_rs, 
    #     df_eda_baja_dmt, df_eda_baja_rs, 
    #     info_eda_alta_dmt, info_eda_alta_rs, 
    #     info_eda_baja_dmt, info_eda_baja_r




# SI NO SE QUIERE GUARDAR: Saltear estas dos celdas y descomentar las que digan "promedio_dmt_..." y hagan las transposicion, se usa en el promediado y en pvalores
#%% Guardado de archivos de SCL para después promediar o comparar con PCA más facil

medida = 'SCL'
preproc_dir = os.path.join(ROOT_DATA, 'Preprocesado', medida)
os.makedirs(preproc_dir, exist_ok=True)

#%%
# Lista de datos procesados pasada a dataframe y guardada en csv
df_alta_guardar = pd.DataFrame(promedio_dmt_alta).T
df_alta_guardar.columns = carpetas
csv = f'{medida}_dmt_alta.csv'
df_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_baja_guardar = pd.DataFrame(promedio_dmt_baja).T
df_baja_guardar.columns = carpetas
csv = f'{medida}_dmt_baja.csv'
df_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_tiempo_alta_guardar = pd.DataFrame(promedio_tiempo_alto).T
df_tiempo_alta_guardar.columns = carpetas
csv = f'{medida}_tiempo_dmt_alta.csv'
df_tiempo_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

df_tiempo_baja_guardar = pd.DataFrame(promedio_tiempo_bajo).T
df_tiempo_baja_guardar.columns = carpetas
csv = f'{medida}_tiempo_dmt_baja.csv'
df_tiempo_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

#%% Leer los archivos preprocesados

scl_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
scl_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
scl_alta_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_alta.csv'))
scl_baja_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_baja.csv'))

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

# Descomentar si no se quiere guardar el preprocesado############
# alta_promedio_listas, alta_errores = calculate_means_and_stdevs(promedio_dmt_alta)
# baja_promedio_listas, baja_errores = calculate_means_and_stdevs(promedio_dmt_baja)
# tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(promedio_tiempo_alto)
# tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(promedio_tiempo_bajo)
#################################################################

alta_promedio_listas, alta_errores = calculate_means_and_stdevs(scl_alta)
baja_promedio_listas, baja_errores = calculate_means_and_stdevs(scl_baja)
tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(scl_alta_tiempo)
tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(scl_baja_tiempo)



#%% Ploteo con fill between y grafico el promedio

# plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores
relleno_neg_alta = alta_promedio_listas - alta_errores


plt.plot(tiempo_alto_listas.values, alta_promedio_listas.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas.values, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores
relleno_neg_baja = baja_promedio_listas - baja_errores

plt.plot(tiempo_bajo_listas.values, baja_promedio_listas.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas.values, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")

plt.ylabel('EDA Tonica restando baseline ($\mu$S)')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.legend()
plt.show()

#%% Test de Wilcoxon entre los dosis alta y dosis baja

from joblib import Parallel, delayed

def test(baja, alta):
    
    pvalor = []
    
    df_alta = alta
    df_baja = baja
    
    # DESCOMENTAR SI SE USA PROMEDIO_DMT_ALTA y BAJA
    # Concateno los sujetos en un dataframe
    # df_baja = pd.concat(baja, axis=1)
    # df_alta = pd.concat(alta, axis=1)
    
    # Paso a arrays para poder hacer las stats
    arr_baja = df_baja.values
    arr_alta = df_alta.values
    
    # Preallocate the result array
    pvalor = np.empty(arr_baja.shape[0])
    
    def compute_pvalue(j):
       return stats.wilcoxon(arr_baja[j], arr_alta[j]).pvalue
   
    pvalor = Parallel(n_jobs=-1)(delayed(compute_pvalue)(j) for j in range(arr_baja.shape[0]))
    
    return pvalor
    
# pvalores = test(promedio_dmt_baja, promedio_dmt_alta)
pvalores = test(scl_baja, scl_alta)


#%% Chequeo de pvalores mayores y menores

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
    
#%% Ploteo de pvalores con respecto al umbral

plt.figure(2)
plt.plot(tiempo_bajo_listas.values,pvalores,'o', label = 'P-valor', color = '#A366A6')
plt.plot(tiempo_bajo_listas.values, 0.05*np.ones(len(tiempo_bajo_listas.values)),'--', label = 'Umbral', color = '#CCA60D')
plt.xlabel('Tiempo(s)')
plt.legend()
plt.tight_layout()

#%% Calculo de pvalores que pasan el umbral para después plotear, ya con fdr

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

#%% AGREGADO: Ploteo con fill between Y Pvalores

size = 14

plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores
relleno_neg_alta = alta_promedio_listas - alta_errores


plt.plot(tiempo_alto_listas.values, alta_promedio_listas.values, label = 'High Dose', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas.values, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.4, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores
relleno_neg_baja = baja_promedio_listas - baja_errores

plt.plot(tiempo_bajo_listas.values, baja_promedio_listas.values, label = 'Low Dose', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas.values, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.4, color = "#FDD835")

# Acá agrego el background de color gris
if len(regiones[0]) != 0: # en este caso el fdr me saca todos los pvalores
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

#%% Ploteo de pvalores con fdr para mostrar qué valores dan

plt.figure(2)
plt.plot(tiempo_bajo_listas.values,pval_fdr,'o', label = 'P-valor', color = '#A366A6')
plt.plot(tiempo_bajo_listas.values, 0.05*np.ones(len(tiempo_bajo_listas.values)),'--', label = 'Umbral', color = '#CCA60D')
plt.xlabel('Tiempo(s)')
plt.legend()
plt.tight_layout()










# %%
