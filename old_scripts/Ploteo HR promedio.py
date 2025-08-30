# -*- coding: utf-8 -*-
#%%
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

def ecg(experimento, carpeta, fname, numero = None, df_rs = None, malo = None):
    
    fname = os.path.join(ROOT_DATA, experimento, carpeta, fname)
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    ################  OJO ACA:  ################################################### 
    # Estas son las longitudes de todos mis datos, me quedé con la mínima, lo hice así pq lo necesitaba hacer una vez
    df = df_data[:293434] #min(326005, 320150, 313305, 315345, 331820, 327750, 339240, 318850, 312215, 330335, 323270, 295405, 340695, 293435, 329035, 350410, 343300, 333035)
    ###############################################################################
    
    if 'ECG' in df.columns:
        
        ecgp = df['ECG']
        
    elif '33' in df.columns:
        
        ecgp = df['33'] #ECG posta
        
        ######### Tengo que usar la frec de sampleo de info ##########
        
    interv = []
    for i in range(0, len(df['time'])-1):
        t = df['time'][i+1] - df['time'][i]
        interv.append(t)
        
    dt = np.mean(interv)
    
    ##############################################################
    
    if df_rs is None:
        df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method
        
        tiempo = None
        
    else:
        df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method
        df_ecg['ECG_Rate'] = df_ecg['ECG_Rate'] - np.mean(df_rs['ECG_Rate'])
        
        plt.figure(numero)
        plt.plot(df['time'], df_ecg['ECG_Rate'], label = f'{carpeta}')
        plt.xlabel('Tiempo (s)')
        plt.legend()
        plt.tight_layout()
        
        tiempo = df['time']
        
    return df_ecg, info_ecg, tiempo
    
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
carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']


#%% Iteración en todas las carpetas

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
        df_ecg_baja_rs, info_ecg_baja_rs, none = ecg(exp, carpet, nombre, j + 1)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_ecg_alta_rs, info_ecg_alta_rs, none = ecg(exp, carpet, nombre, j)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_ecg_baja_dmt, info_ecg_baja_dmt, tiempo_bajo = ecg(exp, carpet, nombre, i + 1, df_ecg_baja_rs)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_ecg_alta_dmt, info_ecg_alta_dmt, tiempo_alto = ecg(exp, carpet, nombre, i, df_ecg_alta_rs)
        
        # print(min(len(df_ecg_alta_dmt), len(df_ecg_baja_dmt)))
        
    # La dosis es alta en la sesion 1
    else:
        
        j = 2
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_ecg_alta_rs, info_ecg_alta_rs, none = ecg(exp, carpet, nombre, j)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_ecg_baja_rs, info_ecg_baja_rs, none = ecg(exp, carpet, nombre , j + 1)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_ecg_alta_dmt, info_ecg_alta_dmt, tiempo_alto = ecg(exp, carpet, nombre, i, df_ecg_alta_rs)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_ecg_baja_dmt, info_ecg_baja_dmt, tiempo_bajo = ecg(exp, carpet, nombre, i+1, df_ecg_baja_rs)
        
        # print(min(len(df_ecg_alta_dmt), len(df_ecg_baja_dmt)))
     
    ## Acá agrego el hecho de que guarde los datos de cada uno en listas para después promediar cada punto
    if len(df_ecg_alta_dmt) != 1 and len(df_ecg_alta_rs) != 1:
        
        prom_dmt = df_ecg_alta_dmt['ECG_Rate']
        prom_time = tiempo_alto
        promedio_dmt_alta.append(prom_dmt)
        promedio_tiempo_alto.append(prom_time)
        
        
    else:
        print(f'Sujeto {carpet} tiene datos (con dosis DMT Alta) de ECG indescifrables')
        
    if len(df_ecg_baja_dmt) != 1 and len(df_ecg_baja_rs) != 1:
        
        prom_dmt = df_ecg_baja_dmt['ECG_Rate']
        prom_time = tiempo_bajo
        promedio_dmt_baja.append(prom_dmt)
        promedio_tiempo_bajo.append(prom_time)
        
    # if len(df_ecg_baja_dmt) != 1 and len(df_ecg_alta_dmt) != 1:  
    #     ######## Estoy pensando si hace falta que le ponga equal_var o alternative = "greater" ###############
    #     ttest = stats.ttest_ind(df_ecg_alta_dmt['ECG_Rate'].values, df_ecg_baja_dmt['ECG_Rate'].values, alternative = 'greater')
    #     pvalores.append(ttest.pvalue)
    #     t_stat.append(ttest.statistic)
    
        
# SI NO SE QUIERE GUARDAR: Saltear estas dos celdas y descomentar las que digan "promedio_dmt_..." y hagan las transposicion, se usa en el promediado y en pvalores
#%% Guardado de archivos de HR para después promediar o comparar con PCA más facil

medida = 'HR'
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

hr_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
hr_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
hr_alta_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_alta.csv'))
hr_baja_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_baja.csv'))



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

alta_promedio_listas, alta_errores = calculate_means_and_stdevs(hr_alta)
baja_promedio_listas, baja_errores = calculate_means_and_stdevs(hr_baja)
tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(hr_alta_tiempo)
tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(hr_baja_tiempo)


#%% Test de Wilcoxon entre los dosis alta y dosis baja

from joblib import Parallel, delayed

def test(baja, alta):
    
    pvalor = []
    
    df_alta = alta
    df_baja = baja
    
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
pvalores = test(hr_baja, hr_alta)

#%% Ploteo con fill between

plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores
relleno_neg_alta = alta_promedio_listas - alta_errores


plt.plot(tiempo_alto_listas.values, alta_promedio_listas.values, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas.values, relleno_neg_alta.values, relleno_pos_alta.values, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores
relleno_neg_baja = baja_promedio_listas - baja_errores

plt.plot(tiempo_bajo_listas.values, baja_promedio_listas.values, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas.values, relleno_neg_baja.values, relleno_pos_baja.values, alpha = 0.2, color = "#FDD835")

plt.ylabel('$\Delta$HR (bpm)')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.legend()
plt.show()


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

#%% Ploteo de Pvalores contra umbral

plt.plot(tiempo_bajo_listas.values, pvalores,'o')
plt.plot(tiempo_bajo_listas.values, 0.05*np.ones(len(tiempo_bajo_listas.values)),'--')

#%% Calculo de pvalores que pasan el umbral para después plotear

arr_pvalores = np.array(pvalores)

# Voy a agregar hacerle el fdr para mejorar el pvalor con un criterio más exacto
rtas, pval_fdr = st.fdrcorrection(arr_pvalores)

# Indices de los pvalores que son menores que el 0.05
# indices_p_pos = (np.where(arr_pvalores < 0.05))[0]

# Indices de los pvalores ya modificados por fdr que dan menores a 0.05
indices_p_pos = (np.where(rtas == True))[0]

regiones = np.split(indices_p_pos, np.where(np.diff(indices_p_pos) != 1)[0] + 1)

#frec_sampleo = int(info_ecg_baja_dmt['sampling_rate'])
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
for i, region in enumerate(regiones):
    start = region[0] + 1
    end = region[-1] + 1
    plt.axvspan(start/frec_sampleo, end/frec_sampleo, color='#C6C6C6', alpha=0.4)

plt.ylabel('HR substracting baseline (bpm)', fontsize = size)
plt.xlabel('Time (s)', fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.tight_layout()
plt.legend(loc ='upper right', fontsize = size)
plt.show()




    







