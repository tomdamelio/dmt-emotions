# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from warnings import warn
import os

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
from scipy import interpolate

#%% Defino sliding window con desviacion estandar

def sliding_window(data_normal, signal, k, step):
    
    window_count = []
    tiempos = []
    
    n = len(signal)
    
    if n <= k:
        print('Array de picos más corto que la ventana propuesta')
        return -1
    
    window = signal[:k]
    window_desv = np.std(window['RSP_Clean'])
    window_count.append(window_desv)
    
    window = data_normal[:k]
    tiempo = np.mean(window['time'])
    tiempos.append(tiempo)

    
    step_inic = step
    
    while step + step_inic < len(signal):
        
        window = signal[step:step + k]
        window_desv = np.std(window['RSP_Clean'])
        window_count.append(window_desv)
        
        window = data_normal[step:step + k]
        tiempo = np.mean(window['time'])
        tiempos.append(tiempo)
        
        step += step_inic
        
    return tiempos, window_count



#%%

# Largo de la ventana para la sliding window, este caso va a ser de 20 segundos
tiempo_sliding_window = 100

ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

def resp(experimento, carpeta, fname, numero, tiempo_w = tiempo_sliding_window, df_rs = None, media_rs = None ,malo = None):
    
    fname = os.path.join(ROOT_DATA, experimento, carpeta, fname)
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    ################  OJO ACA:  ################################################### 
    # Estas son las longitudes de todos mis datos, me quedé con la mínima, lo hice así pq lo necesitaba hacer una vez
    df = df_data[:293434] #min(326005, 324840, 320150, 313305, 327750, 330335, 295405, 340695, 293435, 329035)
    # Nota buena y consistente: EDA y HR tienen la misma longitud minima, espero que resp también
    ###############################################################################
    
    
    
    if 'RESP' in df.columns:
        
        resp = df['RESP']
        
        ######### Tengo que usar la frec de sampleo de info ##########
        
        interv = []
        for i in range(0, len(df['time'])-1):
            t = df['time'][i+1] - df['time'][i]
            interv.append(t)
            
        dt = np.mean(interv)
        
        ##############################################################
        
        if df_rs is None:
            df_resp, info_resp = nk.rsp_process(resp, sampling_rate=1/dt) #neurokit method
            
            # es el largo de la ventana, en este caso son tiempo_w segundos, que es lo que defini antes
            largo = int(tiempo_w/dt)
            
            time, cuenta_picos = sliding_window(df_data, df_resp, largo, int(largo/2))
            interpolo = interpolate.interp1d(time, cuenta_picos, kind = 'cubic')

            x = np.arange(min(time), max(time), 6)
            y = interpolo(x)
            
            # Aca calculo la media de lo que calcule recien, esto es la desviacion estandar en varias sliding windows pero del reposo
            # Esto lo hago para restarselo a la dosis y sacarle la variabilidad de cada sujeto
            media_reposo = np.mean(y)

            tiempo = None
            time = None
            cuenta_picos = media_reposo

            
        else:
            df_resp, info_resp = nk.rsp_process(resp, sampling_rate=1/dt) #neurokit method
            df_resp['RSP_Clean'] = df_resp['RSP_Clean'] - np.mean(df_rs['RSP_Clean'])
            
            # es el largo de la ventana, en este caso son tiempo_w segundos, que es lo que defini antes
            largo = int(tiempo_w/dt)
            
            time, cuenta_picos = sliding_window(df_data, df_resp, largo, int(largo/2))
            interpolo = interpolate.interp1d(time, cuenta_picos, kind = 'cubic')

            x = np.arange(min(time), max(time), 6)
            y = interpolo(x)
            
            #Le resto a los puntos y a la interpolación, la media de la variabilidad del reposo
            cuenta_picos = cuenta_picos - media_rs #este es el más importante para restar pq es el que returneo
            y = y - media_rs
            
####### No me funciona el title porque creo que lo pone cada vez que crea un plot y algo hace raro pq quedan invertidos los titulos ###########
            # if numero == 1:
            #     plt.title('Dosis Alta')
            #     print('Alta', carpeta, numero)
            # else:
            #     plt.title('Dosis Baja')
            #     print('Baja', carpeta, numero)
###############################################################################################################################################            

            plt.figure(numero)
            plt.plot(time, cuenta_picos, 'o')#, color = '#5F9EA0')
            plt.plot(x, y, '-', label = f'{carpeta}')
            plt.ylabel('Desviación estandar')
            plt.xlabel('Tiempo (s)')
            plt.tight_layout()
            plt.legend()
            
            tiempo = df['time']
                
        return df_resp, info_resp, tiempo, time, cuenta_picos
            
            
        # return df_resp, info_resp, tiempo
        
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
carpetas = ['S04','S05','S07','S09','S13','S15','S17','S18','S19','S20'] #agregar S17
# carpetas = ['S05']
## Sujetos que tienen mala señal de respiración:
# S10 (DMT_2) S11 (DMT_2), S18 (DMT_2 - se mide bien el principio nomás), S08 (DMT_2, DMT_1 está bien), 
# S12 (DMT_2 - sólo llega a 0.1, DMT_1 está bien), S17 (DMT_1 - charlar con Tomi, DMT_2 se ve dentro de todo bien),
# S04 (DMT_1 y 2 - creo que con arreglar el codigo a que haya una distancia de picos minima esta bien), 
# S06 (safa), S07 (safa)

#%%

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
        df_resp_baja_rs, info_resp_baja_rs, none, none, media_rs_baja = resp(exp, carpet, nombre, j + 1)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_resp_alta_rs, info_resp_alta_rs, none, none, media_rs_alta = resp(exp, carpet, nombre, j)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_resp_baja_dmt, info_resp_baja_dmt, tiempo_bajo, tiempo_pico_baja, pico_baja = resp(exp, carpet, nombre, i + 1, tiempo_sliding_window, df_resp_baja_rs, media_rs_baja)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_resp_alta_dmt, info_resp_alta_dmt, tiempo_alto, tiempo_pico_alta, pico_alta = resp(exp, carpet, nombre, i, tiempo_sliding_window, df_resp_alta_rs, media_rs_alta)
               
        
    # La dosis es alta en la sesion 1
    else:
        
        j = 2
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_resp_alta_rs, info_resp_alta_rs, none, none, media_rs_alta = resp(exp, carpet, nombre, j)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_resp_baja_rs, info_resp_baja_rs, none, none, media_rs_baja = resp(exp, carpet, nombre , j + 1)
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_resp_alta_dmt, info_resp_alta_dmt, tiempo_alto, tiempo_pico_alta, pico_alta = resp(exp, carpet, nombre, i, tiempo_sliding_window, df_resp_alta_rs, media_rs_alta)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_resp_baja_dmt, info_resp_baja_dmt, tiempo_bajo, tiempo_pico_baja, pico_baja = resp(exp, carpet, nombre, i+1, tiempo_sliding_window, df_resp_baja_rs, media_rs_baja)
        
     
    ## Acá agrego el hecho de que guarde los datos de cada uno en listas para después promediar cada punto
    if len(df_resp_alta_dmt) != 1 and len(df_resp_alta_rs) != 1:
        
        prom_dmt = pico_alta
        prom_time = tiempo_pico_alta
        promedio_dmt_alta.append(prom_dmt)
        promedio_tiempo_alto.append(prom_time)
        
        
    else:
        print(f'Sujeto {carpet} tiene datos (con dosis DMT Alta) de RESP indescifrables')
        
    if len(df_resp_baja_dmt) != 1 and len(df_resp_baja_rs) != 1:
        
        prom_dmt = pico_baja
        prom_time = tiempo_pico_baja
        promedio_dmt_baja.append(prom_dmt)
        promedio_tiempo_bajo.append(prom_time)

    # if len(df_resp_baja_dmt) != 1 and len(df_resp_alta_dmt) != 1:  
    #     ######## Estoy pensando si hace falta que le ponga equal_var o alternative = "greater" ###############
    #     ttest = stats.ttest_ind(df_resp_alta_dmt['RSP_Rate'].values, df_resp_baja_dmt['RSP_Rate'].values, alternative = 'greater')
    #     pvalores.append(ttest.pvalue)
    #     t_stat.append(ttest.statistic)
    


#%% Ploteo de promedio con sombreado de error

def calculate_means_and_stdevs(data):
    
    # Convert the list of lists into a NumPy array
    data_array = np.array(data)
    
    # Calculate mean across columns (axis=0)
    means = np.mean(data_array, axis=0)
    
    # Calculate standard deviation across columns (axis=0) and divide by sqrt of number of rows
    errors = np.std(data_array, axis=0) / np.sqrt(data_array.shape[0])
    
    return means, errors

alta_promedio_listas, alta_errores = calculate_means_and_stdevs(promedio_dmt_alta)
baja_promedio_listas, baja_errores = calculate_means_and_stdevs(promedio_dmt_baja)
tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(promedio_tiempo_alto)
tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(promedio_tiempo_bajo)




#%% Ploteo con fill between

plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores/2
relleno_neg_alta = alta_promedio_listas - alta_errores/2


plt.plot(tiempo_alto_listas, alta_promedio_listas, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas, relleno_neg_alta, relleno_pos_alta, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores/2
relleno_neg_baja = baja_promedio_listas - baja_errores/2

plt.plot(tiempo_bajo_listas, baja_promedio_listas, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas, relleno_neg_baja, relleno_pos_baja, alpha = 0.2, color = "#FDD835")

plt.ylabel('Amplitud de Respiración restando baseline')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.legend()
plt.show()

#%% Test de Wilcoxon entre los dosis alta y dosis baja

from joblib import Parallel, delayed

def test(baja, alta):
    
    # Las listas de listas las paso a dataframe y las hago de la misma manera
    # Concateno los sujetos en un dataframe
    df_baja = pd.concat([pd.DataFrame(b) for b in baja], axis=1)
    df_alta = pd.concat([pd.DataFrame(a) for a in alta], axis=1)
    
    # Paso a arrays para poder hacer las stats
    arr_baja = df_baja.values
    arr_alta = df_alta.values
    
    # Preallocate the result array
    pvalor = np.empty(arr_baja.shape[0])
    
    def compute_pvalue(j):
        return stats.wilcoxon(arr_baja[j], arr_alta[j]).pvalue
   
    pvalor = Parallel(n_jobs=-1)(delayed(compute_pvalue)(j) for j in range(arr_baja.shape[0]))
    
    return pvalor
    
pvalores = test(promedio_dmt_baja, promedio_dmt_alta)



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

#%% Calculo de pvalores que pasan el umbral para después plotear, ya con fdr

#####################################################################################################################################
# Esta celda funciona asi, para calcular pvalores > 0.05, hay que comentar lo de fdr y los indices modificados por fdr
# Dsps descomentar indices menores al 0.05 y graficar en la siguiente celda. Con fdr es al reves y graficar con la siguiente celda.
####################################################################################################################################

arr_pvalores = np.array(pvalores)
# Calculo con fdr los pvalores nuevos
# rtas, pval_fdr = st.fdrcorrection(arr_pvalores)

# Indices de los pvalores que son menores que el 0.05
indices_p_pos = (np.where(arr_pvalores < 0.05))[0]

# Indices de los pvalores ya modificados por fdr que dan menores a 0.05
# indices_p_pos = (np.where(rtas == True))[0]

# Divido en regiones de los valores que cumplen para poder dibujar eso en el grafico
regiones = np.split(indices_p_pos, np.where(np.diff(indices_p_pos) != 1)[0] + 1)

frec_sampleo = int(info_resp_baja_dmt['sampling_rate'])

#%% AGREGADO: Ploteo con fill between Y Pvalores


plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores/2
relleno_neg_alta = alta_promedio_listas - alta_errores/2


plt.plot(tiempo_alto_listas, alta_promedio_listas, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas, relleno_neg_alta, relleno_pos_alta, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores/2
relleno_neg_baja = baja_promedio_listas - baja_errores/2

plt.plot(tiempo_bajo_listas, baja_promedio_listas, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas, relleno_neg_baja, relleno_pos_baja, alpha = 0.2, color = "#FDD835")

# Acá agrego el background de color gris
if len(regiones[0]) != 0: # en este caso el fdr me saca todos los pvalores
    for i, region in enumerate(regiones):
        start = region[0] + 1
        end = region[-1] + 1
        plt.axvspan(start/frec_sampleo, end/frec_sampleo, color='#C6C6C6', alpha=0.3)
        # mid_point = (start + end) / 2
        # plt.text(mid_point, plt.ylim()[1], f'Region {i+1}', horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

plt.ylabel('Amplitud de Respiración Normalizada')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.legend()
plt.show()
    
