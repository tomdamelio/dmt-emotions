# -*- coding: utf-8 -*-
#%%
import os

import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc
from scipy import interpolate
from scipy import stats
import pandas as pd
import statsmodels.stats.multitest as st

#%%
ROOT_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

def eda_methods(
    sampling_rate=1000,
    method="default",
    method_cleaning="default",
    method_peaks="default",
    method_phasic="cvxeda",
    **kwargs,
):
    
    # Sanitize inputs
    method_cleaning = str(method).lower() if method_cleaning == "default" else str(method_cleaning).lower()
    method_phasic = str(method).lower() if method_phasic == "default" else str(method_phasic).lower()
    method_peaks = str(method).lower() if method_peaks == "default" else str(method_peaks).lower()

    # Create dictionary with all inputs
    report_info = {
        "sampling_rate": sampling_rate,
        "method_cleaning": method_cleaning,
        "method_phasic": method_phasic,
        "method_peaks": method_peaks,
        "kwargs": kwargs,
    }

    # Get arguments to be passed to underlying functions
    kwargs_cleaning, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_clean)
    kwargs_phasic, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_phasic)
    kwargs_peaks, report_info = nk.misc.report.get_kwargs(report_info, nk.eda_peaks)

    # Save keyword arguments in dictionary
    report_info["kwargs_cleaning"] = kwargs_cleaning
    report_info["kwargs_phasic"] = kwargs_phasic
    report_info["kwargs_peaks"] = kwargs_peaks

    # Initialize refs list
    refs = []

    # 1. Cleaning
    # ------------
    report_info["text_cleaning"] = f"The raw signal, sampled at {sampling_rate} Hz,"
    if method_cleaning == "biosppy":
        report_info["text_cleaning"] += " was cleaned using the biosppy package."
    elif method_cleaning in ["default", "neurokit", "nk"]:
        report_info["text_cleaning"] += " was cleaned using the default method of the neurokit2 package."
    elif method_cleaning in ["none"]:
        report_info["text_cleaning"] += "was directly used without cleaning."
    else:
        report_info["text_cleaning"] += " was cleaned using the method described in " + method_cleaning + "."

    # 2. Phasic decomposition
    # -----------------------
    # TODO: add descriptions of individual methods
    report_info["text_phasic"] = "The signal was decomposed into phasic and tonic components using"
    if method_phasic is None or method_phasic in ["none"]:
        report_info["text_phasic"] = "There was no phasic decomposition carried out."
    else:
        report_info["text_phasic"] += " the method described in " + method_phasic + "."

    # 3. Peak detection
    # -----------------
    report_info["text_peaks"] = "The cleaned signal was used to detect peaks using"
    if method_peaks in ["gamboa2008", "gamboa"]:
        report_info["text_peaks"] += " the method described in Gamboa et al. (2008)."
        refs.append("""Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci
        and electrophysiology. PhD ThesisUniversidade.""")
    elif method_peaks in ["kim", "kbk", "kim2004", "biosppy"]:
        report_info["text_peaks"] += " the method described in Kim et al. (2004)."
        refs.append("""Kim, K. H., Bang, S. W., & Kim, S. R. (2004). Emotion recognition system using short-term
      monitoring of physiological signals. Medical and biological engineering and computing, 42(3),
      419-427.""")
    elif method_peaks in ["nk", "nk2", "neurokit", "neurokit2"]:
        report_info["text_peaks"] += " the default method of the `neurokit2` package."
        refs.append("https://doi.org/10.21105/joss.01667")
    elif method_peaks in ["vanhalem2020", "vanhalem", "halem2020"]:
        report_info["text_peaks"] += " the method described in Vanhalem et al. (2020)."
        refs.append("""van Halem, S., Van Roekel, E., Kroencke, L., Kuper, N., & Denissen, J. (2020).
      Moments That Matter? On the Complexity of Using Triggers Based on Skin Conductance to Sample
      Arousing Events Within an Experience Sampling Framework. European Journal of Personality.""")
    elif method_peaks in ["nabian2018", "nabian"]:
        report_info["text_peaks"] += " the method described in Nabian et al. (2018)."
        refs.append("""Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., & Ostadabbas, S. (2018). An
      Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data. IEEE
      journal of translational engineering in health and medicine, 6, 2800711.""")
    else:
        report_info[
            "text_peaks"
        ] = f"The peak detection was carried out using the method {method_peaks}."

    # References
    report_info["references"] = list(np.unique(refs))
    return report_info


#%%

def eda_process(
    eda_signal, sampling_rate=1000, method="neurokit", report=None, **kwargs
):
    
    # Sanitize input
    eda_signal = nk.signal_sanitize(eda_signal)
    methods = eda_methods(sampling_rate=sampling_rate, method=method, **kwargs)

    # Preprocess
    # Clean signal
    eda_cleaned = nk.eda_clean(
        eda_signal,
        sampling_rate=sampling_rate,
        method=methods["method_cleaning"],
        **methods["kwargs_cleaning"],
    )
    if methods["method_phasic"] is None or methods["method_phasic"].lower() == "none":
        eda_decomposed = pd.DataFrame({"EDA_Phasic": eda_cleaned})
    else:
        eda_decomposed = nk.eda_phasic(
            eda_cleaned,
            sampling_rate=sampling_rate,
            method=methods["method_phasic"],
            **methods["kwargs_phasic"],
        )

    # Find peaks
    peak_signal, info = nk.eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        method=methods["method_peaks"],
        amplitude_min=0.1,
        **methods["kwargs_peaks"],
    )
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    if report is not None:
        # Generate report containing description and figures of processing
        if ".html" in str(report):
            fig = nk.eda_plot(signals, info, static=False)
        else:
            fig = None
        nk.create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info


#%% Definición del sliding window

def sliding_window(data_normal, signal, k, step):
    
    window_count = []
    tiempos = []
    
    n = len(signal)
    
    if n <= k:
        print('Array de picos más corto que la ventana propuesta')
        return -1
    
    window = signal[:k]
    window_sum = sum(window['SCR_Peaks'])
    window_count.append(window_sum)
    
    window = data_normal[:k]
    tiempo = np.mean(window['time'])
    tiempos.append(tiempo)

    
    step_inic = step
    
    while step + step_inic < len(signal):
        
        window = signal[step:step + k]
        window_sum = sum(window['SCR_Peaks'])
        window_count.append(window_sum)
        
        window = data_normal[step:step + k]
        tiempo = np.mean(window['time'])
        tiempos.append(tiempo)
        
        step += step_inic
        
    return tiempos, window_count



#%%

# Largo de la ventana para la sliding window, este caso va a ser de 120 segundos
tiempo_sliding_window = 120

def scr(experimento, carpeta, fname, tiempo_w = tiempo_sliding_window, numero = None, df_rs = None, malo = None):
    
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
            
        df_eda, info_eda = eda_process(edap, sampling_rate=1/dt) #neurokit method
        # plot_eda = nk.eda_plot(df_eda, info = info_eda)
        
        # es el largo de la ventana, en este caso son tiempo_w segundos, que es lo que defini antes
        largo = int(tiempo_w/dt)
        
        time, cuenta_picos = sliding_window(df_data, df_eda, largo, int(largo/2))
        interpolo = interpolate.interp1d(time, cuenta_picos, kind = 'cubic')

        x = np.arange(min(time), max(time), 6)
        y = interpolo(x)
        
        plt.figure(numero)
        plt.plot(time, cuenta_picos, 'o', color = '#5F9EA0')
        plt.plot(x, y, '-', color = '#A52A2A')
        plt.ylabel('Cantidad de picos')
        plt.xlabel('Tiempo (s)')
        plt.tight_layout()
        plt.legend()
        
        tiempo = df['time']
            
        return df_eda, info_eda, tiempo, time, cuenta_picos
        
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



#%%
## Ploteo por ahora solo las que tengo info
# carpetas = ['S04','S05','S06','S07','S09','S13','S17','S18', 'S19','S20']
carpetas = ['S04','S05','S06','S07','S09','S13','S16','S17','S18', 'S19','S20']

#%%

plt.close('all')  

promedio_dmt_alta = []
promedio_dmt_baja = []
promedio_tiempo_alto = []
promedio_tiempo_bajo = []


for carpet in carpetas:
        
        # Si la dosis es baja en la Sesion 1
    if dosis['Dosis_Sesion_1'][carpet] == 'Baja':
    
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo, tiempo_pico_baja, pico_baja = scr(exp, carpet, nombre, tiempo_sliding_window, i + 1)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto, tiempo_pico_alta, pico_alta = scr(exp, carpet, nombre, tiempo_sliding_window, i)
        
    # La dosis es alta en la sesion 1
    else:
        
        i = 1
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt, tiempo_alto, tiempo_pico_alta, pico_alta = scr(exp, carpet, nombre, tiempo_sliding_window, i)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt, tiempo_bajo, tiempo_pico_baja, pico_baja = scr(exp, carpet, nombre, tiempo_sliding_window, i+1)
        
     
    ## Acá agrego el hecho de que guarde los datos de cada uno en listas para después promediar cada punto
    if len(df_eda_alta_dmt) != 1:
        
        prom_dmt = pico_alta
        prom_time = tiempo_pico_alta
        promedio_dmt_alta.append(prom_dmt)
        promedio_tiempo_alto.append(prom_time)
        
        
    else:
        print(f'Sujeto {carpet} tiene datos (con dosis DMT Alta) de EDA indescifrables')
        
    if len(df_eda_baja_dmt) != 1:
        
        prom_dmt = pico_baja
        prom_time = tiempo_pico_baja
        promedio_dmt_baja.append(prom_dmt)
        promedio_tiempo_bajo.append(prom_time)
    
        

    # if df_eda_alta_dmt is not None:
    #     del df_eda_alta_dmt, df_eda_alta_rs, 
    #     df_eda_baja_dmt, df_eda_baja_rs, 
    #     info_eda_alta_dmt, info_eda_alta_rs, 
    #     info_eda_baja_dmt, info_eda_baja_r
 
# SI NO SE QUIERE GUARDAR: Saltear estas dos celdas y descomentar las que digan "promedio_dmt_..." y hagan las transposicion, se usa en el promediado y en pvalores
#%% Guardado de archivos de HR para después promediar o comparar con PCA más facil

medida = 'SCR'
preproc_dir = os.path.join(ROOT_DATA, 'Preprocesado', medida)
os.makedirs(preproc_dir, exist_ok=True)

#%%

# Lista de datos procesados pasada a dataframe y guardada en csv
array_alta_guardar = np.array(promedio_dmt_alta)
df_alta_guardar = pd.DataFrame(array_alta_guardar).T
df_alta_guardar.columns = carpetas
csv = f'{medida}_dmt_alta.csv'
df_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

array_baja_guardar = np.array(promedio_dmt_baja)
df_baja_guardar = pd.DataFrame(array_baja_guardar).T
df_baja_guardar.columns = carpetas
csv = f'{medida}_dmt_baja.csv'
df_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

array_tiempo_alta_guardar = np.array(promedio_tiempo_alto)
df_tiempo_alta_guardar = pd.DataFrame(array_tiempo_alta_guardar).T
df_tiempo_alta_guardar.columns = carpetas
csv = f'{medida}_tiempo_dmt_alta.csv'
df_tiempo_alta_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

array_tiempo_baja_guardar = np.array(promedio_tiempo_bajo)
df_tiempo_baja_guardar = pd.DataFrame(array_tiempo_baja_guardar).T
df_tiempo_baja_guardar.columns = carpetas
csv = f'{medida}_tiempo_dmt_baja.csv'
df_tiempo_baja_guardar.to_csv(os.path.join(preproc_dir, csv), index=False)

#%% Leer los archivos preprocesados

scr_alta = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_alta.csv'))
scr_baja = pd.read_csv(os.path.join(preproc_dir, f'{medida}_dmt_baja.csv'))
scr_alta_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_alta.csv'))
scr_baja_tiempo = pd.read_csv(os.path.join(preproc_dir, f'{medida}_tiempo_dmt_baja.csv'))
   
 
#%% Ploteo de promedio con sombreado de error

# Mostra promedio dmt antes de correr lo siguiente porque van a ser listas creo

def calculate_means_and_stdevs(data):
    
    ############ Descomentar si corro todo sin el preprocesado ###########
    # Convert the list of lists into a NumPy array
    # data_array = np.array(data)

    # Calculate mean across columns (axis=0)
    # means = np.mean(data_array, axis=0)
    
    # Calculate standard deviation across columns (axis=0) and divide by sqrt of number of rows
    # errors = np.std(data_array, axis=0) / np.sqrt(data_array.shape[0])
    ######################################################################
    
    df = data
    
    # Calculate mean across columns (axis=1)
    means = df.mean(axis=1)

    # Calculate standard deviation across columns (axis=1)
    errors = df.std(axis=1)/np.sqrt(df.shape[1])
    
    return means, errors

# Descomentar si no se quiere guardar el preprocesado############
# alta_promedio_listas, alta_errores = calculate_means_and_stdevs(promedio_dmt_alta)
# baja_promedio_listas, baja_errores = calculate_means_and_stdevs(promedio_dmt_baja)
# tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(promedio_tiempo_alto)
# tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(promedio_tiempo_bajo)
#################################################################

alta_promedio_listas, alta_errores = calculate_means_and_stdevs(scr_alta)
baja_promedio_listas, baja_errores = calculate_means_and_stdevs(scr_baja)
tiempo_alto_listas, tiempo_errores_alto = calculate_means_and_stdevs(scr_alta_tiempo)
tiempo_bajo_listas, tiempo_errores_bajo = calculate_means_and_stdevs(scr_baja_tiempo)


#%% Ploteo con fill between y grafico el promedio

plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores
relleno_neg_alta = alta_promedio_listas - alta_errores


plt.plot(tiempo_alto_listas, alta_promedio_listas, label = 'Dosis Alta', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas, relleno_neg_alta, relleno_pos_alta, alpha=0.2, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores
relleno_neg_baja = baja_promedio_listas - baja_errores

plt.plot(tiempo_bajo_listas, baja_promedio_listas, label = 'Dosis Baja', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas, relleno_neg_baja, relleno_pos_baja, alpha = 0.2, color = "#FDD835")

plt.ylabel('Cantidad de Picos Promediada')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.legend()
plt.show()

#%% Test de Wilcoxon entre los dosis alta y dosis baja

from joblib import Parallel, delayed

def test(baja, alta):
    
    ############### Descomentar esto si corro sin preprocesado #################
    # Las listas de listas las paso a dataframe y las hago de la misma manera
    # Concateno los sujetos en un dataframe
    # df_baja = pd.concat([pd.DataFrame(b) for b in baja], axis=1)
    # df_alta = pd.concat([pd.DataFrame(a) for a in alta], axis=1)
    
    # # Paso a arrays para poder hacer las stats
    # arr_baja = df_baja.values
    # arr_alta = df_alta.values
    
    # # Preallocate the result array
    # pvalor = np.empty(arr_baja.shape[0])
    
    # def compute_pvalue(j):
    #     return stats.wilcoxon(arr_baja[j], arr_alta[j]).pvalue
   
    # pvalor = Parallel(n_jobs=-1)(delayed(compute_pvalue)(j) for j in range(arr_baja.shape[0]))
    ###########################################################################
    
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
pvalores = test(scr_baja, scr_alta)


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
plt.plot(tiempo_bajo_listas,pvalores,'o', label = 'P-valor', color = '#A366A6')
plt.plot(tiempo_bajo_listas, 0.05*np.ones(len(tiempo_bajo_listas)),'--', label = 'Umbral', color = '#CCA60D')
plt.xlabel('Tiempo(s)')
plt.legend()
plt.tight_layout()

#%% Calculo de pvalores que pasan el umbral para después plotear

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
tiempo_sliding_window = 120
size = 14

plt.close('all')

relleno_pos_alta = alta_promedio_listas + alta_errores
relleno_neg_alta = alta_promedio_listas - alta_errores


plt.plot(tiempo_alto_listas, alta_promedio_listas, label = 'High Dose', color = "#9C27B0")
plt.fill_between(tiempo_alto_listas, relleno_neg_alta, relleno_pos_alta, alpha=0.4, color = "#9C27B0")


relleno_pos_baja = baja_promedio_listas + baja_errores
relleno_neg_baja = baja_promedio_listas - baja_errores

plt.plot(tiempo_bajo_listas, baja_promedio_listas, label = 'Low Dose', color = "#FFA726")
plt.fill_between(tiempo_bajo_listas, relleno_neg_baja, relleno_pos_baja, alpha = 0.4, color = "#FDD835")

# Acá agrego el background de color gris
for i, region in enumerate(regiones):
    start = region[0] + 1
    end = region[-1] + 1
    # Como aca tengo los puntos movidos por ventanas, tengo que trasladarlos acordemente
    # NOTA: no pongo la frecuencia de muestreo porque se cancela con la que multiplico con la ventana
    plt.axvspan(start*int(tiempo_sliding_window/2), end*int(tiempo_sliding_window/2), color='#C6C6C6', alpha=0.4)
    # mid_point = (start + end) / 2
    # plt.text(mid_point, plt.ylim()[1], f'Region {i+1}', horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

plt.ylabel('Amount of SCR peaks', fontsize = size)
plt.xlabel('Time (s)', fontsize = size)
plt.xticks(fontsize = size)
plt.yticks(fontsize = size)
plt.tight_layout()
plt.legend(fontsize = size)
plt.show()

#%% Ploteo de pvalores con fdr para mostrar qué valores dan

plt.figure(2)
plt.plot(tiempo_bajo_listas,pval_fdr,'o', label = 'P-valor', color = '#A366A6')
plt.plot(tiempo_bajo_listas, 0.05*np.ones(len(tiempo_bajo_listas)),'--', label = 'Umbral', color = '#CCA60D')
plt.xlabel('Tiempo(s)')
plt.legend()
plt.tight_layout()

# %%
