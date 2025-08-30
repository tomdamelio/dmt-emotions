# -*- coding: utf-8 -*-
import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc
from scipy import interpolate

#%% Uso MNE para descargar la data

fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/DMT_1/S19/S19_DMT_Session1_DMT.vhdr"
# fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Reposo_2/S19/S19_RS_Session2_EC.vhdr"

raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()
# raw_data.copy().pick(['ECG']).plot()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Separo los dataframes en arrays que pueda leer Neurokit, es decir, ECG, EDA, RSP

# edap = df_data['GSR'] #EDA posta (para ponerlo despues sin problema) MicroSiemens
# ecgp = df_data['ECG'] #ECG posta
resp = df_data['RESP'] #RESP
# 
#%% Generalizo el intervalo de sampleo, pongo lo que tenga el instrumental para no perder info

######### ¿Me conviene el promedio, el max o el minimo? ##############

interv = []
for i in range(0, len(df_data['time'])-1):
    t = df_data['time'][i+1] - df_data['time'][i]
    interv.append(t)
    
dt = np.mean(interv)
### se que lo dice el vhdr pero no se si confiar en el sistema de primera

#%% Leo los datos con NeuroKit (aunque RSP analiza con Harrison 2021)

# df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt) #neurokit method
# df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method
df_resp, info_resp = nk.rsp_process(resp, sampling_rate=1/dt) #Harrison 2021 method

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

# es el largo de la ventana, en este caso son tiempo_w segundos, que es lo que defini antes
largo = int(tiempo_sliding_window/dt)

time, cuenta_picos = sliding_window(df_data, df_resp, largo, int(largo/2))

interpolo = interpolate.interp1d(time, cuenta_picos, kind = 'cubic')

x = np.arange(min(time), max(time), 6)
y = interpolo(x)

plt.figure()
plt.plot(time, cuenta_picos, 'o', color = '#5F9EA0')
plt.plot(x, y, '-', color = '#A52A2A')
plt.ylabel('Desviación estandar')
plt.xlabel('Tiempo (s)')
plt.tight_layout()


plt.figure()
plt.plot(df_data['time'],df_resp['RSP_Clean'], label = 'Señal Resp Limpia', alpha = 0.5, color = 'grey')
plt.ylabel('Movimiento cinturón')
plt.xlabel('Tiempo (s)')
plt.legend()



