# -*- coding: utf-8 -*-
import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc

#%% Uso MNE para descargar la data

fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/DMT_2/S16/S16_DMT_Session2_DMT.vhdr"
# fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Reposo_2/S19/S19_RS_Session2_EC.vhdr"

raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()
# raw_data.copy().pick(['ECG']).plot()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Separo los dataframes en arrays que pueda leer Neurokit, es decir, ECG, EDA, RSP

edap = df_data['GSR'] #EDA posta (para ponerlo despues sin problema) MicroSiemens
# ecgp = df_data['ECG'] #ECG posta
# resp = df_data['RESP'] #RESP

# edap = df_data['35'] #EDA posta (para ponerlo despues sin problema) MicroSiemens
# ecgp = df_data['33'] #ECG posta
# resp = df_data['34'] #RESP
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

df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt) #neurokit method
# df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method
# df_resp, info_resp = nk.rsp_process(resp, sampling_rate=1/dt) #Harrison 2021 method


#%% Ploteo de cada señal con NeuroKit

plot_eda = nk.eda_plot(df_eda, info = info_eda)
# plot_ecg = nk.ecg_plot(df_ecg, info = info_ecg)
# plot_resp = nk.rsp_plot(df_resp, info = info_resp)

#%% Procesamiento con HRV

# df, info = nk.bio_process(ecg = ecgp, rsp = resp, sampling_rate = 1/dt)
indices_hrv = nk.hrv(df_ecg['ECG_R_Peaks'], sampling_rate = 1/dt, show = True)
# results = nk.hrv(info, sampling_rate=1/dt, show=True)

#%% Ploteo de cada señal con los metodos alternos

# plot_ecg = nk.ecg_plot(df_ecg2, info = info_ecg2)
# plot_resp = nk.rsp_plot(df_resp2, info = info_resp2)









