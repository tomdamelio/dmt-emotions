# -*- coding: utf-8 -*-
import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

#%% 

def eda(experimento, carpeta, fname, numero):
    
    fname = f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/{experimento}/{carpeta}/{fname}'
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    if 'GSR' in df_data.columns:
        edap = df_data['GSR'] #ECG posta
        interv = []
        for i in range(0, len(df_data['time'])-1):
            t = df_data['time'][i+1] - df_data['time'][i]
            interv.append(t)
            
        dt = np.mean(interv)
        
        plt.figure(numero)
        df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt)
        # nk.eda_plot(df_eda, info = info_eda)
        plt.plot(df_data['time'], df_eda['EDA_Clean'], label = f'{carpeta} - {experimento}')
        plt.xlabel('Tiempo (s)')
        plt.legend()
    
    # interv = []
    # for i in range(0, len(df_data['time'])-1):
    #     t = df_data['time'][i+1] - df_data['time'][i]
    #     interv.append(t)
        
    # dt = np.mean(interv)
    
    # df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt)
    # # nk.eda_plot(df_eda, info = info_eda)
    # plt.plot(df_data['time'], df_eda['EDA_Clean'], label = f'{carpeta}')
    # plt.xlabel('Tiempo (s)')
    # plt.legend()

#%% ECG

def ecg(experimento, carpeta, fname, numero):
    
    fname = f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/{experimento}/{carpeta}/{fname}'
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    if 'ECG' in df_data.columns:
        
        ecgp = df_data['ECG'] #ECG posta
        
    elif '33' in df_data.columns:
        
        ecgp = df_data['33'] #ECG posta
        
    interv = []
    for i in range(0, len(df_data['time'])-1):
        t = df_data['time'][i+1] - df_data['time'][i]
        interv.append(t)
        
    dt = np.mean(interv)
    
    plt.figure(numero)
    df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt)
    # nk.eda_plot(df_eda, info = info_eda)
    plt.plot(df_data['time'], df_ecg['ECG_Clean'], label = f'{carpeta} - {experimento}')
    plt.xlabel('Tiempo (s)')
    plt.legend()
    
#%% Resp

def resp(experimento, carpeta, fname, numero):
    
    fname = f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/{experimento}/{carpeta}/{fname}'
    
    raw_data = mne.io.read_raw_brainvision(fname)
    data = raw_data.load_data()

    df_data = data.to_data_frame()
    print(df_data.columns)
    
    if 'RESP' in df_data.columns:
        
        resp = df_data['RESP'] #ECG posta
        
        interv = []
        for i in range(0, len(df_data['time'])-1):
            t = df_data['time'][i+1] - df_data['time'][i]
            interv.append(t)
            
        dt = np.mean(interv)
        
        plt.figure(numero)
        df_resp, info_resp = nk.rsp_process(resp, sampling_rate=1/dt)
        # nk.eda_plot(df_eda, info = info_eda)
        plt.plot(df_data['time'], df_resp['RSP_Clean'], label = f'{carpeta} - {experimento}')
        plt.xlabel('Tiempo (s)')
        plt.legend()    
        

#%% Carpetas y experimentos
exps = ['DMT_1', 'DMT_2']
carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']

#%% Calculo EDA para varios estudios
# Separo grafico por sujeto

for exp in exps:
    
    i = 0
    
    for carpet in carpetas:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            i += 1
            eda(exp, carpet, nombre, i)

        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            i += 1
            eda(exp, carpet, nombre, i)
            
#%% Calculo ECG para varios estudios
       
for exp in exps:
    
    i = 0
    
    for carpet in carpetas:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            i += 1
            ecg(exp, carpet, nombre, i)

        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            i += 1
            ecg(exp, carpet, nombre, i) 
            
#%% Calculo RESP para varios estudios
       
for exp in exps:
    
    i = 0
    
    for carpet in carpetas:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            i += 1
            resp(exp, carpet, nombre, i)

        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            i += 1
            resp(exp, carpet, nombre, i) 






