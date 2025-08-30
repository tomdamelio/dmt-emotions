# -*- coding: utf-8 -*-

from warnings import warn

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk


#%%

def eda(experimento, carpeta, fname, malo = None):
    
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
        
        df_eda, info_eda = nk.eda_process(edap, sampling_rate=1/dt) #neurokit method
        
        return df_eda, info_eda
        
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
exps = ['DMT_1', 'Reposo_1']
carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
# carpetas = ['S07']
            
#%% Calculo para correlacion entre sesion 1 y sesion 2

plt.close('all')  

promedio_dmt = []
promedio_rs = []
puntos = []
i = 1

for carpet in carpetas:
    
    for exp in exps:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            df_eda1, info_eda1 = eda(exp, carpet, nombre)
            
        if exp == 'Reposo_1':
            nombre = f'{carpet}_RS_Session1_EC.vhdr'
            print(nombre)
            df_eda2, info_eda2 = eda(exp, carpet, nombre)
            print(nombre)
    
    if len(df_eda1) != 1 and len(df_eda2) != 1:
        
        prom_dmt = np.mean(df_eda1['EDA_Raw'])
        promedio_dmt.append(prom_dmt)
        prom_rs = np.mean(df_eda2['EDA_Raw'])
        promedio_rs.append(prom_rs)
        
    else:
        print(f'Sujeto {carpet} tiene datos indescifrables de EDA')
        
    if df_eda1 is not None:
        puntos.append(i)
        del df_eda1, df_eda2, info_eda1, info_eda2
        i += 1

#%% Ploteo de datos para ver correlación

arr_dmt = np.array(promedio_dmt)
arr_rs = np.array(promedio_rs)

plt.close('all')
plt.figure(1)
plt.scatter(arr_rs, arr_dmt)
plt.xlabel('<RS>')
plt.ylabel('<DMT>')
plt.show()

#%%

plt.figure(2)
plt.yscale('log')
plt.scatter(arr_rs, arr_dmt)
plt.xlabel('<RS>')
plt.ylabel('log(<DMT>)')
plt.show()


#%% Calculo correlación para dosis alta y dosis baja

plt.close('all')  

promedio_dmt_alta = []
promedio_dmt_baja = []
promedio_rs_alta = []
promedio_rs_baja = []

for carpet in carpetas:
        
        # Si la dosis es baja en la Sesion 1
    if dosis['Dosis_Sesion_1'][carpet] == 'Baja':
        
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt = eda(exp, carpet, nombre)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt = eda(exp, carpet, nombre)
        
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs = eda(exp, carpet, nombre)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs = eda(exp, carpet, nombre)
        
    # La dosis es alta en la sesion 1
    else:
        
        exp = 'DMT_1'
        nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
        df_eda_alta_dmt, info_eda_alta_dmt = eda(exp, carpet, nombre)

        exp = 'DMT_2'
        nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
        df_eda_baja_dmt, info_eda_baja_dmt = eda(exp, carpet, nombre)
        
        exp = 'Reposo_1'
        nombre = f'{carpet}_RS_Session1_EC.vhdr'
        df_eda_alta_rs, info_eda_alta_rs = eda(exp, carpet, nombre)
        
        exp = 'Reposo_2'
        nombre = f'{carpet}_RS_Session2_EC.vhdr'
        df_eda_baja_rs, info_eda_baja_rs = eda(exp, carpet, nombre)


    if len(df_eda_alta_dmt) != 1 and len(df_eda_alta_rs) != 1:
        
        prom_dmt = np.mean(df_eda_alta_dmt['EDA_Raw'])
        promedio_dmt_alta.append(prom_dmt)
        prom_rs = np.mean(df_eda_alta_rs['EDA_Raw'])
        promedio_rs_alta.append(prom_rs)
        
    else:
        print(f'Sujeto {carpet} tiene datos (con dosis DMT Alta) de EDA indescifrables')
        
    if len(df_eda_baja_dmt) != 1 and len(df_eda_baja_rs) != 1:
        
        prom_dmt = np.mean(df_eda_baja_dmt['EDA_Raw'])
        promedio_dmt_baja.append(prom_dmt)
        prom_rs = np.mean(df_eda_baja_rs['EDA_Raw'])
        promedio_rs_baja.append(prom_rs)

        
    if df_eda_alta_dmt is not None:
        del df_eda_alta_dmt, df_eda_alta_rs, 
        df_eda_baja_dmt, df_eda_baja_rs, 
        info_eda_alta_dmt, info_eda_alta_rs, 
        info_eda_baja_dmt, info_eda_baja_rs   


#%% Ploteo de datos para ver correlación por dosis

arr_dmt_alta = np.array(promedio_dmt_alta)
arr_rs_alta = np.array(promedio_rs_alta)
arr_dmt_baja = np.array(promedio_dmt_baja)
arr_rs_baja = np.array(promedio_rs_baja)

plt.close('all')
plt.figure(1)
plt.title('Dosis Alta')
plt.scatter(arr_rs_alta, arr_dmt_alta)
plt.xlabel('<RS>')
plt.ylabel('<DMT>')
plt.show()

plt.figure(2)
plt.title('Dosis Baja')
plt.scatter(arr_rs_baja, arr_dmt_baja)
plt.xlabel('<RS>')
plt.ylabel('<DMT>')
plt.show()






