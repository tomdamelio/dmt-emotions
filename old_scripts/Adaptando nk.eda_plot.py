# -*- coding: utf-8 -*-
from warnings import warn

import mne
# import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

#%%

def eda_plot_superpuesto(eda_signals1, eda_signals2, info1, info2, experimentos, carpeta, d1, d2, static=True):
    
    if info1 is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category= nk.misc.NeuroKitWarning,
        )

        info1 = {
            "sampling_rate": 1000,
        }

    # Determine unit of x-axis.
    x_label = "Time (seconds)"
    x_axis = np.linspace(0, len(eda_signals1) / info1["sampling_rate"], len(eda_signals1))
    
    if (len(eda_signals2["EDA_Clean"].values) - len(eda_signals1["EDA_Clean"].values)) > 0:
        resta = np.abs(len(eda_signals2["EDA_Clean"].values) - len(eda_signals1["EDA_Clean"].values))
        eda_signals2 = eda_signals2[:-resta]
        
    elif (len(eda_signals2["EDA_Clean"].values) - len(eda_signals1["EDA_Clean"].values)) < 0:
        resta = np.abs(len(eda_signals2["EDA_Clean"].values) - len(eda_signals1["EDA_Clean"].values))
        eda_signals1 = eda_signals1[:-resta]
        x_axis = x_axis[:-resta]   
    

    if static:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True)

        last_ax = fig.get_axes()[-1]
        last_ax.set_xlabel(x_label)

        # Plot cleaned and raw electrodermal activity.
        ax0.set_title(f'Raw and Cleaned Signal - {carpeta}')
        fig.suptitle("Electrodermal Activity (EDA)", fontweight="bold")

        ax0.plot(
            x_axis,
            eda_signals1["EDA_Clean"],
            color="#9C27B0",
            label=f'{experimentos[0]} ({d1})',
            linewidth=1.5,
            zorder=1,
        )
        
        ax0.plot(
            x_axis,
            eda_signals2["EDA_Clean"],
            color="Black",
            label=f'{experimentos[1]} ({d2})',
            linewidth=1.5,
            zorder=1,
            alpha=0.7
        )
        ax0.legend(loc="upper right")

        # Plot skin conductance response.
        ax1.set_title("Skin Conductance Response (SCR)")

        # Plot Phasic.
        ax1.plot(
            x_axis,
            eda_signals1["EDA_Phasic"],
            color="#9C27B0",
            label=f'{experimentos[0]} ({d1})',
            linewidth=1.5,
            zorder=1,
        )
        # Plot Phasic.
        ax1.plot(
            x_axis,
            eda_signals2["EDA_Phasic"],
            color="Black",
            label=f'{experimentos[1]} ({d2})',
            linewidth=1.5,
            zorder=1,
            alpha=0.7
        )
        ax1.legend(loc="upper right")

        # Plot Tonic.
        ax2.set_title("Skin Conductance Level (SCL)")
        ax2.plot(
            x_axis,
            eda_signals1["EDA_Tonic"],
            color="#673AB7",
            label=f'{experimentos[0]} ({d1})',
            linewidth=1.5,
        )
        # Plot Tonic.
        ax2.plot(
            x_axis,
            eda_signals2["EDA_Tonic"],
            color="Black",
            label=f'{experimentos[1]} ({d2})',
            linewidth=1.5,
            alpha = 0.7
        )
        ax2.legend(loc="upper right")




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
exps = ['DMT_1', 'DMT_2']
carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
# carpetas = ['S18','S19']
            
#%% Calculo ECG para varios estudios

plt.close('all')  


for carpet in carpetas:
    
    for exp in exps:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            df_eda1, info_eda1 = eda(exp, carpet, nombre)
            dosis1 = dosis['Dosis_Sesion_1'][carpet]
            
        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            df_eda2, info_eda2 = eda(exp, carpet, nombre) 
            dosis2 = dosis['Dosis_Sesion_2'][carpet]
    
    if len(df_eda1) != 1 and len(df_eda2) != 1:
        eda_plot_superpuesto(df_eda1, df_eda2, info_eda1, info_eda2, exps, carpet, dosis1, dosis2)
    else:
        print(f'Sujeto {carpet} tiene datos indescifrables de EDA')
        
    if df_eda1 is not None:
        del df_eda1, df_eda2, info_eda1, info_eda2

