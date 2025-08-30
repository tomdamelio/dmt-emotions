# -*- coding: utf-8 -*-
import mne
import matplotlib.pyplot as plt

#%%

fname = "C:/Users/Tomas Gil/Desktop/Tesis de Licenciatura/Sesiones/S19_DMT_Session2_DMT.vhdr"
raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Plotear Dataframe

plt.plot(df_data['time'], df_data['GSR'])


