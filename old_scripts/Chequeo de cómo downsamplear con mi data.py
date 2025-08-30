# -*- coding: utf-8 -*-
from warnings import warn

import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
import statsmodels.stats.multitest as st
from scipy import signal

#%% Leer los archivos preprocesados

medida = 'HR' #Opciones: HR, SCL, SCR. Depende de tus carpetas y archivos

df_alta = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida}/{medida}_dmt_alta.csv')
df_baja = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida}/{medida}_dmt_baja.csv')
df_alta_tiempo = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida}/{medida}_tiempo_dmt_alta.csv')
df_baja_tiempo = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida}/{medida}_tiempo_dmt_baja.csv')

#%% Leer los archivos preprocesados de PCA

medida_2 = 'PCA'

pc1_alta = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc1_alta.csv')
pc1_baja = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc1_baja.csv')
pc2_alta = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc2_alta.csv')
pc2_baja = pd.read_csv(f'C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Preprocesado/{medida_2}/{medida_2}_pc2_baja.csv')

#%% Reestructuración de indices para mejor funcionamiento de correlacion

print(len(df_alta),'\n',len(df_baja),'\n',len(df_alta_tiempo),'\n',len(df_baja_tiempo))
print('OJO: esta df_alta acá, asi que si las length de antes son distintas, cagamos')

# plt.plot(df_alta_tiempo['S01'].values,df_alta['S01'].values)
# ydem = signal.decimate(df_alta['S01'].values, 5)
# xdem = signal.decimate(df_alta_tiempo['S01'].values, 5)
# plt.plot(xdem,ydem,'.')
print()
# print('Cantidad de veces que tengo que repetir el signal.decimate: ',int((len(df_alta)/len(pc1_alta))/13))
print('Cantidad de veces que tengo que repetir el signal.decimate: ',int((len(df_alta)/60000)/13))
print('Porque es mejor repetir dsps de 13 que poner un q mayor')

#### ESTA ES LA MANERA DE HACERSELO A TODO EL DATAFRAME DE UNA, QUEDA ORDENADO
ydem = signal.decimate(df_alta.values.T, 4)
ydem = pd.DataFrame(ydem.T)
plt.plot(df_alta_tiempo['S19'].values,df_alta['S19'].values,label='Normal')
xdem = signal.decimate(df_alta_tiempo.values.T, 4)
xdem = pd.DataFrame(xdem.T)
plt.plot(xdem[17],ydem[17],'.', label = 'Downsampleada')
plt.legend()
plt.tight_layout()

# def reindice_y_corr(data):
    
#     valores_corr_alta = pd.DataFrame()
#     valores_corr_baja = pd.DataFrame()
    
#     for i in range(len(data)):
    
#         new_index = np.linspace(data[i].index.min(), data[i].index.max(), num = len(df_alta), dtype = int)
    
#         # Reindex the dataframe to include the new points
#         df_expanded = data[i].reindex(new_index)
    
#         # Interpolate the data
#         df_interpolated = df_expanded.interpolate(method='linear')
#         extendido = df_interpolated.reset_index(drop=True) #Corrijo los indices para que dsps se compare correctamente
        
#         if i == 0 or i == 1:
        
#             corr_alta = df_alta.corrwith(extendido, method = 'pearson')
#             valores_corr_alta[f'PC{i+1}'] = corr_alta
            
#         else:
#             corr_baja = df_baja.corrwith(extendido, method = 'pearson')
#             valores_corr_baja[f'PC{i-1}'] = corr_baja
            
    
#     return valores_corr_alta, valores_corr_baja   


# #%% Chequeo ahora la correlacion entre sujetos

# pca = [pc1_alta, pc2_alta, pc1_baja, pc2_baja]

# valores_corr_alta, valores_corr_baja = reindice_y_corr(pca)
# print('\n', f"Valores de Correlacion de Pearson en Dosis Alta con {medida}",'\n', valores_corr_alta)
# print('\n', f"Valores de Correlacion de Pearson en Dosis Baja con {medida}",'\n', valores_corr_baja)


