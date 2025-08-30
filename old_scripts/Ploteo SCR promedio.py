# -*- coding: utf-8 -*-

import mne
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import scipy.signal as sc
from scipy import interpolate
import pandas as pd

#%%

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



#%% Uso MNE para descargar la data

fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/DMT_2/S13/S13_DMT_Session2_DMT.vhdr"
# fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Reposo_2/S19/S19_RS_Session2_EC.vhdr"

raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()
# raw_data.copy().pick(['ECG']).plot()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Separo los dataframes en arrays que pueda leer Neurokit, es decir, ECG, EDA, RSP

edap = df_data['GSR'] #EDA posta (para ponerlo despues sin problema) MicroSiemens

#%% Generalizo el intervalo de sampleo, pongo lo que tenga el instrumental para no perder info

######### ¿Me conviene el promedio, el max o el minimo? ##############

interv = []
for i in range(0, len(df_data['time'])-1):
    t = df_data['time'][i+1] - df_data['time'][i]
    interv.append(t)
    
dt = np.mean(interv)
### se que lo dice el vhdr pero no se si confiar en el sistema de primera

#%% Leo los datos con cvxeda

df_eda, info_eda = eda_process(edap, sampling_rate=1/dt) #neurokit method

#%% Ploteo de cvxEDA

plot_eda = nk.eda_plot(df_eda, info = info_eda)

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


#%% Hago el calculo de sliding windows

sampling_rate = 1/dt

largo = int(120*sampling_rate) #esta es una ventana de 120 segundos

# picos = np.where(df_eda["SCR_Peaks"] == 1)[0]
# pico_x_values = x_axis[picos]
# pico_y_values = df_eda["EDA_Phasic"][picos].values

time, cuenta_picos = sliding_window(df_data, df_eda, largo, int(largo/2))

#%% Ploteo los puntos en sus ventanas e interpolo para discretizar

interpolo = interpolate.interp1d(time, cuenta_picos, kind = 'cubic')

x = np.arange(min(time), max(time), 6)
y = interpolo(x)

plt.figure(2)
plt.plot(time, cuenta_picos, 'o', color = '#5F9EA0')
plt.plot(x, y, '-', color = '#A52A2A')
plt.ylabel('Cantidad de picos')
plt.xlabel('Tiempo (s)')
plt.tight_layout()
plt.show()

