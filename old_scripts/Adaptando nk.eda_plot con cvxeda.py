# -*- coding: utf-8 -*-

#########################################################################
##### IMPORTANTE: ESTE CODIGO SOLO CORRERLO EN COMPUS RAPIDAS ###########
#########################################################################


from warnings import warn

import mne
# import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

#%% Aca hago uso cvxEDA como default para calcular EDA

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


#%% Aca empieza la parte de hacer multiples plots de EDA


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
        
        df_eda, info_eda = eda_process(edap, sampling_rate=1/dt) #neurokit method
        
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
# carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
carpetas = ['S06']
            
#%% Calculo ECG para varios estudios comparando entre dosis

plt.close('all')  


for carpet in carpetas:
    
    for exp in exps:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            df_eda1, info_eda1 = eda(exp, carpet, nombre)
            dosis1 = dosis['Dosis_Sesion_1'][carpet]
            print(dosis1)
            
        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            df_eda2, info_eda2 = eda(exp, carpet, nombre) 
            dosis2 = dosis['Dosis_Sesion_2'][carpet]
            print(dosis2)
    
    if len(df_eda1) != 1 and len(df_eda2) != 1:
        eda_plot_superpuesto(df_eda1, df_eda2, info_eda1, info_eda2, exps, carpet, dosis1, dosis2)
    else:
        print(f'Sujeto {carpet} tiene datos indescifrables de EDA')
        
    if df_eda1 is not None:
        del df_eda1, df_eda2, info_eda1, info_eda2
        

