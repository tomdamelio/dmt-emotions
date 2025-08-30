# -*- coding: utf-8 -*-
from warnings import warn

import mne
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

#%%

def _signal_rate_plot(
    rate,
    peaks,
    sampling_rate=None,
    interpolation_method=None,
    title="Rate",
    ytitle="Cycle per minute",
    color="black",
    color_mean="orange",
    color_points="red",
    ax=None,
):
    # Prepare plot
    if ax is None:
        fig, ax = plt.subplots()

    if sampling_rate is None:
        x_axis = np.arange(0, len(rate))
        ax.set_xlabel("Time (samples)")
    else:
        x_axis = np.linspace(0, len(rate) / sampling_rate, len(rate))
        ax.set_xlabel("Time (seconds)")

    if interpolation_method is not None:
        title += " (interpolation method: " + str(interpolation_method) + ")"
    ax.set_title(title)
    ax.set_ylabel(ytitle)

    # Plot continuous rate
    ax.plot(
        x_axis,
        rate,
        color=color,
        label="Rate",
        linewidth=1.5,
    )

    # Plot points
    if peaks is not None:
        ax.scatter(
            x_axis[peaks],
            rate[peaks],
            color=color_points,
        )

    # Show average rate
    rate_mean = rate.mean()
    ax.axhline(y=rate_mean, label="Mean", linestyle="--", color=color_mean)

    ax.legend(loc="upper right")

    return ax

#%%

def _ecg_peaks_plot(
    ecg_cleaned,
    info=None,
    sampling_rate=1000,
    raw=None,
    quality=None,
    phase=None,
    ax=None,
):
    x_axis = np.linspace(0, len(ecg_cleaned) / sampling_rate, len(ecg_cleaned))

    # Prepare plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel("Time (seconds)")
    ax.set_title("ECG signal and peaks")

    # Quality Area -------------------------------------------------------------
    if quality is not None:
        quality = nk.stats.rescale(
            quality,
            to=[
                np.min([np.min(raw), np.min(ecg_cleaned)]),
                np.max([np.max(raw), np.max(ecg_cleaned)]),
            ],
        )
        minimum_line = np.full(len(x_axis), quality.min())

        # Plot quality area first
        ax.fill_between(
            x_axis,
            minimum_line,
            quality,
            alpha=0.12,
            zorder=0,
            interpolate=True,
            facecolor="#4CAF50",
            label="Signal quality",
        )

    # Raw Signal ---------------------------------------------------------------
    if raw is not None:
        ax.plot(x_axis, raw, color="#B0BEC5", label="Raw signal", zorder=1)
        label_clean = "Cleaned signal"
    else:
        label_clean = "Signal"

    # Peaks -------------------------------------------------------------------
    ax.scatter(
        x_axis[info["ECG_R_Peaks"]],
        ecg_cleaned[info["ECG_R_Peaks"]],
        color="#FFC107",
        label="R-peaks",
        zorder=2,
    )

    # Artifacts ---------------------------------------------------------------
    _ecg_peaks_plot_artefacts(
        x_axis,
        ecg_cleaned,
        info,
        peaks=info["ECG_R_Peaks"],
        ax=ax,
    )

    # Clean Signal ------------------------------------------------------------
    if phase is not None:
        mask = (phase == 0) | (np.isnan(phase))
        diastole = ecg_cleaned.copy()
        diastole[~mask] = np.nan

        # Create overlap to avoid interuptions in signal
        mask[np.where(np.diff(mask))[0] + 1] = True
        systole = ecg_cleaned.copy()
        systole[mask] = np.nan

        ax.plot(
            x_axis,
            diastole,
            color="#B71C1C",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )
        ax.plot(
            x_axis,
            systole,
            color="#F44336",
            zorder=3,
            linewidth=1,
        )
    else:
        ax.plot(
            x_axis,
            ecg_cleaned,
            color="#F44336",
            label=label_clean,
            zorder=3,
            linewidth=1,
        )

    # Optimize legend
    if raw is not None:
        handles, labels = ax.get_legend_handles_labels()
        order = [2, 0, 1, 3]
        ax.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
        )
    else:
        ax.legend(loc="upper right")

    return ax



def _ecg_peaks_plot_artefacts(
    x_axis,
    signal,
    info,
    peaks,
    ax,
):
    raw = [s for s in info.keys() if str(s).endswith("Peaks_Uncorrected")]
    if len(raw) == 0:
        return "No correction"
    raw = info[raw[0]]
    if len(raw) == 0:
        return "No bad peaks"
    if any([i < len(signal) for i in raw]):
        return "Peak indices longer than signal. Signals might have been cropped. " + "Better skip plotting."

    extra = [i for i in raw if i not in peaks]
    if len(extra) > 0:
        ax.scatter(
            x_axis[extra],
            signal[extra],
            color="#4CAF50",
            label="Peaks removed after correction",
            marker="x",
            zorder=2,
        )

    added = [i for i in peaks if i not in raw]
    if len(added) > 0:
        ax.scatter(
            x_axis[added],
            signal[added],
            color="#FF9800",
            label="Peaks added after correction",
            marker="x",
            zorder=2,
        )
    return ax


#%%

def ecg_plot_superpuesto(ecg_signals1, ecg_signals2, info1, info2, experimentos, carpeta, d1 = None, d2 = None, ax=None):
    
    # Sanity-check input.
    if not isinstance(ecg_signals1, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )
        
    # Sanity-check input.
    if not isinstance(ecg_signals2, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    if info1 is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category = nk.misc.NeuroKitWarning,
        )
        info1 = {"sampling_rate": 250}
        
    # Extract R-peaks.
    if info2 is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category = nk.misc.NeuroKitWarning,
        )
        info1 = {"sampling_rate": 250}

    # Extract R-peaks (take those from df as it might have been cropped)
    if "ECG_R_Peaks" in ecg_signals1.columns:
        info1["ECG_R_Peaks"] = np.where(ecg_signals1["ECG_R_Peaks"] == 1)[0]

    if "ECG_R_Peaks" in ecg_signals2.columns:
        info2["ECG_R_Peaks"] = np.where(ecg_signals2["ECG_R_Peaks"] == 1)[0]
    
    # Prepare plot
    fig, ax1 = plt.subplots()

    rate1 =  ecg_signals1["ECG_Rate"].values
    peaks1 = info1["ECG_R_Peaks"]
    color1 = "#FF5722"
    color_mean1 = "#FFC107"
    color_points1 = "#FF9800"
    
    rate2 =  ecg_signals2["ECG_Rate"].values
    peaks2 = info2["ECG_R_Peaks"]
    color2 = "black"
    color_mean2 = "#5A0014"
    color_points2 = "#800040"
    
    
    sampling_rate = info1["sampling_rate"]
    title = f'Heart Rate - {carpeta}'
    ytitle = "Ciclos por minuto"


    x_axis = np.linspace(0, len(rate1) / sampling_rate, len(rate1))
    ax1.set_xlabel("Tiempo (segundos)")

    ax1.set_title(title) 
    ax1.set_ylabel(ytitle)

    # Plot continuous rate
    ax1.plot(
        x_axis,
        rate1,
        color=color1,
        label=f'Rate - {experimentos[0]} ({d1})',
        linewidth=1.5,
    )

    # Plot points
    # ax1.scatter(
    #     x_axis[peaks1],
    #     rate1[peaks1],
    #     color=color_points1,
    # )

    # Show average rate
    rate_mean1 = rate1.mean()
    ax1.axhline(y = rate_mean1, label=f'Mean - {experimentos[0]}', linestyle="--", color=color_mean1)
    
    if (len(rate2) - len(rate1)) > 0:
        resta = np.abs(len(rate2) - len(rate1))
        rate2 = rate2[:-resta]
        
    elif (len(rate2) - len(rate1)) < 0:
        resta = np.abs(len(rate2) - len(rate1))
        rate1 = rate1[:-resta]
        x_axis = x_axis[:-resta]   
    
    ## 2nd plot
    # Plot continuous rate
    ax1.plot(
        x_axis,
        rate2,
        color=color2,
        label=f'Rate - {experimentos[1]} ({d2})',
        linewidth=1.5,
    )

    # # Plot points
    # ax1.scatter(
    #     x_axis[peaks2],
    #     rate2[peaks2],
    #     color=color_points2,
    # )

    # Show average rate
    rate_mean2 = rate2.mean()
    ax1.axhline(y = rate_mean2, label = f'Mean - {experimentos[1]}', linestyle="--", color=color_mean2)

    ax1.legend(loc = "upper right")

    

#%%

def ecg(experimento, carpeta, fname):
    
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
    
    df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method
    
    return df_ecg, info_ecg
        

#%% Carpetas y experimentos
exps = ['Reposo_1', 'Reposo_2']
carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
# carpetas = ['S16']

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
indices = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S17','S18', 'S19','S20']

dosis = pd.DataFrame(dosis, columns=columnas, index = indices)
            
#%% Calculo ECG para varios estudios

plt.close('all')  


for carpet in carpetas:
    
    for exp in exps:
        # if exp == 'DMT_1':
        if exp == 'Reposo_1':
            # nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            nombre = f'{carpet}_RS_Session1_EC.vhdr'
            df_ecg1, info_ecg1 = ecg(exp, carpet, nombre)
            # dosis1 = dosis['Dosis_Sesion_1'][carpet]
            
        # if exp == 'DMT_2':
        if exp == 'Reposo_2':
            # nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            nombre = f'{carpet}_RS_Session2_EC.vhdr'
            df_ecg2, info_ecg2 = ecg(exp, carpet, nombre) 
            # dosis2 = dosis['Dosis_Sesion_2'][carpet]
            
    ecg_plot_superpuesto(df_ecg1, df_ecg2, info_ecg1, info_ecg2, exps, carpet)#, dosis1, dosis2)
    
    if df_ecg1 is not None:
        del df_ecg1, df_ecg2, info_ecg1, info_ecg2
    
    