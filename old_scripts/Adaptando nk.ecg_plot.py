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

def ecg_plot(ecg_signals, numero, color, color_mean, color_points, info=None, ax=None):
    
    # Sanity-check input.
    if not isinstance(ecg_signals, pd.DataFrame):
        raise ValueError(
            "NeuroKit error: ecg_plot(): The `ecg_signals` argument must be the "
            "DataFrame returned by `ecg_process()`."
        )

    # Extract R-peaks.
    if info is None:
        warn(
            "'info' dict not provided. Some information might be missing."
            + " Sampling rate will be set to 1000 Hz.",
            category = nk.misc.NeuroKitWarning,
        )
        info = {"sampling_rate": 1000}

    # Extract R-peaks (take those from df as it might have been cropped)
    if "ECG_R_Peaks" in ecg_signals.columns:
        info["ECG_R_Peaks"] = np.where(ecg_signals["ECG_R_Peaks"] == 1)[0]

    # Prepare figure and set axes.    
    gs = matplotlib.gridspec.GridSpec(1, 1)
    
    #Esto es lo que me va a adaptar todo para tener dos plots en un mismo grafico
    if ax is None:
        fig = plt.figure(numero)
        ax = fig.add_subplot(1, 1, 1)

    fig = plt.figure(numero)#,constrained_layout=False)
    fig.suptitle("Electrocardiogram (ECG)", fontweight="bold")

    ax1 = fig.add_subplot(gs[0,0])

    # Plot Heart Rate
    ax1 = _signal_rate_plot(
        ecg_signals["ECG_Rate"].values,
        info["ECG_R_Peaks"],
        sampling_rate=info["sampling_rate"],
        title="Heart Rate",
        ytitle="Beats per minute (bpm)",
        color=color,
        color_mean=color_mean,
        color_points=color_points,
        ax=ax1,
    )
    
    return ax
    

#%%

def ecg(experimento, carpeta, fname, num, color, color_mean, color_points):
    
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

    plot_ecg = ecg_plot(df_ecg, num, color, color_mean, color_points, info = info_ecg) 
        

#%% Carpetas y experimentos
exps = ['DMT_1', 'DMT_2']
# carpetas = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10','S11','S12','S13','S15','S16','S17','S18', 'S19','S20']
carpetas = ['S01']
            
#%% Calculo ECG para varios estudios
       
for exp in exps:
    
    i = 0
    for carpet in carpetas:
        if exp == 'DMT_1':
            nombre = f'{carpet}_DMT_Session1_DMT.vhdr'
            i += 1
            ecg(exp, carpet, nombre, i,"#FF5722", "#FF9800","#FFC107")

        if exp == 'DMT_2':
            nombre = f'{carpet}_DMT_Session2_DMT.vhdr'
            i += 1
            ecg(exp, carpet, nombre, i, "black", "orange", "#800040") 
            
#%% 

# First plot
plot_ecg1 = ecg_plot(df_ecg1, num1, color1, color_mean1, color_points1, info=info_ecg1)

# Second plot on the same figure
plot_ecg2 = ecg_plot(df_ecg2, num2, color2, color_mean2, color_points2, info=info_ecg2, ax=plot_ecg1)

## ESTO ES LO QUE ME FALTA AGREGAR QUE YO CREO QUE PONIENDO UN PAR DE F'{}
## Y UN PAR DE DEFINICIONES DE VARIABLES SALE EN DOS TOQUES

#%%
import matplotlib.pyplot as plt
import numpy as np

# Example data for two ECG signals
time = np.linspace(0, 10, 1000)
ecg_signal1 = np.sin(2 * np.pi * 1 * time)  # ECG signal 1
ecg_signal2 = np.sin(2 * np.pi * 2 * time)  # ECG signal 2

# Create the figure and subplots
fig, ax1 = plt.subplots()

# Plot the first ECG signal
ax1.plot(time, ecg_signal1, color='blue', label='ECG Signal 1')

# Plot the second ECG signal on the same subplot
ax1.plot(time, ecg_signal2, color='red', label='ECG Signal 2')

# Add labels and legend
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage')
ax1.set_title('ECG Signals')
ax1.legend()

# Show the plot
plt.show()




