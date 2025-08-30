# -*- coding: utf-8 -*-
import mne
import neurokit2 as nk
import pandas as pd
import numpy as np

#%%

def eda_process(
    eda_signal, sampling_rate=1000, method="neurokit", report=None, **kwargs
):
    """**Process Electrodermal Activity (EDA)**

    Convenience function that automatically processes electrodermal activity (EDA) signal.

    Parameters
    ----------
    eda_signal : Union[list, np.array, pd.Series]
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of ``"eda_signal"`` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"biosppy"`` or ``"neurokit"`` (default).
    report : str
        The filename of a report containing description and figures of processing
        (e.g. ``"myreport.html"``). Needs to be supplied if a report file
        should be generated. Defaults to ``None``. Can also be ``"text"`` to
        just print the text in the console without saving anything.
    **kwargs
        Other arguments to be passed to specific methods. For more information,
        see :func:`.rsp_methods`.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as ``"eda_signal"`` containing the following
        columns:

        * ``"EDA_Raw"``: the raw signal.

        * ``"EDA_Clean"``: the cleaned signal.

        * ``"EDA_Tonic"``: the tonic component of the signal, or the Tonic Skin Conductance Level
          (SCL).

        * ``"EDA_Phasic"``: the phasic component of the signal, or the Phasic Skin Conductance
          Response (SCR).

        * ``"SCR_Onsets"``: the samples at which the onsets of the peaks occur, marked as "1" in a
          list of zeros.

        * ``"SCR_Peaks"``: the samples at which the peaks occur, marked as "1" in a list of zeros.

        * ``"SCR_Height"``: the SCR amplitude of the signal including the Tonic component. Note that
          cumulative effects of close-occurring SCRs might lead to an underestimation of the
          amplitude.

        * ``"SCR_Amplitude"``: the SCR amplitude of the signal excluding the Tonic component.

        * ``"SCR_RiseTime"``: the time taken for SCR onset to reach peak amplitude within the SCR.

        * ``"SCR_Recovery"``: the samples at which SCR peaks recover (decline) to half amplitude,
          marked  as "1" in a list of zeros.
    info : dict
        A dictionary containing the information of each SCR peak (see :func:`eda_findpeaks`),
        as well as the signals' sampling rate.

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_findpeaks, eda_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
      signals, info = nk.eda_process(eda_signal, sampling_rate=1000)

      @savefig p_eda_process.png scale=100%
      nk.eda_plot(signals, info)
      @suppress
      plt.close()

    """
    # Sanitize input
    eda_signal = nk.signal_sanitize(eda_signal)
    methods = nk.eda_methods(sampling_rate=sampling_rate, method=method, **kwargs)

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

#%%

fname = "C:/Users/Tomas Gil/Desktop/Tesis de Licenciatura/Sesiones/S19/S19_DMT_Session2_DMT.vhdr"
# fname = "C:/Users/Tomas Gil/Desktop/Tesis de Licenciatura/Sesiones/S19/S19_RS_Session1_EC.vhdr"

raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Separo los dataframes en arrays que pueda leer Neurokit, es decir, ECG, EDA, RSP

edap = df_data['GSR'] #EDA posta (para ponerlo despues sin problema)
ecgp = df_data['ECG'] #ECG posta
resp = df_data['RESP'] #RESP

interv = []
for i in range(0, len(df_data['time'])-1):
    t = df_data['time'][i+1] - df_data['time'][i]
    interv.append(t)
    
dt = np.mean(interv)
### se que lo dice el vhdr pero no se si confiar en el sistema de primera

#%% Leo los datos con NeuroKit (aunque RSP analiza con Harrison 2021)

df_eda, info_eda = eda_process(edap, sampling_rate=1/dt) #neurokit method

