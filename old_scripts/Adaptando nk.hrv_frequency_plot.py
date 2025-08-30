# -*- coding: utf-8 -*-
#####################################################################
#### PONER TODAS LAS DEPENDENCIAS QUE ME FALTAN DE NEUROKIT      ####
#####################################################################
from warnings import warn

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
import scipy

import neurokit2 as nk

#%% Acá está la definición de intervals_utils que tiene a intervals sanitize

def _intervals_successive(intervals, intervals_time=None, thresh_unequal=10, n_diff=1):

    # Convert to numpy array
    intervals = np.array(intervals)

    if intervals_time is None:
        intervals_time = np.nancumsum(intervals / 1000)
    else:
        intervals_time = np.array(intervals_time).astype(float)

    intervals_time[np.isnan(intervals)] = np.nan

    diff_intervals_time_ms = np.diff(intervals_time, n=n_diff) * 1000

    abs_error_intervals_ref_time = abs(
        diff_intervals_time_ms - np.diff(intervals[1:], n=n_diff - 1)
    )

    successive_intervals = abs_error_intervals_ref_time <= thresh_unequal

    return np.array(successive_intervals)


def _intervals_time_uniform(intervals_time, decimals=3):
    """Check whether timestamps are uniformly spaced.

    Useful for determining whether intervals have been interpolated.

    Parameters
    ----------
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    decimals : int, optional
        The precision of the timestamps. The default is 3.

    Returns
    ----------
    bool
        Whether the timestamps are uniformly spaced

    """
    return len(np.unique(np.round(np.diff(intervals_time), decimals=decimals))) == 1


def _intervals_sanitize(intervals, intervals_time=None, remove_missing=True):
    """**Interval input sanitization**

    Parameters
    ----------
    intervals : list or array
        List or numpy array of intervals, in milliseconds.
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    remove_missing : bool, optional
        Whether to remove NaNs and infinite values from intervals and timestamps.
        The default is True.

    Returns
    -------
    intervals : array
        Sanitized intervals, in milliseconds.
    intervals_time : array
        Sanitized timestamps corresponding to intervals, in seconds.
    intervals_missing : bool
        Whether there were missing intervals detected.

    Examples
    ---------
    .. ipython:: python

      import neurokit2 as nk
      ibi = [500, 400, 700, 500, 300, 800, 500]
      ibi, ibi_time, intervals_missing = intervals_sanitize(ibi)

    """
    if intervals is None:
        return None, None
    else:
        # Ensure that input is numpy array
        intervals = np.array(intervals)
    if intervals_time is None:
        # Impute intervals with median in case of missing values to calculate timestamps
        imputed_intervals = np.where(
            np.isnan(intervals), np.nanmedian(intervals, axis=0), intervals
        )
        # Compute the timestamps of the intervals in seconds
        intervals_time = np.nancumsum(imputed_intervals / 1000)
    else:
        # Ensure that input is numpy array
        intervals_time = np.array(intervals_time)

        # Confirm that timestamps are in seconds
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)

        if np.all(successive_intervals) is False:
            # Check whether intervals appear to be interpolated
            if not _intervals_time_uniform(intervals_time):
                # If none of the differences between timestamps match
                # the length of the R-R intervals in seconds,
                # try converting milliseconds to seconds
                converted_successive_intervals = _intervals_successive(
                    intervals, intervals_time=intervals_time / 1000
                )

                # Check if converting to seconds increased the number of differences
                # between timestamps that match the length of the R-R intervals in seconds
                if len(converted_successive_intervals[converted_successive_intervals]) > len(
                    successive_intervals[successive_intervals]
                ):
                    # Assume timestamps were passed in milliseconds and convert to seconds
                    intervals_time = intervals_time / 1000

    intervals_missing = _intervals_missing(intervals, intervals_time)

    if remove_missing:
        # Remove NaN R-R intervals, if any
        intervals_time = intervals_time[np.isfinite(intervals)]
        intervals = intervals[np.isfinite(intervals)]
    return intervals, intervals_time, intervals_missing


def _intervals_missing(intervals, intervals_time=None):
    if len(intervals[np.isfinite(intervals)]) < len(intervals):
        return True
    elif intervals_time is not None:
        successive_intervals = _intervals_successive(intervals, intervals_time=intervals_time)
        if not np.all(successive_intervals) and np.any(successive_intervals):
            # Check whether intervals appear to be interpolated
            if not _intervals_time_uniform(intervals_time):
                return True
    return False


def _intervals_time_to_sampling_rate(intervals_time, central_measure="mean"):
    """Get sampling rate from timestamps.

    Useful for determining sampling rate used to interpolate intervals.

    Parameters
    ----------
    intervals_time : list or array, optional
        List or numpy array of timestamps corresponding to intervals, in seconds.
    central_measure : str, optional
        The measure of central tendancy used. Either ``"mean"`` (default), ``"median"``, or ``"mode"``.

    Returns
    ----------
    bool
        Whether the timestamps are uniformly spaced

    """
    if central_measure == "mean":
        sampling_rate = float(1 / np.nanmean(np.diff(intervals_time)))
    elif central_measure == "median":
        sampling_rate = float(1 / np.nanmedian(np.diff(intervals_time)))
    else:
        sampling_rate = float(1 / scipy.stats.mode(np.diff(intervals_time)))
    return sampling_rate
#%% Para tener el intervals_to_peaks del siguiente y el intervals_sanitize, uso este codigo

import numpy as np

# from .intervals_utils import _intervals_sanitize, _intervals_successive


def intervals_to_peaks(intervals, intervals_time=None, sampling_rate=1000):
    
    if intervals is None:
        return None

    intervals, intervals_time, intervals_missing = _intervals_sanitize(
        intervals, intervals_time=intervals_time, remove_missing=True
    )

    if intervals_missing:
        # Check for non successive intervals in case of missing data
        non_successive_indices = np.arange(1, len(intervals_time))[
            np.invert(_intervals_successive(intervals, intervals_time))
        ]
    else:
        non_successive_indices = np.array([]).astype(int)
    # The number of peaks should be the number of intervals
    # plus one extra at the beginning of each group of successive intervals
    # (with no missing data there should be N_intervals + 1 peaks)
    to_insert_indices = np.concatenate((np.array([0]), non_successive_indices))

    times_to_insert = intervals_time[to_insert_indices] - intervals[to_insert_indices] / 1000

    peaks_time = np.sort(np.concatenate((intervals_time, times_to_insert)))
    # convert seconds to sample indices
    peaks = peaks_time * sampling_rate

    return np.array([int(np.round(i)) for i in peaks])

#%% Esta parte reemplaza el hrv_utils que pide format input que me falta

# from .intervals_to_peaks import intervals_to_peaks
# from .intervals_utils import _intervals_sanitize


def _hrv_get_rri(peaks=None, sampling_rate=1000):
    if peaks is None:
        return None, None, None
    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri = np.diff(peaks) / sampling_rate * 1000
    rri, rri_time, rri_missing = _intervals_sanitize(rri)
    return rri, rri_time, rri_missing


def _hrv_format_input(peaks=None, sampling_rate=1000, output_format="intervals"):

    if isinstance(peaks, tuple):
        rri, rri_time, rri_missing, sampling_rate = _hrv_sanitize_tuple(peaks, sampling_rate=sampling_rate)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        rri, rri_time, rri_missing, sampling_rate = _hrv_sanitize_dict_or_df(peaks, sampling_rate=sampling_rate)
    else:
        peaks = _hrv_sanitize_peaks(peaks)
        rri, rri_time, rri_missing = _hrv_get_rri(peaks, sampling_rate=sampling_rate)
    if output_format == "intervals":
        return rri, rri_time, rri_missing
    elif output_format == "peaks":
        return (
            intervals_to_peaks(rri, intervals_time=rri_time, sampling_rate=sampling_rate),
            sampling_rate,
        )


# =============================================================================
# Internals
# =============================================================================
def _hrv_sanitize_tuple(peaks, sampling_rate=1000):

    # Get sampling rate
    info = [i for i in peaks if isinstance(i, dict)]
    sampling_rate = info[0]["sampling_rate"]

    # Detect actual sampling rate
    if len(info) < 1:
        peaks, sampling_rate = peaks[0], peaks[1]

    # Get peaks
    if isinstance(peaks[0], (dict, pd.DataFrame)):
        try:
            peaks = _hrv_sanitize_dict_or_df(peaks[0])
        except NameError:
            if isinstance(peaks[1], (dict, pd.DataFrame)):
                try:
                    peaks = _hrv_sanitize_dict_or_df(peaks[1])
                except NameError:
                    peaks = _hrv_sanitize_peaks(peaks[1])
            else:
                peaks = _hrv_sanitize_peaks(peaks[0])

    rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate)

    return rri, rri_time, rri_missing, sampling_rate


def _hrv_sanitize_dict_or_df(peaks, sampling_rate=None):

    # Get columns
    if isinstance(peaks, dict):
        cols = np.array(list(peaks.keys()))
        if "sampling_rate" in cols:
            sampling_rate = peaks["sampling_rate"]
    elif isinstance(peaks, pd.DataFrame):
        cols = peaks.columns.values

    # check whether R-R intervals were passed rather than peak indices
    if "RRI" in cols:
        rri = peaks["RRI"]
        if "RRI_Time" in cols:
            rri_time = peaks["RRI_Time"]
        else:
            rri_time = None
        rri, rri_time, rri_missing = _intervals_sanitize(rri, intervals_time=rri_time)
        return rri, rri_time, rri_missing, sampling_rate

    cols = cols[["Peak" in s for s in cols]]

    if len(cols) > 1:
        cols = cols[[("ECG" in s) or ("PPG" in s) for s in cols]]

    if len(cols) == 0:
        raise NameError(
            "NeuroKit error: hrv(): Wrong input, ",
            "we couldn't extract R-peak indices. ",
            "You need to provide a list of R-peak indices.",
        )

    peaks = _hrv_sanitize_peaks(peaks[cols[0]])

    if sampling_rate is not None:
        rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks, sampling_rate=sampling_rate)
    else:
        rri, rri_time, rri_missing = _hrv_get_rri(peaks=peaks)
    return rri, rri_time, rri_missing, sampling_rate


def _hrv_sanitize_peaks(peaks):

    if isinstance(peaks, pd.Series):
        peaks = peaks.values

    if len(np.unique(peaks)) == 2:
        if np.all(np.unique(peaks) == np.array([0, 1])):
            peaks = np.where(peaks == 1)[0]

    if isinstance(peaks, list):
        peaks = np.array(peaks)

    if peaks is not None:
        if isinstance(peaks, tuple):
            if any(np.diff(peaks[0]) < 0):  # not continuously increasing
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )
        else:
            if any(np.diff(peaks) < 0):
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )

    return peaks

#%% Acá defino signal_psd que después lo necesito


def signal_psd(
    signal,
    sampling_rate=1000,
    method="welch",
    show=False,
    normalize=True,
    min_frequency="default",
    max_frequency=np.inf,
    window=None,
    window_type="hann",
    order=16,
    order_criteria="KIC",
    order_corrected=True,
    silent=True,
    t=None,
    **kwargs,
):
    
    # Constant Detrend
    signal = signal - np.mean(signal)

    # Sanitize method name
    method = method.lower()

    # Sanitize min_frequency
    N = len(signal)
    if isinstance(min_frequency, str):
        if sampling_rate is None:
            # This is to compute min_frequency if both min_frequency and sampling_rate are not provided (#800)
            min_frequency = (2 * np.median(np.diff(t))) / (N / 2)  # for high frequency resolution
        else:
            min_frequency = (2 * sampling_rate) / (N / 2)  # for high frequency resolution

    # MNE
    if method in ["multitaper", "multitapers", "mne"]:
        frequency, power = _signal_psd_multitaper(
            signal, sampling_rate=sampling_rate, min_frequency=min_frequency, max_frequency=max_frequency,
        )

    # FFT (Numpy)
    elif method in ["fft"]:
        frequency, power = _signal_psd_fft(signal, sampling_rate=sampling_rate, **kwargs)

    # Lombscargle (AtroPy)
    elif method.lower() in ["lombscargle", "lomb"]:
        frequency, power = _signal_psd_lomb(
            signal, sampling_rate=sampling_rate, min_frequency=min_frequency, max_frequency=max_frequency, t=t
        )

    # Method that are using a window
    else:
        # Define window length
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize min_frequency

        if window is not None:
            nperseg = int(window * sampling_rate)
        else:
            # to capture at least 2 cycles of min_frequency
            nperseg = int((2 / min_frequency) * sampling_rate)

        # in case duration of recording is not sufficient
        if nperseg > N / 2:
            if silent is False:
                warn(
                    "The duration of recording is too short to support a"
                    " sufficiently long window for high frequency resolution."
                    " Consider using a longer recording or increasing the `min_frequency`",
                    category = nk.misc.NeuroKitWarning,
                )
            nperseg = int(N / 2)

        # Welch (Scipy)
        if method.lower() in ["welch"]:
            frequency, power = _signal_psd_welch(
                signal, sampling_rate=sampling_rate, nperseg=nperseg, window_type=window_type, **kwargs,
            )

        # BURG
        elif method.lower() in ["burg", "pburg", "spectrum"]:
            frequency, power = _signal_psd_burg(
                signal,
                sampling_rate=sampling_rate,
                order=order,
                criteria=order_criteria,
                corrected=order_corrected,
                side="one-sided",
                nperseg=nperseg,
            )

    # Normalize
    if normalize is True:
        power /= np.max(power)

    # Store results
    data = pd.DataFrame({"Frequency": frequency, "Power": power})

    # Filter
    data = data.loc[np.logical_and(data["Frequency"] >= min_frequency, data["Frequency"] <= max_frequency)]
    #    data["Power"] = 10 * np.log(data["Power"])

    if show is True:
        ax = data.plot(x="Frequency", y="Power", title="Power Spectral Density (" + str(method) + " method)")
        ax.set(xlabel="Frequency (Hz)", ylabel="Spectrum")

    return data


# =============================================================================
# Multitaper method
# =============================================================================


def _signal_psd_fft(signal, sampling_rate=1000, n=None):
    # Power-spectrum density (PSD)
    power = np.abs(np.fft.rfft(signal, n=n)) ** 2
    frequency = np.linspace(0, sampling_rate / 2, len(power))
    return frequency, power


# =============================================================================
# Multitaper method
# =============================================================================


def _signal_psd_multitaper(signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf):
    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_psd(): the 'mne'",
            " module is required for the 'mne' method to run.",
            " Please install it first (`pip install mne`).",
        ) from e

    power, frequency = mne.time_frequency.psd_array_multitaper(
        signal,
        sfreq=sampling_rate,
        fmin=min_frequency,
        fmax=max_frequency,
        adaptive=True,
        normalization="full",
        verbose=False,
    )

    return frequency, power


# =============================================================================
# Welch method
# =============================================================================


def _signal_psd_welch(signal, sampling_rate=1000, nperseg=None, window_type="hann", **kwargs):
    if nperseg is not None:
        nfft = int(nperseg * 2)
    else:
        nfft = None

    frequency, power = scipy.signal.welch(
        signal,
        fs=sampling_rate,
        scaling="density",
        detrend=False,
        nfft=nfft,
        average="mean",
        nperseg=nperseg,
        window=window_type,
        **kwargs,
    )

    return frequency, power


# =============================================================================
# Lomb method
# =============================================================================


def _signal_psd_lomb(signal, sampling_rate=1000, min_frequency=0, max_frequency=np.inf, t=None):

    try:
        import astropy.timeseries

        if t is None:
            if max_frequency == np.inf:
                max_frequency = sampling_rate / 2  # sanitize highest frequency
            t = np.arange(len(signal)) / sampling_rate
            frequency, power = astropy.timeseries.LombScargle(t, signal, normalization="psd").autopower(
                minimum_frequency=min_frequency, maximum_frequency=max_frequency
            )
        else:
            # determine maximum frequency with astropy defaults for unevenly spaced data
            # https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html#astropy.timeseries.LombScargle.autopower
            frequency, power = astropy.timeseries.LombScargle(t, signal, normalization="psd").autopower(
                minimum_frequency=min_frequency
            )

    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_psd(): the 'astropy'",
            " module is required for the 'lomb' method to run.",
            " Please install it first (`pip install astropy`).",
        ) from e

    return frequency, power


# =============================================================================
# Burg method
# =============================================================================


def _signal_psd_burg(
    signal, sampling_rate=1000, order=16, criteria="KIC", corrected=True, side="one-sided", nperseg=None,
):

    nfft = int(nperseg * 2)
    ar, rho, _ = _signal_arma_burg(signal, order=order, criteria=criteria, corrected=corrected)
    psd = _signal_psd_from_arma(ar=ar, rho=rho, sampling_rate=sampling_rate, nfft=nfft, side=side)

    # signal is real, not complex
    if nfft % 2 == 0:
        power = psd[0 : int(nfft / 2 + 1)] * 2
    else:
        power = psd[0 : int((nfft + 1) / 2)] * 2

    # angular frequencies, w
    # for one-sided psd, w spans [0, pi]
    # for two-sdied psd, w spans [0, 2pi)
    # for dc-centered psd, w spans (-pi, pi] for even nfft, (-pi, pi) for add nfft
    if side == "one-sided":
        w = np.pi * np.linspace(0, 1, len(power))
    #    elif side == "two-sided":
    #        w = np.pi * np.linspace(0, 2, len(power), endpoint=False)  #exclude last point
    #    elif side == "centerdc":
    #        if nfft % 2 == 0:
    #            w = np.pi * np.linspace(-1, 1, len(power))
    #        else:
    #            w = np.pi * np.linspace(-1, 1, len(power) + 1, endpoint=False)  # exclude last point
    #            w = w[1:]  # exclude first point (extra)

    frequency = (w * sampling_rate) / (2 * np.pi)

    return frequency, power


def _signal_arma_burg(signal, order=16, criteria="KIC", corrected=True):

    # Sanitize order and signal
    N = len(signal)
    if order <= 0.0:
        raise ValueError("Order must be > 0")
    if order > N:
        raise ValueError("Order must be less than length signal minus 2")
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # Initialisation
    # rho is variance of driving white noise process (prediction error)
    rho = sum(abs(signal) ** 2.0) / float(N)
    denominator = rho * 2.0 * N

    ar = np.zeros(0, dtype=complex)  # AR parametric signal model estimate
    ref = np.zeros(0, dtype=complex)  # vector K of reflection coefficients (parcor coefficients)
    ef = signal.astype(complex)  # forward prediction error
    eb = signal.astype(complex)  # backward prediction error
    temp = 1.0

    # Main recursion

    for k in range(0, order):

        # calculate the next order reflection coefficient
        numerator = sum([ef[j] * eb[j - 1].conjugate() for j in range(k + 1, N)])
        denominator = temp * denominator - abs(ef[k]) ** 2 - abs(eb[N - 1]) ** 2
        kp = -2.0 * numerator / denominator

        # Update the prediction error
        temp = 1.0 - abs(kp) ** 2.0
        new_rho = temp * rho

        if criteria is not None:
            # k=k+1 because order goes from 1 to P whereas k starts at 0.
            residual_new = _criteria(criteria=criteria, N=N, k=k + 1, rho=new_rho, corrected=corrected)
            if k == 0:
                residual_old = 2.0 * abs(residual_new)

            # Stop as criteria has reached
            if residual_new > residual_old:
                break

            # This should be after the criteria
            residual_old = residual_new
        rho = new_rho
        if rho <= 0:
            raise ValueError(f"Found a negative value (expected positive strictly) {rho}. Decrease the order.")

        ar = np.resize(ar, ar.size + 1)
        ar[k] = kp
        if k == 0:
            for j in range(N - 1, k, -1):
                ef_previous = ef[j]  # previous value
                ef[j] = ef_previous + kp * eb[j - 1]  # Eq. (8.7)
                eb[j] = eb[j - 1] + kp.conjugate() * ef_previous

        else:
            # Update the AR coeff
            khalf = (k + 1) // 2  # khalf must be an integer
            for j in range(0, khalf):
                ar_previous = ar[j]  # previous value
                ar[j] = ar_previous + kp * ar[k - j - 1].conjugate()  # Eq. (8.2)
                if j != k - j - 1:
                    ar[k - j - 1] = ar[k - j - 1] + kp * ar_previous.conjugate()  # Eq. (8.2)

            # Update the forward and backward prediction errors
            for j in range(N - 1, k, -1):
                ef_previous = ef[j]  # previous value
                ef[j] = ef_previous + kp * eb[j - 1]  # Eq. (8.7)
                eb[j] = eb[j - 1] + kp.conjugate() * ef_previous

        # save the reflection coefficient
        ref = np.resize(ref, ref.size + 1)
        ref[k] = kp

    return ar, rho, ref


# =============================================================================
# Utilities
# =============================================================================


def _criteria(criteria=None, N=None, k=None, rho=None, corrected=True):
    
    if criteria == "AIC":
        if corrected is True:
            residual = np.log(rho) + 2.0 * (k + 1) / (N - k - 2)
        else:
            residual = N * np.log(np.array(rho)) + 2.0 * (np.array(k) + 1)

    elif criteria == "KIC":
        if corrected is True:
            residual = np.log(rho) + k / N / (N - k) + (3.0 - (k + 2.0) / N) * (k + 1.0) / (N - k - 2.0)
        else:
            residual = np.log(rho) + 3.0 * (k + 1.0) / float(N)

    elif criteria == "FPE":
        fpe = rho * (N + k + 1.0) / (N - k - 1)
        return fpe

    elif criteria == "MDL":
        mdl = N * np.log(rho) + k * np.log(N)
        return mdl

    return residual


def _signal_psd_from_arma(ar=None, ma=None, rho=1.0, sampling_rate=1000, nfft=None, side="one-sided"):

    if ar is None and ma is None:
        raise ValueError("Either AR or MA model must be provided")

    psd = np.zeros(nfft, dtype=complex)

    if ar is not None:
        ip = len(ar)
        den = np.zeros(nfft, dtype=complex)
        den[0] = 1.0 + 0j
        for k in range(0, ip):
            den[k + 1] = ar[k]
        denf = np.fft.fft(den, nfft)

    if ma is not None:
        iq = len(ma)
        num = np.zeros(nfft, dtype=complex)
        num[0] = 1.0 + 0j
        for k in range(0, iq):
            num[k + 1] = ma[k]
        numf = np.fft.fft(num, nfft)

    if ar is not None and ma is not None:
        psd = rho / sampling_rate * abs(numf) ** 2.0 / abs(denf) ** 2.0
    elif ar is not None:
        psd = rho / sampling_rate / abs(denf) ** 2.0
    elif ma is not None:
        psd = rho / sampling_rate * abs(numf) ** 2.0

    psd = np.real(psd)  # The PSD is a twosided PSD.

    # convert to one-sided
    if side == "one-sided":
        assert len(psd) % 2 == 0
        one_side_psd = np.array(psd[0 : len(psd) // 2 + 1]) * 2.0
        one_side_psd[0] /= 2.0
        #        one_side_psd[-1] = psd[-1]
        psd = one_side_psd

    # convert to centerdc
    elif side == "centerdc":
        first_half = psd[0 : len(psd) // 2]
        second_half = psd[len(psd) // 2 :]
        rotate_second_half = second_half[-1:] + second_half[:-1]
        center_psd = np.concatenate((rotate_second_half, first_half))
        center_psd[0] = psd[-1]
        psd = center_psd

    return psd


#%% Esta parte reemplaza signal_power

# from .signal_psd import signal_psd


def signal_power(
    signal,
    frequency_band,
    sampling_rate=1000,
    continuous=False,
    show=False,
    normalize=True,
    **kwargs,
):

    if continuous is False:
        out, psd = _signal_power_instant(
            signal,
            frequency_band,
            sampling_rate=sampling_rate,
            show=show,
            normalize=normalize,
            **kwargs,
        )
    else:
        out, psd = _signal_power_continuous(signal, frequency_band, sampling_rate=sampling_rate)

    out = pd.DataFrame.from_dict(out, orient="index").T

    return out, psd


# =============================================================================
# Instant
# =============================================================================


def _signal_power_instant(
    signal,
    frequency_band,
    sampling_rate=1000,
    show=False,
    normalize=True,
    order_criteria="KIC",
    **kwargs,
):
    # Sanitize frequency band
    if isinstance(frequency_band[0], (int, float)):
        frequency_band = [frequency_band]  # put in list to iterate on

    #  Get min-max frequency
    min_freq = min([band[0] for band in frequency_band])
    max_freq = max([band[1] for band in frequency_band])

    # Get PSD
    psd = signal_psd(
        signal,
        sampling_rate=sampling_rate,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs,
    )

    psd = psd[(psd["Frequency"] >= min_freq) & (psd["Frequency"] <= max_freq)]
    
    out = {}
    for band in frequency_band:
        power = _signal_power_instant_compute(psd, band)
        out[f"Hz_{band[0]}_{band[1]}"] = power

    if show:
        _signal_power_instant_plot(psd, out, frequency_band)
    return out, psd


def _signal_power_instant_compute(psd, band):
    """Also used in other instances"""
    where = (psd["Frequency"] >= band[0]) & (psd["Frequency"] < band[1])
    power = np.trapz(y=psd["Power"][where], x=psd["Frequency"][where])
    return np.nan if power == 0.0 else power


def _signal_power_instant_plot(psd, out, frequency_band, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # Sanitize signal
    if isinstance(frequency_band[0], int):
        if len(frequency_band) > 2:
            print(
                "NeuroKit error: signal_power(): The `frequency_band` argument must be a list of tuples"
                " or a tuple of 2 integers"
            )
        else:
            frequency_band = [tuple(i for i in frequency_band)]

    freq = np.array(psd["Frequency"])
    power = np.array(psd["Power"])
    
    # Get indexes for different frequency band
    frequency_band_index = []
    for band in frequency_band:
        indexes = np.logical_and(
            psd["Frequency"] >= band[0], psd["Frequency"] < band[1]
        )  # pylint: disable=E1111
        frequency_band_index.append(np.array(indexes))

    labels = list(out.keys())
    # Reformat labels if of the pattern "Hz_X_Y"
    if len(labels[0].split("_")) == 3:
        labels = [i.split("_") for i in labels]
        labels = [f"{i[1]}-{i[2]} Hz" for i in labels]

    # Get cmap
    cmap = matplotlib.colormaps.get_cmap("Set1")
    colors = cmap.colors
    colors = (
        colors[3],
        colors[1],
        colors[2],
        colors[4],
        colors[0],
        colors[5],
        colors[6],
        colors[7],
        colors[8],
    )  # manually rearrange colors
    colors = colors[0 : len(frequency_band_index)]

    # Plot
    ax.set_title("Power Spectral Density (PSD) for Frequency Domains")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spectrum (ms2/Hz)")

    ax.fill_between(freq, 0, power, color="lightgrey")

    for band_index, label, i in zip(frequency_band_index, labels, colors):
        ax.fill_between(freq[band_index], 0, power[band_index], label=label, color=i)
        ax.legend(prop={"size": 10}, loc="best")

    return fig


# =============================================================================
# Continuous
# =============================================================================


def _signal_power_continuous(signal, frequency_band, sampling_rate=1000):

    out = {}
    if isinstance(frequency_band[0], (list, tuple)):
        for band in frequency_band:
            out.update(_signal_power_continuous_get(signal, band, sampling_rate))
    else:
        out.update(_signal_power_continuous_get(signal, frequency_band, sampling_rate))
    return out


def _signal_power_continuous_get(signal, frequency_band, sampling_rate=1000, precision=20):

    try:
        import mne
    except ImportError as e:
        raise ImportError(
            "NeuroKit error: signal_power(): the 'mne'",
            "module is required. ",
            "Please install it first (`pip install mne`).",
        ) from e  # explicitly raise error from ImportError exception

    out = mne.time_frequency.tfr_array_morlet(
        [[signal]],
        sfreq=sampling_rate,
        freqs=np.linspace(frequency_band[0], frequency_band[1], precision),
        output="power",
    )
    power = np.mean(out[0][0], axis=0)

    out = {}
    out[f"{frequency_band[0]:.2f}-{frequency_band[1]:.2f}Hz"] = power  # use literal string format
    return out

#%% Acá creo hrv_frequency

def hrv_frequency(
    peaks,
    sampling_rate=1000,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 100),
    psd_method="welch",
    show=False,
    silent=True,
    normalize=True,
    order_criteria=None,
    interpolation_rate=250,
    **kwargs
):
    
    # Sanitize input
    # If given peaks, compute R-R intervals (also referred to as NN) in milliseconds
    rri, rri_time, _ = _hrv_format_input(peaks, sampling_rate=sampling_rate)

    # Process R-R intervals (interpolated at 100 Hz by default)
    rri, rri_time, sampling_rate = nk.intervals_process(
        rri, intervals_time=rri_time, interpolate=True, interpolation_rate=interpolation_rate, **kwargs
    )

    if interpolation_rate is None:
        t = rri_time
    else:
        t = None

    frequency_band = [ulf, vlf, lf, hf, vhf]

    # Find maximum frequency
    max_frequency = np.max([np.max(i) for i in frequency_band])

    power, psd = signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=max_frequency,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        t=t,
        **kwargs
    )
    
    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]

    out = power.to_dict(orient="index")[0]
    out_bands = out.copy()  # Components to be entered into plot

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                warn(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                    " Its power is returned as zero.",
                    category = nk.misc.NeuroKitWarning,
                )

    # Normalized
    total_power = np.nansum(power.values)
    out["TP"] = total_power
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])  # pylint: disable=E1111
    
    # Add rri_time to the output dictionary
    lista_rri = rri_time.tolist()
    rr_columna = pd.DataFrame({'RRI_Time': lista_rri})

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")

    out = pd.concat([out, rr_columna], axis=1)
    
    # Plot
    if show:
        _hrv_frequency_show(
            rri,
            out_bands,
            ulf=ulf,
            vlf=vlf,
            lf=lf,
            hf=hf,
            vhf=vhf,
            sampling_rate=sampling_rate,
            psd_method=psd_method,
            order_criteria=order_criteria,
            normalize=normalize,
            max_frequency=max_frequency,
            t=t,
        )
    return out, psd#, plot


def _hrv_frequency_show(
    rri,
    out_bands,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    sampling_rate=1000,
    psd_method="welch",
    order_criteria=None,
    normalize=True,
    max_frequency=0.5,
    t=None,
    **kwargs
):

    if "ax" in kwargs:
        ax = kwargs.get("ax")
        kwargs.pop("ax")
    else:
        __, ax = plt.subplots()

    frequency_band = [ulf, vlf, lf, hf, vhf]

    # Compute sampling rate for plot windows
    if sampling_rate is None:
        med_sampling_rate = np.median(np.diff(t))  # This is just for visualization purposes (#800)
    else:
        med_sampling_rate = sampling_rate

    for i in range(len(frequency_band)):  # pylint: disable=C0200
        min_frequency = frequency_band[i][0]
        if min_frequency == 0:
            min_frequency = 0.001  # sanitize lowest frequency

        window_length = int((2 / min_frequency) * med_sampling_rate)
        if window_length <= len(rri) / 2:
            break

    psd = signal_psd(
        rri,
        sampling_rate=sampling_rate,
        show=False,
        min_frequency=min_frequency,
        method=psd_method,
        max_frequency=max_frequency,
        order_criteria=order_criteria,
        normalize=normalize,
        t=t,
    )

    _signal_power_instant_plot(psd, out_bands, frequency_band, ax=ax)
    
    # return psd
    
#%% Uso MNE para descargar la data
import mne

#%%

fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/DMT_2/S19/S19_DMT_Session2_DMT.vhdr"
# fname = "C:/Users/tomas/OneDrive/Desktop/Tesis de Licenciatura/Sesiones/Reposo/S19/S19_RS_Session2_EC.vhdr"

raw_data = mne.io.read_raw_brainvision(fname)
data = raw_data.load_data()

df_data = data.to_data_frame()
print(df_data.columns)

#%% Generalizo el intervalo de sampleo, pongo lo que tenga el instrumental para no perder info

######### ¿Me conviene el promedio, el max o el minimo? ##############

interv = []
for i in range(0, len(df_data['time'])-1):
    t = df_data['time'][i+1] - df_data['time'][i]
    interv.append(t)
    
dt = np.mean(interv)
### se que lo dice el vhdr pero no se si confiar en el sistema de primera

ecgp = df_data['ECG'] #ECG posta

# Find peaks
df_ecg, info_ecg = nk.ecg_process(ecgp, sampling_rate=1/dt) #neurokit method

# Compute HRV indices using method="welch"
#ME SIRVE PARA DIFERENCIAR LAS DEL PLOT DE LAS CALCULADAS POSTA
# hrv_welch, psd, datos_plot = hrv_frequency(df_ecg['ECG_R_Peaks'], sampling_rate=1/dt, show=True, psd_method="welch")

# USO ESTO PARA FACILITAR TRABAJO
hrv_welch, psd = hrv_frequency(df_ecg['ECG_R_Peaks'], sampling_rate=1/dt, show=True, psd_method="welch")


#%% Voy a hacer el hrv_frequency para varios partes de datos

n_divs = 100
longitud_segmento = int(len(df_ecg['ECG_R_Peaks'])/n_divs) #calculo los primeros 10 picos
inicio = 0 

psd_intervalos = []
frec_intervalos = []
tiempo_intervalos = []


while inicio + longitud_segmento <= (len(df_ecg['ECG_R_Peaks'])):

    hrv_welch, psd = hrv_frequency(df_ecg['ECG_R_Peaks'][inicio:inicio + longitud_segmento], sampling_rate=1/dt, show=False, psd_method="welch")
    frec_intervalos.append(psd['Frequency'].tolist())
    psd_intervalos.append(psd['Power'].tolist())
    tiempo_intervalos.append(df_data['time'].tolist(),)
    
    inicio += longitud_segmento

# plt.figure(2)
# plt.pcolormesh(hrv_welch['RRI_Time'], psd['Frequency'], psd['Power'], shading = "auto")
# plt.colorbar(label = "Power Spectrum")
# plt.title('Time-Frequency Plot')
# plt.xlabel('Tiempo asociado(s)')
# plt.ylabel('Frecuencia de intervalos')
# plt.show()


