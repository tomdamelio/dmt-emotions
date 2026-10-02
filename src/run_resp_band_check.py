# -*- coding: utf-8 -*-
"""Respiratory fundamental frequency against the HRV frequency bands.

Supports the statement in Methods that frequency-domain ratios such as LF/HF
were not computed because respiratory frequency fell below the conventional
high-frequency band (0.15-0.4 Hz) in most participants and was often non-periodic
during DMT, and the numbers given for it in the reply to Reviewer 3 (3.1 ii).

Method (per window, 15-s steps, on the cleaned belt signal RSP_Clean):
  - high-pass 0.04 Hz (removes slow belt drift), decimation 250 -> 25 Hz;
  - normalised autocorrelation; the respiratory period is the FIRST prominent
    local maximum (prominence 0.05) at lags of 1.5-20 s (3 to 40 breaths/min),
    since the highest peak can be a multiple of the period;
  - r_max, the height of that maximum, indexes periodicity: windows with
    r_max < 0.30 are treated as non-periodic (irregular breathing or artefact)
    and get no frequency.
Peak-based rates (NeuroKit RSP_Rate) and Welch spectral peaks were tried first
and rejected: both are biased by non-sinusoidal breaths and irregular segments.

Sample: participants with valid recordings in all three modalities (ECG, EDA
and respiration), the subsample used by the coupling analyses (n = 7).
Main window: 60 s. Sensitivity: 30 s and 120 s.

Outputs -> results/resp/band_check/
    per_window.csv            one row per participant x recording x window length x window
    summary_by_subject.csv    per participant, state and window length
    summary_by_state.csv      pooled over participants, per state and window length
    report.txt                the numbers quoted in the manuscript and the letter

Usage:
    micromamba run -n dmt-emotions python src/run_resp_band_check.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, decimate, filtfilt, find_peaks

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (DERIVATIVES_DATA, SUJETOS_VALIDADOS_ECG,  # noqa: E402
                    SUJETOS_VALIDADOS_EDA, SUJETOS_VALIDADOS_RESP)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTDIR = os.path.join(REPO, "results", "resp", "band_check")
RESP_DIR = os.path.join(DERIVATIVES_DATA, "phys", "resp")

# ---- fixed parameters -------------------------------------------------------
FS, FS_D, Q = 250.0, 25.0, 10          # sampling rate, decimated rate, factor
STEP_S = 15.0
WINDOWS_S = (60.0, 30.0, 120.0)         # main window first, then sensitivity
MAIN_WINDOW_S = 60.0
LAG_MIN_S, LAG_MAX_S = 1.5, 20.0
R_THRESH = 0.30
HF_LOW = 0.15                           # lower edge of the HRV high-frequency band

SUBJECTS = sorted(set(SUJETOS_VALIDADOS_ECG) & set(SUJETOS_VALIDADOS_EDA)
                  & set(SUJETOS_VALIDADOS_RESP))
_B, _A = butter(2, 0.04 / (FS / 2), btype="high")


def recordings(subject: str):
    """[(state, dose, session, path)] for one participant."""
    out = []
    for dose in ("high", "low"):
        for path in glob.glob(os.path.join(RESP_DIR, f"dmt_{dose}", f"{subject}_*.csv")):
            m = re.match(rf"{subject}_(dmt|rs)_(session\d)_{dose}\.csv$", os.path.basename(path))
            if m:
                out.append((m.group(1).upper(), dose, m.group(2), path))
    return sorted(out)


def fundamental(segment: np.ndarray):
    """(frequency in Hz, r_max) of the respiratory fundamental, or (nan, 0)."""
    x = filtfilt(_B, _A, segment)
    x = decimate(x, Q, ftype="fir", zero_phase=True)
    x = x - x.mean()
    if x.std() == 0:
        return np.nan, 0.0
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = int(LAG_MIN_S * FS_D), int(LAG_MAX_S * FS_D)
    peaks, _ = find_peaks(ac[:hi + 1], prominence=0.05)
    peaks = peaks[peaks >= lo]
    if len(peaks) == 0:
        return np.nan, 0.0
    i = peaks[0]
    return FS_D / i, float(ac[i])


def per_window() -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        for state, dose, session, path in recordings(subject):
            d = pd.read_csv(path, usecols=["time", "RSP_Clean"])
            t, x = d["time"].to_numpy(), d["RSP_Clean"].to_numpy()
            step = int(STEP_S * FS)
            for win_s in WINDOWS_S:
                win = int(win_s * FS)
                for start in range(0, len(x) - win + 1, step):
                    seg = x[start:start + win]
                    if not np.all(np.isfinite(seg)):
                        continue
                    f, r = fundamental(seg)
                    rows.append(dict(subject=subject, state=state, dose=dose, session=session,
                                     window_s=win_s, t_centre=float(t[start + win // 2]),
                                     f_resp_hz=f, r_max=r, periodic=bool(r >= R_THRESH)))
    return pd.DataFrame(rows)


def summarise(g: pd.DataFrame) -> pd.Series:
    ok = g[g["periodic"] & g["f_resp_hz"].notna()]
    return pd.Series({
        "n_windows": len(g),
        "pct_periodic": 100 * g["periodic"].mean(),
        "median_f_hz": ok["f_resp_hz"].median() if len(ok) else np.nan,
        "median_breaths_per_min": 60 * ok["f_resp_hz"].median() if len(ok) else np.nan,
        "pct_periodic_below_015": 100 * (ok["f_resp_hz"] < HF_LOW).mean() if len(ok) else np.nan,
    })


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    df = per_window()
    df.to_csv(os.path.join(OUTDIR, "per_window.csv"), index=False)
    by_subject = (df.groupby(["window_s", "state", "subject"])
                  .apply(summarise, include_groups=False).reset_index())
    by_state = (df.groupby(["window_s", "state"])
                .apply(summarise, include_groups=False).reset_index())
    by_subject.to_csv(os.path.join(OUTDIR, "summary_by_subject.csv"), index=False)
    by_state.to_csv(os.path.join(OUTDIR, "summary_by_state.csv"), index=False)

    lines = [f"Respiratory fundamental vs HRV bands (n = {len(SUBJECTS)}: {SUBJECTS})",
             f"autocorrelation, windows of {WINDOWS_S} s in {STEP_S:.0f}-s steps, "
             f"periodic if r_max >= {R_THRESH}", ""]
    for win_s in WINDOWS_S:
        tag = " (main)" if win_s == MAIN_WINDOW_S else " (sensitivity)"
        lines.append(f"=== window {win_s:.0f} s{tag}")
        for state in ("DMT", "RS"):
            s = by_state[(by_state.window_s == win_s) & (by_state.state == state)].iloc[0]
            subj = by_subject[(by_subject.window_s == win_s) & (by_subject.state == state)]
            below = int((subj["median_f_hz"] < HF_LOW).sum())
            lines.append(f"  {state:3s}: periodic windows {s.pct_periodic:.1f}%; of those, "
                         f"{s.pct_periodic_below_015:.1f}% below {HF_LOW} Hz; participant "
                         f"medians below {HF_LOW} Hz: {below} of {len(subj)}")
            if win_s == MAIN_WINDOW_S:
                for _, r in subj.iterrows():
                    lines.append(f"       {r.subject}: median {r.median_f_hz:.3f} Hz "
                                 f"({r.median_breaths_per_min:.1f} breaths/min), "
                                 f"{r.pct_periodic:.0f}% periodic")
        lines.append("")
    with open(os.path.join(OUTDIR, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {OUTDIR}")


if __name__ == "__main__":
    main()
