# -*- coding: utf-8 -*-
"""Ratio of sample-level to window-level standard deviation, per modality.

Supports the statement in Methods (physiological preprocessing, window-level
standardisation) and in the reply to Reviewer 3 (3.3) that the ratio of the
sample-level to the window-level standard deviation has a median of 0.98 for
heart rate, 1.57 for RVT and 10.86 for SMNA. This is the reason the three
signals are standardised over their 30-s window values rather than over their
raw samples: with sample-level scaling, one standard deviation would mean a
different quantity in each modality.

Method (per participant, on the raw, unscaled signals):
  - sample-level SD: the sigma that the pipeline's subject-level z-scoring uses,
    i.e. `zscore_with_subject_baseline` of each analysis module applied to the
    four full recordings (RS High, DMT High, RS Low, DMT Low); NaNs removed,
    and for RVT only samples in the module's physiological range (0, 50000);
  - window-level SD: SD (ddof = 1) of the 72 raw window summaries (18 windows
    of 30 s over 0-9 min x 4 recordings), computed with the module's own window
    function: mean HR, mean RVT, and for SMNA the AUC divided by the 30-s window
    length (the time-averaged SMNA, so that both SDs are in the signal's units);
  - ratio = sample-level SD / window-level SD; the median is taken across the
    participants of each modality.
Only windows present in all four recordings are kept, as in the pipeline (in
practice every included participant has all 72).

Sample: the validated participants of each modality (SUJETOS_VALIDADOS_ECG,
_RESP, _EDA) with all four recordings scalable, i.e. the participants entering
each modality's LME.

Outputs -> results/robustness/
    window_sd_ratio.csv            one row per participant x modality
    window_sd_ratio_report.txt     the medians quoted in the manuscript and the letter

Usage:
    micromamba run -n dmt-emotions python src/run_window_sd_ratio.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import run_ecg_hr_analysis as HR  # noqa: E402
import run_eda_smna_analysis as SM  # noqa: E402
import run_resp_rvt_analysis as RV  # noqa: E402

OUTDIR = os.path.join(REPO, "results", "robustness")
N_WINDOWS = 18
WINDOW_S = 30.0

# Values quoted in Methods and in the letter (3.3), for the side-by-side check.
QUOTED = {"HR": 0.98, "RVT": 1.57, "SMNA": 10.86}


def _hr_loader(path):
    d = HR.load_ecg_csv(path)
    if d is None:
        return None
    return d["time"].to_numpy(), pd.to_numeric(d["ECG_Rate"], errors="coerce").to_numpy()


def _rvt_loader(path):
    d = RV.load_resp_csv(path)
    if d is None:
        return None
    return d["time"].to_numpy(), pd.to_numeric(d["RSP_RVT"], errors="coerce").to_numpy()


def _smna_window(t, y, w):
    a = SM.compute_auc_window(t, y, w)
    return None if a is None else a / WINDOW_S


def _paths(mod, subject):
    """Paths in the order the z-scoring function expects: RS H, DMT H, RS L, DMT L."""
    hs, ls = mod.determine_sessions(subject)
    if mod is HR:
        dh, dl = HR.build_ecg_paths(subject, hs, ls)
        rh, rl = HR.build_rs_ecg_path(subject, hs), HR.build_rs_ecg_path(subject, ls)
    elif mod is RV:
        dh, dl = RV.build_resp_paths(subject, hs, ls)
        rh, rl = RV.build_rs_resp_path(subject, hs), RV.build_rs_resp_path(subject, ls)
    else:
        dh, dl = SM.build_cvx_paths(subject, hs, ls)
        rh, rl = SM.build_rs_cvx_path(subject, hs), SM.build_rs_cvx_path(subject, ls)
    return [rh, dh, rl, dl]


MODALITIES = [
    # name, module, subjects, loader, raw window summary
    ("HR", HR, HR.SUJETOS_VALIDADOS_ECG, _hr_loader, HR.compute_hr_mean_per_window),
    ("RVT", RV, RV.SUJETOS_VALIDADOS_RESP, _rvt_loader, RV.compute_rvt_mean_per_window),
    ("SMNA", SM, SM.SUJETOS_VALIDADOS_EDA, SM.load_cvx_smna, _smna_window),
]


def subject_ratio(mod, subject, loader, window_fn):
    """Dict with the sample-level SD, window-level SD and their ratio, or None."""
    recs = [loader(p) for p in _paths(mod, subject)]
    if any(r is None for r in recs):
        return None
    args = [x for r in recs for x in r]
    *_, diag = mod.zscore_with_subject_baseline(*args)
    if not diag["scalable"]:
        return None
    windows = []
    for w in range(N_WINDOWS):
        vals = [window_fn(t, y, w) for t, y in recs]
        if None not in vals:
            windows.extend(vals)
    if len(windows) < 2:
        return None
    sd_window = float(np.std(windows, ddof=1))
    return dict(sd_sample=diag["sigma"], sd_window=sd_window,
                ratio=diag["sigma"] / sd_window, n_windows=len(windows))


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for name, mod, subjects, loader, window_fn in MODALITIES:
        for s in subjects:
            r = subject_ratio(mod, s, loader, window_fn)
            if r is not None:
                rows.append(dict(modality=name, subject=s, **r))
    df = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(os.path.join(OUTDIR, "window_sd_ratio.csv"), index=False)

    lines = ["Ratio of sample-level to window-level SD (raw signals, per participant)",
             "sample-level SD = sigma of the subject-level z-scoring over the four full recordings",
             "window-level SD = SD of the 72 raw 30-s window values (0-9 min x 4 recordings)",
             "",
             f"{'modality':8s} {'n':>3s} {'median':>8s} {'min':>7s} {'max':>7s}   quoted"]
    for name, _, _, _, _ in MODALITIES:
        r = df.loc[df.modality == name, "ratio"]
        lines.append(f"{name:8s} {len(r):3d} {r.median():8.2f} {r.min():7.2f} {r.max():7.2f}"
                     f"   {QUOTED[name]:.2f}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(OUTDIR, "window_sd_ratio_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"-> {OUTDIR}")


if __name__ == "__main__":
    main()
