# -*- coding: utf-8 -*-
"""Participant descriptives quoted in Methods and in the response letter.

Supports the first paragraph of Methods (Participants): n = 19; sex 17 male /
2 female; age M and SD; prior DMT use M, SD, median, range and the number of
participants with no prior DMT experience. Also gives, per physiological
subsample (ECG, EDA, respiration and their intersection), its size, the number
of women and its age range (the Methods age range for the cardiac sample used
by the intrinsic heart rate estimate, and the reply to Reviewer 3, 3.1: "the
only woman in the cardiac sample").

Inputs (metadata/):
    participants_sex.tsv           participant_id, sex (M/F, self-report)
    prior_dmt_use_session1.tsv     participant_id, prior_dmt_use (occasions)
    participants_age.tsv           participant_id, age  (PRIVATE, optional)
Subsamples: SUJETOS_VALIDADOS_ECG / _EDA / _RESP in config.py.

Privacy: age and sex are never written next to a participant ID; only
aggregates (counts, mean, SD, median, range) are printed or saved.

Outputs -> results/participants/
    descriptives.txt    the numbers quoted in the manuscript and the letter
    descriptives.csv    the same aggregates, one row per sample

Usage:
    micromamba run -n dmt-emotions python src/participant_descriptives.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (SUJETOS_VALIDADOS_ECG, SUJETOS_VALIDADOS_EDA,  # noqa: E402
                    SUJETOS_VALIDADOS_RESP)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
META = os.path.join(REPO, "metadata")
OUTDIR = os.path.join(REPO, "results", "participants")


def norm_id(pid: str) -> str:
    """'sub-04', 'S04', 'S4' -> 'S04' (the convention of config.py)."""
    m = re.search(r"(\d+)", str(pid))
    if m is None:
        raise ValueError(f"unrecognised participant id: {pid!r}")
    return f"S{int(m.group(1)):02d}"


def read_meta(name: str, col: str) -> pd.Series | None:
    """One metadata column indexed by normalised ID, or None if the file is absent."""
    path = os.path.join(META, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", dtype={"participant_id": str})
    df["participant_id"] = df["participant_id"].map(norm_id)
    if df["participant_id"].duplicated().any():
        raise ValueError(f"duplicated participant ids in {name}")
    return df.set_index("participant_id")[col]


def load() -> pd.DataFrame:
    """Merged table (in memory only; never saved, it holds per-person data)."""
    sex = read_meta("participants_sex.tsv", "sex").str.strip().str.upper()
    prior = read_meta("prior_dmt_use_session1.tsv", "prior_dmt_use").astype(float)
    df = pd.concat({"sex": sex, "prior_dmt_use": prior}, axis=1)
    age = read_meta("participants_age.tsv", "age")
    df["age"] = np.nan if age is None else age.astype(float)
    return df


def describe(df: pd.DataFrame, label: str, ids=None) -> dict:
    """Aggregates for a (sub)sample; SD is the sample SD (ddof = 1)."""
    sub = df if ids is None else df.loc[sorted(ids)]
    age = sub["age"].dropna()
    prior = sub["prior_dmt_use"].dropna()
    return {
        "sample": label,
        "n": len(sub),
        "n_male": int((sub["sex"] == "M").sum()),
        "n_female": int((sub["sex"] == "F").sum()),
        "n_age": len(age),
        "age_mean": age.mean() if len(age) else np.nan,
        "age_sd": age.std(ddof=1) if len(age) > 1 else np.nan,
        "age_min": age.min() if len(age) else np.nan,
        "age_max": age.max() if len(age) else np.nan,
        "prior_mean": prior.mean(),
        "prior_sd": prior.std(ddof=1),
        "prior_median": prior.median(),
        "prior_min": prior.min(),
        "prior_max": prior.max(),
        "prior_n_zero": int((prior == 0).sum()),
    }


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    df = load()
    has_age = df["age"].notna().any()

    subsamples = {
        "ECG": set(SUJETOS_VALIDADOS_ECG),
        "EDA": set(SUJETOS_VALIDADOS_EDA),
        "RESP": set(SUJETOS_VALIDADOS_RESP),
    }
    subsamples["ECG&EDA&RESP"] = subsamples["ECG"] & subsamples["EDA"] & subsamples["RESP"]
    for name, ids in subsamples.items():
        missing = ids - set(df.index)
        if missing:
            raise ValueError(f"{name}: ids not in metadata: {sorted(missing)}")

    rows = [describe(df, "all")] + [describe(df, k, v) for k, v in subsamples.items()]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTDIR, "descriptives.csv"), index=False, float_format="%.4f")

    a = rows[0]
    lines = [
        "Participant descriptives (aggregates only; no per-participant age or sex)",
        "=" * 72,
        f"Full sample: n = {a['n']}",
        f"Sex (self-report): {a['n_male']} M / {a['n_female']} F",
    ]
    if has_age:
        lines.append(f"Age (n = {a['n_age']}): M = {a['age_mean']:.1f}, SD = {a['age_sd']:.1f}, "
                     f"range {a['age_min']:.0f}-{a['age_max']:.0f} years")
    else:
        lines.append("Age: metadata/participants_age.tsv not found; age not reported")
    lines += [
        f"Prior DMT use (occasions): M = {a['prior_mean']:.1f}, SD = {a['prior_sd']:.1f}, "
        f"Mdn = {a['prior_median']:g}, range {a['prior_min']:g}-{a['prior_max']:g}; "
        f"{a['prior_n_zero']} with no prior use",
        "",
        "Physiological subsamples (config.py SUJETOS_VALIDADOS_*)",
        "-" * 72,
    ]
    for r in rows[1:]:
        s = f"{r['sample']:<13} n = {r['n']:>2}; {r['n_male']} M / {r['n_female']} F"
        if has_age:
            s += f"; age range {r['age_min']:.0f}-{r['age_max']:.0f} (M = {r['age_mean']:.1f})"
        lines.append(s)
    n_f_ecg = rows[1]["n_female"]
    lines += [
        "",
        f"Women in the cardiac (ECG) sample: {n_f_ecg}"
        + (" -> 'the only woman in the cardiac sample' holds" if n_f_ecg == 1 else ""),
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUTDIR, "descriptives.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
