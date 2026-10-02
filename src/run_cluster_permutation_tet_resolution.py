"""Unit B (R2.2): sensitivity of the TET cluster results to temporal resolution.

Mirrors the published `run_temporal_resolution_sensitivity.py` (imported,
not modified): TET ratings averaged into 4 s (native), 20 s and 30 s bins,
then High-vs-Low (within DMT) and DMT-vs-RS cluster tests per affective
variable. Two-tailed throughout, as in Fig. 4a and Methods.

NOTE: the published sensitivity script uses alternative='greater' for the
dose test, whereas Fig. 4a / Methods use two-tailed. Flagged, not changed.

Usage:
    micromamba run -n dmt-emotions python src/run_cluster_permutation_tet_resolution.py
Outputs -> results/cluster_permutation/tet_resolution_*.csv, tet_resolution_report.txt
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
REPO = os.path.abspath(os.path.join(SRC, ".."))
sys.path[:0] = [HERE, SRC]
os.chdir(REPO)

from cluster_stats import cluster_test, segments                    # noqa: E402
import run_temporal_resolution_sensitivity as trs                   # noqa: E402  (published, read-only)

OUTDIR = os.path.join(REPO, "results", "cluster_permutation")


def wide(df: pd.DataFrame, var: str, bins: List[float]) -> pd.DataFrame:
    return (df[df["time_min"].isin(bins)]
            .groupby(["subject", "time_min"])[var].mean().unstack("time_min"))[bins]


def pair(A: pd.DataFrame, B: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    subj = sorted(set(A.dropna().index) & set(B.dropna().index))
    return A.loc[subj].to_numpy(), B.loc[subj].to_numpy(), subj


def fmt(segs) -> str:
    return ", ".join(f"{a:.1f}-{b:.1f}" for a, b in segs) if segs else "none"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    df = trs.load_and_prepare()
    rows: List[Dict] = []
    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s); lines.append(s)

    for label, bin_sec in trs.RESOLUTIONS.items():
        ds = trs.downsample(df, bin_sec)
        dmt, rs = ds[ds.state == "DMT"], ds[ds.state == "RS"]
        high, low = dmt[dmt.dose == "Alta"], dmt[dmt.dose == "Baja"]
        say("#" * 72)
        say(f"RESOLUTION {label}")
        say("#" * 72)
        for var in trs.AFFECTIVE_VARS:
            for comp, (dfa, dfb) in [("dose", (high, low)), ("state", (dmt, rs))]:
                bins = sorted(set(dfa.time_min.unique()) & set(dfb.time_min.unique()))
                A, B, subj = pair(wide(dfa, var, bins), wide(dfb, var, bins))
                c = cluster_test(A, B, "two-sided")
                x = np.array(bins)
                segs = segments(c["sig"], x)
                say(f"  {var:24s} {comp:5s} n = {len(subj)}  bins = {len(bins):3d}  cluster: {fmt(segs)}")
                for cl in c["clusters"]:
                    w0, w1 = cl["windows"]
                    say(f"      {x[w0-1]:5.1f}-{x[w1-1]:5.1f} min  mass = {cl['mass']:8.2f}  p = {cl['p']:.4f}")
                rows.append(dict(resolution=label, bin_sec=bin_sec, variable=var, comparison=comp,
                                 n=len(subj), n_bins=len(bins), cluster=fmt(segs),
                                 clusters=[(x[c0-1], x[c1-1], round(m, 2), round(p, 4))
                                           for (c0, c1), m, p in
                                           [(cl["windows"], cl["mass"], cl["p"]) for cl in c["clusters"]]]))
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "tet_resolution_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "tet_resolution_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
