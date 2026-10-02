"""Unit B (R2.2), TET part: cluster-based permutation on the Fig. 4a time courses.

Replicates the two inferences of `compute_significance_masks` in the published
`run_tet_analysis.py` (imported, not modified):

  state  -- DMT vs RS at each common 4-s bin (0-10 min), doses pooled within
            subject; two-tailed. Grey shading in Fig. 4a.
  dose   -- High vs Low within DMT at each 4-s bin (0-20 min); two-tailed.
            Black bars in Fig. 4a.

for the same variables (two indices, five items, PC1, PC2). Validation: the
since revision R2.2 the published function returns the cluster masks, so the
cluster masks computed here must equal them bin by bin. The window-wise FDR
masks are kept alongside for Supplementary Table 1.

Usage:
    micromamba run -n dmt-emotions python src/run_cluster_permutation_tet.py
Outputs -> results/cluster_permutation/tet_*.csv, tet_report.txt
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

from cluster_stats import fdr_mask, cluster_test, segments          # noqa: E402
import run_tet_analysis as tet                                       # noqa: E402  (published, read-only)

OUTDIR = os.path.join(REPO, "results", "cluster_permutation")
VARIABLES = ["emotional_intensity_z", "valence_index", "interoception_z",
             "anxiety_z", "unpleasantness_z", "pleasantness_z", "bliss_z", "PC1", "PC2"]


def wide(df: pd.DataFrame, var: str, bins: List[float]) -> pd.DataFrame:
    """subject x time_min matrix of the per-subject mean of `var`."""
    w = (df[df["time_min"].isin(bins)]
         .groupby(["subject", "time_min"])[var].mean().unstack("time_min"))
    return w[bins]


def matrices_state(df: pd.DataFrame, var: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[str]]:
    d, r = df[df.state == "DMT"], df[df.state == "RS"]
    bins = sorted(set(d.time_min.unique()) & set(r.time_min.unique()))
    A, B = wide(d, var, bins), wide(r, var, bins)
    subj = sorted(set(A.dropna().index) & set(B.dropna().index))
    return A.loc[subj].to_numpy(), B.loc[subj].to_numpy(), bins, subj


def matrices_dose(df: pd.DataFrame, var: str) -> Tuple[np.ndarray, np.ndarray, List[float], List[str]]:
    d = df[df.state == "DMT"]
    bins = sorted(d.time_min.unique())
    A, B = wide(d[d.dose == "Alta"], var, bins), wide(d[d.dose == "Baja"], var, bins)
    subj = sorted(set(A.dropna().index) & set(B.dropna().index))
    return A.loc[subj].to_numpy(), B.loc[subj].to_numpy(), bins, subj


def fmt(segs: List[Tuple[float, float]]) -> str:
    return ", ".join(f"{a:.1f}-{b:.1f}" for a, b in segs) if segs else "none"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    df = tet.load_data()
    df, _, _ = tet.compute_pca(df)
    df = df.copy()
    df["valence_index"] = df["pleasantness_z"] - df["unpleasantness_z"]
    published = tet.compute_significance_masks(df)          # the control

    rows: List[Dict] = []
    summary: List[Dict] = []
    lines: List[str] = []
    ok_all = True

    def say(s: str = "") -> None:
        print(s); lines.append(s)

    for var in VARIABLES:
        for comp, builder, key_bins, key_sig in [
            ("state", matrices_state, "state_time_bins", "state_sig"),
            ("dose", matrices_dose, "time_bins", "dose_sig"),
        ]:
            A, B, bins, subj = builder(df, var)
            if np.isnan(A).any() or np.isnan(B).any():
                say(f"{var} {comp}: NaN present -- skipped"); continue
            f = fdr_mask(A, B, "two-sided")
            c = cluster_test(A, B, "two-sided")
            pub_bins = list(published[var][key_bins])
            pub_sig = np.asarray(published[var][key_sig], dtype=bool)
            same = (pub_bins == list(bins)) and np.array_equal(pub_sig, c["sig"])
            ok_all &= same
            x = np.array(bins)
            fdr_segs = [(a, b) for a, b in segments(f["sig"], x)]
            clu_segs = [(a, b) for a, b in segments(c["sig"], x)]
            say("=" * 72)
            say(f"{var} | {comp} | n = {len(subj)} | bins = {len(bins)} ({bins[0]:.1f}-{bins[-1]:.1f} min) | "
                f"perms = {c['n_permutations']} | cluster vs published: {'REPRODUCED' if same else '*** MISMATCH ***'}")
            say(f"  FDR:     {fmt(fdr_segs)}  ({int(f['sig'].sum())} bins)")
            say(f"  cluster: {fmt(clu_segs)}  ({int(c['sig'].sum())} bins)")
            for cl in c["clusters"]:
                w0, w1 = cl["windows"]
                say(f"      {x[w0-1]:.1f}-{x[w1-1]:.1f} min  mass = {cl['mass']:7.2f}  p = {cl['p']:.4f}")
            for k, b in enumerate(bins):
                rows.append(dict(variable=var, comparison=comp, n=len(subj), time_min=b,
                                 t=f["t"][k], p_uncorrected=f["p"][k], p_fdr=f["p_adj"][k],
                                 sig_fdr=bool(f["sig"][k]), p_cluster=c["p_window"][k],
                                 sig_cluster=bool(c["sig"][k])))
            summary.append(dict(variable=var, comparison=comp, n=len(subj), n_bins=len(bins),
                                fdr=fmt(fdr_segs), cluster=fmt(clu_segs),
                                n_sig_fdr=int(f["sig"].sum()), n_sig_cluster=int(c["sig"].sum()),
                                cluster_reproduced=same))
    say()
    say("VALIDATION: " + ("all cluster masks identical to the published function"
                          if ok_all else "*** mismatch -- do not interpret ***"))
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "tet_per_bin.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "tet_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "tet_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
