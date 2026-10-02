"""Build Supplementary Table (tab:supp_fdr_windows): cluster-based vs
window-wise FDR results, one row per comparison, straight from the CSVs
written by run_cluster_permutation*.py. Nothing is typed by hand.

Writes ../dmt-emotions-paper/tables/fdr_windows_table.tex
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results", "cluster_permutation")
PAPER = os.path.abspath(os.path.join(REPO, "..", "..", "dmt-emotions-paper"))
OUT = os.path.join(PAPER, "tables", "fdr_windows_table.tex")

WIN = 0.5


def runs(mask: np.ndarray, start: np.ndarray, end: np.ndarray) -> List[str]:
    """Contiguous True runs -> 'a--b' strings in minutes."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(f"{start[i]:.1f}--{end[j]:.1f}")
            i = j
        i += 1
    return out


def cluster_cells(d: pd.DataFrame) -> str:
    """'a--b (p = .xxx); ...' for significant clusters, from per-window p_cluster."""
    d = d.sort_values("window" if "window" in d else "time_min")
    sig = d.sig_cluster.to_numpy()
    p = d.p_cluster.to_numpy()
    if "window" in d:
        s, e = d.t_start_min.to_numpy(), d.t_end_min.to_numpy()
    else:
        s = e = d.time_min.to_numpy()
    out, i = [], 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j + 1 < len(sig) and sig[j + 1] and np.isclose(p[j + 1], p[i]):
                j += 1
            out.append(f"{s[i]:.1f}--{e[j]:.1f} (${fmt_p(p[i])}$)")
            i = j
        i += 1
    return "; ".join(out) if out else "none"


def fmt_p(p: float) -> str:
    """'p < .001' | 4 decimals below .005 (matches the .0015 quoted in Results) | 3 decimals."""
    if p < 0.001:
        return "p < .001"
    dec = 4 if p < 0.005 else 3
    return f"p = {p:.{dec}f}".replace("= 0.", "= .")


def fdr_cells(d: pd.DataFrame) -> str:
    d = d.sort_values("window" if "window" in d else "time_min")
    if "window" in d:
        s, e = d.t_start_min.to_numpy(), d.t_end_min.to_numpy()
    else:
        s = e = d.time_min.to_numpy()
    r = runs(d.sig_fdr.to_numpy(), s, e)
    return "; ".join(r) if r else "none"


def main() -> None:
    pw = pd.read_csv(os.path.join(RES, "per_window.csv"))
    tet = pd.read_csv(os.path.join(RES, "tet_per_bin.csv"))
    rows = []

    labels = {"HR": "Heart rate", "SMNA": "SMNA", "RVT": "RVT", "Arousal": "Physiological Arousal Index"}
    for mod in ["HR", "SMNA", "RVT", "Arousal"]:
        for state, lab, tail in [("DMT", "DMT", "one-tailed"), ("RS", "RS", "two-tailed"),
                                 ("DMT_extended", "DMT, 0--19\\,min", "one-tailed")]:
            d = pw[(pw.modality == mod) & (pw.state == state)]
            if d.empty:
                continue
            rows.append((labels[mod], lab, tail, int(d.n.iloc[0]), cluster_cells(d), fdr_cells(d)))

    tet_labels = {"emotional_intensity_z": "Emotional Intensity", "valence_index": "Valence Index",
                  "interoception_z": "Interoception", "anxiety_z": "Anxiety",
                  "unpleasantness_z": "Unpleasantness", "pleasantness_z": "Pleasantness",
                  "bliss_z": "Bliss", "PC1": "PC1", "PC2": "PC2"}
    for var, lab in tet_labels.items():
        for comp, clab in [("dose", "DMT High vs.\\ Low"), ("state", "DMT vs.\\ RS")]:
            d = tet[(tet.variable == var) & (tet.comparison == comp)]
            if d.empty:
                continue
            c, f = cluster_cells(d), fdr_cells(d)
            if c == "none" and f == "none":
                continue
            rows.append((f"TET: {lab}", clab, "two-tailed", int(d.n.iloc[0]), c, f))

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \scriptsize",
        r"  \caption{Cluster-based permutation results and window-wise FDR-corrected comparisons for all "
        r"shaded time courses. Cluster extents are approximate (see Methods). Cluster-level $p$ values "
        r"are exact, obtained by exhaustive sign-flip permutation ($2^{n}$ permutations for one-tailed "
        r"tests and $2^{n-1}$ for two-tailed ones). FDR windows list the time points at which the paired "
        r"$t$-test survived Benjamini--Hochberg correction across time points ($p_{\text{FDR}} < .05$). "
        r"Physiological time courses use 30-s windows; TET time courses use native 4-s samples (DMT: "
        r"0--20\,min; RS: 0--10\,min). Resting-state comparisons are listed even when neither test "
        r"returns a significant result, because the manuscript reports the absence of dose separation at "
        r"rest.}",
        r"  \label{tab:supp_fdr_windows}",
        r"  \begin{tabular}{l l l c p{4.0cm} p{3.6cm}}",
        r"    \toprule",
        r"    \textbf{Measure} & \textbf{Comparison} & \textbf{Tail} & $\boldsymbol{n}$ & "
        r"\textbf{Cluster(s), min} & \textbf{FDR windows, min} \\",
        r"    \midrule",
    ]
    for m, c, t, n, cl, fd in rows:
        lines.append(f"    {m} & {c} & {t} & {n} & {cl} & {fd} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print(f"-> {OUT}  ({len(rows)} rows)")
    for r in rows:
        print("  ", " | ".join(str(x) for x in r))


if __name__ == "__main__":
    main()
