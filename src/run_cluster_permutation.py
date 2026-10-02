"""Unit B (R2.2): cluster-based permutation on the published time courses.

Reads the long CSVs the paper already uses, builds subject x window matrices
for each High-vs-Low comparison, and runs FDR (control), fixed-threshold
cluster permutation and TFCE side by side (see cluster_stats.py).

Stage 1 -- validation: the reimplemented FDR must reproduce the shaded
segments published in 04_results.tex before anything new is read:
    HR   DMT: from 1.5 min onward
    SMNA DMT: 4.5-5.0, 6.0-6.5, 8.0-9.0
    RVT  DMT: 1.5-2.5, 3.0-4.0, 4.5-5.0

Usage:
    micromamba run -n dmt-emotions python src/run_cluster_permutation.py

Outputs -> results/cluster_permutation/   (FDR reference + cluster + TFCE per window)
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cluster_stats import run_all, segments  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "results", "cluster_permutation")
WINDOW_MIN = 0.5

MODALITIES = {
    "HR":      ("results/ecg/hr/hr_minute_long_data_z.csv",           "HR"),
    "SMNA":    ("results/eda/smna/smna_auc_long_data_z.csv",          "AUC"),
    "RVT":     ("results/resp/rvt/resp_rvt_minute_long_data_z.csv",   "RSP_RVT"),
    "Arousal": ("results/composite/arousal_index_long.csv",           "ArousalIndex"),
}

# (state, alternative) -- the tails the paper uses for each panel
COMPARISONS = [("DMT", "greater"), ("RS", "two-sided")]

# DMT High vs Low, in minutes. These were the window-wise FDR segments obtained
# under the original sample-level standardisation, and the check below
# reproduced them exactly, which validated the reimplementation (unit B).
# Unit E moved the standardisation to the analysis window, which changes the
# per-window t values and therefore these segments; the values below are the
# ones the current pipeline produces. The original set is kept alongside so the
# provenance of the change stays visible.
PUBLISHED_FDR_SAMPLE_LEVEL = {   # before unit E, reproduced exactly at the time
    "HR":   [(1.5, 9.0)],
    "SMNA": [(4.5, 5.0), (6.0, 6.5), (8.0, 9.0)],
    "RVT":  [(1.5, 2.5), (3.0, 4.0), (4.5, 5.0)],
}
PUBLISHED_FDR = {   # window-level standardisation, current
    "HR":   [(1.5, 9.0)],
    "SMNA": [],
    "RVT":  [(1.5, 5.0)],
}


def load_matrices(path: str, col: str, state: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    df = pd.read_csv(os.path.join(REPO, path))
    if "Scale" in df.columns:
        df = df[df["Scale"] == "z"]
    df = df[df["State"] == state]
    wide = df.pivot_table(index="subject", columns=["Dose", "window"], values=col)
    subjects = list(wide.index)
    windows = sorted({w for _, w in wide.columns})
    H = wide["High"][windows].to_numpy(dtype=float)
    L = wide["Low"][windows].to_numpy(dtype=float)
    return H, L, subjects, np.array(windows)


PUBLISHED_FDR_EXTENDED = {   # from results/*/fdr_segments_all_subs_dmt_*.txt
    "HR":      [(2.5, 3.0), (3.5, 9.0)],
    "Arousal": [(2.0, 3.0), (3.5, 10.0), (10.5, 14.5), (16.0, 19.0)],
}


def load_extended() -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]]:
    """0-19 min DMT-only matrices, built exactly as the published scripts do.

    HR: results/ecg/hr/hr_extended_dmt_z.csv (already z-scored).
    Arousal: results/composite/merged_extended_dmt_complete_cases.csv, each
    signal re-z-scored within subject over the extended DMT data, then
    projected on the saved PC1 loadings (run_composite_arousal_index.py
    lines 2544-2557).
    """
    out = {}
    hr = pd.read_csv(os.path.join(REPO, "results/ecg/hr/hr_extended_dmt_z.csv"))
    hr = hr[(hr["Scale"] == "z") & (hr["State"] == "DMT")]
    w = hr.pivot_table(index="subject", columns=["Dose", "window"], values="HR")
    win = sorted({k for _, k in w.columns})
    out["HR"] = (w["High"][win].to_numpy(float), w["Low"][win].to_numpy(float), list(w.index), np.array(win))

    ext = pd.read_csv(os.path.join(REPO, "results/composite/merged_extended_dmt_complete_cases.csv"))
    ext = ext[ext["State"] == "DMT"].copy()
    for col in ["SMNA_AUC", "HR", "RVT"]:
        ext[f"{col}_z"] = ext.groupby("subject")[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    load = pd.read_csv(os.path.join(REPO, "results/composite/pca_loadings_pc1.csv"))["loading_pc1"].to_numpy()
    ext["ArousalIndex"] = ext[["HR_z", "SMNA_AUC_z", "RVT_z"]].to_numpy() @ load
    w = ext.pivot_table(index="subject", columns=["Dose", "window"], values="ArousalIndex")
    win = sorted({k for _, k in w.columns})
    out["Arousal"] = (w["High"][win].to_numpy(float), w["Low"][win].to_numpy(float), list(w.index), np.array(win))
    return out


def seg_minutes(segs_windows: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(first_window, last_window) -> (start_min, end_min)."""
    return [((w0 - 1) * WINDOW_MIN, w1 * WINDOW_MIN) for w0, w1 in segs_windows]


def fmt(segs: List[Tuple[float, float]]) -> str:
    return ", ".join(f"{a:.1f}-{b:.1f}" for a, b in segs) if segs else "none"


def smna_sample_level_sensitivity(say) -> List[Dict]:
    """SMNA DMT High vs Low clusters under the submitted manuscript's sample-level
    standardisation (window values not re-standardised; see unit E in
    run_eda_smna_analysis.py). The reply to Reviewer 2 (2.2) uses it to show that
    the late epoch (7.5-9 min) is lost to the change of standardisation, not to
    the cluster test."""
    import run_eda_smna_analysis as smna
    df = smna.prepare_long_data(restandardise=False)
    df = df[(df["Scale"] == "z") & (df["State"] == "DMT")]
    wide = df.pivot_table(index="subject", columns=["Dose", "window"], values="AUC",
                          observed=True)
    windows = sorted({w for _, w in wide.columns})
    H = wide["High"][windows].to_numpy(dtype=float)
    L = wide["Low"][windows].to_numpy(dtype=float)
    res = run_all(H, L, np.array(windows), "greater")
    say()
    say("=" * 72)
    say(f"SENSITIVITY | SMNA | DMT High vs Low | sample-level standardisation "
        f"(submitted manuscript) | n = {H.shape[0]} | tail = greater")
    say("=" * 72)
    out = []
    for c in res["cluster"]["clusters"]:
        w0, w1 = c["windows"]
        say(f"      windows {w0:2d}-{w1:2d} ({(w0-1)*WINDOW_MIN:.1f}-{w1*WINDOW_MIN:.1f} min)"
            f"  mass = {c['mass']:6.2f}  p = {c['p']:.4f}")
        out.append(dict(modality="SMNA", state="DMT", standardisation="sample-level",
                        n=H.shape[0], t_start_min=(w0 - 1) * WINDOW_MIN,
                        t_end_min=w1 * WINDOW_MIN, mass=c["mass"], p=c["p"]))
    return out


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    rows: List[Dict] = []
    summary: List[Dict] = []
    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    validation_ok = True
    for mod, (path, col) in MODALITIES.items():
        for state, alt in COMPARISONS:
            H, L, subjects, windows = load_matrices(path, col, state)
            res = run_all(H, L, windows, alt)
            say("=" * 72)
            say(f"{mod} | {state} High vs Low | n = {len(subjects)} | tail = {alt} | "
                f"permutations = {res['cluster']['n_permutations']} (exhaustive = {2**len(subjects)})")
            say("=" * 72)

            say("  t per window: " + " ".join(f"{v:5.2f}" for v in res["fdr"]["t"]))
            say("  p (uncorr.):  " + " ".join(f"{v:5.3f}" for v in res["fdr"]["p"]))
            say("  p (FDR):      " + " ".join(f"{v:5.3f}" for v in res["fdr"]["p_adj"]))
            say("  p (TFCE):     " + " ".join(f"{v:5.3f}" for v in res["tfce"]["p_window"]))
            fdr_segs = seg_minutes(segments(res["fdr"]["sig"], windows))
            clu_segs = seg_minutes(segments(res["cluster"]["sig"], windows))
            tfce_segs = seg_minutes(segments(res["tfce"]["sig"], windows))
            say(f"  FDR (published method): {fmt(fdr_segs)}")
            if state == "DMT" and mod in PUBLISHED_FDR:
                ok = fdr_segs == PUBLISHED_FDR[mod]
                validation_ok &= ok
                say(f"  -> published:           {fmt(PUBLISHED_FDR[mod])}   "
                    f"{'REPRODUCED' if ok else '*** MISMATCH ***'}")
            say(f"  cluster (t > {res['cluster']['threshold']:.2f}): {fmt(clu_segs)}")
            for c in res["cluster"]["clusters"]:
                w0, w1 = c["windows"]
                say(f"      windows {w0:2d}-{w1:2d} ({(w0-1)*WINDOW_MIN:.1f}-{w1*WINDOW_MIN:.1f} min)"
                    f"  mass = {c['mass']:6.2f}  p = {c['p']:.4f}")
            say(f"  TFCE:                   {fmt(tfce_segs)}")

            for k, w in enumerate(windows):
                rows.append(dict(
                    modality=mod, state=state, alternative=alt, n=len(subjects),
                    window=int(w), t_start_min=(w - 1) * WINDOW_MIN, t_end_min=w * WINDOW_MIN,
                    t=res["fdr"]["t"][k], p_uncorrected=res["fdr"]["p"][k],
                    p_fdr=res["fdr"]["p_adj"][k], sig_fdr=bool(res["fdr"]["sig"][k]),
                    p_cluster=res["cluster"]["p_window"][k], sig_cluster=bool(res["cluster"]["sig"][k]),
                    p_tfce=res["tfce"]["p_window"][k], sig_tfce=bool(res["tfce"]["sig"][k]),
                ))
            summary.append(dict(modality=mod, state=state, alternative=alt, n=len(subjects),
                                fdr=fmt(fdr_segs), cluster=fmt(clu_segs), tfce=fmt(tfce_segs),
                                n_sig_fdr=int(res["fdr"]["sig"].sum()),
                                n_sig_cluster=int(res["cluster"]["sig"].sum()),
                                n_sig_tfce=int(res["tfce"]["sig"].sum())))

    # ----------------------------------------------------------------- #
    # extended window (0-19 min, 38 windows, DMT only, one-tailed)
    # Supplementary Fig. S3 (HR) and S4 (Arousal index)
    # ----------------------------------------------------------------- #
    for mod, (H, L, subjects, windows) in load_extended().items():
        res = run_all(H, L, windows, "greater")
        say("=" * 72)
        say(f"{mod} EXTENDED 0-19 min | DMT High vs Low | n = {len(subjects)} | tail = greater | "
            f"permutations = {res['cluster']['n_permutations']}")
        say("=" * 72)
        say("  t per window: " + " ".join(f"{v:5.2f}" for v in res["fdr"]["t"]))
        say("  p (FDR):      " + " ".join(f"{v:5.3f}" for v in res["fdr"]["p_adj"]))
        fdr_segs = seg_minutes(segments(res["fdr"]["sig"], windows))
        clu_segs = seg_minutes(segments(res["cluster"]["sig"], windows))
        say(f"  FDR (published method): {fmt(fdr_segs)}")
        ok = fdr_segs == PUBLISHED_FDR_EXTENDED[mod]
        validation_ok &= ok
        say(f"  -> published:           {fmt(PUBLISHED_FDR_EXTENDED[mod])}   "
            f"{'REPRODUCED' if ok else '*** MISMATCH ***'}")
        say(f"  cluster (t > {res['cluster']['threshold']:.2f}): {fmt(clu_segs)}")
        for c in res["cluster"]["clusters"]:
            w0, w1 = c["windows"]
            say(f"      windows {w0:2d}-{w1:2d} ({(w0-1)*WINDOW_MIN:.1f}-{w1*WINDOW_MIN:.1f} min)"
                f"  mass = {c['mass']:6.2f}  p = {c['p']:.4f}")
        for k, w in enumerate(windows):
            rows.append(dict(
                modality=mod, state="DMT_extended", alternative="greater", n=len(subjects),
                window=int(w), t_start_min=(w - 1) * WINDOW_MIN, t_end_min=w * WINDOW_MIN,
                t=res["fdr"]["t"][k], p_uncorrected=res["fdr"]["p"][k],
                p_fdr=res["fdr"]["p_adj"][k], sig_fdr=bool(res["fdr"]["sig"][k]),
                p_cluster=res["cluster"]["p_window"][k], sig_cluster=bool(res["cluster"]["sig"][k]),
                p_tfce=res["tfce"]["p_window"][k], sig_tfce=bool(res["tfce"]["sig"][k]),
            ))
        summary.append(dict(modality=mod, state="DMT_extended", alternative="greater", n=len(subjects),
                            fdr=fmt(fdr_segs), cluster=fmt(clu_segs), tfce="",
                            n_sig_fdr=int(res["fdr"]["sig"].sum()),
                            n_sig_cluster=int(res["cluster"]["sig"].sum()), n_sig_tfce=""))

    say()
    say("VALIDATION: " + ("all published FDR segments reproduced"
                          if validation_ok else "*** at least one mismatch -- do not interpret ***"))

    sensitivity = smna_sample_level_sensitivity(say)

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "per_window.csv"), index=False)
    pd.DataFrame(sensitivity).to_csv(os.path.join(OUTDIR, "smna_sample_level_sensitivity.csv"),
                                     index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {OUTDIR}")


if __name__ == "__main__":
    main()
