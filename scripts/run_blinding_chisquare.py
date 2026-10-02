"""
run_blinding_chisquare.py

Tests blinding efficacy on the DMT dataset.

Two tests, following Lewis-Healey et al. (2024) on the same dataset:
  1) Goodness-of-fit chi-square against chance (50%) — overall accuracy.
  2) 2x2 chi-square of independence — does identification accuracy differ
     between the High (40 mg) and Low (20 mg) dose conditions?

The participant record is the single source of truth for both administered
dose and post-session dose guess:
    metadata/participants.tsv (the participants.tsv of the Zenodo data deposit)

Reference values (Lewis-Healey et al., 2024):
   26/38 sessions correctly identified
   14/19 (74%) high dose, 12/19 (63%) low dose
   Goodness-of-fit:   chi2(1, N=38) = 5.16, p = .02
   Independence:      chi2(1, N=38) = 0.49, p = .49

Outputs:
   results/blinding/blinding_chisquare_report.txt
   results/blinding/blinding_chisquare_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PARTICIPANTS_TSV = PROJECT_ROOT / "metadata" / "participants.tsv"

RESULTS_DIR = PROJECT_ROOT / "results" / "blinding"
REPORT_TXT = RESULTS_DIR / "blinding_chisquare_report.txt"
SUMMARY_CSV = RESULTS_DIR / "blinding_chisquare_summary.csv"


def load_sessions(participants_tsv):
    """Read participants.tsv and reshape to long format: one row per session."""
    df = pd.read_csv(participants_tsv, sep="\t")
    rows = []
    for _, r in df.iterrows():
        for s in (1, 2):
            rows.append({
                "participant_id": r["participant_id"],
                "session": s,
                "dose_admin_mg": int(r[f"session{s}_dose_mg"]),
                "dose_guess_mg": int(r[f"session{s}_dose_guess_mg"]),
            })
    long = pd.DataFrame(rows)
    long["dose_admin"] = long["dose_admin_mg"].map({40: "high", 20: "low"})
    long["guess"] = long["dose_guess_mg"].map({40: "high", 20: "low"})
    long["correct"] = (long["dose_admin"] == long["guess"]).astype(int)
    return long


def goodness_of_fit_correct(n_correct, n_total):
    """One-sample chi-square against H0: 50% correct (chance)."""
    observed = np.array([n_correct, n_total - n_correct])
    expected = np.array([n_total / 2, n_total / 2])
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=1))
    return chi2, p


def independence_high_low(n_correct_high, n_high, n_correct_low, n_low):
    """2x2 chi-square of independence: dose x correctness, no Yates correction."""
    table = np.array([
        [n_correct_high, n_high - n_correct_high],
        [n_correct_low, n_low - n_correct_low],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof), table


def main():
    if not PARTICIPANTS_TSV.exists():
        raise FileNotFoundError(f"{PARTICIPANTS_TSV} not found")
    long = load_sessions(PARTICIPANTS_TSV)

    lines = [
        "Blinding-efficacy chi-square tests",
        "=" * 60,
        f"Source: {PARTICIPANTS_TSV.relative_to(PROJECT_ROOT).as_posix()}",
        f"Total sessions: {len(long)}",
        "",
        "Reference (Lewis-Healey et al., 2024):",
        "  26/38 sessions correctly identified",
        "  Goodness-of-fit: chi2(1, N=38) = 5.16, p = .02",
        "  Independence:    chi2(1, N=38) = 0.49, p = .49",
    ]

    n_total = len(long)
    n_correct = int(long["correct"].sum())
    high = long[long["dose_admin"] == "high"]
    low = long[long["dose_admin"] == "low"]
    n_high, n_correct_high = len(high), int(high["correct"].sum())
    n_low, n_correct_low = len(low), int(low["correct"].sum())

    lines += [
        "",
        f"N total sessions: {n_total}",
        f"  Overall correct: {n_correct}/{n_total} "
        f"({100 * n_correct / n_total:.1f}%)",
        f"  High (40 mg) correct: {n_correct_high}/{n_high} "
        f"({100 * n_correct_high / n_high:.1f}%)",
        f"  Low  (20 mg) correct: {n_correct_low}/{n_low} "
        f"({100 * n_correct_low / n_low:.1f}%)",
    ]

    chi2_gof, p_gof = goodness_of_fit_correct(n_correct, n_total)
    lines += [
        "",
        "Goodness-of-fit (correct vs chance 50%):",
        f"  chi2(1, N={n_total}) = {chi2_gof:.2f}, p = {p_gof:.3f}",
    ]

    chi2_ind, p_ind, dof, table = independence_high_low(
        n_correct_high, n_high, n_correct_low, n_low
    )
    lines += [
        "",
        "Independence (does correct rate differ between doses?):",
        f"  chi2({dof}, N={n_total}) = {chi2_ind:.2f}, p = {p_ind:.3f}",
        "  contingency table [correct, incorrect] x [high, low]:",
        f"    high: correct={table[0,0]}, incorrect={table[0,1]}",
        f"    low:  correct={table[1,0]}, incorrect={table[1,1]}",
    ]

    summary = pd.DataFrame([{
        "N": n_total,
        "correct": n_correct,
        "pct_correct": 100 * n_correct / n_total,
        "high_correct": n_correct_high, "high_N": n_high,
        "low_correct": n_correct_low, "low_N": n_low,
        "gof_chi2": chi2_gof, "gof_p": p_gof,
        "ind_chi2": chi2_ind, "ind_p": p_ind,
    }])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n".join(lines))
    print(f"\nWrote: {REPORT_TXT}")
    print(f"Wrote: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
