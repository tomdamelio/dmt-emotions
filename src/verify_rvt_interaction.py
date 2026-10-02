"""Is the RVT State x Dose interaction really lost under random slopes?

Context. Under the published random-intercept model the RVT interaction is
significant (beta = 0.324, 95% CI [0.166, 0.482]); once by-subject random
slopes are added it is not (CI crosses zero). Before treating that as a result
it has to be separated from an optimiser failure, because the fits misbehave:
the M4 rung returns a WORSE log-likelihood than M3, which is impossible for a
nested model that actually converged, and it returns a SMALLER standard error
than M3, which is the opposite of what adding random effects should do.

The same models are refitted under several optimisers, on all four modalities
so the method can be judged against cases where the answer is already known.
Agreement means the estimate is real; disagreement means we were reading
optimiser noise.

Usage:
    micromamba run -n dmt-emotions python src/verify_rvt_interaction.py

Outputs -> results/lme_random_slopes/rvt_verification.{txt,csv}
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List

import pandas as pd
import statsmodels.formula.api as smf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "results", "lme_random_slopes")

FIXED = "y ~ State * Dose + window_c + State:window_c + Dose:window_c"
KEY = "State[T.DMT]:Dose[T.High]"
OPTIMISERS = ["lbfgs", "bfgs", "cg", "powell", "nm"]

MODALITIES = {
    "HR":      ("results/ecg/hr/hr_minute_long_data_z.csv",         "HR"),
    "SMNA":    ("results/eda/smna/smna_auc_long_data_z.csv",        "AUC"),
    "RVT":     ("results/resp/rvt/resp_rvt_minute_long_data_z.csv", "RSP_RVT"),
    "Arousal": ("results/composite/arousal_index_long.csv",         "ArousalIndex"),
}


def load(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(REPO, path))
    if "Scale" in df.columns:
        df = df[df["Scale"] == "z"]
    df = df.rename(columns={col: "y"})[["subject", "State", "Dose", "window", "y"]].dropna()
    df["State"] = pd.Categorical(df["State"], categories=["RS", "DMT"], ordered=True)
    df["Dose"] = pd.Categorical(df["Dose"], categories=["Low", "High"], ordered=True)
    df["window_c"] = df["window"] - df["window"].mean()
    df["State_n"] = (df["State"] == "DMT").astype(float)
    df["Dose_n"] = (df["Dose"] == "High").astype(float)
    return df.reset_index(drop=True)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    lines: List[str] = []
    rows: List[Dict] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    for mod, (path, col) in MODALITIES.items():
        df = load(path, col)
        say("=" * 74)
        say(f"{mod}   ({df.subject.nunique()} participants)")
        say("=" * 74)

        # -- same model, several optimisers ---------------------------------
        for re_f, label in [("0 + State_n + Dose_n + State_n:Dose_n", "M3"),
                            ("0 + State_n + Dose_n + State_n:Dose_n + window_c", "M4")]:
            say(f"  {label}  ({re_f})")
            for opt in OPTIMISERS:
                try:
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        r = smf.mixedlm(FIXED, df, groups=df["subject"],
                                        re_formula=re_f).fit(reml=True, method=opt,
                                                             maxiter=4000)
                    b, se = float(r.params[KEY]), float(r.bse[KEY])
                    sing = any("singular" in str(x.message).lower() for x in w)
                    say(f"    {opt:7s} loglik={r.llf:10.1f}  beta={b:+.3f}  SE={se:.4f}  "
                        f"CI=[{b-1.96*se:+.3f},{b+1.96*se:+.3f}]"
                        f"{'  SINGULAR' if sing else ''}")
                    rows.append(dict(modality=mod, model=label, optimiser=opt,
                                     loglik=float(r.llf), beta=b, se=se,
                                     ci_lo=b - 1.96 * se, ci_hi=b + 1.96 * se,
                                     singular=sing))
                except Exception as exc:                        # noqa: BLE001
                    say(f"    {opt:7s} FAILED: {str(exc)[:70]}")

        say()

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "rvt_verification.csv"), index=False)
    with open(os.path.join(OUTDIR, "rvt_verification.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"-> {OUTDIR}/rvt_verification.txt")


if __name__ == "__main__":
    main()
