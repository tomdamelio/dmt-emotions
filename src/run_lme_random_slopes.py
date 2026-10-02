"""Maximal LME with by-subject random slopes, and prior DMT use as a moderator.

Answers two referee requests from the Communications Biology revision at once,
because they are the same refit:

  R3.3 (Reviewer 3) -- "the LME form used (random intercept only) is not
      considered best practice - see e.g. Barr, D. J., Levy, R., Scheepers, C.,
      & Tily, H. J. (2013)".

  R1.1b (Reviewer 1) -- "analyses could be corrected for this previous mean
      use", i.e. does prior DMT experience moderate the physiological response
      (the tolerance hypothesis).

WHY THE TWO ARE ONE ANALYSIS
    R3.3 asks whether the size of the response varies across participants.
    R1.1b asks whether prior use explains that variation. Same variance, looked
    at twice: one measures it, the other tries to explain it.

WHY THE RANDOM INTERCEPT IS DROPPED, NOT JUST REDUCED AWAY
    All outcomes are z-scored within participant across that participant's four
    sessions, which sets every participant's mean to zero. The between-subject
    intercept variance is therefore zero *by construction*, not by accident:
    measured on HR, the between-subject SD of the per-subject mean is 0.094
    against a within-subject SD of 1.005 (ICC = 0.009), and the current model
    duly returns a singular random-effects covariance. Keeping the intercept
    costs one variance plus k covariances to estimate a quantity known to be
    zero. Barr's principle applies here to SLOPES, not intercepts.

MODEL LADDER (fixed effects identical throughout, so REML likelihoods are
comparable across random-effects structures):

    M0  ~ 1 | subject                                current published model
    M1  ~ 0 + State_n | subject                      1 random slope
    M2  ~ 0 + State_n + Dose_n | subject             2
    M3  ~ 0 + State_n * Dose_n | subject             3
    M4  ~ 0 + State_n * Dose_n + window_c | subject  4  maximal, slopes only
    M4i ~ 1 + State_n * Dose_n + window_c | subject  5  Barr maximal with intercept

    State_n and Dose_n are numeric 0/1. This matters: with the categorical,
    "0 + State" is cell-means coding and gives one random effect per level,
    which is the intercept-plus-slope space under another name.

WHY THE LRT BASELINE IS M1 AND NOT M0
    M0's random variance collapses to exactly zero, and statsmodels then returns
    llf = +inf (under REML and ML alike), so no likelihood ratio can be formed
    against it. That degeneracy is itself the finding and is reported as such;
    the ladder is therefore tested from M1 upwards.

CAVEAT ON THE LRT
    Testing whether a variance is zero puts the null on the boundary of the
    parameter space, so the naive chi-square p-value is CONSERVATIVE (the true
    null is a mixture of chi-squares). A significant result is therefore safe to
    report; a non-significant one is weaker evidence than it looks.

Usage:
    micromamba run -n dmt-emotions python src/run_lme_random_slopes.py

PER-PARTICIPANT CONTRASTS
    For each modality, each participant's State x Dose contrast
    d_i = (DMT_High - DMT_Low) - (RS_High - RS_Low), on their mean z-scored
    value across the 18 analysis windows, is saved with their prior DMT use
    (per_subject_contrasts.csv), and summarised as M, SD, Cohen's d = M / SD
    (the definition in Methods), number of positive participants and range
    (per_subject_contrasts_summary.csv, also in report.txt). These are the
    per-participant numbers quoted in Results and in the reply to R1.1b.
    The report also gives the ratio of the State x Dose SE under M3 to that
    under M0 (how much the random slopes widen the interval).

Outputs -> results/lme_random_slopes/
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lme_fit import fit_lbfgs_powell  # noqa: E402
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import statsmodels.formula.api as smf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTDIR = os.path.join(REPO, "results", "lme_random_slopes")
PRIOR_USE_TSV = os.path.join(REPO, "metadata", "prior_dmt_use_session1.tsv")

FIXED = "{y} ~ State * Dose + window_c + State:window_c + Dose:window_c"

# The random-effects formula uses NUMERIC State_n / Dose_n, not the categoricals.
# With a categorical, "0 + State" is cell-means coding and yields one random
# effect per level -- i.e. it silently keeps the intercept, just reparametrised,
# so nothing is saved and the parameter count does not match the argument. With
# a 0/1 numeric, "0 + State_n" is a genuine single random slope.
RANDOM_STRUCTURES = [
    ("M0_intercept",    "1",                                              1),
    ("M1_State",        "0 + State_n",                                    1),
    ("M2_State_Dose",   "0 + State_n + Dose_n",                           2),
    ("M3_StateXDose",   "0 + State_n + Dose_n + State_n:Dose_n",          3),
    ("M4_maximal",      "0 + State_n + Dose_n + State_n:Dose_n + window_c", 4),
    ("M4i_maximal_int", "1 + State_n + Dose_n + State_n:Dose_n + window_c", 5),
]

MODALITIES = {
    "HR":       ("results/ecg/hr/hr_minute_long_data_z.csv",              "HR"),
    "SMNA":     ("results/eda/smna/smna_auc_long_data_z.csv",             "AUC"),
    "RVT":      ("results/resp/rvt/resp_rvt_minute_long_data_z.csv",      "RSP_RVT"),
    "Arousal":  ("results/composite/arousal_index_long.csv",              "ArousalIndex"),
}

# The term the paper leads with: the dose-dependent amplification during DMT.
KEY_TERM = "State[T.DMT]:Dose[T.High]"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_modality(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(REPO, path))
    if "Scale" in df.columns:
        df = df[df["Scale"] == "z"]
    df = df.rename(columns={col: "y"})
    df = df[["subject", "State", "Dose", "window", "y"]].dropna()
    df["State"] = pd.Categorical(df["State"], categories=["RS", "DMT"], ordered=True)
    df["Dose"] = pd.Categorical(df["Dose"], categories=["Low", "High"], ordered=True)
    df["window_c"] = df["window"] - df["window"].mean()
    df["State_n"] = (df["State"] == "DMT").astype(float)
    df["Dose_n"] = (df["Dose"] == "High").astype(float)
    return df.reset_index(drop=True)


def attach_prior_use(df: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(PRIOR_USE_TSV, sep="\t")
    prior["subject"] = prior["participant_id"].str.replace("sub-", "S", regex=False)
    prior = prior.set_index("subject")["prior_dmt_use"]
    out = df.copy()
    out["prior_use"] = out["subject"].map(prior)
    # Centre so that the State and Dose coefficients keep their meaning at the
    # sample's average level of experience rather than at zero occasions.
    out["prior_use_c"] = out["prior_use"] - out["prior_use"].mean()
    return out.dropna(subset=["prior_use"])


def per_subject_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Each participant's State x Dose contrast on cell means across windows:
    (DMT_High - DMT_Low) - (RS_High - RS_Low), with the two simple dose effects.
    """
    cm = (df.groupby(["subject", "State", "Dose"], observed=True)["y"].mean()
          .unstack(["State", "Dose"]))
    out = pd.DataFrame({
        "dose_DMT": cm[("DMT", "High")] - cm[("DMT", "Low")],
        "dose_RS": cm[("RS", "High")] - cm[("RS", "Low")],
    })
    out["state_x_dose"] = out["dose_DMT"] - out["dose_RS"]
    return out.dropna().rename_axis("subject").reset_index()


def summarise_contrast(x: pd.Series) -> Dict:
    """M, SD, Cohen's d = M / SD (Methods definition), positives, range."""
    m, sd = float(x.mean()), float(x.std(ddof=1))
    return {"n": int(len(x)), "mean": m, "sd": sd,
            "cohens_d": m / sd if sd > 0 else np.nan,
            "n_positive": int((x > 0).sum()),
            "min": float(x.min()), "max": float(x.max())}


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit(df: pd.DataFrame, fixed: str, re_formula: Optional[str]) -> Tuple[Optional[object], Dict]:
    info: Dict = {"re_formula": re_formula, "warnings": [], "converged": False}
    try:
        model = smf.mixedlm(fixed, df, groups=df["subject"], re_formula=re_formula)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res, info["optimiser"] = fit_lbfgs_powell(model)
            info["warnings"] = sorted({str(w.message)[:90] for w in caught})
    except Exception as exc:                                    # noqa: BLE001
        info["error"] = str(exc)[:160]
        return None, info

    singular = any("singular" in w.lower() for w in info["warnings"])
    info.update(
        converged=bool(getattr(res, "converged", True)) and not singular,
        singular=singular,
        loglik=float(res.llf),
        n_re_params=int(res.cov_re.shape[0] * (res.cov_re.shape[0] + 1) / 2),
        re_var=np.diag(np.atleast_2d(res.cov_re)).tolist(),
        re_names=list(np.atleast_2d(res.cov_re).shape and res.cov_re.columns)
        if hasattr(res.cov_re, "columns") else [],
        resid_var=float(res.scale),
    )
    return res, info


def check_stable(df: pd.DataFrame, fixed: str, re_formula: str,
                 optimisers=("lbfgs", "bfgs", "powell")) -> Tuple[bool, List[float]]:
    """Refit under several optimisers. A model that actually converged returns
    the same log-likelihood whichever one is used; a spread of more than a
    likelihood point means the optimiser, not the data, is choosing the answer.
    """
    lls: List[float] = []
    for opt in optimisers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(fixed, df, groups=df["subject"],
                                  re_formula=re_formula).fit(reml=True, method=opt,
                                                             maxiter=4000)
            if np.isfinite(res.llf):
                lls.append(float(res.llf))
        except Exception:                                       # noqa: BLE001
            continue
    if len(lls) < 2:
        return False, lls
    return (max(lls) - min(lls)) < 1.0, lls


def lrt(ll_small: float, ll_big: float, df_diff: int) -> Tuple[float, float]:
    """Likelihood-ratio test. p is conservative for variance parameters."""
    stat = 2.0 * (ll_big - ll_small)
    if stat < 0 or df_diff <= 0:
        return stat, float("nan")
    return stat, float(stats.chi2.sf(stat, df_diff))


def term_row(res, term: str) -> Dict:
    if res is None or term not in res.params.index:
        return {}
    b = float(res.params[term])
    se = float(res.bse[term])
    return {"beta": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
            "z": b / se if se else np.nan,
            "p": float(2 * stats.norm.sf(abs(b / se))) if se else np.nan,
            "ci_width": 2 * 1.96 * se}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    ladder_rows: List[Dict] = []
    prior_rows: List[Dict] = []
    contrast_rows: List[pd.DataFrame] = []
    contrast_summary: List[Dict] = []
    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    for mod, (path, col) in MODALITIES.items():
        df = load_modality(path, col)
        fixed = FIXED.format(y="y")
        say("=" * 78)
        say(f"{mod}:  {len(df)} observations, {df.subject.nunique()} subjects")
        say("=" * 78)

        fits: Dict[str, Tuple[Optional[object], Dict]] = {}
        for name, re_f, _k in RANDOM_STRUCTURES:
            res, info = fit(df, fixed, re_f)
            fits[name] = (res, info)
            key = term_row(res, KEY_TERM)
            status = ("ok" if info.get("converged") else
                      ("SINGULAR" if info.get("singular") else "FAILED"))
            row = dict(modality=mod, model=name, re_formula=re_f, status=status,
                       loglik=info.get("loglik", np.nan),
                       n_re_params=info.get("n_re_params", np.nan), **key)
            ladder_rows.append(row)
            if key:
                say(f"  {name:20s} {status:9s} loglik={info.get('loglik', float('nan')):10.1f}  "
                    f"StateXDose beta={key['beta']:+.3f}  SE={key['se']:.4f}  "
                    f"CI=[{key['ci_lo']:+.3f},{key['ci_hi']:+.3f}]")
            else:
                say(f"  {name:20s} {status:9s} {info.get('error', '')}")

        # --- LRT, rung by rung ------------------------------------------------
        # M0 (and any rung whose variance collapses to exactly zero) returns
        # llf = +inf under both REML and ML, so it cannot serve as a baseline.
        # Each rung is therefore tested against the previous one that produced a
        # finite likelihood; the degenerate rungs are reported as degenerate,
        # which is itself the answer to the referee.
        say("\n  LRT, each rung against the previous fittable one"
            " -- p conservative at the boundary:")
        prev = None
        for name, _re, _k in RANDOM_STRUCTURES:
            info = fits[name][1]
            ll = info.get("loglik")
            if ll is None or not np.isfinite(ll):
                say(f"    {name:20s} degenerate (variance -> 0, loglik = {ll})")
                continue
            if prev is not None:
                d = info["n_re_params"] - fits[prev][1]["n_re_params"]
                stat, p = lrt(fits[prev][1]["loglik"], ll, d)
                say(f"    {prev:16s} -> {name:16s} chi2({d:2d}) = {stat:8.2f}   p = {p:.4g}")
            prev = name

        # --- prior use as a moderator ----------------------------------------
        # Fitted on the richest converged random structure, because the term of
        # interest, prior_use x State, is a between-subject comparison of a
        # within-subject effect: its standard error is only honest once the
        # by-subject random slope for State absorbs how much that effect varies
        # across participants.
        #
        # The prior_use MAIN effect is a different matter: it is purely
        # between-subject, and within-subject z-scoring pins every participant's
        # mean to zero, so there is no between-subject variance for it to
        # explain. The model says so itself, returning beta = 0 with a standard
        # error that is nan or astronomically large. That is not a failure to
        # report as a result; it is the design, and it is why the reply to
        # Reviewer 1 can only offer an interaction.
        # Structure "1 + State_n": the intercept makes the between-subject main
        # effect identifiable, the State slope makes the interaction's standard
        # error honest, and two random terms is small enough to converge from 11
        # participants. Fitting these models on a slopes-only structure gives a
        # visibly wrong answer -- the prior_use main effect comes out at
        # p < .001 even though within-subject z-scoring pins it to zero -- which
        # is the same pseudoreplication error, arriving through the intercept.
        #
        # Each specification gets the random structure its own terms require.
        # The two-way needs the intercept and the State slope. The three-way
        # additionally involves Dose and State x Dose, so those slopes have to be
        # there too or their standard errors are anticonservative for the same
        # reason. That is four random terms from 11 participants, so it is
        # checked for stability and reported as unfittable if it is not.
        say("\n  prior DMT use ('*' and not ':': with ':' patsy codes State with "
            "full dummy\n  coding inside the interaction and returns one "
            "coefficient per level, not the contrast)")
        dfp = attach_prior_use(df)
        SPECS = [
            ("State",      " + prior_use_c * State",          "1 + State_n"),
            ("StateXDose", " + prior_use_c * State * Dose",
             "1 + State_n + Dose_n + State_n:Dose_n"),
        ]
        for label, extra, re_f in SPECS:
            stable, lls = check_stable(dfp, fixed + extra, re_f)
            spread = f"{max(lls) - min(lls):.2f}" if len(lls) >= 2 else "n/a"
            verdict = "stable" if stable else "UNSTABLE across optimisers"
            say(f"    [{label}]  random: {re_f}   {verdict} (loglik spread {spread})")
            res, info = fit(dfp, fixed + extra, re_f)
            if res is None:
                say(f"      FAILED: {info.get('error','')}")
                continue
            for term in [t for t in res.params.index if "prior_use_c" in t]:
                k = term_row(res, term)
                prior_rows.append(dict(modality=mod, spec=label, term=term,
                                       random=re_f, stable=stable,
                                       n_subjects=dfp.subject.nunique(), **k))
                say(f"      {term:40s} beta={k['beta']:+.4f}  SE={k['se']:.4f}  "
                    f"p={k['p']:.3f}")

        # Optimiser-free counterpart: does prior use predict each participant's
        # own State x Dose interaction? Same question, 11 numbers, no fitting.
        d_i = (dfp.groupby(["subject", "State", "Dose"], observed=True)["y"].mean()
               .unstack(["State", "Dose"]))
        d_i = ((d_i[("DMT", "High")] - d_i[("DMT", "Low")])
               - (d_i[("RS", "High")] - d_i[("RS", "Low")])).dropna()
        pu = dfp.groupby("subject", observed=True)["prior_use"].first().reindex(d_i.index)
        rho, p_rho = stats.spearmanr(pu, d_i)
        say(f"      [optimiser-free] Spearman(prior use, per-subject State x Dose) "
            f"= {rho:+.3f}, p = {p_rho:.3f}, n = {len(d_i)}")
        prior_rows.append(dict(modality=mod, spec="per-subject-spearman",
                               term="prior_use vs StateXDose", random="none",
                               stable=True, n_subjects=len(d_i),
                               beta=float(rho), p=float(p_rho)))

        # --- per-participant contrasts (all participants, prior use attached
        # where known) ----------------------------------------------------------
        pc = per_subject_contrasts(df)
        prior = pd.read_csv(PRIOR_USE_TSV, sep="\t")
        prior.index = prior["participant_id"].str.replace("sub-", "S", regex=False)
        pc["prior_dmt_use"] = pc["subject"].map(prior["prior_dmt_use"])
        pc.insert(0, "modality", mod)
        pc["rank_state_x_dose"] = (pc["state_x_dose"].rank(ascending=False, method="min")
                                   .astype(int))
        contrast_rows.append(pc)
        say("\n  per-participant contrasts (mean across windows; "
            "Cohen's d = M / SD of the per-participant contrast)")
        for cname, label in [("state_x_dose", "State x Dose"),
                             ("dose_DMT", "High - Low in DMT"),
                             ("dose_RS", "High - Low in RS")]:
            sm = summarise_contrast(pc[cname])
            contrast_summary.append(dict(modality=mod, contrast=cname, **sm))
            say(f"    {label:18s} n={sm['n']:2d}  M={sm['mean']:+.4f}  SD={sm['sd']:.4f}  "
                f"d={sm['cohens_d']:+.3f}  positive {sm['n_positive']}/{sm['n']}  "
                f"range [{sm['min']:+.3f}, {sm['max']:+.3f}]")
        top = pc.sort_values("state_x_dose", ascending=False).head(2)
        for _, r in top.iterrows():
            say(f"    rank {r['rank_state_x_dose']} State x Dose: {r['subject']}  "
                f"{r['state_x_dose']:+.3f}  (prior DMT use {r['prior_dmt_use']:g})")

    # --- SE inflation from the random slopes: M3 against M0 -------------------
    lad = pd.DataFrame(ladder_rows).set_index(["modality", "model"])
    say("=" * 78)
    say("State x Dose SE, M3 (random slopes) / M0 (random intercept)")
    say("=" * 78)
    ratios = []
    for mod in MODALITIES:
        se0 = lad.loc[(mod, "M0_intercept"), "se"]
        se3 = lad.loc[(mod, "M3_StateXDose"), "se"]
        ratios.append(se3 / se0)
        say(f"  {mod:8s} SE M0 = {se0:.4f}  SE M3 = {se3:.4f}  ratio = {se3 / se0:.2f}")
    say(f"  range of ratios: {min(ratios):.2f} to {max(ratios):.2f}")

    pd.concat(contrast_rows).to_csv(os.path.join(OUTDIR, "per_subject_contrasts.csv"),
                                    index=False)
    pd.DataFrame(contrast_summary).to_csv(
        os.path.join(OUTDIR, "per_subject_contrasts_summary.csv"), index=False)
    pd.DataFrame(ladder_rows).to_csv(os.path.join(OUTDIR, "model_ladder.csv"), index=False)
    pd.DataFrame(prior_rows).to_csv(os.path.join(OUTDIR, "prior_use_moderation.csv"), index=False)
    with open(os.path.join(OUTDIR, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n-> {OUTDIR}")


if __name__ == "__main__":
    main()
