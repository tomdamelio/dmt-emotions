"""Time-domain heart rate variability and the intrinsic heart-rate ceiling.

Answers two referee requests from the Communications Biology revision:

  R1.5 (Reviewer 1)  analyse SDNN and RMSSD as parasympathetic indices.
  R3.1 (Reviewer 3)  without a parasympathetic measure the manuscript cannot
                     assert vagal withdrawal; intrinsic-rate bound as the
                     alternative argument.

Pipeline (all steps fixed before any result was looked at):

  1. R peaks: the Kubios-corrected peaks stored in *_info.json (ECG_R_Peaks).
  2. Beat-level artefact handling. An RR interval is invalid if it is outside
     300-2000 ms or deviates > 25 % from the median of its 11 nearest
     neighbours. Invalid intervals are discarded, never interpolated, and
     successive differences are taken only within runs of consecutive valid
     intervals, so no difference spans a gap. A recording is excluded if more
     than 10 % of its intervals are invalid (applies to one: S16 DMT Low,
     22.7 %). A participant is kept only with all four recordings valid.
  3. Windows: 120 s, sliding in 30 s steps; DMT to 19 min, RS to 9 min. A
     window is kept if valid intervals cover >= 80 % of it and number >= 60.
  4. Metrics per window: meanRR, HR (bpm), RMSSD, RMSSD/meanRR, SDNN.
  5. Standardisation at the WINDOW level within participant, over all window
     values of the four recordings (same convention as the other modalities
     since the revision).
  6. LME with by-participant random slopes for State, Dose and State:Dose
     (statsmodels, REML, Wald p), on 0-9 min. Exact sign-flip cluster test
     High vs Low within DMT (one-tailed, 'less' for variability measures,
     'greater' for HR) and DMT vs RS (two-tailed).
  7. Intrinsic heart rate per participant, 118.1 - 0.57 * age (Jose &
     Collison 1970), against peak HR in 60-s windows sliding in 5 s. Counts
     against the estimate and its upper 95% limit (+15%, the least
     favourable case for the sympathetic inference), and a sample-level
     one-tailed test of the margin (t and Wilcoxon).

Outputs -> results/ecg/hrv/
    hrv_windows_long.csv        one row per participant x recording x window
    hrv_qc.txt                  invalid-interval rates and exclusions
    hrv_lme_cluster_report.txt  fixed effects, cluster segments and the descriptives
                                quoted in the manuscript (absolute HR, raw HRV, audit)
    intrinsic_ceiling.csv       per participant and dose
    recording_audit.csv         per ECG recording: Kubios-corrected beats, invalid
                                intervals, change in mean HR from the filter
    hrv_summary.json            the numbers quoted in the manuscript

Figure: make_figure(path) draws the four-panel Supplementary Fig. 4.
Usage:
    micromamba run -n dmt-emotions python src/run_ecg_hrv_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lme_fit import fit_lbfgs_powell  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import (DERIVATIVES_DATA, SUJETOS_VALIDADOS_ECG,  # noqa: E402
                    EDAD_SUJETO, EDAD_MEDIA_MUESTRA, get_edad_sujeto)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUTDIR = os.path.join(REPO, "results", "ecg", "hrv")
ECG_DIR = os.path.join(DERIVATIVES_DATA, "phys", "ecg")

# ---- fixed analysis parameters ---------------------------------------------
WIN, STEP = 120.0, 30.0
MAX_T = {"DMT": 19 * 60.0, "RS": 9 * 60.0}
ANALYSIS_END = 540.0                     # 0-9 min for models and clusters
RR_LO, RR_HI, REL_DEV = 300.0, 2000.0, 0.25
MIN_VALID_FRAC, MIN_BEATS = 0.80, 60
MAX_REC_INVALID = 0.10
VARS = ["HR_bpm", "RMSSD", "RMSSD_n", "SDNN"]
TAIL = {"HR_bpm": "greater", "RMSSD": "less", "RMSSD_n": "less", "SDNN": "less"}
FIXED = "y ~ State * Dose + window_c + State:window_c + Dose:window_c"
RE_SLOPES = "0 + State_n + Dose_n + State_n:Dose_n"
IHR_CI = 0.15                            # 95% limits for individuals of the same age, +-15%
                                         # (Jose & Collison 1970, Fig. 1; Opthof 2000)

NAME_RE = re.compile(r"^(S\d+)_(dmt|rs)_(session\d)_(high|low)_info\.json$")


# =============================================================================
# 1-4  extraction
# =============================================================================
def valid_mask(rr: np.ndarray) -> np.ndarray:
    loc = pd.Series(rr).rolling(11, center=True, min_periods=5).median().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(rr - loc) / loc
    return (rr >= RR_LO) & (rr <= RR_HI) & (rel <= REL_DEV)


def window_metrics(rr: np.ndarray, ok: np.ndarray) -> Dict[str, float]:
    """RMSSD over runs of consecutive valid intervals; meanRR/SDNN over valid."""
    sq, n_d, start = 0.0, 0, None
    for i, good in enumerate(np.append(ok, False)):
        if good and start is None:
            start = i
        elif not good and start is not None:
            run = rr[start:i]
            if len(run) > 1:
                d = np.diff(run)
                sq += float((d ** 2).sum())
                n_d += len(d)
            start = None
    if n_d == 0 or ok.sum() < 2:
        return {}
    mean_rr = float(rr[ok].mean())
    rmssd = float(np.sqrt(sq / n_d))
    return dict(meanRR=mean_rr, HR_bpm=60000.0 / mean_rr, RMSSD=rmssd,
                RMSSD_n=rmssd / mean_rr, SDNN=float(rr[ok].std(ddof=1)))


def load_peaks(path: str) -> Tuple[np.ndarray, float]:
    info = json.load(open(path))
    fs = float(info.get("sampling_rate", 250.0))
    return np.asarray(info["ECG_R_Peaks"], dtype=float) / fs, fs


def extract_windows() -> Tuple[pd.DataFrame, List[str], List[Tuple[str, str, str, float]]]:
    rows, qc, excluded = [], [], []
    for f in sorted(glob.glob(os.path.join(ECG_DIR, "dmt_*", "*_info.json"))):
        m = NAME_RE.match(os.path.basename(f))
        if not m or m.group(1) not in SUJETOS_VALIDADOS_ECG:
            continue
        subj, state, session, dose = m.groups()
        state, dose = state.upper(), dose.capitalize()
        t_pk, _ = load_peaks(f)
        rr_all = np.diff(t_pk) * 1000.0
        ok_all = valid_mask(rr_all)
        frac_bad = float(1 - ok_all.mean())
        qc.append(f"{subj} {state:3s} {dose:4s} {session}: {len(rr_all)} intervals, "
                  f"{100 * frac_bad:.2f} % invalid"
                  + ("   -> RECORDING EXCLUDED" if frac_bad > MAX_REC_INVALID else ""))
        if frac_bad > MAX_REC_INVALID:
            excluded.append((subj, state, dose, 100 * frac_bad))
            continue
        t_end = t_pk[1:]
        for w, t0 in enumerate(np.arange(0.0, MAX_T[state] - WIN + STEP, STEP)):
            sel = (t_end >= t0) & (t_end < t0 + WIN)
            rr, ok = rr_all[sel], ok_all[sel]
            rec = dict(subject=subj, State=state, Dose=dose, session=session,
                       window=w + 1, t_start=t0, t_center=t0 + WIN / 2, t_end=t0 + WIN,
                       n_rr=int(sel.sum()), n_valid=int(ok.sum()),
                       valid_frac=float(rr[ok].sum() / 1000.0 / WIN) if ok.any() else 0.0)
            if rec["n_valid"] >= MIN_BEATS and rec["valid_frac"] >= MIN_VALID_FRAC:
                rec.update(window_metrics(rr, ok))
            rows.append(rec)
    return pd.DataFrame(rows), qc, excluded


def complete_subjects(df: pd.DataFrame) -> List[str]:
    n_rec = df.groupby("subject").apply(
        lambda g: g.groupby(["State", "Dose"]).ngroups, include_groups=False)
    return sorted(n_rec[n_rec == 4].index)


# =============================================================================
# 5-6  standardisation, LME, clusters
# =============================================================================
def zscore_within_subject(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for v in VARS:
        mu = df.groupby("subject")[v].transform("mean")
        sd = df.groupby("subject")[v].transform(lambda s: s.std(ddof=1))
        df[v + "_z"] = (df[v] - mu) / sd
    return df


def fit_lme(d9: pd.DataFrame, y: str, re_formula: str):
    import statsmodels.formula.api as smf
    d = d9.dropna(subset=[y]).rename(columns={y: "y"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(FIXED, d, groups=d["subject"], re_formula=re_formula)
        return fit_lbfgs_powell(model)[0]


def matrix(df: pd.DataFrame, subjects: List[str], y: str, state: str, dose: str,
           t_end_max: float) -> Tuple[np.ndarray, np.ndarray]:
    d = df[(df.State == state) & (df.Dose == dose) & (df.t_end <= t_end_max)]
    w = d.pivot_table(index="subject", columns="t_center", values=y).reindex(subjects)
    return w.to_numpy(), np.array(w.columns) / 60.0


def run_models(df: pd.DataFrame, subjects: List[str], say) -> Dict:
    from cluster_stats import cluster_test, segments
    d9 = df[(df.t_end <= ANALYSIS_END) & (df.subject.isin(subjects))].copy()
    d9["State"] = pd.Categorical(d9.State, ["RS", "DMT"], ordered=True)
    d9["Dose"] = pd.Categorical(d9.Dose, ["Low", "High"], ordered=True)
    d9["State_n"] = (d9.State == "DMT").astype(float)
    d9["Dose_n"] = (d9.Dose == "High").astype(float)
    d9["window_c"] = d9.window - d9.window.mean()

    out: Dict = {"lme": {}, "clusters": {}}
    say("=" * 78 + "\nLME, 0-9 min, window-level z, by-participant random slopes\n" + "=" * 78)
    say(f"subjects {d9.subject.nunique()}, rows {len(d9)}")
    for v in VARS:
        r = fit_lme(d9, v + "_z", RE_SLOPES)
        say(f"\n--- {v} ---   loglik={r.llf:.1f}")
        out["lme"][v] = {}
        for term, name in [("State[T.DMT]", "State"), ("Dose[T.High]", "Dose"),
                           ("State[T.DMT]:Dose[T.High]", "StateXDose")]:
            b, se, p = float(r.params[term]), float(r.bse[term]), float(r.pvalues[term])
            out["lme"][v][name] = dict(beta=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se], p=p)
            say(f"    {name:12s} beta={b:+.3f} CI=[{b-1.96*se:+.3f},{b+1.96*se:+.3f}] p={p:.4f}")

    say("\n" + "=" * 78 + "\nCluster permutation (exact sign-flip)\n" + "=" * 78)
    for label, state, t_max in [("DMT 0-9", "DMT", ANALYSIS_END), ("RS 0-9", "RS", ANALYSIS_END),
                                ("DMT 0-19", "DMT", MAX_T["DMT"])]:
        say(f"\n--- High vs Low, {label} min ---")
        for v in VARS:
            H, x = matrix(df, subjects, v + "_z", state, "High", t_max)
            L, _ = matrix(df, subjects, v + "_z", state, "Low", t_max)
            keep = ~np.isnan(H).any(axis=1) & ~np.isnan(L).any(axis=1)
            alt = TAIL[v] if state == "DMT" else "two-sided"
            res = cluster_test(H[keep], L[keep], alt)
            segs = [(round(a, 1), round(b, 1)) for a, b in segments(res["sig"], x)]
            cl = [dict(windows=c["windows"], mass=round(c["mass"], 2), p=round(c["p"], 4))
                  for c in res["clusters"]]
            out["clusters"][f"{v}|HighVsLow|{label}"] = dict(n=int(keep.sum()), tail=alt,
                                                             segments=segs, clusters=cl)
            say(f"  {v:8s} n={keep.sum():2d} tail={alt:9s} sig={segs}  "
                f"p={[c['p'] for c in cl]}")
    say("\n--- DMT vs RS (doses averaged), 0-9 min, two-sided ---")
    for v in VARS:
        d = df[(df.t_end <= ANALYSIS_END) & (df.subject.isin(subjects))]
        D = d[d.State == "DMT"].pivot_table(index="subject", columns="t_center", values=v + "_z").reindex(subjects)
        R = d[d.State == "RS"].pivot_table(index="subject", columns="t_center", values=v + "_z").reindex(subjects)
        A, B = D.to_numpy(), R.to_numpy()
        keep = ~np.isnan(A).any(axis=1) & ~np.isnan(B).any(axis=1)
        res = cluster_test(A[keep], B[keep], "two-sided")
        x = np.array(D.columns) / 60.0
        segs = [(round(a, 1), round(b, 1)) for a, b in segments(res["sig"], x)]
        cl = [dict(windows=c["windows"], mass=round(c["mass"], 2), p=round(c["p"], 4))
              for c in res["clusters"]]
        out["clusters"][f"{v}|DMTvsRS|0-9"] = dict(n=int(keep.sum()), tail="two-sided",
                                                   segments=segs, clusters=cl)
        say(f"  {v:8s} n={keep.sum():2d} sig={segs}  p={[c['p'] for c in cl]}")
    return out


# =============================================================================
# 7  intrinsic heart-rate ceiling (all ECG participants; HR is robust to the
#    detection noise that excludes S16 Low from the variability analyses)
# =============================================================================
def intrinsic_ceiling(say) -> pd.DataFrame:
    if not EDAD_SUJETO:
        say(f"\n[WARNING] metadata/participants_age.tsv not found: every participant is "
            f"assigned the group mean age ({EDAD_MEDIA_MUESTRA} y). Per-participant ceiling "
            f"counts will differ from the manuscript; see config.py.")
    rows = []
    for f in sorted(glob.glob(os.path.join(ECG_DIR, "dmt_*", "*_info.json"))):
        m = NAME_RE.match(os.path.basename(f))
        if not m or m.group(1) not in SUJETOS_VALIDADOS_ECG or m.group(2) != "dmt":
            continue
        subj, _, _, dose = m.groups()
        t, _ = load_peaks(f)
        rr, tm = np.diff(t) * 1000.0, t[1:]
        ok = valid_mask(rr)
        age = float(get_edad_sujeto(subj))
        rec = dict(subject=subj, Dose=dose.capitalize(), age=age, IHR=118.1 - 0.57 * age)
        for win in (120, 60, 30):
            best = np.nan
            for t0 in np.arange(0, ANALYSIS_END - win + 1, 5):
                sel = (tm >= t0) & (tm < t0 + win) & ok
                if sel.sum() >= max(8, win // 6):
                    hr = 60000.0 / rr[sel].mean()
                    best = hr if not np.isfinite(best) else max(best, hr)
            rec[f"peak_{win}s"] = best
        rs = f.replace("_dmt_", "_rs_")
        if os.path.exists(rs):
            t2, _ = load_peaks(rs)
            rr2 = np.diff(t2) * 1000.0
            o2 = valid_mask(rr2) & (t2[1:] <= ANALYSIS_END)
            rec["rest_HR"] = 60000.0 / rr2[o2].mean()
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["margin_60s"] = df.peak_60s - df.IHR
    # The estimate can err in either direction for one person; the direction
    # that could undo the inference is a true IHR above the estimate, so the
    # conservative count uses the upper 95% limit, not a bound below it.
    df["above_IHR_upper95"] = df.peak_60s > df.IHR * (1 + IHR_CI)
    say("\n" + "=" * 78 + "\nIntrinsic heart-rate ceiling (IHR = 118.1 - 0.57 x age)\n" + "=" * 78)
    for dose in ["High", "Low"]:
        d = df[df.Dose == dose]
        st = ceiling_test(d.margin_60s)
        say(f"  {dose:4s}: rest {d.rest_HR.mean():.1f} bpm, peak(60 s) M={d.peak_60s.mean():.1f} "
            f"SD={d.peak_60s.std(ddof=1):.1f}; above own IHR {int((d.margin_60s > 0).sum())}/{len(d)}, "
            f"above IHR+15% {int(d.above_IHR_upper95.sum())}/{len(d)}; "
            f"margin M={st['mean_margin']:+.1f} (median {st['median_margin']:+.1f}) bpm, "
            f"t({st['df']})={st['t']:.2f} p={st['p_t']:.3f}, "
            f"Wilcoxon p={st['p_wilcoxon']:.3f} (one-tailed)")
    return df


def ceiling_test(margin: pd.Series) -> Dict[str, float]:
    """Sample-level test that peak HR exceeds the estimated IHR. The per-person
    error of the estimate is part of the spread of the margins, so the test
    carries it instead of assuming it away."""
    from scipy import stats
    m = margin.dropna()
    t = stats.ttest_1samp(m, 0.0, alternative="greater")
    return dict(mean_margin=float(m.mean()), median_margin=float(m.median()),
                df=int(len(m) - 1), t=float(t.statistic), p_t=float(t.pvalue),
                p_wilcoxon=float(stats.wilcoxon(m, alternative="greater").pvalue))


# =============================================================================
# descriptives quoted in the manuscript (Methods, Results, Supplementary
# Information) and in the response letter (R1.5, R3.1). They only read the
# extraction above and the R peaks; no estimate of the models changes.
# =============================================================================
KUBIOS_KEYS = ("ECG_fixpeaks_ectopic", "ECG_fixpeaks_missed",
               "ECG_fixpeaks_extra", "ECG_fixpeaks_longshort")
HR_WIN, HR_END = 30.0, 540.0             # the 30-s windows of run_ecg_hr_analysis, 0-9 min


def recording_audit(say) -> pd.DataFrame:
    """Per-recording quality audit.

    Methods: Kubios-corrected beats over every ECG recording in the dataset
    (all *_info.json, not only the validated participants); share of invalid
    intervals across the 44 recordings of the validated participants; and the
    change in mean HR produced by the variability filter, taken in the 30-s,
    0-9 min windows of the HR analysis as HR from the valid intervals minus HR
    from all Kubios-corrected intervals, averaged over windows.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(ECG_DIR, "dmt_*", "*_info.json"))):
        m = NAME_RE.match(os.path.basename(f))
        if not m:
            continue
        subj, state, session, dose = m.groups()
        info = json.load(open(f))
        n_beats = len(info["ECG_R_Peaks"])
        n_kubios = len({int(i) for k in KUBIOS_KEYS for i in info.get(k, [])})
        rec = dict(subject=subj, State=state.upper(), Dose=dose.capitalize(), session=session,
                   validated=subj in SUJETOS_VALIDADOS_ECG,
                   n_beats=n_beats, n_kubios=n_kubios)
        if rec["validated"]:
            t_pk, _ = load_peaks(f)
            rr = np.diff(t_pk) * 1000.0
            ok, t_end = valid_mask(rr), t_pk[1:]
            rec["pct_invalid"] = 100.0 * float(1 - ok.mean())
            d = []
            for t0 in np.arange(0.0, HR_END, HR_WIN):
                sel = (t_end >= t0) & (t_end < t0 + HR_WIN)
                if (sel & ok).any():
                    d.append(60000.0 / rr[sel & ok].mean() - 60000.0 / rr[sel].mean())
            rec["HR_change_filter_bpm"] = float(np.mean(d))
        rows.append(rec)
    df = pd.DataFrame(rows)
    v = df[df.validated]
    worst = v.sort_values("pct_invalid", ascending=False)
    exc = worst.iloc[0]
    say("\n" + "=" * 78 + "\nRecording audit (Kubios correction, invalid intervals, filter effect on HR)\n" + "=" * 78)
    say(f"  Kubios-corrected beats, all {len(df)} ECG recordings: {int(df.n_kubios.sum())} of "
        f"{int(df.n_beats.sum()):,} ({100 * df.n_kubios.sum() / df.n_beats.sum():.2f} %)")
    say(f"  invalid intervals, {len(v)} recordings of the {v.subject.nunique()} validated participants: "
        f"median {v.pct_invalid.median():.3f} %; highest {exc.subject} {exc.State} {exc.Dose} "
        f"{exc.pct_invalid:.1f} %, next highest {worst.iloc[1].pct_invalid:.1f} %")
    say(f"  mean HR change from the filter (30-s windows, 0-9 min, valid minus all intervals): "
        f"median {v.HR_change_filter_bpm.median():+.2f} bpm across {len(v)} recordings; "
        f"{exc.subject} {exc.State} {exc.Dose} {exc.HR_change_filter_bpm:+.2f} bpm")
    return df


def hr_descriptives(ceiling: pd.DataFrame, say) -> Dict:
    """Absolute heart rate (Results): resting and peak HR by dose, the rise of
    the peak over each participant's own rest, the estimated intrinsic rate,
    and the cardiac response of those who stay below it. Ages are reported
    only as a range."""
    from scipy import stats
    c = ceiling.copy()
    c["rise_60s"] = c.peak_60s - c.rest_HR
    w = {v: c.pivot(index="subject", columns="Dose", values=v)
         for v in ["rest_HR", "peak_60s", "rise_60s"]}
    out: Dict = {}
    say("\n" + "=" * 78 + "\nAbsolute heart rate (bpm), all ECG participants\n" + "=" * 78)
    for v, lab in [("rest_HR", "resting HR (0-9 min)"), ("peak_60s", "peak HR (60-s windows)"),
                   ("rise_60s", "rise of peak over own rest")]:
        t = stats.ttest_rel(w[v].High, w[v].Low)
        out[v] = {d: dict(mean=float(w[v][d].mean()), sd=float(w[v][d].std(ddof=1)),
                          min=float(w[v][d].min()), max=float(w[v][d].max())) for d in ["High", "Low"]}
        out[v]["paired_t"] = dict(df=int(len(w[v]) - 1), t=float(t.statistic), p=float(t.pvalue))
        say(f"  {lab:28s} " + "; ".join(
            f"{d}: M={w[v][d].mean():.1f} SD={w[v][d].std(ddof=1):.1f} "
            f"range {w[v][d].min():.1f}-{w[v][d].max():.1f}" for d in ["High", "Low"])
            + f";  High vs Low t({len(w[v]) - 1})={t.statistic:.2f} p={t.pvalue:.3f}")
    per = c.groupby("subject")[["age", "IHR"]].first()
    out["IHR"] = dict(mean=float(per.IHR.mean()), sd=float(per.IHR.std(ddof=1)),
                      age_min=float(per.age.min()), age_max=float(per.age.max()))
    say(f"  estimated intrinsic rate: M={per.IHR.mean():.1f} SD={per.IHR.std(ddof=1):.1f} "
        f"(age range {per.age.min():.0f}-{per.age.max():.0f} y)")
    hi = c[c.Dose == "High"].set_index("subject")
    below = hi[hi.margin_60s <= 0]
    rank = hi.rise_60s.rank(method="min").astype(int)
    out["below_IHR_high"] = {s: dict(rise_60s=float(hi.rise_60s[s]), rise_rank=int(rank[s]))
                             for s in below.index}
    say(f"  High dose, not above own IHR: " + ", ".join(
        f"{s} (rise {hi.rise_60s[s]:.1f} bpm, rank {rank[s]}/{len(hi)})" for s in below.index)
        + f"; the {len(below)} smallest rises are " + ", ".join(hi.rise_60s.nsmallest(len(below)).index))
    return out


def hrv_descriptives(df: pd.DataFrame, subjects: List[str], say) -> Dict:
    """Variability in raw units (Supplementary Information): median RMSSD by
    window, within-participant dependence of RMSSD on the mean interval
    (DMT windows, 0-9 min) and the late (10-19 min) dose difference in z.
    Restricted to the participants retained for the variability analyses."""
    from scipy import stats
    d = df[df.subject.isin(subjects)]
    d9 = d[d.t_end <= ANALYSIS_END]
    med = d9.groupby(["State", "Dose", "t_center"]).RMSSD.median()
    hi, rs_hi = med.loc[("DMT", "High")], med.loc[("RS", "High")]
    t_lo = hi.idxmin()
    out: Dict = dict(rmssd_median_dmt_high={f"{k / 60:.1f}": float(v) for k, v in hi.items()},
                     rmssd_median_rs_high={f"{k / 60:.1f}": float(v) for k, v in rs_hi.items()})
    say("\n" + "=" * 78 + f"\nHRV in raw units, n = {d.subject.nunique()} (0-9 min)\n" + "=" * 78)
    say("  median RMSSD (ms) by window centre (min):")
    for (state, dose), g in med.groupby(level=[0, 1]):
        say(f"    {state:3s} {dose:4s} " + " ".join(f"{k / 60:.1f}:{v:.1f}" for k, v in g.droplevel([0, 1]).items()))
    say(f"  DMT High: first window {hi.iloc[0]:.1f} ms, minimum {hi.min():.1f} ms at {t_lo / 60:.1f} min, "
        f"{hi.loc[480.0]:.1f} ms at 8.0 min; RS High range {rs_hi.min():.1f}-{rs_hi.max():.1f} ms")

    dd = d9[d9.State == "DMT"]
    r = dd.groupby("subject").apply(lambda g: pd.Series({
        "r": stats.pearsonr(g.meanRR, g.RMSSD)[0],
        "slope_loglog": stats.linregress(np.log(g.meanRR), np.log(g.RMSSD)).slope}),
        include_groups=False)
    q1, q3 = r.r.quantile([.25, .75])
    out["r_rmssd_meanrr"] = dict(median=float(r.r.median()), q1=float(q1), q3=float(q3),
                                 slope_loglog_median=float(r.slope_loglog.median()))
    say(f"  within-participant r(RMSSD, meanRR), DMT windows: median {r.r.median():.2f} "
        f"(IQR {q1:.2f}-{q3:.2f}); log-log slope median {r.slope_loglog.median():.2f}")

    late = d[(d.State == "DMT") & (d.t_start >= 600)]
    w = late.groupby(["subject", "Dose"]).RMSSD_z.mean().unstack("Dose")
    diff = (w.High - w.Low).dropna()
    p = float(stats.ttest_1samp(diff, 0.0).pvalue)
    out["late_rebound"] = dict(mean_z=float(diff.mean()), n_positive=int((diff > 0).sum()),
                               n=int(len(diff)), p=p)
    say(f"  late RMSSD (DMT 10-19 min), High - Low: mean {diff.mean():+.2f} z, "
        f"{int((diff > 0).sum())}/{len(diff)} positive, t p = {p:.3f} (two-tailed)")
    return out


# =============================================================================
# figure
# =============================================================================
def make_figure(out_path: str) -> str:
    """Supplementary Fig. 4: RMSSD, RMSSD/meanRR, SDNN, HR (bpm); means +- SEM
    with one trace per participant (High in colour, Low grey dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figure_config as FC
    from cluster_stats import cluster_test, segments

    df = pd.read_csv(os.path.join(OUTDIR, "hrv_windows_long.csv"))
    subjects = json.load(open(os.path.join(OUTDIR, "hrv_summary.json")))["subjects"]
    df = df[df.subject.isin(subjects)]
    dfz = zscore_within_subject(df)
    d9 = df[df.t_end <= ANALYSIS_END]
    GREY, GREY_MEAN, DASH = "#9e9e9e", "#6b6b6b", (0, (2.2, 1.3))
    panels = [("RMSSD", "RMSSD (ms)"), ("RMSSD_n", "RMSSD / mean RR"),
              ("SDNN", "SDNN (ms)"), ("HR_bpm", "Heart rate (bpm)")]

    FC.apply_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=FC.FIG_SIZE_DOUBLE_TALL, sharex=True)
    for ax, (var, ylab) in zip(axes.ravel(), panels):
        H, x = matrix(d9, subjects, var, "DMT", "High", ANALYSIS_END)
        L, _ = matrix(d9, subjects, var, "DMT", "Low", ANALYSIS_END)
        rs = d9[d9.State == "RS"]
        for dose, col in [("High", FC.COLOR_ECG_HIGH), ("Low", GREY_MEAN)]:
            ax.axhline(rs[rs.Dose == dose].groupby("subject")[var].mean().mean(),
                       color=col, lw=0.8, ls=":", alpha=0.8, zorder=1)
        for row in H:
            ax.plot(x, row, color=FC.COLOR_ECG_HIGH, lw=0.7, alpha=0.40, zorder=1)
        for row in L:
            ax.plot(x, row, color=GREY, lw=0.9, alpha=0.80, ls=DASH, zorder=1)
        for M, lab, col in [(H, "High (40 mg)", FC.COLOR_ECG_HIGH), (L, "Low (20 mg)", GREY_MEAN)]:
            mu = np.nanmean(M, axis=0)
            se = np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])
            ax.plot(x, mu, color=col, lw=2.0, label=lab, zorder=5)
            ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.30, lw=0, zorder=4)
        Hz, _ = matrix(dfz[dfz.t_end <= ANALYSIS_END], subjects, var + "_z", "DMT", "High", ANALYSIS_END)
        Lz, _ = matrix(dfz[dfz.t_end <= ANALYSIS_END], subjects, var + "_z", "DMT", "Low", ANALYSIS_END)
        res = cluster_test(Hz, Lz, TAIL[var])
        for a, b in segments(res["sig"], x):
            ax.axvspan(a - 0.25, b + 0.25, color=FC.COLOR_SIG_SHADE, alpha=FC.ALPHA_SIG_SHADE, lw=0, zorder=0)
        ax.set_ylabel(ylab)
        ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
        ax.grid(True, alpha=FC.GRID_ALPHA, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Time from $t_0$ (min, window centre)")
    for ax, lab in zip(axes.ravel(), ["A", "B", "C", "D"]):
        FC.add_panel_label(ax, lab)
    leg = axes[0, 0].legend(loc="lower right", frameon=True, fontsize=8)
    FC.style_legend(leg)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# =============================================================================
def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    lines: List[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    df, qc, excluded = extract_windows()
    subjects = complete_subjects(df)
    df.to_csv(os.path.join(OUTDIR, "hrv_windows_long.csv"), index=False)
    with open(os.path.join(OUTDIR, "hrv_qc.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(qc) + "\n\n")
        fh.write(f"recordings excluded (> {100*MAX_REC_INVALID:.0f} % invalid intervals): {excluded}\n")
        fh.write(f"participants with all four recordings ({len(subjects)}): {subjects}\n")
        fh.write(f"participants dropped: {sorted(set(SUJETOS_VALIDADOS_ECG) - set(subjects))}\n")
    say(f"HRV extraction: {len(df)} windows; excluded recordings {excluded}; "
        f"participants retained {len(subjects)}: {subjects}")

    dfz = zscore_within_subject(df[df.subject.isin(subjects)])
    models = run_models(dfz, subjects, say)
    ceiling = intrinsic_ceiling(say)
    ceiling.to_csv(os.path.join(OUTDIR, "intrinsic_ceiling.csv"), index=False)
    descriptives = dict(hr=hr_descriptives(ceiling, say), hrv=hrv_descriptives(dfz, subjects, say))
    audit = recording_audit(say)
    audit.to_csv(os.path.join(OUTDIR, "recording_audit.csv"), index=False)

    with open(os.path.join(OUTDIR, "hrv_lme_cluster_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    summary = dict(subjects=subjects, excluded_recordings=excluded,
                   lme=models["lme"], clusters=models["clusters"],
                   ceiling={dose: dict(
                       n=int((ceiling.Dose == dose).sum()),
                       above_IHR=int((ceiling[ceiling.Dose == dose].margin_60s > 0).sum()),
                       above_IHR_upper95=int(ceiling[ceiling.Dose == dose].above_IHR_upper95.sum()),
                       peak60_mean=float(ceiling[ceiling.Dose == dose].peak_60s.mean()),
                       rest_mean=float(ceiling[ceiling.Dose == dose].rest_HR.mean()),
                       **ceiling_test(ceiling[ceiling.Dose == dose].margin_60s))
                       for dose in ["High", "Low"]},
                   descriptives=descriptives)
    with open(os.path.join(OUTDIR, "hrv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n-> {OUTDIR}")


if __name__ == "__main__":
    main()
    make_figure(os.path.join(REPO, "results", "figures", "figure_S4.png"))
