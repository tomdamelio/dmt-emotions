#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary Analysis: Temporal Resolution Sensitivity for TET Data

Addresses Reviewer Comment 10 (Section 2.7):
  The reviewer questions whether participants can reliably represent experience at
  4-second temporal granularity during retrospective TET ratings. They suggest
  downsampling to 20-30s timepoints to test if the same results are found at
  coarser resolution (20-30s), more consistent with retrospective memory.

Analysis (replicated at 4s native, 20s, and 30s resolutions):
  1. Temporal t-tests:
       - Dose effect within DMT (High > Low): one-tailed paired t-test + FDR (BH)
       - State effect (DMT vs RS): two-tailed paired t-test + FDR (BH)
  2. LME models (same formula as main paper):
       Y ~ state * dose + state * time_c + dose * time_c + (1|subject)
       Key term: State × Dose interaction (state[T.DMT]:dose[T.Alta])
       Models: Arousal, Valence, and 5 individual dimensions

Key paper results to replicate:
  - Arousal State × Dose: β = 0.47, 95% CI [0.32, 0.61], p < .001
  - Valence State × Dose: β = −0.50, 95% CI [−0.70, −0.30], p < .001

Outputs:
  results/tet/supplementary/temporal_resolution/
    sensitivity_ttest_results.csv     -- t-test significance counts
    sensitivity_ttest_timecourses.csv -- p-values at each timepoint
    sensitivity_lme_results.csv       -- LME State × Dose estimates per resolution
    figure_S_ttest.png                -- t-test comparison figure
    figure_S_lme.png                  -- LME forest plot comparison

Usage:
    python src/run_temporal_resolution_sensitivity.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'results' / 'tet' / 'preprocessed' / 'tet_preprocessed.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'tet' / 'supplementary' / 'temporal_resolution'

DMT_MAX_TIME_SEC = 1200   # 20 minutes
RS_MAX_TIME_SEC = 600     # 10 minutes

RESOLUTIONS = {
    '4s (native)': 4,
    '20s': 20,
    '30s': 30,
}

# Affective variables (z-scored columns + derived valence index)
AFFECTIVE_VARS = {
    'emotional_intensity_z': 'Arousal\n(Emotional Intensity)',
    'valence_index_z':        'Valence\n(Pleasant − Unpleasant)',
    'interoception_z':        'Interoception',
    'anxiety_z':              'Anxiety',
    'unpleasantness_z':       'Unpleasantness',
    'pleasantness_z':         'Pleasantness',
    'bliss_z':                'Bliss',
}

# For LME: individual z-scored dimensions (excluding derived valence_index_z)
LME_DIMS = ['emotional_intensity_z', 'interoception_z', 'anxiety_z',
            'unpleasantness_z', 'pleasantness_z', 'bliss_z']

# Colors
C_4S  = '#2c7bb6'
C_20S = '#d7191c'
C_30S = '#1a9641'
RES_COLORS = [C_4S, C_20S, C_30S]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[
        ((df['state'] == 'DMT') & (df['t_sec'] <= DMT_MAX_TIME_SEC)) |
        ((df['state'] == 'RS')  & (df['t_sec'] <= RS_MAX_TIME_SEC))
    ].copy()
    if 'valence_index_z' not in df.columns:
        df['valence_index_z'] = df['pleasantness_z'] - df['unpleasantness_z']
    print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
    return df


def downsample(df: pd.DataFrame, bin_sec: int) -> pd.DataFrame:
    """Average all affective columns into non-overlapping bins of bin_sec seconds."""
    df = df.copy()
    df['t_bin_sec'] = (df['t_sec'] // bin_sec) * bin_sec
    agg_cols = list(AFFECTIVE_VARS.keys())
    ds = (
        df.groupby(['subject', 'state', 'dose', 't_bin_sec'], observed=True)[agg_cols]
        .mean()
        .reset_index()
    )
    ds['time_min'] = ds['t_bin_sec'] / 60
    return ds


# ---------------------------------------------------------------------------
# Part 1 — Temporal t-tests
# ---------------------------------------------------------------------------

def _paired_ttest_series(df_a, df_b, var, time_col='time_min',
                          alternative='two-sided') -> pd.DataFrame:
    time_bins = sorted(set(df_a[time_col].unique()) & set(df_b[time_col].unique()))
    rows = []
    for t in time_bins:
        a_s = df_a[df_a[time_col] == t].set_index('subject')[var]
        b_s = df_b[df_b[time_col] == t].set_index('subject')[var]
        common = sorted(set(a_s.index) & set(b_s.index))
        if len(common) >= 3:
            t_stat, p = stats.ttest_rel(a_s[common].values, b_s[common].values,
                                        alternative=alternative)
        else:
            t_stat, p = np.nan, np.nan
        rows.append({'time_min': t, 't_stat': t_stat, 'p_raw': p, 'n_pairs': len(common)})
    return pd.DataFrame(rows)


def _apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    valid = df['p_raw'].notna()
    p_fdr = np.full(len(df), np.nan)
    if valid.sum() > 0:
        _, corr, _, _ = multipletests(df.loc[valid, 'p_raw'], method='fdr_bh')
        p_fdr[valid.values] = corr
    df['p_fdr'] = p_fdr
    df['sig'] = df['p_fdr'] < 0.05
    return df


def run_ttests_at_resolution(df: pd.DataFrame, bin_sec: int) -> dict:
    ds = downsample(df, bin_sec)
    dmt = ds[ds['state'] == 'DMT']
    rs  = ds[ds['state'] == 'RS']
    high = dmt[dmt['dose'] == 'Alta']
    low  = dmt[dmt['dose'] == 'Baja']
    # Collapse doses for state comparison
    dmt_avg = dmt.groupby(['subject', 'time_min'], observed=True).mean(numeric_only=True).reset_index()
    rs_avg  = rs.groupby(['subject', 'time_min'], observed=True).mean(numeric_only=True).reset_index()

    results = {}
    for var in AFFECTIVE_VARS:
        dose_df  = _apply_fdr(_paired_ttest_series(high, low, var, alternative='greater'))
        state_df = _apply_fdr(_paired_ttest_series(dmt_avg, rs_avg, var, alternative='two-sided'))
        results[var] = {'dose': dose_df, 'state': state_df}
    return results


# ---------------------------------------------------------------------------
# Part 2 — LME models
# ---------------------------------------------------------------------------

def _fit_single_lme(df_all: pd.DataFrame, outcome: str):
    """Fit one LME model matching the main paper formula."""
    model = smf.mixedlm(
        f"{outcome} ~ state * dose + state * time_c + dose * time_c",
        df_all,
        groups=df_all['subject'],
        re_formula='1'
    )
    return model.fit(reml=True, method='lbfgs')


def run_lme_at_resolution(df: pd.DataFrame, bin_sec: int) -> list:
    """
    Fit LME models on data downsampled to bin_sec resolution.
    Returns a list of dicts with key parameters for State × Dose interaction.
    """
    ds = downsample(df, bin_sec)

    # Prepare: reference levels, centered time, valence index
    ds = ds.copy()
    ds['valence_index'] = ds['pleasantness_z'] - ds['unpleasantness_z']
    ds['state'] = pd.Categorical(ds['state'], categories=['RS', 'DMT'], ordered=False)
    ds['dose']  = pd.Categorical(ds['dose'],  categories=['Baja', 'Alta'], ordered=False)
    ds['time_c'] = ds['time_min'] - ds['time_min'].mean()

    interaction_key = 'state[T.DMT]:dose[T.Alta]'

    outcomes = {
        'arousal':     'emotional_intensity_z',
        'valence':     'valence_index',
        'interoception': 'interoception_z',
        'anxiety':     'anxiety_z',
        'unpleasantness': 'unpleasantness_z',
        'pleasantness': 'pleasantness_z',
        'bliss':       'bliss_z',
    }

    records = []
    for name, outcome in outcomes.items():
        print(f"    Fitting {name}...")
        try:
            fit = _fit_single_lme(ds, outcome)
            if interaction_key in fit.params:
                beta = fit.params[interaction_key]
                ci   = fit.conf_int().loc[interaction_key]
                p    = fit.pvalues[interaction_key]
                records.append({
                    'model':  name,
                    'outcome': outcome,
                    'beta':   beta,
                    'ci_lo':  ci[0],
                    'ci_hi':  ci[1],
                    'p':      p,
                    'sig':    p < 0.05,
                })
            else:
                records.append({'model': name, 'outcome': outcome,
                                'beta': np.nan, 'ci_lo': np.nan,
                                'ci_hi': np.nan, 'p': np.nan, 'sig': False})
        except Exception as e:
            print(f"      WARNING: {e}")
            records.append({'model': name, 'outcome': outcome,
                            'beta': np.nan, 'ci_lo': np.nan,
                            'ci_hi': np.nan, 'p': np.nan, 'sig': False})
    return records


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_ttest_summary(all_ttest: dict) -> pd.DataFrame:
    rows = []
    for res_label, (bin_sec, var_results) in all_ttest.items():
        for var, var_label in AFFECTIVE_VARS.items():
            for effect in ['dose', 'state']:
                edf = var_results[var][effect]
                rows.append({
                    'resolution': res_label, 'bin_sec': bin_sec,
                    'variable': var,
                    'variable_label': var_label.replace('\n', ' '),
                    'effect': effect,
                    'n_timepoints': len(edf),
                    'n_sig': int(edf['sig'].sum()),
                    'pct_sig': 100 * edf['sig'].mean(),
                })
    return pd.DataFrame(rows)


def build_lme_summary(all_lme: dict) -> pd.DataFrame:
    rows = []
    for res_label, (bin_sec, records) in all_lme.items():
        for r in records:
            rows.append({'resolution': res_label, 'bin_sec': bin_sec, **r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_ttest_figure(summary: pd.DataFrame, output_path: Path):
    variables = list(AFFECTIVE_VARS.keys())
    n_vars = len(variables)
    res_labels = list(RESOLUTIONS.keys())

    fig, axes = plt.subplots(2, n_vars, figsize=(3.2 * n_vars, 7),
                             sharey='row')
    fig.suptitle(
        'Temporal Resolution Sensitivity — Pointwise t-tests\n'
        '(Supplementary, addresses Reviewer Comment 10)',
        fontsize=11, fontweight='bold', y=0.99
    )
    effect_info = [
        ('dose',  'Dose effect (High > Low, DMT)\none-tailed, FDR-corrected'),
        ('state', 'State effect (DMT vs RS)\ntwo-tailed, FDR-corrected'),
    ]

    for row_idx, (effect_key, effect_label) in enumerate(effect_info):
        for col_idx, var in enumerate(variables):
            ax = axes[row_idx, col_idx]
            var_label = AFFECTIVE_VARS[var].replace('\n', ' ')

            x = np.arange(len(res_labels))
            sub = summary[(summary['variable'] == var) & (summary['effect'] == effect_key)]
            for i, (res_label, color) in enumerate(zip(res_labels, RES_COLORS)):
                row = sub[sub['resolution'] == res_label]
                if len(row) == 0:
                    continue
                n_sig = int(row['n_sig'].iloc[0])
                n_tot = int(row['n_timepoints'].iloc[0])
                pct   = float(row['pct_sig'].iloc[0])
                ax.bar(i, pct, color=color, alpha=0.8, width=0.6)
                ax.text(i, pct + 1, f'{n_sig}/{n_tot}', ha='center',
                        va='bottom', fontsize=7.5)

            ax.set_xticks(x)
            ax.set_xticklabels(res_labels, rotation=30, ha='right', fontsize=8)
            ax.set_ylim(0, 115)
            if col_idx == 0:
                ax.set_ylabel('% significant timepoints\n(FDR p < .05)', fontsize=8.5)
            ax.set_title(var_label, fontsize=8.5, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        axes[row_idx, 0].annotate(
            effect_label, xy=(-0.4, 0.5), xycoords='axes fraction',
            fontsize=8.5, va='center', ha='right', rotation=90,
            fontweight='bold', annotation_clip=False
        )

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.8, label=l)
               for c, l in zip(RES_COLORS, res_labels)]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_lme_figure(lme_summary: pd.DataFrame, output_path: Path):
    """
    Forest plot comparing LME State × Dose interaction betas across resolutions,
    for each model. Layout: one column per resolution, one row per model.
    """
    models_order = ['arousal', 'valence', 'interoception', 'anxiety',
                    'unpleasantness', 'pleasantness', 'bliss']
    model_labels = {
        'arousal':       'Arousal\n(Emotional Intensity)',
        'valence':       'Valence\n(Pleasant − Unpleasant)',
        'interoception': 'Interoception',
        'anxiety':       'Anxiety',
        'unpleasantness':'Unpleasantness',
        'pleasantness':  'Pleasantness',
        'bliss':         'Bliss',
    }
    res_labels = list(RESOLUTIONS.keys())
    n_res = len(res_labels)
    n_models = len(models_order)

    fig, axes = plt.subplots(1, n_res, figsize=(4 * n_res, 6), sharey=True)
    fig.suptitle(
        'LME State × Dose Interaction (β) across Temporal Resolutions\n'
        'Term: state[T.DMT]:dose[T.Alta]  |  Reference: RS, Low dose\n'
        '(Supplementary, addresses Reviewer Comment 10)',
        fontsize=10, fontweight='bold', y=1.01
    )

    # Paper reference values
    paper_ref = {'arousal': 0.47, 'valence': -0.50}

    for col_idx, (res_label, color) in enumerate(zip(res_labels, RES_COLORS)):
        ax = axes[col_idx]
        sub = lme_summary[lme_summary['resolution'] == res_label]

        y_positions = np.arange(n_models)
        for yi, model_name in enumerate(models_order):
            row = sub[sub['model'] == model_name]
            if len(row) == 0 or row['beta'].isna().all():
                ax.plot(0, yi, 'x', color='gray', ms=8)
                continue
            beta  = float(row['beta'].iloc[0])
            ci_lo = float(row['ci_lo'].iloc[0])
            ci_hi = float(row['ci_hi'].iloc[0])
            p     = float(row['p'].iloc[0])
            sig   = bool(row['sig'].iloc[0])

            # Error bar (CI)
            ax.errorbar(beta, yi, xerr=[[beta - ci_lo], [ci_hi - beta]],
                        fmt='o', color=color if sig else 'gray',
                        ecolor=color if sig else 'lightgray',
                        capsize=4, ms=6, linewidth=1.5,
                        zorder=3 if sig else 2)

            # p-value annotation
            p_str = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else 'ns')
            ax.text(ci_hi + 0.02, yi, p_str, va='center', fontsize=8,
                    color=color if sig else 'gray')

            # Mark paper reference (4s only)
            if col_idx == 0 and model_name in paper_ref:
                ax.axvline(paper_ref[model_name], color='black',
                           linestyle=':', linewidth=1, alpha=0.5)

        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_yticks(y_positions)
        if col_idx == 0:
            ax.set_yticklabels([model_labels[m] for m in models_order], fontsize=8.5)
        ax.set_xlabel('β (State × Dose interaction)', fontsize=9)
        ax.set_title(res_label, fontsize=10, fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_yaxis()

    # Shared note
    fig.text(0.5, -0.02,
             'Dotted vertical lines = paper reference values (4s analysis).\n'
             'Filled markers = p < .05; grey = not significant.',
             ha='center', fontsize=8, style='italic')

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY: Temporal Resolution Sensitivity (TET Data)")
    print("Addresses Reviewer Comment 10, Section 2.7")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare()

    # ------------------------------------------------------------------
    # Part 1: Temporal t-tests
    # ------------------------------------------------------------------
    print("\n--- PART 1: Temporal t-tests ---")
    all_ttest = {}
    for res_label, bin_sec in RESOLUTIONS.items():
        print(f"\n  Resolution: {res_label}")
        var_results = run_ttests_at_resolution(df, bin_sec)
        all_ttest[res_label] = (bin_sec, var_results)
        for var in ['emotional_intensity_z', 'valence_index_z']:
            dose_df  = var_results[var]['dose']
            state_df = var_results[var]['state']
            vl = AFFECTIVE_VARS[var].replace('\n', ' ')[:30]
            print(f"    {vl:<32} dose: {dose_df['sig'].sum()}/{len(dose_df)}"
                  f"  state: {state_df['sig'].sum()}/{len(state_df)}")

    ttest_summary = build_ttest_summary(all_ttest)
    ttest_summary.to_csv(OUTPUT_DIR / 'sensitivity_ttest_results.csv', index=False)

    # Save timecourse p-values
    tc_rows = []
    for res_label, (bin_sec, var_results) in all_ttest.items():
        for var in AFFECTIVE_VARS:
            for effect in ['dose', 'state']:
                tmp = var_results[var][effect].copy()
                tmp['resolution'] = res_label
                tmp['bin_sec']    = bin_sec
                tmp['variable']   = var
                tmp['effect']     = effect
                tc_rows.append(tmp)
    pd.concat(tc_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / 'sensitivity_ttest_timecourses.csv', index=False)

    # ------------------------------------------------------------------
    # Part 2: LME models
    # ------------------------------------------------------------------
    print("\n--- PART 2: LME models (State × Dose interaction) ---")
    all_lme = {}
    for res_label, bin_sec in RESOLUTIONS.items():
        print(f"\n  Resolution: {res_label} (bin = {bin_sec}s)")
        records = run_lme_at_resolution(df, bin_sec)
        all_lme[res_label] = (bin_sec, records)

    lme_summary = build_lme_summary(all_lme)
    lme_summary.to_csv(OUTPUT_DIR / 'sensitivity_lme_results.csv', index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n--- Generating figures ---")
    plot_ttest_figure(ttest_summary, OUTPUT_DIR / 'figure_S_ttest.png')
    plot_lme_figure(lme_summary, OUTPUT_DIR / 'figure_S_lme.png')

    # ------------------------------------------------------------------
    # Print text summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n=== T-tests: % significant timepoints (FDR p < .05) ===")
    for effect_key, effect_label in [('dose', 'Dose (High > Low, DMT, one-tailed)'),
                                      ('state', 'State (DMT vs RS, two-tailed)')]:
        print(f"\n  {effect_label}")
        sub = ttest_summary[ttest_summary['effect'] == effect_key]
        print(f"  {'Variable':<35} {'4s':>14} {'20s':>14} {'30s':>14}")
        print("  " + "-" * 78)
        for var, var_label in AFFECTIVE_VARS.items():
            vals = []
            for res_label in RESOLUTIONS:
                r = sub[(sub['resolution'] == res_label) & (sub['variable'] == var)]
                if len(r):
                    n_sig = int(r['n_sig'].iloc[0])
                    n_tot = int(r['n_timepoints'].iloc[0])
                    pct   = float(r['pct_sig'].iloc[0])
                    vals.append(f"{n_sig}/{n_tot} ({pct:.0f}%)")
                else:
                    vals.append("N/A")
            vl = var_label.replace('\n', ' ')[:34]
            print(f"  {vl:<35} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")

    print("\n=== LME: State × Dose interaction (β [95% CI], p) ===")
    print(f"  {'Model':<18} {'4s (native)':^30} {'20s':^30} {'30s':^30}")
    print("  " + "-" * 108)
    for model_name in ['arousal', 'valence', 'interoception', 'anxiety',
                        'unpleasantness', 'pleasantness', 'bliss']:
        row_parts = []
        for res_label in RESOLUTIONS:
            r = lme_summary[(lme_summary['resolution'] == res_label) &
                            (lme_summary['model'] == model_name)]
            if len(r) and not np.isnan(r['beta'].iloc[0]):
                beta  = r['beta'].iloc[0]
                ci_lo = r['ci_lo'].iloc[0]
                ci_hi = r['ci_hi'].iloc[0]
                p     = r['p'].iloc[0]
                sig   = '*' if p < 0.05 else ''
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<.001"
                row_parts.append(f"{beta:+.2f} [{ci_lo:+.2f},{ci_hi:+.2f}] {p_str}{sig}")
            else:
                row_parts.append("N/A")
        print(f"  {model_name:<18} {row_parts[0]:^30} {row_parts[1]:^30} {row_parts[2]:^30}")

    print("\n" + "=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
