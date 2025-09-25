# -*- coding: utf-8 -*-
"""
Within-subject 2x2 ANOVA (RS vs DMT) x (Low vs High) for SMNA AUC.

Provides a helper to run a repeated-measures ANOVA and simple-effects tests
given per-subject AUCs for the four cells: RS_Low, RS_High, DMT_Low, DMT_High.

Usage from another script:

    from test.stats_anova_2x2 import run_anova_2x2_within
    results = run_anova_2x2_within(records, out_report_path)

Where `records` is a list of dicts like:
    {
      'subject': 'S04',
      'RS_Low':  ..., 'RS_High':  ...,
      'DMT_Low': ..., 'DMT_High': ...
    }

The function will write a text report if out_report_path is provided.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def _long_format(records: List[Dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        sid = rec['subject']
        rows.append({'subject': sid, 'Task': 'RS',  'Dose': 'Low',  'AUC': rec['RS_Low']})
        rows.append({'subject': sid, 'Task': 'RS',  'Dose': 'High', 'AUC': rec['RS_High']})
        rows.append({'subject': sid, 'Task': 'DMT', 'Dose': 'Low',  'AUC': rec['DMT_Low']})
        rows.append({'subject': sid, 'Task': 'DMT', 'Dose': 'High', 'AUC': rec['DMT_High']})
    df = pd.DataFrame(rows)
    # Ensure categorical ordering
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    return df


def _paired_t(a: np.ndarray, b: np.ndarray):
    # Lazy paired t without external deps; fallback to scipy if present
    try:
        from scipy import stats
        t, p = stats.ttest_rel(a, b, nan_policy='omit')
        return float(t), float(p)
    except Exception:
        x = a - b
        x = x[~np.isnan(x)]
        n = len(x)
        if n <= 1:
            return np.nan, np.nan
        mean = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        if sd == 0:
            return np.inf, 0.0
        t = mean / (sd / np.sqrt(n))
        # No exact p without scipy; return NaN
        return float(t), np.nan


def run_anova_2x2_within(records: List[Dict], out_report_path: Optional[str] = None) -> Dict:
    df = _long_format(records)
    subjects = sorted(df['subject'].unique())
    report_lines = []
    report_lines.append(f"N subjects: {len(subjects)}")

    # Try pingouin, then statsmodels; else skip ANOVA
    anova_table = None
    anova_ok = False
    try:
        import pingouin as pg
        aov = pg.rm_anova(dv='AUC', within=['Task', 'Dose'], subject='subject', data=df, detailed=True)
        anova_table = aov
        anova_ok = True
        report_lines.append("Repeated-measures ANOVA via pingouin:")
        report_lines.append(str(aov))
    except Exception:
        try:
            import statsmodels.api as sm
            from statsmodels.stats.anova import AnovaRM
            aovrm = AnovaRM(df, depvar='AUC', subject='subject', within=['Task', 'Dose'])
            res = aovrm.fit()
            anova_table = res.anova_table
            anova_ok = True
            report_lines.append("Repeated-measures ANOVA via statsmodels:")
            report_lines.append(str(anova_table))
        except Exception as e:
            report_lines.append(f"ANOVA not available: {e}")

    # Planned comparisons / simple effects
    # RS: Low vs High
    rs = df[df['Task'] == 'RS']
    rs_wide = rs.pivot(index='subject', columns='Dose', values='AUC')
    t_rs, p_rs = _paired_t(rs_wide['Low'].to_numpy(), rs_wide['High'].to_numpy())
    report_lines.append(f"RS Low vs High (paired t): t={t_rs:.3f}, p={p_rs if p_rs==p_rs else 'NA'}")

    # DMT: Low vs High
    dmt = df[df['Task'] == 'DMT']
    dmt_wide = dmt.pivot(index='subject', columns='Dose', values='AUC')
    t_dmt, p_dmt = _paired_t(dmt_wide['Low'].to_numpy(), dmt_wide['High'].to_numpy())
    report_lines.append(f"DMT Low vs High (paired t): t={t_dmt:.3f}, p={p_dmt if p_dmt==p_dmt else 'NA'}")

    # Main effect Task: average across Dose within subject
    rs_avg = rs_wide.mean(axis=1).to_numpy()
    dmt_avg = dmt_wide.mean(axis=1).to_numpy()
    t_task, p_task = _paired_t(rs_avg, dmt_avg)
    report_lines.append(f"Main effect Task (RS vs DMT): t={t_task:.3f}, p={p_task if p_task==p_task else 'NA'}")

    # Summarize means
    means = df.groupby(['Task', 'Dose'])['AUC'].mean().unstack()
    report_lines.append("\nCell means (AUC):")
    report_lines.append(str(means))

    # Write report
    if out_report_path:
        out_dir = os.path.dirname(out_report_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    return {
        'anova_ok': anova_ok,
        'anova': anova_table,
        't_rs': t_rs, 'p_rs': p_rs,
        't_dmt': t_dmt, 'p_dmt': p_dmt,
        't_task': t_task, 'p_task': p_task,
        'cell_means': means,
        'n_subjects': len(subjects)
    }


def _render_anova_results(results: Dict) -> str:
    lines = []
    lines.append(f"N subjects: {results.get('n_subjects', 'NA')}")
    aov = results.get('anova', None)
    if aov is not None:
        lines.append(str(aov))
    lines.append(f"RS Low vs High (paired t): t={results.get('t_rs', float('nan')):.3f}, p={results.get('p_rs', 'NA')}")
    lines.append(f"DMT Low vs High (paired t): t={results.get('t_dmt', float('nan')):.3f}, p={results.get('p_dmt', 'NA')}")
    lines.append(f"Main effect Task (RS vs DMT): t={results.get('t_task', float('nan')):.3f}, p={results.get('p_task', 'NA')}")
    cm = results.get('cell_means', None)
    if cm is not None:
        lines.append("Cell means (AUC):")
        lines.append(str(cm))
    return "\n".join(lines)


def _paired_cohens_d(diff: np.ndarray) -> float:
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n <= 1:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return np.inf
    return float(np.mean(diff) / sd)


def _bootstrap_ci_diff(diff: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> tuple:
    rng = np.random.default_rng(12345)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    stats = []
    for _ in range(n_boot):
        sample = diff[rng.integers(0, n, size=n)]
        d = _paired_cohens_d(sample)
        stats.append(d)
    lo = np.percentile(stats, (1.0 - ci) / 2.0 * 100.0)
    hi = np.percentile(stats, (1.0 + ci) / 2.0 * 100.0)
    return float(lo), float(hi)


def run_anova_2x2_per_minute(records_by_minute: Dict[int, List[Dict]], out_report_path: Optional[str] = None) -> str:
    """Run 2x2 within-subject ANOVA for each minute window and write a combined report.
    Applies Holm correction across the 10 interaction p-values and reports paired d_z with CIs.
    """
    all_lines = []
    all_lines.append("Repeated-measures ANOVA 2x2 per minute window (first 10 minutes)")
    all_lines.append("Design: Within-subjects (Task: RS vs DMT) × (Dose: Low vs High)")
    all_lines.append("=" * 72)

    minutes = sorted(records_by_minute.keys())
    p_inter_list = []
    per_minute_blocks = []

    for m in minutes:
        recs = records_by_minute[m]
        header = f"\n--- Minute window {m:02d}:00–{m:02d}:59 ---"
        if not recs:
            per_minute_blocks.append(header + "\nNo data")
            p_inter_list.append(np.nan)
            continue

        res = run_anova_2x2_within(recs, out_report_path=None)

        # Try to extract interaction p-value
        p_inter = np.nan
        aov = res.get('anova', None)
        try:
            if aov is not None and hasattr(aov, 'index'):
                for key in ['Task * Dose', 'Task:Dose', 'Task * Dose (sphericity corrected)']:
                    if key in aov.index:
                        row = aov.loc[key]
                        p_inter = float(row.get('p-unc', np.nan) if 'p-unc' in row else row.get('Pr > F', np.nan))
                        break
        except Exception:
            pass
        p_inter_list.append(p_inter if (p_inter==p_inter) else 1.0)

        # Paired effect sizes with CIs for RS and DMT High–Low
        df_long = _long_format(recs)
        rs = df_long[df_long['Task'] == 'RS']
        dmt = df_long[df_long['Task'] == 'DMT']
        rs_w = rs.pivot(index='subject', columns='Dose', values='AUC')
        dmt_w = dmt.pivot(index='subject', columns='Dose', values='AUC')
        d_rs = rs_w['High'].to_numpy() - rs_w['Low'].to_numpy()
        d_dmt = dmt_w['High'].to_numpy() - dmt_w['Low'].to_numpy()
        d_rs_val = _paired_cohens_d(d_rs)
        d_dmt_val = _paired_cohens_d(d_dmt)
        ci_rs = _bootstrap_ci_diff(d_rs)
        ci_dmt = _bootstrap_ci_diff(d_dmt)

        block = [header, _render_anova_results(res), "Effect sizes per-minute:",
                 f"  RS High–Low: d_z={d_rs_val:.3f}, 95% CI [{ci_rs[0]:.3f}, {ci_rs[1]:.3f}]",
                 f"  DMT High–Low: d_z={d_dmt_val:.3f}, 95% CI [{ci_dmt[0]:.3f}, {ci_dmt[1]:.3f}]" ]
        per_minute_blocks.append("\n".join(block))

    # Holm correction across 10 minutes for interaction p-values
    def _holm(pvals):
        m = len(pvals)
        order = np.argsort(pvals)
        p_sorted = np.array(pvals)[order]
        adj = np.minimum.accumulate(np.maximum.accumulate((m - np.arange(m)) * p_sorted, axis=0))[::-1][::-1]
        out = np.empty(m)
        out[order] = np.minimum(adj, 1.0)
        return out

    p_clean = [p if (p==p) else 1.0 for p in p_inter_list]
    p_holm = _holm(p_clean)

    # Append blocks with Holm-corrected p
    for k, m in enumerate(minutes):
        all_lines.append(per_minute_blocks[k])
        all_lines.append(f"  Holm-corrected p (interaction, across minutes): {p_holm[k]:.4f}")

    text = "\n".join(all_lines)
    if out_report_path:
        out_dir = os.path.dirname(out_report_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_report_path, 'w', encoding='utf-8') as f:
            f.write(text)
    return text


if __name__ == '__main__':
    # Minimal smoke test with fake data
    recs = []
    rng = np.random.default_rng(0)
    for i in range(10):
        recs.append({
            'subject': f'S{i:02d}',
            'RS_Low':  rng.normal(10, 2),
            'RS_High': rng.normal(10, 2),
            'DMT_Low': rng.normal(12, 2),
            'DMT_High': rng.normal(16, 2),
        })
    run_anova_2x2_within(recs, out_report_path=os.path.join('test', 'eda', 'smna_10min', 'anova_2x2_report.txt'))


