# -*- coding: utf-8 -*-
"""
Unified SCR (EmotiPhai) Analysis: Poisson GEE modeling and visualization (first 9 minutes).

This script adapts the EmotiPhai SCR rate analysis and adds statistical modeling:
  1) Load EmotiPhai SCR events and compute per-minute counts (0–8)
  2) Fit a Poisson GEE (log link) with Task × Dose and time effects
  3) Apply BH-FDR within hypothesis families and report conditional contrasts
  4) Produce publication-ready plots and a textual model report

Outputs are written to: results/eda/scr/ (plots in results/eda/scr/plots/)

Run:
  python scripts/run_eda_scr_analysis.py
"""

import os
import sys
from typing import List, Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_EDA,
    get_dosis_sujeto,
)

# Statistical packages
try:
    import patsy
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.cov_struct import Exchangeable
except Exception:
    GEE = None

try:
    from scipy import stats as scistats
except Exception:
    scistats = None


#############################
# Plot aesthetics & centralized hyperparameters
#############################

# Centralized font sizes and legend settings (aligned with SCL script)
AXES_TITLE_SIZE = 29
AXES_LABEL_SIZE = 36
TICK_LABEL_SIZE = 28
TICK_LABEL_SIZE_SMALL = 24

LEGEND_FONTSIZE = 14
LEGEND_FONTSIZE_SMALL = 12
LEGEND_MARKERSCALE = 1.6
LEGEND_BORDERPAD = 0.6
LEGEND_HANDLELENGTH = 3.0

# Stacked per-subject figure specific sizes
STACKED_AXES_LABEL_SIZE = 26
STACKED_TICK_LABEL_SIZE = 18
STACKED_SUBJECT_FONTSIZE = 36

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': AXES_TITLE_SIZE,
    'axes.labelsize': AXES_LABEL_SIZE,
    'axes.titlepad': 8.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'legend.fontsize': LEGEND_FONTSIZE,
    'legend.borderpad': LEGEND_BORDERPAD,
    'legend.handlelength': LEGEND_HANDLELENGTH,
    'xtick.labelsize': TICK_LABEL_SIZE_SMALL,
    'ytick.labelsize': TICK_LABEL_SIZE_SMALL,
})

# Fixed colors matching project convention
# RS High=green, RS Low=purple; DMT High=red, DMT Low=blue
COLOR_RS_HIGH = 'tab:green'
COLOR_RS_LOW = 'tab:purple'
COLOR_DMT_HIGH = 'tab:red'
COLOR_DMT_LOW = 'tab:blue'

# Analysis window: first 9 minutes
N_MINUTES = 9  # store minutes as 1..9 in long data
MAX_TIME_SEC = 60 * N_MINUTES


def _fmt_mmss(x, pos):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m:02d}:{s:02d}"


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_emotiphai_scr_path(subject: str, task: str, session: str, dose: str) -> str:
    """Build path to EmotiPhai SCR CSV file."""
    dose_dir = f'dmt_{dose}'
    filename = f"{subject}_{task}_{session}_{dose}_emotiphai_scr.csv"
    return os.path.join(DERIVATIVES_DATA, 'phys', 'eda', dose_dir, filename)


def load_emotiphai_scr(path: str, fs: float = 250.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load EmotiPhai SCR events (onsets, amplitudes). Returns (onsets_sec, amplitudes)."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df) == 0:
            return None
        if 'SCR_Onsets_Emotiphai' not in df.columns or 'SCR_Amplitudes_Emotiphai' not in df.columns:
            return None
        onsets = df['SCR_Onsets_Emotiphai'].values
        amplitudes = df['SCR_Amplitudes_Emotiphai'].values
        valid = ~(np.isnan(onsets) | np.isnan(amplitudes))
        onsets = onsets[valid]
        amplitudes = amplitudes[valid]
        if len(onsets) == 0:
            return None
        onsets_sec = onsets / float(fs)
        return onsets_sec, amplitudes
    except Exception:
        return None


def compute_scr_rate_per_minute(onsets_sec: np.ndarray, max_time: float = MAX_TIME_SEC) -> np.ndarray:
    """Count SCRs in 1-minute bins for the first N minutes (length N_MINUTES)."""
    n_minutes = int(np.ceil(max_time / 60.0))
    counts = np.zeros(n_minutes, dtype=float)
    for onset in onsets_sec:
        if onset < max_time:
            minute_idx = int(onset // 60.0)
            if minute_idx < n_minutes:
                counts[minute_idx] += 1
    return counts


def prepare_long_data_scr() -> pd.DataFrame:
    """Build long-format per-minute SCR counts for first 9 minutes."""
    rows: List[Dict] = []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            # DMT
            high_session, low_session = determine_sessions(subject)
            path_dmt_high = build_emotiphai_scr_path(subject, 'dmt', high_session, 'high')
            path_dmt_low = build_emotiphai_scr_path(subject, 'dmt', low_session, 'low')
            data_high = load_emotiphai_scr(path_dmt_high)
            data_low = load_emotiphai_scr(path_dmt_low)
            onsets_dmt_high = np.array([]) if data_high is None else data_high[0]
            onsets_dmt_low = np.array([]) if data_low is None else data_low[0]

            # RS
            path_rs_high = build_emotiphai_scr_path(subject, 'rs', high_session, 'high')
            path_rs_low = build_emotiphai_scr_path(subject, 'rs', low_session, 'low')
            data_rs_high = load_emotiphai_scr(path_rs_high)
            data_rs_low = load_emotiphai_scr(path_rs_low)
            onsets_rs_high = np.array([]) if data_rs_high is None else data_rs_high[0]
            onsets_rs_low = np.array([]) if data_rs_low is None else data_rs_low[0]

            # Per-minute counts for first 9 minutes
            dmt_h_counts = compute_scr_rate_per_minute(onsets_dmt_high)
            dmt_l_counts = compute_scr_rate_per_minute(onsets_dmt_low)
            rs_h_counts = compute_scr_rate_per_minute(onsets_rs_high)
            rs_l_counts = compute_scr_rate_per_minute(onsets_rs_low)

            # Build rows for minutes 1..9 (labels shifted by +1 for interpretability)
            for minute in range(N_MINUTES):
                minute_label = minute + 1
                rows.extend([
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'High', 'count': float(dmt_h_counts[minute])},
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'Low', 'count': float(dmt_l_counts[minute])},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'High', 'count': float(rs_h_counts[minute])},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'Low', 'count': float(rs_l_counts[minute])},
                ])
        except Exception:
            continue

    if not rows:
        raise ValueError('No SCR data found!')

    df = pd.DataFrame(rows)
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['minute_c'] = df['minute'] - df['minute'].mean()
    return df


def fit_gee_poisson_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    """Fit a Poisson GEE with exchangeable working correlation by subject."""
    if GEE is None:
        return None, {'error': 'statsmodels GEE not available'}
    # Use patsy to build design matrices with categorical encoding
    formula = 'count ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c'
    try:
        y, X = patsy.dmatrices(formula, df, return_type='dataframe')
        groups = df['subject']
        model = GEE(endog=y, exog=X, groups=groups, family=Poisson(), cov_struct=Exchangeable())
        result = model.fit()
    except Exception as e:
        return None, {'error': str(e)}

    diagnostics = {
        'n_obs': int(result.nobs),
        'n_groups': int(df['subject'].nunique()),
        'working_corr': 'Exchangeable',
        'converged': getattr(result, 'converged', True),
    }
    return result, diagnostics


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[order[i]] = sorted_p[i]
        else:
            adjusted[order[i]] = min(sorted_p[i] * n / (i + 1), adjusted[order[i + 1]])
    return np.minimum(adjusted, 1.0).tolist()


def hypothesis_testing_with_fdr(result, df: pd.DataFrame) -> Dict:
    if result is None:
        return {}
    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    conf_int = result.conf_int()
    # Determine CI column labels robustly
    ci_cols = list(conf_int.columns)
    lower_col = ci_cols[0]
    upper_col = ci_cols[1]

    out: Dict[str, Dict] = {
        'all_params': params.to_dict(),
        'all_pvalues': pvalues.to_dict(),
        'all_stderr': bse.to_dict(),
        'conf_int': {param: {'lower': float(conf_int.loc[param, lower_col]),
                             'upper': float(conf_int.loc[param, upper_col])}
                    for param in params.index},
    }

    families: Dict[str, List[str]] = {'Task': [], 'Dose': [], 'Interaction': []}
    for p in ['Task[T.DMT]', 'Task[T.DMT]:minute_c']:
        if p in pvalues.index:
            families['Task'].append(p)
    for p in ['Dose[T.High]', 'Dose[T.High]:minute_c']:
        if p in pvalues.index:
            families['Dose'].append(p)
    for p in ['Task[T.DMT]:Dose[T.High]']:
        if p in pvalues.index:
            families['Interaction'].append(p)

    fdr_results: Dict[str, Dict] = {}
    for fam, plist in families.items():
        if not plist:
            continue
        raw = [pvalues[p] for p in plist]
        adj = benjamini_hochberg_correction(raw)
        fam_dict: Dict[str, Dict] = {}
        for i, p in enumerate(plist):
            ci_low, ci_up = conf_int.loc[p, lower_col], conf_int.loc[p, upper_col]
            fam_dict[p] = {
                'beta': float(params[p]),
                'se': float(bse[p]),
                'p_raw': float(pvalues[p]),
                'p_fdr': float(adj[i]),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_up),
                'exp_beta': float(np.exp(params[p])),
                'exp_ci_lower': float(np.exp(ci_low)),
                'exp_ci_upper': float(np.exp(ci_up)),
            }
        fdr_results[fam] = fam_dict

    # Conditional contrasts (log rate ratio interpretation)
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        ci = conf_int.loc['Dose[T.High]']
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(bse['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'exp_beta': float(np.exp(params['Dose[T.High]'])),
            'exp_ci_lower': float(np.exp(ci[lower_col])),
            'exp_ci_upper': float(np.exp(ci[upper_col])),
            'description': 'High - Low within RS (log rate ratio)',
        }
    if 'Task[T.DMT]:Dose[T.High]' in params.index:
        ci = conf_int.loc['Task[T.DMT]:Dose[T.High]']
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': float(params['Task[T.DMT]:Dose[T.High]']),
            'se': float(bse['Task[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['Task[T.DMT]:Dose[T.High]']),
            'exp_beta': float(np.exp(params['Task[T.DMT]:Dose[T.High]'])),
            'exp_ci_lower': float(np.exp(ci[lower_col])),
            'exp_ci_upper': float(np.exp(ci[upper_col])),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }

    out['fdr_families'] = fdr_results
    out['conditional_contrasts'] = contrasts
    return out


def generate_report(result, diagnostics: Dict, hyp: Dict, df: pd.DataFrame, output_dir: str) -> str:
    lines: List[str] = [
        '=' * 80,
        'GEE ANALYSIS REPORT: SCR counts per minute (first 9 minutes)',
        '=' * 80,
        '',
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        '',
        'DESIGN:',
        '  Within-subjects 2×2: Task (RS vs DMT) × Dose (Low vs High)',
        '  Time windows: 9 one-minute windows (0-8 minutes)',
        '  Dependent variable: SCR count per minute',
        '',
        'MODEL SPECIFICATION:',
        '  Poisson GEE with log link, Exchangeable working correlation',
        '  count ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c',
        '',
        'MODEL FIT SUMMARY:',
        f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
        f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
        f"  Working corr: {diagnostics.get('working_corr', 'N/A')}",
        f"  Converged: {diagnostics.get('converged', 'N/A')}",
        '',
    ]

    if result is not None:
        lines.extend(['PARAMETERS (β on log scale; exp(β) rate ratio):', '-' * 65])
        summary_df = pd.DataFrame({
            'beta': result.params,
            'se': result.bse,
            'p': result.pvalues,
        })
        ci = result.conf_int()
        summary_df['ci_lower'] = ci[0]
        summary_df['ci_upper'] = ci[1]
        summary_df['exp_beta'] = np.exp(summary_df['beta'])
        summary_df['exp_ci_lower'] = np.exp(summary_df['ci_lower'])
        summary_df['exp_ci_upper'] = np.exp(summary_df['ci_upper'])
        for idx, row in summary_df.iterrows():
            lines.extend([
                f"  {idx}:",
                f"    β = {row['beta']:8.4f} (SE={row['se']:6.4f}), Wald p={row['p']:6.4f}",
                f"    95% CI: [{row['ci_lower']:8.4f}, {row['ci_upper']:8.4f}]",
                f"    exp(β) = {row['exp_beta']:8.4f} [ {row['exp_ci_lower']:8.4f}, {row['exp_ci_upper']:8.4f} ]",
                '',
            ])

    if 'fdr_families' in hyp:
        lines.extend(['FDR-CORRECTED HYPOTHESES (by family):', '=' * 60, ''])
        for fam, famres in hyp['fdr_families'].items():
            lines.extend([f'FAMILY {fam.upper()}:', '-' * 30])
            for param, res in famres.items():
                sig = '***' if res['p_fdr'] < 0.001 else '**' if res['p_fdr'] < 0.01 else '*' if res['p_fdr'] < 0.05 else ''
                lines.extend([
                    f"  {param}:",
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    exp(β) = {res['exp_beta']:8.4f} [ {res['exp_ci_lower']:8.4f}, {res['exp_ci_upper']:8.4f} ]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {sig}",
                    '',
                ])
            lines.append('')

    if 'conditional_contrasts' in hyp:
        lines.extend(['CONDITIONAL CONTRASTS (log scale and rate ratio):', '-' * 40])
        for _, res in hyp['conditional_contrasts'].items():
            sig = '***' if res['p_raw'] < 0.001 else '**' if res['p_raw'] < 0.01 else '*' if res['p_raw'] < 0.05 else ''
            lines.extend([
                f"  {res['description']}:",
                f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {sig}",
                f"    exp(β) = {res['exp_beta']:8.4f} [ {res['exp_ci_lower']:8.4f}, {res['exp_ci_upper']:8.4f} ]",
                '',
            ])

    # Summary statistics
    lines.extend(['', 'DATA SUMMARY:', '-' * 30])
    cell = df.groupby(['Task', 'Dose'])['count'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Cell means (count by Task × Dose):', str(cell), ''])
    trend = df.groupby('minute')['count'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Time trend (count by minute):', str(trend), ''])
    lines.extend(['', '=' * 80])

    out_path = os.path.join(output_dir, 'gee_analysis_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def prepare_coefficient_table(result) -> pd.DataFrame:
    if result is None:
        raise ValueError('No GEE result available')
    coef_df = pd.DataFrame({
        'parameter': result.params.index,
        'beta': result.params.values,
        'se': result.bse.values,
        'p': result.pvalues.values,
    })
    ci = result.conf_int()
    coef_df['ci_lower'] = ci.iloc[:, 0].values
    coef_df['ci_upper'] = ci.iloc[:, 1].values
    coef_df['exp_beta'] = np.exp(coef_df['beta'])
    coef_df['exp_ci_lower'] = np.exp(coef_df['ci_lower'])
    coef_df['exp_ci_upper'] = np.exp(coef_df['ci_upper'])
    # Determine family for color coding
    def _family(param: str) -> str:
        if param in ['Intercept', 'minute_c'] or param.startswith('Task[T.') is False and param.startswith('Dose[T.') is False:
            return 'Other'
        if param.startswith('Task[T.') and ':Dose' in param:
            return 'Interaction'
        if param.startswith('Task[T.'):
            return 'Task'
        if param.startswith('Dose[T.'):
            return 'Dose'
        return 'Other'
    coef_df['family'] = coef_df['parameter'].apply(_family)
    coef_df['significant'] = coef_df['p'] < 0.05
    return coef_df


def create_gee_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    # Plot β with 95% CI; annotate exp(β) in labels
    order_params = [
        'Task[T.DMT]',
        'Dose[T.High]',
        'Task[T.DMT]:minute_c',
        'Dose[T.High]:minute_c',
        'Task[T.DMT]:Dose[T.High]'
    ]
    plot_df = coef_df[coef_df['parameter'].isin(order_params)].copy()
    plot_df['order'] = plot_df['parameter'].apply(lambda p: order_params.index(p) if p in order_params else 999)
    plot_df = plot_df.sort_values('order')

    family_colors = {
        'Task': COLOR_DMT_HIGH,
        'Dose': COLOR_RS_HIGH,
        'Interaction': COLOR_DMT_LOW,
        'Other': '#666666',
    }

    labels = {
        'Task[T.DMT]': 'Task (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'Task[T.DMT]:minute_c': 'Task × Time',
        'Dose[T.High]:minute_c': 'Dose × Time',
        'Task[T.DMT]:Dose[T.High]': 'Task × Dose',
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    y_positions = np.arange(len(plot_df))
    for i, row in enumerate(plot_df.itertuples(index=False)):
        color = family_colors.get(row.family, '#666666')
        ax.plot([row.ci_lower, row.ci_upper], [i, i], color=color, linewidth=3.0 if row.significant else 2.5, alpha=1.0 if row.significant else 0.8)
        ax.scatter(row.beta, i, color=color, s=70 if row.significant else 55, alpha=1.0 if row.significant else 0.8, edgecolors='white', linewidths=1)
        # Annotate exp(beta)
        ax.text(row.ci_upper + 0.02, i, f"exp(β)={row.exp_beta:.2f}", va='center', fontsize=26, color=color)

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([labels.get(p, p) for p in plot_df['parameter']], fontsize=33)
    ax.set_xlabel('Coefficient (β, log rate)\nwith 95% CI')
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    grouped = df.groupby(['minute', 'Task', 'Dose'])['count']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'Task', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['Task'].astype(str) + '_' + stats_df['Dose'].astype(str)
    alpha = 1 - confidence
    t_crit = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1) if scistats is not None else 1.96
    stats_df['ci_lower'] = stats_df['mean'] - t_crit * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_crit * stats_df['se']
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    return stats_df


def _compute_fdr_significant_segments(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> List[Tuple[float, float]]:
    """Return contiguous x-intervals where High vs Low differ after BH-FDR."""
    if scistats is None:
        return []
    n_time = A.shape[1]
    pvals = np.full(n_time, np.nan, dtype=float)
    for t in range(n_time):
        a = A[:, t]
        b = B[:, t]
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) >= 2:
            try:
                _, p = scistats.ttest_rel(a[mask], b[mask])
                pvals[t] = float(p)
            except Exception:
                pvals[t] = np.nan
    valid = ~np.isnan(pvals)
    if not np.any(valid):
        return []
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid].tolist())
    adj[valid] = np.array(adj_vals)
    sig = adj < alpha
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segments.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return segments


def _compute_fdr_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> Dict:
    if scistats is None:
        return {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    n_time = A.shape[1]
    pvals = np.full(n_time, np.nan, dtype=float)
    for t in range(n_time):
        a = A[:, t]
        b = B[:, t]
        mask = (~np.isnan(a)) & (~np.isnan(b))
        if np.sum(mask) >= 2:
            try:
                _, p = scistats.ttest_rel(a[mask], b[mask])
                pvals[t] = float(p)
            except Exception:
                pvals[t] = np.nan
    valid = ~np.isnan(pvals)
    if not np.any(valid):
        return {'alpha': alpha, 'pvals': pvals.tolist(), 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid].tolist())
    adj[valid] = np.array(adj_vals)
    sig = adj < alpha
    segments: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segments.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return {'alpha': alpha, 'pvals': pvals.tolist(), 'pvals_adj': adj.tolist(), 'sig_mask': sig.tolist(), 'segments': segments}


def create_timecourse_plot(df: pd.DataFrame, stats_df: pd.DataFrame, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    # RS
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(data['minute'], data['mean'], color=color, lw=2.5, marker='o', markersize=5, label=condition.replace('RS_', ''))
        # Use SEM shading (to match legacy test/ EmotiPhai plot)
        ax1.fill_between(data['minute'], data['mean'] - data['se'], data['mean'] + data['se'], color=color, alpha=0.2)
    # Shade significant segments (paired across subjects High vs Low)
    try:
        # Build per-subject matrices for RS
        subjects = list(df['subject'].cat.categories)
        n_time = N_MINUTES
        H = np.full((len(subjects), n_time), np.nan, dtype=float)
        L = np.full((len(subjects), n_time), np.nan, dtype=float)
        for si, subj in enumerate(subjects):
            sdf = df[df['subject'] == subj]
            for minute in range(1, N_MINUTES + 1):
                row_h = sdf[(sdf['Task'] == 'RS') & (sdf['Dose'] == 'High') & (sdf['minute'] == minute)]['count']
                row_l = sdf[(sdf['Task'] == 'RS') & (sdf['Dose'] == 'Low') & (sdf['minute'] == minute)]['count']
                if len(row_h) == 1:
                    H[si, minute - 1] = float(row_h.iloc[0])
                if len(row_l) == 1:
                    L[si, minute - 1] = float(row_l.iloc[0])
        x_grid = np.arange(1, N_MINUTES + 1)
        segs = _compute_fdr_significant_segments(H, L, x_grid)
        for x0, x1 in segs:
            ax1.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
        # Write RS report
        rs_res = _compute_fdr_results(H, L, x_grid)
        with open(os.path.join(os.path.dirname(output_path), '..', 'fdr_segments_all_subs_scr_rs.txt'), 'w', encoding='utf-8') as f:
            lines = [
                'FDR COMPARISON: RS High vs Low (SCR rate)',
                f"Alpha = {rs_res.get('alpha', 0.05)}",
                '',
            ]
            segs2 = rs_res.get('segments', [])
            lines.append(f"Significant segments (count={len(segs2)}):")
            if len(segs2) == 0:
                lines.append('  - None')
            for (a, b) in segs2:
                lines.append(f"  - {a:.1f}–{b:.1f} min")
            p_adj = [v for v in rs_res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
            if p_adj:
                lines.append(f"Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
            f.write('\n'.join(lines))
    except Exception:
        pass
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('SCR')
    ticks = list(range(1, N_MINUTES + 1))
    ax1.set_xticks(ticks)
    ax1.set_xlim(0.8, N_MINUTES + 0.2)
    ax1.set_title('Resting State (RS)', fontweight='bold')
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    # DMT
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(data['minute'], data['mean'], color=color, lw=2.5, marker='o', markersize=5, label=condition.replace('DMT_', ''))
        # Use SEM shading (to match legacy test/ EmotiPhai plot)
        ax2.fill_between(data['minute'], data['mean'] - data['se'], data['mean'] + data['se'], color=color, alpha=0.2)
    try:
        # Build per-subject matrices for DMT
        subjects = list(df['subject'].cat.categories)
        n_time = N_MINUTES
        H = np.full((len(subjects), n_time), np.nan, dtype=float)
        L = np.full((len(subjects), n_time), np.nan, dtype=float)
        for si, subj in enumerate(subjects):
            sdf = df[df['subject'] == subj]
            for minute in range(1, N_MINUTES + 1):
                row_h = sdf[(sdf['Task'] == 'DMT') & (sdf['Dose'] == 'High') & (sdf['minute'] == minute)]['count']
                row_l = sdf[(sdf['Task'] == 'DMT') & (sdf['Dose'] == 'Low') & (sdf['minute'] == minute)]['count']
                if len(row_h) == 1:
                    H[si, minute - 1] = float(row_h.iloc[0])
                if len(row_l) == 1:
                    L[si, minute - 1] = float(row_l.iloc[0])
        x_grid = np.arange(1, N_MINUTES + 1)
        segs = _compute_fdr_significant_segments(H, L, x_grid)
        for x0, x1 in segs:
            ax2.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
        # Write DMT report
        dmt_res = _compute_fdr_results(H, L, x_grid)
        with open(os.path.join(os.path.dirname(output_path), '..', 'fdr_segments_all_subs_scr_dmt.txt'), 'w', encoding='utf-8') as f:
            lines = [
                'FDR COMPARISON: DMT High vs Low (SCR rate)',
                f"Alpha = {dmt_res.get('alpha', 0.05)}",
                '',
            ]
            segs2 = dmt_res.get('segments', [])
            lines.append(f"Significant segments (count={len(segs2)}):")
            if len(segs2) == 0:
                lines.append('  - None')
            for (a, b) in segs2:
                lines.append(f"  - {a:.1f}–{b:.1f} min")
            p_adj = [v for v in dmt_res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
            if p_adj:
                lines.append(f"Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
            f.write('\n'.join(lines))
    except Exception:
        pass
    ax2.set_xlabel('Time (minutes)')
    ax2.set_xticks(ticks)
    ax2.set_xlim(0.8, N_MINUTES + 0.2)
    ax2.set_title('DMT', fontweight='bold')
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_task_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    task_means = stats_df.groupby(['minute', 'Task']).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    task_se = stats_df.groupby(['minute', 'Task'])['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    task_means = task_means.merge(task_se, on=['minute', 'Task'], how='left')
    t_crit = 1.96
    task_means['ci_lower'] = task_means['mean'] - t_crit * task_means['se']
    task_means['ci_upper'] = task_means['mean'] + t_crit * task_means['se']

    fig, ax = plt.subplots(figsize=(10, 6))
    for task, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        data = task_means[task_means['Task'] == task].sort_values('minute')
        ax.plot(data['minute'], data['mean'], color=color, lw=3, marker='o', markersize=6, label=f'{task}')
        ax.fill_between(data['minute'], data['ci_lower'], data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('SCR')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    leg = ax.legend(loc='upper right', frameon=True, fancybox=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    ticks = list(range(1, N_MINUTES + 1))
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(data['minute'], data['mean'], color=color, lw=2.5, marker='o', markersize=4, label=condition.replace('RS_', ''))
        ax1.fill_between(data['minute'], data['ci_lower'], data['ci_upper'], color=color, alpha=0.2)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('SCR')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    ax1.set_xticks(ticks)
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)

    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(data['minute'], data['mean'], color=color, lw=2.5, marker='o', markersize=4, label=condition.replace('DMT_', ''))
        ax2.fill_between(data['minute'], data['ci_lower'], data['ci_upper'], color=color, alpha=0.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    ax2.set_xticks(ticks)
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_stacked_subjects_plot(df: pd.DataFrame, output_path: str) -> None:
    # Pivot by subject to reconstruct per-subject minute curves per condition
    subjects = list(df['subject'].cat.categories)
    n = len(subjects)
    fig, axes = plt.subplots(n, 2, figsize=(18, max(6.0, 3.2 * n)), sharex=True, sharey=True, gridspec_kw={'hspace': 0.8, 'wspace': 0.35})
    if n == 1:
        axes = np.array([axes])
    minute_ticks = list(range(1, N_MINUTES + 1))
    from matplotlib.lines import Line2D
    for i, subj in enumerate(subjects):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]
        sdf = df[df['subject'] == subj]
        # RS
        for cond, color in [('Low', COLOR_RS_LOW), ('High', COLOR_RS_HIGH)]:
            d = sdf[(sdf['Task'] == 'RS') & (sdf['Dose'] == cond)].sort_values('minute')
            ax_rs.plot(d['minute'], d['count'], color=color, lw=1.4)
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.set_ylabel('SCR', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xticks(minute_ticks)
        ax_rs.set_xlim(0.8, N_MINUTES + 0.2)
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        # RS legend
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=COLOR_RS_HIGH, lw=1.4, label='RS High'),
            Line2D([0], [0], color=COLOR_RS_LOW, lw=1.4, label='RS Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)
        # DMT
        for cond, color in [('Low', COLOR_DMT_LOW), ('High', COLOR_DMT_HIGH)]:
            d = sdf[(sdf['Task'] == 'DMT') & (sdf['Dose'] == cond)].sort_values('minute')
            ax_dmt.plot(d['minute'], d['count'], color=color, lw=1.4)
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.set_ylabel('SCR', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xticks(minute_ticks)
        ax_dmt.set_xlim(0.8, N_MINUTES + 0.2)
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        # DMT legend
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=COLOR_DMT_HIGH, lw=1.4, label='DMT High'),
            Line2D([0], [0], color=COLOR_DMT_LOW, lw=1.4, label='DMT Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)
        # Subject row title
    plt.tight_layout(pad=2.0)
    # Place subject labels after layout using axes positions
    for i, subj in enumerate(subjects):
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 1].get_position()
        y_center = (pos_left.y0 + pos_left.y1) / 2.0
        x_center = (pos_left.x1 + pos_right.x0) / 2.0
        fig.text(x_center, y_center + 0.02, subj, ha='center', va='bottom', fontweight='bold', fontsize=STACKED_SUBJECT_FONTSIZE, transform=fig.transFigure)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    table = coef_df[['parameter', 'beta', 'se', 'p', 'ci_lower', 'ci_upper', 'exp_beta', 'exp_ci_lower', 'exp_ci_upper', 'family']].copy()
    table = table.round({'beta': 4, 'se': 4, 'p': 6, 'ci_lower': 4, 'ci_upper': 4, 'exp_beta': 4, 'exp_ci_lower': 4, 'exp_ci_upper': 4})
    # Interpretation column
    def _interp(row):
        if row['p'] < 0.05:
            return f"Significant rate ratio {row['exp_beta']:.2f}"
        return f"Non-significant (rate ratio {row['exp_beta']:.2f})"
    table['interpretation'] = table.apply(_interp, axis=1)
    table.to_csv(output_path, index=False)


def generate_captions_file(output_dir: str, n_subjects: int) -> None:
    captions = [
        'Figure: SCR Rate Time Course (EmotiPhai)\n\n'
        'Group-level mean ± 95% CI for SCR counts per minute over the first 9 minutes. '
        'Left: RS with High (green) vs Low (purple). Right: DMT with High (red) vs Low (blue).',
        '',
        'Figure: GEE Coefficients (log-scale and rate ratios)\n\n'
        'Fixed-effects β (log rate) with 95% CI; annotated with exp(β) (rate ratios).',
        '',
        'Figure: Marginal Means Over Time\n\n'
        'Empirical means ± 95% CI for each condition (RS Low/High; DMT Low/High) across minutes 0–8.',
        '',
        'Figure: Task × Dose Interaction Panels\n\n'
        'Separate RS and DMT panels, High vs Low dose, with mean ± 95% CI over time.',
        '',
        'Supplementary: Stacked Subject Time Series (9 minutes)\n\n'
        f'One row per subject (N={n_subjects}); RS left, DMT right; shows within-subject variability in SCR rates.',
    ]
    with open(os.path.join(output_dir, 'captions_scr.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def main() -> bool:
    out_dir = os.path.join('results', 'eda', 'scr')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    try:
        # Data
        df = prepare_long_data_scr()
        df.to_csv(os.path.join(out_dir, 'scr_counts_long_data.csv'), index=False)

        # GEE model
        result, diagnostics = fit_gee_poisson_model(df)
        hyp = hypothesis_testing_with_fdr(result, df)

        # Report
        report_path = generate_report(result, diagnostics, hyp, df, out_dir)

        # Coefficient table and plot
        coef_df = prepare_coefficient_table(result)
        create_effect_sizes_table(coef_df, os.path.join(plots_dir, 'effect_sizes_table.csv'))
        create_gee_coefficient_plot(coef_df[coef_df['parameter'].isin([
            'Task[T.DMT]', 'Dose[T.High]', 'Task[T.DMT]:minute_c', 'Dose[T.High]:minute_c', 'Task[T.DMT]:Dose[T.High]'
        ])], os.path.join(plots_dir, 'gee_coefficient_plot.png'))

        # Empirical means and derivative plots
        stats_df = compute_empirical_means_and_ci(df)
        create_timecourse_plot(df, stats_df, os.path.join(plots_dir, 'all_subs_scr_rate_timecourse.png'))
        create_task_effect_plot(stats_df, os.path.join(plots_dir, 'task_main_effect.png'))
        create_interaction_plot(stats_df, os.path.join(plots_dir, 'task_dose_interaction.png'))

        # Stacked per-subject (optional, for parity with SCL)
        create_stacked_subjects_plot(df, os.path.join(plots_dir, 'stacked_subs_scr_rate.png'))

        # Captions
        generate_captions_file(out_dir, len(df['subject'].unique()))
    except Exception as e:
        print(f'SCR analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)


