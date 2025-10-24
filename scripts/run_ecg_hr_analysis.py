# -*- coding: utf-8 -*-
"""
Unified HR (Heart Rate) Analysis: LME modeling and visualization (first 9 minutes).

This script processes ECG-derived heart rate (HR) from NeuroKit ECG_Rate column:
  1) Build long-format per-minute HR dataset (mean HR per minute, 0–8)
  2) Fit LME with Task × Dose and time effects; apply BH-FDR per family
  3) Create coefficient, marginal means, interaction, diagnostics plots
  4) Write model summary as TXT and figure captions
  5) Generate group-level timecourse plot for the first 9 minutes with FDR

Outputs are written to: results/ecg/hr/

Run:
  python scripts/run_ecg_hr_analysis.py
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
    SUJETOS_VALIDADOS_ECG,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
except Exception:
    mixedlm = None

try:
    from scipy import stats as scistats
except Exception:
    scistats = None


#############################
# Plot aesthetics & centralized hyperparameters
#############################

# Centralized font sizes and legend settings
AXES_TITLE_SIZE = 29
AXES_LABEL_SIZE = 36
TICK_LABEL_SIZE = 28
TICK_LABEL_SIZE_SMALL = 24

# Legend sizes (global and a smaller variant for dense, per-subject panels)
LEGEND_FONTSIZE = 18
LEGEND_FONTSIZE_SMALL = 14
LEGEND_MARKERSCALE = 1.6
LEGEND_BORDERPAD = 0.6
LEGEND_HANDLELENGTH = 3.0
LEGEND_LABELSPACING = 0.7
LEGEND_BORDERAXESPAD = 0.9

# Stacked per-subject figure specific sizes
STACKED_AXES_LABEL_SIZE = 22
STACKED_TICK_LABEL_SIZE = 14
STACKED_SUBJECT_FONTSIZE = 30

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

# Fixed colors to match the project's convention
COLOR_RS_HIGH = 'tab:green'
COLOR_RS_LOW = 'tab:purple'
COLOR_DMT_HIGH = 'tab:red'
COLOR_DMT_LOW = 'tab:blue'

# Analysis window: first 9 minutes
N_MINUTES = 9  # minutes 0..8
MAX_TIME_SEC = 60 * N_MINUTES

# Baseline correction flag (optional, disabled by default)
BASELINE_CORRECTION = False


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_ecg_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    """Build paths to DMT High and Low ECG CSVs."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv")
    return high_csv, low_csv


def build_rs_ecg_path(subject: str, session: str) -> str:
    """Build path to RS ECG CSV."""
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)  # 'Alta' or 'Baja'
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'ecg', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}.csv")


def load_ecg_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load ECG CSV and validate required columns."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    
    # Validate required columns
    if 'ECG_Rate' not in df.columns:
        return None
    
    # Reconstruct time if missing
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    
    return df


def compute_hr_mean_per_minute(t: np.ndarray, hr: np.ndarray, minute: int) -> Optional[float]:
    """Compute mean HR for a specific minute window."""
    start_time = minute * 60.0
    end_time = (minute + 1) * 60.0
    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None
    hr_win = hr[mask]
    hr_win_valid = hr_win[(hr_win >= 35) & (hr_win <= 200)]  # Physiological range
    if len(hr_win_valid) < 2:
        return None
    return float(np.nanmean(hr_win_valid))


def prepare_long_data_hr() -> pd.DataFrame:
    """Build long-format per-minute HR table (first 9 minutes)."""
    rows: List[Dict] = []
    for subject in SUJETOS_VALIDADOS_ECG:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_ecg_paths(subject, high_session, low_session)
        rs_high_path = build_rs_ecg_path(subject, high_session)
        rs_low_path = build_rs_ecg_path(subject, low_session)

        dmt_high = load_ecg_csv(dmt_high_path)
        dmt_low = load_ecg_csv(dmt_low_path)
        rs_high = load_ecg_csv(rs_high_path)
        rs_low = load_ecg_csv(rs_low_path)
        
        if any(x is None for x in (dmt_high, dmt_low, rs_high, rs_low)):
            continue

        # Extract time and HR
        t_dmt_high = dmt_high['time'].to_numpy()
        hr_dmt_high = pd.to_numeric(dmt_high['ECG_Rate'], errors='coerce').to_numpy()
        t_dmt_low = dmt_low['time'].to_numpy()
        hr_dmt_low = pd.to_numeric(dmt_low['ECG_Rate'], errors='coerce').to_numpy()
        t_rs_high = rs_high['time'].to_numpy()
        hr_rs_high = pd.to_numeric(rs_high['ECG_Rate'], errors='coerce').to_numpy()
        t_rs_low = rs_low['time'].to_numpy()
        hr_rs_low = pd.to_numeric(rs_low['ECG_Rate'], errors='coerce').to_numpy()

        # Optional baseline correction
        if BASELINE_CORRECTION:
            # Subtract mean of first minute
            baseline_dmt_high = compute_hr_mean_per_minute(t_dmt_high, hr_dmt_high, 0)
            baseline_dmt_low = compute_hr_mean_per_minute(t_dmt_low, hr_dmt_low, 0)
            baseline_rs_high = compute_hr_mean_per_minute(t_rs_high, hr_rs_high, 0)
            baseline_rs_low = compute_hr_mean_per_minute(t_rs_low, hr_rs_low, 0)
            if baseline_dmt_high is not None:
                hr_dmt_high = hr_dmt_high - baseline_dmt_high
            if baseline_dmt_low is not None:
                hr_dmt_low = hr_dmt_low - baseline_dmt_low
            if baseline_rs_high is not None:
                hr_rs_high = hr_rs_high - baseline_rs_high
            if baseline_rs_low is not None:
                hr_rs_low = hr_rs_low - baseline_rs_low

        for minute in range(N_MINUTES):
            hr_dmt_h = compute_hr_mean_per_minute(t_dmt_high, hr_dmt_high, minute)
            hr_dmt_l = compute_hr_mean_per_minute(t_dmt_low, hr_dmt_low, minute)
            hr_rs_h = compute_hr_mean_per_minute(t_rs_high, hr_rs_high, minute)
            hr_rs_l = compute_hr_mean_per_minute(t_rs_low, hr_rs_low, minute)
            
            if None not in (hr_dmt_h, hr_dmt_l, hr_rs_h, hr_rs_l):
                minute_label = minute + 1  # store minutes as 1..9 instead of 0..8
                rows.extend([
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'High', 'HR': hr_dmt_h},
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'Low', 'HR': hr_dmt_l},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'High', 'HR': hr_rs_h},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'Low', 'HR': hr_rs_l},
                ])

    if not rows:
        raise ValueError('No valid HR data found for any subject!')

    df = pd.DataFrame(rows)
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['minute_c'] = df['minute'] - df['minute'].mean()
    return df


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    if mixedlm is None:
        return None, {'error': 'statsmodels not available'}
    try:
        formula = 'HR ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c'
        model = mixedlm(formula, df, groups=df['subject'])  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fitted = model.fit()
            convergence_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        return None, {'error': str(e)}
    diagnostics = {
        'aic': getattr(fitted, 'aic', np.nan),
        'bic': getattr(fitted, 'bic', np.nan),
        'loglik': getattr(fitted, 'llf', np.nan),
        'n_obs': getattr(fitted, 'nobs', len(df)),
        'n_groups': len(df['subject'].unique()),
        'convergence_warnings': convergence_warnings,
        'random_effects_var': getattr(fitted, 'cov_re', None),
        'residual_var': getattr(fitted, 'scale', np.nan),
    }
    return fitted, diagnostics


def plot_model_diagnostics(fitted_model, df: pd.DataFrame, output_dir: str) -> None:
    if fitted_model is None:
        return
    fitted_vals = fitted_model.fittedvalues
    residuals = fitted_model.resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    try:
        if scistats is not None:
            scistats.probplot(residuals, dist='norm', plot=axes[0, 1])
    except Exception:
        pass
    subject_means = df.groupby('subject').apply(lambda x: residuals[x.index].mean())
    axes[1, 0].bar(range(len(subject_means)), subject_means.values, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('Mean Residual')
    minute_residuals = df.groupby('minute').apply(lambda x: residuals[x.index].mean())
    axes[1, 1].plot(minute_residuals.index, minute_residuals.values, 'o-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Minute')
    axes[1, 1].set_ylabel('Mean Residual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lme_diagnostics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    order = np.argsort(p_array)
    sorted_p = p_array[order]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[order[i]] = sorted_p[i]
        else:
            adjusted[order[i]] = min(sorted_p[i] * n / (i + 1), adjusted[order[i + 1]])
    return np.minimum(adjusted, 1.0).tolist()


def hypothesis_testing_with_fdr(fitted_model) -> Dict:
    if fitted_model is None:
        return {}
    params = fitted_model.params
    pvalues = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    stderr = fitted_model.bse
    results: Dict[str, Dict] = {
        'all_params': params.to_dict(),
        'all_pvalues': pvalues.to_dict(),
        'all_stderr': stderr.to_dict(),
        'conf_int': conf_int.to_dict(),
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
            fam_dict[p] = {
                'beta': float(params[p]),
                'se': float(stderr[p]),
                'p_raw': float(pvalues[p]),
                'p_fdr': float(adj[i]),
                'ci_lower': float(conf_int.loc[p, 0]),
                'ci_upper': float(conf_int.loc[p, 1]),
            }
        fdr_results[fam] = fam_dict
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(stderr['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'description': 'High - Low within RS',
        }
    if all(k in params.index for k in ['Dose[T.High]', 'Task[T.DMT]:Dose[T.High]']):
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': float(params['Task[T.DMT]:Dose[T.High]']),
            'se': float(stderr['Task[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['Task[T.DMT]:Dose[T.High]']),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }
    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    return results


def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, df: pd.DataFrame, output_dir: str) -> str:
    lines: List[str] = [
        '=' * 80,
        'LME ANALYSIS REPORT: HR (Heart Rate) by Minute (first 9 minutes)',
        '=' * 80,
        '',
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        '',
        'DESIGN:',
        '  Within-subjects 2×2: Task (RS vs DMT) × Dose (Low vs High)',
        '  Time windows: 9 one-minute windows (0-8 minutes)',
        '  Dependent variable: Mean HR per minute (bpm)',
        f'  Baseline correction: {BASELINE_CORRECTION}',
        '',
        'MODEL SPECIFICATION:',
        '  Fixed effects: HR ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c',
        '  Random effects: ~ 1 | subject',
        '  Where minute_c = minute - mean(minute) [centered time]',
        '',
    ]
    if fitted_model is not None:
        lines.extend([
            'MODEL FIT STATISTICS:',
            f"  AIC: {diagnostics.get('aic', np.nan):.2f}",
            f"  BIC: {diagnostics.get('bic', np.nan):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', np.nan):.6f}",
            '',
        ])
        warns = diagnostics.get('convergence_warnings', [])
        if warns:
            lines.extend(['CONVERGENCE WARNINGS:', *[f'  - {w}' for w in warns], ''])
        else:
            lines.append('Model converged without warnings\n')
    if 'fdr_families' in hypothesis_results:
        lines.extend(['HYPOTHESIS TESTING RESULTS (with BH-FDR correction):', '=' * 60, ''])
        for fam, famres in hypothesis_results['fdr_families'].items():
            lines.extend([f'FAMILY {fam.upper()}:', '-' * 30])
            for param, res in famres.items():
                sig = '***' if res['p_fdr'] < 0.001 else '**' if res['p_fdr'] < 0.01 else '*' if res['p_fdr'] < 0.05 else ''
                lines.extend([
                    f'  {param}:',
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {sig}",
                    '',
                ])
            lines.append('')
    if 'conditional_contrasts' in hypothesis_results:
        lines.extend(['CONDITIONAL CONTRASTS:', '-' * 30])
        for _, res in hypothesis_results['conditional_contrasts'].items():
            sig = '***' if res['p_raw'] < 0.001 else '**' if res['p_raw'] < 0.01 else '*' if res['p_raw'] < 0.05 else ''
            lines.extend([f"  {res['description']}:", f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {sig}", ''])
    lines.extend(['', 'DATA SUMMARY:', '-' * 30])
    cell = df.groupby(['Task', 'Dose'], observed=False)['HR'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Cell means (HR by Task × Dose):', str(cell), ''])
    trend = df.groupby('minute', observed=False)['HR'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Time trend (HR by minute):', str(trend), ''])
    lines.extend(['', '=' * 80])
    out_path = os.path.join(output_dir, 'lme_analysis_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def load_lme_results_from_report(report_path: str) -> Dict:
    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    coefficients: Dict[str, Dict] = {}
    current_family: Optional[str] = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('FAMILY TASK:'):
            current_family = 'Task'
        elif line.startswith('FAMILY DOSE:'):
            current_family = 'Dose'
        elif line.startswith('FAMILY INTERACTION:'):
            current_family = 'Interaction'
        elif line.startswith('CONDITIONAL CONTRASTS:'):
            current_family = 'Contrasts'
        if current_family and line.endswith(':') and not line.startswith('FAMILY') and not line.startswith('CONDITIONAL'):
            name = line.rstrip(':').strip()
            if i + 3 < len(lines):
                beta_line = lines[i + 1].strip()
                ci_line = lines[i + 2].strip()
                p_line = lines[i + 3].strip()
                try:
                    if beta_line.startswith('β ='):
                        parts = beta_line.split(',')
                        beta = float(parts[0].split('=')[1].strip())
                        se = float(parts[1].split('=')[1].strip())
                    if ci_line.startswith('95% CI:'):
                        txt = ci_line.replace('95% CI:', '').replace('[', '').replace(']', '').strip()
                        ci_lower = float(txt.split(',')[0])
                        ci_upper = float(txt.split(',')[1])
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw = float([p for p in p_parts if 'p_raw' in p][0].split('=')[1].strip().split()[0])
                        p_fdr_text = [p for p in p_parts if 'p_FDR' in p][0].split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        significance = '***' if '***' in p_fdr_text else '**' if '**' in p_fdr_text else '*' if '*' in p_fdr_text else ''
                    coefficients[name] = {
                        'family': current_family,
                        'beta': beta,
                        'se': se,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'p_raw': p_raw,
                        'p_fdr': p_fdr,
                        'significance': significance,
                    }
                except Exception:
                    continue
    return coefficients


def prepare_coefficient_data(coefficients: Dict) -> pd.DataFrame:
    order = [
        'Task[T.DMT]',
        'Dose[T.High]',
        'Task[T.DMT]:minute_c',
        'Dose[T.High]:minute_c',
        'Task[T.DMT]:Dose[T.High]'
    ]
    labels = {
        'Task[T.DMT]': 'Task (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'Task[T.DMT]:minute_c': 'Task × Time',
        'Dose[T.High]:minute_c': 'Dose × Time',
        'Task[T.DMT]:Dose[T.High]': 'Task × Dose'
    }
    fam_colors = {
        'Task': COLOR_DMT_HIGH,
        'Dose': COLOR_RS_HIGH,
        'Interaction': COLOR_DMT_LOW,
    }
    rows: List[Dict] = []
    for i, p in enumerate(order):
        if p in coefficients:
            c = coefficients[p]
            rows.append({
                'parameter': p,
                'label': labels.get(p, p),
                'beta': c['beta'],
                'se': c['se'],
                'ci_lower': c['ci_lower'],
                'ci_upper': c['ci_upper'],
                'p_raw': c['p_raw'],
                'p_fdr': c['p_fdr'],
                'significance': c['significance'],
                'family': c['family'],
                'order': i,
                'significant': c['p_fdr'] < 0.05,
                'color': fam_colors.get(c['family'], '#666666'),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No coefficient data to plot')
    return df


def create_coefficient_plot(coef_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    coef_df = coef_df.sort_values('order')
    y_positions = np.arange(len(coef_df))
    for _, row in coef_df.iterrows():
        y_pos = y_positions[row['order']]
        linewidth = 3.5 if row['significant'] else 2.5
        alpha = 1.0 if row['significant'] else 0.8
        marker_size = 70 if row['significant'] else 55
        ax.plot([row['ci_lower'], row['ci_upper']], [y_pos, y_pos], color=row['color'], linewidth=linewidth, alpha=alpha)
        ax.scatter(row['beta'], y_pos, color=row['color'], s=marker_size, alpha=alpha, edgecolors='white', linewidths=1)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(coef_df['label'], fontsize=33)
    ax.set_xlabel('Coefficient Estimate (β)\nwith 95% CI')
    ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.subplots_adjust(left=0.28)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def compute_empirical_means_and_ci(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    grouped = df.groupby(['minute', 'Task', 'Dose'], observed=False)['HR']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'Task', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['Task'].astype(str) + '_' + stats_df['Dose'].astype(str)
    alpha = 1 - confidence
    t_critical = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1) if scistats is not None else 1.96
    stats_df['ci_lower'] = stats_df['mean'] - t_critical * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + t_critical * stats_df['se']
    stats_df['ci_lower'] = stats_df['ci_lower'].fillna(stats_df['mean'])
    stats_df['ci_upper'] = stats_df['ci_upper'].fillna(stats_df['mean'])
    return stats_df


def create_marginal_means_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    conditions = stats_df['condition'].unique()
    for condition in sorted(conditions):
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        if len(cond_data) == 0:
            continue
        if condition == 'RS_Low':
            color = COLOR_RS_LOW
        elif condition == 'RS_High':
            color = COLOR_RS_HIGH
        elif condition == 'DMT_Low':
            color = COLOR_DMT_LOW
        elif condition == 'DMT_High':
            color = COLOR_DMT_HIGH
        else:
            color = '#666666'
        ax.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('_', ' '), marker='o', markersize=5)
        ax.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('HR (bpm)')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_task_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    task_means = stats_df.groupby(['minute', 'Task'], observed=False).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    task_se = stats_df.groupby(['minute', 'Task'], observed=False)['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    task_means = task_means.merge(task_se, on=['minute', 'Task'], how='left')
    t_crit = 1.96
    task_means['ci_lower'] = task_means['mean'] - t_crit * task_means['se']
    task_means['ci_upper'] = task_means['mean'] + t_crit * task_means['se']
    fig, ax = plt.subplots(figsize=(10, 6))
    for task, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        task_data = task_means[task_means['Task'] == task].sort_values('minute')
        ax.plot(task_data['minute'], task_data['mean'], color=color, linewidth=3, label=f'{task}', marker='o', markersize=6)
        ax.fill_between(task_data['minute'], task_data['ci_lower'], task_data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('HR (bpm)')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('RS_', ''), marker='o', markersize=4)
        ax1.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('HR (bpm)')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    ticks = list(range(1, N_MINUTES + 1))
    ax1.set_xticks(ticks)
    ax1.set_xlim(0.8, N_MINUTES + 0.2)
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('DMT_', ''), marker='o', markersize=4)
        ax2.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    ax2.set_xticks(ticks)
    ax2.set_xlim(0.8, N_MINUTES + 0.2)
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    table = coef_df[['label', 'beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'significance', 'family']].copy()
    numeric = ['beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr']
    table[numeric] = table[numeric].round(4)
    table['interpretation'] = table.apply(lambda r: (
        f"{'Significant' if r['p_fdr'] < 0.05 else 'Non-significant'} "
        f"{'increase' if r['beta'] > 0 else 'decrease'} in HR"), axis=1)
    table.to_csv(output_path, index=False)


def create_model_summary_txt(diagnostics: Dict, coef_df: pd.DataFrame, output_path: str) -> None:
    lines: List[str] = [
        'LME MODEL SUMMARY',
        '=' * 60,
        '',
        'Fixed Effects Formula:',
        'HR ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c',
        '',
        'Random Effects: ~ 1 | subject',
        '',
        'Model Fit Statistics:',
        f"AIC: {diagnostics.get('aic', np.nan):.2f}",
        f"BIC: {diagnostics.get('bic', np.nan):.2f}",
        f"Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
        f"N observations: {diagnostics.get('n_obs', 'N/A')}",
        f"N subjects: {diagnostics.get('n_groups', 'N/A')}",
        '',
        'Significant Fixed Effects (p_FDR < 0.05):',
    ]
    sig = coef_df[coef_df['p_fdr'] < 0.05]
    if len(sig) == 0:
        lines.append('• None')
    else:
        for _, row in sig.iterrows():
            lines.append(f"• {row['label']}: β = {row['beta']:.3f} {row['significance']}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _resample_to_grid(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.full_like(t_grid, np.nan, dtype=float)
    t_valid = t[valid]
    y_valid = y[valid]
    yg = np.full_like(t_grid, np.nan, dtype=float)
    mask = (t_grid >= t_valid[0]) & (t_grid <= t_valid[-1])
    if np.any(mask):
        yg[mask] = np.interp(t_grid[mask], t_valid, y_valid)
    return yg


def _compute_fdr_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> Dict:
    """Compute paired t-test across time, apply BH-FDR, and summarize results."""
    result: Dict[str, object] = {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    if scistats is None:
        return result
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
    valid_idx = np.where(~np.isnan(pvals))[0]
    if len(valid_idx) == 0:
        return result
    adj = np.full_like(pvals, np.nan, dtype=float)
    adj_vals = benjamini_hochberg_correction(pvals[valid_idx].tolist())
    adj[valid_idx] = np.array(adj_vals, dtype=float)
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
    result['pvals'] = pvals.tolist()
    result['pvals_adj'] = adj.tolist()
    result['sig_mask'] = sig.tolist()
    result['segments'] = segments
    return result


def create_combined_summary_plot(out_dir: str) -> Optional[str]:
    """Create RS+DMT summary (9 minutes) with FDR shading."""
    t_grid = np.arange(0.0, 541.0, 0.5)

    task_data: Dict[str, Dict[str, np.ndarray]] = {}
    for kind in ['RS', 'DMT']:
        high_curves: List[np.ndarray] = []
        low_curves: List[np.ndarray] = []
        for subject in SUJETOS_VALIDADOS_ECG:
            try:
                if kind == 'DMT':
                    high_session, low_session = determine_sessions(subject)
                    p_high, p_low = build_ecg_paths(subject, high_session, low_session)
                    d_high = load_ecg_csv(p_high)
                    d_low = load_ecg_csv(p_low)
                    if any(x is None for x in (d_high, d_low)):
                        continue
                    th = d_high['time'].to_numpy()
                    yh = pd.to_numeric(d_high['ECG_Rate'], errors='coerce').to_numpy()
                    tl = d_low['time'].to_numpy()
                    yl = pd.to_numeric(d_low['ECG_Rate'], errors='coerce').to_numpy()
                else:  # RS
                    high_session, low_session = determine_sessions(subject)
                    p_rsh = build_rs_ecg_path(subject, high_session)
                    p_rsl = build_rs_ecg_path(subject, low_session)
                    r_high = load_ecg_csv(p_rsh)
                    r_low = load_ecg_csv(p_rsl)
                    if any(x is None for x in (r_high, r_low)):
                        continue
                    th = r_high['time'].to_numpy()
                    yh = pd.to_numeric(r_high['ECG_Rate'], errors='coerce').to_numpy()
                    tl = r_low['time'].to_numpy()
                    yl = pd.to_numeric(r_low['ECG_Rate'], errors='coerce').to_numpy()

                # Trim to 0..540s
                mh = (th >= 0.0) & (th <= 540.0)
                ml = (tl >= 0.0) & (tl <= 540.0)
                th, yh = th[mh], yh[mh]
                tl, yl = tl[ml], yl[ml]

                # Resample to common grid
                high_curves.append(_resample_to_grid(th, yh, t_grid))
                low_curves.append(_resample_to_grid(tl, yl, t_grid))
            except Exception:
                continue

        if high_curves and low_curves:
            H = np.vstack(high_curves)
            L = np.vstack(low_curves)
            mean_h = np.nanmean(H, axis=0)
            mean_l = np.nanmean(L, axis=0)
            sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
            sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])
            task_data[kind] = {
                'mean_h': mean_h,
                'mean_l': mean_l,
                'sem_h': sem_h,
                'sem_l': sem_l,
                'H_mat': H,
                'L_mat': L,
            }
        else:
            return None

    if len(task_data) != 2:
        return None

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    c_rs_high, c_rs_low = COLOR_RS_HIGH, COLOR_RS_LOW

    # RS (left)
    rs = task_data['RS']
    rs_fdr = _compute_fdr_results(rs['H_mat'], rs['L_mat'], t_grid)
    for x0, x1 in rs_fdr.get('segments', []):
        ax1.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
    line_h1 = ax1.plot(t_grid, rs['mean_h'], color=c_rs_high, lw=2.0, marker=None, label='High')[0]
    ax1.fill_between(t_grid, rs['mean_h'] - rs['sem_h'], rs['mean_h'] + rs['sem_h'], color=c_rs_high, alpha=0.25)
    line_l1 = ax1.plot(t_grid, rs['mean_l'], color=c_rs_low, lw=2.0, marker=None, label='Low')[0]
    ax1.fill_between(t_grid, rs['mean_l'] - rs['sem_l'], rs['mean_l'] + rs['sem_l'], color=c_rs_low, alpha=0.25)
    legend1 = ax1.legend([line_h1, line_l1], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('HR (bpm)')
    ax1.set_title('Resting State (RS)', fontweight='bold')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')

    # DMT (right)
    dmt = task_data['DMT']
    dmt_fdr = _compute_fdr_results(dmt['H_mat'], dmt['L_mat'], t_grid)
    for x0, x1 in dmt_fdr.get('segments', []):
        ax2.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
    line_h2 = ax2.plot(t_grid, dmt['mean_h'], color=c_dmt_high, lw=2.0, marker=None, label='High')[0]
    ax2.fill_between(t_grid, dmt['mean_h'] - dmt['sem_h'], dmt['mean_h'] + dmt['sem_h'], color=c_dmt_high, alpha=0.25)
    line_l2 = ax2.plot(t_grid, dmt['mean_l'], color=c_dmt_low, lw=2.0, marker=None, label='Low')[0]
    ax2.fill_between(t_grid, dmt['mean_l'] - dmt['sem_l'], dmt['mean_l'] + dmt['sem_l'], color=c_dmt_low, alpha=0.25)
    legend2 = ax2.legend([line_h2, line_l2], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title('DMT', fontweight='bold')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')

    # X ticks 0..9:00 every minute
    minute_ticks = np.arange(0, 10)
    for ax in (ax1, ax2):
        ax.set_xticks(minute_ticks * 60)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x // 60)}"))
        ax.set_xlim(0.0, 540.0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_ecg_hr.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    # Write FDR report
    try:
        report_lines: List[str] = []
        report_lines.append('FDR COMPARISON: High vs Low (All Subjects, RS and DMT)')
        report_lines.append(f"Alpha = {rs_fdr.get('alpha', 0.05)}")
        report_lines.append('')
        def _panel_section(name: str, res: Dict):
            report_lines.append(f'PANEL {name}:')
            segs = res.get('segments', [])
            report_lines.append(f"  Significant segments (count={len(segs)}):")
            if len(segs) == 0:
                report_lines.append('    - None')
            for (x0, x1) in segs:
                report_lines.append(f"    - {x0:.1f}s–{x1:.1f}s ( {x0/60:.2f}–{x1/60:.2f} min )")
            p_adj = [v for v in res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
            if p_adj:
                report_lines.append(f"  Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
            report_lines.append('')
        _panel_section('RS', rs_fdr)
        _panel_section('DMT', dmt_fdr)
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_ecg_hr.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    except Exception:
        pass
    return out_path


def create_dmt_only_20min_plot(out_dir: str) -> Optional[str]:
    """Create DMT-only extended plot (~19 minutes) with FDR shading.
    
    Saves results/ecg/hr/all_subs_dmt_ecg_hr.png
    """
    t_grid = np.arange(0.0, 1150.0, 0.5)
    high_curves: List[np.ndarray] = []
    low_curves: List[np.ndarray] = []
    for subject in SUJETOS_VALIDADOS_ECG:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_ecg_paths(subject, high_session, low_session)
            d_high = load_ecg_csv(p_high)
            d_low = load_ecg_csv(p_low)
            if any(x is None for x in (d_high, d_low)):
                continue
            th = d_high['time'].to_numpy()
            yh = pd.to_numeric(d_high['ECG_Rate'], errors='coerce').to_numpy()
            tl = d_low['time'].to_numpy()
            yl = pd.to_numeric(d_low['ECG_Rate'], errors='coerce').to_numpy()
            mh = (th >= 0.0) & (th < 1150.0)
            ml = (tl >= 0.0) & (tl < 1150.0)
            th, yh = th[mh], yh[mh]
            tl, yl = tl[ml], yl[ml]
            high_curves.append(_resample_to_grid(th, yh, t_grid))
            low_curves.append(_resample_to_grid(tl, yl, t_grid))
        except Exception:
            continue
    if not (high_curves and low_curves):
        return None
    H = np.vstack(high_curves)
    L = np.vstack(low_curves)
    mean_h = np.nanmean(H, axis=0)
    mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # Colors: DMT High red, DMT Low blue
    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    # Shade significant differences (High vs Low) after FDR and collect report
    fdr_res = _compute_fdr_results(H, L, t_grid)
    for x0, x1 in fdr_res.get('segments', []):
        ax.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
    line_h = ax.plot(t_grid, mean_h, color=c_dmt_high, lw=2.0, marker=None, label='High')[0]
    ax.fill_between(t_grid, mean_h - sem_h, mean_h + sem_h, color=c_dmt_high, alpha=0.25)
    line_l = ax.plot(t_grid, mean_l, color=c_dmt_low, lw=2.0, marker=None, label='Low')[0]
    ax.fill_between(t_grid, mean_l - sem_l, mean_l + sem_l, color=c_dmt_low, alpha=0.25)

    legend = ax.legend([line_h, line_l], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('HR (bpm)')
    ax.set_title('DMT', fontweight='bold')
    # Subtle grid: y-only, light alpha
    ax.grid(True, which='major', axis='y', alpha=0.25)
    ax.grid(False, which='major', axis='x')
    minute_ticks = np.arange(0.0, 1141.0, 60.0)
    ax.set_xticks(minute_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x//60)}"))
    ax.set_xlim(0.0, 1150.0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_dmt_ecg_hr.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    # Write FDR report for DMT-only plot
    try:
        lines: List[str] = [
            'FDR COMPARISON: High vs Low (All Subjects, DMT only)',
            f"Alpha = {fdr_res.get('alpha', 0.05)}",
            '',
        ]
        segs = fdr_res.get('segments', [])
        lines.append(f"Significant segments (count={len(segs)}):")
        if len(segs) == 0:
            lines.append('  - None')
        for (x0, x1) in segs:
            lines.append(f"  - {x0:.1f}s–{x1:.1f}s ( {x0/60:.2f}–{x1/60:.2f} min )")
        p_adj = [v for v in fdr_res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
        if p_adj:
            lines.append(f"Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_dmt_ecg_hr.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path


def create_stacked_subjects_plot(out_dir: str) -> Optional[str]:
    """Create a stacked per-subject figure (RS left, DMT right) for 9 minutes."""
    limit_sec = 550.0
    rows: List[Dict] = []
    for subject in SUJETOS_VALIDADOS_ECG:
        try:
            # DMT
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_ecg_paths(subject, high_session, low_session)
            d_high = load_ecg_csv(p_high)
            d_low = load_ecg_csv(p_low)
            if any(x is None for x in (d_high, d_low)):
                continue
            th = d_high['time'].to_numpy()
            yh = pd.to_numeric(d_high['ECG_Rate'], errors='coerce').to_numpy()
            tl = d_low['time'].to_numpy()
            yl = pd.to_numeric(d_low['ECG_Rate'], errors='coerce').to_numpy()
            mh = (th >= 0.0) & (th <= limit_sec)
            ml = (tl >= 0.0) & (tl <= limit_sec)
            th, yh = th[mh], yh[mh]
            tl, yl = tl[ml], yl[ml]

            # RS
            p_r1 = build_rs_ecg_path(subject, 'session1')
            p_r2 = build_rs_ecg_path(subject, 'session2')
            r1 = load_ecg_csv(p_r1)
            r2 = load_ecg_csv(p_r2)
            if any(x is None for x in (r1, r2)):
                continue
            tr1 = r1['time'].to_numpy()
            yr1 = pd.to_numeric(r1['ECG_Rate'], errors='coerce').to_numpy()
            tr2 = r2['time'].to_numpy()
            yr2 = pd.to_numeric(r2['ECG_Rate'], errors='coerce').to_numpy()
            m1 = (tr1 >= 0.0) & (tr1 <= limit_sec)
            m2 = (tr2 >= 0.0) & (tr2 <= limit_sec)
            tr1, yr1 = tr1[m1], yr1[m1]
            tr2, yr2 = tr2[m2], yr2[m2]

            # RS dose mapping
            try:
                dose_s1 = get_dosis_sujeto(subject, 1)
            except Exception:
                dose_s1 = 'Alta'
            cond1 = 'High' if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a') else 'Low'
            cond2 = 'Low' if cond1 == 'High' else 'High'

            rows.append({
                'subject': subject,
                't_rs1': tr1, 'y_rs1': yr1, 'cond1': cond1,
                't_rs2': tr2, 'y_rs2': yr2, 'cond2': cond2,
                't_dmt_h': th, 'y_dmt_h': yh,
                't_dmt_l': tl, 'y_dmt_l': yl,
            })
        except Exception:
            continue

    if not rows:
        return None

    n = len(rows)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(18, max(6.0, 3.2 * n)),
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0.8, 'wspace': 0.35}
    )
    if n == 1:
        axes = np.array([axes])

    c_dmt_high, c_dmt_low = COLOR_DMT_HIGH, COLOR_DMT_LOW
    c_rs_high, c_rs_low = COLOR_RS_HIGH, COLOR_RS_LOW
    minute_ticks = np.arange(0.0, 541.0, 60.0)

    from matplotlib.lines import Line2D

    for i, row in enumerate(rows):
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]

        # RS
        if row['cond1'] == 'High':
            ax_rs.plot(row['t_rs1'], row['y_rs1'], color=c_rs_high, lw=1.4)
        else:
            ax_rs.plot(row['t_rs1'], row['y_rs1'], color=c_rs_low, lw=1.4)
        if row['cond2'] == 'High':
            ax_rs.plot(row['t_rs2'], row['y_rs2'], color=c_rs_high, lw=1.4)
        else:
            ax_rs.plot(row['t_rs2'], row['y_rs2'], color=c_rs_low, lw=1.4)
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.set_ylabel('HR (bpm)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(0.0, limit_sec)
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=c_rs_high, lw=1.4, label='RS High'),
            Line2D([0], [0], color=c_rs_low, lw=1.4, label='RS Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)

        # DMT
        ax_dmt.plot(row['t_dmt_h'], row['y_dmt_h'], color=c_dmt_high, lw=1.4)
        ax_dmt.plot(row['t_dmt_l'], row['y_dmt_l'], color=c_dmt_low, lw=1.4)
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.set_ylabel('HR (bpm)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(0.0, limit_sec)
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=c_dmt_high, lw=1.4, label='DMT High'),
            Line2D([0], [0], color=c_dmt_low, lw=1.4, label='DMT Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)

        # ticks & limits
        ax_rs.set_xticks(minute_ticks)
        ax_rs.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x//60)}"))
        ax_dmt.set_xticks(minute_ticks)
        ax_dmt.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x//60)}"))

    fig.tight_layout(pad=2.0)

    # Add subject codes centered between columns
    for i, row in enumerate(rows):
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 1].get_position()
        y_center = (pos_left.y0 + pos_left.y1) / 2.0
        x_center = (pos_left.x1 + pos_right.x0) / 2.0
        fig.text(
            x_center,
            y_center + 0.02,
            row['subject'],
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=STACKED_SUBJECT_FONTSIZE,
            transform=fig.transFigure,
        )
    out_path = os.path.join(out_dir, 'plots', 'stacked_subs_ecg_hr.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def generate_captions_file(output_dir: str) -> None:
    captions = [
        'Figure: LME Coefficients (HR)\n\n'
        'Point estimates (β) and 95% CIs for fixed effects from the mixed model. '
        'Reference line at zero aids interpretation. Significant effects are visually emphasized.',
        '',
        'Figure: Marginal Means Over Time (RS vs DMT × High vs Low)\n\n'
        'Group-level mean ± 95% CI of HR (bpm) across the first 9 minutes for each condition (RS Low/High, DMT Low/High). '
        'Legends indicate dose levels; shading shows uncertainty.',
        '',
        'Figure: Main Task Effect Over Time\n\n'
        'Mean ± 95% CI for RS and DMT (averaged across dose) across minutes 0–8. '
        'Illustrates overall task separation and temporal trend.',
        '',
        'Figure: Task × Dose Interaction (Panels)\n\n'
        'Left: RS Low vs High; Right: DMT Low vs High. Lines show mean ± 95% CI across minutes 0–8. '
        'Highlights how dose effects differ between tasks.',
        '',
        'Figure: Group-level HR Timecourse (9 min)\n\n'
        'Two panels (RS, DMT) showing mean ± SEM over time; High vs Low dose with legends. '
        'Gray shading indicates FDR-significant differences (High vs Low) across time. '
        'Time axis in minutes (0–9).',
        '',
        'Figure: DMT-only HR Timecourse (~19 min)\n\n'
        'Extended timecourse plot showing DMT High vs Low over approximately 19 minutes. '
        'Gray shading indicates FDR-significant differences (High vs Low) across time. '
        'Mean ± SEM for all subjects. Time axis in minutes (0–19).',
        '',
        'Figure: Stacked Per-Subject HR Timecourse (9 min)\n\n'
        'Individual subject traces for RS (left) and DMT (right) conditions. '
        'High/Low dose traces shown in respective colors. Subject codes centered between panels.',
    ]
    with open(os.path.join(output_dir, 'captions_hr.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def main() -> bool:
    out_dir = os.path.join('results', 'ecg', 'hr')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    try:
        # Data
        print("Preparing long-format HR data...")
        df = prepare_long_data_hr()
        df.to_csv(os.path.join(out_dir, 'hr_minute_long_data.csv'), index=False)
        print(f"  ✓ Loaded {len(df['subject'].unique())} subjects with {len(df)} observations")
        
        # LME
        print("Fitting LME model...")
        fitted, diagnostics = fit_lme_model(df)
        plot_model_diagnostics(fitted, df, plots_dir)
        
        # Hypothesis testing + report
        print("Performing hypothesis testing with FDR correction...")
        hyp = hypothesis_testing_with_fdr(fitted)
        report_path = generate_report(fitted, diagnostics, hyp, df, out_dir)
        print(f"  ✓ Report saved: {report_path}")
        
        # Coefficients
        print("Generating plots...")
        coefs = load_lme_results_from_report(report_path)
        coef_df = prepare_coefficient_data(coefs)
        create_coefficient_plot(coef_df, os.path.join(plots_dir, 'lme_coefficient_plot.png'))
        create_effect_sizes_table(coef_df, os.path.join(plots_dir, 'effect_sizes_table.csv'))
        
        # Marginal means + derived plots
        stats_df = compute_empirical_means_and_ci(df)
        create_marginal_means_plot(stats_df, os.path.join(plots_dir, 'marginal_means_all_conditions.png'))
        create_task_effect_plot(stats_df, os.path.join(plots_dir, 'task_main_effect.png'))
        create_interaction_plot(stats_df, os.path.join(plots_dir, 'task_dose_interaction.png'))
        
        # Summary statistics
        overall = stats_df.groupby('condition').agg({
            'mean': 'mean',
            'se': lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1)),
            'n': 'mean'
        }).round(4)
        overall['ci_lower'] = overall['mean'] - 1.96 * overall['se']
        overall['ci_upper'] = overall['mean'] + 1.96 * overall['se']
        overall.to_csv(os.path.join(plots_dir, 'summary_statistics.csv'))
        
        # Model summary txt
        create_model_summary_txt(diagnostics, coef_df, os.path.join(out_dir, 'model_summary.txt'))
        
        # Combined summary (9 min with FDR)
        print("Creating combined summary plot with FDR...")
        create_combined_summary_plot(out_dir)
        
        # DMT-only extended (19 min with FDR)
        print("Creating DMT-only extended plot...")
        create_dmt_only_20min_plot(out_dir)
        
        # Stacked per-subject
        print("Creating stacked per-subject plot...")
        create_stacked_subjects_plot(out_dir)
        
        # Captions
        generate_captions_file(out_dir)
        
        print(f"\n✓ HR analysis complete! Results in: {out_dir}")
    except Exception as e:
        print(f'HR analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)

