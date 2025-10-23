# -*- coding: utf-8 -*-
"""
Unified SMNA Analysis: LME modeling and visualization (first 9 minutes).

This script combines and streamlines the SMNA AUC LME analysis and plotting into
one reproducible pipeline that:
  1) Prepares long-format SMNA AUC data by minute (0-8) across conditions
  2) Fits an LME model with Task × Dose and time effects
  3) Applies BH-FDR to families of hypotheses
  4) Produces publication-ready plots with consistent aesthetics
  5) Writes a plain-text model summary and captions for figures

Outputs are written to: results/eda/smna/

Run:
  python scripts/run_eda_smna_analysis.py
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
import seaborn as sns  # optional

# Import project config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import (
    DERIVATIVES_DATA,
    SUJETOS_VALIDADOS_EDA,
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import shapiro
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


#############################
# Plot aesthetics & centralized hyperparameters
#############################

# Centralized font sizes and legend settings (aligned with SCL/SCR)
AXES_TITLE_SIZE = 29
AXES_LABEL_SIZE = 36
TICK_LABEL_SIZE = 28
TICK_LABEL_SIZE_SMALL = 24

LEGEND_FONTSIZE = 14
LEGEND_FONTSIZE_SMALL = 12
LEGEND_MARKERSCALE = 1.6
LEGEND_BORDERPAD = 0.6
LEGEND_HANDLELENGTH = 3.0

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

# Fixed colors to match EmotiPhai SCR rate script
COLOR_RS_HIGH = 'tab:green'
COLOR_RS_LOW = 'tab:purple'
COLOR_DMT_HIGH = 'tab:red'
COLOR_DMT_LOW = 'tab:blue'

# Analysis window: first 9 minutes
N_MINUTES = 9  # minutes 0..8


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_cvx_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    """Build full paths for DMT high and DMT low CVX decomposition CSVs for EDA."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high_cvx_decomposition.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low_cvx_decomposition.csv")
    return high_csv, low_csv


def build_rs_cvx_path(subject: str, session: str) -> str:
    """Build RS CVX decomposition path for a given subject/session using session dose."""
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)  # 'Alta' or 'Baja'
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}_cvx_decomposition.csv")


def load_cvx_smna(csv_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load time and SMNA arrays from CVX decomposition CSV.

    Returns (t, smna) or None if missing/invalid.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if 'SMNA' not in df.columns:
        return None

    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)

    smna = pd.to_numeric(df['SMNA'], errors='coerce').fillna(0.0).to_numpy()
    return t, smna


def compute_auc_minute_window(t: np.ndarray, y: np.ndarray, minute: int) -> Optional[float]:
    """Compute AUC for a specific 1-minute window (minute=0..8)."""
    start_time = minute * 60.0
    end_time = (minute + 1) * 60.0

    mask = (t >= start_time) & (t < end_time)
    if not np.any(mask):
        return None

    t_win = t[mask]
    y_win = y[mask]
    if len(t_win) < 2:
        return None
    return float(np.trapz(y_win, t_win))


def prepare_long_data() -> pd.DataFrame:
    """Prepare data in long format for LME analysis (first 9 minutes).

    Returns:
        DataFrame with columns: subject, minute, Task, Dose, AUC, minute_c
    """
    rows: List[Dict] = []
    n_processed = 0

    for subject in SUJETOS_VALIDADOS_EDA:
        high_session, low_session = determine_sessions(subject)

        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs_high_path = build_rs_cvx_path(subject, high_session)
        rs_low_path = build_rs_cvx_path(subject, low_session)

        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs_high = load_cvx_smna(rs_high_path)
        rs_low = load_cvx_smna(rs_low_path)

        if None in (dmt_high, dmt_low, rs_high, rs_low):
            continue

        t_dmt_high, smna_dmt_high = dmt_high
        t_dmt_low, smna_dmt_low = dmt_low
        t_rs_high, smna_rs_high = rs_high
        t_rs_low, smna_rs_low = rs_low

        subject_rows: List[Dict] = []
        for minute in range(N_MINUTES):  # 0..8
            auc_dmt_high = compute_auc_minute_window(t_dmt_high, smna_dmt_high, minute)
            auc_dmt_low = compute_auc_minute_window(t_dmt_low, smna_dmt_low, minute)
            auc_rs_high = compute_auc_minute_window(t_rs_high, smna_rs_high, minute)
            auc_rs_low = compute_auc_minute_window(t_rs_low, smna_rs_low, minute)

            if None not in (auc_dmt_high, auc_dmt_low, auc_rs_high, auc_rs_low):
                minute_label = minute + 1
                subject_rows.extend([
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high},
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'High', 'AUC': auc_rs_high},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low},
                ])

        if subject_rows:
            rows.extend(subject_rows)
            n_processed += 1

    if not rows:
        raise ValueError("No valid data found for any subject!")

    df = pd.DataFrame(rows)

    # Set categorical variables with proper ordering
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])

    # Centered minute
    mean_minute = df['minute'].mean()
    df['minute_c'] = df['minute'] - mean_minute

    return df


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    """Fit the LME model with specified fixed and random effects."""
    try:
        formula = "AUC ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c"
        model = mixedlm(formula, df, groups=df["subject"])  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
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
    """Create diagnostic plots (titles removed)."""
    if fitted_model is None:
        return

    fitted_vals = fitted_model.fittedvalues
    residuals = fitted_model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')

    # Q-Q plot of residuals
    try:
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    except Exception:
        pass

    # Residuals by subject
    subject_means = df.groupby('subject').apply(lambda x: residuals[x.index].mean())
    axes[1, 0].bar(range(len(subject_means)), subject_means.values, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('Mean Residual')

    # Residuals by minute
    minute_residuals = df.groupby('minute').apply(lambda x: residuals[x.index].mean())
    axes[1, 1].plot(minute_residuals.index, minute_residuals.values, 'o-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Minute')
    axes[1, 1].set_ylabel('Mean Residual')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'lme_diagnostics.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def benjamini_hochberg_correction(p_values: List[float]) -> List[float]:
    p_array = np.array(p_values, dtype=float)
    n = len(p_array)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    adjusted_p = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted_p[sorted_indices[i]] = min(sorted_p[i] * n / (i + 1), adjusted_p[sorted_indices[i + 1]])
    return np.minimum(adjusted_p, 1.0).tolist()


def _compute_fdr_significant_segments(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> List[Tuple[float, float]]:
    if not SCIPY_AVAILABLE:
        return []
    from scipy import stats as scistats
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
    segs: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segs.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return segs


def _compute_fdr_results(A: np.ndarray, B: np.ndarray, x_grid: np.ndarray, alpha: float = 0.05) -> Dict:
    if not SCIPY_AVAILABLE:
        return {'alpha': alpha, 'pvals': [], 'pvals_adj': [], 'sig_mask': [], 'segments': []}
    from scipy import stats as scistats
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
    segs: List[Tuple[float, float]] = []
    i = 0
    while i < len(sig):
        if sig[i]:
            start = i
            while i + 1 < len(sig) and sig[i + 1]:
                i += 1
            end = i
            segs.append((float(x_grid[start]), float(x_grid[end])))
        i += 1
    return {'alpha': alpha, 'pvals': pvals.tolist(), 'pvals_adj': adj.tolist(), 'sig_mask': sig.tolist(), 'segments': segs}

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

    # Family (i): Task effects
    for param in ['Task[T.DMT]', 'Task[T.DMT]:minute_c']:
        if param in pvalues.index:
            families['Task'].append(param)

    # Family (ii): Dose effects
    for param in ['Dose[T.High]', 'Dose[T.High]:minute_c']:
        if param in pvalues.index:
            families['Dose'].append(param)

    # Family (iii): Interaction
    for param in ['Task[T.DMT]:Dose[T.High]']:
        if param in pvalues.index:
            families['Interaction'].append(param)

    fdr_results: Dict[str, Dict] = {}
    for family_name, param_list in families.items():
        if not param_list:
            continue
        family_pvals = [pvalues[param] for param in param_list]
        fdr_pvals = benjamini_hochberg_correction(family_pvals)
        family_dict: Dict[str, Dict] = {}
        for i, param in enumerate(param_list):
            family_dict[param] = {
                'beta': float(params[param]),
                'se': float(stderr[param]),
                'p_raw': float(pvalues[param]),
                'p_fdr': float(fdr_pvals[i]),
                'ci_lower': float(conf_int.loc[param, 0]),
                'ci_upper': float(conf_int.loc[param, 1]),
            }
        fdr_results[family_name] = family_dict

    # Conditional contrasts
    contrasts: Dict[str, Dict] = {}
    if 'Dose[T.High]' in params.index:
        contrasts['High_Low_within_RS'] = {
            'beta': float(params['Dose[T.High]']),
            'se': float(stderr['Dose[T.High]']),
            'p_raw': float(pvalues['Dose[T.High]']),
            'description': 'High - Low within RS',
        }
    if all(p in params.index for p in ['Dose[T.High]', 'Task[T.DMT]:Dose[T.High]']):
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
    """Generate comprehensive analysis report (TXT)."""
    report_lines: List[str] = []

    report_lines.extend([
        "=" * 80,
        "LME ANALYSIS REPORT: SMNA AUC by Minute (first 9 minutes)",
        "=" * 80,
        "",
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        "",
        "DESIGN:",
        "  Within-subjects 2×2: Task (RS vs DMT) × Dose (Low vs High)",
        "  Time windows: 9 one-minute windows (0-8 minutes)",
        "  Dependent variable: AUC of SMNA signal",
        "",
        "MODEL SPECIFICATION:",
        "  Fixed effects: AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c",
        "  Random effects: ~ 1 | subject",
        "  Where minute_c = minute - mean(minute) [centered time]",
        "",
    ])

    if fitted_model is not None:
        report_lines.extend([
            "MODEL FIT STATISTICS:",
            f"  AIC: {diagnostics.get('aic', np.nan):.2f}",
            f"  BIC: {diagnostics.get('bic', np.nan):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', np.nan):.6f}",
            "",
        ])
        warnings_list = diagnostics.get('convergence_warnings', [])
        if warnings_list:
            report_lines.extend(["CONVERGENCE WARNINGS:"] + [f"  - {w}" for w in warnings_list] + [""])
        else:
            report_lines.append("Model converged without warnings\n")

    if 'fdr_families' in hypothesis_results:
        report_lines.extend([
            "HYPOTHESIS TESTING RESULTS (with BH-FDR correction):",
            "=" * 60,
            "",
        ])
        for family_name, family_results in hypothesis_results['fdr_families'].items():
            report_lines.extend([f"FAMILY {family_name.upper()}:", "-" * 30])
            for param, res in family_results.items():
                significance = "***" if res['p_fdr'] < 0.001 else "**" if res['p_fdr'] < 0.01 else "*" if res['p_fdr'] < 0.05 else ""
                report_lines.extend([
                    f"  {param}:",
                    f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}",
                    f"    95% CI: [{res['ci_lower']:8.4f}, {res['ci_upper']:8.4f}]",
                    f"    p_raw = {res['p_raw']:6.4f}, p_FDR = {res['p_fdr']:6.4f} {significance}",
                    "",
                ])
            report_lines.append("")

    if 'conditional_contrasts' in hypothesis_results:
        report_lines.extend(["CONDITIONAL CONTRASTS:", "-" * 30])
        for _, res in hypothesis_results['conditional_contrasts'].items():
            significance = "***" if res['p_raw'] < 0.001 else "**" if res['p_raw'] < 0.01 else "*" if res['p_raw'] < 0.05 else ""
            report_lines.extend([
                f"  {res['description']}:",
                f"    β = {res['beta']:8.4f}, SE = {res['se']:6.4f}, p = {res['p_raw']:6.4f} {significance}",
                "",
            ])

    # Data summary
    report_lines.extend(["", "DATA SUMMARY:", "-" * 30])
    summary_stats = df.groupby(['Task', 'Dose'])['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend(["Cell means (AUC by Task × Dose):", str(summary_stats), ""]) 
    time_stats = df.groupby('minute')['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend(["Time trend (AUC by minute):", str(time_stats), ""]) 

    report_lines.extend(["", "=" * 80])

    report_path = os.path.join(output_dir, 'lme_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    return report_path


def load_lme_results_from_report(report_path: str) -> Dict:
    """Parse the LME analysis report to extract coefficients for plotting."""
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
            param_name = line.rstrip(':').strip()
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
                        ci_text = ci_line.replace('95% CI:', '').replace('[', '').replace(']', '').strip()
                        ci_lower = float(ci_text.split(',')[0])
                        ci_upper = float(ci_text.split(',')[1])
                    if 'p_raw =' in p_line and 'p_FDR =' in p_line:
                        p_parts = p_line.split(',')
                        p_raw = float([p for p in p_parts if 'p_raw' in p][0].split('=')[1].strip().split()[0])
                        p_fdr_text = [p for p in p_parts if 'p_FDR' in p][0].split('=')[1].strip()
                        p_fdr = float(p_fdr_text.split()[0])
                        significance = ''
                        if '***' in p_fdr_text:
                            significance = '***'
                        elif '**' in p_fdr_text:
                            significance = '**'
                        elif '*' in p_fdr_text:
                            significance = '*'
                    coefficients[param_name] = {
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
    param_order = [
        'Task[T.DMT]',
        'Dose[T.High]',
        'Task[T.DMT]:minute_c',
        'Dose[T.High]:minute_c',
        'Task[T.DMT]:Dose[T.High]'
    ]
    param_labels = {
        'Task[T.DMT]': 'Task (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'Task[T.DMT]:minute_c': 'Task × Time',
        'Dose[T.High]:minute_c': 'Dose × Time',
        'Task[T.DMT]:Dose[T.High]': 'Task × Dose'
    }
    family_colors = {
        'Task': COLOR_DMT_HIGH,        # Map Task-related to DMT color family
        'Dose': COLOR_RS_HIGH,         # Dose-related in RS color family for contrast in figure
        'Interaction': COLOR_DMT_LOW,  # Interaction in contrasting color
    }

    rows: List[Dict] = []
    for i, param in enumerate(param_order):
        if param in coefficients:
            c = coefficients[param]
            rows.append({
                'parameter': param,
                'label': param_labels.get(param, param),
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
                'color': family_colors.get(c['family'], '#666666'),
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
    grouped = df.groupby(['minute', 'Task', 'Dose'])['AUC']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'Task', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['Task'].astype(str) + '_' + stats_df['Dose'].astype(str)
    alpha = 1 - confidence
    from scipy import stats as scistats  # local import in case scipy missing at top
    t_critical = scistats.t.ppf(1 - alpha/2, stats_df['n'] - 1)
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
        # Map colors based on condition
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
    ax.set_ylabel('SMNA AUC')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_task_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    task_means = stats_df.groupby(['minute', 'Task']).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    # Rough SE aggregation
    task_se = stats_df.groupby(['minute', 'Task'])['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    task_means = task_means.merge(task_se, on=['minute', 'Task'], how='left')
    t_crit = 1.96
    task_means['ci_lower'] = task_means['mean'] - t_crit * task_means['se']
    task_means['ci_upper'] = task_means['mean'] + t_crit * task_means['se']

    fig, ax = plt.subplots(figsize=(10, 6))
    # Ensure legend order: DMT first, RS second
    for task, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        task_data = task_means[task_means['Task'] == task].sort_values('minute')
        ax.plot(task_data['minute'], task_data['mean'], color=color, linewidth=3, label=f'{task}', marker='o', markersize=6)
        ax.fill_between(task_data['minute'], task_data['ci_lower'], task_data['ci_upper'], color=color, alpha=0.2)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('SMNA AUC')
    ticks = list(range(1, N_MINUTES + 1))
    ax.set_xticks(ticks)
    ax.set_xlim(0.8, N_MINUTES + 0.2)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


def create_interaction_plot(stats_df: pd.DataFrame, output_path: str, df_raw: Optional[pd.DataFrame] = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    # Compute FDR-based significant segments per panel using subject-wise data
    try:
        subjects = sorted(stats_df['minute'].index.unique())  # dummy to trigger except if malformed
    except Exception:
        pass

    # RS panel (High above Low in legend)
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('RS_', ''), marker='o', markersize=4)
        ax1.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('SMNA AUC')
    ax1.grid(True, alpha=0.3)
    ticks = list(range(1, N_MINUTES + 1))
    ax1.set_xticks(ticks)
    ax1.set_xlim(0.8, N_MINUTES + 0.2)
    leg1 = ax1.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.9)

    # DMT panel (High above Low in legend)
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('DMT_', ''), marker='o', markersize=4)
        ax2.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    ax2.set_xlabel('Time (minutes)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(ticks)
    ax2.set_xlim(0.8, N_MINUTES + 0.2)
    leg2 = ax2.legend(loc='upper right', frameon=True, fancybox=True, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.9)

    # Optional: add FDR-based shading if raw df provided
    if df_raw is not None:
        try:
            subjects = list(df_raw['subject'].cat.categories) if 'category' in str(df_raw['subject'].dtype) else list(df_raw['subject'].unique())
            n_time = N_MINUTES
            # RS arrays
            H = np.full((len(subjects), n_time), np.nan, dtype=float)
            L = np.full((len(subjects), n_time), np.nan, dtype=float)
            for si, subj in enumerate(subjects):
                sdf = df_raw[df_raw['subject'] == subj]
                for minute in range(1, N_MINUTES + 1):
                    row_h = sdf[(sdf['Task'] == 'RS') & (sdf['Dose'] == 'High') & (sdf['minute'] == minute)]['AUC']
                    row_l = sdf[(sdf['Task'] == 'RS') & (sdf['Dose'] == 'Low') & (sdf['minute'] == minute)]['AUC']
                    if len(row_h) == 1:
                        H[si, minute - 1] = float(row_h.iloc[0])
                    if len(row_l) == 1:
                        L[si, minute - 1] = float(row_l.iloc[0])
            x_grid = np.arange(1, N_MINUTES + 1)
            segs = _compute_fdr_significant_segments(H, L, x_grid)
            for x0, x1 in segs:
                ax1.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
            rs_res = _compute_fdr_results(H, L, x_grid)

            # DMT arrays
            H = np.full((len(subjects), n_time), np.nan, dtype=float)
            L = np.full((len(subjects), n_time), np.nan, dtype=float)
            for si, subj in enumerate(subjects):
                sdf = df_raw[df_raw['subject'] == subj]
                for minute in range(1, N_MINUTES + 1):
                    row_h = sdf[(sdf['Task'] == 'DMT') & (sdf['Dose'] == 'High') & (sdf['minute'] == minute)]['AUC']
                    row_l = sdf[(sdf['Task'] == 'DMT') & (sdf['Dose'] == 'Low') & (sdf['minute'] == minute)]['AUC']
                    if len(row_h) == 1:
                        H[si, minute - 1] = float(row_h.iloc[0])
                    if len(row_l) == 1:
                        L[si, minute - 1] = float(row_l.iloc[0])
            segs = _compute_fdr_significant_segments(H, L, x_grid)
            for x0, x1 in segs:
                ax2.axvspan(x0, x1, color='0.85', alpha=0.35, zorder=0)
            dmt_res = _compute_fdr_results(H, L, x_grid)

            # Write FDR report
            out_dir = os.path.dirname(os.path.dirname(output_path))
            lines: List[str] = [
                'FDR COMPARISON: SMNA High vs Low (Interaction panels)',
                f"Alpha = {rs_res.get('alpha', 0.05)}",
                ''
            ]
            def _sec(name: str, res: Dict):
                lines.append(f'PANEL {name}:')
                segs2 = res.get('segments', [])
                lines.append(f"  Significant segments (count={len(segs2)}):")
                if len(segs2) == 0:
                    lines.append('    - None')
                for (a, b) in segs2:
                    lines.append(f"    - {a:.1f}–{b:.1f} min")
                p_adj = [v for v in res.get('pvals_adj', []) if isinstance(v, (int, float)) and not np.isnan(v)]
                if p_adj:
                    lines.append(f"  Min p_FDR: {np.nanmin(p_adj):.6f}; Median p_FDR: {np.nanmedian(p_adj):.6f}")
                lines.append('')
            _sec('RS', rs_res)
            _sec('DMT', dmt_res)
            with open(os.path.join(out_dir, 'fdr_segments_smna_interaction.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()


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


def create_combined_summary_plot(out_dir: str) -> Optional[str]:
    """SMNA combined summary using per-minute AUC (first 9 minutes).

    Saves: results/eda/smna/plots/all_subs_smna.png
    """
    # Build per-subject per-minute AUC matrices for RS and DMT (High/Low)
    H_RS, L_RS, H_DMT, L_DMT = [], [], [], []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            # DMT paths
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_smna(p_high)
            d_low = load_cvx_smna(p_low)
            # RS paths
            p_rsh = build_rs_cvx_path(subject, high_session)
            p_rsl = build_rs_cvx_path(subject, low_session)
            r_high = load_cvx_smna(p_rsh)
            r_low = load_cvx_smna(p_rsl)
            if None in (d_high, d_low, r_high, r_low):
                continue
            th, yh = d_high; tl, yl = d_low
            trh, yrh = r_high; trl, yrl = r_low
            # Compute per-minute AUC 1..9
            auc_dmt_h = [compute_auc_minute_window(th, yh, m) for m in range(N_MINUTES)]
            auc_dmt_l = [compute_auc_minute_window(tl, yl, m) for m in range(N_MINUTES)]
            auc_rs_h = [compute_auc_minute_window(trh, yrh, m) for m in range(N_MINUTES)]
            auc_rs_l = [compute_auc_minute_window(trl, yrl, m) for m in range(N_MINUTES)]
            if None in auc_dmt_h or None in auc_dmt_l or None in auc_rs_h or None in auc_rs_l:
                continue
            H_RS.append(np.array(auc_rs_h, dtype=float))
            L_RS.append(np.array(auc_rs_l, dtype=float))
            H_DMT.append(np.array(auc_dmt_h, dtype=float))
            L_DMT.append(np.array(auc_dmt_l, dtype=float))
        except Exception:
            continue
    if not (H_RS and L_RS and H_DMT and L_DMT):
        return None
    H_RS = np.vstack(H_RS); L_RS = np.vstack(L_RS)
    H_DMT = np.vstack(H_DMT); L_DMT = np.vstack(L_DMT)

    def mean_sem(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.nanmean(M, axis=0), np.nanstd(M, axis=0, ddof=1) / np.sqrt(M.shape[0])

    rs_mean_h, rs_sem_h = mean_sem(H_RS)
    rs_mean_l, rs_sem_l = mean_sem(L_RS)
    dmt_mean_h, dmt_sem_h = mean_sem(H_DMT)
    dmt_mean_l, dmt_sem_l = mean_sem(L_DMT)

    x = np.arange(1, N_MINUTES + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    # RS panel
    rs_segs = _compute_fdr_significant_segments(H_RS, L_RS, x)
    for x0, x1 in rs_segs:
        ax1.axvspan(x0 - 0.5, x1 + 0.5, color='0.85', alpha=0.35, zorder=0)
    l1 = ax1.plot(x, rs_mean_h, color=COLOR_RS_HIGH, lw=2.0, label='High')[0]
    ax1.fill_between(x, rs_mean_h - rs_sem_h, rs_mean_h + rs_sem_h, color=COLOR_RS_HIGH, alpha=0.25)
    l2 = ax1.plot(x, rs_mean_l, color=COLOR_RS_LOW, lw=2.0, label='Low')[0]
    ax1.fill_between(x, rs_mean_l - rs_sem_l, rs_mean_l + rs_sem_l, color=COLOR_RS_LOW, alpha=0.25)
    leg1 = ax1.legend([l1, l2], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg1.get_frame().set_facecolor('white'); leg1.get_frame().set_alpha(0.9)
    ax1.set_xlabel('Time (minutes)'); ax1.set_ylabel('SMNA AUC'); ax1.set_title('Resting State (RS)', fontweight='bold')
    ax1.grid(True, which='major', axis='y', alpha=0.25); ax1.grid(False, which='major', axis='x')
    # DMT panel
    dmt_segs = _compute_fdr_significant_segments(H_DMT, L_DMT, x)
    for x0, x1 in dmt_segs:
        ax2.axvspan(x0 - 0.5, x1 + 0.5, color='0.85', alpha=0.35, zorder=0)
    l3 = ax2.plot(x, dmt_mean_h, color=COLOR_DMT_HIGH, lw=2.0, label='High')[0]
    ax2.fill_between(x, dmt_mean_h - dmt_sem_h, dmt_mean_h + dmt_sem_h, color=COLOR_DMT_HIGH, alpha=0.25)
    l4 = ax2.plot(x, dmt_mean_l, color=COLOR_DMT_LOW, lw=2.0, label='Low')[0]
    ax2.fill_between(x, dmt_mean_l - dmt_sem_l, dmt_mean_l + dmt_sem_l, color=COLOR_DMT_LOW, alpha=0.25)
    leg2 = ax2.legend([l3, l4], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg2.get_frame().set_facecolor('white'); leg2.get_frame().set_alpha(0.9)
    ax2.set_xlabel('Time (minutes)'); ax2.set_title('DMT', fontweight='bold')
    ax2.grid(True, which='major', axis='y', alpha=0.25); ax2.grid(False, which='major', axis='x')

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xlim(0.8, N_MINUTES + 0.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_smna.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    try:
        lines: List[str] = [
            'FDR COMPARISON: SMNA AUC High vs Low (All Subjects, RS and DMT)',
            'Alpha = 0.05',
            ''
        ]
        def _sect(name: str, segs: List[Tuple[float, float]]):
            lines.append(f'PANEL {name}:')
            lines.append(f'  Significant segments (count={len(segs)}):')
            if len(segs) == 0:
                lines.append('    - None')
            for (a, b) in segs:
                lines.append(f"    - minute {a:.0f}–{b:.0f}")
            lines.append('')
        _sect('RS', rs_segs)
        _sect('DMT', dmt_segs)
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_smna.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path


def create_dmt_only_20min_plot(out_dir: str) -> Optional[str]:
    """SMNA DMT-only extended plot using per-minute AUC (~19 minutes)."""
    limit_sec = 1150.0
    total_minutes = int(np.floor(limit_sec / 60.0))  # ~19
    H_list, L_list = [] , []
    for subject in SUJETOS_VALIDADOS_EDA:
        try:
            high_session, low_session = determine_sessions(subject)
            p_high, p_low = build_cvx_paths(subject, high_session, low_session)
            d_high = load_cvx_smna(p_high)
            d_low = load_cvx_smna(p_low)
            if None in (d_high, d_low):
                continue
            th, yh = d_high; tl, yl = d_low
            auc_h = [compute_auc_minute_window(th, yh, m) for m in range(total_minutes)]
            auc_l = [compute_auc_minute_window(tl, yl, m) for m in range(total_minutes)]
            if None in auc_h or None in auc_l:
                continue
            H_list.append(np.array(auc_h, dtype=float))
            L_list.append(np.array(auc_l, dtype=float))
        except Exception:
            continue
    if not (H_list and L_list):
        return None
    H = np.vstack(H_list); L = np.vstack(L_list)
    mean_h = np.nanmean(H, axis=0); mean_l = np.nanmean(L, axis=0)
    sem_h = np.nanstd(H, axis=0, ddof=1) / np.sqrt(H.shape[0])
    sem_l = np.nanstd(L, axis=0, ddof=1) / np.sqrt(L.shape[0])

    x = np.arange(1, total_minutes + 1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    segs = _compute_fdr_significant_segments(H, L, x)
    for x0, x1 in segs:
        ax.axvspan(x0 - 0.5, x1 + 0.5, color='0.85', alpha=0.35, zorder=0)
    l1 = ax.plot(x, mean_h, color=COLOR_DMT_HIGH, lw=2.0, label='High')[0]
    ax.fill_between(x, mean_h - sem_h, mean_h + sem_h, color=COLOR_DMT_HIGH, alpha=0.25)
    l2 = ax.plot(x, mean_l, color=COLOR_DMT_LOW, lw=2.0, label='Low')[0]
    ax.fill_between(x, mean_l - sem_l, mean_l + sem_l, color=COLOR_DMT_LOW, alpha=0.25)
    leg = ax.legend([l1, l2], ['High', 'Low'], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
    leg.get_frame().set_facecolor('white'); leg.get_frame().set_alpha(0.9)
    ax.set_xlabel('Time (minutes)'); ax.set_ylabel('SMNA AUC'); ax.set_title('DMT', fontweight='bold')
    ax.grid(True, which='major', axis='y', alpha=0.25); ax.grid(False, which='major', axis='x')
    ax.set_xticks(x); ax.set_xlim(0.8, total_minutes + 0.2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'plots', 'all_subs_dmt_smna.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()

    try:
        lines: List[str] = [
            'FDR COMPARISON: SMNA AUC High vs Low (DMT only)',
            'Alpha = 0.05',
            ''
        ]
        lines.append(f"Significant segments (count={len(segs)}):")
        if len(segs) == 0:
            lines.append('  - None')
        for (a, b) in segs:
            lines.append(f"  - minute {a:.0f}–{b:.0f}")
        with open(os.path.join(out_dir, 'fdr_segments_all_subs_dmt_smna.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    except Exception:
        pass
    return out_path

def create_effect_sizes_table(coef_df: pd.DataFrame, output_path: str) -> None:
    table_data = coef_df[['label', 'beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr', 'significance', 'family']].copy()
    numeric_cols = ['beta', 'se', 'ci_lower', 'ci_upper', 'p_raw', 'p_fdr']
    table_data[numeric_cols] = table_data[numeric_cols].round(4)
    table_data['interpretation'] = table_data.apply(lambda r: (
        f"{'Significant' if r['p_fdr'] < 0.05 else 'Non-significant'} "
        f"{'increase' if r['beta'] > 0 else 'decrease'} in SMNA AUC"), axis=1)
    table_data.to_csv(output_path, index=False)


def create_model_summary_txt(diagnostics: Dict, coef_df: pd.DataFrame, output_path: str) -> None:
    lines: List[str] = [
        "LME MODEL SUMMARY",
        "=" * 60,
        "",
        "Fixed Effects Formula:",
        "AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c",
        "",
        "Random Effects: ~ 1 | subject",
        "",
        "Model Fit Statistics:",
        f"AIC: {diagnostics.get('aic', np.nan):.2f}",
        f"BIC: {diagnostics.get('bic', np.nan):.2f}",
        f"Log-likelihood: {diagnostics.get('loglik', np.nan):.2f}",
        f"N observations: {diagnostics.get('n_obs', 'N/A')}",
        f"N subjects: {diagnostics.get('n_groups', 'N/A')}",
        "",
        "Significant Fixed Effects (p_FDR < 0.05):",
    ]
    significant = coef_df[coef_df['p_fdr'] < 0.05]
    if len(significant) == 0:
        lines.append("• None")
    else:
        for _, row in significant.iterrows():
            lines.append(f"• {row['label']}: β = {row['beta']:.3f} {row['significance']}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def generate_captions_file(output_dir: str, n_subjects: int) -> None:
    captions = [
        "Figure: LME Coefficients (SMNA AUC)\n\n"
        "Point estimates (β) and 95% CIs for fixed effects from the mixed model. "
        "Reference line at zero aids interpretation. Significant effects are visually emphasized.",
        "",
        "Figure: Marginal Means Over Time (RS vs DMT × High vs Low)\n\n"
        "Group-level mean ± 95% CI of SMNA AUC across the first 9 minutes for each condition (RS Low/High, DMT Low/High). "
        "Legends indicate dose levels; shading shows uncertainty.",
        "",
        "Figure: Main Task Effect Over Time\n\n"
        "Mean ± 95% CI for RS and DMT (averaged across dose) across minutes 0–8. "
        "Illustrates overall task separation and temporal trend.",
        "",
        "Figure: Task × Dose Interaction (Panels)\n\n"
        "Left: RS Low vs High; Right: DMT Low vs High. Lines show mean ± 95% CI across minutes 0–8. "
        "Highlights how dose effects differ between tasks.",
    ]
    path = os.path.join(output_dir, 'captions_smna.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def main() -> bool:
    out_dir = os.path.join('results', 'eda', 'smna')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Data
        df = prepare_long_data()
        data_path = os.path.join(out_dir, 'smna_auc_long_data.csv')
        df.to_csv(data_path, index=False)

        # LME
        fitted_model, diagnostics = fit_lme_model(df)

        # Diagnostics plot (no titles)
        plot_model_diagnostics(fitted_model, df, plots_dir)

        # Hypotheses + report
        hypothesis_results = hypothesis_testing_with_fdr(fitted_model)
        report_path = generate_report(fitted_model, diagnostics, hypothesis_results, df, out_dir)

        # Coefficients (parse from report for consistency)
        coefficients = load_lme_results_from_report(report_path)
        coef_df = prepare_coefficient_data(coefficients)

        coef_plot_path = os.path.join(plots_dir, 'lme_coefficient_plot.png')
        create_coefficient_plot(coef_df, coef_plot_path)

        # Effect sizes table
        table_path = os.path.join(plots_dir, 'effect_sizes_table.csv')
        create_effect_sizes_table(coef_df, table_path)

        # Marginal means
        stats_df = compute_empirical_means_and_ci(df)
        mm_all_path = os.path.join(plots_dir, 'marginal_means_all_conditions.png')
        create_marginal_means_plot(stats_df, mm_all_path)
        task_main_path = os.path.join(plots_dir, 'task_main_effect.png')
        create_task_effect_plot(stats_df, task_main_path)
        interaction_path = os.path.join(plots_dir, 'task_dose_interaction.png')
        create_interaction_plot(stats_df, interaction_path, df)

        # Summary statistics
        summary_path = os.path.join(plots_dir, 'summary_statistics.csv')
        overall_means = stats_df.groupby('condition').agg({
            'mean': 'mean',
            'se': lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1)),
            'n': 'mean'
        }).round(4)
        overall_means['ci_lower'] = overall_means['mean'] - 1.96 * overall_means['se']
        overall_means['ci_upper'] = overall_means['mean'] + 1.96 * overall_means['se']
        overall_means.to_csv(summary_path)

        # Text model summary (instead of PNG)
        model_summary_txt = os.path.join(out_dir, 'model_summary.txt')
        create_model_summary_txt(diagnostics, coef_df, model_summary_txt)

        # Combined summary (first 9 min) and DMT-only extended
        create_combined_summary_plot(out_dir)
        create_dmt_only_20min_plot(out_dir)

        # Captions
        generate_captions_file(out_dir, len(df['subject'].unique()))

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)


