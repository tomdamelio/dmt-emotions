# -*- coding: utf-8 -*-
"""
Unified HRV (Heart Rate Variability) Analysis: LME modeling and visualization (first 9 minutes).

This script processes ECG-derived HRV from R-peak intervals:
  1) Extract RR intervals from ECG_R_Peaks column
  2) Compute HRV features per minute (RMSSD primary, SDNN/pNN50/LF/HF secondary)
  3) Build long-format per-minute RMSSD dataset
  4) Fit LME with State × Dose and time effects; apply BH-FDR per family
  5) Create coefficient, marginal means, interaction, diagnostics plots
  6) Write model summary as TXT and figure captions
  7) Generate discrete timecourse plot for the first 9 minutes with FDR

Outputs are written to: results/ecg/hrv/

Run:
  python scripts/run_ecg_hrv_analysis.py
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
    from scipy import signal as scisignal
except Exception:
    scistats = None
    scisignal = None


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
    if 'ECG_R_Peaks' not in df.columns:
        return None
    
    # Reconstruct time if missing
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    
    return df


def extract_rr_intervals(df: pd.DataFrame, sampling_rate: float = 250.0) -> Tuple[np.ndarray, np.ndarray]:
    """Extract RR intervals from R-peak binary column.
    
    Returns:
        (rr_times_ms, rr_values_ms): Cumulative time and RR interval duration arrays
    """
    r_peaks = pd.to_numeric(df['ECG_R_Peaks'], errors='coerce').to_numpy()
    r_idx = np.where(r_peaks == 1)[0]
    
    if len(r_idx) < 2:
        return np.array([]), np.array([])
    
    # Compute RR intervals in ms
    rr_samples = np.diff(r_idx)
    rr_ms = rr_samples / sampling_rate * 1000.0
    
    # Filter physiologically plausible RR (300-2000 ms)
    valid_mask = (rr_ms >= 300) & (rr_ms <= 2000)
    rr_ms_valid = rr_ms[valid_mask]
    
    # Compute cumulative time for each RR (in ms)
    # Time of RR_i is at the start of the interval (time of R_i)
    rr_times_samples = r_idx[:-1][valid_mask]  # Start time of each valid RR
    rr_times_ms = rr_times_samples / sampling_rate * 1000.0
    
    return rr_times_ms, rr_ms_valid


def window_rr_by_minute(rr_times_ms: np.ndarray, rr_values_ms: np.ndarray, minute: int) -> np.ndarray:
    """Window RR intervals by minute.
    
    Include RR by timestamp of initial beat (start of interval).
    """
    start_ms = minute * 60.0 * 1000.0
    end_ms = (minute + 1) * 60.0 * 1000.0
    mask = (rr_times_ms >= start_ms) & (rr_times_ms < end_ms)
    return rr_values_ms[mask]


def compute_hrv_features(rr_ms: np.ndarray) -> Dict[str, float]:
    """Compute HRV features from RR intervals.
    
    Primary: RMSSD
    Secondary: SDNN, pNN50, LF, HF, LF/HF, SD1, SD2
    """
    features = {
        'RMSSD': np.nan,
        'SDNN': np.nan,
        'pNN50': np.nan,
        'LF': np.nan,
        'HF': np.nan,
        'LF_HF': np.nan,
        'SD1': np.nan,
        'SD2': np.nan,
    }
    
    if len(rr_ms) < 2:
        return features
    
    # Time-domain features
    # RMSSD: Root Mean Square of Successive Differences
    diff_rr = np.diff(rr_ms)
    features['RMSSD'] = float(np.sqrt(np.mean(diff_rr**2)))
    
    # SDNN: Standard Deviation of NN intervals
    features['SDNN'] = float(np.std(rr_ms, ddof=1))
    
    # pNN50: Percentage of successive differences > 50 ms
    if len(diff_rr) > 0:
        nn50 = np.sum(np.abs(diff_rr) > 50)
        features['pNN50'] = float(100.0 * nn50 / len(diff_rr))
    
    # Frequency-domain features (require at least 10 RR intervals)
    if len(rr_ms) >= 10 and scisignal is not None:
        try:
            # Interpolate RR series to 4 Hz
            time_s = np.cumsum(rr_ms) / 1000.0  # Cumulative time in seconds
            time_s = np.concatenate([[0], time_s])
            rr_interp = np.concatenate([[rr_ms[0]], rr_ms])
            
            fs = 4.0  # Target sampling rate
            time_regular = np.arange(0, time_s[-1], 1.0/fs)
            if len(time_regular) < 10:
                return features
            
            rr_regular = np.interp(time_regular, time_s, rr_interp)
            
            # Detrend
            rr_detrend = scisignal.detrend(rr_regular, type='linear')
            
            # Welch PSD
            nperseg = min(256, len(rr_detrend))
            freqs, psd = scisignal.welch(rr_detrend, fs=fs, nperseg=nperseg)
            
            # LF: 0.04-0.15 Hz, HF: 0.15-0.40 Hz
            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
            hf_mask = (freqs >= 0.15) & (freqs < 0.40)
            
            lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
            hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask]))
            
            features['LF'] = lf_power
            features['HF'] = hf_power
            if hf_power > 0:
                features['LF_HF'] = lf_power / hf_power
        except Exception:
            pass
    
    # Nonlinear features: Poincaré plot (SD1, SD2)
    if len(diff_rr) >= 2:
        try:
            # SD1: Standard deviation perpendicular to line of identity
            features['SD1'] = float(np.std(diff_rr, ddof=1) / np.sqrt(2))
            
            # SD2: Standard deviation along line of identity
            # Approximation: SD2 = sqrt(2*SDNN^2 - SD1^2)
            sdnn = features['SDNN']
            sd1 = features['SD1']
            if not np.isnan(sdnn) and not np.isnan(sd1):
                sd2_sq = 2 * sdnn**2 - sd1**2
                if sd2_sq > 0:
                    features['SD2'] = float(np.sqrt(sd2_sq))
        except Exception:
            pass
    
    return features


def prepare_long_data_hrv() -> Tuple[pd.DataFrame, str]:
    """Build long-format per-minute RMSSD table (first 9 minutes).
    
    Also exports full feature CSVs per subject-condition.
    
    Returns:
        (DataFrame, features_dir): Long RMSSD data and path to features directory
    """
    features_dir = os.path.join('results', 'ecg', 'hrv', 'features_per_minute')
    os.makedirs(features_dir, exist_ok=True)
    
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

        # Extract RR intervals
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        rrt_dmt_h, rrv_dmt_h = extract_rr_intervals(dmt_high, sr)
        rrt_dmt_l, rrv_dmt_l = extract_rr_intervals(dmt_low, sr)
        rrt_rs_h, rrv_rs_h = extract_rr_intervals(rs_high, sr)
        rrt_rs_l, rrv_rs_l = extract_rr_intervals(rs_low, sr)
        
        if any(len(x) < 2 for x in [rrv_dmt_h, rrv_dmt_l, rrv_rs_h, rrv_rs_l]):
            continue
        
        # Per-condition feature storage
        cond_features = {
            'dmt_high': [],
            'dmt_low': [],
            'rs_high': [],
            'rs_low': [],
        }
        
        for minute in range(N_MINUTES):
            # Window RR by minute
            rr_dmt_h = window_rr_by_minute(rrt_dmt_h, rrv_dmt_h, minute)
            rr_dmt_l = window_rr_by_minute(rrt_dmt_l, rrv_dmt_l, minute)
            rr_rs_h = window_rr_by_minute(rrt_rs_h, rrv_rs_h, minute)
            rr_rs_l = window_rr_by_minute(rrt_rs_l, rrv_rs_l, minute)
            
            # Compute features
            feat_dmt_h = compute_hrv_features(rr_dmt_h)
            feat_dmt_l = compute_hrv_features(rr_dmt_l)
            feat_rs_h = compute_hrv_features(rr_rs_h)
            feat_rs_l = compute_hrv_features(rr_rs_l)
            
            # Store for CSV export
            feat_dmt_h['minute'] = minute + 1
            feat_dmt_l['minute'] = minute + 1
            feat_rs_h['minute'] = minute + 1
            feat_rs_l['minute'] = minute + 1
            
            cond_features['dmt_high'].append(feat_dmt_h)
            cond_features['dmt_low'].append(feat_dmt_l)
            cond_features['rs_high'].append(feat_rs_h)
            cond_features['rs_low'].append(feat_rs_l)
            
            # Add to long data if RMSSD valid for all conditions
            rmssd_values = [
                feat_dmt_h['RMSSD'],
                feat_dmt_l['RMSSD'],
                feat_rs_h['RMSSD'],
                feat_rs_l['RMSSD'],
            ]
            if not any(np.isnan(v) for v in rmssd_values):
                minute_label = minute + 1
                rows.extend([
                    {'subject': subject, 'minute': minute_label, 'State': 'DMT', 'Dose': 'High', 'RMSSD': rmssd_values[0]},
                    {'subject': subject, 'minute': minute_label, 'State': 'DMT', 'Dose': 'Low', 'RMSSD': rmssd_values[1]},
                    {'subject': subject, 'minute': minute_label, 'State': 'RS', 'Dose': 'High', 'RMSSD': rmssd_values[2]},
                    {'subject': subject, 'minute': minute_label, 'State': 'RS', 'Dose': 'Low', 'RMSSD': rmssd_values[3]},
                ])
        
        # Export full features per condition
        for cond, feat_list in cond_features.items():
            if feat_list:
                feat_df = pd.DataFrame(feat_list)
                feat_path = os.path.join(features_dir, f"{subject}_{cond}.csv")
                feat_df.to_csv(feat_path, index=False)

    if not rows:
        raise ValueError('No valid HRV data found for any subject!')

    df = pd.DataFrame(rows)
    df['State'] = pd.Categorical(df['State'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['minute_c'] = df['minute'] - df['minute'].mean()
    return df, features_dir


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    if mixedlm is None:
        return None, {'error': 'statsmodels not available'}
    try:
        formula = 'RMSSD ~ State * Dose + minute_c + State:minute_c + Dose:minute_c'
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
    families: Dict[str, List[str]] = {'State': [], 'Dose': [], 'Interaction': []}
    for p in ['State[T.DMT]', 'State[T.DMT]:minute_c']:
        if p in pvalues.index:
            families['State'].append(p)
    for p in ['Dose[T.High]', 'Dose[T.High]:minute_c']:
        if p in pvalues.index:
            families['Dose'].append(p)
    for p in ['State[T.DMT]:Dose[T.High]']:
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
    if all(k in params.index for k in ['Dose[T.High]', 'State[T.DMT]:Dose[T.High]']):
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': float(params['State[T.DMT]:Dose[T.High]']),
            'se': float(stderr['State[T.DMT]:Dose[T.High]']),
            'p_raw': float(pvalues['State[T.DMT]:Dose[T.High]']),
            'description': '(High - Low within DMT) - (High - Low within RS)',
        }
    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    return results


def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, df: pd.DataFrame, output_dir: str) -> str:
    lines: List[str] = [
        '=' * 80,
        'LME ANALYSIS REPORT: HRV (RMSSD) by Minute (first 9 minutes)',
        '=' * 80,
        '',
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        '',
        'DESIGN:',
        '  Within-subjects 2×2: State (RS vs DMT) × Dose (Low vs High)',
        '  Time windows: 9 one-minute windows (0-8 minutes)',
        '  Dependent variable: RMSSD (Root Mean Square of Successive Differences) per minute (ms)',
        f'  Baseline correction: {BASELINE_CORRECTION}',
        '',
        'MODEL SPECIFICATION:',
        '  Fixed effects: RMSSD ~ State*Dose + minute_c + State:minute_c + Dose:minute_c',
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
    cell = df.groupby(['State', 'Dose'], observed=False)['RMSSD'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Cell means (RMSSD by State × Dose):', str(cell), ''])
    trend = df.groupby('minute', observed=False)['RMSSD'].agg(['count', 'mean', 'std']).round(4)
    lines.extend(['Time trend (RMSSD by minute):', str(trend), ''])
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
        if line.startswith('FAMILY STATE:'):
            current_family = 'State'
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
        'State[T.DMT]',
        'Dose[T.High]',
        'State[T.DMT]:minute_c',
        'Dose[T.High]:minute_c',
        'State[T.DMT]:Dose[T.High]'
    ]
    labels = {
        'State[T.DMT]': 'State (DMT vs RS)',
        'Dose[T.High]': 'Dose (High vs Low)',
        'State[T.DMT]:minute_c': 'State × Time',
        'Dose[T.High]:minute_c': 'Dose × Time',
        'State[T.DMT]:Dose[T.High]': 'State × Dose'
    }
    fam_colors = {
        'State': COLOR_DMT_HIGH,
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
    grouped = df.groupby(['minute', 'State', 'Dose'], observed=False)['RMSSD']
    stats_df = grouped.agg(['count', 'mean', 'std', 'sem']).reset_index()
    stats_df.columns = ['minute', 'State', 'Dose', 'n', 'mean', 'std', 'se']
    stats_df['condition'] = stats_df['State'].astype(str) + '_' + stats_df['Dose'].astype(str)
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
    ax.set_ylabel('RMSSD (ms)')
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


def create_state_effect_plot(stats_df: pd.DataFrame, output_path: str) -> None:
    state_means = stats_df.groupby(['minute', 'State'], observed=False).agg({'mean': 'mean', 'n': 'sum'}).reset_index()
    state_se = stats_df.groupby(['minute', 'State'], observed=False)['se'].apply(lambda x: np.sqrt(np.sum(x**2) / max(len(x), 1))).reset_index(name='se')
    state_means = state_means.merge(state_se, on=['minute', 'State'], how='left')
    t_crit = 1.96
    state_means['ci_lower'] = state_means['mean'] - t_crit * state_means['se']
    state_means['ci_upper'] = state_means['mean'] + t_crit * state_means['se']
    fig, ax = plt.subplots(figsize=(10, 6))
    for state, color in [('DMT', COLOR_DMT_HIGH), ('RS', COLOR_RS_HIGH)]:
        state_data = state_means[state_means['State'] == state].sort_values('minute')
        ax.plot(state_data['minute'], state_data['mean'], color=color, linewidth=3, label=f'{state}', marker='o', markersize=6)
        ax.fill_between(state_data['minute'], state_data['ci_lower'], state_data['ci_upper'], color=color, alpha=0.2)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('RMSSD (ms)')
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
    ax1.set_ylabel('RMSSD (ms)')
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
        f"{'increase' if r['beta'] > 0 else 'decrease'} in RMSSD"), axis=1)
    table.to_csv(output_path, index=False)


def create_model_summary_txt(diagnostics: Dict, coef_df: pd.DataFrame, output_path: str) -> None:
    lines: List[str] = [
        'LME MODEL SUMMARY',
        '=' * 60,
        '',
        'Fixed Effects Formula:',
        'RMSSD ~ State*Dose + minute_c + State:minute_c + Dose:minute_c',
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


def create_timecourse_hrv_rmssd(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Create discrete timecourse plot with FDR by minute."""
    # Compute per-subject, per-minute, per-condition means for FDR test
    subj_minute_data = df.pivot_table(
        index=['subject', 'minute'],
        columns=['State', 'Dose'],
        values='RMSSD',
        observed=False
    ).reset_index()
    
    if subj_minute_data.empty:
        return None
    
    # Prepare marginal means for plotting
    stats_df = compute_empirical_means_and_ci(df)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    
    # RS panel
    for condition, color in [('RS_High', COLOR_RS_HIGH), ('RS_Low', COLOR_RS_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax1.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('RS_', ''), marker='o', markersize=6)
        ax1.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    
    # DMT panel
    for condition, color in [('DMT_High', COLOR_DMT_HIGH), ('DMT_Low', COLOR_DMT_LOW)]:
        cond_data = stats_df[stats_df['condition'] == condition].sort_values('minute')
        ax2.plot(cond_data['minute'], cond_data['mean'], color=color, linewidth=2.5, label=condition.replace('DMT_', ''), marker='o', markersize=6)
        ax2.fill_between(cond_data['minute'], cond_data['ci_lower'], cond_data['ci_upper'], color=color, alpha=0.2)
    
    # FDR per minute: paired t-test High vs Low
    rs_pvals = []
    dmt_pvals = []
    for minute in range(1, N_MINUTES + 1):
        minute_data = df[df['minute'] == minute]
        
        # RS High vs Low
        rs_high = minute_data[(minute_data['State'] == 'RS') & (minute_data['Dose'] == 'High')]['RMSSD'].values
        rs_low = minute_data[(minute_data['State'] == 'RS') & (minute_data['Dose'] == 'Low')]['RMSSD'].values
        if len(rs_high) >= 2 and len(rs_low) >= 2:
            try:
                _, p = scistats.ttest_rel(rs_high, rs_low)
                rs_pvals.append((minute, float(p)))
            except Exception:
                rs_pvals.append((minute, np.nan))
        else:
            rs_pvals.append((minute, np.nan))
        
        # DMT High vs Low
        dmt_high = minute_data[(minute_data['State'] == 'DMT') & (minute_data['Dose'] == 'High')]['RMSSD'].values
        dmt_low = minute_data[(minute_data['State'] == 'DMT') & (minute_data['Dose'] == 'Low')]['RMSSD'].values
        if len(dmt_high) >= 2 and len(dmt_low) >= 2:
            try:
                _, p = scistats.ttest_rel(dmt_high, dmt_low)
                dmt_pvals.append((minute, float(p)))
            except Exception:
                dmt_pvals.append((minute, np.nan))
        else:
            dmt_pvals.append((minute, np.nan))
    
    # BH-FDR correction
    rs_valid = [(m, p) for m, p in rs_pvals if not np.isnan(p)]
    dmt_valid = [(m, p) for m, p in dmt_pvals if not np.isnan(p)]
    
    rs_sig_minutes = []
    dmt_sig_minutes = []
    
    if rs_valid:
        rs_p_only = [p for _, p in rs_valid]
        rs_adj = benjamini_hochberg_correction(rs_p_only)
        rs_sig_minutes = [m for (m, _), padj in zip(rs_valid, rs_adj) if padj < 0.05]
    
    if dmt_valid:
        dmt_p_only = [p for _, p in dmt_valid]
        dmt_adj = benjamini_hochberg_correction(dmt_p_only)
        dmt_sig_minutes = [m for (m, _), padj in zip(dmt_valid, dmt_adj) if padj < 0.05]
    
    # Shade significant minutes
    for minute in rs_sig_minutes:
        ax1.axvspan(minute - 0.4, minute + 0.4, color='0.85', alpha=0.35, zorder=0)
    
    for minute in dmt_sig_minutes:
        ax2.axvspan(minute - 0.4, minute + 0.4, color='0.85', alpha=0.35, zorder=0)
    
    # Styling
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('RMSSD (ms)')
    ax1.set_title('Resting State (RS)', fontweight='bold')
    ax1.grid(True, which='major', axis='y', alpha=0.25)
    ax1.grid(False, which='major', axis='x')
    ax1.set_xlim(0.8, N_MINUTES + 0.2)
    ax1.set_xticks(range(1, N_MINUTES + 1))
    legend1 = ax1.legend(loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_alpha(0.9)
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_title('DMT', fontweight='bold')
    ax2.grid(True, which='major', axis='y', alpha=0.25)
    ax2.grid(False, which='major', axis='x')
    ax2.set_xlim(0.8, N_MINUTES + 0.2)
    ax2.set_xticks(range(1, N_MINUTES + 1))
    legend2 = ax2.legend(loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD, labelspacing=LEGEND_LABELSPACING, borderaxespad=LEGEND_BORDERAXESPAD)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_alpha(0.9)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plots', 'timecourse_hrv_rmssd.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close()
    
    # Write FDR report
    try:
        report_lines = [
            'FDR COMPARISON: High vs Low by Minute (RMSSD)',
            'Alpha = 0.05',
            '',
            'RS Panel:',
            f'  Significant minutes (count={len(rs_sig_minutes)}):',
        ]
        if rs_sig_minutes:
            report_lines.extend([f'    - Minute {m}' for m in rs_sig_minutes])
        else:
            report_lines.append('    - None')
        
        report_lines.extend([
            '',
            'DMT Panel:',
            f'  Significant minutes (count={len(dmt_sig_minutes)}):',
        ])
        if dmt_sig_minutes:
            report_lines.extend([f'    - Minute {m}' for m in dmt_sig_minutes])
        else:
            report_lines.append('    - None')
        
        with open(os.path.join(output_dir, 'fdr_minutes_all_subs_ecg_hrv.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    except Exception:
        pass
    
    return out_path


def create_stacked_subjects_plot(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    """Create stacked per-subject RMSSD plot (discrete by minute)."""
    subjects = df['subject'].unique()
    n = len(subjects)
    
    if n == 0:
        return None
    
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
    
    from matplotlib.lines import Line2D
    
    for i, subject in enumerate(subjects):
        subj_data = df[df['subject'] == subject]
        
        ax_rs = axes[i, 0]
        ax_dmt = axes[i, 1]
        
        # RS
        rs_high = subj_data[(subj_data['State'] == 'RS') & (subj_data['Dose'] == 'High')].sort_values('minute')
        rs_low = subj_data[(subj_data['State'] == 'RS') & (subj_data['Dose'] == 'Low')].sort_values('minute')
        
        ax_rs.plot(rs_high['minute'], rs_high['RMSSD'], color=c_rs_high, linewidth=1.8, marker='o', markersize=4)
        ax_rs.plot(rs_low['minute'], rs_low['RMSSD'], color=c_rs_low, linewidth=1.8, marker='o', markersize=4)
        
        ax_rs.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.set_ylabel('RMSSD (ms)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_rs.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_rs.set_title('Resting State (RS)', fontweight='bold')
        ax_rs.set_xlim(0.8, N_MINUTES + 0.2)
        ax_rs.set_xticks(range(1, N_MINUTES + 1))
        ax_rs.grid(True, which='major', axis='y', alpha=0.25)
        ax_rs.grid(False, which='major', axis='x')
        
        legend_rs = ax_rs.legend(handles=[
            Line2D([0], [0], color=c_rs_high, lw=1.8, marker='o', label='RS High'),
            Line2D([0], [0], color=c_rs_low, lw=1.8, marker='o', label='RS Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_rs.get_frame().set_facecolor('white')
        legend_rs.get_frame().set_alpha(0.9)
        
        # DMT
        dmt_high = subj_data[(subj_data['State'] == 'DMT') & (subj_data['Dose'] == 'High')].sort_values('minute')
        dmt_low = subj_data[(subj_data['State'] == 'DMT') & (subj_data['Dose'] == 'Low')].sort_values('minute')
        
        ax_dmt.plot(dmt_high['minute'], dmt_high['RMSSD'], color=c_dmt_high, linewidth=1.8, marker='o', markersize=4)
        ax_dmt.plot(dmt_low['minute'], dmt_low['RMSSD'], color=c_dmt_low, linewidth=1.8, marker='o', markersize=4)
        
        ax_dmt.set_xlabel('Time (minutes)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.set_ylabel('RMSSD (ms)', fontsize=STACKED_AXES_LABEL_SIZE)
        ax_dmt.tick_params(axis='both', labelsize=STACKED_TICK_LABEL_SIZE)
        ax_dmt.set_title('DMT', fontweight='bold')
        ax_dmt.set_xlim(0.8, N_MINUTES + 0.2)
        ax_dmt.set_xticks(range(1, N_MINUTES + 1))
        ax_dmt.grid(True, which='major', axis='y', alpha=0.25)
        ax_dmt.grid(False, which='major', axis='x')
        
        legend_dmt = ax_dmt.legend(handles=[
            Line2D([0], [0], color=c_dmt_high, lw=1.8, marker='o', label='DMT High'),
            Line2D([0], [0], color=c_dmt_low, lw=1.8, marker='o', label='DMT Low'),
        ], loc='upper right', frameon=True, fancybox=False, fontsize=LEGEND_FONTSIZE_SMALL, markerscale=LEGEND_MARKERSCALE, borderpad=LEGEND_BORDERPAD)
        legend_dmt.get_frame().set_facecolor('white')
        legend_dmt.get_frame().set_alpha(0.9)
    
    fig.tight_layout(pad=2.0)
    
    # Add subject codes centered between columns
    for i, subject in enumerate(subjects):
        pos_left = axes[i, 0].get_position()
        pos_right = axes[i, 1].get_position()
        y_center = (pos_left.y0 + pos_left.y1) / 2.0
        x_center = (pos_left.x1 + pos_right.x0) / 2.0
        fig.text(
            x_center,
            y_center + 0.02,
            subject,
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=STACKED_SUBJECT_FONTSIZE,
            transform=fig.transFigure,
        )
    
    out_path = os.path.join(output_dir, 'plots', 'stacked_subs_ecg_hrv.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path


def generate_captions_file(output_dir: str) -> None:
    captions = [
        'Figure: LME Coefficients (HRV RMSSD)\n\n'
        'Point estimates (β) and 95% CIs for fixed effects from the mixed model. '
        'Reference line at zero aids interpretation. Significant effects are visually emphasized.',
        '',
        'Figure: Marginal Means Over Time (RS vs DMT × High vs Low)\n\n'
        'Group-level mean ± 95% CI of RMSSD (ms) across the first 9 minutes for each condition (RS Low/High, DMT Low/High). '
        'Legends indicate dose levels; shading shows uncertainty.',
        '',
        'Figure: Main State Effect Over Time\n\n'
        'Mean ± 95% CI for RS and DMT (averaged across dose) across minutes 0–8. '
        'Illustrates overall state separation and temporal trend.',
        '',
        'Figure: State × Dose Interaction (Panels)\n\n'
        'Left: RS Low vs High; Right: DMT Low vs High. Lines show mean ± 95% CI across minutes 0–8. '
        'Highlights how dose effects differ between states.',
        '',
        'Figure: Discrete HRV RMSSD Timecourse (9 min)\n\n'
        'Two panels (RS, DMT) showing mean ± 95% CI at each minute (discrete points with markers). '
        'Gray shading indicates FDR-significant minutes (High vs Low). Time axis: minutes 1–9.',
        '',
        'Figure: Stacked Per-Subject HRV RMSSD (9 min)\n\n'
        'Individual subject RMSSD traces for RS (left) and DMT (right) conditions. '
        'High/Low dose traces shown with discrete markers. Subject codes centered between panels.',
    ]
    with open(os.path.join(output_dir, 'captions_hrv.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(captions))


def main() -> bool:
    out_dir = os.path.join('results', 'ecg', 'hrv')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    try:
        # Data
        print("Preparing long-format HRV data (RMSSD)...")
        df, features_dir = prepare_long_data_hrv()
        df.to_csv(os.path.join(out_dir, 'hrv_rmssd_long_data.csv'), index=False)
        print(f"  ✓ Loaded {len(df['subject'].unique())} subjects with {len(df)} observations")
        print(f"  ✓ Full HRV features exported to: {features_dir}")
        
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
        create_state_effect_plot(stats_df, os.path.join(plots_dir, 'state_main_effect.png'))
        create_interaction_plot(stats_df, os.path.join(plots_dir, 'state_dose_interaction.png'))
        
        # Model summary txt
        create_model_summary_txt(diagnostics, coef_df, os.path.join(out_dir, 'model_summary.txt'))
        
        # Discrete timecourse with FDR
        print("Creating discrete timecourse plot with FDR...")
        create_timecourse_hrv_rmssd(df, out_dir)
        
        # Stacked per-subject
        print("Creating stacked per-subject plot...")
        create_stacked_subjects_plot(df, out_dir)
        
        # Captions
        generate_captions_file(out_dir)
        
        print(f"\n✓ HRV analysis complete! Results in: {out_dir}")
    except Exception as e:
        print(f'HRV analysis failed: {e}')
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == '__main__':
    ok = main()
    if not ok:
        sys.exit(1)

