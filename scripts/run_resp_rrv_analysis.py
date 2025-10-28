# -*- coding: utf-8 -*-
"""
Unified RSP_RRV (Respiratory Rate Variability) Analysis: LME modeling and visualization (first 9 minutes).

This script processes respiration-derived respiratory rate variability (RSP_RRV) using NeuroKit2's rsp_rrv():
  1) Compute RRV metrics per minute from RSP_Clean signal (RMSSD, SDBB, SD1, SD2, etc.)
  2) Build long-format per-minute RRV dataset (0–8 minutes)
  3) Fit LME with Task × Dose and time effects; apply BH-FDR per family
  4) Create coefficient, marginal means, interaction, diagnostics plots
  5) Write model summary as TXT and figure captions
  6) Generate group-level timecourse plot for the first 9 minutes with FDR

Outputs are written to: results/resp/rrv/

Run:
  python scripts/run_resp_rrv_analysis.py
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
    SUJETOS_VALIDADOS_RESP,  
    get_dosis_sujeto,
    NEUROKIT_PARAMS,
)

# NeuroKit2 for RRV computation
try:
    import neurokit2 as nk
except ImportError:
    nk = None

# Statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
except Exception:
    mixedlm = None

try:
    from scipy import stats as scistats
    from scipy.stats import multipletests
except Exception:
    scistats = None
    multipletests = None


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

# RRV metrics to analyze (select key time-domain metrics)
RRV_METRICS = ['RRV_RMSSD', 'RRV_SDBB', 'RRV_SD1', 'RRV_SD2']


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'


def build_resp_paths(subject: str, high_session: str, low_session: str) -> Tuple[str, str]:
    """Build paths to DMT High and Low RESP CSVs."""
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low.csv")
    return high_csv, low_csv


def build_rs_resp_path(subject: str, session: str) -> str:
    """Build path to RS RESP CSV."""
    ses_num = 1 if session == 'session1' else 2
    dose = get_dosis_sujeto(subject, ses_num)  # 'Alta' or 'Baja'
    cond = 'high' if str(dose).lower().startswith('alta') or str(dose).lower().startswith('a') else 'low'
    base = os.path.join(DERIVATIVES_DATA, 'phys', 'resp', f'dmt_{cond}')
    return os.path.join(base, f"{subject}_rs_{session}_{cond}.csv")


def load_resp_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """Load RESP CSV and validate required columns."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    
    # Validate required columns for RRV computation
    required_cols = ['RSP_Clean', 'RSP_Rate', 'RSP_Troughs']
    if not all(col in df.columns for col in required_cols):
        return None
    
    # Reconstruct time if missing
    if 'time' not in df.columns:
        sr = NEUROKIT_PARAMS.get('sampling_rate', 250)
        df['time'] = np.arange(len(df)) / float(sr)
    
    return df


def compute_rrv_per_minute(df: pd.DataFrame, minute: int, sampling_rate: int = 250) -> Optional[Dict[str, float]]:
    """
    Compute RRV metrics for a specific minute window using NeuroKit2's rsp_rrv().
    
    Returns dict with RRV metrics or None if insufficient data.
    """
    if nk is None:
        return None
    
    start_time = minute * 60.0
    end_time = (minute + 1) * 60.0
    mask = (df['time'] >= start_time) & (df['time'] < end_time)
    
    if not np.any(mask):
        return None
    
    df_win = df[mask].copy()
    
    # Extract required signals
    rsp_rate = pd.to_numeric(df_win['RSP_Rate'], errors='coerce').to_numpy()
    troughs_idx = pd.to_numeric(df_win['RSP_Troughs'], errors='coerce').to_numpy()
    
    # Filter valid troughs (marked as 1)
    troughs_samples = np.where(troughs_idx == 1)[0]
    
    # Need at least 2 breath cycles for RRV
    if len(troughs_samples) < 2:
        return None
    
    try:
        # Compute RRV using NeuroKit2
        rrv_df = nk.rsp_rrv(
            rsp_rate=rsp_rate,
            troughs={'RSP_Troughs': troughs_samples},
            sampling_rate=sampling_rate,
            show=False,
            silent=True
        )
        
        # Extract key metrics
        result = {}
        for metric in RRV_METRICS:
            if metric in rrv_df.columns:
                val = rrv_df[metric].iloc[0]
                # Handle inf/-inf values
                if np.isfinite(val):
                    result[metric] = float(val)
                else:
                    result[metric] = np.nan
            else:
                result[metric] = np.nan
        
        return result if any(np.isfinite(v) for v in result.values()) else None
        
    except Exception as e:
        warnings.warn(f"RRV computation failed for minute {minute}: {e}")
        return None


def prepare_long_data_resp_rrv(metric: str = 'RRV_RMSSD') -> pd.DataFrame:
    """
    Build long-format per-minute RRV table (first 9 minutes) for a specific metric.
    
    Args:
        metric: Which RRV metric to extract (default: 'RRV_RMSSD')
    """
    rows: List[Dict] = []
    sampling_rate = NEUROKIT_PARAMS.get('sampling_rate', 250)
    
    for subject in SUJETOS_VALIDADOS_RESP:
        high_session, low_session = determine_sessions(subject)
        dmt_high_path, dmt_low_path = build_resp_paths(subject, high_session, low_session)
        rs_high_path = build_rs_resp_path(subject, high_session)
        rs_low_path = build_rs_resp_path(subject, low_session)

        dmt_high = load_resp_csv(dmt_high_path)
        dmt_low = load_resp_csv(dmt_low_path)
        rs_high = load_resp_csv(rs_high_path)
        rs_low = load_resp_csv(rs_low_path)
        
        if any(x is None for x in (dmt_high, dmt_low, rs_high, rs_low)):
            continue

        for minute in range(N_MINUTES):
            rrv_dmt_h = compute_rrv_per_minute(dmt_high, minute, sampling_rate)
            rrv_dmt_l = compute_rrv_per_minute(dmt_low, minute, sampling_rate)
            rrv_rs_h = compute_rrv_per_minute(rs_high, minute, sampling_rate)
            rrv_rs_l = compute_rrv_per_minute(rs_low, minute, sampling_rate)
            
            # Extract the specific metric
            vals = []
            for rrv_dict in [rrv_dmt_h, rrv_dmt_l, rrv_rs_h, rrv_rs_l]:
                if rrv_dict is not None and metric in rrv_dict:
                    vals.append(rrv_dict[metric])
                else:
                    vals.append(None)
            
            if all(v is not None and np.isfinite(v) for v in vals):
                minute_label = minute + 1  # store minutes as 1..9 instead of 0..8
                rows.extend([
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'High', metric: vals[0]},
                    {'subject': subject, 'minute': minute_label, 'Task': 'DMT', 'Dose': 'Low', metric: vals[1]},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'High', metric: vals[2]},
                    {'subject': subject, 'minute': minute_label, 'Task': 'RS', 'Dose': 'Low', metric: vals[3]},
                ])

    if not rows:
        raise ValueError(f'No valid {metric} data found for any subject!')

    df = pd.DataFrame(rows)
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    df['minute_c'] = df['minute'] - df['minute'].mean()
    return df



def fit_lme_model(df: pd.DataFrame, metric: str) -> Tuple[Optional[object], Dict]:
    if mixedlm is None:
        return None, {'error': 'statsmodels not available'}
    try:
        formula = f'{metric} ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c'
        model = mixedlm(formula, df, groups=df['subject'])  # type: ignore[arg-type]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fitted = model.fit()
            convergence_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        return None, {'error': str(e)}
    
    info = {
        'aic': fitted.aic,
        'bic': fitted.bic,
        'llf': fitted.llf,
        'convergence_warnings': convergence_warnings
    }
    return fitted, info


def apply_fdr_correction(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction."""
    if multipletests is None:
        return np.zeros_like(pvals, dtype=bool), pvals
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


def extract_model_coefficients(fitted) -> pd.DataFrame:
    """Extract coefficients with p-values from fitted LME model."""
    coef_df = pd.DataFrame({
        'Coefficient': fitted.params.index,
        'Estimate': fitted.params.values,
        'SE': fitted.bse.values,
        'z': fitted.tvalues.values,
        'p_value': fitted.pvalues.values
    })
    
    # Apply FDR correction
    reject, pvals_corrected = apply_fdr_correction(coef_df['p_value'].values)
    coef_df['p_value_fdr'] = pvals_corrected
    coef_df['significant_fdr'] = reject
    
    return coef_df


def compute_marginal_means(df: pd.DataFrame, fitted, metric: str) -> pd.DataFrame:
    """Compute marginal means for Task × Dose combinations."""
    conditions = [
        ('RS', 'Low'), ('RS', 'High'),
        ('DMT', 'Low'), ('DMT', 'High')
    ]
    
    results = []
    for task, dose in conditions:
        subset = df[(df['Task'] == task) & (df['Dose'] == dose)]
        if len(subset) > 0:
            mean_val = subset[metric].mean()
            se_val = subset[metric].sem()
            results.append({
                'Task': task,
                'Dose': dose,
                'Mean': mean_val,
                'SE': se_val
            })
    
    return pd.DataFrame(results)


def save_model_summary(fitted, coef_df: pd.DataFrame, marginal_df: pd.DataFrame, 
                       info: Dict, output_dir: str, metric: str):
    """Save model summary to text file."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f'{metric}_model_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"LME Model Summary: {metric}\n")
        f.write(f"=" * 80 + "\n\n")
        
        f.write(str(fitted.summary()) + "\n\n")
        
        f.write(f"Model Fit Statistics:\n")
        f.write(f"  AIC: {info['aic']:.2f}\n")
        f.write(f"  BIC: {info['bic']:.2f}\n")
        f.write(f"  Log-Likelihood: {info['llf']:.2f}\n\n")
        
        if info.get('convergence_warnings'):
            f.write(f"Convergence Warnings:\n")
            for w in info['convergence_warnings']:
                f.write(f"  - {w}\n")
            f.write("\n")
        
        f.write(f"Coefficients with FDR Correction:\n")
        f.write(coef_df.to_string(index=False) + "\n\n")
        
        f.write(f"Marginal Means (Task × Dose):\n")
        f.write(marginal_df.to_string(index=False) + "\n")
    
    print(f"Model summary saved: {summary_path}")


def plot_coefficients(coef_df: pd.DataFrame, output_dir: str, metric: str):
    """Plot model coefficients with FDR-corrected significance."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Exclude intercept
    plot_df = coef_df[coef_df['Coefficient'] != 'Intercept'].copy()
    
    y_pos = np.arange(len(plot_df))
    colors = ['red' if sig else 'gray' for sig in plot_df['significant_fdr']]
    
    ax.barh(y_pos, plot_df['Estimate'], xerr=plot_df['SE'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df['Coefficient'], fontsize=TICK_LABEL_SIZE_SMALL)
    ax.set_xlabel('Coefficient Estimate', fontsize=AXES_LABEL_SIZE)
    ax.set_title(f'{metric}: Model Coefficients (FDR-corrected)', fontsize=AXES_TITLE_SIZE)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_coefficients.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Coefficients plot saved: {plot_path}")


def plot_marginal_means(marginal_df: pd.DataFrame, output_dir: str, metric: str):
    """Plot marginal means for Task × Dose."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x_labels = [f"{row['Task']}\n{row['Dose']}" for _, row in marginal_df.iterrows()]
    x_pos = np.arange(len(x_labels))
    
    colors = [COLOR_RS_LOW, COLOR_RS_HIGH, COLOR_DMT_LOW, COLOR_DMT_HIGH]
    
    ax.bar(x_pos, marginal_df['Mean'], yerr=marginal_df['SE'],
           color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel(metric, fontsize=AXES_LABEL_SIZE)
    ax.set_title(f'{metric}: Marginal Means', fontsize=AXES_TITLE_SIZE)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_marginal_means.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Marginal means plot saved: {plot_path}")


def plot_interaction(df: pd.DataFrame, output_dir: str, metric: str):
    """Plot Task × Dose interaction."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for dose in ['Low', 'High']:
        subset = df[df['Dose'] == dose]
        means = subset.groupby('Task')[metric].mean()
        sems = subset.groupby('Task')[metric].sem()
        
        color = COLOR_DMT_LOW if dose == 'Low' else COLOR_DMT_HIGH
        ax.errorbar(means.index, means.values, yerr=sems.values,
                   marker='o', markersize=10, linewidth=2.5, capsize=5,
                   label=f'Dose: {dose}', color=color)
    
    ax.set_xlabel('Task', fontsize=AXES_LABEL_SIZE)
    ax.set_ylabel(metric, fontsize=AXES_LABEL_SIZE)
    ax.set_title(f'{metric}: Task × Dose Interaction', fontsize=AXES_TITLE_SIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_interaction.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Interaction plot saved: {plot_path}")


def plot_timecourse(df: pd.DataFrame, output_dir: str, metric: str):
    """Plot group-level timecourse for first 9 minutes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    conditions = [
        ('RS', 'High', COLOR_RS_HIGH, 'RS High'),
        ('RS', 'Low', COLOR_RS_LOW, 'RS Low'),
        ('DMT', 'High', COLOR_DMT_HIGH, 'DMT High'),
        ('DMT', 'Low', COLOR_DMT_LOW, 'DMT Low'),
    ]
    
    for task, dose, color, label in conditions:
        subset = df[(df['Task'] == task) & (df['Dose'] == dose)]
        means = subset.groupby('minute')[metric].mean()
        sems = subset.groupby('minute')[metric].sem()
        
        ax.errorbar(means.index, means.values, yerr=sems.values,
                   marker='o', markersize=8, linewidth=2.5, capsize=4,
                   label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Time (minutes)', fontsize=AXES_LABEL_SIZE)
    ax.set_ylabel(metric, fontsize=AXES_LABEL_SIZE)
    ax.set_title(f'{metric}: Timecourse (First 9 Minutes)', fontsize=AXES_TITLE_SIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best', 
             markerscale=LEGEND_MARKERSCALE,
             borderpad=LEGEND_BORDERPAD,
             handlelength=LEGEND_HANDLELENGTH,
             labelspacing=LEGEND_LABELSPACING)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, N_MINUTES + 1))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_timecourse.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Timecourse plot saved: {plot_path}")


def plot_diagnostics(fitted, df: pd.DataFrame, output_dir: str, metric: str):
    """Create diagnostic plots for model validation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Residuals
    residuals = fitted.resid
    fitted_vals = fitted.fittedvalues
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.5, s=30)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0, 0].set_xlabel('Fitted Values', fontsize=AXES_LABEL_SIZE)
    axes[0, 0].set_ylabel('Residuals', fontsize=AXES_LABEL_SIZE)
    axes[0, 0].set_title('Residuals vs Fitted', fontsize=AXES_TITLE_SIZE)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Q-Q Plot
    if scistats is not None:
        scistats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot', fontsize=AXES_TITLE_SIZE)
        axes[0, 1].grid(alpha=0.3)
    
    # 3. Scale-Location
    standardized_resid = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted_vals, np.sqrt(np.abs(standardized_resid)), alpha=0.5, s=30)
    axes[1, 0].set_xlabel('Fitted Values', fontsize=AXES_LABEL_SIZE)
    axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=AXES_LABEL_SIZE)
    axes[1, 0].set_title('Scale-Location', fontsize=AXES_TITLE_SIZE)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Histogram of Residuals
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals', fontsize=AXES_LABEL_SIZE)
    axes[1, 1].set_ylabel('Frequency', fontsize=AXES_LABEL_SIZE)
    axes[1, 1].set_title('Residual Distribution', fontsize=AXES_TITLE_SIZE)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{metric}_diagnostics.png')
    plt.savefig(plot_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Diagnostics plot saved: {plot_path}")


def analyze_metric(metric: str, base_output_dir: str):
    """Run complete analysis pipeline for a single RRV metric."""
    print(f"\n{'='*80}")
    print(f"Analyzing {metric}")
    print(f"{'='*80}\n")
    
    output_dir = os.path.join(base_output_dir, metric.lower())
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare data
    print(f"Preparing long-format {metric} data...")
    df = prepare_long_data_resp_rrv(metric)
    print(f"  Data shape: {df.shape}")
    print(f"  Subjects: {df['subject'].nunique()}")
    
    # Save data
    data_path = os.path.join(output_dir, f'{metric}_data.csv')
    df.to_csv(data_path, index=False)
    print(f"  Data saved: {data_path}")
    
    # 2. Fit LME model
    print(f"\nFitting LME model for {metric}...")
    fitted, info = fit_lme_model(df, metric)
    
    if fitted is None:
        print(f"  ERROR: {info.get('error', 'Unknown error')}")
        return
    
    print(f"  Model fit successful")
    print(f"  AIC: {info['aic']:.2f}, BIC: {info['bic']:.2f}")
    
    # 3. Extract coefficients and marginal means
    coef_df = extract_model_coefficients(fitted)
    marginal_df = compute_marginal_means(df, fitted, metric)
    
    # 4. Save model summary
    save_model_summary(fitted, coef_df, marginal_df, info, output_dir, metric)
    
    # 5. Create plots
    print(f"\nGenerating plots for {metric}...")
    plot_coefficients(coef_df, output_dir, metric)
    plot_marginal_means(marginal_df, output_dir, metric)
    plot_interaction(df, output_dir, metric)
    plot_timecourse(df, output_dir, metric)
    plot_diagnostics(fitted, df, output_dir, metric)
    
    print(f"\n{metric} analysis complete!")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("Respiratory Rate Variability (RRV) Analysis")
    print("="*80)
    
    if nk is None:
        print("ERROR: NeuroKit2 not available. Install with: pip install neurokit2")
        return
    
    if mixedlm is None:
        print("ERROR: statsmodels not available. Install with: pip install statsmodels")
        return
    
    base_output_dir = os.path.join('results', 'resp', 'rrv')
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Analyze each RRV metric
    for metric in RRV_METRICS:
        try:
            analyze_metric(metric, base_output_dir)
        except Exception as e:
            print(f"\n{metric} analysis failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("RRV Analysis Complete!")
    print(f"Results saved to: {base_output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
