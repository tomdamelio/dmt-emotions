# -*- coding: utf-8 -*-
"""
Análisis LME para AUC de SMNA por minutos: DMT vs RS x Alta vs Baja dosis.

Este script implementa un modelo lineal mixto (LME) para analizar el AUC de SMNA
por ventanas de 1 minuto durante los primeros 10 minutos, considerando:

Diseño: 2x2 within-subjects (Task: RS vs DMT) x (Dose: Low vs High)
Modelo: AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c + (1 | subject)

Donde minute_c = minute - mean(minute) es el tiempo centrado.

El análisis incluye:
- Preparación de datos en formato long
- Ajuste del modelo LME con diagnósticos
- Tres familias de hipótesis con corrección BH-FDR:
  (i) Task: DMT vs RS y Task:minute_c
  (ii) Dose: High vs Low y Dose:minute_c  
  (iii) Interacción Task:Dose y contrastes condicionados

Usage:
  python test/lme_analysis_smna_auc.py
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
import seaborn as sns

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
    from statsmodels.stats.outliers_influence import summary_table
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels not available. LME analysis will be limited.")

try:
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Some diagnostics will be limited.")

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 400,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def determine_sessions(subject: str) -> Tuple[str, str]:
    """Return (high_session, low_session) strings: 'session1' or 'session2'."""
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)  # 'Alta' or 'Baja'
    except Exception:
        dose_s1 = 'Alta'
    if dose_s1 == 'Alta':
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
        print(f"⚠️  Missing CVX file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️  Failed to read CVX file {os.path.basename(csv_path)}: {e}")
        return None

    if 'SMNA' not in df.columns:
        print(f"⚠️  SMNA column not found in {os.path.basename(csv_path)}")
        return None

    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)

    smna = pd.to_numeric(df['SMNA'], errors='coerce').fillna(0.0).to_numpy()
    return t, smna


def compute_auc_minute_window(t: np.ndarray, y: np.ndarray, minute: int) -> Optional[float]:
    """Compute AUC for a specific 1-minute window.
    
    Args:
        t: time array in seconds
        y: signal array  
        minute: minute window (0-9 for first 10 minutes)
        
    Returns:
        AUC for that minute window or None if insufficient data
    """
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
    """Prepare data in long format for LME analysis.
    
    Returns:
        DataFrame with columns: subject, minute, Task, Dose, AUC, minute_c
    """
    print("📊 Preparing long-format data for LME analysis...")
    
    rows = []
    n_processed = 0
    
    for subject in SUJETOS_VALIDADOS_EDA:
        print(f"  Processing {subject}...")
        
        # Determine sessions for high/low dose
        high_session, low_session = determine_sessions(subject)
        
        # Build paths
        dmt_high_path, dmt_low_path = build_cvx_paths(subject, high_session, low_session)
        rs_high_path = build_rs_cvx_path(subject, high_session)
        rs_low_path = build_rs_cvx_path(subject, low_session)
        
        # Load data
        dmt_high = load_cvx_smna(dmt_high_path)
        dmt_low = load_cvx_smna(dmt_low_path)
        rs_high = load_cvx_smna(rs_high_path)
        rs_low = load_cvx_smna(rs_low_path)
        
        if None in (dmt_high, dmt_low, rs_high, rs_low):
            print(f"    ⚠️  Skipping {subject}: missing data files")
            continue
            
        # Extract time and SMNA arrays
        t_dmt_high, smna_dmt_high = dmt_high
        t_dmt_low, smna_dmt_low = dmt_low
        t_rs_high, smna_rs_high = rs_high
        t_rs_low, smna_rs_low = rs_low
        
        # Process each minute window (0-9)
        subject_rows = []
        for minute in range(10):
            # Compute AUC for each condition in this minute
            auc_dmt_high = compute_auc_minute_window(t_dmt_high, smna_dmt_high, minute)
            auc_dmt_low = compute_auc_minute_window(t_dmt_low, smna_dmt_low, minute)
            auc_rs_high = compute_auc_minute_window(t_rs_high, smna_rs_high, minute)
            auc_rs_low = compute_auc_minute_window(t_rs_low, smna_rs_low, minute)
            
            # Only include if all four conditions have valid data for this minute
            if None not in (auc_dmt_high, auc_dmt_low, auc_rs_high, auc_rs_low):
                subject_rows.extend([
                    {'subject': subject, 'minute': minute, 'Task': 'DMT', 'Dose': 'High', 'AUC': auc_dmt_high},
                    {'subject': subject, 'minute': minute, 'Task': 'DMT', 'Dose': 'Low', 'AUC': auc_dmt_low},
                    {'subject': subject, 'minute': minute, 'Task': 'RS', 'Dose': 'High', 'AUC': auc_rs_high},
                    {'subject': subject, 'minute': minute, 'Task': 'RS', 'Dose': 'Low', 'AUC': auc_rs_low},
                ])
        
        if subject_rows:
            rows.extend(subject_rows)
            n_processed += 1
            print(f"    ✅ Added {len(subject_rows)} rows for {subject}")
        else:
            print(f"    ⚠️  No valid minute windows for {subject}")
    
    if not rows:
        raise ValueError("No valid data found for any subject!")
        
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Set categorical variables with proper ordering
    df['Task'] = pd.Categorical(df['Task'], categories=['RS', 'DMT'], ordered=True)
    df['Dose'] = pd.Categorical(df['Dose'], categories=['Low', 'High'], ordered=True)
    df['subject'] = pd.Categorical(df['subject'])
    
    # Create centered minute variable
    mean_minute = df['minute'].mean()
    df['minute_c'] = df['minute'] - mean_minute
    
    print(f"✅ Long-format data prepared:")
    print(f"    {len(df)} total observations")
    print(f"    {n_processed} subjects with valid data")
    print(f"    {len(df['minute'].unique())} minute windows")
    print(f"    minute_c centered at {mean_minute:.2f}")
    
    return df


def fit_lme_model(df: pd.DataFrame) -> Tuple[Optional[object], Dict]:
    """Fit the LME model with specified fixed and random effects.
    
    Model: AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c + (1 | subject)
    
    Returns:
        (fitted_model, diagnostics_dict)
    """
    if not STATSMODELS_AVAILABLE:
        print("❌ Cannot fit LME model: statsmodels not available")
        return None, {}
        
    print("🔧 Fitting LME model...")
    print("   Formula: AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c")
    print("   Random effects: ~ 1 | subject")
    
    # Prepare formula - statsmodels handles interaction notation
    formula = "AUC ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c"
    
    try:
        # Fit model with warnings captured
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            model = mixedlm(formula, df, groups=df["subject"])
            fitted = model.fit()
            
            # Capture any convergence warnings
            convergence_warnings = [str(warning.message) for warning in w]
            
    except Exception as e:
        print(f"❌ Model fitting failed: {e}")
        return None, {'error': str(e)}
    
    print("✅ Model fitted successfully")
    
    # Extract model diagnostics
    diagnostics = {
        'aic': fitted.aic,
        'bic': fitted.bic,
        'loglik': fitted.llf,
        'n_obs': fitted.nobs,
        'n_groups': len(df['subject'].unique()),
        'convergence_warnings': convergence_warnings,
        'random_effects_var': fitted.cov_re,
        'residual_var': fitted.scale,
    }
    
    print(f"   AIC: {diagnostics['aic']:.2f}")
    print(f"   BIC: {diagnostics['bic']:.2f}")
    print(f"   Log-likelihood: {diagnostics['loglik']:.2f}")
    print(f"   N observations: {diagnostics['n_obs']}")
    print(f"   N subjects: {diagnostics['n_groups']}")
    
    if convergence_warnings:
        print(f"   ⚠️  Convergence warnings: {len(convergence_warnings)}")
        for warning in convergence_warnings:
            print(f"      {warning}")
    
    return fitted, diagnostics


def plot_model_diagnostics(fitted_model, df: pd.DataFrame, output_dir: str):
    """Create diagnostic plots for the LME model."""
    if fitted_model is None:
        print("⚠️  Cannot create diagnostics: no fitted model")
        return
        
    print("📊 Creating model diagnostic plots...")
    
    # Get fitted values and residuals
    fitted_vals = fitted_model.fittedvalues
    residuals = fitted_model.resid
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('LME Model Diagnostics', fontsize=14, y=0.95)
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_vals, residuals, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Add lowess smooth
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted_vals, frac=0.3)
        axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=2)
    except:
        pass
    
    # 2. Q-Q plot of residuals
    if SCIPY_AVAILABLE:
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        
        # Add Shapiro-Wilk test result
        try:
            sw_stat, sw_p = shapiro(residuals)
            axes[0, 1].text(0.05, 0.95, f'Shapiro-Wilk p={sw_p:.4f}', 
                           transform=axes[0, 1].transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except:
            pass
    else:
        # Manual Q-Q plot
        sorted_resid = np.sort(residuals)
        n = len(sorted_resid)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
        axes[0, 1].scatter(theoretical_quantiles, sorted_resid, alpha=0.6)
        axes[0, 1].plot(theoretical_quantiles, theoretical_quantiles, 'r--')
        axes[0, 1].set_xlabel('Theoretical Quantiles')
        axes[0, 1].set_ylabel('Sample Quantiles')
        axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # 3. Residuals by subject (random effects check)
    subject_means = df.groupby('subject').apply(lambda x: residuals[x.index].mean())
    axes[1, 0].bar(range(len(subject_means)), subject_means.values, alpha=0.7)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('Mean Residual')
    axes[1, 0].set_title('Mean Residuals by Subject')
    
    # 4. Residuals by minute (time trend check)
    minute_residuals = df.groupby('minute').apply(lambda x: residuals[x.index].mean())
    axes[1, 1].plot(minute_residuals.index, minute_residuals.values, 'o-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Minute')
    axes[1, 1].set_ylabel('Mean Residual')
    axes[1, 1].set_title('Mean Residuals by Minute')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'lme_diagnostics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diagnostic plots saved: {output_path}")


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    p_array = np.array(p_values)
    n = len(p_array)
    
    # Sort p-values and get order
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    # Apply BH correction
    adjusted_p = np.zeros(n)
    for i in range(n-1, -1, -1):  # Work backwards
        if i == n-1:
            adjusted_p[sorted_indices[i]] = sorted_p[i]
        else:
            adjusted_p[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted_p[sorted_indices[i+1]]
            )
    
    return np.minimum(adjusted_p, 1.0).tolist()


def hypothesis_testing_with_fdr(fitted_model, output_dir: str) -> Dict:
    """Perform hypothesis testing with BH-FDR correction within families.
    
    Three families of hypotheses:
    (i) Task: DMT vs RS and Task:minute_c  
    (ii) Dose: High vs Low and Dose:minute_c
    (iii) Interaction: Task:Dose and conditional contrasts
    """
    if fitted_model is None:
        print("⚠️  Cannot perform hypothesis testing: no fitted model")
        return {}
        
    print("🔬 Performing hypothesis testing with FDR correction...")
    
    # Get parameter estimates and p-values
    params = fitted_model.params
    pvalues = fitted_model.pvalues
    conf_int = fitted_model.conf_int()
    stderr = fitted_model.bse
    
    # Print all available parameters for debugging
    print(f"📋 Available parameters: {list(params.index)}")
    
    results = {
        'all_params': params.to_dict(),
        'all_pvalues': pvalues.to_dict(),
        'all_stderr': stderr.to_dict(),
        'conf_int': conf_int.to_dict(),
    }
    
    # Define hypothesis families
    families = {
        'Task': [],
        'Dose': [], 
        'Interaction': []
    }
    
    # Family (i): Task effects
    task_params = ['Task[T.DMT]', 'Task[T.DMT]:minute_c']
    for param in task_params:
        if param in pvalues.index:
            families['Task'].append(param)
    
    # Family (ii): Dose effects  
    dose_params = ['Dose[T.High]', 'Dose[T.High]:minute_c']
    for param in dose_params:
        if param in pvalues.index:
            families['Dose'].append(param)
    
    # Family (iii): Interaction effects
    interaction_params = ['Task[T.DMT]:Dose[T.High]']
    for param in interaction_params:
        if param in pvalues.index:
            families['Interaction'].append(param)
    
    # Apply BH-FDR correction within each family
    fdr_results = {}
    
    for family_name, param_list in families.items():
        if not param_list:
            print(f"   ⚠️  No parameters found for {family_name} family")
            continue
            
        print(f"\n📊 {family_name} family ({len(param_list)} tests):")
        
        # Extract p-values for this family
        family_pvals = [pvalues[param] for param in param_list]
        
        # Apply BH-FDR correction
        fdr_pvals = benjamini_hochberg_correction(family_pvals)
        
        family_results = {}
        for i, param in enumerate(param_list):
            beta = params[param]
            se = stderr[param]
            p_raw = family_pvals[i]
            p_fdr = fdr_pvals[i]
            ci_lower = conf_int.loc[param, 0]
            ci_upper = conf_int.loc[param, 1]
            
            family_results[param] = {
                'beta': beta,
                'se': se,
                'p_raw': p_raw,
                'p_fdr': p_fdr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            print(f"   {param}:")
            print(f"     β = {beta:.4f}, SE = {se:.4f}")
            print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"     p_raw = {p_raw:.4f}, p_FDR = {p_fdr:.4f}")
        
        fdr_results[family_name] = family_results
    
    # Conditional contrasts: High-Low within DMT and within RS
    print(f"\n📊 Conditional contrasts:")
    
    # These require manual computation from the model coefficients
    # High-Low within DMT = Dose[T.High] + Task[T.DMT]:Dose[T.High]
    # High-Low within RS = Dose[T.High]
    
    contrasts = {}
    
    # Get base dose effect (High-Low within RS)
    if 'Dose[T.High]' in params.index:
        dose_effect = params['Dose[T.High]']
        dose_se = stderr['Dose[T.High]']
        dose_p = pvalues['Dose[T.High]']
        
        contrasts['High_Low_within_RS'] = {
            'beta': dose_effect,
            'se': dose_se,
            'p_raw': dose_p,
            'description': 'High - Low within RS'
        }
        
        print(f"   High - Low within RS:")
        print(f"     β = {dose_effect:.4f}, SE = {dose_se:.4f}, p = {dose_p:.4f}")
    
    # High-Low within DMT requires combining main effect + interaction
    if all(param in params.index for param in ['Dose[T.High]', 'Task[T.DMT]:Dose[T.High]']):
        # This is more complex as we need the covariance between parameters
        # For now, report the interaction effect as the difference
        interaction_effect = params['Task[T.DMT]:Dose[T.High]']
        interaction_se = stderr['Task[T.DMT]:Dose[T.High]']
        interaction_p = pvalues['Task[T.DMT]:Dose[T.High]']
        
        contrasts['High_Low_within_DMT_vs_RS'] = {
            'beta': interaction_effect,
            'se': interaction_se,
            'p_raw': interaction_p,
            'description': '(High - Low within DMT) - (High - Low within RS)'
        }
        
        print(f"   (High - Low within DMT) - (High - Low within RS):")
        print(f"     β = {interaction_effect:.4f}, SE = {interaction_se:.4f}, p = {interaction_p:.4f}")
    
    results['fdr_families'] = fdr_results
    results['conditional_contrasts'] = contrasts
    
    return results


def generate_report(fitted_model, diagnostics: Dict, hypothesis_results: Dict, 
                   df: pd.DataFrame, output_dir: str):
    """Generate comprehensive analysis report."""
    print("📄 Generating comprehensive analysis report...")
    
    report_lines = []
    
    # Header
    report_lines.extend([
        "=" * 80,
        "LME ANALYSIS REPORT: SMNA AUC by Minute",
        "=" * 80,
        "",
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: {len(df)} observations from {len(df['subject'].unique())} subjects",
        "",
        "DESIGN:",
        "  Within-subjects 2×2: Task (RS vs DMT) × Dose (Low vs High)",
        "  Time windows: 10 one-minute windows (0-9 minutes)",
        "  Dependent variable: AUC of SMNA signal",
        "",
        "MODEL SPECIFICATION:",
        "  Fixed effects: AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c",
        "  Random effects: ~ 1 | subject",
        "  Where minute_c = minute - mean(minute) [centered time]",
        "",
    ])
    
    # Model fit information
    if fitted_model is not None:
        report_lines.extend([
            "MODEL FIT STATISTICS:",
            f"  AIC: {diagnostics.get('aic', 'N/A'):.2f}",
            f"  BIC: {diagnostics.get('bic', 'N/A'):.2f}",
            f"  Log-likelihood: {diagnostics.get('loglik', 'N/A'):.2f}",
            f"  N observations: {diagnostics.get('n_obs', 'N/A')}",
            f"  N subjects: {diagnostics.get('n_groups', 'N/A')}",
            f"  Random effects variance: {diagnostics.get('random_effects_var', 'N/A')}",
            f"  Residual variance: {diagnostics.get('residual_var', 'N/A'):.6f}",
            "",
        ])
        
        # Convergence warnings
        warnings = diagnostics.get('convergence_warnings', [])
        if warnings:
            report_lines.extend([
                "CONVERGENCE WARNINGS:",
                *[f"  - {w}" for w in warnings],
                "",
            ])
        else:
            report_lines.append("✅ Model converged without warnings\n")
    
    # Hypothesis testing results
    if 'fdr_families' in hypothesis_results:
        report_lines.extend([
            "HYPOTHESIS TESTING RESULTS (with BH-FDR correction):",
            "=" * 60,
            "",
        ])
        
        for family_name, family_results in hypothesis_results['fdr_families'].items():
            report_lines.extend([
                f"FAMILY {family_name.upper()}:",
                "-" * 30,
            ])
            
            for param, results in family_results.items():
                beta = results['beta']
                se = results['se']
                p_raw = results['p_raw']
                p_fdr = results['p_fdr']
                ci_lower = results['ci_lower']
                ci_upper = results['ci_upper']
                
                significance = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else ""
                
                report_lines.extend([
                    f"  {param}:",
                    f"    β = {beta:8.4f}, SE = {se:6.4f}",
                    f"    95% CI: [{ci_lower:8.4f}, {ci_upper:8.4f}]", 
                    f"    p_raw = {p_raw:6.4f}, p_FDR = {p_fdr:6.4f} {significance}",
                    "",
                ])
            
            report_lines.append("")
    
    # Conditional contrasts
    if 'conditional_contrasts' in hypothesis_results:
        report_lines.extend([
            "CONDITIONAL CONTRASTS:",
            "-" * 30,
        ])
        
        for contrast_name, results in hypothesis_results['conditional_contrasts'].items():
            beta = results['beta']
            se = results['se']
            p_raw = results['p_raw']
            desc = results['description']
            
            significance = "***" if p_raw < 0.001 else "**" if p_raw < 0.01 else "*" if p_raw < 0.05 else ""
            
            report_lines.extend([
                f"  {desc}:",
                f"    β = {beta:8.4f}, SE = {se:6.4f}, p = {p_raw:6.4f} {significance}",
                "",
            ])
    
    # Data summary
    report_lines.extend([
        "",
        "DATA SUMMARY:",
        "-" * 30,
    ])
    
    # Summary statistics by condition
    summary_stats = df.groupby(['Task', 'Dose'])['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend([
        "Cell means (AUC by Task × Dose):",
        str(summary_stats),
        "",
    ])
    
    # Time trend
    time_stats = df.groupby('minute')['AUC'].agg(['count', 'mean', 'std']).round(4)
    report_lines.extend([
        "Time trend (AUC by minute):",
        str(time_stats),
        "",
    ])
    
    # Footer
    report_lines.extend([
        "",
        "INTERPRETATION NOTES:",
        "- Positive β coefficients indicate higher AUC",
        "- Task[T.DMT]: DMT vs RS (reference = RS)",
        "- Dose[T.High]: High vs Low dose (reference = Low)",
        "- minute_c: Linear time trend (centered)",
        "- Interactions test differential effects",
        "- FDR correction applied within hypothesis families",
        "",
        "=" * 80,
    ])
    
    # Write report
    report_path = os.path.join(output_dir, 'lme_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Analysis report saved: {report_path}")
    
    return report_path


def main():
    """Main analysis pipeline."""
    print("🚀 Starting LME analysis of SMNA AUC by minute...")
    
    # Create output directory
    output_dir = os.path.join('test', 'eda', 'lme_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Prepare long-format data
        df = prepare_long_data()
        
        # Save the prepared dataset
        data_path = os.path.join(output_dir, 'smna_auc_long_data.csv')
        df.to_csv(data_path, index=False)
        print(f"📁 Long-format data saved: {data_path}")
        
        # Step 2: Fit LME model
        fitted_model, diagnostics = fit_lme_model(df)
        
        # Step 3: Create diagnostic plots
        plot_model_diagnostics(fitted_model, df, output_dir)
        
        # Step 4: Hypothesis testing with FDR correction
        hypothesis_results = hypothesis_testing_with_fdr(fitted_model, output_dir)
        
        # Step 5: Generate comprehensive report
        report_path = generate_report(fitted_model, diagnostics, hypothesis_results, df, output_dir)
        
        print(f"\n🎯 Analysis completed successfully!")
        print(f"📊 Results saved in: {output_dir}")
        print(f"📄 Main report: {os.path.basename(report_path)}")
        
        # Quick summary of key findings
        if fitted_model is not None and 'fdr_families' in hypothesis_results:
            print(f"\n🔍 QUICK SUMMARY:")
            
            # Check for significant effects
            significant_effects = []
            for family_name, family_results in hypothesis_results['fdr_families'].items():
                for param, results in family_results.items():
                    if results['p_fdr'] < 0.05:
                        significant_effects.append(f"{param} (p_FDR = {results['p_fdr']:.4f})")
            
            if significant_effects:
                print(f"   Significant effects (p_FDR < 0.05):")
                for effect in significant_effects:
                    print(f"     - {effect}")
            else:
                print(f"   No significant effects after FDR correction")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1)
