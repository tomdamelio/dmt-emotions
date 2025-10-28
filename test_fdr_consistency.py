"""
Test FDR consistency between 9-min and 19-min plots for EDA SCL.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from config import DERIVATIVES_DATA, SUJETOS_VALIDADOS_EDA, get_dosis_sujeto, NEUROKIT_PARAMS

def determine_sessions(subject: str):
    try:
        dose_s1 = get_dosis_sujeto(subject, 1)
    except Exception:
        dose_s1 = 'Alta'
    if str(dose_s1).lower().startswith('alta') or str(dose_s1).lower().startswith('a'):
        return 'session1', 'session2'
    return 'session2', 'session1'

def build_cvx_paths(subject: str, high_session: str, low_session: str):
    base_high = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_high')
    base_low = os.path.join(DERIVATIVES_DATA, 'phys', 'eda', 'dmt_low')
    high_csv = os.path.join(base_high, f"{subject}_dmt_{high_session}_high_cvx_decomposition.csv")
    low_csv = os.path.join(base_low, f"{subject}_dmt_{low_session}_low_cvx_decomposition.csv")
    return high_csv, low_csv

def compute_baseline_scl(t, y):
    mask = (t >= 0.0) & (t < 1.0)
    if np.any(mask):
        baseline = np.nanmean(y[mask])
        if not np.isnan(baseline):
            return float(baseline)
    valid_mask = ~np.isnan(y)
    if np.any(valid_mask):
        return float(y[valid_mask][0])
    return 0.0

def load_cvx_scl(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if 'EDL' not in df.columns:
        return None
    if 'time' in df.columns:
        t = df['time'].to_numpy()
    else:
        sr = NEUROKIT_PARAMS.get('sampling_rate_default', 250)
        t = np.arange(len(df)) / float(sr)
    y = pd.to_numeric(df['EDL'], errors='coerce').to_numpy()
    baseline = compute_baseline_scl(t, y)
    y_corr = y - baseline
    return t, y_corr

def _resample_to_grid(t, y, t_grid):
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

# Load DMT data for 9 minutes
print("Loading DMT data for 9 minutes...")
t_grid_9min = np.arange(0.0, 541.0, 0.5)
high_curves_9 = []
low_curves_9 = []

for subject in SUJETOS_VALIDADOS_EDA:
    try:
        high_session, low_session = determine_sessions(subject)
        p_high, p_low = build_cvx_paths(subject, high_session, low_session)
        d_high = load_cvx_scl(p_high)
        d_low = load_cvx_scl(p_low)
        if None in (d_high, d_low):
            continue
        th, yh = d_high
        tl, yl = d_low
        
        # Trim to 0..540s
        mh = (th >= 0.0) & (th <= 540.0)
        ml = (tl >= 0.0) & (tl <= 540.0)
        th, yh = th[mh], yh[mh]
        tl, yl = tl[ml], yl[ml]
        
        high_curves_9.append(_resample_to_grid(th, yh, t_grid_9min))
        low_curves_9.append(_resample_to_grid(tl, yl, t_grid_9min))
    except Exception:
        continue

H_9 = np.vstack(high_curves_9)
L_9 = np.vstack(low_curves_9)
print(f"9-min data: {H_9.shape[0]} subjects, {H_9.shape[1]} time points")

# Load DMT data for 19 minutes
print("\nLoading DMT data for 19 minutes...")
t_grid_19min = np.arange(0.0, 1150.0, 0.5)
high_curves_19 = []
low_curves_19 = []

for subject in SUJETOS_VALIDADOS_EDA:
    try:
        high_session, low_session = determine_sessions(subject)
        p_high, p_low = build_cvx_paths(subject, high_session, low_session)
        d_high = load_cvx_scl(p_high)
        d_low = load_cvx_scl(p_low)
        if None in (d_high, d_low):
            continue
        th, yh = d_high
        tl, yl = d_low
        
        # Trim to 0..1150s
        mh = (th >= 0.0) & (th < 1150.0)
        ml = (tl >= 0.0) & (tl < 1150.0)
        th, yh = th[mh], yh[mh]
        tl, yl = tl[ml], yl[ml]
        
        high_curves_19.append(_resample_to_grid(th, yh, t_grid_19min))
        low_curves_19.append(_resample_to_grid(tl, yl, t_grid_19min))
    except Exception:
        continue

H_19 = np.vstack(high_curves_19)
L_19 = np.vstack(low_curves_19)
print(f"19-min data: {H_19.shape[0]} subjects, {H_19.shape[1]} time points")

# Extract first 9 minutes from 19-min data
n_points_9min = len(t_grid_9min)
H_19_first9 = H_19[:, :n_points_9min]
L_19_first9 = L_19[:, :n_points_9min]
print(f"\nFirst 9 min from 19-min data: {H_19_first9.shape}")

# Compare the data
print("\n=== DATA COMPARISON ===")
print(f"Are the matrices identical? {np.allclose(H_9, H_19_first9, equal_nan=True) and np.allclose(L_9, L_19_first9, equal_nan=True)}")
print(f"Max difference in High: {np.nanmax(np.abs(H_9 - H_19_first9))}")
print(f"Max difference in Low: {np.nanmax(np.abs(L_9 - L_19_first9))}")

# Compute some sample t-tests
from scipy import stats as scistats

print("\n=== SAMPLE T-TESTS (first 5 time points) ===")
for t_idx in range(5):
    a_9 = H_9[:, t_idx]
    b_9 = L_9[:, t_idx]
    mask_9 = (~np.isnan(a_9)) & (~np.isnan(b_9))
    
    a_19 = H_19_first9[:, t_idx]
    b_19 = L_19_first9[:, t_idx]
    mask_19 = (~np.isnan(a_19)) & (~np.isnan(b_19))
    
    if np.sum(mask_9) >= 2:
        _, p_9 = scistats.ttest_rel(a_9[mask_9], b_9[mask_9])
        _, p_19 = scistats.ttest_rel(a_19[mask_19], b_19[mask_19])
        print(f"Time point {t_idx}: p_9min={p_9:.6f}, p_19min={p_19:.6f}, diff={abs(p_9-p_19):.2e}")
