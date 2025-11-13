#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inspection of LME results
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("LME RESULTS INSPECTION")
print("=" * 70)

# Load LME results
print("\n1. LME RESULTS")
lme = pd.read_csv('results/tet/lme/lme_results.csv')
print(f"   Shape: {lme.shape}")
print(f"   Columns: {list(lme.columns)}")

# Show significant effects
print(f"\n   Significant effects (FDR < 0.05):")
sig = lme[lme['significant'] == True].sort_values('p_fdr')
print(f"   Total: {len(sig)}")

print(f"\n   By effect type:")
for effect in sig['effect'].unique():
    n = len(sig[sig['effect'] == effect])
    print(f"     {effect}: {n} dimensions")

print(f"\n   Top 10 most significant:")
top10 = sig.nsmallest(10, 'p_fdr')[['dimension', 'effect', 'beta', 'p_value', 'p_fdr']]
print(top10.to_string(index=False))

# Load contrasts
print("\n2. DOSE CONTRASTS")
contrasts = pd.read_csv('results/tet/lme/lme_contrasts.csv')
print(f"   Shape: {contrasts.shape}")
print(f"   Columns: {list(contrasts.columns)}")

# Show significant contrasts
print(f"\n   Significant contrasts (FDR < 0.05):")
sig_contrasts = contrasts[contrasts['significant'] == True].sort_values('p_fdr')
print(f"   Total: {len(sig_contrasts)}")

print(f"\n   DMT High vs Low (significant):")
dmt_sig = sig_contrasts[sig_contrasts['contrast'] == 'dmt_high_vs_low']
print(f"   {len(dmt_sig)} dimensions")
if len(dmt_sig) > 0:
    print(dmt_sig[['dimension', 'estimate', 'p_fdr']].to_string(index=False))

print(f"\n   RS High vs Low (significant):")
rs_sig = sig_contrasts[sig_contrasts['contrast'] == 'rs_high_vs_low']
print(f"   {len(rs_sig)} dimensions")
if len(rs_sig) > 0:
    print(rs_sig[['dimension', 'estimate', 'p_fdr']].to_string(index=False))

# Summary statistics
print("\n3. SUMMARY STATISTICS")
print(f"\n   State effect (DMT vs RS):")
state_effect = lme[lme['effect'] == 'state[T.DMT]']
print(f"     Mean beta: {state_effect['beta'].mean():.3f}")
print(f"     Range: [{state_effect['beta'].min():.3f}, {state_effect['beta'].max():.3f}]")
print(f"     Significant: {(state_effect['significant'] == True).sum()}/{len(state_effect)}")

print(f"\n   Dose effect (Alta vs Baja):")
dose_effect = lme[lme['effect'] == 'dose[T.Alta]']
print(f"     Mean beta: {dose_effect['beta'].mean():.3f}")
print(f"     Range: [{dose_effect['beta'].min():.3f}, {dose_effect['beta'].max():.3f}]")
print(f"     Significant: {(dose_effect['significant'] == True).sum()}/{len(dose_effect)}")

print(f"\n   State:Dose interaction:")
interaction = lme[lme['effect'] == 'state[T.DMT]:dose[T.Alta]']
print(f"     Mean beta: {interaction['beta'].mean():.3f}")
print(f"     Range: [{interaction['beta'].min():.3f}, {interaction['beta'].max():.3f}]")
print(f"     Significant: {(interaction['significant'] == True).sum()}/{len(interaction)}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
