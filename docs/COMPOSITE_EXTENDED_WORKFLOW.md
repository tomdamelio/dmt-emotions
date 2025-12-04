# Composite Arousal Index: Extended DMT Analysis Workflow

## Overview

This document explains how to generate the extended DMT plot (~19 minutes) for the composite autonomic arousal index.

## Problem

The unimodal scripts (HR, SMNA, RVT) can generate extended DMT plots (~19 minutes) because they load all available data directly from raw CSVs. However, the composite script only had access to the first 9 minutes because it was designed for LME analysis.

## Solution

We've implemented a two-step workflow:

### Step 1: Extract Extended DMT Data

Run the extraction script to save extended DMT data (~19 minutes) for all three modalities:

```bash
python scripts/save_extended_dmt_data.py
```

This script:
- Loads raw physiological data (HR, SMNA, RVT) for all available windows (~38 windows = ~19 minutes)
- Applies subject-level z-scoring (using RS + DMT from both sessions as baseline)
- Computes per-window metrics (mean HR, SMNA AUC, mean RVT)
- Saves three CSV files:
  - `results/composite/extended_dmt_hr_z.csv`
  - `results/composite/extended_dmt_smna_z.csv`
  - `results/composite/extended_dmt_rvt_z.csv`

### Step 2: Generate Composite Analysis

Run the composite analysis script:

```bash
python pipelines/run_composite_arousal_index.py
```

This script:
- Performs LME analysis on first 9 minutes (18 windows)
- Generates standard plots (coefficients, RS+DMT summary, stacked subjects)
- **NEW**: Checks for pre-saved extended DMT data
  - If found: Uses it to generate extended DMT plot (~19 minutes)
  - If not found: Falls back to loading all available data (slower)

## Complete Workflow

To generate all physiological figures in one go:

```bash
python scripts/generate_phys_figures.py
```

This master script runs:
1. `pipelines/run_resp_rvt_analysis.py` - RVT analysis
2. `pipelines/run_ecg_hr_analysis.py` - HR analysis
3. `pipelines/run_eda_smna_analysis.py` - SMNA analysis
4. `scripts/save_extended_dmt_data.py` - Extract extended DMT data
5. `pipelines/run_composite_arousal_index.py` - Composite analysis

## Output Files

### Extended DMT Data (Intermediate)
- `results/composite/extended_dmt_hr_z.csv`
- `results/composite/extended_dmt_smna_z.csv`
- `results/composite/extended_dmt_rvt_z.csv`

### Composite Analysis Outputs
- `results/composite/plots/all_subs_composite.png` - RS+DMT summary (9 min)
- `results/composite/plots/all_subs_dmt_composite.png` - **DMT-only extended (~19 min)**
- `results/composite/plots/stacked_subs_composite.png` - Per-subject traces
- `results/composite/plots/lme_coefficient_plot.png` - LME coefficients
- `results/composite/plots/pca_pc1_loadings.png` - PCA loadings
- `results/composite/plots/pca_scree.png` - Scree plot
- `results/composite/plots/corr_heatmap_RS_vs_DMT.png` - Cross-correlations
- `results/composite/plots/dynamic_autonomic_coherence_window2.png` - Dynamic coherence
- `results/composite/plots/pca_3d_loadings_interactive.html` - 3D PCA visualization

## Technical Details

### Z-Scoring Strategy

All three modalities use **subject-level z-scoring**:
- Baseline: All sessions from subject (RS_high + DMT_high + RS_low + DMT_low)
- Parameters: Single μ and σ computed from all valid data
- Benefit: Consistent normalization across all conditions for each subject

### Data Format

Extended DMT CSVs have the following structure:

```
subject,window,Dose,HR/SMNA_AUC/RVT
S04,1,High,0.234
S04,1,Low,-0.156
S04,2,High,0.412
...
```

Where:
- `window`: 1-based window index (1-38 for ~19 minutes)
- `Dose`: 'High' or 'Low'
- Values are z-scored within subject

### PCA Application

The composite script:
1. Loads extended data for all three modalities
2. Merges on complete cases (subjects with all three signals)
3. Z-scores each signal within subject (again, for consistency)
4. Applies saved PCA loadings from 9-minute analysis
5. Projects onto PC1 to get ArousalIndex
6. Generates extended plot with FDR analysis

## Troubleshooting

### Missing Extended Data

If `all_subs_dmt_composite.png` is not generated:
1. Check if extended DMT CSVs exist in `results/composite/`
2. If missing, run `python scripts/save_extended_dmt_data.py`
3. Re-run composite analysis

### Inconsistent Subjects

The composite analysis only includes subjects in the intersection:
- `SUBJECTS_INTERSECTION = ['S04', 'S06', 'S07', 'S16', 'S18', 'S19', 'S20']`

If a subject is missing from extended data, check:
- Raw data availability in `derivatives/phys/`
- Z-scoring diagnostics (printed during extraction)

### PCA Loadings Not Found

If PCA loadings are missing:
1. Run the 9-minute analysis first: `python pipelines/run_composite_arousal_index.py`
2. This generates `results/composite/pca_loadings_pc1.csv`
3. Re-run extended plot generation

## Maintenance

### Adding New Subjects

1. Update subject lists in `config.py`:
   - `SUJETOS_VALIDADOS_ECG`
   - `SUJETOS_VALIDADOS_EDA`
   - `SUJETOS_VALIDADOS_RESP`
2. Update intersection in `pipelines/run_composite_arousal_index.py`:
   - `SUBJECTS_INTERSECTION`
3. Re-run extraction and analysis

### Changing Time Window

To analyze a different time range:
1. Edit `LIMIT_SEC` in `scripts/save_extended_dmt_data.py`
2. Re-run extraction
3. Re-run composite analysis

## References

- Unimodal analyses: `pipelines/run_*_analysis.py`
- Composite analysis: `pipelines/run_composite_arousal_index.py`
- Data extraction: `scripts/save_extended_dmt_data.py`
- Master script: `scripts/generate_phys_figures.py`
