"""Assemble the Source Data workbook that backs every figure panel.

Communications Biology asks that the numerical data underlying each figure
panel be provided either in a repository or as a Supplementary Data file.
This script collects the per-participant tables written by the ``run_*``
scripts into one Excel workbook (``results/source_data/Source_Data.xlsx``,
one sheet per figure / panel group) and into a folder of plain CSV files
(``results/source_data/csv/``) for the repository route.

Run after the full pipeline (``python src/run_figures.py``)::

    python src/make_source_data.py

Figure numbers and panel letters follow the manuscript. Fig. 1 is a
schematic and has no source-data sheet.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR.parent))

from config import PROJECT_ROOT  # noqa: E402

RESULTS = Path(PROJECT_ROOT) / 'results'
OUT_DIR = RESULTS / 'source_data'
CSV_DIR = OUT_DIR / 'csv'

DOSE_MAP = {'Alta': 'High', 'Baja': 'Low'}

# (sheet name <= 31 chars, path relative to results/, description)
SHEETS = [
    # Figure 2 -----------------------------------------------------------
    ('Fig2AB_HR_9min', 'ecg/hr/hr_minute_long_data_z.csv',
     'Fig. 2a-b and Supp. Fig. 1a: heart rate, 30-s windows, RS and DMT, 0-9 min, window-level z per participant'),
    ('Fig2CD_SMNA_9min', 'eda/smna/smna_auc_long_data_z.csv',
     'Fig. 2c-d and Supp. Fig. 1b: SMNA AUC, 30-s windows, RS and DMT, 0-9 min, window-level z'),
    ('Fig2EF_RVT_9min', 'resp/rvt/resp_rvt_minute_long_data_z.csv',
     'Fig. 2e-f and Supp. Fig. 1c: respiratory volume per time, 30-s windows, RS and DMT, 0-9 min, window-level z'),
    ('Fig2B_HR_LME', 'ecg/hr/plots/effect_sizes_table.csv',
     'Fig. 2b: LME fixed-effect coefficients, 95% CI and BH-FDR p for heart rate'),
    ('Fig2D_SMNA_LME', 'eda/smna/plots/effect_sizes_table.csv',
     'Fig. 2d: LME fixed-effect coefficients, 95% CI and BH-FDR p for SMNA'),
    ('Fig2F_RVT_LME', 'resp/rvt/plots/effect_sizes_table.csv',
     'Fig. 2f: LME fixed-effect coefficients, 95% CI and BH-FDR p for RVT'),
    ('Fig2_clusters_per_window', 'cluster_permutation/per_window.csv',
     'Fig. 2 shading / Supp. Table: per-window paired statistics entering the cluster permutation'),
    ('Fig2_clusters_summary', 'cluster_permutation/summary.csv',
     'Fig. 2 shading / Supp. Table: cluster masses and exact permutation p-values'),
    # Figure 3 -----------------------------------------------------------
    ('Fig3A_PCA_variance', 'composite/pca_variance_explained.txt',
     'Fig. 3a: explained variance ratio of the three principal components'),
    ('Fig3B_LME', 'composite/lme_analysis_report.txt',
     'Fig. 3b: LME fixed-effect coefficients, 95% CI and BH-FDR p for the composite arousal index'),
    ('Fig3C_PC1_loadings', 'composite/pca_loadings_pc1.csv',
     'Fig. 3c: PC1 loadings of the composite arousal index'),
    ('Fig3BD_composite', 'composite/arousal_index_long.csv',
     'Fig. 3b, 3d and Supp. Fig. 5: composite arousal index and modality z-scores per participant and window, RS and DMT (n = 7)'),
    # Figure 4 -----------------------------------------------------------
    ('Fig4_TET_preprocessed', 'tet/preprocessed/tet_preprocessed.csv',
     'Fig. 4 traces: Temporal Experience Tracing, per participant, session and time bin'),
    ('Fig4_TET_PC_scores', 'tet/pca/pca_scores.csv',
     'Fig. 4 PC1/PC2 traces: TET principal-component scores per participant and bin'),
    ('Fig4_TET_PC_loadings', 'tet/pca/pca_loadings.csv',
     'Fig. 4: TET PCA loadings'),
    ('Fig4_TET_PC_variance', 'tet/pca/pca_variance_explained.csv',
     'Fig. 4: TET PCA variance explained'),
    ('Fig4_TET_LME', 'tet/lme/lme_results.csv',
     'Fig. 4 / Results: LME results for the TET dimensions'),
    ('Fig4_TET_clusters_per_bin', 'cluster_permutation/tet_per_bin.csv',
     'Fig. 4 shading: per-bin paired statistics for the TET cluster permutation'),
    ('Fig4_TET_clusters_summary', 'cluster_permutation/tet_summary.csv',
     'Fig. 4 shading: TET cluster masses and exact permutation p-values'),
    # Figure 5 -----------------------------------------------------------
    ('Fig5_merged_physio_TET', 'coupling/merged_physio_tet_data.csv',
     'Fig. 5 and Supp. Fig. 6: merged physiological and TET data per participant and window'),
    ('Fig5_regression', 'coupling/regression_tet_arousal_index.csv',
     'Fig. 5 / Supp. Fig. 6: regressions of TET dimensions on the composite arousal index'),
    ('Fig5_correlations', 'coupling/correlations_tet_physio.csv',
     'Fig. 5: correlations between TET dimensions and physiological measures'),
    ('Fig5_CCA_loadings', 'coupling/cca_loadings.csv', 'Fig. 5: CCA loadings'),
    ('Fig5D_CCA_scores', 'coupling/cca_scores_dmt_cv1.csv',
     'Fig. 5d: in-sample physiological (U1) and affective (V1) canonical scores during DMT'),
    ('Fig5_CCA_CV_folds', 'coupling/cca_cross_validation_folds.csv',
     'Fig. 5: CCA cross-validation, per fold'),
    ('Fig5_CCA_CV_summary', 'coupling/cca_cross_validation_summary.csv',
     'Fig. 5: CCA cross-validation summary'),
    ('Fig5_CCA_CV_signif', 'coupling/cca_cv_significance.csv',
     'Fig. 5: CCA cross-validation significance'),
    ('Fig5_CCA_perm_p', 'coupling/cca_permutation_pvalues.csv',
     'Fig. 5: CCA permutation p-values'),
    # Supplementary figures ----------------------------------------------
    ('FigS2_HR_20min', 'ecg/hr/hr_extended_dmt_z.csv',
     'Supp. Fig. 2: heart rate, DMT only, 0-20 min, window-level z'),
    ('FigS3_HRV_windows', 'ecg/hrv/hrv_windows_long.csv',
     'Supp. Fig. 3: RMSSD, normalised RMSSD, SDNN and HR per 2-min window (step 30 s); the 11 participants with valid ECG, of whom the 10 passing the variability quality criterion enter Supp. Fig. 3 (S16 excluded)'),
    ('FigS4_composite_20min', 'composite/arousal_index_extended_dmt.csv',
     'Supp. Fig. 4: modality z-scores and composite arousal index (projected on the 9-min PC1), DMT 0-20 min, complete cases'),
    ('Res_intrinsic_ceiling', 'ecg/hrv/intrinsic_ceiling.csv',
     'Results (intrinsic-rate ceiling): peak 60-s HR per participant and dose against the age-predicted intrinsic rate'),
]

NOTES = [
    'Fig. 1: schematic, no numerical source data.',
    'Supp. Fig. 1: participant-level 0-9 min trajectories; same data as sheets Fig2AB_HR_9min, Fig2CD_SMNA_9min and Fig2EF_RVT_9min.',
    'Supp. Fig. 5: same data as sheet Fig3BD_composite.',
    'Supp. Fig. 6: same data as sheets Fig5_merged_physio_TET and Fig5_regression.',
    'Dose labels Alta/Baja in the TET-derived tables were translated to High/Low.',
    'Sheet Res_intrinsic_ceiling: per-participant age, predicted intrinsic rate and margin are withheld '
    '(age is a quasi-identifier in this sample); the two booleans give the comparison reported in Results.',
]


# Columns that would disclose a participant's age (age itself, or quantities
# from which it can be recovered exactly). Replaced by the two booleans that
# the manuscript actually reports.
AGE_DISCLOSING = {'age', 'IHR', 'margin_60s'}


LME_LABELS = {'State[T.DMT]': 'State (DMT vs RS)', 'Dose[T.High]': 'Dose (High vs Low)',
              'State[T.DMT]:window_c': 'State × Time', 'Dose[T.High]:window_c': 'Dose × Time',
              'State[T.DMT]:Dose[T.High]': 'State × Dose'}


def _parse_lme_report(path: Path) -> pd.DataFrame:
    """Read the fixed-effect tests from an lme_analysis_report.txt (same columns as
    the per-modality effect_sizes_table.csv)."""
    text = path.read_text(encoding='utf-8').split('HYPOTHESIS TESTING')[1].split('CONDITIONAL CONTRASTS')[0]
    pat = re.compile(r'FAMILY (\w+):|^  (\S+):\n\s+β =\s+(\S+), SE = (\S+)\n\s+95% CI: \[\s*(\S+),\s+(\S+)\]\n'
                     r'\s+p_raw = (\S+), p_FDR = (\S+) ?(\**)', re.M)
    rows, family = [], None
    for m in pat.finditer(text):
        if m.group(1):
            family = m.group(1).capitalize()
            continue
        term = m.group(2)
        rows.append({'label': LME_LABELS.get(term, term), 'beta': float(m.group(3)),
                     'se': float(m.group(4)), 'ci_lower': float(m.group(5)),
                     'ci_upper': float(m.group(6)), 'p_raw': float(m.group(7)),
                     'p_fdr': float(m.group(8)), 'significance': m.group(9), 'family': family})
    return pd.DataFrame(rows)


def _load(path: Path, sheet: str) -> pd.DataFrame:
    if path.name == 'lme_analysis_report.txt':
        return _parse_lme_report(path)
    if path.suffix == '.txt':
        # 'PCk_explained_variance_ratio = x' lines written by run_composite_arousal_index.py
        rows = [line.split(' = ') for line in path.read_text(encoding='utf-8').splitlines() if ' = ' in line]
        return pd.DataFrame({'component': [k.split('_')[0] for k, _ in rows],
                             'explained_variance_ratio': [float(v) for _, v in rows]})
    df = pd.read_csv(path)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].isin(DOSE_MAP).any():
            df[col] = df[col].replace(DOSE_MAP)
    if sheet == 'Res_intrinsic_ceiling' and 'IHR' in df.columns:
        df['above_own_IHR'] = df['peak_60s'] > df['IHR']
        df['above_own_IHR_upper95'] = df['peak_60s'] > df['IHR'] * 1.15
        df = df.drop(columns=[c for c in AGE_DISCLOSING if c in df.columns])
    return df


def main() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_DIR.exists():
        # Clear the contents rather than the directory itself: on OneDrive-backed
        # paths the directory handle is often held open and rmdir raises WinError 5.
        for old_file in CSV_DIR.iterdir():
            if old_file.is_file():
                old_file.unlink()
            else:
                shutil.rmtree(old_file, ignore_errors=True)
    CSV_DIR.mkdir(exist_ok=True)

    index_rows, missing, frames = [], [], []
    for sheet, rel, desc in SHEETS:
        assert len(sheet) <= 31, sheet
        path = RESULTS / rel
        if not path.exists():
            missing.append(rel)
            continue
        df = _load(path, sheet)
        frames.append((sheet, df))
        df.to_csv(CSV_DIR / f'{sheet}.csv', index=False)
        index_rows.append({'sheet': sheet, 'figure_panel': desc,
                           'pipeline_file': f'results/{rel}',
                           'n_rows': len(df), 'n_cols': df.shape[1]})
    readme = pd.DataFrame(index_rows)
    notes = pd.DataFrame({'sheet': [''] * len(NOTES), 'figure_panel': NOTES})

    xlsx = OUT_DIR / 'Source_Data.xlsx'
    with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
        pd.concat([readme, notes], ignore_index=True).to_excel(
            writer, sheet_name='README', index=False)
        for sheet, df in frames:
            df.to_excel(writer, sheet_name=sheet, index=False)
    readme.to_csv(CSV_DIR / 'README_index.csv', index=False)

    print(f'Wrote {xlsx} ({len(index_rows)} data sheets)')
    print(f'CSV copies in {CSV_DIR}')
    if missing:
        print('MISSING (run the pipeline first):')
        for m in missing:
            print('  ', m)
    return xlsx


if __name__ == '__main__':
    main()
