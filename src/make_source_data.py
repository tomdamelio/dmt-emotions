"""Assemble the Source Data workbook that backs every figure panel.

Communications Biology asks that the numerical data underlying each figure
panel be provided either in a repository or as a Supplementary Data file.
This script collects the per-participant tables written by the ``run_*``
scripts into one Excel workbook (``results/source_data/Source_Data.xlsx``,
one sheet per figure / panel group) and into a folder of plain CSV files
(``results/source_data/csv/``) for the repository route.

Run after the full pipeline (``python src/run_figures.py``)::

    python src/make_source_data.py

Figure numbers follow the manuscript. Supplementary Fig. 1 is a hand-drawn
schematic and Supplementary Fig. 2 shows the continuous per-participant
signals themselves, whose source is the Zenodo derivatives deposit; neither
has a tabular source-data sheet.
"""

from __future__ import annotations

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
    ('Fig2A_HR_9min', 'ecg/hr/hr_minute_long_data_z.csv',
     'Fig. 2A: heart rate, 30-s windows, RS and DMT, 0-9 min, window-level z per participant'),
    ('Fig2B_SMNA_9min', 'eda/smna/smna_auc_long_data_z.csv',
     'Fig. 2B: SMNA AUC, 30-s windows, RS and DMT, 0-9 min, window-level z'),
    ('Fig2C_RVT_9min', 'resp/rvt/resp_rvt_minute_long_data_z.csv',
     'Fig. 2C: respiratory volume per time, 30-s windows, RS and DMT, 0-9 min, window-level z'),
    ('Fig2D_HR_20min', 'ecg/hr/hr_extended_dmt_z.csv',
     'Fig. 2D: heart rate, DMT only, 0-20 min, window-level z'),
    ('Fig2E_SMNA_20min', 'eda/smna/smna_extended_dmt_z.csv',
     'Fig. 2E: SMNA AUC, DMT only, 0-20 min, window-level z'),
    ('Fig2F_RVT_20min', 'resp/rvt/resp_rvt_extended_dmt_z.csv',
     'Fig. 2F: RVT, DMT only, 0-20 min, window-level z'),
    ('Fig2_HR_effects', 'ecg/hr/plots/effect_sizes_table.csv',
     'Fig. 2 / Results: per-window effect sizes for heart rate'),
    ('Fig2_SMNA_effects', 'eda/smna/plots/effect_sizes_table.csv',
     'Fig. 2 / Results: per-window effect sizes for SMNA'),
    ('Fig2_RVT_effects', 'resp/rvt/plots/effect_sizes_table.csv',
     'Fig. 2 / Results: per-window effect sizes for RVT'),
    ('Fig2_clusters_per_window', 'cluster_permutation/per_window.csv',
     'Fig. 2 shading / Supp. Table: per-window paired statistics entering the cluster permutation'),
    ('Fig2_clusters_summary', 'cluster_permutation/summary.csv',
     'Fig. 2 shading / Supp. Table: cluster masses and exact permutation p-values'),
    # Figure 3 -----------------------------------------------------------
    ('Fig3A_PC1_loadings', 'composite/pca_loadings_pc1.csv',
     'Fig. 3A: PC1 loadings of the composite arousal index'),
    ('Fig3CD_composite', 'composite/arousal_index_long.csv',
     'Fig. 3C-D: composite arousal index per participant and window, RS and DMT (n = 7)'),
    ('Fig3_composite_extended', 'composite/merged_extended_dmt_complete_cases.csv',
     'Fig. 3D extended view: modality z-scores and composite, DMT 0-20 min, complete cases'),
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
     'Fig. 5 and Supp. Fig. 7: merged physiological and TET data per participant and window'),
    ('Fig5_regression', 'coupling/regression_tet_arousal_index.csv',
     'Fig. 5 / Supp. Fig. 7: regressions of TET dimensions on the composite arousal index'),
    ('Fig5_correlations', 'coupling/correlations_tet_physio.csv',
     'Fig. 5: correlations between TET dimensions and physiological measures'),
    ('Fig5_CCA_loadings', 'coupling/cca_loadings.csv', 'Fig. 5: CCA loadings'),
    ('Fig5_CCA_CV_folds', 'coupling/cca_cross_validation_folds.csv',
     'Fig. 5: CCA cross-validation, per fold'),
    ('Fig5_CCA_CV_summary', 'coupling/cca_cross_validation_summary.csv',
     'Fig. 5: CCA cross-validation summary'),
    ('Fig5_CCA_CV_signif', 'coupling/cca_cv_significance.csv',
     'Fig. 5: CCA cross-validation significance'),
    ('Fig5_CCA_perm_p', 'coupling/cca_permutation_pvalues.csv',
     'Fig. 5: CCA permutation p-values'),
    # Supplementary figures ----------------------------------------------
    ('FigS3_HR_scales', 'ecg/hr/hr_extended_dmt_all_scales.csv',
     'Supp. Fig. 3: heart rate, DMT 0-20 min, all window scales'),
    ('FigS3_SMNA_scales', 'eda/smna/smna_extended_dmt_all_scales.csv',
     'Supp. Fig. 3: SMNA, DMT 0-20 min, all window scales'),
    ('FigS3_RVT_scales', 'resp/rvt/resp_rvt_extended_dmt_all_scales.csv',
     'Supp. Fig. 3: RVT, DMT 0-20 min, all window scales'),
    ('FigS6_composite_by_subj', 'composite/merged_z_by_subject.csv',
     'Supp. Fig. 6: per-participant composite index and modality z-scores'),
    ('FigS4_HRV_windows', 'ecg/hrv/hrv_windows_long.csv',
     'Supp. Fig. 4: RMSSD, normalised RMSSD, SDNN and HR per 2-min window (step 30 s); the 11 participants with valid ECG, of whom the 10 passing the variability quality criterion enter Supp. Fig. 4 (S16 excluded)'),
    ('FigS4_intrinsic_ceiling', 'ecg/hrv/intrinsic_ceiling.csv',
     'Results (intrinsic-rate ceiling): peak 60-s HR per participant and dose against the age-predicted intrinsic rate'),
]

NOTES = [
    'Fig. 1 and Supp. Fig. 1: schematics, no numerical source data.',
    'Supp. Fig. 2: continuous per-participant signals; source is the derivatives folder of the Zenodo data deposit.',
    'Supp. Fig. 5: same data as sheet Fig3CD_composite.',
    'Supp. Fig. 7: same data as sheets Fig5_merged_physio_TET and Fig5_regression.',
    'Dose labels Alta/Baja in the TET-derived tables were translated to High/Low.',
    'Sheet FigS4_intrinsic_ceiling: per-participant age, predicted intrinsic rate and margin are withheld '
    '(age is a quasi-identifier in this sample); the two booleans give the comparison reported in Results.',
]


# Columns that would disclose a participant's age (age itself, or quantities
# from which it can be recovered exactly). Replaced by the two booleans that
# the manuscript actually reports.
AGE_DISCLOSING = {'age', 'IHR', 'margin_60s'}


def _load(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].isin(DOSE_MAP).any():
            df[col] = df[col].replace(DOSE_MAP)
    if sheet == 'FigS4_intrinsic_ceiling' and 'IHR' in df.columns:
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
