# -*- coding: utf-8 -*-
"""
DMT Emotions project configuration.

This module centralises all project-wide configuration parameters,
including subject-to-dose mapping, validated subject lists, and signal
processing parameters.

Usage:
    Central configuration module imported by every script in the
    dmt-emotions pipeline. It centralises in a single location:
      - BIDS-style paths to raw data, derivatives, and TET reports.
      - Subject -> dose (High/Low) mapping per session, and per-signal
        validated subject lists (EDA, ECG, RESP).
      - Acquisition and processing parameters for the peripheral
        physiological signals (channels, sampling rate, expected session
        durations, validation tolerances).
      - TET analysis configuration: dimensions, expected session lengths,
        aggregation to 30 s bins, and composite indices (e.g., valence).
      - Utility functions (get_dosis_sujeto, get_nombre_archivo, ...) and
        an internal-consistency checker (validar_configuracion).

    Import as `import config` from the physiological preprocessing and
    TET analysis scripts to ensure that every analysis shares the same
    source of truth for subjects, doses, and parameters.

    Note on language: variable, function and string-value names are kept
    in their original (Spanish) form throughout the codebase so that the
    rest of the pipeline does not need to be rewritten. Only the
    comments and docstrings are in English.
"""

import pandas as pd
import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project base path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'data'))

# Data paths following BIDS conventions
PHYSIOLOGY_DATA = os.path.join(DATA_ROOT, 'original', 'physiology')
DERIVATIVES_DATA = os.path.join(DATA_ROOT, 'derivatives', 'preprocessing')
REPORTS_DATA = os.path.join(DATA_ROOT, 'original', 'reports', 'resampled')

# =============================================================================
# EDA ANALYSIS CONFIGURATION
# =============================================================================

# Toggle which EDA analysis methods to run.
# Enable/disable each method depending on project needs.
EDA_ANALYSIS_CONFIG = {
    'neurokit': True,    # Standard NeuroKit2 analysis (always required)
    'emotiphai': True,   # BioSPPy emotiphai SCR method
    'cvx': True          # BioSPPy CVX decomposition (EDR, SMNA, EDL)
}

# =============================================================================
# SUBJECTS AND DOSE CONFIGURATION
# =============================================================================

# Per-subject, per-session dose mapping.
# Each row is a subject; each column is a session.
DOSIS_RAW = [
    ['Alta', 'Baja'],  # S01
    ['Baja', 'Alta'],  # S02
    ['Baja', 'Alta'],  # S03
    ['Alta', 'Baja'],  # S04
    ['Alta', 'Baja'],  # S05
    ['Baja', 'Alta'],  # S06
    ['Baja', 'Alta'],  # S07
    ['Baja', 'Alta'],  # S08
    ['Alta', 'Baja'],  # S09
    ['Alta', 'Baja'],  # S10
    ['Baja', 'Alta'],  # S11
    ['Baja', 'Alta'],  # S12
    ['Baja', 'Alta'],  # S13
    ['Alta', 'Baja'],  # S15
    ['Baja', 'Alta'],  # S16
    ['Alta', 'Baja'],  # S17
    ['Alta', 'Baja'],  # S18
    ['Baja', 'Alta'],  # S19
    ['Baja', 'Alta']   # S20
]

# Column names and subject IDs
COLUMNAS_DOSIS = ['Dosis_Sesion_1', 'Dosis_Sesion_2']
SUJETOS_INDICES = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                   'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

# Build the dose DataFrame
DOSIS = pd.DataFrame(DOSIS_RAW, columns=COLUMNAS_DOSIS, index=SUJETOS_INDICES)

# Full list of all subjects (S01-S20, except S14) - 19 subjects in total
TODOS_LOS_SUJETOS = SUJETOS_INDICES.copy()  # ['S01', 'S02', ..., 'S13', 'S15', ..., 'S20']

# Subjects with previously validated data (verified subset)
SUJETOS_VALIDOS = ['S04', 'S05', 'S06', 'S07', 'S09', 'S13', 'S16', 'S17', 'S18', 'S19', 'S20']

# Subjects used for quick testing (small subset for sanity checks)
SUJETOS_TEST = ['S04']

# Execution mode configuration
TEST_MODE = False  # True = use SUJETOS_TEST only; False = use PROCESSING_MODE
PROCESSING_MODE = 'ALL'  # 'VALID' = validated subjects only; 'ALL' = every available subject

# Subjects with known problematic EDA recordings (documented for reference)
SUJETOS_EDA_PROBLEMATICA = {
    'S08': ['DMT_2'],  # DMT_2 has flat / dead signal
    'S10': ['DMT_2'],  # DMT_2 has flat / dead signal
    'S11': ['DMT_2'],  # DMT_2 has flat / dead signal
    'S12': ['DMT_2'],  # DMT_2 has flat / dead signal; DMT_1 is OK
    'S15': ['DMT_2']   # DMT_2 has flat / dead signal
}

# =============================================================================
# PER-SIGNAL VALIDATED SUBJECTS (based on validation_log.json + dmt_bad_subjects.json)
# =============================================================================
# Inclusion criterion: subjects with all four files (DMT_1, DMT_2, Reposo_1,
# Reposo_2) classified as 'good' or 'acceptable' for the given signal.


SUJETOS_VALIDADOS_EDA = [
    'S04', 'S05', 'S06', 'S07', 'S09', 'S13', 'S16', 'S17', 'S18', 'S19', 'S20'
] # 11 subjects

SUJETOS_VALIDADOS_ECG= [
    'S04', 'S06', 'S07', 'S08', 'S10', 'S11',
    'S15', 'S16', 'S18', 'S19', 'S20'
] # 11 subjects

SUJETOS_VALIDADOS_RESP = [
    'S04', 'S05', 'S06', 'S07', 'S09', 'S13',
    'S15', 'S16', 'S17', 'S18', 'S19', 'S20'
] # 12 subjects



# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Available experiment types
EXPERIMENTOS = ['DMT_1', 'DMT_2', 'Reposo_1', 'Reposo_2']

# Filename patterns
PATRONES_ARCHIVOS = {
    'DMT_1': '{sujeto}_DMT_Session1_DMT.vhdr',
    'DMT_2': '{sujeto}_DMT_Session2_DMT.vhdr',
    'Reposo_1': '{sujeto}_RS_Session1_EC.vhdr',
    'Reposo_2': '{sujeto}_RS_Session2_EC.vhdr'
}

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Expected session durations (seconds)
DURACIONES_ESPERADAS = {
    'DMT': 20 * 60 + 15,     # 20 minutes 15 seconds
    'Reposo': 10 * 60 + 15   # 10 minutes 15 seconds
}

# NeuroKit parameters
NEUROKIT_PARAMS = {
    'method': 'neurokit',
    'sampling_rate_default': 250  # Hz - default fallback if not derivable from header
}

# Physiological data channels
CANALES = {
    'EDA': 'GSR',
    'ECG': 'ECG',
    'RESP': 'RESP'
}

# Tolerance for session-duration validation (seconds)
TOLERANCIA_DURACION = 0.1

# =============================================================================
# TET CONFIGURATION (TEMPORAL EXPERIENCE TRACING)
# =============================================================================

# TET data path
TET_DATA_PATH = os.path.join(DATA_ROOT, 'tet', 'tet_data.csv')

# TET results directory
TET_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'tet')

# TET dimension columns (15 subjective dimensions).
# Column order matches the original .mat files (see Lewis-Healey et al. 2025
# Supplementary Methods for the full TET battery definition).
# Each .mat file contains a 'dimensions' matrix of shape (n_bins, 15)
# whose columns map to the following dimensions, in order:
TET_DIMENSION_COLUMNS = [
    'pleasantness',        # 1. Subjective intensity of the "pleasant" aspects of the experience
    'unpleasantness',      # 2. Subjective intensity of the "unpleasant" aspects of the experience
    'emotional_intensity', # 3. Emotional intensity, valence-independent
    'elementary_imagery',  # 4. Basic visual sensations (flashes, colours, patterns)
    'complex_imagery',     # 5. Complex visual sensations (vivid scenes, visions)
    'auditory',            # 6. Auditory sensations (external sounds or hallucinatory)
    'interoception',       # 7. Intensity of internal bodily sensations ("body load")
    'bliss',               # 8. Experience of ecstasy or deep peace
    'anxiety',             # 9. Experience of dysphoria or anxiety
    'entity',              # 10. Perceived presence of "autonomous entities"
    'selfhood',            # 11. Alterations in the experience of the self (ego dissolution)
    'disembodiment',       # 12. Experience of NOT identifying with one's own body (disembodiment)
    'salience',            # 13. Subjective sense of profound meaning and importance
    'temporality',         # 14. Alterations in the subjective experience of time
    'general_intensity'    # 15. Overall subjective intensity of the DMT effects
]

# Affective / autonomic subset of dimensions used in the statistical analyses.
# These dimensions capture emotional and bodily aspects of the experience.
TET_AFFECTIVE_COLUMNS = [
    'pleasantness',        # 1. Subjective intensity of the "pleasant" aspects of the experience
    'unpleasantness',      # 2. Subjective intensity of the "unpleasant" aspects of the experience
    'emotional_intensity', # 3. Emotional intensity, valence-independent (arousal proxy)
    'interoception',       # 7. Intensity of internal bodily sensations ("body load")
    'bliss',               # 8. Experience of ecstasy or deep peace
    'anxiety',             # 9. Experience of dysphoria or anxiety
]

# Expected session lengths.
# The .mat files contain data uniformly down-sampled to 0.25 Hz (1 point every 4 s):
#   - DMT: 20 min = 1200 s -> 300 points @ 0.25 Hz
#   - RS:  10 min =  600 s -> 150 points @ 0.25 Hz
#
# The original paper specifies 30 s bins (N=40 for DMT, N=20 for RS), but we
# keep the original 0.25 Hz resolution for analysis. Aggregation to 30 s bins
# is done only when required by specific statistical analyses (e.g., LME).
EXPECTED_SESSION_LENGTHS = {
    'RS': 150,   # 150 points @ 0.25 Hz = 600 s = 10 min
    'DMT': 300   # 300 points @ 0.25 Hz = 1200 s = 20 min
}

# Temporal resolution of the data (sampling rate)
TET_SAMPLING_RATE_HZ = 0.25  # 0.25 Hz = 1 point every 4 seconds
TET_SAMPLING_INTERVAL_SEC = 4  # 4 seconds between points

# Parameters for aggregation into 30 s bins (when needed)
TET_BIN_DURATION_SEC = 30  # Bin duration as in the original paper
TET_AGGREGATION_FACTOR = TET_BIN_DURATION_SEC / TET_SAMPLING_INTERVAL_SEC  # 7.5 points/bin

# Valid range for TET dimension values
TET_VALUE_RANGE = (0, 10)

# Required columns in the TET dataset
TET_REQUIRED_COLUMNS = [
    'subject',
    'session_id',
    'state',
    'dose',
    't_bin',
    't_sec'
] + TET_DIMENSION_COLUMNS

# Definitions of the TET composite indices.
# Each index combines several dimensions into a higher-level construct.
# All formulas operate on z-scored values (within-subject standardisation).
COMPOSITE_INDEX_DEFINITIONS = {
    'valence_index_z': {
        'formula': 'pleasantness_z - unpleasantness_z',
        'components': {
            'positive': ['pleasantness_z'],
            'negative': ['unpleasantness_z']
        },
        'interpretation': (
            'Affective valence index. Positive values indicate a predominance of '
            'pleasant experiences; negative values indicate a predominance of '
            'unpleasant experiences. Typical range: -3 to +3 (z-scores).'
        ),
        'directionality': 'higher = more positive affective valence'
    },
    'valence_pos': {
        'formula': 'pleasantness',
        'components': {
            'positive': ['pleasantness'],
            'negative': []
        },
        'interpretation': (
            'Positive valence. Direct copy of the pleasantness dimension for use '
            'in two-dimensional valence analyses (positive / negative as separate '
            'axes).'
        ),
        'directionality': 'higher = more pleasant'
    },
    'valence_neg': {
        'formula': 'unpleasantness',
        'components': {
            'positive': ['unpleasantness'],
            'negative': []
        },
        'interpretation': (
            'Negative valence. Direct copy of the unpleasantness dimension for '
            'use in two-dimensional valence analyses (positive / negative as '
            'separate axes).'
        ),
        'directionality': 'higher = more unpleasant'
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_dosis_sujeto(sujeto, sesion):
    """
    Get the dose for a given subject and session.

    Args:
        sujeto (str): Subject code (e.g., 'S01')
        sesion (int): Session number (1 or 2)

    Returns:
        str: 'Alta' (high) or 'Baja' (low)
    """
    columna = f'Dosis_Sesion_{sesion}'
    return DOSIS.loc[sujeto, columna]

def get_experimento_por_dosis(sujeto, dosis):
    """
    Get which experiment(s) correspond to a given dose for a given subject.

    Args:
        sujeto (str): Subject code (e.g., 'S01')
        dosis (str): 'Alta' or 'Baja'

    Returns:
        list: List of experiments matching that dose, e.g. ['DMT_1'] or ['DMT_2']
    """
    experimentos = []
    for sesion in [1, 2]:
        if get_dosis_sujeto(sujeto, sesion) == dosis:
            experimentos.append(f'DMT_{sesion}')
    return experimentos

def get_nombre_archivo(experimento, sujeto):
    """
    Build the filename for a given experiment and subject.

    Args:
        experimento (str): Experiment type (e.g., 'DMT_1')
        sujeto (str): Subject code (e.g., 'S01')

    Returns:
        str: Name of the .vhdr file
    """
    patron = PATRONES_ARCHIVOS.get(experimento)
    if patron:
        return patron.format(sujeto=sujeto)
    else:
        raise ValueError(f"Unknown experiment: {experimento}")

def get_duracion_esperada(experimento):
    """
    Return the expected duration for a given experiment type.

    Args:
        experimento (str): Experiment type (e.g., 'DMT_1')

    Returns:
        float: Duration in seconds
    """
    if 'DMT' in experimento:
        return DURACIONES_ESPERADAS['DMT']
    else:  # Reposo
        return DURACIONES_ESPERADAS['Reposo']

def sujeto_tiene_problema_eda(sujeto, experimento):
    """
    Check whether a subject has known EDA-quality issues for a given experiment.

    Args:
        sujeto (str): Subject code (e.g., 'S08')
        experimento (str): Experiment type (e.g., 'DMT_2')

    Returns:
        bool: True if the subject has a documented issue for that experiment
    """
    return experimento in SUJETOS_EDA_PROBLEMATICA.get(sujeto, [])

def aggregate_tet_to_30s_bins(data, method='mean'):
    """
    Aggregate TET data from 4 s resolution into 30 s bins, as in the original paper.

    The .mat files contain data uniformly down-sampled to:
      - DMT: 300 points @ 4 s = 1200 s (20 min)
      - RS:  150 points @ 4 s =  600 s (10 min)

    The original paper specifies 30 s bins:
      - DMT: N=40 bins x 30 s = 1200 s (20 min)
      - RS:  N=20 bins x 30 s =  600 s (10 min)

    This function groups every 7.5 points (30 s / 4 s) into a single bin and
    applies an aggregation function (mean by default).

    Args:
        data (pd.DataFrame): TET DataFrame in long format
        method (str): Aggregation method ('mean' or 'median')

    Returns:
        pd.DataFrame: Aggregated DataFrame with 30 s bins

    Example:
        >>> from tet.data_loader import TETDataLoader
        >>> import config
        >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
        >>> data = loader.load_data()
        >>> data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')
        >>> # DMT: 300 points -> 40 bins
        >>> # RS:  150 points -> 20 bins
    """
    import pandas as pd
    import numpy as np

    data = data.copy()

    # Compute real time in seconds (t_bin is an index, not seconds).
    # Each point represents 4 seconds.
    data['t_sec_real'] = data['t_bin'] * TET_RAW_RESOLUTION_SEC

    # Build the 30 s bins
    data['bin_30s'] = (data['t_sec_real'] // TET_BIN_DURATION_SEC).astype(int)

    # Grouping columns
    group_cols = ['subject', 'session_id', 'state', 'dose', 'bin_30s']

    # Columns to aggregate (the dimension columns)
    agg_cols = TET_DIMENSION_COLUMNS

    # Apply aggregation
    if method == 'mean':
        aggregated = data.groupby(group_cols)[agg_cols].mean().reset_index()
    elif method == 'median':
        aggregated = data.groupby(group_cols)[agg_cols].median().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'mean' or 'median'.")

    # Rename bin_30s -> t_bin and recompute t_sec
    aggregated = aggregated.rename(columns={'bin_30s': 't_bin'})
    aggregated['t_sec'] = aggregated['t_bin'] * TET_BIN_DURATION_SEC

    # Reorder columns
    column_order = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec'] + TET_DIMENSION_COLUMNS
    aggregated = aggregated[column_order]

    return aggregated

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validar_configuracion():
    """Validate that the configuration is internally consistent."""

    # The number of dose rows must match the number of subjects
    assert len(DOSIS_RAW) == len(SUJETOS_INDICES), (
        f"Mismatch: {len(DOSIS_RAW)} dose rows vs {len(SUJETOS_INDICES)} subjects"
    )

    # Every "valid" subject must be in the full subject list
    assert all(s in TODOS_LOS_SUJETOS for s in SUJETOS_VALIDOS), (
        "Some valid subjects are missing from the full subject list"
    )

    # Every dose must be one of the known categories
    for fila in DOSIS_RAW:
        for dosis in fila:
            assert dosis in ['Alta', 'Baja'], f"Invalid dose: {dosis}"

    print("Configuration validated successfully.")

if __name__ == "__main__":
    validar_configuracion()
    print("\nConfiguration summary:")
    print(f"   Total subjects: {len(TODOS_LOS_SUJETOS)}")
    print(f"   Validated subjects: {len(SUJETOS_VALIDOS)}")
    print(f"   Experiments: {len(EXPERIMENTOS)}")
    print(f"   DMT duration: {DURACIONES_ESPERADAS['DMT']/60:.1f} min")
    print(f"   Resting State duration: {DURACIONES_ESPERADAS['Reposo']/60:.1f} min")
