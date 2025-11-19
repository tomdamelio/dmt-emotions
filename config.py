# -*- coding: utf-8 -*-
"""
Configuración del proyecto DMT Emotions

Este archivo contiene todos los parámetros de configuración del proyecto,
incluyendo mapeo de dosis por sujeto, listas de sujetos válidos,
y parámetros de procesamiento.
"""

import pandas as pd
import os

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================

# Ruta base del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'data'))

# Rutas de datos siguiendo estándares BIDS
PHYSIOLOGY_DATA = os.path.join(DATA_ROOT, 'original', 'physiology')
DERIVATIVES_DATA = os.path.join(DATA_ROOT, 'derivatives', 'preprocessing')
REPORTS_DATA = os.path.join(DATA_ROOT, 'original', 'reports', 'resampled')

# =============================================================================
# CONFIGURACIÓN DE ANÁLISIS EDA
# =============================================================================

# Configuración de qué análisis de EDA ejecutar
# Activa/desactiva cada método según necesidades del proyecto
EDA_ANALYSIS_CONFIG = {
    'neurokit': True,    # Análisis estándar NeuroKit2 (siempre requerido)
    'emotiphai': True,   # Método emotiphai SCR de BioSPPy
    'cvx': True          # Descomposición CVX de BioSPPy (EDR, SMNA, EDL)
}

# =============================================================================
# CONFIGURACIÓN DE SUJETOS Y DOSIS
# =============================================================================

# Mapeo de dosis por sujeto y sesión
# Cada fila representa un sujeto, cada columna una sesión
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

# Nombres de columnas y sujetos
COLUMNAS_DOSIS = ['Dosis_Sesion_1', 'Dosis_Sesion_2']
SUJETOS_INDICES = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                   'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']

# Crear DataFrame de dosis
DOSIS = pd.DataFrame(DOSIS_RAW, columns=COLUMNAS_DOSIS, index=SUJETOS_INDICES)

# Lista completa de todos los sujetos (S01-S20, excepto S14) - 19 sujetos total
TODOS_LOS_SUJETOS = SUJETOS_INDICES.copy()  # ['S01', 'S02', ..., 'S13', 'S15', ..., 'S20']

# Sujetos con datos válidos disponibles (subconjunto verificado previamente)
SUJETOS_VALIDOS = ['S04', 'S05', 'S06', 'S07', 'S09', 'S13', 'S16', 'S17', 'S18', 'S19', 'S20']

# Sujetos para testing rápido (subconjunto pequeño para pruebas)
SUJETOS_TEST = ['S04']

# Configuración de modo de ejecución
TEST_MODE = False  # True = solo SUJETOS_TEST, False = procesar según PROCESSING_MODE
PROCESSING_MODE = 'ALL'  # 'VALID' = solo sujetos válidos, 'ALL' = todos los sujetos disponibles

# Sujetos con señal de EDA problemática (documentado para referencia)
SUJETOS_EDA_PROBLEMATICA = {
    'S08': ['DMT_2'],  # DMT_2 tiene señal muerta
    'S10': ['DMT_2'],  # DMT_2 tiene señal muerta
    'S11': ['DMT_2'],  # DMT_2 tiene señal muerta
    'S12': ['DMT_2'],  # DMT_2 tiene señal muerta, DMT_1 está bien
    'S15': ['DMT_2']   # DMT_2 tiene señal muerta
}

# =============================================================================
# SUJETOS VALIDADOS POR SEÑAL (basado en validation_log.json + dmt_bad_subjects.json)
# =============================================================================
# Criterio: sujetos con los 4 archivos (DMT_1, DMT_2, Reposo_1, Reposo_2)
# en categoría 'good' o 'acceptable' para cada señal.


SUJETOS_VALIDADOS_EDA = [
    'S04', 'S05', 'S06', 'S07', 'S09', 'S13', 'S16', 'S17', 'S18', 'S19', 'S20'
] # 11 sujetos

SUJETOS_VALIDADOS_ECG= [
    'S04', 'S06', 'S07', 'S08', 'S10', 'S11',
    'S15', 'S16', 'S18', 'S19', 'S20'
] # 11 sujetos 

SUJETOS_VALIDADOS_RESP = [
    'S04', 'S05', 'S06', 'S07', 'S09', 'S13',
    'S15', 'S16', 'S17', 'S18', 'S19', 'S20'
] # 12 sujetos



# =============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# =============================================================================

# Tipos de experimentos disponibles
EXPERIMENTOS = ['DMT_1', 'DMT_2', 'Reposo_1', 'Reposo_2']

# Patrones de nombres de archivos
PATRONES_ARCHIVOS = {
    'DMT_1': '{sujeto}_DMT_Session1_DMT.vhdr',
    'DMT_2': '{sujeto}_DMT_Session2_DMT.vhdr',
    'Reposo_1': '{sujeto}_RS_Session1_EC.vhdr',
    'Reposo_2': '{sujeto}_RS_Session2_EC.vhdr'
}

# =============================================================================
# PARÁMETROS DE PROCESAMIENTO
# =============================================================================

# Duraciones esperadas (en segundos)
DURACIONES_ESPERADAS = {
    'DMT': 20 * 60 + 15,     # 20 minutos 15 segundos
    'Reposo': 10 * 60 + 15   # 10 minutos 15 segundos
}

# Parámetros de NeuroKit
NEUROKIT_PARAMS = {
    'method': 'neurokit',
    'sampling_rate_default': 250  # Hz - valor por defecto si no se puede calcular
}

# Canales de datos fisiológicos
CANALES = {
    'EDA': 'GSR',
    'ECG': 'ECG',
    'RESP': 'RESP'
}

# Tolerancia para validación de duración (en segundos)
TOLERANCIA_DURACION = 0.1

# =============================================================================
# CONFIGURACIÓN DE TET (TEMPORAL EXPERIENCE TRACKING)
# =============================================================================

# Ruta de datos TET
TET_DATA_PATH = os.path.join(DATA_ROOT, 'tet', 'tet_data.csv')

# Directorio de resultados TET
TET_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'tet')

# Columnas de dimensiones TET (15 dimensiones subjetivas)
# Orden definido según archivos .mat originales (ver docs/PIPELINE.md)
# Cada archivo .mat contiene matriz 'dimensions' de shape (n_bins, 15)
# donde las columnas corresponden a las siguientes dimensiones en orden:
TET_DIMENSION_COLUMNS = [
    'pleasantness',        # 1. Intensidad subjetiva de lo "bueno" de la experiencia
    'unpleasantness',      # 2. Intensidad subjetiva de lo "malo" de la experiencia
    'emotional_intensity', # 3. Intensidad emocional independiente de valencia
    'elementary_imagery',  # 4. Sensaciones visuales básicas (destellos, colores, patrones)
    'complex_imagery',     # 5. Sensaciones visuales complejas (escenas vívidas, visiones)
    'auditory',            # 6. Sensaciones auditivas (sonidos externos o alucinatorios)
    'interoception',       # 7. Intensidad de sensaciones corporales internas ("body load")
    'bliss',               # 8. Experiencia de éxtasis o paz profunda
    'anxiety',             # 9. Experiencia de disforia o ansiedad
    'entity',              # 10. Presencia percibida de "entidades autónomas"
    'selfhood',            # 11. Alteraciones en la experiencia del "yo" (disolución del ego)
    'disembodiment',       # 12. Experiencia de NO identificarse con el propio cuerpo (desencarnación)
    'salience',            # 13. Sentido subjetivo de significado profundo e importancia
    'temporality',         # 14. Alteraciones en la experiencia subjetiva del tiempo
    'general_intensity'    # 15. Intensidad general subjetiva de los efectos del DMT
]

# Subconjunto de dimensiones afectivas/autonómicas para análisis estadísticos
# Estas dimensiones capturan aspectos emocionales y corporales de la experiencia
TET_AFFECTIVE_COLUMNS = [
    'pleasantness',        # 1. Intensidad subjetiva de lo "bueno" de la experiencia
    'unpleasantness',      # 2. Intensidad subjetiva de lo "malo" de la experiencia
    'emotional_intensity', # 3. Intensidad emocional independiente de valencia (arousal proxy)
    'interoception',       # 7. Intensidad de sensaciones corporales internas ("body load")
    'bliss',               # 8. Experiencia de éxtasis o paz profunda
    'anxiety',             # 9. Experiencia de disforia o ansiedad
]

# Longitudes esperadas de sesión
# Los archivos .mat contienen datos down-sampled uniformemente a 0.25 Hz (1 punto cada 4s)
# - DMT: 20 min = 1200s → 300 puntos @ 0.25 Hz
# - RS: 10 min = 600s → 150 puntos @ 0.25 Hz
#
# Paper original especifica bins de 30s (N=40 DMT, N=20 RS), pero mantenemos
# la resolución original (0.25 Hz) para análisis. La agregación a 30s se hace
# solo cuando es necesario para análisis estadísticos específicos (e.g., LME).
EXPECTED_SESSION_LENGTHS = {
    'RS': 150,   # 150 puntos @ 0.25 Hz = 600s = 10 min
    'DMT': 300   # 300 puntos @ 0.25 Hz = 1200s = 20 min
}

# Resolución temporal de datos (frecuencia de muestreo)
TET_SAMPLING_RATE_HZ = 0.25  # 0.25 Hz = 1 punto cada 4 segundos
TET_SAMPLING_INTERVAL_SEC = 4  # 4 segundos entre puntos

# Parámetros para agregación a bins de 30s (cuando sea necesario)
TET_BIN_DURATION_SEC = 30  # Duración de bin según paper original
TET_AGGREGATION_FACTOR = TET_BIN_DURATION_SEC / TET_SAMPLING_INTERVAL_SEC  # 7.5 puntos/bin

# Rango válido para valores de dimensiones TET
TET_VALUE_RANGE = (0, 10)

# Columnas requeridas en datos TET
TET_REQUIRED_COLUMNS = [
    'subject',
    'session_id',
    'state',
    'dose',
    't_bin',
    't_sec'
] + TET_DIMENSION_COLUMNS

# Definiciones de índices compuestos TET
# Cada índice combina múltiples dimensiones para capturar constructos de alto nivel
# Todas las fórmulas operan sobre valores z-scored (estandarizados dentro de sujeto)
COMPOSITE_INDEX_DEFINITIONS = {
    'valence_index_z': {
        'formula': 'pleasantness_z - unpleasantness_z',
        'components': {
            'positive': ['pleasantness_z'],
            'negative': ['unpleasantness_z']
        },
        'interpretation': (
            'Índice de valencia afectiva. Valores positivos indican predominio de experiencias '
            'placenteras, valores negativos indican predominio de experiencias displacenteras. '
            'Rango típico: -3 a +3 (z-scores).'
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
            'Valencia positiva. Copia directa de la dimensión pleasantness para análisis '
            'de valencia bidimensional (positiva/negativa como dimensiones separadas).'
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
            'Valencia negativa. Copia directa de la dimensión unpleasantness para análisis '
            'de valencia bidimensional (positiva/negativa como dimensiones separadas).'
        ),
        'directionality': 'higher = more unpleasant'
    }
}

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def get_dosis_sujeto(sujeto, sesion):
    """
    Obtiene la dosis para un sujeto y sesión específicos
    
    Args:
        sujeto (str): Código del sujeto (ej: 'S01')
        sesion (int): Número de sesión (1 o 2)
    
    Returns:
        str: 'Alta' o 'Baja'
    """
    columna = f'Dosis_Sesion_{sesion}'
    return DOSIS.loc[sujeto, columna]

def get_experimento_por_dosis(sujeto, dosis):
    """
    Obtiene qué experimento corresponde a una dosis específica para un sujeto
    
    Args:
        sujeto (str): Código del sujeto (ej: 'S01')
        dosis (str): 'Alta' o 'Baja'
    
    Returns:
        list: Lista de experimentos que corresponden a esa dosis ['DMT_1'] o ['DMT_2']
    """
    experimentos = []
    for sesion in [1, 2]:
        if get_dosis_sujeto(sujeto, sesion) == dosis:
            experimentos.append(f'DMT_{sesion}')
    return experimentos

def get_nombre_archivo(experimento, sujeto):
    """
    Genera el nombre de archivo para un experimento y sujeto específicos
    
    Args:
        experimento (str): Tipo de experimento (ej: 'DMT_1')
        sujeto (str): Código del sujeto (ej: 'S01')
    
    Returns:
        str: Nombre del archivo .vhdr
    """
    patron = PATRONES_ARCHIVOS.get(experimento)
    if patron:
        return patron.format(sujeto=sujeto)
    else:
        raise ValueError(f"Experimento desconocido: {experimento}")

def get_duracion_esperada(experimento):
    """
    Obtiene la duración esperada para un tipo de experimento
    
    Args:
        experimento (str): Tipo de experimento (ej: 'DMT_1')
    
    Returns:
        float: Duración en segundos
    """
    if 'DMT' in experimento:
        return DURACIONES_ESPERADAS['DMT']
    else:  # Reposo
        return DURACIONES_ESPERADAS['Reposo']

def sujeto_tiene_problema_eda(sujeto, experimento):
    """
    Verifica si un sujeto tiene problemas conocidos de EDA en un experimento
    
    Args:
        sujeto (str): Código del sujeto (ej: 'S08')
        experimento (str): Tipo de experimento (ej: 'DMT_2')
    
    Returns:
        bool: True si tiene problemas conocidos
    """
    return experimento in SUJETOS_EDA_PROBLEMATICA.get(sujeto, [])

def aggregate_tet_to_30s_bins(data, method='mean'):
    """
    Agrega datos TET de resolución 4s a bins de 30s según paper original.
    
    Los archivos .mat contienen datos down-sampled uniformemente:
    - DMT: 300 puntos @ 4s = 1200s (20 min)
    - RS: 150 puntos @ 4s = 600s (10 min)
    
    Paper original especifica bins de 30s:
    - DMT: N=40 bins × 30s = 1200s (20 min)
    - RS: N=20 bins × 30s = 600s (10 min)
    
    Esta función agrupa cada 7.5 puntos (30s / 4s) en un bin y aplica
    una función de agregación (por defecto, promedio).
    
    Args:
        data (pd.DataFrame): DataFrame con datos TET cargados (formato largo)
        method (str): Método de agregación ('mean', 'median')
    
    Returns:
        pd.DataFrame: DataFrame agregado con bins de 30s
        
    Example:
        >>> from tet.data_loader import TETDataLoader
        >>> import config
        >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
        >>> data = loader.load_data()
        >>> data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')
        >>> # DMT: 300 puntos → 40 bins
        >>> # RS: 150 puntos → 20 bins
    """
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    
    # Calcular tiempo real en segundos (t_bin es índice, no tiempo)
    # Cada punto representa 4 segundos
    data['t_sec_real'] = data['t_bin'] * TET_RAW_RESOLUTION_SEC
    
    # Crear bins de 30 segundos
    data['bin_30s'] = (data['t_sec_real'] // TET_BIN_DURATION_SEC).astype(int)
    
    # Columnas de agrupación
    group_cols = ['subject', 'session_id', 'state', 'dose', 'bin_30s']
    
    # Columnas a agregar (dimensiones)
    agg_cols = TET_DIMENSION_COLUMNS
    
    # Aplicar agregación
    if method == 'mean':
        aggregated = data.groupby(group_cols)[agg_cols].mean().reset_index()
    elif method == 'median':
        aggregated = data.groupby(group_cols)[agg_cols].median().reset_index()
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'mean' or 'median'.")
    
    # Renombrar bin_30s a t_bin y calcular t_sec
    aggregated = aggregated.rename(columns={'bin_30s': 't_bin'})
    aggregated['t_sec'] = aggregated['t_bin'] * TET_BIN_DURATION_SEC
    
    # Reordenar columnas
    column_order = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec'] + TET_DIMENSION_COLUMNS
    aggregated = aggregated[column_order]
    
    return aggregated

# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# =============================================================================

def validar_configuracion():
    """Valida que la configuración sea consistente"""
    
    # Verificar que el número de filas de dosis coincida con el número de sujetos
    assert len(DOSIS_RAW) == len(SUJETOS_INDICES), (
        f"Mismatch: {len(DOSIS_RAW)} filas de dosis vs {len(SUJETOS_INDICES)} sujetos"
    )
    
    # Verificar que todos los sujetos válidos estén en la lista completa
    assert all(s in TODOS_LOS_SUJETOS for s in SUJETOS_VALIDOS), (
        "Algunos sujetos válidos no están en la lista completa"
    )
    
    # Verificar que todas las dosis sean válidas
    for fila in DOSIS_RAW:
        for dosis in fila:
            assert dosis in ['Alta', 'Baja'], f"Dosis inválida: {dosis}"
    
    print("✅ Configuración validada correctamente")

if __name__ == "__main__":
    validar_configuracion()
    print("\n📊 Resumen de configuración:")
    print(f"   Total sujetos: {len(TODOS_LOS_SUJETOS)}")
    print(f"   Sujetos válidos: {len(SUJETOS_VALIDOS)}")
    print(f"   Experimentos: {len(EXPERIMENTOS)}")
    print(f"   Duración DMT: {DURACIONES_ESPERADAS['DMT']/60:.1f} min")
    print(f"   Duración Reposo: {DURACIONES_ESPERADAS['Reposo']/60:.1f} min")
