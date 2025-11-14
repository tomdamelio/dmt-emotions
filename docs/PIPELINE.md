# Pipeline de análisis y generación de resultados

Este documento describe, en orden de ejecución, los scripts necesarios para reproducir los análisis y figuras del proyecto, desde el procesamiento de datos hasta la creación de las figuras finales. Para cada paso se indica: entradas esperadas, qué hace el script y qué produce como salida.

**Nota**: Los scripts de desarrollo y testeo se encuentran en `/test` y `/old_scripts`. Este documento solo describe los scripts de producción en `/scripts` que conforman el pipeline principal.


## Estructura actual de los datos

### 📁 **Datos originales** (`../data/original/`)

#### **Fisiología** (`../data/original/physiology/`)
Los datos de fisiología están organizados en **4 condiciones experimentales**:

- **`DMT_1/`** - Primera sesión con DMT
- **`DMT_2/`** - Segunda sesión con DMT  
- **`Reposo_1/`** - Primera sesión de reposo
- **`Reposo_2/`** - Segunda sesión de reposo

Cada condición contiene carpetas por sujeto (`S01/`, `S02/`, ..., `S20/`) y cada sujeto tiene **3 archivos BrainVision**:
- `.vhdr` - Header file (metadatos de canales, frecuencia de muestreo, etc.)
- `.eeg` - Datos binarios de EEG/fisiología (ECG, GSR, RESP)
- `.vmrk` - Marcadores/eventos temporales (no nos imorta esto en principio)

**Sujetos disponibles**: S01-S13, S15-S20 (falta S14, probablemente excluido del estudio)

#### **Autorreportes** (`../data/original/reports/`)
Los autorreportes están disponibles en **dos versiones**:

1. **`resampled/`** - Autorreportes estándar (sin sujeto S12)
2. **`resampled - con el s12/`** - Autorreportes incluyendo datos del sujeto S12 (revisar despues si es mejor usar estos archivos o `resampled` directamente)

Cada carpeta contiene archivos `.mat` con el patrón de nomenclatura:
- `sXX_RS_Session1_EC.mat` - Reposo sesión 1 (eyes closed)
- `sXX_DMT_Session1_DMT.mat` - DMT sesión 1 
- `sXX_RS_Session2_EC.mat` - Reposo sesión 2 (eyes closed)
- `sXX_DMT_Session2_DMT.mat` - DMT sesión 2 

**Contenido de archivos .mat**: Matriz `dimensions` de shape variable (15 dimensiones × n_puntos):

> **⚠️ NOTAS CRÍTICAS**:
> 1. **Orden de dimensiones**: Fundamental para la validez del análisis. Implementado en `config.TET_DIMENSION_COLUMNS`.
> 2. **Resolución temporal**: Los archivos .mat contienen datos down-sampled uniformemente a **0.25 Hz** (1 punto cada 4s):
>    - DMT: 300 puntos @ 0.25 Hz = 1200s = 20 minutos
>    - RS: 150 puntos @ 0.25 Hz = 600s = 10 minutos
> 3. **Columna `t_sec`**: Contiene el timing exacto en segundos (0, 4, 8, 12, ...) para análisis temporales precisos.
> 4. **Paper original**: Especifica bins de 30s (N=40 DMT, N=20 RS), pero los datos se mantienen en resolución original (0.25 Hz). La agregación a 30s se realiza solo cuando es necesario para análisis estadísticos específicos usando `config.aggregate_tet_to_30s_bins()`.

1. **Pleasantness** - Intensidad subjetiva de lo "bueno" de la experiencia
2. **Unpleasantness** - Intensidad subjetiva de lo "malo" de la experiencia  
3. **Emotional_Intensity** - Intensidad emocional independiente de valencia
4. **Elementary_Imagery** - Sensaciones visuales básicas (destellos, colores, patrones geométricos)
5. **Complex_Imagery** - Sensaciones visuales complejas (escenas vívidas, visiones fantásticas)
6. **Auditory** - Sensaciones en el dominio auditivo (sonidos externos o alucinatorios)
7. **Interoception** - Intensidad de sensaciones corporales internas ("body load")
8. **Bliss** - Experiencia de éxtasis o paz profunda
9. **Anxiety** - Experiencia de disforia o ansiedad
10. **Entity** - Presencia percibida de "entidades autónomas" (guías, espíritus, aliens)
11. **Selfhood** - Alteraciones en la experiencia del "yo" (disolución del ego)
12. **Disembodiment** - Experiencia de NO identificarse con el propio cuerpo (desencarnación)
13. **Salience** - Sentido subjetivo de significado profundo e importancia del momento
14. **Temporality** - Alteraciones en la experiencia subjetiva del tiempo
15. **General_Intensity** - Intensidad general subjetiva de los efectos del DMT

### 📁 **Datos "old" procesados** (`../data/old/`)

#### **Datos preprocesados** (`../data/old/Preprocesado/`)
CSVs generados por los scripts del pipeline, organizados por modalidad fisiológica:

- **`HR/`** - Frecuencia cardíaca derivada de ECG
- **`SCL/`** - Conductancia tónica (componente EDA)
- **`SCR/`** - Respuestas fásicas (picos EDA)

Cada modalidad contiene 4 archivos:
- `<MEDIDA>_dmt_alta.csv` - Series temporales para dosis alta
- `<MEDIDA>_dmt_baja.csv` - Series temporales para dosis baja  
- `<MEDIDA>_tiempo_dmt_alta.csv` - Vector temporal correspondiente (dosis alta)
- `<MEDIDA>_tiempo_dmt_baja.csv` - Vector temporal correspondiente (dosis baja)

#### **Datos para clustering** (`../data/old/Data Cluster/`)
- `Datos_reportes_para_clusterizar_sin_reposo.csv` - Dataset concatenado de autorreportes (solo condiciones DMT, excluyendo reposo) preparado para análisis multivariado

### 📁 **Derivatives** (`../data/derivatives/`)

Directorio que seguirá el estándar **BIDS-Derivatives** para almacenar datos procesados de fisiología. La estructura propuesta organizará los outputs de preprocesamiento por modalidad y pipeline:


---

## Pipeline de análisis (orden de ejecución)

### 0) Configuración del proyecto

#### **Archivo**: `config.py`
- **Descripción**: Configuración central del proyecto que define paths, constantes, listas de sujetos, tabla de dosis, y parámetros de procesamiento.
- **Contenido principal**:
  - `PHYSIOLOGY_DATA`, `DERIVATIVES_DATA`: Paths a datos originales y procesados
  - `DOSIS`: Tabla que mapea sujetos a dosis alta/baja por sesión
  - `SUJETOS_VALIDADOS_EDA`, `SUJETOS_VALIDADOS_ECG`, `SUJETOS_VALIDADOS_RESP`: Listas de sujetos validados por modalidad de señal
  - `CANALES`: Mapeo de nombres de canales (EDA='GSR', ECG='ECG', RESP='RESP')
  - `DURACIONES_ESPERADAS`: Duración de cada tipo de sesión (DMT: 20:15 min, Reposo: 10:15 min)
  - `NEUROKIT_PARAMS`: Parámetros para procesamiento con NeuroKit
  - `EDA_ANALYSIS_CONFIG`: Configuración de métodos de análisis EDA (emotiphai, cvx, etc.)
  - Funciones auxiliares: `get_dosis_sujeto()`, `get_nombre_archivo()`, `get_duracion_esperada()`
- **Nota**: Este archivo debe ser revisado antes de ejecutar cualquier script del pipeline.

---

### 1) Preprocesamiento de señales fisiológicas

#### **Script**: `scripts/preprocess_phys.py`
- **Input**:
  - Archivos BrainVision `.vhdr` en `../data/original/physiology/`:
    - `DMT_1/`, `DMT_2/`: Sesiones DMT (20 sujetos: S01-S13, S15-S20)
    - `Reposo_1/`, `Reposo_2/`: Sesiones de reposo
  - Configuración desde `config.py` (tabla de dosis, sujetos, parámetros)
  - Canales procesados: `GSR` (EDA), `ECG`, `RESP`
  
- **Proceso**:
  - **EDA**: Procesamiento con NeuroKit (`nk.eda_process`) + análisis adicionales:
    - **CVX decomposition** (BioSPPy): Descomposición en EDL (tonic/SCL), SMNA, EDR (phasic)
    - **EmotiPhai SCR** (BioSPPy): Detección de eventos SCR (onsets, peaks, amplitudes)
  - **ECG**: Procesamiento con NeuroKit (`nk.ecg_process`) para obtener HR y HRV
  - **RESP**: Procesamiento con NeuroKit (`nk.rsp_process`, método khodadad2018)
  - **Estrategia de duración**: Truncamiento o NaN-padding para estandarizar todas las señales a la duración esperada
  - **Sin corrección de baseline** en este paso (preserva datos crudos procesados)
  
- **Output**:
  - CSVs organizados por señal y condición (high/low) en `../data/derivatives/phys/`:
    - `eda/dmt_high/`, `eda/dmt_low/`: Datos EDA con todas las variables de NeuroKit
    - `ecg/dmt_high/`, `ecg/dmt_low/`: Datos ECG con todas las variables de NeuroKit
    - `resp/dmt_high/`, `resp/dmt_low/`: Datos RESP con todas las variables de NeuroKit
  - Archivos adicionales para EDA:
    - `*_cvx_decomposition.csv`: EDL, SMNA, EDR por sesión (time-series)
    - `*_emotiphai_scr.csv`: Eventos SCR detectados (onsets, peaks, amplitudes)
  - JSON de info de NeuroKit por archivo: `*_info.json`
  - Log de procesamiento: `../data/derivatives/logs/physiology_preprocessing_log.json`
  
- **Formato de archivos**: `{subject}_{session_type}_{experiment}_{condition}.csv`
  - Ejemplo: `S04_dmt_session1_high.csv`, `S04_rs_session1_high.csv`

- **Comando**:
  ```bash
  python scripts/preprocess_phys.py
  ```

---

### 2) Validación de calidad de datos y selección de sujetos

#### **Validación manual** (realizada previamente)
- **Archivo de validación**:
  - `validation_log.json`: Log de validación de todos los sujetos por señal (EDA, ECG, RESP)
  
- **Script de validación interactiva**: `test/test_phys_preprocessing.py`
  - **Propósito**: Script usado para realizar la validación manual de calidad de señales fisiológicas
  - **Proceso**:
    1. Carga los CSVs procesados de `preprocess_phys.py`
    2. Genera plots interactivos de NeuroKit para cada señal
    3. Para EDA: genera plots adicionales de CVX decomposition (EDL, SMNA, EDR) con eventos EmotiPhai superpuestos
    4. Solicita al usuario evaluar la calidad de cada señal (categorías: good, acceptable, maybe, bad)
    5. Permite agregar notas cualitativas sobre artefactos, ruido, etc.
    6. Actualiza automáticamente `validation_log.json` con las evaluaciones
  - **⚠️ IMPORTANTE**: Este script **NO debe correrse nuevamente** salvo que sea estrictamente necesario revalidar señales, ya que la validación manual es un proceso largo y laborioso (evalúa ~20 sujetos × 3 señales × 4 archivos = ~240 archivos)
  - **Uso**: Solo si necesitas re-evaluar sujetos específicos o validar datos reprocesados
  
- **Criterio de validación**:
  - Cada sujeto debe tener los 4 archivos (DMT_1, DMT_2, Reposo_1, Reposo_2) clasificados como 'good' o 'acceptable' para ser incluido
  - Sujetos con señales 'poor' o 'bad' son excluidos para esa modalidad
  - Evaluación visual basada en plots de NeuroKit y análisis de componentes EDA
  
- **Resultado**: Listas de sujetos validados registradas en `config.py`:
  - `SUJETOS_VALIDADOS_EDA`: 11 sujetos (S04, S05, S06, S07, S09, S13, S16, S17, S18, S19, S20)
  - `SUJETOS_VALIDADOS_ECG`: 15 sujetos
  - `SUJETOS_VALIDADOS_RESP`: 12 sujetos

- **Nota**: Los scripts de análisis posteriores usan estas listas de sujetos validados automáticamente desde `config.py`.

---

### 3) Análisis de EDA (Electrodermal Activity)

Los siguientes tres scripts procesan diferentes componentes de la señal EDA y generan modelos estadísticos (LME/GEE) con corrección FDR y visualizaciones.

#### 3.1) **Script**: `scripts/run_eda_scl_analysis.py`
- **Componente**: SCL (Skin Conductance Level) / EDL (tónico)
- **Input**:
  - CSVs de CVX decomposition: `*_cvx_decomposition.csv` desde `../data/derivatives/phys/eda/dmt_{high,low}/`
  - Solo sujetos de `SUJETOS_VALIDADOS_EDA` (11 sujetos)
  - Ventana de análisis: **primeros 9 minutos** (0-8 min)
  
- **Proceso**:
  1. Carga señal EDL (SCL tónica) de CVX decomposition
  2. **Corrección de baseline**: Resta media del primer segundo (0-1s)
  3. Calcula **AUC por minuto** (1-min windows) para cada condición (Task × Dose: RS/DMT × Low/High)
  4. Ajusta **modelo LME** (Linear Mixed Effects):
     - Formula: `AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c`
     - Efectos aleatorios: `~ 1 | subject`
  5. Aplica **corrección BH-FDR** por familias de hipótesis (Task, Dose, Interaction)
  6. Genera plots de diagnóstico, coeficientes, medias marginales, interacciones
  7. Crea plots de series temporales completas (9 min y 19 min) con significancia FDR
  8. Genera plot stacked por sujeto individual
  
- **Output**:
  - `results/eda/scl/`:
    - `scl_auc_long_data.csv`: Datos en formato long (AUC por minuto)
    - `lme_analysis_report.txt`: Reporte completo del modelo LME con coeficientes y p-valores FDR
    - `model_summary.txt`: Resumen del modelo
    - `captions_scl.txt`: Captions para figuras
  - `results/eda/scl/plots/`:
    - `lme_coefficient_plot.png`: Coeficientes β con IC 95%
    - `marginal_means_all_conditions.png`: Medias marginales por condición
    - `task_main_effect.png`: Efecto principal de Task
    - `task_dose_interaction.png`: Interacción Task × Dose
    - `all_subs_eda_scl.png`: **Plot grupal RS+DMT (9 min)** con shading FDR
    - `all_subs_dmt_eda_scl.png`: **Plot DMT-only (~19 min)** con shading FDR
    - `stacked_subs_eda_scl.png`: **Plot stacked por sujeto (9 min)**
    - `lme_diagnostics.png`: Diagnósticos del modelo
    - `effect_sizes_table.csv`, `summary_statistics.csv`
    - FDR reports: `fdr_segments_all_subs_eda_scl.txt`, `fdr_segments_all_subs_dmt_eda_scl.txt`

- **Comando**:
  ```bash
  python scripts/run_eda_scl_analysis.py
  ```

#### 3.2) **Script**: `scripts/run_eda_smna_analysis.py`
- **Componente**: SMNA (Sympathetic Nervous Activity, componente fásico continuo)
- **Input**: 
  - CSVs de CVX decomposition: `*_cvx_decomposition.csv` (columna SMNA)
  - Ventana de análisis: **primeros 9 minutos**
  
- **Proceso**: Idéntico a SCL pero usando señal SMNA (sin corrección de baseline)
  - Modelo LME con misma especificación
  - Corrección BH-FDR por familias
  - Plots equivalentes a SCL
  
- **Output**:
  - `results/eda/smna/`:
    - Estructura idéntica a `results/eda/scl/`
    - `lme_analysis_report.txt`: Reporte del modelo SMNA
    - Plots en `results/eda/smna/plots/`:
      - `lme_coefficient_plot.png`
      - `all_subs_smna.png`: **Plot grupal RS+DMT (9 min)**
      - `all_subs_dmt_smna.png`: **Plot DMT-only (~19 min)**
      - `stacked_subs_smna.png`: **Plot stacked por sujeto (9 min)**
      - Otros plots de medias marginales, interacciones, diagnósticos
      
- **Comando**:
  ```bash
  python scripts/run_eda_smna_analysis.py
  ```

#### 3.3) **Script**: `scripts/run_eda_scr_analysis.py`
- **Componente**: SCR (Skin Conductance Responses, eventos fásicos discretos)
- **Input**:
  - CSVs de EmotiPhai: `*_emotiphai_scr.csv` (eventos SCR con onsets, peaks, amplitudes)
  - Ventana de análisis: **primeros 9 minutos**
  
- **Proceso**:
  1. Carga eventos SCR detectados por EmotiPhai
  2. Calcula **conteo de SCRs por minuto** para cada condición
  3. Ajusta **modelo Poisson GEE** (Generalized Estimating Equations):
     - Distribución: Poisson (para datos de conteo)
     - Link: log
     - Working correlation: Exchangeable
     - Formula: `count ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c`
  4. Aplica **corrección BH-FDR** por familias
  5. Genera plots con coeficientes en escala log y rate ratios exp(β)
  6. Series temporales con SEM (no CI) para mantener consistencia con plots legacy
  
- **Output**:
  - `results/eda/scr/`:
    - `scr_counts_long_data.csv`: Datos en formato long (conteo por minuto)
    - `gee_analysis_report.txt`: Reporte del modelo GEE con coeficientes y rate ratios
    - `captions_scr.txt`: Captions para figuras
  - `results/eda/scr/plots/`:
    - `gee_coefficient_plot.png`: Coeficientes β (log scale) con exp(β) anotado
    - `all_subs_scr_rate_timecourse.png`: **Plot grupal RS+DMT (9 min)** con shading FDR
    - `stacked_subs_scr_rate.png`: **Plot stacked por sujeto (9 min)**
    - `task_main_effect.png`, `task_dose_interaction.png`
    - `effect_sizes_table.csv`
    - FDR reports: `fdr_segments_all_subs_scr_rs.txt`, `fdr_segments_all_subs_scr_dmt.txt`
      
- **Comando**:
  ```bash
  python scripts/run_eda_scr_analysis.py
  ```

---

### 4) Generación de paneles de figuras para publicación

#### **Script**: `scripts/generate_eda_figures.py`
- **Input**:
  - Plots individuales generados por los scripts de análisis EDA en:
    - `results/eda/scl/plots/`
    - `results/eda/smna/plots/`
  
- **Proceso**:
  - Lee imágenes PNG pre-generadas
  - Construye 3 paneles compuestos con labels (A, B, C, D) y layout optimizado
  
- **Output**: `results/eda/panels/`
  - **`panel_1.png`** (2×2 grid, 34×16 inches):
    - A (top-left): `all_subs_eda_scl.png` - SCL grupal RS+DMT
    - B (top-right): `scl/plots/lme_coefficient_plot.png` - Coeficientes LME de SCL
    - C (bottom-left): `all_subs_smna.png` - SMNA grupal RS+DMT
    - D (bottom-right): `smna/plots/lme_coefficient_plot.png` - Coeficientes LME de SMNA
  
  - **`panel_2.png`** (1×2 vertical, 12×12 inches):
    - A (top): `all_subs_dmt_eda_scl.png` - SCL DMT-only (~19 min)
    - B (bottom): `all_subs_dmt_smna.png` - SMNA DMT-only (~19 min)
  
  - **`panel_3.png`** (1×2 horizontal, 12×26 inches):
    - A (left): `stacked_subs_eda_scl.png` - SCL por sujeto (9 min)
    - B (right): `stacked_subs_smna.png` - SMNA por sujeto (9 min)

- **Comando**:
  ```bash
  python scripts/generate_eda_figures.py
  ```

- **Nota**: Este script debe ejecutarse **después** de los tres análisis de EDA para asegurar que todos los plots de input existan.

---

### 5) Análisis de ECG (Electrocardiografía)

Los siguientes dos scripts procesan la señal ECG para obtener métricas de frecuencia cardíaca (HR) y variabilidad de la frecuencia cardíaca (HRV), generando modelos estadísticos LME con corrección FDR y visualizaciones.

#### 5.1) **Script**: `scripts/run_ecg_hr_analysis.py`
- **Componente**: HR (Heart Rate / Frecuencia Cardíaca)
- **Input**:
  - CSVs de ECG: `*_ecg.csv` desde `../data/derivatives/phys/ecg/dmt_{high,low}/`
  - Solo sujetos de `SUJETOS_VALIDADOS_ECG` (15 sujetos)
  - Columna utilizada: `ECG_Rate` (bpm)
  - Ventana de análisis: **primeros 9 minutos** (0-8 min)
  
- **Proceso**:
  1. Carga señal `ECG_Rate` (HR instantánea en bpm) de NeuroKit
  2. **Sin corrección de baseline** por defecto (flag `BASELINE_CORRECTION=False`)
  3. Calcula **HR media por minuto** (1-min windows) para cada condición (Task × Dose: RS/DMT × Low/High)
  4. Ajusta **modelo LME** (Linear Mixed Effects):
     - Formula: `HR ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c`
     - Efectos aleatorios: `~ 1 | subject`
  5. Aplica **corrección BH-FDR** por familias de hipótesis (Task, Dose, Interaction)
  6. Genera plots de diagnóstico, coeficientes, medias marginales, interacciones
  7. Crea plots de series temporales completas (9 min) con significancia FDR (High vs Low)
  8. Genera plot stacked por sujeto individual
  
- **Output**:
  - `results/ecg/hr/`:
    - `hr_minute_long_data.csv`: Datos en formato long (HR media por minuto)
    - `lme_analysis_report.txt`: Reporte completo del modelo LME con coeficientes y p-valores FDR
    - `model_summary.txt`: Resumen del modelo
    - `captions_hr.txt`: Captions para figuras
  - `results/ecg/hr/plots/`:
    - `lme_coefficient_plot.png`: Coeficientes β con IC 95%
    - `marginal_means_all_conditions.png`: Medias marginales por condición
    - `task_main_effect.png`: Efecto principal de Task
    - `task_dose_interaction.png`: Interacción Task × Dose
    - `all_subs_ecg_hr.png`: **Plot grupal RS+DMT (9 min)** con shading FDR
    - `all_subs_dmt_ecg_hr.png`: **Plot DMT-only extendido (~19 min)** con shading FDR
    - `stacked_subs_ecg_hr.png`: **Plot stacked por sujeto (9 min)**
    - `lme_diagnostics.png`: Diagnósticos del modelo
    - `effect_sizes_table.csv`, `summary_statistics.csv`
  - `results/ecg/hr/fdr_segments_all_subs_ecg_hr.txt`: Reporte de segmentos FDR (9 min, RS+DMT)
  - `results/ecg/hr/fdr_segments_all_subs_dmt_ecg_hr.txt`: Reporte de segmentos FDR (19 min, DMT-only)

- **Comando**:
  ```bash
  python scripts/run_ecg_hr_analysis.py
  ```

#### 5.2) **Script**: `scripts/run_ecg_hrv_analysis.py`
- **Componente**: HRV (Heart Rate Variability / Variabilidad de la Frecuencia Cardíaca)
- **Input**: 
  - Mismos CSVs de ECG, columna `ECG_R_Peaks` (binaria)
  - Extracción de intervalos RR desde R-peaks
  - Ventana de análisis: **primeros 9 minutos**
  
- **Proceso**:
  1. Extrae intervalos RR desde `ECG_R_Peaks` (detección binaria de picos R)
  2. Filtra RR fisiológicamente plausibles (300-2000 ms)
  3. Calcula **HRV features por minuto**:
     - **Primaria**: RMSSD (Root Mean Square of Successive Differences)
     - **Secundarias**: SDNN, pNN50, LF, HF, LF/HF, SD1, SD2
     - Frecuencia: Interpolación a 4 Hz, Welch PSD, bandas LF (0.04-0.15 Hz) / HF (0.15-0.40 Hz)
  4. Ajusta **modelo LME** con RMSSD como DV:
     - Formula: `RMSSD ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c`
  5. Aplica **corrección BH-FDR** por familias
  6. Genera plots con timecourse discreto (puntos por minuto) y FDR por minuto
  7. Exporta todas las features HRV en CSVs por sujeto-condición
  
- **Output**:
  - `results/ecg/hrv/`:
    - `hrv_rmssd_long_data.csv`: Datos en formato long (RMSSD por minuto)
    - `features_per_minute/`: CSVs con todas las features HRV por sujeto-condición
    - `lme_analysis_report.txt`: Reporte del modelo HRV
    - `model_summary.txt`: Resumen del modelo
    - `captions_hrv.txt`: Captions para figuras
  - `results/ecg/hrv/plots/`:
    - `lme_coefficient_plot.png`: Coeficientes β con IC 95%
    - `marginal_means_all_conditions.png`: Medias marginales por condición
    - `task_main_effect.png`: Efecto principal de Task
    - `task_dose_interaction.png`: Interacción Task × Dose
    - `timecourse_hrv_rmssd.png`: **Plot discreto (9 min)** con shading FDR por minuto
    - `stacked_subs_ecg_hrv.png`: **Plot stacked por sujeto (9 min)**
    - `lme_diagnostics.png`: Diagnósticos del modelo
    - `effect_sizes_table.csv`
  - `results/ecg/hrv/fdr_minutes_all_subs_ecg_hrv.txt`: Reporte de minutos FDR significativos
      
- **Comando**:
  ```bash
  python scripts/run_ecg_hrv_analysis.py
  ```

---

### 6) Análisis de Respiración (pendiente)

Los siguientes análisis están planificados pero aún no implementados:

- **Respiración**:
  - Análisis de patrones respiratorios y variabilidad
  - Sujetos validados: 12 (ver `SUJETOS_VALIDADOS_RESP` en `config.py`)
  - Input: CSVs de `../data/derivatives/phys/resp/`

---

### 7) Generación de Figuras TET (Temporal Experience Tracking)

#### **Script**: `scripts/generate_all_figures.py`
- **Descripción**: Script maestro que orquesta la generación de todas las figuras de análisis TET para publicación
- **Input**:
  - Directorio de resultados TET (default: `results/tet/`)
  - Datos preprocesados: `tet_preprocessed.csv`
  - Resultados LME: `lme/lme_results.csv`, `lme/lme_contrasts.csv`
  - Datos descriptivos: `descriptive/time_course_all_dimensions.csv`
  - Resultados Peak/AUC: `peak_auc/peak_auc_metrics.csv`, `peak_auc/peak_auc_tests.csv`
  - Resultados PCA: `pca/pca_variance_explained.csv`, `pca/pca_loadings.csv`
  - Resultados clustering: `clustering/clustering_kmeans_assignments.csv`

- **Proceso**:
  1. **Verificación de archivos**: Comprueba la existencia de archivos requeridos antes de generar cada tipo de figura
  2. **Generación selectiva**: Permite generar todos los tipos de figuras o solo tipos específicos
  3. **Manejo de errores**: Usa bloques try-except para cada tipo de figura, continúa con las disponibles si algunas fallan
  4. **Reporte de estado**: Registra figuras generadas, omitidas y fallidas
  5. **Índice HTML**: Crea un índice HTML con enlaces a todas las figuras generadas

- **Tipos de figuras generadas**:
  
  **7.1) Series Temporales Anotadas (Req 8.1)**
  - Script subyacente: `plot_time_series.py`
  - Archivo: `timeseries_all_dimensions.png`
  - Contenido:
    - Trayectorias medias con sombreado SEM para dosis Low (20mg) y High (40mg)
    - Línea base RS como punto de referencia
    - Sombreado gris de fondo para bins temporales donde DMT difiere significativamente de RS
    - Barras horizontales negras para bins con interacciones State:Dose significativas
    - Dimensiones ordenadas por fuerza del efecto State principal
  - Dimensiones: 12 × 10 pulgadas, 300 DPI
  - Requisitos: Datos preprocesados, resultados LME, contrastes LME, cursos temporales
  
  **7.2) Gráficos de Bosque de Coeficientes LME (Req 8.2)**
  - Script subyacente: `plot_lme_coefficients.py`
  - Archivo: `lme_coefficients_forest.png`
  - Contenido:
    - Estimaciones β con intervalos de confianza del 95%
    - Círculos rellenos para p_fdr < 0.05, círculos abiertos para p_fdr ≥ 0.05
    - Codificación por colores según tipo de efecto (State, Dose, Interacción)
    - Paneles separados para cada tipo de efecto fijo
    - Dimensiones ordenadas por fuerza del efecto State
  - Dimensiones: 10 × 8 pulgadas, 300 DPI
  - Requisitos: Resultados LME
  
  **7.3) Boxplots de Comparación de Dosis Peak/AUC (Req 8.3)**
  - Script subyacente: `plot_peak_auc.py`
  - Archivos:
    - `peak_dose_comparison.png`: Valores pico (z-scores)
    - `time_to_peak_dose_comparison.png`: Tiempo hasta el pico (minutos)
    - `auc_dose_comparison.png`: AUC 0-9 min (z-score × min)
  - Contenido:
    - Cajas azules para dosis Low, cajas rojas para dosis High
    - Líneas grises conectando observaciones pareadas (mismo sujeto)
    - Estrellas de significancia: * p < 0.05, ** p < 0.01, *** p < 0.001
    - Tamaños de efecto (r) con IC 95% de bootstrap
    - Solo anotaciones para p_fdr < 0.05
  - Dimensiones: 10 × 6 pulgadas por figura, 300 DPI
  - Requisitos: Métricas Peak/AUC, resultados de pruebas
  
  **7.4) Figuras PCA (Req 8.4, 8.5)**
  - Estado: Pendiente de implementación
  - Figuras planificadas:
    - Gráfico de sedimentación mostrando varianza explicada
    - Mapas de calor o gráficos de barras de cargas
  - Requisitos: Varianza explicada PCA, cargas PCA
  
  **7.5) Figuras de Clustering (Req 8.6, 8.7)**
  - Script subyacente: `plot_state_results.py`
  - Nota: Usar directamente `plot_state_results.py` para figuras de clustering
  - Figuras generadas:
    - Perfiles de centroides KMeans (similar a Fig. 3.5)
    - Gráficos de cursos temporales de probabilidad de cluster (similar a Fig. 3.6)
    - Cursos temporales de estado GLHMM
    - Mapas de calor de correspondencia KMeans-GLHMM
  - Requisitos: Datos preprocesados, asignaciones KMeans, rutas Viterbi GLHMM, probabilidades GLHMM
  
  **7.6) Figuras GLHMM (Req 8.8)**
  - Estado: Trabajo futuro opcional (no implementado aún)
  - Razón: El análisis GLHMM es trabajo futuro opcional

- **Output**:
  - Figuras PNG en `results/tet/figures/` con resolución especificada (default: 300 DPI)
  - `index.html`: Índice HTML con enlaces a todas las figuras generadas
  - Reporte de resumen en consola con estado de cada tipo de figura
  - Registro de figuras generadas, omitidas y fallidas

- **Opciones de línea de comandos**:
  ```bash
  # Generar todas las figuras con configuración predeterminada
  python scripts/generate_all_figures.py
  
  # Especificar directorios personalizados de entrada/salida
  python scripts/generate_all_figures.py --input results/tet --output results/tet/figures
  
  # Generar solo tipos de figuras específicos
  python scripts/generate_all_figures.py --figures time-series lme peak-auc
  
  # Salida de alta resolución
  python scripts/generate_all_figures.py --dpi 600 --verbose
  ```

- **Manejo de errores**:
  - Verifica archivos requeridos antes de cada generación de figura
  - Registra advertencias para figuras omitidas (archivos faltantes)
  - Registra errores para generaciones de figuras fallidas
  - Continúa con figuras disponibles si algunas fallan
  - Genera reporte de resumen de generadas vs omitidas

- **Interpretación de figuras**:
  
  **Series Temporales (8.1)**:
  - Eje X: Tiempo en minutos (0-20 para DMT)
  - Eje Y: Intensidad en z-score
  - Línea azul: Dosis Low (20mg) con sombreado SEM
  - Línea roja: Dosis High (40mg) con sombreado SEM
  - Línea vertical gris discontinua: Inicio de DMT (fin de línea base RS)
  - Fondo gris: Bins temporales donde DMT difiere significativamente de RS (p < 0.05)
  - Barras negras: Bins temporales con interacción State:Dose significativa (p < 0.05)
  - Orden de dimensiones: Por fuerza del efecto State principal (más fuerte primero)
  
  **Coeficientes LME (8.2)**:
  - Eje X: Valor del coeficiente β
  - Eje Y: Dimensiones (ordenadas por efecto State)
  - Puntos: Estimaciones β
  - Barras de error: Intervalos de confianza del 95%
  - Círculos rellenos: p_fdr < 0.05 (significativo)
  - Círculos abiertos: p_fdr ≥ 0.05 (no significativo)
  - Línea vertical en x=0: Referencia de efecto nulo
  - Paneles separados: Por tipo de efecto (State, Dose, State:Dose, etc.)
  
  **Boxplots Peak/AUC (8.3)**:
  - Eje X: Dimensiones (ordenadas por tamaño del efecto)
  - Eje Y: Valor de métrica (pico, tiempo hasta el pico, o AUC)
  - Cajas azules: Dosis Low (20mg)
  - Cajas rojas: Dosis High (40mg)
  - Líneas grises: Observaciones pareadas (mismo sujeto entre dosis)
  - Estrellas: Significancia después de corrección FDR
    - * p < 0.05
    - ** p < 0.01
    - *** p < 0.001
  - Interpretación:
    - Pico: Intensidad máxima alcanzada (z-scores más altos = experiencias más fuertes)
    - Tiempo hasta el pico: Minutos hasta la intensidad máxima (valores más tempranos = inicio más rápido)
    - AUC: Intensidad acumulada en el tiempo (AUC más alto = experiencias elevadas sostenidas)

- **Solución de problemas**:
  
  **Problema**: Figuras omitidas debido a archivos faltantes
  - **Solución**: Ejecutar los scripts de análisis requeridos primero:
    1. `python scripts/preprocess_tet_data.py` (para datos preprocesados)
    2. `python scripts/compute_descriptive_stats.py` (para cursos temporales)
    3. `python scripts/fit_lme_models.py` (para resultados LME)
    4. `python scripts/compute_peak_auc.py` (para análisis Peak/AUC)
    5. `python scripts/compute_pca_analysis.py` (para análisis PCA)
    6. `python scripts/compute_clustering_analysis.py` (para análisis de clustering)
  
  **Problema**: Generación de figuras fallida con error
  - **Solución**: Verificar el registro de errores en la salida de consola
  - Verificar que los archivos de entrada tengan el formato correcto
  - Verificar que todas las dependencias estén instaladas
  - Ejecutar con `--verbose` para salida de depuración detallada
  
  **Problema**: Figuras PCA o clustering omitidas
  - **Solución**: Estas figuras requieren scripts de análisis específicos
  - Para PCA: Ejecutar `python scripts/compute_pca_analysis.py` primero
  - Para clustering: Usar `python scripts/plot_state_results.py` directamente
  
  **Problema**: Figuras de baja calidad o pixeladas
  - **Solución**: Aumentar la resolución DPI:
    ```bash
    python scripts/generate_all_figures.py --dpi 600
    ```
  
  **Problema**: Índice HTML no muestra imágenes
  - **Solución**: Verificar que las figuras se generaron exitosamente
  - Verificar que el directorio de salida sea correcto
  - Abrir `index.html` desde el directorio de salida

- **Personalización de figuras**:
  - Resolución: Usar flag `--dpi` (default: 300, recomendado para publicación: 300-600)
  - Tipos de figuras: Usar flag `--figures` para generar solo tipos específicos
  - Directorios: Usar flags `--input` y `--output` para rutas personalizadas
  - Estilo: Modificar scripts individuales de graficación para personalización detallada

---

## Resumen del orden de ejecución

### Pipeline de Fisiología

```bash
# 1. Preprocesamiento de señales fisiológicas
python scripts/preprocess_phys.py

# 2. Validación de datos (ya realizada, sujetos validados en config.py)

# 3. Análisis de EDA (los tres scripts pueden correrse en paralelo)
python scripts/run_eda_scl_analysis.py   # SCL/EDL (tónico)
python scripts/run_eda_smna_analysis.py  # SMNA (fásico continuo)
python scripts/run_eda_scr_analysis.py   # SCR (eventos discretos)

# 4. Generación de paneles compuestos EDA
python scripts/generate_eda_figures.py

# 5. Análisis de ECG (los dos scripts pueden correrse en paralelo)
python scripts/run_ecg_hr_analysis.py    # Heart Rate (HR)
python scripts/run_ecg_hrv_analysis.py   # Heart Rate Variability (HRV)

# 6. Análisis de RESP (pendiente)
```

### Pipeline de TET (Temporal Experience Tracking)

```bash
# 1. Preprocesamiento de datos TET
python scripts/preprocess_tet_data.py

# 2. Estadísticas descriptivas y cursos temporales
python scripts/compute_descriptive_stats.py

# 3. Modelos de efectos mixtos lineales (LME)
python scripts/fit_lme_models.py

# 4. Análisis de pico y AUC
python scripts/compute_peak_auc.py

# 5. Análisis de componentes principales (PCA)
python scripts/compute_pca_analysis.py

# 6. Análisis de clustering y modelado de estados
python scripts/compute_clustering_analysis.py

# 7. Generación de todas las figuras TET
python scripts/generate_all_figures.py

# Alternativamente, generar solo tipos específicos de figuras:
python scripts/generate_all_figures.py --figures time-series lme peak-auc

# O generar figuras de clustering directamente:
python scripts/plot_state_results.py
```

---

## Dependencias principales

- **Python**: 3.11
- **Procesamiento de señales**: `mne`, `neurokit2`, `biosppy`
- **Análisis numérico**: `numpy`, `scipy`, `pandas`
- **Estadística**: `statsmodels` (LME, GEE), `patsy`
- **Visualización**: `matplotlib`, `seaborn`
- **Gestión de ambiente**: `micromamba` (ver `environment.yml`)

## Configuración antes de ejecutar

1. **Revisar `config.py`**:
   - Verificar paths a datos originales (`PHYSIOLOGY_DATA`)
   - Confirmar lista de sujetos a procesar (`TEST_MODE`, `PROCESSING_MODE`)
   - Ajustar parámetros de análisis EDA si necesario (`EDA_ANALYSIS_CONFIG`)

2. **Ambiente**:
   ```bash
   micromamba create -n dmt-emotions -f environment.yml
   micromamba activate dmt-emotions
   pip install -e .[dev]
   ```

3. **Estructura de directorios**: Los scripts crean automáticamente los directorios de output necesarios.



    - Usar `--dpi 600` para mayor resolución
    - Verificar que matplotlib esté actualizado

---

### 8) Generación de Reporte Integral de Resultados TET

#### **Script**: `scripts/generate_comprehensive_report.py`
- **Descripción**: Genera un documento markdown integral que sintetiza todos los hallazgos del análisis TET
- **Input**:
  - Directorio de resultados TET (default: `results/tet/`)
  - Todos los archivos CSV de análisis:
    - Estadísticas descriptivas: `descriptive/time_course_all_dimensions.csv`, `descriptive/session_summaries.csv`
    - Resultados LME: `lme/lme_results.csv`, `lme/lme_contrasts.csv`
    - Análisis Peak/AUC: `peak_auc/peak_auc_metrics.csv`, `peak_auc/peak_auc_tests.csv`
    - Resultados PCA: `pca/pca_loadings.csv`, `pca/pca_variance_explained.csv`, `pca/pca_lme_results.csv`
    - Resultados clustering: `clustering/clustering_evaluation.csv`, `clustering/clustering_state_metrics.csv`, etc.

- **Proceso**:
  1. **Carga de resultados**: Lee todos los archivos CSV de análisis disponibles
  2. **Detección de actualizaciones**: Compara timestamps de archivos de resultados vs reporte existente
  3. **Síntesis de hallazgos**: Extrae y organiza hallazgos clave de cada análisis
  4. **Ranking de efectos**: Ordena dimensiones por tamaño de efecto y significancia
  5. **Integración cruzada**: Identifica patrones convergentes entre métodos de análisis
  6. **Generación de documento**: Crea documento markdown estructurado con todas las secciones

- **Secciones del reporte**:
  
  **8.1) Executive Summary**
  - 3-5 hallazgos más importantes del análisis completo
  - Tamaños de efecto y niveles de significancia
  - Contexto del estudio (n sujetos, condiciones)
  
  **8.2) Descriptive Statistics**
  - Dinámicas temporales para cada dimensión
  - Patrones de timing de picos (onset, plateau, offset)
  - Comparaciones de intensidad por dosis
  - Métricas de variabilidad inter-sujeto
  
  **8.3) LME Results**
  - Efectos significativos de State, Dose, e interacciones
  - Coeficientes estandarizados con intervalos de confianza
  - P-valores corregidos por FDR
  - Organizado por tipo de efecto y magnitud
  
  **8.4) Peak and AUC Analysis**
  - Comparaciones de dosis para métricas de intensidad y duración
  - Tamaños de efecto con intervalos de confianza bootstrap
  - Ranking de sensibilidad a dosis por dimensión
  - Identificación de dimensiones con mayor/menor respuesta
  
  **8.5) Dimensionality Reduction (PCA)**
  - Interpretación de componentes principales retenidos
  - Cargas de dimensiones y patrones de agrupamiento
  - Varianza explicada por cada componente
  - Dinámicas temporales de scores de componentes
  
  **8.6) Clustering Analysis**
  - Caracterización de estados experienciales discretos
  - Perfiles de dimensiones para cada cluster
  - Patrones de prevalencia temporal por condición
  - Efectos de dosis en métricas de ocupación
  - Métricas de estabilidad (bootstrap ARI)
  
  **8.7) Cross-Analysis Integration**
  - Hallazgos convergentes entre múltiples métodos
  - Rankings de dimensiones por método
  - Correlaciones de rango entre métodos
  - Patrones temporales concordantes
  
  **8.8) Methodological Notes**
  - Problemas de calidad de datos documentados
  - Supuestos de modelos y limitaciones
  - Decisiones analíticas que afectan interpretación
  
  **8.9) Further Investigation**
  - Preguntas sin resolver priorizadas
  - Hallazgos ambiguos o contradictorios
  - Sugerencias de análisis de seguimiento
  - Estimaciones de esfuerzo de implementación

- **Output**:
  - `docs/tet_comprehensive_results.md`: Documento markdown integral
  - Formato consistente con notación estadística estandarizada
  - Referencias a figuras específicas con rutas relativas
  - Organización jerárquica con encabezados claros

- **Opciones de línea de comandos**:
  ```bash
  # Generar reporte con configuración predeterminada
  python scripts/generate_comprehensive_report.py
  
  # Especificar directorios personalizados
  python scripts/generate_comprehensive_report.py \\
      --results-dir results/tet \\
      --output docs/tet_comprehensive_results.md
  
  # Forzar regeneración incluso si está actualizado
  python scripts/generate_comprehensive_report.py --force
  
  # Solo verificar si necesita actualización (no regenerar)
  python scripts/generate_comprehensive_report.py --check-only
  
  # Salida detallada para depuración
  python scripts/generate_comprehensive_report.py --verbose
  ```

- **Generación automática**:
  
  El reporte se genera automáticamente al final de:
  - `scripts/generate_all_figures.py` (después de generar figuras)
  - `scripts/compute_clustering_analysis.py` (después de análisis de clustering)
  
  Para omitir la generación automática (útil para depuración rápida):
  ```bash
  # Omitir reporte en generate_all_figures.py
  python scripts/generate_all_figures.py --skip-report
  
  # Omitir reporte en compute_clustering_analysis.py
  python scripts/compute_clustering_analysis.py --skip-report
  ```

- **Detección de actualizaciones**:
  
  El script detecta automáticamente si los archivos de resultados son más recientes que el reporte:
  - Si el reporte no existe → se genera
  - Si hay archivos de resultados más nuevos → se regenera
  - Si el reporte está actualizado → se omite (usar `--force` para regenerar)
  
  Verificar estado de actualización sin regenerar:
  ```bash
  python scripts/generate_comprehensive_report.py --check-only
  ```
  
  Esto mostrará:
  - Si el reporte necesita actualización
  - Lista de archivos más nuevos que el reporte
  - No regenerará el reporte

- **Estructura del reporte**:
  
  El documento generado sigue esta estructura:
  ```markdown
  # TET Comprehensive Results Report
  
  ## Executive Summary
  - Top 3-5 findings with effect sizes
  
  ## Descriptive Statistics
  ### Dimension 1
  - Peak timing, intensity, variability
  ### Dimension 2
  ...
  
  ## LME Results
  ### State Effects
  - Significant dimensions ranked by |β|
  ### Dose Effects
  ...
  ### Interaction Effects
  ...
  
  ## Peak and AUC Analysis
  ### Peak Values
  - Dose comparisons with effect sizes
  ### Time to Peak
  ...
  ### AUC (0-9 min)
  ...
  
  ## Dimensionality Reduction
  ### PC1
  - Loadings, interpretation, temporal dynamics
  ### PC2
  ...
  
  ## Clustering Analysis
  ### Cluster 1
  - Profile, prevalence, dose effects
  ### Cluster 2
  ...
  
  ## Cross-Analysis Integration
  - Convergent findings
  - Method comparisons
  
  ## Methodological Notes
  - Data quality, assumptions, limitations
  
  ## Further Investigation
  - Unresolved questions
  - Suggested follow-up analyses
  
  ## Appendix
  - Statistical notation guide
  - Figure index
  ```

- **Interpretación del reporte**:
  
  **Notación estadística**:
  - `β`: Coeficiente de regresión (LME)
  - `r`: Tamaño de efecto (Wilcoxon)
  - `p_fdr`: P-valor corregido por FDR
  - `CI [X, Y]`: Intervalo de confianza del 95%
  
  **Niveles de significancia**:
  - `p_fdr < 0.001`: Altamente significativo (***)
  - `p_fdr < 0.01`: Muy significativo (**)
  - `p_fdr < 0.05`: Significativo (*)
  - `p_fdr ≥ 0.05`: No significativo (ns)
  
  **Magnitud de efectos**:
  - LME β: |β| > 1.5 (fuerte), 0.8-1.5 (moderado), < 0.8 (débil)
  - Wilcoxon r: |r| > 0.5 (grande), 0.3-0.5 (mediano), < 0.3 (pequeño)
  
  **Referencias a figuras**:
  - Rutas relativas desde `docs/` a `results/tet/figures/`
  - Ejemplo: `../results/tet/figures/timeseries_all_dimensions.png`

- **Solución de problemas**:
  
  **Problema**: Reporte vacío o con secciones faltantes
  - **Solución**: Verificar que los análisis requeridos se hayan ejecutado
  - Ejecutar scripts de análisis en orden:
    1. `python scripts/preprocess_tet_data.py`
    2. `python scripts/compute_descriptive_stats.py`
    3. `python scripts/fit_lme_models.py`
    4. `python scripts/compute_peak_auc.py`
    5. `python scripts/compute_pca_analysis.py`
    6. `python scripts/compute_clustering_analysis.py`
  - Luego regenerar reporte: `python scripts/generate_comprehensive_report.py --force`
  
  **Problema**: Reporte no se actualiza automáticamente
  - **Solución**: Verificar timestamps de archivos
  - Usar `--check-only` para ver estado de actualización
  - Usar `--force` para forzar regeneración
  - Verificar que los archivos de resultados existan en `results/tet/`
  
  **Problema**: Errores al cargar datos
  - **Solución**: Verificar formato de archivos CSV
  - Verificar que las columnas requeridas estén presentes
  - Ejecutar con `--verbose` para ver detalles de errores
  - Revisar logs de scripts de análisis para errores upstream
  
  **Problema**: Hallazgos faltantes en Executive Summary
  - **Solución**: El ranking automático requiere efectos significativos
  - Verificar que haya efectos con p_fdr < 0.05 en los análisis
  - Si no hay efectos significativos, el Executive Summary estará vacío
  - Revisar resultados de análisis individuales para confirmar

- **Uso en manuscritos**:
  
  El reporte está diseñado para facilitar la escritura de manuscritos:
  - **Executive Summary** → Abstract/Highlights
  - **LME Results** → Results section (main effects)
  - **Peak/AUC Analysis** → Results section (dose-response)
  - **PCA** → Results section (dimensionality)
  - **Clustering** → Results section (experiential states)
  - **Cross-Analysis Integration** → Discussion (convergent evidence)
  - **Methodological Notes** → Methods/Limitations
  - **Further Investigation** → Discussion/Future directions
  
  Copiar secciones relevantes directamente al manuscrito y ajustar formato según revista.

---

## Orden de ejecución recomendado (Pipeline completo TET)

Para reproducir el análisis completo de TET desde cero:

```bash
# 1. Preprocesamiento de datos TET
python scripts/preprocess_tet_data.py

# 2. Estadísticas descriptivas
python scripts/compute_descriptive_stats.py

# 3. Modelos LME
python scripts/fit_lme_models.py

# 4. Análisis Peak/AUC
python scripts/compute_peak_auc.py

# 5. Análisis PCA
python scripts/compute_pca_analysis.py

# 6. Análisis de clustering (incluye generación de reporte)
python scripts/compute_clustering_analysis.py

# 7. Generación de figuras (incluye generación de reporte)
python scripts/generate_all_figures.py

# 8. (Opcional) Regenerar solo el reporte
python scripts/generate_comprehensive_report.py --force
```

**Nota**: Los pasos 6 y 7 generan automáticamente el reporte integral al finalizar. Si se ejecutan ambos, el reporte se regenerará dos veces (una vez después del clustering y otra después de las figuras). Para evitar regeneración duplicada, usar `--skip-report` en uno de los scripts.

**Ejecución rápida para depuración**:
```bash
# Omitir reporte en clustering (más rápido)
python scripts/compute_clustering_analysis.py --skip-report --n-bootstrap 100 --n-permutations 100

# Omitir reporte en figuras
python scripts/generate_all_figures.py --skip-report

# Regenerar solo reporte al final
python scripts/generate_comprehensive_report.py --force
```

---

## Verificación de resultados

Para verificar que el pipeline se ejecutó correctamente:

```bash
# Verificar existencia de archivos de resultados
ls results/tet/tet_preprocessed.csv
ls results/tet/descriptive/
ls results/tet/lme/
ls results/tet/peak_auc/
ls results/tet/pca/
ls results/tet/clustering/

# Verificar figuras generadas
ls results/tet/figures/

# Verificar reporte integral
ls docs/tet_comprehensive_results.md

# Verificar estado de actualización del reporte
python scripts/generate_comprehensive_report.py --check-only
```

Si algún archivo falta, ejecutar el script correspondiente del pipeline.
