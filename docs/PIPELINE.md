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

**Contenido de archivos .mat**: Matriz `dimensions` de 15×300 (15 dimensiones fenomenológicas × 300 puntos temporales):
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
12. **Disembodiment** - Experiencia de no identificarse con el propio cuerpo
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
  
- **Criterio de validación**:
  - Cada sujeto debe tener los 4 archivos (DMT_1, DMT_2, Reposo_1, Reposo_2) clasificados como 'good' o 'acceptable' para ser incluido
  - Sujetos con señales 'poor' o 'bad' son excluidos para esa modalidad
  
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

### 5) Análisis de ECG y Respiración (pendientes)

Los siguientes análisis están planificados pero aún no implementados:

- **ECG (Heart Rate / HRV)**:
  - Análisis de frecuencia cardíaca y variabilidad
  - Sujetos validados: 15 (ver `SUJETOS_VALIDADOS_ECG` en `config.py`)
  - Input: CSVs de `../data/derivatives/phys/ecg/`
  
- **Respiración**:
  - Análisis de patrones respiratorios y variabilidad
  - Sujetos validados: 12 (ver `SUJETOS_VALIDADOS_RESP` en `config.py`)
  - Input: CSVs de `../data/derivatives/phys/resp/`

---

## Resumen del orden de ejecución

```bash
# 1. Preprocesamiento de señales fisiológicas
python scripts/preprocess_phys.py

# 2. Validación de datos (ya realizada, sujetos validados en config.py)

# 3. Análisis de EDA (los tres scripts pueden correrse en paralelo)
python scripts/run_eda_scl_analysis.py   # SCL/EDL (tónico)
python scripts/run_eda_smna_analysis.py  # SMNA (fásico continuo)
python scripts/run_eda_scr_analysis.py   # SCR (eventos discretos)

# 4. Generación de paneles compuestos
python scripts/generate_eda_figures.py

# 5. Análisis de ECG y RESP (pendientes)
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


