## Pipeline de análisis y generación de resultados

Este documento describe, en orden de ejecución, los scripts necesarios para reproducir los análisis y figuras del proyecto, desde el procesamiento de datos hasta la creación de las figuras finales. Para cada paso se indica: entradas esperadas, qué hace el script y qué produce como salida.

Notas generales
- Los datos de fisiología se leen mayormente desde archivos BrainVision (`.vhdr`) organizados por sujeto y sesión en la carpeta `../data/original/physiology/` con subcarpetas `DMT_1`, `DMT_2`, `Reposo_1`, `Reposo_2` y `SXX/` para cada sujeto.
- Los autorreportes se cargan desde archivos `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/` (cada archivo contiene la matriz `dimensions` con 15 dimensiones, 300 puntos por sujeto, y metadatos en el nombre para identificar sujeto y condición).
- Varios scripts guardan CSVs intermedios en `../data/old/Preprocesado/<MEDIDA>/` para facilitar análisis posteriores.

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

### 📁 **Datos procesados** (`../data/old/`)

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
Carpeta preparada para procesamientos futuros siguiendo estándar BIDS-Derivatives (actualmente vacía).


### 1) Preprocesamiento de señales fisiológicas (por modalidad)

#### 1.1) SCL (EDA tónica)
- Script: `scripts/Ploteo EDA promedio.py`
- Input:
  - BrainVision `.vhdr` por sujeto en: `../data/original/physiology/DMT_1`, `DMT_2`, `Reposo_1`, `Reposo_2` (canal `GSR`).
  - Tabla de dosis embebida en el script para mapear qué sesión es alta/baja por sujeto.
- Proceso:
  - Procesa EDA con NeuroKit (`nk.eda_process`), separa SCL/SCR, resta baseline de reposo correspondiente por sujeto.
  - Agrega series por sujeto y calcula promedio temporal con banda de error (EE del promedio).
  - Test de Wilcoxon punto a punto (alta vs baja) y opción de FDR para sombrear regiones significativas.
- Output (CSV):
  - `../data/old/Preprocesado/SCL/SCL_dmt_alta.csv`
  - `../data/old/Preprocesado/SCL/SCL_dmt_baja.csv`
  - `../data/old/Preprocesado/SCL/SCL_tiempo_dmt_alta.csv`
  - `../data/old/Preprocesado/SCL/SCL_tiempo_dmt_baja.csv`
- Output (Figuras): curvas promedio ± error; p-valores/zonas significativas.

#### 1.2) HR (frecuencia cardíaca desde ECG)
- Script: `scripts/Ploteo HR promedio.py`
- Input:
  - BrainVision `.vhdr` por sujeto en: `../data/original/physiology/DMT_1`, `DMT_2`, `Reposo_1`, `Reposo_2` (canal `ECG`).
  - Tabla de dosis embebida.
- Proceso:
  - Procesa ECG con NeuroKit (`nk.ecg_process`) para obtener `ECG_Rate`.
  - Resta baseline de reposo por sujeto, agrega y calcula promedio ± EE.
  - Test de Wilcoxon y FDR para marcar regiones significativas.
- Output (CSV):
  - `../data/old/Preprocesado/HR/HR_dmt_alta.csv`
  - `../data/old/Preprocesado/HR/HR_dmt_baja.csv`
  - `../data/old/Preprocesado/HR/HR_tiempo_dmt_alta.csv`
  - `../data/old/Preprocesado/HR/HR_tiempo_dmt_baja.csv`
- Output (Figuras): curvas HR promedio ± error; p-valores/zonas significativas.

#### 1.3) SCR (frecuencia de respuestas fásicas)
- Script: `scripts/Ploteo SCR todos.py`
- Input:
  - BrainVision `.vhdr` por sujeto en: `../data/original/physiology/DMT_1`, `DMT_2` (canal `GSR`).
- Proceso:
  - Procesa EDA con método fásico basado en cvxEDA (implementación incluida en el script).
  - Calcula cantidad de picos SCR en ventanas deslizantes (tamaño configurable, por defecto 120 s) y los interpola.
  - Agrega por sujeto, promedia y calcula EE; Wilcoxon y zonas significativas (por índices de ventana).
- Output (CSV):
  - `../data/old/Preprocesado/SCR/SCR_dmt_alta.csv`
  - `../data/old/Preprocesado/SCR/SCR_dmt_baja.csv`
  - `../data/old/Preprocesado/SCR/SCR_tiempo_dmt_alta.csv`
  - `../data/old/Preprocesado/SCR/SCR_tiempo_dmt_baja.csv`
- Output (Figuras): curvas de cantidad de picos ± error; p-valores/zonas significativas.

#### 1.4) Respiración (variabilidad/amplitud) -> Entiendo que respiracion no funciona, pero revisar.
- Script: `scripts/Ploteo Resp promedio.py`
- Input:
  - BrainVision `.vhdr` por sujeto en: `../data/original/physiology/DMT_1`, `DMT_2`, `Reposo_1`, `Reposo_2` (canal `RESP`).
- Proceso:
  - Procesa respiración con NeuroKit (`nk.rsp_process`).
  - Define ventanas deslizantes y calcula desviación estándar de `RSP_Clean` por ventana; resta baseline de reposo.
  - Agrega, promedia ± EE; Wilcoxon y zonas significativas.
- Output: Figuras de promedio ± error y p-valores. (CSV no persistido por defecto; se puede adaptar siguiendo el patrón de HR/SCL/SCR).

Apoyo/variantes
- `scripts/Eda_process con cvxEDA.py` y `scripts/Adaptando nk.eda_plot con cvxeda.py`: funciones auxiliares para usar cvxEDA con NeuroKit y ploteos; útiles para inspecciones o variantes de procesamiento.
- `scripts/Estudio con NeuroKit.py` y `scripts/Estudio multiple con NeuroKit.py`: ejemplos de lectura y procesamiento (útiles como referencia, no forman parte del pipeline masivo).

---

### 2) Preparación de autorreportes para clustering/PCA

#### 2.1) Armado del dataset de autorreportes (sin reposo)
- Script: `scripts/Armado de Dataframe para clusterizar.py`
- Input:
  - `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/` con la matriz `dimensions` (15 columnas: Pleasantness, Unpleasantness, Emotional_Intensity, …, General_Intensity).
  - Tabla de dosis (embebida) para clasificar alta/baja y distinguir DMT vs RS.
- Proceso:
  - Recorre archivos, parsea sujeto/condición desde el nombre, separa por dosis y condición.
  - Concatena y construye el dataset final (en el script hay una variante que excluye reposo).
- Output (CSV):
  - `../data/old/Data Cluster/Datos_reportes_para_clusterizar_sin_reposo.csv`
 - Nota (interpretación del CSV): el archivo concatena primero todas las series de DMT alta y luego todas las de DMT baja (no incluye reposo). Si hay Nsuj sujetos y cada uno aporta 300 puntos:
   - Filas 0 a (300×Nsuj − 1), es decir Filas 1 a 5400: High Dose (DMT alta)
   - Filas 300×Nsuj a (600×Nsuj − 1), es decir Filas 5401 a 10800: Low Dose (DMT baja)
   - Opcional: si se prefiere una etiqueta explícita, añadir en el script una columna `Source` con `High Dose`/`Low Dose` antes de concatenar.

---

### 3) PCA de autorreportes y figuras asociadas

#### 3.1) PCA final (alta vs baja) y curvas PC1/PC2 con estadística
- Script: `scripts/Final PCA.py`
- Input: `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/` (mismas dimensiones que arriba).
- Proceso:
  - Concatena DMT alta/baja (o condiciones seleccionadas), estandariza y aplica PCA (por defecto 2 componentes).
  - Obtiene loadings y reporta top cargas por PC; separa sujetos por condición y plotea trayectorias PC1 vs PC2.
  - Guarda PC1/PC2 por sujeto, calcula promedios ± EE; Wilcoxon punto a punto y FDR; sombreado de regiones.
- Output (CSV): `../data/old/Preprocesado/PCA/`
  - `PCA_pc1_alta.csv`, `PCA_pc1_baja.csv`, `PCA_pc2_alta.csv`, `PCA_pc2_baja.csv`
- Output (Figuras):
  - Trayectorias PC1–PC2 por sujeto/condición, curvas promedio PC1/PC2 ± EE y regiones significativas.

#### 3.2) PCA sobre todas las condiciones y visualizaciones
- Script: `scripts/PCA todos.py`
- Input: `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/` (alta, baja, RS alta, RS baja).
- Proceso: PCA global (2 componentes), separación por condición, ploteo de trayectorias y curvas.
- Output: Figuras (y CSVs de PC opcionales siguiendo el patrón de `Final PCA.py`).

---

### 4) Correlaciones entre autorreportes y fisiología

#### 4.1) Correlación lineal con SCL/HR
- Script: `scripts/Correlacion lineal.py`
- Input:
  - CSVs de `../data/old/Preprocesado/<MEDIDA>/` según `medida = 'SCL'` o `'HR'` (ej. `SCL_dmt_alta.csv`, `SCL_dmt_baja.csv`, sus tiempos).
  - `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/` (autorreportes). 
- Proceso:
  - Alinea temporalmente (upsample/interpola reportes y/o downsamplea señales) y calcula correlaciones Pearson/Spearman por dimensión y sujeto.
  - Reorganiza DataFrames para boxplots (alta vs baja) y ejecuta Wilcoxon por dimensión.
- Output: Figuras (boxplots Pearson/Spearman por dimensión y condición); tabla impresa de p-valores.

#### 4.2) Correlación lineal específica para SCR
- Script: `scripts/Correlacion lineal SCR.py`
- Input:
  - CSVs de `../data/old/Preprocesado/SCR/` (dosis alta/baja y tiempos).
  - `.mat` en `../data/original/reports/resampled/` o `../data/original/reports/resampled - con el s12/`.
- Proceso: downsamplea reportes para empatar con SCR agregado, calcula Pearson/Spearman por dimensión/sujeto, boxplots y Wilcoxon por dimensión.
- Output: Figuras (boxplots) y tabla impresa de p-valores.

---

### 5) Visualización agregada y clustering de autorreportes

#### 5.1) Visualización y KMeans
- Script: `scripts/Visualizacion Reportes y Clustering.py`
- Input:
  - `../data/old/Data Cluster/Datos_reportes_para_clusterizar_sin_reposo.csv` (generado por el script de armado de dataframe).
- Proceso:
  - Plotea promedios y bandas de error por dimensión (alta vs baja).
  - Ejecuta KMeans (k=2 por defecto), grafica probabilidades de estados por condición y muestra centroides por dimensión.
  - Calcula Silhouette Score y curva Elbow para explorar cantidad de clusters.
- Output: Figuras de promedios, estados medios, centroides y diagnóstico de clusters.

---

## Orden recomendado de ejecución (de principio a fin)
1. Preprocesar fisiología por modalidad para generar CSVs en `../data/old/Preprocesado/`:
   - `scripts/Ploteo EDA promedio.py` (SCL)
   - `scripts/Ploteo HR promedio.py` (HR)
   - `scripts/Ploteo SCR todos.py` (SCR)
   - `scripts/Ploteo Resp promedio.py` (Resp; figuras)
2. Preparar dataset de autorreportes:
   - `scripts/Armado de Dataframe para clusterizar.py`
3. PCA de autorreportes y figuras:
   - `scripts/Final PCA.py` (y/o `scripts/PCA todos.py`)
4. Correlaciones reportes–fisiología:
   - `scripts/Correlacion lineal.py` (SCL/HR)
   - `scripts/Correlacion lineal SCR.py` (SCR)
5. Visualización y clustering:
   - `scripts/Visualizacion Reportes y Clustering.py`

---

## Dependencias principales
- Python 3.x, `mne`, `neurokit2`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `joblib`.

## Ajustes a verificar antes de correr
- Listas de sujetos (`carpetas`) y tabla `dosis` embebida.
- Parámetros de ventanas deslizantes (e.g., 120 s para SCR, 100 s para Resp) y down/upsampling.


