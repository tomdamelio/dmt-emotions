
SEGUIR DESDE ACA ->
Implemente hasta el requirement 6 (del PCA)
Pero ahora me falta implementar del 7 al 10 (que deberia ser mas rapido).

A continuacion, el proximo prompt que deberia correr es:
"Quiero que crees el documento `design_req7.md` a partir de implementar lo mencionado como `### Requirement 7: Clustering of Experiential States` en el documento `requirements.md`"

Algunos de estos reqs incluyen hacer figuras, que me dejaran mas en claro como seguir a partir de esto. Capaz, dependiendo como van las figuras, podria considerar hacer analisis en concreto sobre las variables afectivas, e.g. arousal (`emotional_intensity`), and "valence" (mean between     'pleasantness' and minus (-) 'unpleasantness'?). Ver si la primera compoennte capta bien el arousal, y la segunda la valencia.

Una vez que esto quede mas claro, adaptar porque se crearon MUCHOS archivos que hay que eliminar. Sobre todos los que estan en `scripts`. De hecho, capaz pueda crear una tarea como requirement organizar esto, para asegurar que se haga correctamente.

---

Para despues ->

Una vez que todo esto este, recien pasar a la relacion entre estos PC1 (y PC2) con las medidas fisiologicas obtenidas



---

# Especificación técnica en Markdown


## 1) Análisis de TET

### 1.0 Datos esperados

* Tabla larga a 30 s por sesión:

  * `subject`, `session_id`, `state` ∈ {RS, DMT}, `dose` ∈ {Low, High}, `t_bin` ∈ {0,1,…}, `t_sec` = 30 * `t_bin`
  * 15 dimensiones en 0 a 10:
    `pleasantness`, `unpleasantness`, `emotional_intensity`, `elementary_imagery`, `complex_imagery`, `auditory`, `interoception`, `bliss`, `anxiety`, `entity`, `selfhood`, `disembodiment` (valores altos = desencarnación), `salience`, `temporality`, `general_intensity`
* Tabla de sujetos con edad, sexo, exposición previa y flags de calidad si aplica.

### 1.1 Preprocesamiento

* Validación de longitud: RS 20 muestras, DMT 40. Recorte a RS 0–10 min y DMT 0–20 min.
* Clampeo a [0, 10] con registro de ajustes.
* Variables de valencia: `valence_pos = pleasantness`, `valence_neg = unpleasantness`.
* Estandarización dentro de sujeto para cada dimensión con las cuatro sesiones combinadas. Conservar escala original para descripciones.
* Subíndices compuestos:

  * `affect_index_z` = media de z(`pleasantness`, `bliss`, `emotional_intensity`) menos z(`anxiety`, `unpleasantness`).
  * `imagery_index_z` = media de z(`elementary_imagery`, `complex_imagery`).
  * `self_index_z` = invertir z(`disembodiment`) y combinar con z(`selfhood`) para que valores altos indiquen mayor integración del yo (menos desencarnación, más sentido del yo).
* Documentar composición y signo de cada índice.

### 1.2 Descriptivos y figuras

* Curvas promedio ± SEM por `t_bin` en RS y DMT, Low vs High, por dimensión e índices. Exportar CSV por ventana.
* Resumen por sesión: pico, tiempo al pico, AUC 0–9 min, pendiente 0–2 min, pendiente 5–9 min.

### 1.3 Modelos LME por dimensión

* Outcome `Y_z` en 0–9 min para empatar con fisiología.
* Fijos: `State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c`.
* Aleatorio: intercepto por `subject`.
* `Time_c` centrado. Estimación ML.
* Objetivo principal: `State × Dose`. Reportar β, IC 95, p ajustado.
* Multiplicidad con BH-FDR por familia:

  * 15 dimensiones.
  * Subíndices compuestos.
  * Derivados resumen.
* Contrastes derivados:

  * En DMT: High vs Low y diferencias de pendiente.
  * En RS: High vs Low.
* Figuras: coefplots con IC 95 y p_FDR, trayectorias con sombreado de ventanas significativas por t pareados con BH-FDR sobre ventanas.

### 1.4 Picos y AUC

* Wilcoxon dentro de DMT para `peak`, `time_to_peak`, `AUC_0_9`. BH-FDR en 15 dimensiones e índices.
* Efecto r de Wilcoxon con IC por bootstrap.

### 1.5 Reducción de dimensionalidad y estados

* PCA en ventanas TET z dentro de sujeto. Conservar componentes hasta 70–80 por ciento de varianza. Modelar PC1 y PC2 con LME. Figuras: scree, loadings, coefplots.
* Clustering de ventanas: KMeans y GMM con k ∈ {2,3,4}. Selección por silhouette y BIC. Estabilidad por bootstrap. Curvas de probabilidad de cluster por tiempo, estado y dosis. Tests High vs Low en DMT por ventana con BH-FDR.

### 1.6 Robustez

* Repetir modelos con `embodiment` sin invertir.
* Excluir sujetos con TET incompletos o alta influencia (DFBETAS).
* Repetir con bins de 60 s.

---

## 2) Relación TET y fisiología

### 2.0 Datos conjuntos

* Merge por `subject`, `state`, `dose`, `t_bin` en 0–9 min.
* Fisiología z dentro de sujeto: `hr_z`, `smna_auc_z`, `rvt_z`, `arousal_pc1_z`.
* TET z e índices compuestos.

### 2.1 Asociaciones concurrentes

**Modelos mixtos por outcome fisiológico**

* Para cada `Phys_z ∈ {hr_z, smna_auc_z, rvt_z, arousal_pc1_z}` y cada dimensión TET:

  * `Phys_z ~ TET_dim_z + State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c` con intercepto aleatorio por `subject`.
  * BH-FDR por familia de 15 dimensiones y por outcome.
* Modelo multivariable parsimonioso:

  * LASSO con validación leave-one-subject-out sobre las 15 TET para preselección.
  * Refit LME con predictores retenidos y términos de diseño.
* Opcional: interacción `TET_dim_z × State` si hay hipótesis de cambio entre RS y DMT. Ajustar multiplicidad.

**Correlaciones parciales por sujeto**

* En DMT, correlación de Pearson entre `Phys_z` y `TET_dim_z` controlando `Time_c` por residuales.
* Meta análisis de r transformadas con Fisher z. BH-FDR sobre dimensiones. Forest plot con IC 95.

### 2.2 Desfase temporal

**CCF por sujeto**

* Lags de −6 a +6 ventanas. Remover tendencia lineal o usar residuales AR(1).
* Pico de CCF y lag óptimo por dimensión y outcome. Meta análisis y p por permutación circular dentro de serie. BH-FDR global.

**LME con lag**

* Para lags {0,1,2,3}:

  * `Phys_z[t] ~ TET_dim_z[t−lag] + State + Dose + Time_c + (1 | subject)` en DMT.
* Curvas de β vs lag con IC por bootstrap. BH-FDR por dimensión en los lags probados.

### 2.3 Modelos multivariados

* PLS para predecir `arousal_pc1_z` con 15 TET z en DMT. Validación leave-one-subject-out. Reportar RMSE, R², pesos.
* CCA entre bloque TET y bloque fisiología `{hr_z, smna_auc_z, rvt_z}`. Correlaciones canónicas, pesos, IC por bootstrap. BH-FDR si hay múltiples correlaciones.
* LME con índices compuestos `affect_index_z`, `imagery_index_z`, `self_index_z` como predictores concurrentes y con lag 1–3.

### 2.4 Estados latentes compartidos

* Usar probabilidades de cluster TET como predictores:

  * `Phys_z ~ Cluster_prob_k + Time_c + (1 | subject)` en DMT.
* Comparar High vs Low en probabilidad de cluster por ventana con BH-FDR y superponer con ventanas significativas en fisiología.

### 2.5 Robustez

* Excluir primeras 2 ventanas posinhalación.
* Winsorización al 2 por ciento en TET y fisiología.
* VIF en modelos multivariables y PCA previa si hay colinealidad.
* Sensibilidad a la estandarización usando rangos normalizados por sujeto.

---

## 3) Salidas

### 3.1 Tablas

* **T1** LME por dimensión TET con β, IC 95, p_FDR para `State`, `Dose`, `State × Dose`, `State × Time`, `Dose × Time`.
* **T2** Picos, tiempo al pico, AUC, pendientes. Tests High vs Low en DMT con p_FDR.
* **T3** Asociaciones concurrentes TET → fisiología por outcome y dimensión con p_FDR.
* **T4** Lags óptimos por dimensión y outcome con p por permutación y p_FDR.
* **T5** PLS y CCA con desempeño, varianza explicada, correlaciones canónicas, pesos e IC.

### 3.2 Figuras

* **F-TET-1** Curvas TET promedio ± SEM por estado y dosis con sombreado de ventanas significativas.
* **F-TET-2** Coefplots LME por dimensión con p_FDR.
* **F-TET-3** Boxplots de picos y AUC con Wilcoxon y p_FDR.
* **F-TET-4** PCA TET: scree, loadings, coefplots de PC1 y PC2.
* **F-TET-5** Probabilidad temporal de clusters por dosis.
* **F-LINK-1** Heatmap de β concurrentes TET → fisiología por outcome con celdas significativas.
* **F-LINK-2** β vs lag por outcome para dimensiones clave.
* **F-LINK-3** PLS: pesos y desempeño.
* **F-LINK-4** CCA: correlaciones canónicas e IC, pesos por bloque.

---

## 4) Implementación en Python

### 4.1 Paquetes

`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `pingouin`, `matplotlib`.
Semilla global 22.

### 4.2 Utilidades

* `standardize_within_subject(df, cols, group_cols=['subject'])`
* `fit_lme(formula_like, data, groups)`
* `fdrcorrect_series(series, alpha=0.05)`
* `paired_t_by_time(df, value, group='subject', cond='dose', time='t_bin', state='DMT')` con BH-FDR
* `ccf_by_subject(x, y, lags, detrend='linear', ar1=True, n_perm=2000)`

### 4.3 Carpeta de resultados

* `results/tet/`

  * `descriptives/*.csv`
  * `lme/*.csv`
  * `figures/*.png`
* `results/tet_physio_link/`

  * `concurrent/*.csv`
  * `lags/*.csv`
  * `multivariate/*.csv`
  * `figures/*.png`

### 4.4 Registro y QC

* JSON por corrida con:

  * N de sujetos, N de ventanas, varianza explicada, hiperparámetros, versión de datos, fecha.
* Listas de exclusión y motivos.



Para despues
------------

[ ] Hacer una figura de la metodologia, usando al imagen de Evan para crear en chatgpt u boceto de la toma de datos. O si no hacerla usando otra app. Usar los colores que defini mas arriba para las 3 señales fisiologicas para marcar en esa figura las 3 señales registradas

--
-

## 5. State Clustering Visualization (Task 25 - Completed)

### 5.1 Overview

Implemented comprehensive visualization tools for clustering and state modelling results through the `TETStateVisualization` class and `plot_state_results.py` script.

### 5.2 Visualization Capabilities

#### 5.2.1 KMeans Centroid Profiles (Fig. 3.5-like)
* Computes centroid coordinates in z-scored dimensions
* Normalizes each centroid by maximum absolute value to [0, 1]
* Generates horizontal bar plots showing relative dimension contributions per cluster
* Allows comparison of characteristic profiles across clusters

#### 5.2.2 Time-Course Cluster Probability Plots (Fig. 3.6-like)
* Visualizes evolution of cluster probabilities over time
* Computes mean ± SEM across subjects for High/Low dose in DMT
* Optional inclusion of RS condition for comparison
* Separate panels for each state × dose combination

#### 5.2.3 GLHMM State Time-Course Plots
* Plots GLHMM posterior probabilities (gamma) over time
* Supports filtering by experimental state (RS vs DMT)
* Allows subsetting to specific GLHMM states (e.g., S=2 solution)
* Mean ± SEM curves with shaded error bands

#### 5.2.4 KMeans-GLHMM Correspondence Analysis
* Computes contingency table between KMeans clusters and GLHMM states
* Normalizes to conditional probabilities: P(GLHMM state | KMeans cluster)
* Generates heatmap visualization
* Exports contingency table as CSV for further analysis

### 5.3 Usage

#### Basic Usage
```bash
# Generate all visualization figures with default settings
python scripts/plot_state_results.py

# Specify custom paths
python scripts/plot_state_results.py \
    --input-dir results/tet/clustering \
    --output-dir results/tet/figures

# Include RS condition and plot k=2 clusters
python scripts/plot_state_results.py \
    --k 2 \
    --include-rs

# Plot subset of GLHMM states (S=2 solution)
python scripts/plot_state_results.py \
    --subset-glhmm-states 0 1

# High-resolution output
python scripts/plot_state_results.py \
    --dpi 600 \
    --verbose
```

#### Programmatic Usage
```python
from tet.state_visualization import TETStateVisualization
import pandas as pd

# Load data
data = pd.read_csv('results/tet/tet_preprocessed.csv')
kmeans = pd.read_csv('results/tet/clustering/clustering_kmeans_assignments.csv')
glhmm_viterbi = pd.read_csv('results/tet/clustering/clustering_glhmm_viterbi.csv')
glhmm_probs = pd.read_csv('results/tet/clustering/clustering_glhmm_probabilities.csv')

# Initialize visualizer
viz = TETStateVisualization(
    data=data,
    kmeans_assignments=kmeans,
    glhmm_viterbi=glhmm_viterbi,
    glhmm_probabilities=glhmm_probs
)

# Generate individual plots
viz.plot_kmeans_centroid_profiles(k=2, output_dir='results/tet/figures')
viz.plot_kmeans_cluster_timecourses(k=2, include_rs=True, output_dir='results/tet/figures')
viz.plot_glhmm_state_timecourses(output_dir='results/tet/figures')
viz.plot_kmeans_glhmm_crosswalk(k=2, output_dir='results/tet/figures')
```

### 5.4 Output Files

Generated figures are saved to `results/tet/figures/`:
* `clustering_kmeans_centroids_k2.png` - Centroid profiles
* `clustering_kmeans_prob_timecourses_with_rs.png` - Cluster time courses (with RS)
* `clustering_kmeans_prob_timecourses_dmt_only.png` - Cluster time courses (DMT only)
* `glhmm_state_prob_timecourses.png` - GLHMM state time courses
* `kmeans_glhmm_crosswalk.png` - Correspondence heatmap
* `kmeans_glhmm_crosswalk.csv` - Contingency table (counts and probabilities)

### 5.5 Implementation Details

#### Normalization Strategy
Centroid profiles are normalized by maximum absolute value within each cluster to enable comparison of relative dimension importance independent of absolute intensity. This highlights which dimensions are most characteristic of each cluster.

#### Soft Probability Computation
KMeans soft probabilities are computed using normalized inverse distances to cluster centers:
```
prob_k = (1/dist_k) / sum(1/dist_j for all j)
```

This provides a probabilistic interpretation where observations closer to a cluster center have higher probability of belonging to that cluster.

#### Correspondence Analysis
The KMeans-GLHMM correspondence analysis helps understand whether the two methods identify similar experiential states or capture different aspects of the data. High correspondence (diagonal heatmap) suggests agreement, while off-diagonal patterns indicate complementary information.

### 5.6 Integration with Analysis Pipeline

The visualization tools integrate seamlessly with the clustering analysis pipeline:

1. Run clustering analysis: `python scripts/compute_clustering_analysis.py`
2. Generate visualizations: `python scripts/plot_state_results.py`
3. Inspect results: `python scripts/inspect_clustering_results.py`

All scripts use consistent file naming conventions and directory structures for easy workflow integration.

### 5.7 Notes

* All plots use publication-ready defaults (300 DPI, appropriate figure sizes)
* Color schemes are consistent across plots (Set2 for clusters, Set1 for states)
* Error bands represent standard error of the mean (SEM) across subjects
* Time is displayed in minutes for easier interpretation
* Missing data is handled gracefully with informative warnings

