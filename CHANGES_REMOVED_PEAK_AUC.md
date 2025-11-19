# Cambios: Eliminación del Análisis de Peak y AUC

**Fecha**: 2025-11-18  
**Objetivo**: Eliminar los análisis de time-to-peak y AUC, manteniendo únicamente los valores promedio de las dimensiones afectivas por ventana de 30 segundos.

## Resumen

Se eliminó completamente el análisis de métricas temporales (peak, time-to-peak, AUC) del pipeline de análisis TET. El pipeline ahora se enfoca exclusivamente en:

1. **Valores promedio** de las dimensiones afectivas por ventana de 30 segundos
2. **Análisis LME** de efectos temporales y de dosis
3. **Análisis PCA** de reducción dimensional
4. **Análisis de clustering** de estados experienciales

## Archivos Modificados

### 1. Pipeline Principal (`pipelines/run_tet_analysis.py`)

**Cambios**:
- Eliminada la etapa `peak_auc` de la lista de stages
- Eliminado el método `_run_peak_auc_analysis()`
- Actualizado `_validate_analysis_results()` para no requerir archivos peak_auc
- Actualizado `_validate_all_results()` para no requerir directorio peak_auc
- Actualizado argparse choices para eliminar 'peak_auc' de las opciones

**Stages actuales**:
1. preprocessing
2. descriptive
3. lme
4. pca
5. clustering
6. figures
7. report

### 2. Generación de Figuras (`scripts/generate_all_figures.py`)

**Cambios**:
- Eliminada la función `generate_peak_auc_figures()`
- Eliminada la llamada a generación de figuras peak_auc
- Actualizado argparse choices para eliminar 'peak-auc' de las opciones

**Figuras actuales**:
- time-series: Series temporales anotadas
- lme: Coeficientes LME (forest plots)
- pca: Scree plot y heatmap de loadings
- clustering: Centroides y cursos temporales
- glhmm: (futuro)

### 3. Reporte Comprehensivo (`scripts/generate_comprehensive_report.py`)

**Cambios**:
- Actualizado conteo de componentes de 5 a 4
- Eliminada referencia a "Peak/AUC Analysis" en loaded_components
- Actualizado missing_components para no incluir Peak/AUC

### 4. Sintetizador de Resultados (`scripts/tet/results_synthesizer.py`)

**Cambios principales**:
- Eliminado import de `extract_peak_auc_effects`
- Eliminado atributo `self.peak_auc_results`
- Eliminado método `_load_peak_auc()`
- Eliminado método `generate_peak_auc_section()`
- Actualizado `load_all_results()` para no cargar peak_auc
- Actualizado conteo de componentes de 5 a 4
- Actualizado `generate_executive_summary()` para pasar None en lugar de peak_auc_results
- Actualizado `generate_integration_section()` para pasar None en lugar de peak_auc_results
- Actualizado `generate_further_investigation()` para pasar None en lugar de peak_auc_results
- Eliminada llamada a `generate_peak_auc_section()` en `generate_report()`
- Renumeradas las secciones del reporte (PCA ahora es sección 3, clustering sección 4, etc.)

## Archivos NO Modificados (pero ya no se usan)

Los siguientes archivos permanecen en el repositorio pero ya no son llamados por el pipeline:

- `scripts/compute_peak_auc.py` - Script de cálculo de métricas peak/AUC
- `scripts/plot_peak_auc.py` - Script de visualización de comparaciones peak/AUC
- `scripts/tet/peak_auc_analyzer.py` - Módulo de análisis peak/AUC

**Nota**: Estos archivos pueden ser eliminados o movidos a `old_scripts/` si se desea limpiar el repositorio.

## Estructura de Resultados Actualizada

```
results/tet/
├── preprocessed/
│   └── tet_preprocessed.csv          # Datos preprocesados con valores promedio
├── descriptive/
│   ├── time_course_all_dimensions.csv
│   └── session_summaries.csv
├── lme/
│   ├── lme_results.csv
│   └── lme_contrasts.csv
├── pca/
│   ├── pca_variance_explained.csv
│   ├── pca_loadings.csv
│   └── pca_lme_results.csv
├── clustering/
│   ├── clustering_kmeans_assignments.csv
│   ├── clustering_kmeans_metrics.csv
│   └── clustering_kmeans_tests.csv
└── figures/
    ├── timeseries_all_dimensions.png
    ├── lme_coefficients_forest.png
    ├── pca_scree_plot.png
    ├── pca_loadings_heatmap.png
    └── clustering_*.png
```

**Directorio eliminado**: `results/tet/peak_auc/`

## Datos Analizados

El pipeline ahora trabaja exclusivamente con:

### Dimensiones Afectivas (valores promedio por ventana de 30s)
- Valence (valencia emocional)
- Arousal (activación)
- Anxiety (ansiedad)
- Unpleasantness (displacer)
- Intensity (intensidad)
- Complexity (complejidad)
- Mystical (experiencia mística)
- Insight (insight)
- Emotional_breakthrough (avance emocional)
- Challenging (desafiante)
- Grief (duelo)
- Joy (alegría)
- Love (amor)
- Surrender (entrega)
- Valence_index (índice de valencia compuesto)

### Variables de Diseño
- **Tiempo**: Ventanas de 30 segundos (0-9 minutos)
- **Dosis**: Baja (20mg) vs Alta (40mg)
- **Sujeto**: Efectos aleatorios por participante
- **Sesión**: Sesiones DMT vs Placebo

## Análisis Estadísticos Actuales

1. **Estadísticas Descriptivas**
   - Cursos temporales promedio por dosis
   - Resúmenes por sesión

2. **Modelos LME**
   - Efectos de tiempo, dosis, y su interacción
   - Efectos aleatorios por sujeto
   - Contrastes de dosis dentro de ventanas temporales

3. **Análisis PCA**
   - Reducción dimensional de las 15 dimensiones afectivas
   - Interpretación de componentes principales
   - Análisis LME de scores de componentes

4. **Análisis de Clustering**
   - Identificación de estados experienciales
   - Caracterización de centroides
   - Comparaciones de dosis

## Impacto en Documentación

Los siguientes documentos necesitarán actualización:

- `docs/PIPELINE.md` - Eliminar referencias a análisis peak/AUC
- `docs/methods_tet.md` - Actualizar sección de métodos
- `docs/tet_comprehensive_results.md` - Se regenerará automáticamente sin sección peak/AUC
- Especificaciones en `.kiro/specs/tet-analysis-pipeline/` - Actualizar requirements y design docs

## Comandos de Ejecución Actualizados

```bash
# Pipeline completo (sin peak_auc)
python pipelines/run_tet_analysis.py

# Stages específicos
python pipelines/run_tet_analysis.py --stages preprocessing descriptive lme

# Desde un stage específico
python pipelines/run_tet_analysis.py --from-stage pca

# Saltar stages
python pipelines/run_tet_analysis.py --skip-stages clustering
```

## Validación

Para verificar que los cambios funcionan correctamente:

```bash
# 1. Dry run del pipeline
python pipelines/run_tet_analysis.py --dry-run

# 2. Ejecutar solo preprocessing y descriptive
python pipelines/run_tet_analysis.py --stages preprocessing descriptive

# 3. Ejecutar pipeline completo
python pipelines/run_tet_analysis.py --verbose
```

## Próximos Pasos

1. Ejecutar el pipeline completo para verificar funcionamiento
2. Actualizar documentación en `docs/`
3. Considerar mover scripts peak_auc a `old_scripts/`
4. Actualizar tests si existen
5. Regenerar reporte comprehensivo

## Notas Técnicas

- Los datos preprocesados (`tet_preprocessed.csv`) ya contienen los valores promedio por ventana
- No se requieren cambios en el script de preprocesamiento
- Los análisis LME, PCA y clustering trabajan directamente con estos valores promedio
- La eliminación de peak/AUC simplifica el pipeline y reduce tiempo de ejecución
