# Implementación de Sombreado FDR en Series Temporales TET

## Resumen

Se implementó el sombreado estilo FDR (False Discovery Rate) en `timeseries_all_dimensions.png`, similar al usado en el análisis de ECG (`run_ecg_hr_analysis.py`).

## Cambios Realizados

### 1. Corrección Benjamini-Hochberg FDR

**Archivo**: `scripts/tet/time_series_visualizer.py`

Se agregó el método `_benjamini_hochberg_correction()` que implementa la corrección FDR:

```python
def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values (List[float]): List of raw p-values
        
    Returns:
        List[float]: FDR-corrected p-values
    """
```

### 2. Detección de Interacciones de Dosis con FDR

Se modificó `_identify_dose_interactions()` para:

- Realizar t-tests independientes en cada bin temporal (High vs Low)
- Aplicar corrección BH-FDR **por dimensión** a través de todos los bins temporales
- Retornar p-values crudos, p-values ajustados (p_FDR), y significancia

**Antes**: Usaba p < 0.05 sin corrección (muchos falsos positivos)

**Ahora**: Usa p_FDR < 0.05 con corrección BH (control de tasa de falsos descubrimientos)

### 3. Sombreado FDR en Gráficos

Se modificó `_plot_dimension()` para usar sombreado estilo ECG:

**Antes**: 
- Barras negras en la parte superior del gráfico
- Sombreado gris individual por cada bin (sin agrupar)

**Ahora**: 
- Regiones sombreadas grises (`color='0.85', alpha=0.35`) que cubren todo el eje Y
- **Agrupa bins consecutivos significativos en segmentos contiguos**
- Usa `axvspan()` para crear regiones sombreadas verticales continuas
- Similar al estilo usado en `run_ecg_hr_analysis.py`

**Importante**: Se eliminó el sombreado de efectos principales (DMT vs RS) para evitar confusión visual. Ahora solo se muestra el sombreado FDR para diferencias de dosis (High vs Low).

### 4. Reporte FDR

Se agregó el método `export_fdr_report()` que genera un archivo de texto con:

- Segmentos significativos por dimensión
- Rango temporal de cada segmento (en bins, segundos y minutos)
- p_FDR mínimo por segmento
- Estadísticas resumen (total bins significativos, min/median p_FDR)

**Archivo generado**: `timeseries_all_dimensions_fdr_report.txt`

### 5. Actualización de `export_figure()`

Se agregó parámetro `export_fdr_report=True` para generar automáticamente el reporte FDR junto con la figura.

## Comparación con Análisis ECG

| Aspecto | ECG (HR) | TET (Emociones) |
|---------|----------|-----------------|
| **Corrección FDR** | ✓ BH por condición | ✓ BH por dimensión |
| **Sombreado** | ✓ axvspan gris | ✓ axvspan gris |
| **Segmentos contiguos** | ✓ Agrupados | ✓ Agrupados |
| **Reporte FDR** | ✓ Archivo .txt | ✓ Archivo .txt |
| **Estilo visual** | Idéntico | Idéntico |

## Interpretación

### Sombreado Gris
- **Qué muestra**: Períodos donde High (40mg) difiere significativamente de Low (20mg) después de corrección FDR
- **Nivel de significancia**: p_FDR < 0.05 (control de tasa de falsos descubrimientos al 5%)
- **Ventaja**: Reduce falsos positivos comparado con p < 0.05 sin corrección

### Ejemplo de Resultados

Para **Emotional Intensity**:
- Segmento significativo: bins 17-168 (1.13-11.27 min)
- p_FDR mínimo: 0.0204
- Interpretación: La dosis alta (40mg) produce diferencias sostenidas en intensidad emocional durante ~10 minutos

Para **Interoception**:
- Sin segmentos significativos
- Interpretación: No hay diferencias robustas entre dosis en interocepción

## Archivos Modificados

1. `scripts/tet/time_series_visualizer.py`
   - Agregado: `_benjamini_hochberg_correction()`
   - Modificado: `_identify_dose_interactions()`
   - Modificado: `_plot_dimension()`
   - Agregado: `export_fdr_report()`
   - Modificado: `export_figure()`

2. `test/tet/test_fdr_shading.py` (nuevo)
   - Tests de corrección FDR
   - Tests de detección de interacciones
   - Tests de generación de figuras

## Uso

```python
from scripts.tet.time_series_visualizer import TETTimeSeriesVisualizer

# Cargar datos
data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
lme_results = pd.read_csv('results/tet/lme/lme_results.csv')
lme_contrasts = pd.read_csv('results/tet/lme/lme_contrasts.csv')
time_courses = pd.read_csv('results/tet/descriptive/time_course_all_dimensions.csv')

# Crear visualizador
viz = TETTimeSeriesVisualizer(data, lme_results, lme_contrasts, time_courses)

# Exportar figura con reporte FDR
viz.export_figure(
    'results/tet/figures/timeseries_all_dimensions.png',
    dpi=300,
    export_fdr_report=True  # Genera también el reporte FDR
)
```

## Validación

✓ Corrección FDR es monótona
✓ p_FDR ≥ p_raw (siempre)
✓ Segmentos contiguos agrupados correctamente
✓ Reporte FDR generado automáticamente
✓ Estilo visual consistente con análisis ECG

## Referencias

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
- Implementación basada en `pipelines/run_ecg_hr_analysis.py`
