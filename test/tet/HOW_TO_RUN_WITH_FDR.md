# Cómo Ejecutar el Pipeline TET con Sombreado FDR

## Resumen

El sombreado FDR ya está integrado en el pipeline TET. Cuando ejecutes el pipeline, automáticamente se generará `timeseries_all_dimensions.png` con el nuevo estilo de sombreado FDR (similar al análisis de ECG).

## Ejecución Completa del Pipeline

```bash
# Ejecutar pipeline completo (todas las etapas)
python pipelines/run_tet_analysis.py
```

Esto ejecutará todas las etapas en orden:
1. preprocessing
2. descriptive
3. lme
4. pca
5. ica
6. physio_correlation
7. clustering
8. **figures** ← Aquí se genera timeseries_all_dimensions.png con FDR
9. report

## Ejecutar Solo la Generación de Figuras

Si ya tienes los análisis previos y solo quieres regenerar las figuras:

```bash
# Solo regenerar figuras
python pipelines/run_tet_analysis.py --stages figures
```

O directamente:

```bash
# Generar todas las figuras
python scripts/generate_all_figures.py

# Solo la figura de series temporales
python scripts/generate_all_figures.py --figures time-series
```

## Archivos Generados

Cuando ejecutes el pipeline, se generarán:

### Figura Principal
- **`results/tet/figures/timeseries_all_dimensions.png`**
  - Series temporales con sombreado FDR
  - Regiones grises indican diferencias significativas High vs Low (p_FDR < 0.05)
  - Corrección Benjamini-Hochberg aplicada por dimensión

### Reporte FDR
- **`results/tet/figures/timeseries_all_dimensions_fdr_report.txt`**
  - Segmentos significativos por dimensión
  - Rangos temporales (bins, segundos, minutos)
  - Estadísticas p_FDR (mínimo, mediana)

## Flujo de Ejecución

```
pipelines/run_tet_analysis.py
    └─> scripts/generate_all_figures.py
        └─> scripts/plot_time_series.py
            └─> scripts/tet/time_series_visualizer.py
                ├─> _benjamini_hochberg_correction()  [NUEVO]
                ├─> _identify_dose_interactions()     [MODIFICADO con FDR]
                ├─> _plot_dimension()                 [MODIFICADO con sombreado]
                └─> export_fdr_report()               [NUEVO]
```

## Verificación

Para verificar que el sombreado FDR funciona correctamente:

```bash
# Ejecutar test
python test/tet/test_fdr_shading.py
```

Esto generará:
- `test/tet/test_fdr_shading_output.png` (figura de prueba)
- `test/tet/test_fdr_shading_output_fdr_report.txt` (reporte de prueba)

## Ejemplo de Salida del Reporte FDR

```
FDR COMPARISON: High (40mg) vs Low (20mg) Dose Effects Over Time
Benjamini-Hochberg FDR correction applied per dimension across all time bins
Alpha = 0.05

DIMENSION: Emotional Intensity
------------------------------------------------------------
  Significant segments (count=1):
    - Bins 17-168: 1.13-11.27 min (68-676s), min p_FDR=0.0204
  Total significant bins: 152
  Min p_FDR: 0.020353
  Median p_FDR: 0.022845

DIMENSION: Interoception
------------------------------------------------------------
  No significant dose differences (p_FDR < 0.05)
```

## Diferencias Visuales

### Antes (Barras Negras)
- Barras negras horizontales en la parte superior del gráfico
- Sin corrección FDR (p < 0.05 sin ajuste)
- Difícil de ver cuando los datos son altos

### Ahora (Sombreado FDR)
- Regiones sombreadas grises que cubren todo el eje Y
- Corrección Benjamini-Hochberg FDR (p_FDR < 0.05)
- Segmentos contiguos agrupados automáticamente
- Estilo idéntico al análisis de ECG

## Parámetros de Sombreado

Los parámetros son idénticos a los usados en el análisis de ECG:

```python
# Color y transparencia
color='0.85'    # Gris claro
alpha=0.35      # 35% de opacidad
zorder=0        # Detrás de las líneas de datos
```

## Interpretación

### Sombreado Gris
- **Qué muestra**: Períodos donde High (40mg) difiere significativamente de Low (20mg)
- **Corrección**: Benjamini-Hochberg FDR aplicada por dimensión
- **Nivel**: p_FDR < 0.05 (control de tasa de falsos descubrimientos al 5%)

### Sin Sombreado
- No hay diferencias robustas entre dosis en ese período
- O las diferencias no sobreviven la corrección FDR

## Troubleshooting

### Error: "Missing files"
```bash
# Ejecutar etapas previas primero
python pipelines/run_tet_analysis.py --stages preprocessing descriptive lme
```

### Error: "Module not found: tet.time_series_visualizer"
- El import ya fue corregido a `scripts.tet.time_series_visualizer`
- Si persiste, verificar que `scripts/tet/` esté en el path

### Figura sin sombreado
- Verificar que no haya diferencias significativas (revisar reporte FDR)
- Si todas las dimensiones muestran "No significant dose differences", es correcto

## Referencias

- Implementación: `scripts/tet/time_series_visualizer.py`
- Tests: `test/tet/test_fdr_shading.py`
- Documentación: `test/tet/FDR_SHADING_IMPLEMENTATION.md`
- Comparación visual: `test/tet/FDR_VISUAL_COMPARISON.md`
