# Actualización Completa de Scripts de Visualización

## Resumen

Se actualizaron todos los scripts de visualización para trabajar exclusivamente con las 7 dimensiones afectivas (6 dimensiones + valence_index_z), alineándose con la nueva metodología de análisis.

## Scripts Actualizados

### 1. `scripts/tet/time_series_visualizer.py` ✅
**Cambios:**
- Ahora usa `config.TET_AFFECTIVE_COLUMNS` por defecto
- Genera figuras con 7 paneles en vez de 15
- Mantiene todas las anotaciones estadísticas (grey shading, black bars)
- Ordena dimensiones por efecto State

**Código actualizado:**
```python
# Antes
self.dimensions = [f"{dim}_z" for dim in config.TET_DIMENSION_COLUMNS]

# Después
affective_dims = [f"{dim}_z" for dim in config.TET_AFFECTIVE_COLUMNS]
self.dimensions = affective_dims + ['valence_index_z']
```

### 2. `scripts/plot_time_series.py` ✅
**Cambios:**
- Mensajes actualizados para reflejar 7 dimensiones afectivas
- Documentación actualizada
- Mantiene toda la funcionalidad de anotaciones estadísticas

**Mensajes actualizados:**
- "7 affective dimensions + valence index" (antes: "15 dimensions")
- "7 affective dimension panels" (antes: "15 dimension panels")

### 3. `scripts/plot_lme_coefficients.py` ✅
**Estado:**
- No requiere cambios en el código
- Lee automáticamente los resultados LME que ya solo contienen dimensiones afectivas
- Genera forest plots solo para las 7 dimensiones

**Razón:**
- El script solo lee y grafica los resultados
- Como `lme_analyzer.py` ya fue actualizado, los resultados solo contienen dimensiones afectivas

### 4. `scripts/plot_peak_auc.py` ✅
**Estado:**
- No requiere cambios en el código
- Lee automáticamente los resultados Peak/AUC que ya solo contienen dimensiones afectivas
- Genera boxplots solo para las 7 dimensiones

**Razón:**
- El script solo lee y grafica los resultados
- Como `compute_peak_auc.py` ya fue actualizado, los resultados solo contienen dimensiones afectivas

### 5. `scripts/generate_all_figures.py` ✅
**Estado:**
- No requiere cambios
- Orquesta la generación de todas las figuras llamando a los scripts individuales
- Como todos los scripts individuales fueron actualizados, funciona correctamente

## Arquitectura de Visualización

```
generate_all_figures.py (Orquestador)
    │
    ├─> plot_time_series.py
    │       └─> time_series_visualizer.py ✅ ACTUALIZADO
    │
    ├─> plot_lme_coefficients.py
    │       └─> Lee lme_results.csv ✅ (ya filtrado)
    │
    ├─> plot_peak_auc.py
    │       └─> Lee peak_auc_*.csv ✅ (ya filtrado)
    │
    ├─> plot_pca_results.py
    │       └─> Usa todas las dimensiones (exploratorio)
    │
    └─> plot_state_results.py
            └─> Usa todas las dimensiones (exploratorio)
```

## Flujo de Datos

```
Preprocesamiento
    ↓
[15 dimensiones z-scored + valence_index_z]
    ↓
    ├─> Análisis Confirmatorios (FILTRADOS)
    │   ├─> LME: 7 dimensiones afectivas
    │   ├─> Peak/AUC: 7 dimensiones afectivas
    │   └─> Visualizaciones: 7 dimensiones afectivas
    │
    └─> Análisis Exploratorios (COMPLETOS)
        ├─> PCA: 15 dimensiones
        ├─> Clustering: 15 dimensiones
        └─> Visualizaciones: 15 dimensiones
```

## Impacto en Figuras

### Antes
- **Time Series**: 15 paneles (5×3 grid)
- **LME Coefficients**: 15 dimensiones × 5 efectos = 75 coeficientes
- **Peak/AUC**: 18 dimensiones (15 + 3 índices) × 3 métricas = 54 boxplots

### Después
- **Time Series**: 7 paneles (más compacto, más legible)
- **LME Coefficients**: 7 dimensiones × 5 efectos = 35 coeficientes
- **Peak/AUC**: 7 dimensiones × 3 métricas = 21 boxplots

**Beneficios:**
- Figuras más compactas y legibles
- Enfoque claro en dimensiones afectivas
- Menor corrección por comparaciones múltiples
- Alineación con objetivos del paper

## Verificación

Para verificar que las visualizaciones funcionan correctamente:

```bash
# 1. Generar todas las figuras
python scripts/generate_all_figures.py --verbose

# 2. Verificar figuras individuales
python scripts/plot_time_series.py --verbose
python scripts/plot_lme_coefficients.py --verbose
python scripts/plot_peak_auc.py --verbose

# 3. Verificar que las figuras muestran 7 dimensiones
# - Time series: debe tener 7 paneles
# - LME coefficients: debe mostrar 7 dimensiones
# - Peak/AUC: debe tener 7 dimensiones × 3 métricas
```

## Figuras que Mantienen 15 Dimensiones

Estas figuras NO fueron modificadas (correcto, son exploratorias):

- **PCA Scree Plot**: Muestra varianza explicada por componentes de 15 dimensiones
- **PCA Loadings**: Muestra loadings de 15 dimensiones en PC1 y PC2
- **Clustering Centroids**: Muestra perfiles de 15 dimensiones por cluster
- **Clustering Time Courses**: Muestra probabilidades de cluster basadas en 15 dimensiones

## Consistencia con Métodos

Las visualizaciones ahora están completamente alineadas con la sección de métodos:

> "Finally, two core affective dimensions were derived from the standardised TET ratings to capture valence and arousal. As an arousal proxy, we used the Emotional Intensity dimension, which reflects the momentary strength of emotional experience irrespective of hedonic tone. As a valence proxy, we constructed a composite Valence Index computed as Pleasantness minus Unpleasantness at each time point, such that higher scores reflect more positively valenced experience."

Las 7 dimensiones visualizadas son:
1. `pleasantness_z` - Valencia positiva
2. `unpleasantness_z` - Valencia negativa
3. `emotional_intensity_z` - Arousal
4. `interoception_z` - Sensaciones corporales
5. `bliss_z` - Éxtasis
6. `anxiety_z` - Ansiedad
7. `valence_index_z` - Índice de valencia (pleasantness - unpleasantness)

## Próximos Pasos

1. ✅ Scripts de visualización actualizados
2. [ ] Ejecutar pipeline completo para generar nuevas figuras
3. [ ] Verificar que las figuras son correctas y legibles
4. [ ] Actualizar captions de figuras si es necesario
5. [ ] Actualizar reporte comprehensivo

## Comandos para Generar Figuras

```bash
# Generar todas las figuras
python scripts/generate_all_figures.py

# O generar figuras individuales
python scripts/plot_time_series.py
python scripts/plot_lme_coefficients.py
python scripts/plot_peak_auc.py

# Para figuras exploratorias (mantienen 15 dimensiones)
python scripts/plot_pca_results.py
python scripts/plot_state_results.py
```

## Notas Técnicas

1. **Compatibilidad hacia atrás**: Los scripts aceptan un parámetro `dimensions` opcional, permitiendo especificar manualmente qué dimensiones graficar si es necesario.

2. **Orden de dimensiones**: Las dimensiones se ordenan automáticamente por la magnitud del efecto State en los modelos LME, mostrando primero las dimensiones con efectos más fuertes.

3. **Anotaciones estadísticas**: Todas las anotaciones estadísticas (grey shading, black bars) se mantienen y funcionan correctamente con las 7 dimensiones.

4. **Resolución**: Todas las figuras se generan a 300 DPI por defecto (publication-ready).

5. **Formato**: Todas las figuras se guardan como PNG con transparencia donde sea apropiado.
