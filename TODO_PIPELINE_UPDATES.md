# TODO: Actualizaciones Necesarias en Pipeline TET

## Estado Actual

✅ **Completado:**
1. Variable compuesta simplificada a `valence_index_z` únicamente
2. Definidas dimensiones afectivas en `config.TET_AFFECTIVE_COLUMNS`
3. LME analyzer actualizado para usar solo dimensiones afectivas
4. Peak/AUC analyzer actualizado para usar solo dimensiones afectivas
5. Time series visualizer actualizado para dimensiones afectivas
6. Scripts de visualización actualizados (plot_time_series, plot_lme_coefficients, plot_peak_auc)
7. Mensajes de salida actualizados para reflejar 7 dimensiones afectivas

## Cambios Pendientes

### 1. Scripts de Visualización

#### `scripts/plot_time_series.py`
- [x] Actualizar para mostrar solo dimensiones afectivas + valence_index
- [x] Verificar que el orden de dimensiones sea por efecto State (más fuerte primero)
- [x] Asegurar que se muestren las anotaciones estadísticas correctas
- **Nota**: El visualizador ahora usa `config.TET_AFFECTIVE_COLUMNS` automáticamente

#### `scripts/plot_lme_coefficients.py`
- [x] Actualizar para mostrar solo resultados de dimensiones afectivas
- [x] Verificar que se ordenen por magnitud del efecto State
- **Nota**: Lee los resultados LME que ya solo contienen dimensiones afectivas

#### `scripts/plot_peak_auc.py`
- [x] Actualizar para mostrar solo dimensiones afectivas + valence_index
- [x] Reducir número de paneles en figuras
- **Nota**: Lee los resultados que ya solo contienen dimensiones afectivas

### 2. Scripts de Reporte

#### `scripts/generate_comprehensive_report.py`
- [ ] Actualizar para enfocarse en dimensiones afectivas
- [ ] Modificar secciones de resultados para reflejar nuevo enfoque
- [ ] Actualizar interpretaciones y conclusiones

#### `scripts/generate_all_figures.py`
- [ ] Verificar que solo genere figuras para dimensiones afectivas en LME/Peak-AUC
- [ ] Mantener figuras completas para PCA y clustering (exploratorios)

### 3. Documentación

#### `docs/methods_tet.md`
- [x] Ya actualizado con nueva metodología
- [ ] Verificar consistencia con implementación

#### `docs/PIPELINE.md`
- [ ] Actualizar para reflejar enfoque en dimensiones afectivas
- [ ] Documentar qué análisis usan todas las dimensiones vs solo afectivas

### 4. Validación y Testing

#### `scripts/verify_preprocessing.py`
- [x] Ya actualizado para valence_index_z
- [ ] Agregar verificación de que TET_AFFECTIVE_COLUMNS existen

#### `test/tet/` scripts
- [ ] Revisar tests para asegurar compatibilidad con cambios
- [ ] Actualizar tests que asuman 15 dimensiones en todos los análisis

### 5. Metadata y Configuración

#### `scripts/tet/metadata.py`
- [x] Ya actualizado para valence_index_z
- [ ] Agregar documentación de TET_AFFECTIVE_COLUMNS

## Análisis que Mantienen Todas las Dimensiones

Estos NO requieren cambios (son exploratorios):

✅ **PCA Analysis** (`compute_pca_analysis.py`)
- Usa todas las 15 dimensiones
- Objetivo: identificar estructura latente completa

✅ **Clustering Analysis** (`compute_clustering_analysis.py`)
- Usa todas las 15 dimensiones
- Objetivo: identificar estados experienciales discretos

✅ **Descriptive Stats** (`compute_descriptive_stats.py`)
- Reporta todas las 15 dimensiones
- Objetivo: caracterización completa de datos

## Prioridad de Implementación

### Alta Prioridad (Crítico para análisis)
1. ✅ LME analyzer - COMPLETADO
2. ✅ Peak/AUC analyzer - COMPLETADO
3. ✅ Visualización de time series - COMPLETADO
4. ✅ Visualización de coeficientes LME - COMPLETADO

### Media Prioridad (Importante para reportes)
5. ✅ Visualización de Peak/AUC - COMPLETADO
6. [ ] Reporte comprehensivo
7. ✅ Generador de todas las figuras - COMPLETADO (llama a scripts actualizados)

### Baja Prioridad (Documentación)
8. [ ] Actualizar PIPELINE.md
9. [ ] Revisar tests
10. [ ] Actualizar metadata

## Comandos para Verificar Cambios

```bash
# 1. Verificar que preprocesamiento crea valence_index_z
python scripts/preprocess_tet_data.py
python scripts/verify_preprocessing.py

# 2. Verificar que LME usa solo dimensiones afectivas
python scripts/fit_lme_models.py --verbose
# Debe mostrar: "Analyzing 7 dimensions" (6 afectivas + valence_index)

# 3. Verificar que Peak/AUC usa solo dimensiones afectivas
python scripts/compute_peak_auc.py --verbose
# Debe mostrar: "Found 7 z-scored dimensions"

# 4. Verificar que PCA usa todas las dimensiones
python scripts/compute_pca_analysis.py --verbose
# Debe mostrar: "Found 15 z-scored dimensions"
```

## Notas Importantes

1. **No modificar análisis exploratorios**: PCA y clustering deben mantener todas las dimensiones

2. **Consistencia en figuras**: Las figuras de LME y Peak/AUC deben mostrar solo dimensiones afectivas

3. **Documentación clara**: Explicar en métodos por qué algunos análisis usan todas las dimensiones y otros solo afectivas

4. **Validación**: Después de cada cambio, ejecutar el pipeline completo para verificar compatibilidad

## Checklist Final

Antes de considerar completa la actualización:

- [ ] Pipeline completo ejecuta sin errores
- [ ] Figuras muestran solo dimensiones afectivas donde corresponde
- [ ] Reportes reflejan nuevo enfoque metodológico
- [ ] Documentación actualizada y consistente
- [ ] Tests pasan correctamente
- [ ] Resultados interpretables y alineados con objetivos del paper
