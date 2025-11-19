# Cambios Metodológicos: Enfoque en Dimensiones Afectivas

## Resumen

Se actualizó la metodología de análisis TET para enfocarse exclusivamente en dimensiones afectivas/autonómicas, alineándose con el objetivo de estudiar la relación entre experiencia subjetiva afectiva y respuestas fisiológicas.

## Cambios Conceptuales

### Antes
- Análisis de todas las 15 dimensiones TET
- 3 variables compuestas (affect, imagery, self)
- Enfoque amplio en toda la fenomenología

### Después
- Análisis de 6 dimensiones afectivas/autonómicas específicas
- 1 variable compuesta (valence_index_z)
- 1 proxy de arousal (emotional_intensity_z)
- Enfoque en aspectos afectivos y corporales

## Dimensiones Afectivas Seleccionadas

```python
TET_AFFECTIVE_COLUMNS = [
    'pleasantness',        # Valencia positiva
    'unpleasantness',      # Valencia negativa
    'emotional_intensity', # Arousal (intensidad emocional)
    'interoception',       # Sensaciones corporales ("body load")
    'bliss',               # Éxtasis/paz profunda
    'anxiety',             # Disforia/ansiedad
]
```

### Variables Derivadas

1. **Valence Index** (`valence_index_z`):
   - Fórmula: `pleasantness_z - unpleasantness_z`
   - Captura valencia afectiva neta

2. **Arousal Proxy** (`emotional_intensity_z`):
   - Dimensión directa de intensidad emocional
   - Independiente de valencia

## Justificación

Las dimensiones seleccionadas capturan:
- **Valencia afectiva**: pleasantness, unpleasantness, bliss, anxiety, valence_index
- **Arousal**: emotional_intensity
- **Componente autonómico**: interoception (sensaciones corporales)

Estas dimensiones son las más relevantes para:
1. Correlacionar con medidas fisiológicas (HR, EDA, respiración)
2. Entender la respuesta afectiva al DMT
3. Evaluar efectos de dosis en experiencia emocional

## Archivos Modificados

### Configuración

1. **config.py**
   - Nueva constante: `TET_AFFECTIVE_COLUMNS`
   - Define las 6 dimensiones afectivas para análisis

### Módulos de Análisis

2. **scripts/tet/lme_analyzer.py**
   - Ahora usa solo `TET_AFFECTIVE_COLUMNS` + `valence_index_z` por defecto
   - Actualizada documentación

3. **scripts/compute_peak_auc.py**
   - Ahora analiza solo dimensiones afectivas + valence_index_z
   - Reduce número de comparaciones múltiples

### Análisis NO Modificados (mantienen todas las dimensiones)

Los siguientes análisis mantienen todas las 15 dimensiones porque son exploratorios:

- **PCA** (`compute_pca_analysis.py`): Usa todas las dimensiones para identificar componentes principales
- **Clustering** (`compute_clustering_analysis.py`): Usa todas las dimensiones para identificar estados experienciales
- **Descriptive stats** (`compute_descriptive_stats.py`): Reporta todas las dimensiones

## Impacto en Resultados

### Reducción de Comparaciones Múltiples

**Antes:**
- LME: 15 dimensiones × 5 efectos = 75 tests
- Peak/AUC: 18 comparaciones (15 dims + 3 índices) × 3 métricas = 54 tests

**Después:**
- LME: 7 dimensiones (6 afectivas + valence) × 5 efectos = 35 tests
- Peak/AUC: 7 dimensiones × 3 métricas = 21 tests

**Beneficio**: Mayor poder estadístico al reducir corrección FDR

### Interpretación Más Clara

- Enfoque en constructos afectivos bien definidos
- Alineación directa con medidas fisiológicas
- Resultados más interpretables para publicación

## Próximos Pasos

1. Re-ejecutar preprocesamiento (ya incluye valence_index_z):
   ```bash
   python scripts/preprocess_tet_data.py
   ```

2. Re-ejecutar análisis LME y Peak/AUC:
   ```bash
   python scripts/fit_lme_models.py
   python scripts/compute_peak_auc.py
   ```

3. Verificar que solo se analizan dimensiones afectivas en resultados

4. Actualizar figuras y reportes para reflejar enfoque afectivo

## Notas Metodológicas para Paper

Para la sección de métodos, enfatizar:

1. **Selección a priori** de dimensiones afectivas basada en:
   - Relevancia teórica para respuesta afectiva
   - Conexión con sistemas autonómicos
   - Hipótesis sobre correlatos fisiológicos

2. **Arousal vs Valence**:
   - Emotional Intensity como proxy de arousal
   - Valence Index como medida bipolar de valencia

3. **Análisis exploratorios** (PCA, clustering) mantienen todas las dimensiones para:
   - Identificar estructura latente completa
   - No perder información sobre otros aspectos fenomenológicos
   - Permitir descubrimientos no anticipados

## Referencias para Justificación

- Modelo circumplejo de afecto (Russell, 1980)
- Teoría de valencia-arousal en emoción
- Conexión interoception-emoción (Craig, 2002)
- Respuestas autonómicas a psicodélicos (Carhart-Harris et al.)
