# Trazabilidad del Mapeo de Dimensiones TET

## Resumen

Este documento establece la trazabilidad del mapeo entre las columnas de los archivos `.mat` y los nombres de las 15 dimensiones fenomenológicas TET (Temporal Experience Tracking).

## Fuente Original

El orden de las dimensiones está definido en el script original del proyecto:

**Archivo**: `old_scripts/Visualizacion Reportes y Clustering.py`

```python
dimensiones = ['Pleasantness', 'Unpleasantness', 'Emotional_Intensity', 
               'Elementary_Imagery', 'Complex_Imagery', 'Auditory', 
               'Interoception', 'Bliss', 'Anxiety', 'Entity', 'Selfhood', 
               'Disembodiment', 'Salience', 'Temporality', 'General_Intensity']
```

Este script fue usado para el análisis preliminar de los datos y establece el orden canónico de las dimensiones.

## Implementación Actual

El mapeo está implementado en `config.py`:

```python
TET_DIMENSION_COLUMNS = [
    'pleasantness',        # 1. Intensidad subjetiva de lo "bueno" de la experiencia
    'unpleasantness',      # 2. Intensidad subjetiva de lo "malo" de la experiencia
    'emotional_intensity', # 3. Intensidad emocional independiente de valencia
    'elementary_imagery',  # 4. Sensaciones visuales básicas (destellos, colores, patrones)
    'complex_imagery',     # 5. Sensaciones visuales complejas (escenas vívidas, visiones)
    'auditory',            # 6. Sensaciones auditivas (sonidos externos o alucinatorios)
    'interoception',       # 7. Intensidad de sensaciones corporales internas ("body load")
    'bliss',               # 8. Experiencia de éxtasis o paz profunda
    'anxiety',             # 9. Experiencia de disforia o ansiedad
    'entity',              # 10. Presencia percibida de "entidades autónomas"
    'selfhood',            # 11. Alteraciones en la experiencia del "yo" (disolución del ego)
    'disembodiment',       # 12. Experiencia de NO identificarse con el propio cuerpo
    'salience',            # 13. Sentido subjetivo de significado profundo e importancia
    'temporality',         # 14. Alteraciones en la experiencia subjetiva del tiempo
    'general_intensity'    # 15. Intensidad general subjetiva de los efectos del DMT
]
```

## Formato de Datos

Los archivos `.mat` contienen una matriz llamada `dimensions` con shape `(n_puntos, 15)` donde:
- `n_puntos`: Número de puntos temporales (down-sampled uniformemente)
  - RS (Reposo): 150 puntos @ 4s = 600s = 10 minutos
  - DMT: 300 puntos @ 4s = 1200s = 20 minutos
- `15`: Las 15 dimensiones fenomenológicas en el orden especificado arriba

**Importante**: 
- Los archivos `.mat` NO contienen nombres de columnas. El mapeo se realiza por posición ordinal.
- Los datos están down-sampled a 1 punto cada 4 segundos.
- Para análisis según paper original (bins de 30s), usar `config.aggregate_tet_to_30s_bins()`:
  - DMT: 300 puntos → 40 bins
  - RS: 150 puntos → 20 bins

## Verificación

Se han creado dos scripts de verificación:

1. **`scripts/verify_tet_dimensions.py`**: Lista el orden actual de dimensiones
2. **`scripts/compare_tet_dimensions.py`**: Compara config.py con el script original

Para verificar la consistencia:

```bash
python scripts/compare_tet_dimensions.py
```

Debe mostrar: `✅ ÉXITO: Todas las dimensiones coinciden con el script original`

## Notas Importantes

### Dimensión 12: Disembodiment

Esta dimensión mide la experiencia de **desencarnación** o **NO identificarse con el propio cuerpo**.

- **Valores altos** (cercanos a 10): Mayor desencarnación, sensación de estar fuera del cuerpo
- **Valores bajos** (cercanos a 0): Mayor identificación con el cuerpo físico

**Nota histórica**: En versiones anteriores del código se usó erróneamente el nombre `embodiment` (opuesto semántico). Esto fue corregido el 2025-01-12 para coincidir con el script original.

### Índices Compuestos

Según `research_diary/TET.md`, se pueden crear índices compuestos:

- **`affect_index_z`**: Combina valencia positiva (pleasantness, bliss) vs negativa (anxiety, unpleasantness)
- **`imagery_index_z`**: Combina imagery elemental y complejo
- **`self_index_z`**: Combina selfhood con disembodiment invertido para medir integración del yo

Para el índice `self_index_z`, se debe **invertir** la escala de `disembodiment` (multiplicar por -1 después de estandarizar) para que valores altos indiquen mayor integración del yo.

## Referencias

- Script original: `old_scripts/Visualizacion Reportes y Clustering.py`
- Configuración: `config.py` (variable `TET_DIMENSION_COLUMNS`)
- Documentación: `docs/PIPELINE.md` (sección "Autorreportes")
- Especificación de análisis: `research_diary/TET.md`

## Historial de Cambios

- **2025-01-12**: Corrección de `embodiment` → `disembodiment` para coincidir con script original
- **2025-01-12**: Creación de scripts de verificación y documentación de trazabilidad
- **2025-01-12**: Agregados comentarios detallados en `config.py`

---

**Última verificación**: 2025-01-12  
**Estado**: ✅ Verificado contra script original
