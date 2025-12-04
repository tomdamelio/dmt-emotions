# Implementación: Plot Extendido de DMT para Composite Arousal Index

## Problema

Los scripts unimodales (HR, SMNA, RVT) generan plots de ~19 minutos porque cargan **toda la data disponible** directamente desde los CSVs originales. Sin embargo, el script composite solo tenía acceso a los primeros 9 minutos porque usa `load_and_prepare(limit_to_9min=True)` para el análisis LME.

## Solución Implementada

### Arquitectura de Dos Pasos

**Paso 1: Extracción de Data Extendida**
- Nuevo script: `scripts/save_extended_dmt_data.py`
- Carga data cruda de las 3 modalidades (~38 ventanas = ~19 minutos)
- Aplica z-scoring a nivel de sujeto (usando RS + DMT de ambas sesiones)
- Guarda 3 CSVs intermedios en `results/composite/`:
  - `extended_dmt_hr_z.csv`
  - `extended_dmt_smna_z.csv`
  - `extended_dmt_rvt_z.csv`

**Paso 2: Generación del Plot Composite**
- Modificado: `pipelines/run_composite_arousal_index.py`
- Nueva función: `create_dmt_only_extended_plot_from_saved()`
- Lógica inteligente:
  - Si existen los CSVs guardados → los usa (rápido)
  - Si no existen → fallback al método original (lento)

### Archivos Creados

1. **`scripts/save_extended_dmt_data.py`** (nuevo)
   - Extrae y guarda data extendida de DMT (~19 min)
   - Funciones helper copiadas de scripts individuales
   - Z-scoring a nivel de sujeto para consistencia

2. **`scripts/generate_phys_figures.py`** (nuevo)
   - Script maestro que ejecuta todo el pipeline
   - Orden: RVT → HR → SMNA → Extract Extended → Composite
   - Reporte de éxito/fallo al final

3. **`docs/COMPOSITE_EXTENDED_WORKFLOW.md`** (nuevo)
   - Documentación completa del workflow
   - Troubleshooting y mantenimiento
   - Referencias técnicas

### Archivos Modificados

1. **`pipelines/run_composite_arousal_index.py`**
   - Nueva función: `create_dmt_only_extended_plot_from_saved()`
   - Modificada: `create_dmt_only_extended_plot()` con lógica de detección
   - Carga CSVs guardados si existen
   - Aplica PCA loadings guardados
   - Genera plot extendido con FDR

## Uso

### Opción 1: Pipeline Completo (Recomendado)

```bash
python scripts/generate_phys_figures.py
```

Ejecuta todo en orden:
1. Análisis RVT (genera plot de ~19 min)
2. Análisis HR (genera plot de ~19 min)
3. Análisis SMNA (genera plot de ~19 min)
4. Extracción de data extendida
5. Análisis composite (genera plot de ~19 min usando data guardada)

### Opción 2: Paso a Paso

```bash
# Paso 1: Extraer data extendida
python scripts/save_extended_dmt_data.py

# Paso 2: Análisis composite (usará data guardada)
python pipelines/run_composite_arousal_index.py
```

### Opción 3: Solo Composite (sin data guardada)

```bash
# Análisis composite (fallback a método original)
python pipelines/run_composite_arousal_index.py
```

## Outputs

### Nuevos CSVs Intermedios
- `results/composite/extended_dmt_hr_z.csv`
- `results/composite/extended_dmt_smna_z.csv`
- `results/composite/extended_dmt_rvt_z.csv`

### Plot Extendido
- `results/composite/plots/all_subs_dmt_composite.png` (ahora ~19 min)
- `results/composite/fdr_segments_all_subs_dmt_composite.txt` (reporte FDR)

## Ventajas de Esta Solución

1. **No Modifica Scripts Existentes**: Los scripts unimodales quedan intactos
2. **Reutilización de Data**: Los CSVs guardados se pueden reusar múltiples veces
3. **Fallback Automático**: Si no hay CSVs guardados, funciona igual que antes
4. **Consistencia**: Usa el mismo z-scoring que los análisis unimodales
5. **Modular**: Cada paso es independiente y puede ejecutarse por separado

## Detalles Técnicos

### Z-Scoring a Nivel de Sujeto

Todas las modalidades usan **subject-level z-scoring**:
```python
# Baseline: TODAS las sesiones del sujeto
y_all = concatenate([RS_high, DMT_high, RS_low, DMT_low])
mu = mean(y_all)
sigma = std(y_all)

# Normalización consistente para todas las condiciones
y_normalized = (y - mu) / sigma
```

### Formato de CSVs Guardados

```csv
subject,window,Dose,HR/SMNA_AUC/RVT
S04,1,High,0.234
S04,1,Low,-0.156
S04,2,High,0.412
...
```

- `window`: 1-38 (ventanas de 30 segundos)
- `Dose`: 'High' o 'Low'
- Valores: z-scored dentro de sujeto

### Aplicación de PCA

El script composite:
1. Carga data extendida de las 3 modalidades
2. Merge en casos completos (sujetos con las 3 señales)
3. Z-score cada señal dentro de sujeto (de nuevo, para consistencia)
4. Aplica PCA loadings guardados del análisis de 9 minutos
5. Proyecta en PC1 → ArousalIndex
6. Genera plot extendido con análisis FDR

## Testing

Para verificar que funciona:

```bash
# 1. Extraer data extendida
python scripts/save_extended_dmt_data.py

# Verificar que se crearon los CSVs
ls results/composite/extended_dmt_*.csv

# 2. Ejecutar análisis composite
python pipelines/run_composite_arousal_index.py

# Verificar que se creó el plot extendido
ls results/composite/plots/all_subs_dmt_composite.png
```

## Próximos Pasos

Si se desea optimizar aún más:

1. **Cache de PCA**: Guardar el modelo PCA completo (no solo loadings)
2. **Validación**: Agregar checks de consistencia entre data guardada y PCA
3. **Paralelización**: Procesar sujetos en paralelo durante extracción
4. **Logging**: Agregar logs más detallados para debugging

## Notas

- Los sujetos en la intersección están hardcoded: `['S04', 'S06', 'S07', 'S16', 'S18', 'S19', 'S20']`
- El límite de tiempo está en `LIMIT_SEC = 1150.0` (~19 minutos)
- Las ventanas son de 30 segundos (`WINDOW_SIZE_SEC = 30`)
- Total de ventanas: ~38 (`TOTAL_WINDOWS = int(floor(1150.0 / 30))`)

## Referencias

- Script de extracción: `scripts/save_extended_dmt_data.py`
- Script composite modificado: `pipelines/run_composite_arousal_index.py`
- Script maestro: `scripts/generate_phys_figures.py`
- Documentación: `docs/COMPOSITE_EXTENDED_WORKFLOW.md`
