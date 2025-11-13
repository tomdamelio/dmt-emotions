# Resolución Temporal de Datos TET

## Resumen Ejecutivo

Los datos TET en los archivos `.mat` están **down-sampled uniformemente** a una resolución de **0.25 Hz** (1 punto cada 4 segundos). 

**Estrategia de análisis**:
- **Por defecto**: Mantener resolución original (0.25 Hz) para máxima precisión temporal
- **Cuando sea necesario**: Agregar a bins de 30s usando `config.aggregate_tet_to_30s_bins()` para análisis estadísticos específicos (e.g., LME)

## Paper Original

> "Based on the assumptions of previous datasets [30, 31], each datapoint represented 30s of subjective experience, with N=20 for RS (10-minutes long) and N=40 for DMT (20-minutes long) conditions."

## Datos Actuales en Archivos .mat

### Resolución Temporal

Los datos están down-sampled uniformemente:

| Condición | Duración | Puntos | Resolución | Cálculo |
|-----------|----------|--------|------------|---------|
| **DMT** | 20 min | 300 | 4s/punto | 1200s / 300 = 4s |
| **RS** | 10 min | 150 | 4s/punto | 600s / 150 = 4s |

### Verificación

```python
# Primera sesión DMT (S01, session 1)
# t_bin: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...
# t_sec: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, ...
# Diferencia: 4s entre puntos consecutivos
```

## Agregación a Bins de 30s

### Objetivo

Según paper original:
- **DMT**: N=40 bins × 30s = 1200s = 20 minutos
- **RS**: N=20 bins × 30s = 600s = 10 minutos

### Factor de Agregación

```
Factor = 30s / 4s = 7.5 puntos por bin
```

Esto significa que cada bin de 30s agrupa aproximadamente 7-8 puntos consecutivos.

### Implementación

```python
from tet.data_loader import TETDataLoader
import config

# Cargar datos (resolución 4s)
loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
data = loader.load_data()

# Verificar resolución actual
print(f"DMT puntos por sesión: {data[data['state']=='DMT'].groupby(['subject','session_id']).size().unique()}")
# Output: [300]

print(f"RS puntos por sesión: {data[data['state']=='RS'].groupby(['subject','session_id']).size().unique()}")
# Output: [150]

# Agregar a bins de 30s
data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')

# Verificar bins resultantes
print(f"DMT bins por sesión: {data_30s[data_30s['state']=='DMT'].groupby(['subject','session_id']).size().unique()}")
# Output: [40]

print(f"RS bins por sesión: {data_30s[data_30s['state']=='RS'].groupby(['subject','session_id']).size().unique()}")
# Output: [20]
```

### Método de Agregación

Por defecto, se usa el **promedio (mean)** de los puntos dentro de cada bin de 30s:

```python
# Promedio (recomendado)
data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')

# Mediana (alternativa robusta a outliers)
data_30s = config.aggregate_tet_to_30s_bins(data, method='median')
```

## Ejemplo Práctico

### Datos Originales (4s)

```
Bin 0: t=0s, pleasantness=0.382
Bin 1: t=4s, pleasantness=0.466
Bin 2: t=8s, pleasantness=0.491
Bin 3: t=12s, pleasantness=0.494
Bin 4: t=16s, pleasantness=0.507
Bin 5: t=20s, pleasantness=0.507
Bin 6: t=24s, pleasantness=0.507
Bin 7: t=28s, pleasantness=0.508
...
```

### Datos Agregados (30s)

```
Bin 0: t=0-30s, pleasantness=0.483 (promedio de bins 0-7)
Bin 1: t=30-60s, pleasantness=0.481 (promedio de bins 8-15)
Bin 2: t=60-90s, pleasantness=0.587 (promedio de bins 16-23)
...
```

## Configuración en config.py

```python
# Resolución temporal de datos crudos (segundos por punto)
TET_RAW_RESOLUTION_SEC = 4  # 1 punto cada 4 segundos

# Duración de bin temporal según paper (segundos)
TET_BIN_DURATION_SEC = 30  # 1 bin cada 30 segundos

# Factor de agregación: cuántos puntos crudos por bin de 30s
TET_AGGREGATION_FACTOR = TET_BIN_DURATION_SEC / TET_RAW_RESOLUTION_SEC  # 30/4 = 7.5

# Longitudes esperadas (datos crudos @ 4s)
EXPECTED_SESSION_LENGTHS_RAW = {
    'RS': 150,   # 150 puntos @ 4s = 600s = 10 min
    'DMT': 300   # 300 puntos @ 4s = 1200s = 20 min
}

# Longitudes según paper original (bins @ 30s)
EXPECTED_SESSION_LENGTHS_30S = {
    'RS': 20,    # 20 bins × 30s = 600s = 10 minutos
    'DMT': 40    # 40 bins × 30s = 1200s = 20 minutos
}
```

## Cuándo Usar Cada Formato

### Datos Originales @ 0.25 Hz (300/150 puntos) - **RECOMENDADO POR DEFECTO**

**Usar cuando**:
- Análisis generales y exploratorios
- Máxima precisión temporal
- Análisis de series temporales
- Comparación con otras modalidades fisiológicas
- Visualizaciones detalladas

**Ventajas**:
- ✅ Resolución temporal completa (0.25 Hz)
- ✅ Timing exacto en columna `t_sec`
- ✅ Más puntos de datos
- ✅ Captura dinámicas temporales finas
- ✅ Flexible para diferentes análisis

**Desventajas**:
- Más datos para procesar
- Puede requerir agregación para algunos análisis estadísticos

### Datos Agregados @ 30s (40/20 bins) - **SOLO CUANDO SEA NECESARIO**

**Usar cuando**:
- Modelos LME que requieran bins de 30s
- Replicar análisis específicos del paper original
- Reducir complejidad computacional en modelos pesados

**Ventajas**:
- Coincide con especificación del paper original
- Reduce ruido mediante promediado
- Menos datos para modelos complejos

**Desventajas**:
- ❌ Pérdida de resolución temporal
- ❌ Puede perder variaciones rápidas
- ❌ Menos puntos de datos

## Recomendación

**Estrategia general**:
1. ✅ **Cargar y trabajar con datos @ 0.25 Hz** (resolución original)
2. ✅ **Agregar a 30s solo cuando sea necesario** para análisis estadísticos específicos
3. ✅ **Documentar claramente** qué resolución se usó en cada análisis

**Ejemplo de workflow**:
```python
# Cargar datos en resolución original (0.25 Hz)
loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
data = loader.load_data()

# Análisis exploratorios, visualizaciones → usar data @ 0.25 Hz
plot_timecourse(data)
analyze_dynamics(data)

# Modelos LME que requieran bins de 30s → agregar solo para ese análisis
data_30s = config.aggregate_tet_to_30s_bins(data, method='mean')
fit_lme_model(data_30s)
```

## Scripts de Demostración

- **`scripts/demo_tet_30s_aggregation.py`**: Demostración completa de agregación
- **`scripts/investigate_tet_timing.py`**: Investigación de estructura temporal

---

**Última actualización**: 2025-01-12  
**Estado**: ✅ Verificado y documentado
