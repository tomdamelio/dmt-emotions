# Simplificación del Layout - timeseries_all_dimensions.png

## Resumen de Cambios

Se realizaron dos modificaciones importantes:

1. **Eliminación del panel "Valence Index"**
2. **Aumento del tamaño de los indicadores de dosis** (20mg vs 40mg)

## Cambio 1: Eliminación de Valence Index

### Layout Anterior

```
┌──────────────────────────────────────────────────────────────┐
│      Arousal                          Valence                │
│  (Emotional Intensity)    (Pleasantness-Unpleasantness)     │
│  [gráfico 50% ancho]          [gráfico 50% ancho]           │
└──────────────────────────────────────────────────────────────┘
                    ↓ hspace=0.50
┌──────────────────────────────────────────────────────────────┐
│  Interoception  Anxiety  Unpleasantness  Pleasantness  Bliss│
│  [5 gráficos pequeños]                                       │
└──────────────────────────────────────────────────────────────┘
```

### Layout Nuevo

```
┌──────────────────────────────────────────────────────────────┐
│                        Arousal                                │
│                  (Emotional Intensity)                        │
│              [gráfico 100% ancho]                            │
└──────────────────────────────────────────────────────────────┘
                    ↓ hspace=0.50
┌──────────────────────────────────────────────────────────────┐
│  Interoception  Anxiety  Unpleasantness  Pleasantness  Bliss│
│  [5 gráficos pequeños]                                       │
└──────────────────────────────────────────────────────────────┘
```

### Ventajas

1. **Mayor espacio para Arousal**: Panel principal ocupa todo el ancho
2. **Mejor visibilidad**: Más espacio para datos y anotaciones
3. **Enfoque claro**: Arousal como dimensión principal destacada
4. **Menos saturación**: Elimina redundancia con Pleasantness/Unpleasantness

### Implementación

**Antes:**
```python
# Row 1: Two large panels
main_dimensions = ['emotional_intensity_z', 'valence_index_z']

ax_arousal = fig.add_subplot(gs[0, 0:5])   # Columns 0-4 (50%)
ax_valence = fig.add_subplot(gs[0, 5:10])  # Columns 5-9 (50%)
```

**Ahora:**
```python
# Row 1: Single large panel (Arousal only)
main_dimension = 'emotional_intensity_z'

ax_arousal = fig.add_subplot(gs[0, :])  # All columns 0-10 (100%)
```

## Cambio 2: Indicadores de Dosis Más Grandes

### Leyenda Anterior

```
┌─────────────────┐
│ ─── 20mg        │  ← fontsize=10
│ ─── 40mg        │
└─────────────────┘
```

### Leyenda Nueva

```
┌─────────────────┐
│ ──── 20mg       │  ← fontsize=14 (+40%)
│ ──── 40mg       │     markerscale=1.5
└─────────────────┘     handlelength=2.5
```

### Parámetros Modificados

| Parámetro | Antes | Ahora | Cambio |
|-----------|-------|-------|--------|
| **fontsize** | 10 | 14 | +40% |
| **markerscale** | 1.0 (default) | 1.5 | +50% |
| **handlelength** | 2.0 (default) | 2.5 | +25% |

### Implementación

**Antes:**
```python
ax_arousal.legend(loc='upper right', fontsize=10, framealpha=0.9)
```

**Ahora:**
```python
ax_arousal.legend(loc='upper right', fontsize=14, framealpha=0.9, 
                 markerscale=1.5, handlelength=2.5)
```

## Comparación Visual Completa

### Antes

```
┌────────────────────────────────────────────────────────────────┐
│  Arousal                    │  Valence                         │
│  (Emotional Intensity)      │  (Pleasantness-Unpleasantness)  │
│                             │                                  │
│  ┌─────────────┐            │                                  │
│  │ ─ 20mg      │ ← pequeño  │                                  │
│  │ ─ 40mg      │            │                                  │
│  └─────────────┘            │                                  │
│  [gráfico 50%]              │  [gráfico 50%]                   │
└────────────────────────────────────────────────────────────────┘
```

### Ahora

```
┌────────────────────────────────────────────────────────────────┐
│                        Arousal                                  │
│                  (Emotional Intensity)                          │
│                                                                 │
│                                      ┌──────────────┐           │
│                                      │ ──── 20mg    │ ← grande  │
│                                      │ ──── 40mg    │           │
│                                      └──────────────┘           │
│                    [gráfico 100%]                               │
└────────────────────────────────────────────────────────────────┘
```

## Dimensiones Totales de la Figura

### Configuración

```python
fig = plt.figure(figsize=(20, 8), dpi=300)
gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.50, wspace=0.4)
```

### Layout Grid

```
Fila 0 (superior):  Arousal (columnas 0-10, 100% ancho)
                    ↓ hspace=0.50
Fila 1 (inferior):  5 paneles secundarios (2 columnas cada uno)
                    - Interoception (cols 0-1)
                    - Anxiety (cols 2-3)
                    - Unpleasantness (cols 4-5)
                    - Pleasantness (cols 6-7)
                    - Bliss (cols 8-9)
```

## Dimensiones Incluidas en la Figura

### Fila Superior (1 dimensión)
1. **Arousal** (Emotional Intensity) - Panel grande, ancho completo

### Fila Inferior (5 dimensiones)
1. **Interoception**
2. **Anxiety**
3. **Unpleasantness**
4. **Pleasantness**
5. **Bliss**

### Dimensión Eliminada
- ~~Valence Index~~ (redundante con Pleasantness/Unpleasantness)

## Ventajas del Nuevo Layout

1. **Enfoque claro**: Arousal como dimensión principal
2. **Mejor uso del espacio**: Panel principal más grande
3. **Leyenda más legible**: Indicadores de dosis más visibles
4. **Menos redundancia**: Elimina Valence Index
5. **Jerarquía visual**: Clara distinción entre dimensión principal y secundarias

## Tamaños de Fuente Finales

| Elemento | Tamaño | Peso |
|----------|--------|------|
| **Arousal (línea 1)** | 18pt | Negrita |
| **Arousal (línea 2)** | 14pt | Normal |
| **Leyenda dosis** | 14pt | Normal |
| **Títulos inferiores** | 14pt | Negrita |
| **Ejes** | 22pt | Normal |

## Ejecución

```bash
# Regenerar figura con nuevo layout
python pipelines/run_tet_analysis.py --stages figures

# O directamente
python scripts/generate_all_figures.py --figures time-series
```

## Archivos Modificados

- **Archivo**: `scripts/tet/time_series_visualizer.py`
- **Método**: `generate_figure()` (líneas ~483-540)
- **Cambios**:
  - Eliminado panel de Valence Index
  - Arousal ocupa toda la fila superior (gs[0, :])
  - Leyenda con fontsize=14, markerscale=1.5, handlelength=2.5

## Referencias

- Implementación: `scripts/tet/time_series_visualizer.py`
- Formato de títulos: `test/tet/TITLE_FORMATTING_CHANGES.md`
- Diseño FDR: `test/tet/FINAL_TWO_LAYER_DESIGN.md`
