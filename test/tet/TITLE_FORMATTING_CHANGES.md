# Cambios en Formato de Títulos - timeseries_all_dimensions.png

## Resumen de Cambios

Se implementaron tres mejoras en el formato de títulos de la figura:

1. **Títulos de dos líneas** en paneles superiores (Arousal y Valence)
2. **Títulos más grandes y en negrita** en paneles inferiores
3. **Mayor separación** entre fila superior e inferior

## Cambio 1: Títulos de Dos Líneas (Paneles Superiores)

### Arousal (Emotional Intensity)

**Antes:**
```
Arousal (Emotional Intensity)
```
- Todo en una línea
- Todo en negrita
- Tamaño: 14pt

**Ahora:**
```
Arousal
(Emotional Intensity)
```
- **Primera línea**: "Arousal" en negrita, 18pt
- **Segunda línea**: "(Emotional Intensity)" sin negrita, 14pt
- Separación vertical entre líneas

### Valence (Pleasantness-Unpleasantness)

**Antes:**
```
Valence (Pleasantness-Unpleasantness)
```
- Todo en una línea
- Todo en negrita
- Tamaño: 14pt

**Ahora:**
```
Valence
(Pleasantness-Unpleasantness)
```
- **Primera línea**: "Valence" en negrita, 18pt
- **Segunda línea**: "(Pleasantness-Unpleasantness)" sin negrita, 14pt
- Separación vertical entre líneas

### Implementación

```python
# Arousal
ax_arousal.text(0.5, 1.08, 'Arousal', 
              transform=ax_arousal.transAxes,
              fontsize=18, fontweight='bold', 
              ha='center', va='bottom')
ax_arousal.text(0.5, 1.02, '(Emotional Intensity)', 
              transform=ax_arousal.transAxes,
              fontsize=14, fontweight='normal', 
              ha='center', va='bottom')

# Valence
ax_valence.text(0.5, 1.08, 'Valence', 
              transform=ax_valence.transAxes,
              fontsize=18, fontweight='bold', 
              ha='center', va='bottom')
ax_valence.text(0.5, 1.02, '(Pleasantness-Unpleasantness)', 
              transform=ax_valence.transAxes,
              fontsize=14, fontweight='normal', 
              ha='center', va='bottom')
```

## Cambio 2: Títulos Más Grandes (Paneles Inferiores)

### Paneles Secundarios

**Antes:**
- Tamaño: 11pt
- Negrita: Sí

**Ahora:**
- Tamaño: **14pt** (27% más grande)
- Negrita: Sí

### Dimensiones Afectadas

1. Interoception
2. Anxiety
3. Unpleasantness
4. Pleasantness
5. Bliss

### Implementación

```python
# Títulos de paneles secundarios
dim_name = dimension.replace('_z', '').replace('_', ' ').title()
ax.set_title(dim_name, fontsize=14, fontweight='bold')  # Antes: fontsize=11
```

## Cambio 3: Mayor Separación Entre Filas

### Espaciado Vertical (hspace)

**Antes:**
```python
gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.35, wspace=0.4)
```
- hspace = 0.35

**Ahora:**
```python
gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.50, wspace=0.4)
```
- hspace = **0.50** (43% más espacio)

### Efecto Visual

```
Antes:
┌─────────────────────────────────────┐
│  Arousal (Emotional Intensity)      │
│  [gráfico]                           │
└─────────────────────────────────────┘
       ↓ espacio pequeño (0.35)
┌─────────────────────────────────────┐
│  Interoception  Anxiety  ...        │
│  [gráficos]                          │
└─────────────────────────────────────┘

Ahora:
┌─────────────────────────────────────┐
│  Arousal                             │
│  (Emotional Intensity)               │
│  [gráfico]                           │
└─────────────────────────────────────┘
       ↓ espacio mayor (0.50)
       ↓
┌─────────────────────────────────────┐
│  Interoception  Anxiety  ...        │
│  [gráficos]                          │
└─────────────────────────────────────┘
```

## Comparación Visual Completa

### Layout Anterior

```
┌──────────────────────────────────────────────────────────────┐
│  Arousal (Emotional Intensity)    Valence (Pleasantness-... │ ← 14pt, todo negrita
│  [gráfico grande]                 [gráfico grande]          │
└──────────────────────────────────────────────────────────────┘
                    ↓ hspace=0.35
┌──────────────────────────────────────────────────────────────┐
│  Interoception  Anxiety  Unpleasantness  Pleasantness  Bliss│ ← 11pt, negrita
│  [5 gráficos pequeños]                                       │
└──────────────────────────────────────────────────────────────┘
```

### Layout Nuevo

```
┌──────────────────────────────────────────────────────────────┐
│      Arousal                          Valence                │ ← 18pt, negrita
│  (Emotional Intensity)    (Pleasantness-Unpleasantness)     │ ← 14pt, normal
│  [gráfico grande]                 [gráfico grande]          │
└──────────────────────────────────────────────────────────────┘
                    ↓ hspace=0.50 (más espacio)
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Interoception  Anxiety  Unpleasantness  Pleasantness  Bliss│ ← 14pt, negrita
│  [5 gráficos pequeños]                                       │
└──────────────────────────────────────────────────────────────┘
```

## Resumen de Tamaños de Fuente

| Elemento | Antes | Ahora | Cambio |
|----------|-------|-------|--------|
| **Arousal/Valence (línea 1)** | 14pt negrita | 18pt negrita | +29% |
| **Arousal/Valence (línea 2)** | 14pt negrita | 14pt normal | Mismo tamaño, sin negrita |
| **Paneles inferiores** | 11pt negrita | 14pt negrita | +27% |
| **Separación vertical** | hspace=0.35 | hspace=0.50 | +43% |

## Ventajas del Nuevo Diseño

1. **Jerarquía visual clara**: Títulos principales destacan más
2. **Mejor legibilidad**: Títulos más grandes en paneles inferiores
3. **Menos saturación**: Subtítulos sin negrita reducen peso visual
4. **Mejor organización**: Mayor separación entre filas
5. **Profesional**: Formato de dos líneas común en publicaciones

## Ejecución

```bash
# Regenerar figura con nuevo formato
python pipelines/run_tet_analysis.py --stages figures

# O directamente
python scripts/generate_all_figures.py --figures time-series
```

## Archivo Modificado

- **Archivo**: `scripts/tet/time_series_visualizer.py`
- **Método**: `generate_figure()` (líneas ~498-540)
- **Cambios**:
  - hspace: 0.35 → 0.50
  - Títulos superiores: texto de dos líneas con formato diferenciado
  - Títulos inferiores: fontsize 11 → 14

## Referencias

- Implementación: `scripts/tet/time_series_visualizer.py`
- Tests: `test/tet/test_fdr_shading.py`
- Documentación FDR: `test/tet/FINAL_TWO_LAYER_DESIGN.md`
