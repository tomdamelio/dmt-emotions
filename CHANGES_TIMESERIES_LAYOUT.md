# Cambios: Nuevo Layout para Time Series Figure

**Fecha**: 2025-11-18  
**Objetivo**: Reorganizar el layout de `timeseries_all_dimensions.png` con un diseño personalizado de 2 filas.

## Nuevo Layout

### Estructura de la Figura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TET Time Series: Dose Effects                     │
├──────────────────────────────────┬──────────────────────────────────┤
│                                  │                                  │
│   Arousal (Emotional Intensity)  │  Valence (Pleasantness-Unpl.)   │
│         (5 columnas)             │         (5 columnas)             │
│                                  │                                  │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│          │          │          │          │                        │
│ Interoc. │ Anxiety  │ Unpleas. │ Pleasan. │        Bliss          │
│(2 cols)  │(2 cols)  │(2 cols)  │(2 cols)  │      (2 cols)         │
│          │          │          │          │                        │
└──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

### Fila 1: Dimensiones Principales (Paneles Grandes)
- **Arousal (Emotional Intensity)** - Columnas 0-4 (5 columnas)
- **Valence (Pleasantness-Unpleasantness)** - Columnas 5-9 (5 columnas)

### Fila 2: Dimensiones Secundarias (Paneles Pequeños)
Cada panel ocupa 2 columnas:
1. **Interoception** - Columnas 0-1
2. **Anxiety** - Columnas 2-3
3. **Unpleasantness** - Columnas 4-5
4. **Pleasantness** - Columnas 6-7
5. **Bliss** - Columnas 8-9

## Archivos Modificados

### 1. `scripts/tet/time_series_visualizer.py`

**Método modificado**: `generate_figure()`

**Cambios principales**:
- Implementado layout personalizado usando `matplotlib.gridspec.GridSpec(2, 10)`
- Definidas dimensiones específicas en orden fijo (no ordenadas por efecto estadístico)
- Fila 1: 2 paneles grandes (5 columnas cada uno)
- Fila 2: 5 paneles pequeños (2 columnas cada uno)
- Ajustado tamaño de figura a `(20, 8)` para acomodar el nuevo layout
- Ajustados espaciados: `hspace=0.35, wspace=0.4`
- Títulos más grandes para paneles principales (fontsize=14)
- Títulos medianos para paneles secundarios (fontsize=11)

**Dimensiones específicas**:
```python
main_dimensions = ['arousal_z', 'valence_index_z']
secondary_dimensions = [
    'interoception_z',
    'anxiety_z', 
    'unpleasantness_z',
    'pleasantness_z',
    'bliss_z'
]
```

### 2. `scripts/plot_time_series.py`

**Cambios**:
- Actualizados mensajes de consola para reflejar el nuevo layout
- Descripción actualizada de componentes de la figura

## Características Visuales Mantenidas

Todos los elementos visuales originales se mantienen:

- ✓ **Líneas de dosis**: Azul (20mg), Rojo (40mg)
- ✓ **Sombreado SEM**: Áreas sombreadas alrededor de las medias
- ✓ **Línea vertical punteada**: Marca el inicio de DMT (t=0)
- ✓ **Fondo gris**: Ventanas temporales con efecto significativo DMT vs RS
- ✓ **Barras negras superiores**: Ventanas con diferencias significativas entre dosis
- ✓ **Leyenda**: Solo en el primer panel (Arousal)
- ✓ **Ejes**: Tiempo en minutos (-1 a 20), Z-scores (-3 a 3)
- ✓ **Grid**: Líneas de cuadrícula sutiles

## Dimensiones del Config

Las dimensiones utilizadas corresponden a:

```python
# De config.py
TET_AFFECTIVE_COLUMNS = [
    'arousal',           # → arousal_z (Fila 1, Panel 1)
    'valence_index',     # → valence_index_z (Fila 1, Panel 2)
    'interoception',     # → interoception_z (Fila 2, Panel 1)
    'anxiety',           # → anxiety_z (Fila 2, Panel 2)
    'unpleasantness',    # → unpleasantness_z (Fila 2, Panel 3)
    'pleasantness',      # → pleasantness_z (Fila 2, Panel 4)
    'bliss'              # → bliss_z (Fila 2, Panel 5)
]
```

**Nota**: Se utilizan las versiones z-scored (`_z`) de cada dimensión.

## Tamaño de Figura

- **Anterior**: `(15, 10)` - 5 filas × 3 columnas
- **Nuevo**: `(20, 8)` - 2 filas × 10 columnas (con GridSpec)
- **Resolución**: 300 DPI (sin cambios)
- **Formato**: PNG con fondo blanco

## Espaciado

- **hspace**: 0.35 (espacio vertical entre filas)
- **wspace**: 0.4 (espacio horizontal entre columnas)

## Títulos

- **Título principal**: Fontsize 16, bold
- **Paneles principales** (Arousal, Valence): Fontsize 14, bold
- **Paneles secundarios**: Fontsize 11, bold

## Comando para Regenerar

```bash
# Regenerar solo la figura de time series
python scripts/plot_time_series.py --verbose

# O regenerar todas las figuras
python scripts/generate_all_figures.py --verbose

# O ejecutar pipeline completo desde figuras
python pipelines/run_tet_analysis.py --from-stage figures --verbose
```

## Ubicación del Archivo

```
results/tet/figures/timeseries_all_dimensions.png
```

## Validación

Para verificar que el nuevo layout funciona:

```bash
# 1. Verificar que los datos necesarios existen
ls results/tet/preprocessed/tet_preprocessed.csv
ls results/tet/lme/lme_results.csv
ls results/tet/lme/lme_contrasts.csv
ls results/tet/descriptive/time_course_all_dimensions.csv

# 2. Regenerar la figura
python scripts/plot_time_series.py --verbose

# 3. Verificar que se creó el archivo
ls -lh results/tet/figures/timeseries_all_dimensions.png
```

## Notas Técnicas

1. **GridSpec**: Permite control preciso sobre el layout de subplots
2. **Orden fijo**: Las dimensiones ya no se ordenan por efecto estadístico, sino que siguen el orden especificado
3. **Compatibilidad**: El cambio es compatible con el resto del pipeline
4. **Flexibilidad**: El código puede adaptarse fácilmente si se necesitan otras dimensiones

## Próximos Pasos

1. Regenerar la figura con el nuevo layout
2. Verificar visualmente que el layout es correcto
3. Actualizar documentación si es necesario
4. Considerar si se necesitan ajustes adicionales en tamaños de fuente o espaciado
