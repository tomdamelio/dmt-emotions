# Diseño Final de Dos Capas: Efectos Principales y de Dosis

## Resumen

La figura `timeseries_all_dimensions.png` muestra **dos capas de información estadística**:

1. **Barra negra horizontal superior**: Efectos principales DMT vs RS (p < 0.05)
2. **Sombreado gris FDR**: Diferencias de dosis High vs Low (p_FDR < 0.05)

## Visualización Final

```
┌─────────────────────────────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████████  ← Barra negra (DMT vs RS)
│     │                                                            │
│  2 ┤     ╱─────╲                                                │
│     │    ╱       ╲                                              │
│  1 ┤   ╱         ╲___                                           │
│     │  ╱              ╲___                                      │
│  0 ┼──────────────────────────────────────────────────────────  │
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Sombreado gris FDR  │
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     (High vs Low)       │
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                         │
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                         │
│     │                                                            │
│ -3 ┤                                                            │
│     └────────────────────────────────────────────────────────── │
│         0    5    10   15   20  (minutos)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Capa 1: Barra Negra Horizontal (Efectos Principales)

### Qué Muestra
- Períodos donde **DMT difiere significativamente de RS** (baseline)
- Test: One-sample t-test (DMT vs RS baseline) en cada bin temporal
- Nivel: p < 0.05 (sin corrección FDR)

### Propósito
- Identificación visual rápida de efectos DMT
- Fácil de ver independientemente de la posición de los datos
- Contraste claro contra el fondo blanco

### Parámetros Visuales
```python
color='black'
linewidth=4
y_position=2.85  # 95% de y_max (3.0)
zorder=3  # Encima de todo
solid_capstyle='butt'  # Extremos cuadrados
```

### Agrupación
- Bins consecutivos significativos se agrupan en segmentos contiguos
- Cada segmento se representa como una barra horizontal continua

## Capa 2: Sombreado Gris FDR (Diferencias de Dosis)

### Qué Muestra
- Períodos donde **High (40mg) difiere significativamente de Low (20mg)**
- Test: Independent samples t-test (High vs Low) en cada bin temporal
- Nivel: p_FDR < 0.05 (con corrección Benjamini-Hochberg)

### Propósito
- Identificar efectos de dosis robustos
- Control de tasa de falsos descubrimientos (FDR)
- Más conservador que efectos principales

### Parámetros Visuales
```python
color='0.85'  # Gris claro
alpha=0.35
zorder=1  # Detrás de datos, delante de grid
```

### Agrupación
- Bins consecutivos significativos se agrupan en segmentos contiguos
- Corrección FDR aplicada **por dimensión** a través de todos los bins
- Estilo idéntico al análisis de ECG

## Interpretación Combinada

### Escenario 1: Solo Barra Negra
```
┌─────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████  │ ← Barra negra
│  2 ┤     ╱─────╲                        │
│  1 ┤   ╱         ╲___                   │
│  0 ┼────────────────────────────────────│
│ -1 ┤                                    │ ← Sin sombreado
│ -2 ┤                                    │
└─────────────────────────────────────────┘

Interpretación: DMT tiene efecto vs RS, pero no hay diferencias 
                robustas entre dosis (High vs Low)
```

### Escenario 2: Barra Negra + Sombreado Gris
```
┌─────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████  │ ← Barra negra
│  2 ┤  ░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░  │
│  1 ┤  ░╱░░░░░░░░░╲___░░░░░░░░░░░░░░░░  │
│  0 ┼────────────────────────────────────│
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ ← Sombreado gris
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
└─────────────────────────────────────────┘

Interpretación: DMT tiene efecto vs RS, Y hay diferencias robustas 
                entre dosis que sobreviven corrección FDR
```

### Escenario 3: Solo Sombreado Gris
```
┌─────────────────────────────────────────┐
│  3 ┤                                    │ ← Sin barra negra
│  2 ┤     ╱─────╲                        │
│  1 ┤   ╱         ╲___                   │
│  0 ┼────────────────────────────────────│
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ ← Sombreado gris
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
└─────────────────────────────────────────┘

Interpretación: Diferencias de dosis detectables incluso sin efecto 
                principal fuerte (raro, pero posible)
```

### Escenario 4: Sin Anotaciones
```
┌─────────────────────────────────────────┐
│  3 ┤                                    │
│  2 ┤     ╱─────╲                        │
│  1 ┤   ╱         ╲___                   │
│  0 ┼────────────────────────────────────│
│ -1 ┤                                    │
│ -2 ┤                                    │
└─────────────────────────────────────────┘

Interpretación: No hay efectos detectables en ese período
```

## Orden de Renderizado (Z-order)

```
Capa más atrás (zorder=0):  Grid del gráfico
                            ↓
Capa intermedia (zorder=1): Sombreado gris FDR (dosis)
                            ↓
Líneas de datos (zorder=2): Series temporales High/Low
                            ↓
Capa más adelante (zorder=3): Barra negra horizontal (efectos principales)
```

## Código Implementado

```python
# Capa 1: Barra negra horizontal (efectos principales DMT vs RS)
sig_bins = self.significance_annotations[
    (self.significance_annotations['dimension'] == dimension) &
    (self.significance_annotations['main_effect_sig'] == True)
]['t_bin'].values

if len(sig_bins) > 0:
    # Agrupar bins consecutivos
    main_effect_groups = []
    current_group = [sig_bins[0]]
    
    for i in range(1, len(sig_bins)):
        if sig_bins[i] == current_group[-1] + 1:
            current_group.append(sig_bins[i])
        else:
            main_effect_groups.append(current_group)
            current_group = [sig_bins[i]]
    main_effect_groups.append(current_group)
    
    # Dibujar barras negras
    y_max = 3
    y_bar = y_max * 0.95
    
    for group in main_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.plot([t_start, t_end], [y_bar, y_bar], 
               color='black', linewidth=4, solid_capstyle='butt', zorder=3)

# Capa 2: Sombreado gris FDR (diferencias de dosis)
dose_sig_bins = self.dose_interaction_bins[
    (self.dose_interaction_bins['dimension'] == dimension) &
    (self.dose_interaction_bins['dose_effect_sig'] == True)
]['t_bin'].values

if len(dose_sig_bins) > 0:
    # Agrupar bins consecutivos
    dose_effect_groups = []
    current_group = [dose_sig_bins[0]]
    
    for i in range(1, len(dose_sig_bins)):
        if dose_sig_bins[i] == current_group[-1] + 1:
            current_group.append(dose_sig_bins[i])
        else:
            dose_effect_groups.append(current_group)
            current_group = [dose_sig_bins[i]]
    dose_effect_groups.append(current_group)
    
    # Dibujar sombreado FDR
    for group in dose_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=1)
```

## Ventajas del Diseño de Dos Capas

1. **Visual limpio**: Sin superposición de sombreados
2. **Jerarquía clara**: Barra negra (efectos principales) vs sombreado gris (dosis FDR)
3. **Identificación rápida**: Barra negra visible desde lejos
4. **Control estadístico**: FDR solo para diferencias de dosis (más conservador)
5. **Consistente con ECG**: Sombreado gris idéntico al análisis de frecuencia cardíaca

## Comparación con Análisis ECG

| Aspecto | ECG (HR) | TET (Emociones) |
|---------|----------|-----------------|
| **Sombreado FDR** | ✓ axvspan gris | ✓ axvspan gris |
| **Color** | `'0.85'` | `'0.85'` |
| **Alpha** | `0.35` | `0.35` |
| **Zorder** | `0` | `1` |
| **Agrupación** | ✓ Contiguos | ✓ Contiguos |
| **Corrección FDR** | ✓ BH | ✓ BH |
| **Barra superior** | ✗ No | ✓ Sí (efectos principales) |

## Ejecución

```bash
# Regenerar figuras con nuevo diseño
python pipelines/run_tet_analysis.py --stages figures

# O directamente
python scripts/generate_all_figures.py --figures time-series
```

## Archivos Generados

1. **`results/tet/figures/timeseries_all_dimensions.png`**
   - Series temporales con dos capas de anotaciones
   - Barra negra: efectos principales DMT vs RS
   - Sombreado gris: diferencias de dosis FDR

2. **`results/tet/figures/timeseries_all_dimensions_fdr_report.txt`**
   - Reporte detallado de segmentos FDR significativos
   - Estadísticas p_FDR por dimensión

## Referencias

- Archivo: `scripts/tet/time_series_visualizer.py`
- Método: `_plot_dimension()` (líneas ~395-450)
- Estilo ECG: `pipelines/run_ecg_hr_analysis.py` (líneas ~1150)
- Tests: `test/tet/test_fdr_shading.py`
