# Visualización de Tres Capas: Efectos Principales y de Dosis

## Resumen

La figura `timeseries_all_dimensions.png` ahora muestra **tres capas de información estadística**:

1. **Sombreado gris claro**: Efectos principales DMT vs RS (p < 0.05, sin corrección FDR)
2. **Sombreado gris oscuro**: Diferencias de dosis High vs Low (p_FDR < 0.05, con corrección BH)
3. **Barra negra horizontal superior**: Efectos principales DMT vs RS (para identificación rápida)

## Capas Visuales

```
┌─────────────────────────────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████████  ← Barra negra (Capa 3)
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← Sombreado gris claro (Capa 1)
│  2 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│  1 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
│  0 ┼──────────────────────────────────────
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Sombreado gris oscuro (Capa 2)
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     (FDR-corrected)
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│     │
│ -3 ┤
│     └────────────────────────────────────
│         0    5    10   15   20  (minutos)
└─────────────────────────────────────────────────────────────────┘
```

## Capa 1: Sombreado Gris Claro (Efectos Principales)

### Qué Muestra
- Períodos donde **DMT difiere significativamente de RS** (baseline)
- Test: One-sample t-test (DMT vs RS baseline) en cada bin temporal
- Nivel: p < 0.05 (sin corrección FDR)

### Propósito
- Identificar cuándo el efecto de DMT es detectable
- Contexto general de la respuesta temporal

### Parámetros Visuales
```python
color='grey'
alpha=0.2
zorder=0  # Detrás de todo
```

### Agrupación
- Bins consecutivos significativos se agrupan en segmentos contiguos
- Evita el efecto de "rayas verticales"

## Capa 2: Sombreado Gris Oscuro (Diferencias de Dosis FDR)

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
color='0.85'  # Gris más oscuro que Capa 1
alpha=0.35
zorder=1  # Encima de Capa 1, debajo de datos
```

### Agrupación
- Bins consecutivos significativos se agrupan en segmentos contiguos
- Corrección FDR aplicada **por dimensión** a través de todos los bins

## Capa 3: Barra Negra Horizontal (Indicador Visual)

### Qué Muestra
- **Misma información que Capa 1** (efectos principales DMT vs RS)
- Representación alternativa para identificación rápida

### Propósito
- Fácil identificación visual de períodos con efecto DMT
- Complementa el sombreado de fondo
- Útil cuando los datos están en la parte superior del gráfico

### Parámetros Visuales
```python
color='black'
linewidth=4
y_position=2.85  # 95% de y_max (3.0)
zorder=3  # Encima de todo
solid_capstyle='butt'  # Extremos cuadrados
```

### Agrupación
- Usa los mismos grupos que Capa 1
- Segmentos contiguos se representan como barras continuas

## Interpretación Combinada

### Escenario 1: Solo Capa 1 (gris claro + barra negra)
```
Interpretación: DMT tiene efecto vs RS, pero no hay diferencias 
                robustas entre dosis (High vs Low)
```

### Escenario 2: Capa 1 + Capa 2 (gris claro + gris oscuro + barra negra)
```
Interpretación: DMT tiene efecto vs RS, Y hay diferencias robustas 
                entre dosis que sobreviven corrección FDR
```

### Escenario 3: Solo Capa 2 (gris oscuro)
```
Interpretación: Diferencias de dosis detectables incluso sin efecto 
                principal fuerte (raro, pero posible)
```

### Escenario 4: Sin sombreado
```
Interpretación: No hay efectos detectables en ese período
```

## Ejemplo Visual Completo

```
Arousal (Emotional Intensity)
┌─────────────────────────────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████████████████████████████  │ ← Barra negra
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │ ← Gris claro
│  2 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   (DMT vs RS)
│     │  ░░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  1 ┤  ░░╱░░░░░░░░░╲___░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│     │  ░╱░░░░░░░░░░░░░░╲___░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  0 ┼──────────────────────────────────────────────────────────  │
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │ ← Gris oscuro
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │   (High vs Low FDR)
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│     │                                                            │
│ -3 ┤                                                            │
│     └────────────────────────────────────────────────────────── │
│         0    5    10   15   20  (minutos)                       │
└─────────────────────────────────────────────────────────────────┘
      ↑                                                          ↑
   DMT onset                                                  20 min
```

**Interpretación**: 
- Barra negra + gris claro: DMT tiene efecto fuerte vs RS durante todo el período
- Gris oscuro: Diferencias de dosis (High > Low) robustas y sostenidas (FDR-corrected)

## Orden de Renderizado (Z-order)

```
Capa más atrás (zorder=0):  Sombreado gris claro (efectos principales)
                            ↓
Capa intermedia (zorder=1): Sombreado gris oscuro (dosis FDR)
                            ↓
Líneas de datos (zorder=2): Series temporales High/Low
                            ↓
Capa más adelante (zorder=3): Barra negra horizontal
```

## Código Implementado

```python
# Capa 1: Sombreado gris claro + Barra negra (efectos principales)
if len(sig_bins) > 0:
    # Agrupar bins consecutivos
    main_effect_groups = [...]
    
    # Sombreado de fondo
    for group in main_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.axvspan(t_start, t_end, color='grey', alpha=0.2, zorder=0)
    
    # Barra negra superior
    y_bar = 2.85  # 95% de y_max
    for group in main_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.plot([t_start, t_end], [y_bar, y_bar], 
               color='black', linewidth=4, solid_capstyle='butt', zorder=3)

# Capa 2: Sombreado gris oscuro (dosis FDR)
if len(dose_sig_bins) > 0:
    # Agrupar bins consecutivos
    dose_effect_groups = [...]
    
    # Sombreado FDR-corrected
    for group in dose_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=1)
```

## Ventajas del Diseño de Tres Capas

1. **Información completa**: Muestra tanto efectos principales como de dosis
2. **Jerarquía visual clara**: Diferentes niveles de evidencia estadística
3. **Identificación rápida**: Barra negra permite escaneo visual rápido
4. **Control de FDR**: Diferencias de dosis con corrección múltiple
5. **Segmentos contiguos**: Agrupación automática evita ruido visual

## Referencias

- Archivo: `scripts/tet/time_series_visualizer.py`
- Método: `_plot_dimension()` (líneas ~395-450)
- Tests: `test/tet/test_fdr_shading.py`
