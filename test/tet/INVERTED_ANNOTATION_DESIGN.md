# Diseño Invertido de Anotaciones - timeseries_all_dimensions.png

## Resumen del Cambio

Se invirtió la asignación de las anotaciones visuales:

### Antes (Diseño Original)
- **Barra negra horizontal**: Efectos principales (DMT vs RS)
- **Sombreado gris**: Diferencias de dosis (High vs Low, FDR)

### Ahora (Diseño Invertido)
- **Sombreado gris**: Efectos principales (DMT vs RS)
- **Barra negra horizontal**: Diferencias de dosis (High vs Low, FDR)

## Visualización del Cambio

### Diseño Anterior

```
┌─────────────────────────────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████████  ← Barra negra (DMT vs RS)
│     │                                                            │
│  2 ┤     ╱─────╲                                                │
│     │    ╱       ╲                                              │
│  1 ┤   ╱         ╲___                                           │
│     │  ╱              ╲___                                      │
│  0 ┼──────────────────────────────────────────────────────────  │
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← Sombreado gris      │
│ -1 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     (High vs Low FDR)   │
│     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                         │
│ -2 ┤  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                         │
└─────────────────────────────────────────────────────────────────┘
```

### Diseño Nuevo (Invertido)

```
┌─────────────────────────────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████████  ← Barra negra       │
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     (High vs Low FDR)│
│  2 ┤  ░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░░░░                       │
│     │  ░░╱░░░░░░░╲░░░░░░░░░░░░░░░░░░░░░░                       │
│  1 ┤  ░╱░░░░░░░░░╲___░░░░░░░░░░░░░░░░░░                        │
│     │  ╱░░░░░░░░░░░░░░╲___░░░░░░░░░░░░░░                       │
│  0 ┼──────────────────────────────────────────────────────────  │
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  ← Sombreado gris    │
│ -1 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     (DMT vs RS)       │
│     │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                      │
│ -2 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                      │
└─────────────────────────────────────────────────────────────────┘
```

## Capa 1: Sombreado Gris (Efectos Principales DMT vs RS)

### Qué Muestra
- Períodos donde **DMT difiere significativamente de RS** (baseline)
- Test: One-sample t-test (DMT vs RS baseline) en cada bin temporal
- Nivel: p < 0.05 (sin corrección FDR)

### Propósito
- Identificar cuándo el efecto de DMT es detectable
- Contexto general de la respuesta temporal
- Fondo visual que cubre todo el rango Y

### Parámetros Visuales
```python
color='0.85'  # Gris claro
alpha=0.35
zorder=0  # Detrás de todo
```

### Agrupación
- Bins consecutivos significativos se agrupan en segmentos contiguos
- Evita el efecto de "rayas verticales"

## Capa 2: Barra Negra Horizontal (Diferencias de Dosis FDR)

### Qué Muestra
- Períodos donde **High (40mg) difiere significativamente de Low (20mg)**
- Test: Independent samples t-test (High vs Low) en cada bin temporal
- Nivel: p_FDR < 0.05 (con corrección Benjamini-Hochberg)

### Propósito
- Identificar efectos de dosis robustos
- Control de tasa de falsos descubrimientos (FDR)
- Más conservador que efectos principales
- Fácil identificación visual (barra negra destacada)

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

## Interpretación Combinada

### Escenario 1: Solo Sombreado Gris
```
┌─────────────────────────────────────────┐
│  3 ┤                                    │ ← Sin barra negra
│  2 ┤  ░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░  │
│  1 ┤  ░╱░░░░░░░░░╲___░░░░░░░░░░░░░░░░  │
│  0 ┼────────────────────────────────────│
│ -1 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │ ← Sombreado gris
│ -2 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└─────────────────────────────────────────┘

Interpretación: DMT tiene efecto vs RS, pero no hay diferencias 
                robustas entre dosis (High vs Low)
```

### Escenario 2: Sombreado Gris + Barra Negra
```
┌─────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████  │ ← Barra negra
│  2 ┤  ░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░  │
│  1 ┤  ░╱░░░░░░░░░╲___░░░░░░░░░░░░░░░░  │
│  0 ┼────────────────────────────────────│
│ -1 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │ ← Sombreado gris
│ -2 ┤  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└─────────────────────────────────────────┘

Interpretación: DMT tiene efecto vs RS, Y hay diferencias robustas 
                entre dosis que sobreviven corrección FDR
```

### Escenario 3: Solo Barra Negra
```
┌─────────────────────────────────────────┐
│  3 ┤  ████████████████████████████████  │ ← Barra negra
│  2 ┤     ╱─────╲                        │
│  1 ┤   ╱         ╲___                   │
│  0 ┼────────────────────────────────────│
│ -1 ┤                                    │ ← Sin sombreado
│ -2 ┤                                    │
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
Capa más atrás (zorder=0):  Sombreado gris (efectos principales DMT vs RS)
                            ↓
Capa intermedia (zorder=1): [vacío]
                            ↓
Líneas de datos (zorder=2): Series temporales High/Low
                            ↓
Capa más adelante (zorder=3): Barra negra horizontal (dosis FDR)
```

## Ventajas del Diseño Invertido

1. **Barra negra para FDR**: Destaca los efectos más robustos (con corrección múltiple)
2. **Sombreado para contexto**: Efectos principales como fondo informativo
3. **Jerarquía visual clara**: Lo más importante (FDR) en primer plano
4. **Consistente con ECG**: Sombreado gris para efectos generales
5. **Fácil interpretación**: Barra negra = diferencias de dosis confirmadas

## Código Implementado

```python
# Capa 1: Sombreado gris (efectos principales DMT vs RS)
if len(sig_bins) > 0:
    # Agrupar bins consecutivos
    main_effect_groups = [...]
    
    # Sombreado de fondo
    for group in main_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=0)

# Capa 2: Barra negra (diferencias de dosis FDR)
if len(dose_sig_bins) > 0:
    # Agrupar bins consecutivos
    dose_effect_groups = [...]
    
    # Barra negra superior
    y_bar = 2.85  # 95% de y_max
    for group in dose_effect_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.plot([t_start, t_end], [y_bar, y_bar], 
               color='black', linewidth=4, solid_capstyle='butt', zorder=3)
```

## Comparación con Diseño Original

| Aspecto | Diseño Original | Diseño Invertido |
|---------|----------------|------------------|
| **Barra negra** | DMT vs RS | High vs Low (FDR) |
| **Sombreado gris** | High vs Low (FDR) | DMT vs RS |
| **Énfasis visual** | Efectos principales | Diferencias de dosis |
| **Corrección FDR** | Sombreado | Barra negra |
| **Z-order barra** | 3 (top) | 3 (top) |
| **Z-order sombreado** | 1 (medio) | 0 (fondo) |

## Ejecución

```bash
# Regenerar figura con diseño invertido
python pipelines/run_tet_analysis.py --stages figures

# O directamente
python scripts/generate_all_figures.py --figures time-series
```

## Archivos Modificados

- **Archivo**: `scripts/tet/time_series_visualizer.py`
- **Método**: `_plot_dimension()` (líneas ~395-455)
- **Cambios**:
  - Sombreado gris: efectos principales (DMT vs RS)
  - Barra negra: diferencias de dosis (High vs Low, FDR)
  - Z-order ajustado: sombreado=0, barra=3

## Referencias

- Implementación: `scripts/tet/time_series_visualizer.py`
- Diseño anterior: `test/tet/FINAL_TWO_LAYER_DESIGN.md`
- Formato de títulos: `test/tet/TITLE_FORMATTING_CHANGES.md`
