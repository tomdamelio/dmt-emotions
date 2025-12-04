# Comparación Visual: Antes vs Después del Sombreado FDR

## Enfoque Anterior (Barras Negras)

```
┌─────────────────────────────────────────────────────────┐
│                    ████                                 │  ← Barras negras en top
│  3 ┤                                                    │
│    │                                                    │
│  2 ┤     ╱─────╲                                       │
│    │    ╱       ╲                                      │
│  1 ┤   ╱         ╲___                                  │
│    │  ╱              ╲___                              │
│  0 ┼─────────────────────────────────────────────────  │
│    │                                                    │
│ -1 ┤                                                    │
│    │                                                    │
│ -2 ┤                                                    │
│    │                                                    │
│ -3 ┤                                                    │
│    └────────────────────────────────────────────────── │
│         0    5    10   15   20  (minutos)              │
└─────────────────────────────────────────────────────────┘

Problemas:
- Barras negras solo en top (difícil ver con datos altos)
- Sin corrección FDR (muchos falsos positivos)
- No agrupa segmentos contiguos visualmente
```

## Enfoque Nuevo (Sombreado FDR)

```
┌─────────────────────────────────────────────────────────┐
│  3 ┤                                                    │
│    │    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │  ← Sombreado gris
│  2 ┤    ░░░╱─────╲░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │     cubre todo Y
│    │    ░░╱░░░░░░░╲░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│  1 ┤   ░╱░░░░░░░░░░╲___░░░░░░░░░░░░░░░░░░░░░░░░░░░░   │
│    │  ░╱░░░░░░░░░░░░░░░╲___░░░░░░░░░░░░░░░░░░░░░░░░   │
│  0 ┼─────────────────────────────────────────────────  │
│    │                                                    │
│ -1 ┤                                                    │
│    │                                                    │
│ -2 ┤                                                    │
│    │                                                    │
│ -3 ┤                                                    │
│    └────────────────────────────────────────────────── │
│         0    5    10   15   20  (minutos)              │
└─────────────────────────────────────────────────────────┘

Ventajas:
✓ Sombreado gris visible en todo el rango Y
✓ Corrección BH-FDR (control de falsos descubrimientos)
✓ Segmentos contiguos agrupados automáticamente
✓ Estilo consistente con análisis ECG
```

## Código de Sombreado

### Antes (Barras Negras)
```python
# Barras negras en la parte superior
if len(dose_sig_bins) > 0:
    y_max = 3
    y_bar = y_max * 0.95
    
    for group in bin_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.plot([t_start, t_end], [y_bar, y_bar], 
               color='black', linewidth=4, solid_capstyle='butt', zorder=3)
```

### Ahora (Sombreado FDR)
```python
# Sombreado gris estilo ECG
if len(dose_sig_bins) > 0:
    # Agrupar bins consecutivos
    bin_groups = []
    current_group = [dose_sig_bins[0]]
    
    for i in range(1, len(dose_sig_bins)):
        if dose_sig_bins[i] == current_group[-1] + 1:
            current_group.append(dose_sig_bins[i])
        else:
            bin_groups.append(current_group)
            current_group = [dose_sig_bins[i]]
    bin_groups.append(current_group)
    
    # Dibujar regiones sombreadas
    for group in bin_groups:
        t_start = group[0] * 4 / 60
        t_end = (group[-1] + 1) * 4 / 60
        ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=0)
```

## Ejemplo de Reporte FDR

```
FDR COMPARISON: High (40mg) vs Low (20mg) Dose Effects Over Time
Benjamini-Hochberg FDR correction applied per dimension across all time bins
Alpha = 0.05

DIMENSION: Emotional Intensity
------------------------------------------------------------
  Significant segments (count=1):
    - Bins 17-168: 1.13-11.27 min (68-676s), min p_FDR=0.0204
  Total significant bins: 152
  Min p_FDR: 0.020353
  Median p_FDR: 0.022845

DIMENSION: Interoception
------------------------------------------------------------
  No significant dose differences (p_FDR < 0.05)

DIMENSION: Anxiety
------------------------------------------------------------
  Significant segments (count=2):
    - Bins 25-45: 1.67-3.00 min (100-184s), min p_FDR=0.0312
    - Bins 89-102: 5.93-6.80 min (356-412s), min p_FDR=0.0445
  Total significant bins: 35
  Min p_FDR: 0.031234
  Median p_FDR: 0.038901
```

## Interpretación Estadística

### Sin Corrección FDR (Antes)
- **Problema**: Con 300 bins temporales y α=0.05, esperamos ~15 falsos positivos por azar
- **Resultado**: Muchas "diferencias significativas" que son ruido estadístico

### Con Corrección BH-FDR (Ahora)
- **Ventaja**: Controla la tasa de falsos descubrimientos al 5%
- **Resultado**: Solo diferencias robustas que sobreviven corrección múltiple
- **Interpretación**: Mayor confianza en los efectos reportados

## Consistencia con Análisis ECG

El nuevo enfoque es **idéntico** al usado en `run_ecg_hr_analysis.py`:

| Característica | ECG | TET |
|----------------|-----|-----|
| Corrección FDR | ✓ BH | ✓ BH |
| Sombreado | ✓ axvspan gris | ✓ axvspan gris |
| Color | `'0.85'` | `'0.85'` |
| Alpha | `0.35` | `0.35` |
| Zorder | `0` | `0` |
| Agrupación | ✓ Contiguos | ✓ Contiguos |
| Reporte | ✓ .txt | ✓ .txt |

## Referencias Visuales

### Estilo ECG (Modelo)
```python
# De run_ecg_hr_analysis.py línea ~1150
for w0, w1 in rs_segs:
    t0 = (w0 - 1) * WINDOW_SIZE_SEC / 60.0
    t1 = w1 * WINDOW_SIZE_SEC / 60.0
    ax1.axvspan(t0, t1, color='0.85', alpha=0.35, zorder=0)
```

### Estilo TET (Implementado)
```python
# De time_series_visualizer.py línea ~420
for group in bin_groups:
    t_start = group[0] * 4 / 60
    t_end = (group[-1] + 1) * 4 / 60
    ax.axvspan(t_start, t_end, color='0.85', alpha=0.35, zorder=0)
```

**Resultado**: Estilo visual idéntico y consistente entre modalidades.
