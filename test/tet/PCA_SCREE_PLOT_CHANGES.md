# Cambios en PCA Scree Plot - pca_scree_plot.png

## Resumen de Cambios

Se implementaron tres mejoras visuales en el gráfico de scree plot:

1. **Colores diferenciados**: PC1-PC2 en violeta, PC3-PC5 en gris
2. **Números de varianza**: Colores coinciden con las barras
3. **Sin cuadrícula**: Fondo limpio sin grid

## Cambio 1: Colores Diferenciados por Componente

### Antes
```
Todos los componentes en gradiente violeta:
PC1: Violeta oscuro
PC2: Violeta oscuro
PC3: Violeta medio
PC4: Violeta claro
PC5: Violeta muy claro
```

### Ahora
```
Componentes principales en violeta, resto en gris:
PC1: Violeta (tab20c[12])
PC2: Violeta (tab20c[12])
PC3: Gris (#808080)
PC4: Gris (#808080)
PC5: Gris (#808080)
```

### Implementación

```python
bar_colors = []
text_colors = []
for i in range(len(variance_df)):
    if i < 2:
        # PC1-PC2: violet (darkest purple from tab20c)
        bar_colors.append(COLOR_PRIMARY)  # tab20c[12]
        text_colors.append(COLOR_PRIMARY)
    else:
        # PC3-PC5: grey
        bar_colors.append('#808080')  # Medium grey
        text_colors.append('#808080')
```

## Cambio 2: Números de Varianza con Colores Coincidentes

### Antes
```
Todos los números en el mismo color que las barras (gradiente violeta)
```

### Ahora
```
Números coinciden con el color de las barras:
- PC1-PC2: Números en violeta
- PC3-PC5: Números en gris
```

### Implementación

```python
for i, var in enumerate(variance_df['variance_explained']):
    # Use violet for PC1-PC2, grey for PC3-PC5
    text_color = text_colors[i]
    ax1.text(i, var * 100 + 1, f'{var*100:.1f}%', ha='center', va='bottom', 
             fontsize=TICK_LABEL_SIZE-2, fontweight='bold', color=text_color)
```

## Cambio 3: Eliminación de Cuadrícula

### Antes
```python
ax1.grid(axis='y', alpha=0.25, linestyle='-', linewidth=0.5)
```
- Cuadrícula horizontal visible en el eje Y
- Alpha=0.25 (25% de opacidad)

### Ahora
```python
ax1.grid(False)
```
- Sin cuadrícula
- Fondo completamente limpio

## Comparación Visual

### Antes

```
┌────────────────────────────────────────┐
│  Variance Explained (%)                │
│                                        │
│  60 ┤ ─────────────────────────────── │ ← Grid lines
│     │                                  │
│  50 ┤ ─────────────────────────────── │
│     │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓       │
│  40 ┤ ─────────────────────────────── │
│     │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓       │
│  30 ┤ ─────────────────────────────── │
│     │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓       │
│  20 ┤ ─────────────────────────────── │
│     │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓       │
│  10 ┤ ─────────────────────────────── │
│     │  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓  ▓▓▓       │
│   0 ┼────────────────────────────────  │
│      PC1  PC2  PC3  PC4  PC5          │
└────────────────────────────────────────┘
     ↑    ↑    ↑    ↑    ↑
   Todos en gradiente violeta
```

### Ahora

```
┌────────────────────────────────────────┐
│  Variance Explained (%)                │
│                                        │
│  60 ┤                                  │ ← Sin grid
│     │  52.3% 18.7%                    │
│  50 ┤  ████  ████  ░░░  ░░░  ░░░     │
│     │  ████  ████  ░░░  ░░░  ░░░     │
│  40 ┤  ████  ████  ░░░  ░░░  ░░░     │
│     │  ████  ████  ░░░  ░░░  ░░░     │
│  30 ┤  ████  ████  ░░░  ░░░  ░░░     │
│     │  ████  ████  ░░░  ░░░  ░░░     │
│  20 ┤  ████  ████  ░░░  ░░░  ░░░     │
│     │  ████  ████  ░░░  ░░░  ░░░     │
│  10 ┤  ████  ████  ░░░  ░░░  ░░░     │
│     │  ████  ████  ░░░  ░░░  ░░░     │
│   0 ┼────────────────────────────────  │
│      PC1  PC2  PC3  PC4  PC5          │
└────────────────────────────────────────┘
     ↑    ↑    ↑    ↑    ↑
   Violeta  Violeta  Gris  Gris  Gris
```

## Paleta de Colores

### Violeta (PC1-PC2)
```python
COLOR_PRIMARY = tab20c_colors[12]  # RGB: (0.459, 0.420, 0.694)
```
- Violeta oscuro de la paleta tab20c
- Índice 12 (primer color del grupo púrpura)

### Gris (PC3-PC5)
```python
grey_color = '#808080'  # RGB: (0.5, 0.5, 0.5)
```
- Gris medio
- Neutral, no compite visualmente con violeta

## Justificación del Diseño

### Por qué PC1-PC2 en Violeta

1. **Componentes principales**: PC1 y PC2 explican la mayor parte de la varianza
2. **Énfasis visual**: El violeta destaca estos componentes como los más importantes
3. **Consistencia**: Violeta es el color de la modalidad TET en todo el análisis

### Por qué PC3-PC5 en Gris

1. **Componentes secundarios**: Explican menos varianza
2. **Jerarquía visual**: El gris indica menor importancia relativa
3. **Claridad**: Reduce saturación visual, enfoca atención en PC1-PC2

### Por qué Sin Cuadrícula

1. **Limpieza visual**: Menos elementos gráficos = más claridad
2. **Enfoque en datos**: Las barras y números son suficientes
3. **Estilo moderno**: Publicaciones actuales prefieren fondos limpios

## Ventajas del Nuevo Diseño

1. **Jerarquía clara**: PC1-PC2 destacan visualmente
2. **Menos ruido**: Sin cuadrícula de fondo
3. **Colores informativos**: Violeta = importante, gris = secundario
4. **Consistencia**: Números coinciden con colores de barras
5. **Profesional**: Estilo limpio y moderno

## Ejecución

```bash
# Regenerar figura PCA
python scripts/plot_pca_simple.py

# O a través del pipeline
python pipelines/run_tet_analysis.py --stages pca figures
```

## Archivos Generados

- **`results/tet/figures/pca_scree_plot.png`**
  - Scree plot con colores diferenciados
  - PC1-PC2 en violeta, PC3-PC5 en gris
  - Sin cuadrícula de fondo

- **`results/tet/figures/pca_loadings_heatmap.png`**
  - Heatmap de loadings (sin cambios)
  - Solo PC1 y PC2
  - Colormap amarillo-violeta divergente

## Archivo Modificado

- **Archivo**: `scripts/plot_pca_simple.py`
- **Líneas modificadas**: ~60-85
- **Cambios**:
  - `bar_colors`: Violeta para PC1-PC2, gris para PC3-PC5
  - `text_colors`: Coinciden con colores de barras
  - `ax1.grid(False)`: Elimina cuadrícula

## Referencias

- Implementación: `scripts/plot_pca_simple.py`
- Paleta de colores: tab20c (violeta) + gris estándar
- Estilo: Nature Human Behaviour
