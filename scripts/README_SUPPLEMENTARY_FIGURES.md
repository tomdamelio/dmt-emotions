# Edición Interactiva de Figuras Suplementarias

Este directorio contiene scripts para editar interactivamente las figuras suplementarias S1-S5 usando `pylustrator`.

## Scripts Disponibles

- `edit_figure_S1_interactive.py` - Editar Figure S1 (Stacked subjects: ECG, EDA, Resp)
- `edit_figure_S2_interactive.py` - Editar Figure S2 (DMT ECG HR extended)
- `edit_figure_S3_interactive.py` - Editar Figure S3 (Stacked subjects composite)
- `edit_figure_S4_interactive.py` - Editar Figure S4 (DMT composite extended)
- `edit_figure_S5_interactive.py` - Editar Figure S5 (Arousal Index vs TET)

## Uso

### 1. Asegúrate de que las figuras base existan

Primero, genera las figuras suplementarias:

```powershell
micromamba run -n dmt-emotions python src/run_figures.py --figures S1 S2 S3 S4 S5
```

### 2. Abre el editor interactivo

Para editar una figura específica, ejecuta:

```powershell
# Editar Figure S1
micromamba run -n dmt-emotions python scripts/edit_figure_S1_interactive.py

# Editar Figure S2
micromamba run -n dmt-emotions python scripts/edit_figure_S2_interactive.py

# Editar Figure S3
micromamba run -n dmt-emotions python scripts/edit_figure_S3_interactive.py

# Editar Figure S4
micromamba run -n dmt-emotions python scripts/edit_figure_S4_interactive.py

# Editar Figure S5
micromamba run -n dmt-emotions python scripts/edit_figure_S5_interactive.py
```

### 3. Controles del Editor

Una vez abierto el editor interactivo:

- **Click izquierdo**: Seleccionar elemento
- **Arrastrar**: Mover elemento
- **Manijas**: Redimensionar elemento
- **Click derecho**: Menú contextual con más opciones
- **Ctrl+Z**: Deshacer
- **Ctrl+Y**: Rehacer

### 4. Guardar Cambios

Cuando cierres la ventana, los cambios se guardarán automáticamente en el script. Pylustrator añadirá código al final del script con las posiciones y tamaños actualizados.

### 5. Aplicar Cambios

Los cambios se aplicarán automáticamente la próxima vez que ejecutes el script. Para regenerar la figura con los cambios:

```powershell
micromamba run -n dmt-emotions python scripts/edit_figure_SX_interactive.py
```

Luego puedes guardar la figura manualmente o actualizar `src/run_figures.py` para usar las nuevas posiciones.

## Notas

- Los scripts cargan las imágenes pre-generadas de `results/*/plots/`
- Para Figure S1, S2, S3, S4: Los scripts cargan imágenes PNG existentes
- Para Figure S5: El script regenera el plot desde los datos CSV
- Todos los cambios de layout se guardan en el mismo archivo del script
- Puedes ejecutar los scripts múltiples veces para refinar el layout

## Ejemplo de Workflow

1. Generar figuras base:
   ```powershell
   micromamba run -n dmt-emotions python src/run_figures.py --figures S1 S2 S3 S4 S5
   ```

2. Editar Figure S1 interactivamente:
   ```powershell
   micromamba run -n dmt-emotions python scripts/edit_figure_S1_interactive.py
   ```

3. Ajustar posiciones y tamaños en la ventana interactiva

4. Cerrar ventana para guardar

5. Volver a abrir para verificar cambios:
   ```powershell
   micromamba run -n dmt-emotions python scripts/edit_figure_S1_interactive.py
   ```

6. Repetir para otras figuras según sea necesario
