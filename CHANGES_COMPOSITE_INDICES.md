# Cambios en Variables Compuestas TET

## Resumen

Se simplificó la creación de variables compuestas de 3 índices a 1 solo índice de valencia.

## Cambios Realizados

### Antes
- `affect_index_z`: mean(pleasantness_z, bliss_z) - mean(anxiety_z, unpleasantness_z)
- `imagery_index_z`: mean(elementary_imagery_z, complex_imagery_z)
- `self_index_z`: -disembodiment_z + selfhood_z

### Después
- `valence_index_z`: pleasantness_z - unpleasantness_z

## Archivos Modificados

### Módulos Core (scripts/tet/)

1. **scripts/tet/preprocessor.py**
   - Método `create_composite_indices()`: Simplificado para crear solo valence_index_z
   - Método `preprocess_all()`: Actualizada documentación
   - Docstrings: Actualizados para reflejar 1 índice en vez de 3

2. **scripts/tet/metadata.py**
   - `composite_indices` dict: Reemplazadas 3 definiciones por 1 (valence_index_z)
   - Docstrings: Actualizados para "composite index" (singular)

3. **scripts/tet/session_metrics.py**
   - Lista de dimensiones por defecto: Actualizada para incluir solo valence_index_z
   - Docstrings: Actualizados

4. **scripts/tet/time_course.py**
   - Lista de dimensiones por defecto: Actualizada para incluir solo valence_index_z
   - Docstrings: Actualizados

5. **scripts/tet/lme_analyzer.py**
   - Docstrings: Actualizados para "composite index" (singular)

6. **scripts/tet/pca_analyzer.py**
   - Ejemplo en docstring: Actualizado para excluir solo valence_index_z

7. **scripts/tet/state_visualization.py**
   - Filtro de dimensiones: Actualizado para excluir solo valence_index_z

### Scripts de Análisis

8. **scripts/compute_clustering_analysis.py**
   - Filtro de dimensiones z: Actualizado para excluir solo valence_index_z

9. **scripts/compute_pca_analysis.py**
   - Filtro de dimensiones z (2 lugares): Actualizado para excluir solo valence_index_z
   - Comentarios: Actualizados

10. **scripts/preprocess_tet_data.py**
    - Mensajes de salida: Actualizados para reportar 1 composite index
    - Docstrings: Actualizados

11. **scripts/verify_preprocessing.py**
    - Verificación de fórmulas: Actualizada para verificar solo valence_index_z
    - Checks finales: Actualizados
    - Mensajes: Actualizados

### Configuración y Documentación

12. **config.py**
    - `COMPOSITE_INDEX_DEFINITIONS`: Reemplazadas 3 definiciones por 1 (valence_index_z)

13. **docs/methods_tet.md**
    - Sección "Composite Index": Actualizada para describir solo valence_index_z

## Total: 13 archivos modificados

## Próximos Pasos

Para aplicar estos cambios:

1. Ejecutar el pipeline de preprocesamiento:
   ```bash
   python scripts/preprocess_tet_data.py
   ```

2. Verificar que el preprocesamiento es correcto:
   ```bash
   python scripts/verify_preprocessing.py
   ```

3. Re-ejecutar análisis completo:
   ```bash
   python pipelines/run_tet_analysis.py
   ```

## Notas

- La variable `valence_index_z` captura la valencia afectiva neta (placer vs displacer)
- Valores positivos = experiencias más placenteras
- Valores negativos = experiencias más displacenteras
- Esta simplificación facilita la interpretación y se alinea con constructos afectivos estándar
- Todos los análisis posteriores (LME, peak/AUC, PCA, clustering) ahora trabajarán con esta única variable compuesta
