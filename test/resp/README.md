# Respiratory Signal Quality Validation

Este directorio contiene scripts para validar la calidad de las señales respiratorias preprocesadas.

## Script: `validate_resp_quality.py`

### Propósito

Valida la calidad de los datos respiratorios preprocesados mediante:

1. **Verificación de rangos fisiológicos**
   - RSP_Rate: 5-40 respiraciones por minuto (rpm)
   - RSP_Amplitude: 0-3000 unidades arbitrarias

2. **Validación de detección de picos/troughs**
   - Número de picos y troughs detectados
   - Intervalos inter-pico (IPI) e inter-trough (ITI)
   - Detección de intervalos anormalmente cortos (< 1.5s)

3. **Detección de artefactos**
   - Valores NaN en señales
   - Saltos súbitos en la señal limpia
   - Porcentaje de datos fuera de rango

4. **Visualizaciones**
   - Señal respiratoria con picos/troughs marcados (primeros 30s)
   - Distribución de frecuencias respiratorias
   - Comparación entre condiciones
   - Gráficos de calidad de datos

### Uso

```bash
python test/resp/validate_resp_quality.py
```

### Outputs

Todos los resultados se guardan en `test/resp/validation_results/`:

1. **`validation_summary.csv`**: Tabla con todas las métricas de calidad por sujeto y condición
2. **`rate_distribution_summary.png`**: Gráficos de distribución y calidad general
3. **`{subject}_{condition}_signal.png`**: Visualización de señal con picos/troughs (48 archivos)

### Métricas de calidad

#### Por condición (subject × condition):

- **Duración**: Duración total de la grabación en segundos
- **% muestras válidas**: Porcentaje de muestras dentro del rango fisiológico
- **Frecuencia respiratoria**: Media ± SD (rpm)
- **Picos detectados**: Número total de ciclos respiratorios
- **Artefactos**: Porcentaje de valores NaN y saltos súbitos

#### Criterios de calidad:

✅ **Buena calidad**:
- ≥ 90% de muestras válidas
- < 5% de valores NaN
- < 1% de saltos súbitos

⚠️ **Calidad cuestionable**:
- 70-90% de muestras válidas
- 5-10% de valores NaN
- 1-5% de saltos súbitos

❌ **Mala calidad**:
- < 70% de muestras válidas
- > 10% de valores NaN
- > 5% de saltos súbitos

### Resultados del análisis

**Resumen general** (48 grabaciones):
- Media de muestras válidas: **88.9% ± 13.9%**
- Frecuencia respiratoria media: **12.2 ± 6.0 rpm**
- Media de NaN: **0.73%**
- Media de saltos súbitos: **4.86%**

**Grabaciones con problemas de calidad**:
- S09 (dmt_high, rs_high): < 52% válido
- S17 (dmt_high): 66.5% válido (frecuencia muy alta: 37 rpm)
- S19 (rs_high): 38.4% válido (frecuencia muy baja: 4.9 rpm)
- Varios sujetos con 5-9% de NaN en condiciones RS

### Interpretación

La mayoría de las grabaciones tienen buena calidad (> 90% válido), pero algunos sujetos muestran:

1. **Frecuencias extremas**: S17 en DMT High (37 rpm) sugiere hiperventilación
2. **Frecuencias muy bajas**: S09, S19, S20 en algunas condiciones (< 6 rpm)
3. **Valores NaN**: Principalmente en condiciones RS (resting state)

Estos casos requieren inspección visual de las señales generadas para determinar si son:
- Artefactos técnicos (problemas de grabación/preprocesamiento)
- Patrones fisiológicos reales (ej. apneas, hiperventilación)

### Próximos pasos

1. Revisar visualmente las señales de los sujetos problemáticos
2. Considerar exclusión o corrección manual de segmentos con artefactos
3. Documentar decisiones de inclusión/exclusión en el análisis final
4. Actualizar `SUJETOS_VALIDADOS_RESP` en `config.py` si es necesario

### Dependencias

- numpy
- pandas
- matplotlib
- seaborn
- config.py (configuración del proyecto)

### Notas

- El script usa los mismos sujetos definidos en `SUJETOS_VALIDADOS_RESP`
- Las visualizaciones muestran solo los primeros 30 segundos de cada grabación
- Los rangos fisiológicos son configurables en el script
