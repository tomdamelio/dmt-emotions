# Resumen: Comparaciones con Baseline (DMT vs Resting State)

## ✅ Estado: COMPLETADO

Se implementaron y ejecutaron exitosamente las comparaciones de features extraídos entre DMT (dosis colapsadas) y Resting State baseline, respondiendo al comentario del supervisor sobre comparaciones con baseline.

---

## Justificación Científica

**Comentario del supervisor:**
> "Algo que podría sumar un poco (aunque no hablaría de diferencias entre dosis baja y dosis alta) es sumar comparaciones con el baseline. Las comparaciones tiempo a tiempo con el baseline no tienen sentido porque no hay 'dinámica' en el baseline, pero si miramos algunas de las cosas que puse en el punto anterior (por ejemplo, altura del máximo, etc) capaz tendría sentido hacer la comparación."

**Implementación:**
- ✅ Extracción de features temporales (peak amplitude, time-to-peak, threshold crossings)
- ✅ Comparación DMT (dosis colapsadas) vs RS baseline usando paired t-tests
- ✅ Cuantifica magnitud del efecto DMT independiente de comparaciones dosis-dependientes
- ✅ Colapsar dosis aumenta poder estadístico para detectar efectos del fármaco

⚠️ **IMPORTANTE:** Estos análisis NO abordan efectos dosis-dependientes. Solo cuantifican la magnitud general del cambio inducido por DMT relativo a baseline.

---

## Resultados por Modalidad

### 1. Electrocardiography (ECG/HR)

#### DMT vs RS Baseline:
| Feature | DMT Mean | RS Mean | t | df | p | Cohen's d | Sig |
|---------|----------|---------|---|----|----|-----------|-----|
| **Peak Amplitude** | 2.482 | -0.040 | 14.236 | 10 | **<0.001** | 4.292 | **✓✓✓** |
| **Time to Peak** | 1.909 min | 5.023 min | -6.803 | 10 | **<0.001** | -2.051 | **✓✓✓** |
| **Time to 33%** | 0.889 min | 4.139 min | -3.677 | 10 | **0.006** | -1.226 | **✓✓** |
| **Time to 50%** | 1.139 min | 4.583 min | -5.139 | 10 | **<0.001** | -1.713 | **✓✓✓** |

**Interpretación:**
- **Peak amplitude MUCHO mayor en DMT** (d = 4.292, efecto muy grande)
- **Time-to-peak MUCHO más rápido en DMT** (1.9 vs 5.0 min, d = -2.051)
- **Threshold crossings más rápidos en DMT** (todos p < 0.01)
- DMT induce respuesta cardiovascular rápida y de gran magnitud vs baseline estable

**Hallazgo clave:** ECG muestra diferencias dramáticas entre DMT y RS en TODAS las features temporales.

---

### 2. Electrodermal Activity (EDA/SMNA)

#### DMT vs RS Baseline:
| Feature | DMT Mean | RS Mean | t | df | p | Cohen's d | Sig |
|---------|----------|---------|---|----|----|-----------|-----|
| **Peak Amplitude** | 8.002 | 1.052 | 5.963 | 10 | **<0.001** | 1.798 | **✓✓✓** |
| **Time to Peak** | 1.750 min | 2.614 min | -2.132 | 10 | 0.059 | -0.643 | Tendencia |
| **Time to 33%** | 0.679 min | 1.536 min | -1.716 | 10 | 0.137 | -0.648 | |
| **Time to 50%** | 0.821 min | 1.857 min | -2.117 | 10 | 0.079 | -0.800 | Tendencia |

**Interpretación:**
- **Peak amplitude MUCHO mayor en DMT** (d = 1.798, efecto muy grande)
- Time-to-peak muestra tendencia hacia onset más rápido en DMT (p = 0.059)
- Threshold crossings muestran tendencias hacia respuesta más rápida en DMT
- DMT induce activación electrodermal de gran magnitud vs baseline

**Hallazgo clave:** EDA muestra diferencia dramática en magnitud (peak amplitude) pero timing menos claro.

---

### 3. Respiration (RESP/RVT)

#### DMT vs RS Baseline:
| Feature | DMT Mean | RS Mean | t | df | p | Cohen's d | Sig |
|---------|----------|---------|---|----|----|-----------|-----|
| **Peak Amplitude** | 1.039 | 0.351 | 4.271 | 11 | **0.001** | 1.233 | **✓✓** |
| **Time to Peak** | 3.792 min | 4.812 min | -1.391 | 11 | 0.192 | -0.402 | |
| **Time to 33%** | 2.271 min | 1.521 min | 1.195 | 11 | 0.257 | 0.345 | |
| **Time to 50%** | 2.792 min | 2.062 min | 1.025 | 11 | 0.327 | 0.296 | |

**Interpretación:**
- **Peak amplitude significativamente mayor en DMT** (d = 1.233, efecto grande)
- Timing features no muestran diferencias significativas
- DMT induce aumento en variabilidad respiratoria vs baseline

**Hallazgo clave:** RESP muestra diferencia robusta en magnitud pero no en timing.

---

## Resumen Comparativo

### Peak Amplitude (Magnitud de Respuesta):

| Modalidad | DMT Mean | RS Mean | Cohen's d | Interpretación |
|-----------|----------|---------|-----------|----------------|
| **ECG** | 2.482 | -0.040 | **4.292** | Efecto muy grande ✓✓✓ |
| **EDA** | 8.002 | 1.052 | **1.798** | Efecto muy grande ✓✓✓ |
| **RESP** | 1.039 | 0.351 | **1.233** | Efecto grande ✓✓ |

**Conclusión:** TODAS las modalidades muestran aumentos significativos y robustos en peak amplitude durante DMT vs RS.

### Time-to-Peak (Velocidad de Onset):

| Modalidad | DMT Mean | RS Mean | Cohen's d | Interpretación |
|-----------|----------|---------|-----------|----------------|
| **ECG** | 1.909 min | 5.023 min | **-2.051** | DMT mucho más rápido ✓✓✓ |
| **EDA** | 1.750 min | 2.614 min | -0.643 | Tendencia más rápido |
| **RESP** | 3.792 min | 4.812 min | -0.402 | No significativo |

**Conclusión:** ECG muestra onset dramáticamente más rápido en DMT. EDA muestra tendencia. RESP no muestra diferencias en timing.

### Threshold Crossings:

| Modalidad | t_33 Sig | t_50 Sig | Interpretación |
|-----------|----------|----------|----------------|
| **ECG** | ✓✓ | ✓✓✓ | Crossings mucho más rápidos en DMT |
| **EDA** | No | Tendencia | Crossings posiblemente más rápidos |
| **RESP** | No | No | Sin diferencias en crossings |

---

## Integración con Análisis Previos

### Consistencia con Análisis Dosis-Dependientes:

**ECG:**
- ✓ Análisis dosis: High > Low en peak amplitude (p = 0.042)
- ✓ Análisis baseline: DMT >> RS en peak amplitude (p < 0.001, d = 4.292)
- **Conclusión:** Efecto robusto tanto dosis-dependiente como vs baseline

**EDA:**
- ✗ Análisis dosis: No diferencias significativas en features
- ✓ Análisis baseline: DMT >> RS en peak amplitude (p < 0.001, d = 1.798)
- **Conclusión:** Efecto robusto vs baseline pero sutil entre dosis

**RESP:**
- ✗ Análisis dosis: Solo tendencia en peak amplitude (p = 0.085)
- ✓ Análisis baseline: DMT > RS en peak amplitude (p = 0.001, d = 1.233)
- **Conclusión:** Efecto robusto vs baseline pero sutil entre dosis

### Patrón General:

1. **Magnitud de respuesta (peak amplitude):**
   - Todas las modalidades: DMT >> RS (efectos grandes a muy grandes)
   - Solo ECG muestra diferencias dosis-dependientes significativas

2. **Velocidad de onset (time-to-peak, thresholds):**
   - ECG: DMT mucho más rápido que RS
   - EDA/RESP: Diferencias menos claras

3. **Implicación científica:**
   - DMT induce cambios autonómicos robustos vs baseline
   - Efectos dosis-dependientes son más sutiles y específicos de modalidad
   - ECG es la modalidad más sensible a efectos dosis-dependientes

---

## Archivos Generados

### Para cada modalidad (ECG, EDA, RESP):

**Datos:**
- `extracted_features.csv` - Features de DMT y RS
- `feature_comparisons.csv` - Comparaciones dosis (DMT High vs Low)
- `baseline_comparison.csv` - Comparaciones DMT vs RS

**Visualizaciones:**
- `feature_comparison.png` - Comparación dosis (DMT High vs Low)
- `baseline_comparison.png` - Comparación DMT vs RS con barras y significancia
- `baseline_comparison_report.txt` - Reporte textual completo

**Ubicación:**
```
results/
├── ecg/hr/supplementary/
│   ├── baseline_comparison.csv
│   ├── baseline_comparison.png
│   └── baseline_comparison_report.txt
├── eda/smna/supplementary/
│   ├── baseline_comparison.csv
│   ├── baseline_comparison.png
│   └── baseline_comparison_report.txt
└── resp/rvt/supplementary/
    ├── baseline_comparison.csv
    ├── baseline_comparison.png
    └── baseline_comparison_report.txt
```

---

## Interpretación para el Paper

### Main Text:
- Análisis FDR time-to-time (one-tailed) para efectos dosis-dependientes
- Análisis por fases para caracterizar dinámica temporal

### Supplementary Materials:

**Sección 1: Análisis Dosis-Dependientes (Features)**
- Comparaciones High vs Low en features extraídos
- ECG muestra diferencias significativas en peak amplitude
- EDA y RESP muestran tendencias

**Sección 2: Comparaciones con Baseline**
- DMT (dosis colapsadas) vs RS
- **Todas las modalidades muestran aumentos robustos en peak amplitude**
- ECG muestra onset dramáticamente más rápido en DMT
- Cuantifica magnitud general del efecto DMT

**Mensaje clave:**
> "DMT induce cambios autonómicos robustos y de gran magnitud comparado con Resting State baseline (Cohen's d = 1.2-4.3 para peak amplitude). Los efectos dosis-dependientes son más sutiles pero detectables, especialmente en ECG (High > Low, p = 0.042)."

---

## Respuesta al Supervisor

✅ **Comentario implementado exitosamente:**

**Supervisor:** "Comparaciones con baseline... altura del máximo, etc."

**Implementación:**
1. ✅ Extracción de features temporales (peak amplitude, time-to-peak, thresholds)
2. ✅ Comparaciones DMT vs RS usando paired t-tests
3. ✅ Visualizaciones con barras y significancia
4. ✅ Reportes textuales completos

**Resultados:**
- **Peak amplitude:** Todas las modalidades muestran diferencias muy significativas (p < 0.001-0.001)
- **Time-to-peak:** ECG muestra diferencias dramáticas (p < 0.001)
- **Threshold crossings:** ECG muestra diferencias significativas

**Conclusión:** Las comparaciones con baseline revelan efectos robustos de DMT en magnitud y timing de respuestas autonómicas, complementando los análisis dosis-dependientes.

---

## Script de Ejecución

**Archivo:** `src/run_supplementary_analyses.py`

**Funcionalidad:**
1. Análisis por fases (onset vs recovery)
2. Extracción de features (peak, timing, thresholds)
3. Comparaciones dosis (DMT High vs Low)
4. **Comparaciones baseline (DMT vs RS)** ← NUEVO

**Ejecución:**
```bash
micromamba run -n dmt-emotions python src/run_supplementary_analyses.py
```

**Módulos utilizados:**
- `scripts/phase_analyzer.py`
- `scripts/feature_extractor.py`
- `scripts/baseline_comparator.py` ← NUEVO
- `scripts/statistical_reporter.py`
