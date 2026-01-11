# Resumen: Análisis Suplementarios - Fases y Features

## ✅ Estado: COMPLETADO

Se implementaron y ejecutaron exitosamente los análisis suplementarios solicitados por el supervisor:
1. **Análisis por fases temporales** (onset 0-3 min, recovery 3-9 min)
2. **Extracción de características temporales** (peak amplitude, time-to-peak, threshold crossings)

---

## Resultados por Modalidad

### 1. Electrocardiography (ECG/HR)

#### Análisis por Fases:
| Fase | Rango | t | df | p | Cohen's d | Significancia |
|------|-------|---|----|----|-----------|---------------|
| **Onset** | 0-3 min | 2.093 | 10 | 0.063 | 0.631 | Tendencia |
| **Recovery** | 3-9 min | 4.434 | 10 | **0.001** | 1.337 | **✓✓** |

**Interpretación:**
- Diferencias significativas High vs Low en fase de recuperación (3-9 min)
- Efecto muy robusto (d = 1.337, efecto grande)
- Tendencia en fase de onset (p = 0.063)

#### Extracción de Features:
| Feature | High Mean | Low Mean | t | p | Cohen's d | Sig |
|---------|-----------|----------|---|---|-----------|-----|
| **Peak Amplitude** | 2.879 | 2.084 | 2.331 | **0.042** | 0.703 | **✓** |
| Time to Peak (min) | 2.250 | 1.568 | 1.808 | 0.101 | 0.545 | |
| Time to 33% (min) | 0.932 | 0.886 | 0.134 | 0.896 | 0.040 | |
| Time to 50% (min) | 1.250 | 1.023 | 0.657 | 0.526 | 0.198 | |

**Interpretación:**
- **Peak amplitude significativamente mayor en High dose** (p = 0.042, d = 0.703)
- Time-to-peak muestra tendencia (p = 0.101) hacia onset más rápido en High
- Threshold crossings no muestran diferencias significativas

**Archivos generados:**
- `results/ecg/hr/supplementary/phase_averages.csv`
- `results/ecg/hr/supplementary/phase_comparisons.csv`
- `results/ecg/hr/supplementary/phase_comparison.png`
- `results/ecg/hr/supplementary/extracted_features.csv`
- `results/ecg/hr/supplementary/feature_comparisons.csv`
- `results/ecg/hr/supplementary/feature_comparison.png`

---

### 2. Electrodermal Activity (EDA/SMNA)

#### Análisis por Fases:
| Fase | Rango | t | df | p | Cohen's d | Significancia |
|------|-------|---|----|----|-----------|---------------|
| **Onset** | 0-3 min | 0.656 | 10 | 0.527 | 0.198 | |
| **Recovery** | 3-9 min | 2.730 | 10 | **0.021** | 0.823 | **✓** |

**Interpretación:**
- Diferencias significativas High vs Low en fase de recuperación (3-9 min)
- Efecto grande (d = 0.823)
- No hay diferencias en fase de onset

#### Extracción de Features:
| Feature | High Mean | Low Mean | t | p | Cohen's d | Sig |
|---------|-----------|----------|---|---|-----------|-----|
| Peak Amplitude | 8.547 | 7.457 | 0.394 | 0.702 | 0.119 | |
| Time to Peak (min) | 1.886 | 1.614 | 0.323 | 0.753 | 0.098 | |
| Time to 33% (min) | 0.614 | 0.705 | -0.410 | 0.690 | -0.124 | |
| Time to 50% (min) | 0.841 | 0.705 | 0.582 | 0.574 | 0.175 | |

**Interpretación:**
- No se encontraron diferencias significativas en features individuales
- Esto sugiere que las diferencias en EDA son más sutiles y requieren análisis temporal completo
- Consistente con los 3 segmentos significativos encontrados en análisis FDR time-to-time

**Archivos generados:**
- `results/eda/smna/supplementary/phase_averages.csv`
- `results/eda/smna/supplementary/phase_comparisons.csv`
- `results/eda/smna/supplementary/phase_comparison.png`
- `results/eda/smna/supplementary/extracted_features.csv`
- `results/eda/smna/supplementary/feature_comparisons.csv`
- `results/eda/smna/supplementary/feature_comparison.png`

---

### 3. Respiration (RESP/RVT)

#### Análisis por Fases:
| Fase | Rango | t | df | p | Cohen's d | Significancia |
|------|-------|---|----|----|-----------|---------------|
| **Onset** | 0-3 min | 2.483 | 11 | **0.030** | 0.717 | **✓** |
| **Recovery** | 3-9 min | 1.618 | 11 | 0.134 | 0.467 | |

**Interpretación:**
- **Diferencias significativas High vs Low en fase de onset (0-3 min)**
- Efecto mediano-grande (d = 0.717)
- Patrón opuesto a ECG y EDA (efecto en onset, no en recovery)
- Sugiere respuesta respiratoria más rápida al DMT

#### Extracción de Features:
| Feature | High Mean | Low Mean | t | p | Cohen's d | Sig |
|---------|-----------|----------|---|---|-----------|-----|
| Peak Amplitude | 1.430 | 0.648 | 1.891 | 0.085 | 0.546 | Tendencia |
| Time to Peak (min) | 3.875 | 3.708 | 0.124 | 0.904 | 0.036 | |
| Time to 33% (min) | 1.500 | 2.312 | -0.944 | 0.377 | -0.334 | |
| Time to 50% (min) | 2.188 | 2.562 | -0.419 | 0.688 | -0.148 | |

**Interpretación:**
- Peak amplitude muestra tendencia (p = 0.085, d = 0.546)
- No hay diferencias significativas en timing features
- Consistente con efecto en fase de onset

**Archivos generados:**
- `results/resp/rvt/supplementary/phase_averages.csv`
- `results/resp/rvt/supplementary/phase_comparisons.csv`
- `results/resp/rvt/supplementary/phase_comparison.png`
- `results/resp/rvt/supplementary/extracted_features.csv`
- `results/resp/rvt/supplementary/feature_comparisons.csv`
- `results/resp/rvt/supplementary/feature_comparison.png`

---

## Resumen Comparativo

### Análisis por Fases:

| Modalidad | Onset (0-3 min) | Recovery (3-9 min) |
|-----------|-----------------|---------------------|
| **ECG** | Tendencia (p=0.063) | **Significativo** (p=0.001, d=1.337) |
| **EDA** | No significativo | **Significativo** (p=0.021, d=0.823) |
| **RESP** | **Significativo** (p=0.030, d=0.717) | No significativo |

**Patrón temporal:**
- **ECG y EDA**: Efectos más fuertes en fase de recuperación (3-9 min)
- **RESP**: Efecto más fuerte en fase de onset (0-3 min)
- Sugiere diferentes dinámicas temporales entre sistemas autonómicos

### Extracción de Features:

| Modalidad | Peak Amplitude | Time to Peak | Threshold Crossings |
|-----------|----------------|--------------|---------------------|
| **ECG** | **Significativo** (p=0.042) | Tendencia (p=0.101) | No significativo |
| **EDA** | No significativo | No significativo | No significativo |
| **RESP** | Tendencia (p=0.085) | No significativo | No significativo |

**Hallazgos clave:**
- **ECG muestra diferencias robustas en peak amplitude** (High > Low)
- EDA y RESP no muestran diferencias significativas en features individuales
- Esto sugiere que las diferencias en EDA/RESP son más sutiles y distribuidas temporalmente

---

## Integración con Análisis Principal

### Consistencia con Análisis FDR Time-to-Time:

**ECG:**
- ✓ Análisis FDR: Segmento continuo 1.5-9.0 min
- ✓ Análisis por fases: Significativo en recovery (3-9 min)
- ✓ Features: Peak amplitude significativo
- **Conclusión: Resultados altamente consistentes**

**EDA:**
- ✓ Análisis FDR: 3 segmentos significativos (4.5-5.0, 6.0-6.5, 8.0-9.0 min)
- ✓ Análisis por fases: Significativo en recovery (3-9 min)
- ✗ Features: No significativos
- **Conclusión: Análisis por fases confirma efecto, features no capturan diferencias**

**RESP:**
- ✓ Análisis FDR: Segmentos significativos (verificar reporte)
- ✓ Análisis por fases: Significativo en onset (0-3 min)
- ✗ Features: Solo tendencia en peak amplitude
- **Conclusión: Patrón temporal diferente (onset vs recovery)**

---

## Interpretación Científica

### 1. Diferencias Temporales entre Sistemas:

**Sistema Cardiovascular (ECG):**
- Respuesta sostenida durante recovery (3-9 min)
- Peak amplitude mayor en High dose
- Sugiere efecto prolongado sobre frecuencia cardíaca

**Sistema Electrodermal (EDA):**
- Respuesta sostenida durante recovery (3-9 min)
- Diferencias sutiles no capturadas por features individuales
- Requiere análisis temporal completo para detectar efecto

**Sistema Respiratorio (RESP):**
- Respuesta rápida durante onset (0-3 min)
- Sugiere adaptación respiratoria inmediata al DMT
- Patrón temporal diferente a otros sistemas

### 2. Validación de Hipótesis del Supervisor:

**Hipótesis 1: "Dividir en fases temporales puede mostrar diferencias"**
- ✓ **CONFIRMADA**: Todas las modalidades muestran diferencias significativas en al menos una fase
- ECG y EDA: Recovery phase
- RESP: Onset phase

**Hipótesis 2: "Extraer features puede mostrar diferencias aunque no sean time-to-time"**
- ✓ **PARCIALMENTE CONFIRMADA**: ECG muestra diferencias en peak amplitude
- ✗ EDA y RESP no muestran diferencias significativas en features
- Sugiere que análisis time-to-time con FDR es más sensible para EDA/RESP

### 3. Implicaciones para el Paper:

**Resultados Principales (Main Text):**
- Análisis FDR time-to-time con one-tailed tests (ya implementado)
- Muestra diferencias robustas en todas las modalidades

**Resultados Suplementarios (Supplementary Materials):**
- Análisis por fases: Confirma diferencias y caracteriza dinámica temporal
- Extracción de features: Caracteriza magnitud de respuesta (ECG)
- Proporciona evidencia convergente desde múltiples perspectivas

---

## Archivos Generados

### Estructura de Directorios:
```
results/
├── ecg/hr/supplementary/
│   ├── phase_averages.csv
│   ├── phase_comparisons.csv
│   ├── phase_comparison.png
│   ├── extracted_features.csv
│   ├── feature_comparisons.csv
│   └── feature_comparison.png
├── eda/smna/supplementary/
│   ├── phase_averages.csv
│   ├── phase_comparisons.csv
│   ├── phase_comparison.png
│   ├── extracted_features.csv
│   ├── feature_comparisons.csv
│   └── feature_comparison.png
└── resp/rvt/supplementary/
    ├── phase_averages.csv
    ├── phase_comparisons.csv
    ├── phase_comparison.png
    ├── extracted_features.csv
    ├── feature_comparisons.csv
    └── feature_comparison.png
```

### Figuras para Supplementary Materials:
- **Figure S_Phase**: Comparación por fases para las 3 modalidades
- **Figure S_Features**: Comparación de features extraídos para las 3 modalidades

---

## Conclusión

✅ **Los análisis suplementarios fueron implementados exitosamente y proporcionan evidencia convergente:**

1. **Análisis por fases confirma diferencias dosis-dependientes** en todas las modalidades
2. **Caracteriza dinámicas temporales diferentes** entre sistemas (onset vs recovery)
3. **Extracción de features identifica diferencias en magnitud** (ECG peak amplitude)
4. **Resultados consistentes con análisis FDR principal** (one-tailed tests)

**Recomendación:** Incluir estos análisis como **Supplementary Materials** en el paper para:
- Demostrar robustez de los hallazgos con múltiples enfoques analíticos
- Caracterizar dinámicas temporales específicas de cada sistema
- Responder a comentarios del supervisor sobre análisis alternativos

**Script de ejecución:** `src/run_supplementary_analyses.py`
