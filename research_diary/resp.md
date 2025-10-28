```markdown
# Pipeline de Análisis de Datos de Respiración  
**Archivo:** `run_resp_analysis.py`  
**Propósito:** reproducir el análisis estadístico y visualización de los efectos fisiológicos de DMT sobre la respiración, siguiendo el mismo formato y estructura que el análisis de ECG/EDA.  

---

## 1. Preparación del entorno de trabajo  

### Requerimientos
- **Python 3.x**
- Librerías:
  - [`NeuroKit2`](https://doi.org/10.3758/s13428-020-01516-y)
  - [`BioSPPy`](https://doi.org/10.1016/j.softx.2024.101712)
  - `numpy`, `pandas`, `matplotlib`, `statsmodels`, `scipy`
- Asegurarse de replicar los estilos de figuras y convenciones gráficas del script de ECG (`run_ecg_hr_analysis.py`):
  - Colores:  
    - RS High → verde (`tab:green`)  
    - RS Low → violeta (`tab:purple`)  
    - DMT High → rojo (`tab:red`)  
    - DMT Low → azul (`tab:blue`)  

---

## 2. Estructura de carpetas y archivos  

Los datos se encuentran en:  
```

./data/derivatives/preprocessing/phys/resp/dmt_high/
./data/derivatives/preprocessing/phys/resp/dmt_low/

```

Cada archivo corresponde a un participante y sesión, por ejemplo:  
`S02_dmt_session2_high.csv`

**Columnas esperadas:**
```

time, RSP_Raw, RSP_Clean, RSP_Amplitude, RSP_Rate, RSP_RVT,
RSP_Phase, RSP_Phase_Completion, RSP_Symmetry_PeakTrough,
RSP_Symmetry_RiseDecay, RSP_Peaks, RSP_Troughs

````

---

## 3. Selección y justificación de métricas  

Basado en literatura de neurofisiología respiratoria y en el workshop [ASSC Physio 2025](https://github.com/irebollo/ASSC_Physio_2025/tree/main/Respiration):

| Métrica | Descripción | Interpretación fisiológica | Uso en modelo |
|----------|--------------|----------------------------|----------------|
| **RSP_Rate** | Respiraciones por minuto | Aceleración respiratoria (activación simpática) | Variable principal |
| **RSP_Amplitude** | Amplitud media de respiración | Profundidad/inspiración más amplia (relajación o activación) | Variable secundaria |
| **RSP_RVT** | Respiratory Volume per Time = Amplitud × Frecuencia | Índice compuesto de ventilación | Variable integrada |
| *(Opcional)* **RRV / Symmetry** | Variabilidad o asimetría ciclo a ciclo | Control respiratorio fino | Exploratorio |

Estas métricas ya se encuentran calculadas en los CSV de NeuroKit2, por lo que no es necesario recalcularlas a menos que se quiera verificar artefactos.

---

## 4. Construcción del dataset largo  

1. Definir lista de participantes válidos (`SUJETOS_VALIDADOS_RESP`).  
2. Determinar qué sesión es alta o baja dosis usando `get_dosis_sujeto()`.  
3. Cargar los archivos `RS` y `DMT` correspondientes para cada sujeto y dosis.  
4. Para cada minuto entre **0–9 min post t₀**, calcular:

   ```python
   mean_rate = np.nanmean(RSP_Rate[min_sec:max_sec])
   mean_amp  = np.nanmean(RSP_Amplitude[min_sec:max_sec])
   mean_rvt  = np.nanmean(RSP_RVT[min_sec:max_sec])
````

Filtros fisiológicos recomendados:

* `5 ≤ RSP_Rate ≤ 40` respiraciones/min
* `RSP_Amplitude > 0`

5. Guardar los promedios en formato largo:

   | subject | minute | Task | Dose | RSP_Rate | RSP_Amplitude | RSP_RVT |
   | ------- | ------ | ---- | ---- | -------- | ------------- | ------- |
   | S02     | 1      | DMT  | High | 13.4     | 0.85          | 11.4    |
   | S02     | 1      | RS   | High | 10.1     | 0.63          | 6.2     |

6. Exportar el dataset:

   ```
   results/resp/resp_minute_long_data.csv
   ```

---

## 5. Modelado estadístico (LME)

### Especificación general

Para cada métrica (`RSP_Rate`, `RSP_Amplitude`, `RSP_RVT`), ajustar un **modelo lineal mixto**:

```python
formula = 'RSP_Rate ~ Task * Dose + minute_c + Task:minute_c + Dose:minute_c'
model = mixedlm(formula, df, groups=df['subject'])
fit = model.fit()
```

Donde:

* **Task:** RS vs DMT (intra-sesión)
* **Dose:** Low vs High (inter-sesión)
* **minute_c:** tiempo centrado (1–9 minutos)
* **Efectos aleatorios:** intercepto por sujeto

### Corrección FDR

Aplicar corrección Benjamini–Hochberg dentro de familias de hipótesis:

* Task
* Dose
* Task × Dose

### Contrastes simples

* High – Low dentro de RS
* High – Low dentro de DMT
* Interacción: (High–Low en DMT) – (High–Low en RS)

Guardar resultados y métricas del modelo en:

```
results/resp/lme_analysis_report.txt
```

---

## 6. Visualización

### 6.1. Gráficos de coeficientes

Forest plot de β ± IC para cada efecto fijo (Task, Dose, Time, interacciones).

### 6.2. Trayectorias temporales

Graficar `mean ± SEM` de cada condición:

* Panel izquierdo: RS High vs RS Low
* Panel derecho: DMT High vs DMT Low

Usar sombreado gris para intervalos con diferencias significativas (FDR-corr.).

### 6.3. Efectos principales e interacciones

* **Main Task Effect:** promedio RS vs DMT (colapsando dosis).
* **Task × Dose Interaction:** dos paneles comparando High vs Low en RS y DMT.

Guardar figuras en:

```
results/resp/plots/
```

### 6.4. Gráficos individuales

Generar figura “stacked per subject” con RS (izq.) y DMT (der.) mostrando curvas High/Low para cada participante.

---

## 7. Reportes y tablas

1. Generar un archivo `.txt` con:

   * Resumen de modelo (AIC, BIC, N, varianzas)
   * Coeficientes con p_FDR
   * Contrastes condicionales
   * Promedios por celda Task×Dose

2. Exportar tablas descriptivas y CSVs de estadísticas agregadas:

   ```
   summary_statistics.csv
   effect_sizes_table.csv
   model_summary.txt
   ```

---

## 8. Validación y control de calidad

* Verificar convergencia del modelo (`fit.converged`).
* Inspeccionar residuos y outliers.
* Revisar trazas minuto a minuto para posibles artefactos (e.g., hiperventilación transitoria).
* Confirmar consistencia temporal con HR y EDA (aumento inicial seguido de decaimiento).

---

## 9. Interpretación fisiológica esperada

| Métrica             | Tendencia esperada durante DMT                     | Interpretación                      |
| ------------------- | -------------------------------------------------- | ----------------------------------- |
| **RSP_Rate**        | Aumento breve (~2–3 min) seguido de estabilización | Activación autonómica simpática     |
| **RSP_Amplitude**   | Aumento o mantenimiento alto en dosis alta         | Inspiraciones más profundas         |
| **RSP_RVT**         | Aumento sostenido en DMT High                      | Incremento de ventilación total     |
| **RRV / Asimetría** | Posible aumento en irregularidad                   | Disrupción del control respiratorio |

Estas tendencias pueden compararse con las observadas en HR y SCL para evaluar coherencia del perfil autonómico.

---

## 10. Estructura final esperada de resultados

```
results/
└── resp/
    ├── resp_minute_long_data.csv
    ├── lme_analysis_report.txt
    ├── model_summary.txt
    ├── summary_statistics.csv
    ├── effect_sizes_table.csv
    └── plots/
        ├── lme_coefficient_plot.png
        ├── marginal_means_all_conditions.png
        ├── task_main_effect.png
        ├── task_dose_interaction.png
        ├── all_subs_resp_rate.png
        ├── all_subs_dmt_resp_rate.png
        └── stacked_subs_resp_rate.png
```

---

## 11. Referencias

1. Makowski D, Pham T, Lau ZJ, Brammer JC, Lespinasse F, Pham H, et al. *NeuroKit2: A Python toolbox for neurophysiological signal processing.* Behav Res Methods 2021;53:1689–96.
2. Bota P, Silva R, Carreiras C, Fred A, da Silva HP. *BioSPPy: A Python toolbox for physiological signal processing.* SoftwareX 2024;26:101712.
3. Irebollo, I. (2025). *ASSC Physio Workshop — Respiration Analysis Scripts.* GitHub Repository.
4. Boucsein W. *Electrodermal Activity* (2nd ed.). Springer; 2012. (para comparaciones autonómicas).

---

**Output final:**
El script `run_resp_analysis.py` debe replicar la estructura del pipeline de ECG (lectura, agregación por minuto, modelado LME, corrección FDR, figuras, reportes y captions), adaptando las funciones para trabajar con las columnas de respiración.
De esta forma, el/la asistente podrá ejecutar el análisis completo de respiración de manera reproducible y consistente con los otros módulos fisiológicos del estudio DMT.

```
```

[x] Ahora crear un script analogo a `run_resp_amplitude_analysis.py` y run_resp_rate_analysis.py` pero que esta vez tome `RSP_RVT` como metrica de comparacion. De esta forma, el script nuevo se debe llamar `run_resp_rvy_analysis.py`


[x] Dejar funcional `run_resp_amplitude_analysis.py y ver sus resultados

# SEGUIR DESDE ACA (ya esta corriendo a la derecha...)

[ ] Ahora crear un script analogo a `run_resp_rvy_analysis.py`  pero que esta vez tome `RRV ` como metrica de comparacion. De esta forma, el script nuevo se debe llamar `run_resp_rrv_analysis.py`

[ ] Claro — vamos a examinar con más detalle los análisis realizados en el tutorial de respiración del repositorio NeuroKit2 (y el folder “Respiration” del workshop ASSC_Physio_2025), pero adaptándolos a **Python** (en lugar de MATLAB) para que puedas comprender cómo se implementan y cómo los estás usando en tu estudio con la señal de respiración.

---

[ ] Hacer un script que se guarde en `./test/resp` (nuevo dir) que haga la validacion de la data de respiracion.

### Validación de calidad y control de artefactos

Revisar que los picos/troughs sean razonables, que no haya intervalos respiratorios absurdamente cortos o largos, y que la señal esté limpia antes de derivar métricas. En Python, podrías:

```python
# Filtrar tasas fuera de rango:
agg_filtered = agg[(agg['RSP_Rate'] >= 5) & (agg['RSP_Rate'] <= 40)]
```

Y también visualizar unos cuantos segundos de la señal con picos/troughs:

```python
import matplotlib.pyplot as plt
plt.plot(signals['time'], signals['RSP_Clean'])
plt.scatter(signals.loc[signals['RSP_Peaks']==1, 'time'], 
            signals.loc[signals['RSP_Peaks']==1, 'RSP_Clean'], color='red')
plt.show()
```

Esto ayuda a ver si la detección de respiraciones fue adecuada.

### Visualización exploratoria

Hacer gráficos de la señal respiratoria, picos/troughs, tasas temporales, etc. En Python, se puede hacer:

```python
# Ejemplo: graficar RSP_Rate en el tiempo
plt.plot(signals['time'], signals['RSP_Rate'])
plt.xlabel("Time (s)")
plt.ylabel("Resp Rate (rpm)")
plt.show()
```

O bien boxplots/comparaciones entre condiciones.


---

SEGUIR DESDE ACA:

[x] Dejar bien puesto los `y labels` tanto de EDA, ECG y Resp (marcando que señal es y tambien la metrica extraida).
