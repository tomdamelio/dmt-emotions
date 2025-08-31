# Tesis de Licenciatura en Ciencias Físicas

La idea de este repositorio es tener a mano todo el código que vaya escribiendo para hacer análisis de los datos de DMT y los experimentos de Emociones.

Explicación de cada código:

- Estudio con NeuroKit es el primero que hice, ese te plotea la señal que vos elijas procesando EDA, ECG y Resp, después te plotea lo de HRV.
- Estudio multiple con NeuroKit es igual que el anterior sólo que repite el estudio para todos los quieras.
- Dsps todos los que dicen "Adaptando", es lo mismo que hace el primero pero con la posibilidad de que podes tocar dentro del paquete de neurokit porque defini las funciones que necesitaba para correrlo.
- Eda_process con cvxEDA es otro "Adaptando" solo que para cvxEDA, pq necesitaba una config distinta
- Correlación es el que chequea correlación para EDA, divide en dosis alta y dosis baja.
- Ploteo EDA promedio es el que resta el promedio de reposo a la señal con dosis, dividiendo en dosis alta y dosis baja, calculando el promedio entre sujetos y el error estandar y grafica la comparación de esto entre dosis alta y dosis baja.
- Ploteo HR promedio es igual al EDA pero con HR.
