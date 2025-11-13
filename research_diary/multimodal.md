
[x] Crear un unico panel con ECG (HR), EDA (SCL) y Resp (RVT). Este es el (por ahora) `panel_6`

SEGUIR DESDE ACA

[x]  En el panel_6, hacer que el color con que esta escrito `Electrocardiography`, `Electordermal Activity` y `Respiration` en el label del eje Y en los subplots A, C y E sean colores distintivos de esa señal fisiologica. Y que esos 3 colores sean con los cuales (en disitnos matices tal vez) se pinten los betas y CIs de los subplots de la derecha (en los subplots B, D y F, respectivamnete). Asi qeuda claro qeu es forma parte de la medida que le corresponde. Moodificar #generate_phys_figures.py considerando esto.


[x] Editar #generate_phys_figures.py para que el `panel_6` pase a ser `figure_1`.
Lo que es actualmente `panel_1` deberia desaparecer, y en cambio deberia generarse un plot llamado `figure_S1` similar al panel `panel_4.png` pero con los plots resultantes de correr `run_eda_smna_analysis (subplot A: `all_subs_smna.png`, subplot B: `lme_coefficient_plot.png`). 
`panel_3.png` deberia llamarse ahora `figure_S2`
`panel_4.png` y `panel_5.png` deberian eliminarse del script (no los quiero mas como output de correr el script).

[x] Quiero cambiar los colores de los subplots de `figure_1`. Esto quiere decir, que quiero cambiar los colores con que estan escritos `Electrocardiography`, `Electordermal Activity` y `Respiration` en el label del eje Y en los subplots A, C y E sean colores distintivos de esa señal fisiologica. Y que esos 3 colores sean con los cuales (en disitnos matices tal vez) se pinten los betas y CIs de los subplots de la derecha (en los subplots B, D y F, respectivamnete). Asi qeuda claro qeu es forma parte de la medida que le corresponde. Pero tambien quiero cambiar los colores de las comparaciones dentro de RS (low, y high) y DMT (low, y high) para que tambein usen dos matices distintos del mismo color. De este modo, cada señal (i.e. `Electrocardiography`, `Electordermal Activity` y `Respiration`)
tendra colores distintivos para cada uno de sus dos subplots.
Quiero usar rojo, azul y verde como los tres colores para las 3 señales fisiologicas (y los matices de esos colores para los subplots)
Modifica los siguintes scripts para que estos cambios se vean reflejados:
scripts/run_ecg_hr_analysis.py
scripts/run_eda_scl_analysis.py
scripts/run_resp_rvt_analysis.py
Empecemos modificando `run_ecg_hr_analysis.py` para ver si esto genera los cambios que quiero generar.

[x] Crear la caption de este panel multimodal en la seccion de resultados del paper  (y eliminar luego todas las figuras que ya no van ahi)



[x] Pasar todas las figuras que correspondan a material suplementario.

1. Figuras de las 3 medidas de 19 min long (o dejo solo la de ECG como esta ahora?). Dejo solo la de ECG, como esta ahora.
2. Figras de las 3 medidas stackeadas para todos lso participantes (o stackeo nomas para los sujetos qeu tenga para las 3 medidas?). Por ahora stackeo todas.

- Adaptar las captions y el texto de la seccion de resulados de forma acorde a estas nuevas figuras
- Cambiar tambien (y extender) la seccion de materiales suplementarios.

[x] Arreglar los filenames de las figuras, dejando claro cuales son figuras principales (numeradas en orden) y cuales de materiales suplementarios (y eliminar los paneles deprecados)

[x] Revisar que la caption de la tabla S1 este bien (esta chequeando eso desde chatgpt y claude). Si eso no marca errores, seguir con el resto de los comentarios del documento.


[x] Revisar todos los comentarios que estan en el documento (seccion resultados)

SEGUIR DESDE ACA ->

Analisis multimodales
---------------------

Diego me dijo que podria iterar con chatgpt para usar los 6 o 7 sujetos, y hacer un LME con esos participantes.
De esta forma, ademas de tener coeficientes por cada una de las condiciones, ademas tendria un coeficientes particulares por el tipo de señal y un coeficiente por el promedio de señales. Hace sentido esto? iterarlo con chatgpt.

[x] Generar analisis y figuras compuestas de LME con PCA incorproado.

[x] Escribri las seciones de RESULTADOS considerando estos analisis realizados.

[x] Hacer analisis extra de correlaciones cruzadas / coherencia.
    No me dio nada muy interesante esto, asi qeu lo exclui.

[x] Escribri las seciones de METODOS considerando estos analisis realizados

SEGUIR DESDE ACA ->
Queda para terminar con esta parte fisiologica (antes de ir a los TET, y la relacion TET con señales)
[ ] Revisar el delta SCL, que me parece que me esta trayendo algunos problemas. 
Lo que hice fue lo siguiente (ver la instruccion):
    Objetivo y regla clave

    Usar RS como línea de base por sesión y participante.

    Hacer el z-scoring directamente sobre la señal continua de SCL, sin agregar ni resumir antes.

    Para cada ventana de un minuto, la métrica de agregación ya no es AUC, sino el promedio de la ventana del SCL z-scoreado.

    RS es siempre pre DMT. No se esperan diferencias por dosis en RS. Se utilizará como chequeo de QC.

    Cambios en el script, paso a paso
    1) Configuración general

    Añade banderas de configuración para activar el flujo “RS como base” y, si quieres, exportar tanto escala absoluta como z.

    Define trims opcionales para RS y para el inicio de DMT.

    Define un mínimo de muestras por ventana para aceptar una sesión.

    2) Escalado por sesión usando RS

    Para cada sujeto y sesión, carga las series RS y DMT como ya haces.

    Recorta RS y DMT con los trims si fueran necesarios.

    Ajusta media (mu) y desviación (sigma) usando únicamente la serie de RS de esa sesión.

    Aplica esos parámetros a la serie completa de RS y a la serie completa de DMT de esa sesión para obtener SCL_z.

    Este z-scoring se hace en la señal continua, sin ningún promedio previo.

    Si sigma es 0 o no es finita, marca esa sesión como no escalable, registra el motivo y exclúyela del análisis z. Conserva sólo salidas descriptivas en absoluto si hiciera falta.

    3) Definición de ventanas y agregación

    Mantén el esquema de 9 ventanas de 1 minuto (0–8) que ya usas.

    Para cada minuto y condición, extrae la ventana correspondiente de la señal SCL_z ya escalada.

    Calcula el promedio de SCL_z en esa ventana como métrica final.

    Esta métrica reemplaza por completo a AUC en el long-format.

    Opcional: si quieres conservar una columna espejo en escala absoluta para controles descriptivos, calcula también el promedio de la señal SCL absoluta por ventana. No mezcles AUC.

    4) Reemplazo de la construcción del long-format

    La función que hoy construye la tabla larga por minuto debe:

    Estimar mu y sigma con RS por sesión.

    Generar SCL_z continuo para RS y DMT de esa sesión.

    Calcular promedio por ventana para RS y DMT en SCL_z.

    Devolver una tabla larga con columnas: subject, minute (1..9), State (RS, DMT), Dose (Low, High), WindowMean (promedio de SCL_z), y una columna Scale con valor z.

    Si decides exportar también absoluto, añade filas paralelas con Scale = 'abs' y WindowMean_abs como promedio de SCL sin escalar. Evita nombres o archivos con “AUC”.

    5) Nombres de archivos y columnas

    Cambia el nombre del archivo principal guardado a algo del tipo:
    results/eda/scl/scl_window_mean_long_data_all_scales.csv

    Si vas a alimentar el modelo con z por defecto, guarda también:
    results/eda/scl/scl_window_mean_long_data_z.csv

    Sustituye todas las referencias a AUC por WindowMean en variables, columnas y etiquetas.

    Evita conservar funciones o nombres que aludan a AUC para este flujo. Si quedan utilidades de AUC por compatibilidad, no las uses en el pipeline principal.

    6) Modelo LME y estadísticas

    Alimenta el LME con el DataFrame filtrado a Scale == 'z'.

    La variable dependiente pasa a ser WindowMean (promedio por minuto del SCL z-scoreado).

    Mantén la misma estructura de factores: State × Dose más efectos de tiempo (minute centrado) e interacciones con tiempo si sigues ese enfoque.

    Actualiza el reporte de modelo y los textos de especificación para que indiquen:
    “Dependent variable: per-minute mean of z-scored SCL”.

    7) Gráficos y etiquetas

    En las funciones de gráficos basadas en el long-format:

    Cambia el eje Y para reflejar que se trata de promedio de SCL z-scoreado por ventana.

    Ajusta títulos, leyendas y captions para sustituir “AUC” por “Window mean (z-SCL)” o “SCL z-mean”.

    Los gráficos basados en series continuas (p. ej. “combined summary” o “DMT only”) pueden mantenerse en absoluto si no deseas duplicar la lógica. Si decides hacer versiones en z, aplica el mismo escalado de RS a las series continuas antes de re-muestrear y trazar.

    8) QC de la premisa “RS no difiere por dosis”

    Genera un chequeo simple con el dataset en z: media de WindowMean para RS por dosis y guarda un TXT con esos promedios.

    Es descriptivo y sirve para documentar que RS, al ser pre DMT, no muestra una separación sistemática por dosis.

    9) Reportes y captions

    En el informe de análisis reemplaza toda mención a AUC por “per-minute mean of z-scored SCL”.

    En las captions de figuras, actualiza la descripción de la variable dependiente y del eje Y.

    Indica claramente que el z-scoring se calculó con RS de cada sesión y que esos parámetros se aplicaron a DMT de esa misma sesión.

    10) Compatibilidad y limpieza

    Mantén BASELINE_CORRECTION desactivado, ya que el z-scoring por RS cumple el rol de normalización.

    Si dejas utilidades de AUC para otros análisis, no las llames desde el flujo principal.

    Documenta en el README del proyecto que el análisis de SCL ahora usa promedios por ventana de SCL z-scoreado como métrica principal y que el z-scoring se realiza sobre la señal continua, no sobre agregados.

```

[x] Implementar el Z scoring por sujeto (y no por sesion) para ECG
[x] Implementar el Z scoring por sujeto (y no por sesion) para EDA
[x] Implementar el Z scoring por sujeto (y no por sesion) para Resp

[x] Crear nuevamente los plots con de arousal index


Ya hice todos los cambios necesarios en los plots, y analisis estadisticos de las señales a nive unimodal y PC1. AHora falta cabiar lo necsario en el manuscrito.

[x] Cambiar la seccion de Metodos y Resultados, de acuerdo a esta nueva forma de tratar los datos
[x] Replicar los analisis de Tomi de los datos de TET

