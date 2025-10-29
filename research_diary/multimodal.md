
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

SEGUIR DESDE ACA ->

[ ] Pasar todas las figuras que correspondan a material suplementario.

1. Figuras de las 3 medidas de 19 min long (o dejo solo la de ECG como esta ahora?)
2. Figras de las 3 medidas stackeadas para todos lso participantes (o stackeo nomas para los sujetos qeu tenga para las 3 medidas?)

- Adaptar las captions y el texto de la seccion de resulados de forma acorde a estas nuevas figuras
- Cambiar tambien (y extender) la seccion de materiales suplementarios.

[ ] Arreglar los filenames de las figuras, dejando claro cuales son figuras principales (numeradas en orden) y cuales de materiales suplementarios (y eliminar los paneles deprecados)

[ ] Revisar todos los comentarios que estan en el documento


Analisis multimodales
---------------------

[ ] Hacer analisis multimodales con los 6 sujetos compartidos con las 3 señales fisiologicas:
S04, S06, S07, S16, S18, S20 (n = 6).
Hacer plots tipo radar plots, pero simplemente con estas 3 señales (un triangulo) que busque patrones comunes en los sujetos para estar 3 señales.
El objetivo seria ver que patrones comunes se puede extraer de las 3 señales combinadas.
Por ejemplo, tal vez se puede hacer una correlacion de las metricas agregadas por minuto de:
- SCL
- HR
- RVT
Para medir consistencia de la fisiologia.
Tambien la idea seria ver cuanto explica una unica componente las 3 señales (esperamos que pueda explicar mucho).

---

[ ] Hacer una figura de la metodologia, usando al imagen de Evan para crear en chatgpt u boceto de la toma de datos. O si no hacerla usando otra app. Usar los colores que defini mas arriba para las 3 señales fisiologicas para marcar en esa figura las 3 señales registradas