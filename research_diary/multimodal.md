
[x] Crear un unico panel con ECG (HR), EDA (SCL) y Resp (RVT). Este es el (por ahora) `panel_6`

SEGUIR DESDE ACA

[ ]  Hacer que el color con que esta escrito `Electrocardiography`, `Electordermal Activity` y `Respiration` sean colores distintivos de esa señal fisiologica. Y que esos 3 colores sean con los cuales (en disitnos matices tal vez) se pinten los betas y CIs de los subplots de la derecha. Asi qeuda claro qeu es forma parte de la medida que le corresponde.

[ ] Crear la caption de este panel multimodal en la seccion de resultados del paper  (y eliminar luego todas las figuras que ya no van ahi)

[ ] El resto de las figuras van a pasar a materiales suplpementarios

1. Figuras de las 3 medidas de 19 min long
2. Figras de las 3 medidas stackeadas para todos lso participantes

- Adaptar las captions y el texto de la seccion de resulados de forma acorde a estas nuevas figuras
- Cambiar tambien (y extender) la seccion de materiales suplementarios.

[ ] Arreglar los filenames de las figuras, dejando claro cuales son figuras principales (numeradas en orden) y cuales de materiales suplementarios (y eliminar los paneles deprecados)

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