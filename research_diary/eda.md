Sabado 30 de Agosto
--------------------
[x] Armar en `docs/` el readme del proyecto con el orden en que deben correrse.
[x] Guardar los archivos de datos viejos en `../data/old/`
[x] Guardar la data originar en `../data/original/`
[x] Pushear el proyecto en un nuevo repo (ahora esta en el repo de Tomi)

Domingo 31 de Agosto
--------------------
[x] Editar `preprocess_eda.py` para usar 15 segundos extra (20:15 min DMT, 10:15 min Reposo) para plots más limpios.
[x] Crear archivo `config.py` con todos los paths y datos de config del proyecto.
[x] Adaptar preprocess_eda.py para extraer todas las metricas de 
interes de EDA
[x] Hacer testeo exploratorio de la version actual de `preprocess_eda.py` en un nuevo script a guardarse en `./test`.
Este script debe testear las señales de EDA extraidas para el sujeto de prueba (S04). Para eso, es necesario primero  filtrar el dataframe guardado en el .csv para quitar la columna ´time´. Luego de eso, con `eda_plot()` puedo plotear los 4 archivos resultantes.

Comentario -> Parece que la extracción de la actividad de SMNA no esta resuelta en NueroKit2, y de hecho parece que cvxEDA podria tener mejoras (e.g con transformers o a partir de sumar un prior sparse. Pero indagar esto va mas alla de los objetivos de este trabajo, y voy a continuar con los datos extraidos como variables de NeuroKit2).

[x] Generalizar `preprocess_eda.py` para que sea `preprocess_phys.py`y
y obtener asi todos los features tambien de resp y ecg (ademas de eda).
[x] Crear `test_phys_preprocessing.py` para testear que los plots esten bien para las 3 medidas (con la generalizacion de `preprocess_eda.py` a `preprocess_phys.py`).
[x] Correr `python scripts/preprocess_phys.py` que me deberia preprocesar todos los particiapantes (TODOS y no solo los validos).

Lunes 15 de Septiembre
[x] Modificar `test_phys_preprocessing.py`para que al correrlo (por ahora solo para el S01) me permita anotar manualmente en consola / terminal las anotaciones que luego se reflejaran en el json (ver los cambios que hice en el chat de cursor a la derecha, para entener el proceso).


[x] En `preprocess_phys.py`, implementar el metodo `kbk_scr()` de biosppy como metodo extra / complementario sobre la señal de EDA. Esto nos dara 3 arrays (i.e. SCR onsets, SCR peaks and SCR amplitudes) que deben combinarse en un unico DataFrame y guardarse como un archivo extra al archivo de eda, agregandole el sufijo (`_kbk_scr`). Por ejemplo, si el archivo resultante de EDA de correr `preprocess_phys.py` es `S02_dmt_session2_high.csv`, este nuevo archivo complementario sera  `S02_dmt_session2_high_kbk_scr.csv`

[x]  En `preprocess_phys.py`, implementar exactamente lo mismo que hice en el paso anterior con `kbk_scr()` pero con la funcion `emotiphai_eda()` (que tambien devuelve 3 arrays de onsets, peaks y amplitudes). Del mismo modo, el archivo resultante podria llamarse (a modo de ejemplo) `S02_dmt_session2_high_kbk_scr_emotiphai_eda.csv`


[x] Finalmente,  en `preprocess_phys.py` guardar un cuarto archivo complementario de EDA. Esta vez, se imlementara el metodo `biosppy.signals.eda.cvx_decomposition()` como metodo extra / complementario sobre la señal de EDA. Esto nos dara 3 arrays (i.e. edr, smna, edl) que debe guardarse como un archivo extra al archivo de eda, agregandole el sufijo (`_cvx_decomposition`). El resto de las cuatro variables que devuelve cvx_decomposition (i.e. tonic_coeff, linear_drift, res, obj) no quiero qeu se guarden. En este proceso por ejemplo, si el archivo resultante de EDA de correr `preprocess_phys.py` es `S02_dmt_session2_high.csv`, este nuevo archivo complementario sera  `S02_dmt_session2_high_cvx_decomposition.csv`

[x] Agregar en `preprocess_phys.py` algun paso de validacion para corroborar que los lenghts de los archivos de kbk_scr, emotiphai_eda y cvx_decomposition tenga la misma dimensioanlidad (imprimir los lenghts de ambos archivos e incluir un assert de eso)

Por ahora tengo implementado en `preprocess_phys.py` para obtener los features de las 3 señales y ademas el emotiphai_eda y cvx_decomposition (kbk_scr lo borre por una cuestion de incompatibilidades de dependencias con numpu).
Recien chequee que los plots de `test_phys_preprocessing.py` me dan algo decente, asi que ahora dejo corriendo `preprocess_phys.py` para todos los participantes. `test_phys_preprocessing.py` todavia puede mejorarse, sobre todo tal vez se puede combinar la infom de emotiphai_eda con la info de cvx eda (todo junto en un mismo plot).
Ademas, tengo la idae de quedarme con la envolvente de SMNA como proxy de arousal (aunque tal vez el SCR extraido con el cxvEDA es suficiente)

[x] Quiero que en `test_phys_preprocessing.py`, al plot de cvxEDA le sumes los eventos extraido de "Emotiphai SCR". En concreto, quiero que en el plot de cvxEDA incluyas el `Onset` y el `Peak` como un punto en la serie temporal de cvxEDA. A su vez, representa `Amplitud` como una barra vertical qeu desciende desde el `Peak`. Tene en cuenta que el plot de cvxEDA esta en segundos mientras que el de Emotiphai tiene los timestamps en timepoints (sampling rate de 250Hz).

[x] Fijare si corrio bien `preprocess_phys.py` para todos los participantes

----
24/9
----

[x] Creo que esta fallando la forma en que se itera para buscar archivo en @test_phys_preprocessing.py.

Por ejemplo, el orden en que se estan procesando los archivos actualmente es el siguiente:

s01 ses02 dmt low 12 min eda
s01 ses 02 rest low 6 min eda

s01 ses02 dmt low 12 min ecg
s01 ses 02 rest low 6 min ecg

s01 ses02 dmt low 12 min resp
s01 ses 02 rest low 6 min resp

(luego no se proceso ni el s02 ni el s03, sino que pasa directamente al s04)

s04 ses01 dmt high 12 min eda
s04 ses01 rest high 6 min eda
s04 ses02 dmt low 12 min eda
s04 ses02 rest low 6 min eda

s04 ses01 dmt high 12 min ecg
s04 ses01 rest high 6 min ecg
s04 ses02 dmt low 12 min ecg
s04 ses02 rest low 6 min ecg

s04 ses01 dmt high 12 min resp
s04 ses01 rest high 6 min resp
s04 ses02 dmt low 12 min resp
s04 ses02 rest low 6 min resp

s05 ses01 dmt high 12 min eda
s05 ses01 resthigh 6 min eda
s05 ses02 dmt low 12 min eda
s05 ses02 rest low 6 min eda

s05 ses01 dmt high 12 min ecg
s05 ses01 resthigh 6 min ecg
s05 ses02 dmt low 12 min ecg
s05 ses02 rest low 6 min ecg

s05 ses01 dmt high 12 min resp
s05 ses01 resthigh 6 min resp
s05 ses02 dmt low 12 min resp
s05 ses02 rest low 6 min resp

(luego no se proceso ni el s06, s07 ni el s08, sino que pasa directamente al s09).


Chequear porque sucede esto, porque por ejemplo si vamos a `dmt_high` encontramos archivos para s02, s03, s06, s07, s08. Y en `dmt_low` encontramos a los sujetos s06, s07 y s08.

Sera que al iterar busca una combinación exacta entre sesión y condición que si no encuentra pasa a la siguiente iteración?

Revisar el archivo para dar respuesta a este problema en el procesamiento / iteración de archivos en @test_phys_preprocessing.py



SEGUIR DESDE ACA ->
[ ]  Correr `test_phys_preprocessing.py` para ver archivo a archivo de cada participante si genera o no datos validos. Ir anotando este en algun archivo de logeo que luego `process_eda.py` tome como input para entender cuales son los archivos validos y cuales no. Comparar este logeo de archivos validos con las anotaciones manuales de Tomi a ver si coinciden.

Aca voy a poner las anotaciones generales que vaya teniendo de cada sujeto, uniendo con la data que tenia de acuerdo a anotaciones de Tomi tambien:

    -  La estimacion de HR es muy ruidosa, y estimo que eso va a ser asi para todos los participantes.
    - La medicion de respiracion parece ruidosa (mas en algunos sujetos que en otros), pero capaz es salvable. Deberia juntarme con Ignacio Rebollo para anlizar si tiene sentido ver esta data.
    - Revisar por que tengo este error:    `⚠️  NeuroKit2 plotting failed: Length of values (112) does not match length of index (111)`. Esto me paso en el s13 eda dmt ses02 high. Intentar resolver a ver que esta sucediendo. Me paso en otros sujetos tambien.

# Sujetos señales problematicas 

| Sujeto | Estado |
|--------|--------|
| S01    | INVALID | Faltan archivos de 1 sesion completa y no entiendo por que (buscarla y rechequear)
| S02    | INVALID |
| S03    | INVALID |
| S04    | válido |
| S05    | válido |
| S06    | válido |
| S07    | válido |
| S08    | DMT_2 (señal muerta), INVALID |
| S09    | válido |
| S10    | DMT_2 (señal muerta), INVALID |
| S11    | DMT_2 (señal muerta), INVALID |
| S12    | DMT_2 (señal muerta, DMT_1 está bien), INVALID |
| S13    | válido |
| S15    | DMT_2 (señal muerta), INVALID |
| S16    | válido |
| S17    | válido |
| S18    | válido |
| S19    | válido |
| S20    | válido |

Faltan archivos del sujeto 01, 02 y 03:
  Missing files:
   📂 EDA:
      ❌ S01 - EDA DMT High Dose
      ❌ S01 - EDA Resting High Dose
      ❌ S02 - EDA DMT Low Dose
      ❌ S02 - EDA Resting Low Dose
      ❌ S03 - EDA DMT Low Dose
      ❌ S03 - EDA Resting Low Dose
   📂 ECG:
      ❌ S01 - ECG DMT High Dose
      ❌ S01 - ECG Resting High Dose
      ❌ S02 - ECG DMT Low Dose
      ❌ S02 - ECG Resting Low Dose
      ❌ S03 - ECG DMT Low Dose
      ❌ S03 - ECG Resting Low Dose
   📂 RESP:
      ❌ S01 - RESP DMT High Dose
      ❌ S01 - RESP Resting High Dose
      ❌ S02 - RESP DMT Low Dose
      ❌ S02 - RESP Resting Low Dose
      ❌ S03 - RESP DMT Low Dose
      ❌ S03 - RESP Resting Low Dose


[x] Revisar este error que tuve al correr @test_phys_preprocessing.py:    `⚠️  NeuroKit2 plotting failed: Length of values (112) does not match length of index (111)`. Esto me paso en al procesar al sujeto `s13` ara el archivo de la señal de EDA en la condicion dmt high. Por que sucede este error? Hacer una prubea minima que intente  plotearlo con neurokit2 para entender que sucede especificamente con este sujeto. Crear un script especificamente para esta prueba minima de ploteo de la señal de eda de este archivo (que debe guardarse en `/test`)

[x] Crear un script en `./test` que lo que haga sea chequear en @validation_log.json cuales son los archivos que fueron marcados como "bad" en "category". Luego, si un sujeto (e.g. `s12`) tiene alguno de los dos archivos de la condicion `DMT` (`low` o `high`) como bad, ese sujeto debe marcarse como bad en un nuevo json que genere como output. Este json debe tener para cada señal separda (`eda`, `ecg` y `resp`) cuales son los sujetos marcados como `bad` (porquepara esa señal tiene uno de los dos archivos de la condicion DMT como `bad`).

SEGUIR DESDE ACA ->

[x] Revisar que los archivos marcados como bad para cada sujeto / señal sea correcta (ya lo estoy revisando con chatgpt)

[x] Con la informacion de validez de los datos guardada en @validation_log.json y considerando ahora los marcados como "bad subjects" para cada señal en @dmt_bad_subjects.json, quiero que me armes una lista para cada señal (eda, ecg y resp) con los sujetos validados (no "bad" ni vacios) para dicha señal (con valor de `notes` que sea `good` o `acceptable`). Quiero que agregues esto como 3 nuevas listas en el config.py file (i.e. `SUJETOS_VALIDADOS_EDA`, `SUJETOS_VALIDADOS_ECG`, `SUJETOS_VALIDADOS_RESP`)

[x] Buscar ahora cual es la mejor representacion para diferenciar entre dosis alta y dosis baja de EDA.
Ahora que ya tengo los datos de los participantes guardados `../dmt\data\derivatives\preprocessing\phys` para las 3 señales de interes (`ecg`, `eda`, `resp`) y que tengo en @config.py los sujetos validos para cada señal (en las listas `SUJETOS_VALIDADOS_EDA`, `SUJETOS_VALIDADOS_ECG` y `SUJETOS_VALIDADOS_RESP` dentro de ese archivo), puedo empezar a inspeccionar los datos de la señal de EDA para los sujetos validos.
Para empezar, hacer ploteos por participante comparando el AUC de SMNA (columna `SMNA` de `*cvx_decomposition.csv`) para el caso de dosis alta y dosis baja. Esto nos va a ayudar a explorar mejor los datos sujeto a sujeto (nos falta todavia ese insight).
De este modo, para todos los sujetos validos de la señal EDA, hacer plots individuales por sujeto en el que se compare el SMNA de dosis alta vs dosis baja. El Area bajo la curva debe estar pintada, con una opacidad media (para que se note cuando ambos colores se superponen). Se espera que, sobre todo en la primera mitad de los registros, el AUC de SMNA para dosis alta supere a dosis baja.
Guardar estos plots por sujeto en `/test/eda/smna`.

[x] Quiero que cada participante tenga un unico plot con dos subplots: el subplot de la dosis (dmt high vs dmt low) y el subplot de resting state (resting state - high vs resting state - low).



Takeaways del analisis estadistico (guardado en smna_auc_2x2_boxplot.png)

“takeaways”:

Efecto de Tarea (RS vs DMT) robusto y consistente. Global 2×2: Task significativo (F=15.94, p=0.0025). Por minuto, el efecto se replica en la mayoría de ventanas (0–7 y 9; p≈0.001–0.050) y en el agregado 0:00–4:59 (p=0.0012). Interpretación: DMT muestra AUC mayor que RS de forma estable.

Dosis (Low vs High) no muestra efecto global, pero emerge en ventanas tardías: sin efecto global (p=0.147), con señales por minuto alrededor de min 4–5 y especialmente min 8 (p=0.013). Patrón: High > Low aparece cuando avanza la sesión.

Interacción Task×Dose: tendencia global y pico puntual. Globalmente marginal (p=0.0789). Por minuto, significativa en min 8 (p=0.0246) y cercana en 2, 4–6 y 9. Sin embargo, tras corrección Holm a través de minutos no se mantiene (p_corr≈0.25), por lo que la evidencia de interacción es sugerente pero no concluyente.

Comparaciones apareadas confirman el patrón esperado de “RCT intra-sujeto”. En RS: Low vs High nunca difiere (todos p>0.22). En DMT: Low vs High sí difiere en ventanas específicas (min 4: p=0.033; min 8: p=0.0118; ~min 5/9: p≈0.05). Esto explica la tendencia a interacción: la dosis modula la respuesta solo en DMT, no en RS.

Medias de celda globales coherentes con el patrón. RS: Low 6.22 → High 3.90 (ligera baja con dosis). DMT: Low 16.11 → High 27.59 (fuerte subida con dosis). Resultado: gran efecto de Task y tendencia a interacción impulsada por DMT.

Conclusión operativa. Hay un efecto principal fuerte de Tarea, evidencia de efecto de Dosis en fases medias–tardías dentro de DMT, y señales de interacción que requieren confirmación con control por múltiples comparaciones (o un modelo que integre el eje tiempo).

[x] Prepará los datos en formato “long” (columnas: subject, minute [0–9], Task {RS,DMT}, Dose {Low,High}, AUC) y creá minute_c = minute - mean(minute). Ajustá primero un LME con statsmodels (MixedLM) con efectos fijos AUC ~ Task*Dose + minute_c + Task:minute_c + Dose:minute_c (dejar Task:Dose:minute_c solo si converge/está justificado) y efectos aleatorios ~ 1 | subject; si converge y aporta varianza, subí a ~ 1 + minute_c | subject. En paralelo, corré un análisis de sensibilidad con GEE (statsmodels.GEE) usando la misma fórmula de fijos y AR(1) para autocorrelación temporal dentro de sujeto. Definí tres familias de hipótesis: (i) Task (incluye Task y Task:minute_c), (ii) Dose (incluye Dose y Dose:minute_c), (iii) Task:Dose (y Task:Dose:minute_c si se incluyó); para cada familia, construí contrastes (p.ej., DMT vs RS promediando minutos; pendiente vs minuto; High–Low dentro de DMT y dentro de RS) y aplicá BH–FDR a los p‐values resultantes.

[x]  Curva de AUC acumulada (cumAUC) High vs Low
- Por sujeto y condición, calculá cumAUC[m] = sum(AUC[0:m]).
- Graficá media de cumAUC para High y Low con IC95% por bootstrap estratificado por sujeto (resample sujetos con reemplazo; dentro de cada resample mantené sus 20 observaciones emparejadas Task×Dose). Líneas finas por sujeto con alpha bajo.
- Añadí banda de diferencia cumAUC_High − cumAUC_Low con IC95% bootstrap; resaltá los minutos donde el IC no cruza 0 (interpretable como mayor “carga total” bajo High).


[x] Hacer tambien el plot para comparar dosis baja y dosis alta para el EDR (columna `EDR` de `*cvx_decomposition.csv`) y EDL (columna `EDL` de `*cvx_decomposition.csv`) por participante (tambien quedandote con los priemeros 10 minutos de registro). Es decir, ahora en `/test/eda/scr`(para `EDR`) y `/test/eda/scl`(para `EDL`) se deben guardar plots analogos a los que se guarda actualmente en `test\eda\smna_10min` pero con esta otra data especifciada (i.e. edr y edl).
Vamos a empezar primero con EDR (i.e. SCR).

[x] A Ahora vamos a hacer ECL (i.e. SCL)

[x]  Considerar que existe `baseline_correction = True` como metodo para el callculo de SCRs en neirokit2. Revisar.
Respuesta -> No, no es necesario, porque esto se hace cuando tengo eventos / epocas. Y lo mio es un unico registro completo (no lo tengo epocheado)

SEGUIR DESDE ACA. REVISAR LAS TAREAS HECHAS Y PASAR A LAS TAREAS SIGUIENTES

[x] Me esta faltando estandarizar la señal? Chequear. Deberia estar usando en algun momento algo tipo `nk.standardize(eda_signal)` pero tal vez `eda_process` ya lo hace directamente.
Respuesta -> Corta: no hace falta. Como usamos LME con intercepto aleatorio por sujeto (1|subject), ese término ya absorbe las diferencias de nivel entre sujetos. Estandarizar la variable respuesta puede quitarte interpretabilidad en µS e incluso atenuar diferencias RS vs DMT si lo haces por sesión/condición.


[x] Resolve problema de 0 padding. Por ahora, esto esta generando 0 padding para completar los archivos. Pero esto no es una buena idea, porque me genera artefactos al momento de analizar los datos. Lo que deberia hacer es producir el padding con el ultimo valor de la señal de EDA. Y despues generar todas las features a partir de esta solucion. Faltaria ver como resulev esto mismo para la señal de ECG y RESP, para dar una solucion general antes de implementar. Tambien habria que ver cuantos archivos estan en esta situacion.

1. Documentar que participantes tienen este problema de padding y en que señales

S04 EDA REST SES02 LOW
* S05 EDA DMT SES01 HIGH
* S06 EDA REST SES01 LOW
* S07 EDA REST SES01 LOW
* S07 EDA DMT SES02 HIGH
S09 EDA REST SES01 HIGH
S10 EDA REST SES02 LOW
S11 EDA REST SES02 HIGH
S12 EDA REST SES01 LOW
S12 EDA REST SES02 HIGH
S13 EDA REST SES02 HIGH
S15 EDA REST SES01 HIGH
S15 EDA REST SES02 LOW
* S18 EDA REST SES01 HIGH
S20 EDA REST SES01 LOW
* S20 EDA REST SES02 HIGH


2. Idear como resolver par acada señal el problema de padding:
    - Opcion 1:En los archivos que tengan valores faltantes para completar los 20 minutos 15 segundos (o 10 minutos 15 segundos, en caso de la condicion RS), completar estos valores faltantes con el ultimo valor valido de la serie de tiempo en vez de reemplazar con valores `0` (0 padding).
    - Opcion 2: En vez de reemplazar los archivos originales, cambiar solamente la columna `EDL`.
    - Opcion 3 (bonus): usar 20 o 10 minutos (y no los 15 segundos extras que me estan trayendo problemas)

3. Implementarlo

[x] Modificar  `./scripts/preprocess_phys.py`. En la version actual del script, si los archivos de condicion `DMT` tienen una duracion menor a 20 minutos 15 segundos, o los archivos de condicio `RS` tienen ua duracion menor a 10 minutos 15 segundos, estos se completan usando como tecnica "0 Padding" (i.e. se agragan ceros hasta completar los registors). Pero esto no es una buena idea, porque me genera artefactos al momento de analizar los datos. Por el contrario, lo que se debera hacer es completar hasta la duracion final (20:15 o 10:15, segun la condicion) con valores NaN. Hacer este cambio en preprocess_phys.py

* Guarde la carpeta `phys` como backup en escritorio



[x] Hacer el plot de los 20 minutos comparando dosis baja y dosis alta, que ahora no deberia tener el 0 padding. Al hacer los plots, hacerlo con los 20 minutos 15 segndos, y marcar claramente el momento de administracion de DMT. Tene en cuenta que los eje X debe ir de 0 a 20 minutos (que lo anterior debe ser negativo, no debe ser un eje X de 20 minutos 15 segundos.)

SEGUIR DESDE ACA ->

[x] Implementar los siguientes cambios en `plot_eda_components.py`:
Archivos individuales por sujeto:
- En el y label de cada subplot, en vez de "(a.u.)" deberia decir "(μS)"
- En el x label de cada subplot, la escala temporal deberia tener numeros espaciados cada un minuto (0:00, 1:00, ..., 9:00). Y el label deberia decir `Time (minutes)`, o algo asi.
- Tene en cuenta que los registros de cada sujeto son de 9 minutos y 10 segundos. Pero quiero que los 10 primeros segundos sean "linea de base", y que el registro real sea de 00:00 a 09:00. Para eso, en el 00:00 marca una linea punteada vertical, para dar cuenta que ese es el inicio real.
- En el label de las variables eliminar los valores medios e.g. "(mean=0.045)" luego de definir el color de cada variable.
- Los colores de cada variable deberian ser los mismos para todos los sujetos: DMT High en rojo, DMT Low en azul, RS High en verde y RS Low en violeta.
- La escala del eje Y deberia ser igual para todos los sujetos (entre 5 y -5)
- La escala del eje X deberia ser igual para todos los participantes.
 - En el titulo de cada subplot deberia decir solamente el numero de sujeto (e.g. S05) y nada sobre EDA, EDL ni (first 10 min)
 - En los titutlos de cada subplot solamente deberia decir `DMT` (y no `DMT High vs Low (first 10 min)`) y `Resting State (RS)` (y no `RS ses01 vs ses02 (first 10 min)`).

[x]  Implementar los siguientes cambios en `plot_eda_components.py`:
Archivos individuales por sujeto:
- No quiero mas las lineas verticales en 00:00. De hecho, el registro debe comenzar en 00:00 y terminar en 09:10 (manteniendo las marcas en el x label en 00:00, 01:00, ..., 09:00)
- Los titulos DMT, Resting State (RS) y el numero de sujeto (e.g. S04). Deben ir todos en bold. Y el titulo del sujeto debe ir ademas en un formato de letra el doble de grande.
- El orden de los labels indicando la condicion en Resting State -RS High y RS Low- deben seguir siempre *ese* orden (high arriba y low abajo)

[x]  Implementar los siguientes cambios en `plot_eda_components.py`:
Archivos individuales por sujeto:
- Los subplots de Resting State deben ir a la izquierda y los de DMT a la derecha (al reves de como estan ahora)
- Incluir tambien en el subplot de la derecha el Y label con los Y ticks de los valores en microsiemens (ahora solo aparecen en el subplot de a derecha)
- Crear de forma adicional un plot que stackee de forma vertical todos los sujetos. Esta figura sera agregada como material suplementario en el paper
Archivos `summary`:
- Mantener todas las propiedades de los elementos graficos analogas a los plots individuales enlos plots sumary (e.g. titulos de los subplots, titulos de los ejes X e Y, formato de los labels, colores de las lineas, etc)
- El subplot de Resting state deben ir a la izquierda y los de DMT a la derecha.
- Eliminarle el titulo general a estos plots summary. En cambio guardar un nuevo .txt con el caption de esta figura y tambien dela figura de los sujetos indivudales stackeada (figura de material suplementario)

[x]  Implementar los siguientes cambios en `plot_eda_components.py`:
Archivos individuales por sujeto:
- Los filenames de los archivos por sujeto deberian decir `9min` en vez de `10min`.
Archivo "stacked_subject":
- Los numeros de los sujetos (e.g. S04) aparece superpuesto co el Y label (e.g. EDL (uS)). Hay que dejar mas espacio en el margen izquierdo para que entre el numero del sujeto. Ademas, el numero del sujeot debe estar el doble de grande al menos.
Archivo "summary dmt only":
- Mantener todas las propiedades de los elementos graficos analogas a los plots individuales enlos plots sumary (e.g. titulos de los subplots, titulos de los ejes X e Y, formato de los labels, colores de las lineas, etc). Incluir tambien una caption de esta figura

- Todas las caption deben ser en ingles, y *mucho* mas completas. Te paso un  ejemplo de una caption bien completa:
Fig. 1. Cumulative dose effect on SMNA (High–Low) during DMT and rest.
Each curve shows, per task, the cumulative dose effect calculated as the sum over time (0–9 min) of the differences per minute in SMNA AUC between high and low doses (High–Low). The thick lines represent the group mean with 95% CI (shaded); the thin lines are the individual trajectories (N = 11). 

[x] Implementar los siguientes cambios en `plot_eda_components.py`:
En todos los plots:
- El Y label debe ser el **delta** de SCL en microsiemens, y *no* directamente SCL en microsiemens. Replicar esto en todos los archivos por sujeto
Archivos stackeado por sujeto:
- Quiero que el archivo stakkeado sea literalmente una version stakeada verticalmente de los plots individaules. De este modo, cada subplot por participante debe incluir todos los labels y titulos identifico a los plots por participantes, con el mismo estilo etc. Asi, el titulo de sujeto debe estar en la parte superior, debe incluir subtitulos de resting state y DMT, debe incluir x e y label cada subplot de cada participante, asi como tambien los indicadores de las condiciones / colores en cada subplo de cada sujeto
En ambos plots summary (summary dmt only y summary combined dmt rs mean sem edl):
- El indicador de la codnicion high o low no debe tener ese relieve/sombreado, sino qeu debe ser igual esteticamente a como aparece ese indicador en los archivos por sujeto
- El Y label del delta de SCL debe tener limite en -1.5 y 1.5

[x] Para los plots de SCR de @plot_eda_components.py, considera que:
- Para los plots por sujeto (y tambein el plot stacked_subs_eda_scr), el eje y deberia tener limite entre -0.25 y 1.0. Pero que los y ticks deberian ir de 0.0 a 1.0 (0.0, 0.1, ..., 1.0)
- Para los plots por all_subs, el eje y deberia tener limite entre -0.05 y 0.25. Pero que los y ticks deberian ir de 0.00 a 0.25 (0.00, 0.05, ..., 0.25)
- Por algun motivo no se creo el plot all_subs_dmt_eda_scr.png, que deberia haberse creado al igual que corriendo para scl


- [x] Ahora que ya implemente los cambios, correrlo para SCR y ver resultados


SEGUIR DESDE ACA

[x] Incluir en plot_eda_components.py --SCR analisis sobre los datos de `*emotiphai_scr.csv` (en vez de la columna EDR de la extraccion de CVX EDA).

Tene en cuenta que los datos de emotiphai tienen este formato (ver datos de las primeras filas como ejemplo de una sesion de un paticipante, e.g. `S02_dmt_session2_high_emotiphai_scr.csv`):

```
SCR_Onsets_Emotiphai,SCR_Peaks_Emotiphai,SCR_Amplitudes_Emotiphai
929,1555,0.49777427216092107
2022,2417,0.16037486767340248
3588,4278,0.4908677882173871
5753,6415,0.4167111851163048
7010,7562,0.4408856160456054
```

Esto quiere decir que ya no tenemos una serie de tiempo de valores de EDA, sino tenemos todos los SCRs encontrados con los tiempos del onset y del peak (en timepoints) y su amplitud asociada.

Para empezar, quiero que me propongas posibles plots para evaluar condiciones (DMT y RS) y dosis (alta y baja) con este tipo de datos de SCR (cantidad de SCRs y amplitud).

Con esos plots, voy a decidir  si vale la pena incluir algun grafico a los resultados segun los analisis de extraccion de SCR counts (y/o amplitud de los SCRs) extraido de los archivos `emotiphai_scr.csv` guardados en `../data/derivatives/preprocessing/phys/eda/*`


[x] Objetivo: rehacer los analisis de SMNA (que ya hice antes) pero con los nuevos datos, porque cambie un poco las tablas de donde salieron los datos.
Para esto, quiero crear un nuevo script que sea un mix de estos siguientes scripts que antes corria secuencialmente:
Primero: python test/lme_analysis_smna_auc.py (análisis principal)
Segundo: python test/plot_lme_coefficients.py (coeficientes)
Tercero: python test/plot_lme_marginal_means.py (medias marginales)
En concreto, aunar todo en un unico script, que considere unicamente usar los 9 primeros minutos de datos para los analisis (tanto plots como analsiis estadisticos), de forma analoga a como hice con los scripts de SCL en @plot_eda_components.py 

Ademas, es necesaio hacer algunas cambios menores extras:
- Imitar todo el resto de los patrones esteticos del plot (labels, formato, colores para cada condicion, nombre de las variables, orden de las condiciones, etc) como hice en `plot_emotiphai_scr_rate.py`
- Eliminar los titulos de todos los plots
- Elimnar el bold de todos los `x label` y `y label` de todos los plots
- modificar para que en vez de crear una figura de  `model_summary.png`, qeu esto sea un .txt
- Eliminar de todos los plots los rectangulos de texto con los key findings (eso se debe agregar luego al texto del paper, o al caption de las figuras).
- Al correr esto se debe crear automaticamente un .txt con las caption de todas las figuras

[x] Ahora quiero hacer algunos cambios pequeños a scripts/run_eda_smna_analysis.py:
- En `task_main_effect.png`, el y label debe ser "SMNA AUC". En los labels de las condiciones arriba a la derecha, debe decir DMT y RS en ese orden, y no al reves como esta ahora. Y sin el aparentesis de "avg across dose".
- En `task_dose_interaction.png`, el y label debe ser "SMNA AUC" (esto vale tambien para `marginal_means_all_conditions.png`). Y los labels de arriba a la derecha deben estar dentro de un recuadro como el resto de los labels en los otros plots. Ademas, debe mantenerse la logia de "High" arriba y "Low" abajo (y no al reves como esta ahora)

[x] Implementar estos mismos analisis estadisticos que hice para SMNA pero para SCL.  Corriendo el anterior script `plot_eda_components.py --component SCL` (que ahora adaptare y creare un nuevo script en `./scripts`) se deberian crear directamente estos analisis. Importante: ahora tanto los plots (que hacia antes) como los nuevos analisis y plots resultantes que se agregan ahora se guardaran en `./results/eda/scl`. El nuevo script  `run_eda_scl_analysis.py` (adaptado de `plot_eda_components.py --component SCL`) deberia generar todo esto que describi anteriormente.

[x] Algunos cambios que son necesarios hacer todavia en run_eda_scl_analysis.py:
- Todos los plots deben guardarse en `plots` (y no directamente en `scl`)
- Por algun motivo, las cuadriculas de fondo en los plots estan en un color muy solido. Deberian estar mas tranlucidas / grises, como estaban en el resto de los plots de antes.
- El y label debe ser `ΔSCL (µS)`
- En la nueva version de `stacked_subs_eda_scl.png` falta agregar en el titulo de cada subplot stackeado el numero de los particiapntes, como estaba antes en @plot_eda_components.py --scl

[x] Bien. Ahora ya terminamos de adaptar por completo y de la forma que buscaba  los analisis que antes formaban parte de  `plot_eda_components.py --component SCL` en `scripts/run_eda_scl_analysis.py` (incluyendo elementos de diseño grafico aprioiado, etc).
Ahora quiero que hagas LO MISMO para adaptar los analisis de `plot_eda_components_scr_emotiphai.py` en un nuevo script `run_eda_scr_analysis.py`. Considera agregar tambien los analisis de LME considerando los SCR counts extraidos.

De SCR el unico grafico que me interesa es all_subs_eda_scr.png, que incluso podria ir a Anexos. El resto me parece que no vale la pena.



SEGUIR DESDE ACA
----------------

[x] Editar la figura tambien de resultados suplementarios ara agregar significancia estadistica. Esto NO da diferencias significativas en ninguna plot / subplot. Es por eso que no aparece ningun segmento como sombreado, porque ninguno sobrevive correccion por FDR. Los resultados quedaron guardados en .txt en results\eda\scl. Los p valor minimos son siempre mayores a 0.05 (corregido por FDR).
[x] Ver comentario de Diego " En caso de que RS sea placebo (lo que sería menos confuso que decir RS) por que hay diferencia enrte high y low".
Revisar esto, porque efectivamente tiene razon, es raro.
La respuesta es que no hay diferencias significativas (si no apareceria como sombreado en el plot, y no lo esta). 
[x] Extender los cambios realizados en SCL a SCR
[x] Extender los cambios realizados en SCL a SMNA
[x] Crear un script auxialiar en `results/eda/` llamado `generate_eda_figures.py`, que genere 2 paneles de plots, a partir de los plots ya generados y guardados en `results/eda`.

1er panel:
Parte superior del panel:
Subplot A (disposicion izquierda): all_subs_eda_scl.png
Subplot B(disposicion izquierda): lme_coefficient_plot.png
Parte inferior del panel:
Subplot C (disposicion izquierda): all_subs_smna.png
Subplot D(disposicion izquierda): lme_coefficient_plot.png

2do panel:
Parte superior del panel (Subplot A): all_subs_dmt_eda_scl.png
Parte inferior del panel (Subplot B): all_subs_dmt_smna

Tene en cuenta que estos ya seran los plots finales que seráne enviados para publicar a Nature Human Behaviour. Por lo tanto, dispone los plots esteticamente ya como plots finales, secuenciando los subplots de cada panel en orden segun las letras mencioandas mas arriba (e.g. A, B, C, ...)

[x] Agrandar un poco el tamaño de los boxes de labels en el margen superior derecho de cada subplot, modificando para eso `run_eda_scl_analysis` y ``run_eda_smna_analysis`. Con eso volver a correr la generaicon de plots con `generate_eda_figures`

[x] Editar `generate_eda_figures` para que:
- Las letras que titulan los subplots del panel (A,B,C y D) sean un poco mas grandes
- El margen (espacio) vertical que separa el panel A del panel B, y tambien el panel C del panel D (es decir, los paneles de la izquierda de los de la derecha) sea considerablemnete menor. Ahora hay mucho espacio entre si.

[x] EDitar en `Results` y en `Material Suplementario` las captions de acuerdo a estas nuevas figuras.

SEGUIR DEDE ACA
[ ] Documentar todos los cambios realizados.
[ ] Planear las tareas de ECG e implementarlas hasta donde pueda con los creditos de cursor (vrevisar lo que hizo Tomi para saber como seguir). Pero esto ya hacerlo en un nuevo documento `ecg.md` dentro de `research_diary`.


EXTRA (no lo implementamos):

   [ ] Hacer espectrograma pero con los datos de EDA a nivel exploratorio, e integrarlo tambien en el documento .doc.

   [ ] Hacer espectrograma pero con los datos de ECG a nivel exploratorio.

   [ ] Analisis de ECG. En el S02 ECg Rest ses 02 high hay que corregir la señal de ECG, pero es salvable interpolando  desde el segundo 210 hasta el segundo 310.



PROXIMOS PASOS ->

[ ] Que pasa si comparamos el PSD de EEG en los momentos de SMNA > 0 comparados con SMNA = 0?. COmo cambia el PSD dependiendo la actividad autonomica?

