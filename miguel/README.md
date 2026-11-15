# Parte 4 — ChemBERTa-2: aportación de Miguel

Todo lo que hay aquí se construye sobre el `models/chemberta.py` que subió Daniel.
Nada de esta carpeta pisa archivos de la raíz: para usar el código ampliado hay que
copiarlo encima (ver más abajo).

## Qué hay

| Carpeta | Contenido |
|---|---|
| `models/chemberta.py` | Versión ampliada del modelo: cuatro cambios y dos ajustes nuevos |
| `conf/model/chemberta.yaml` | Su configuración |
| `batch.sh`, `batch_lr.sh` | Lanzan varios entrenamientos dentro de un mismo trabajo de la cola |
| `probe_chemberta.py`, `probe_pooling.py` | Comprobaciones sueltas sobre el modelo y el pooling |
| `figuras/` | Seis figuras para el informe, en PDF y PNG, con el script que las genera |
| `informe/` | `parte4-chemberta-resultados.pdf`: el registro completo de experimentos |
| `resultados/` | Métricas e historiales de las 74 ejecuciones, y `resumen_runs.csv` |

## Cambios respecto a `models/chemberta.py` de la raíz

1. **`add_pooling_layer=False`** al cargar el encoder. Esa capa no viene en los
   checkpoints, así que nacía al azar, y `forward()` no la usa. Con el arreglo el
   encoder se cuenta como 3,28 M pesos (los reales) en vez de 3,43 M, y desaparece
   el aviso `MISSING`.
2. **`pool()` lanza error** ante un `pooling` desconocido. Antes un error de
   escritura (`CLS` en mayúsculas) caía en silencio en la media, y habríamos
   lanzado un experimento creyendo que usaba CLS.
3. **`pooling: concat`** (nuevo): concatena CLS y media; la cabeza recibe 768 en vez
   de 384, y el tamaño de entrada se ajusta solo.
4. **`pretrained`** (nuevo, defecto `true`). Con `false` monta la misma arquitectura
   con pesos al azar: sirve de *ablation* para medir cuánto aporta el
   preentrenamiento. No es lo mismo que `freeze_encoder`: `pretrained` fija con qué
   valores **arranca** el encoder, `freeze_encoder` si esos valores **cambian**.
5. **`augment`** (nuevo, defecto `false`): aleatorización de SMILES. Renumera los
   átomos al azar y escribe sin canonizar. Se aplica **solo en entrenamiento**, así
   que validación y test siguen viendo el SMILES canónico.

Para usarlo:

```bash
cp miguel/models/chemberta.py models/
cp miguel/conf/model/chemberta.yaml conf/model/
```

## Resultados principales (validación)

Todas las ejecuciones: 20 épocas, `encoder_lr` 1e-5, readout MLP, semilla 42 salvo
donde se indique. Detalle completo en `informe/` y en `resultados/resumen_runs.csv`.

| Configuración | Regresión (ST-RAE ↓) | Clasificación (MCC ↑) |
|---|---|---|
| `mlm_cls` | 0,905 | 0,142 |
| `mtr_cls` | **0,741** | 0,273 |
| `mtr_mean` | 0,753 | **0,292** |
| sin preentrenar + `cls` | 0,982 | 0,000 |

- **MTR gana a MLM** en las dos tareas y con los tres *pooling*.
- **Sin preentrenar no aprende nada**: 0,982 es exactamente predecir la media
  (0,983), y 0,000 es decir «no» a todo.
- Con MTR, los tres *pooling* empatan: la diferencia (0,741 / 0,753 / 0,741) es
  menor que lo que oscila sola la validación entre épocas (0,026–0,039).

Se descartaron, con medidas: la aleatorización de SMILES (3 semillas y 60 épocas,
sin efecto), elegir el umbral del MCC (ninguna regla supera al 0,5 fijo) y `concat`
(empata y duplica la cabeza).

## El learning rate: medido y cerrado

Daniel encontró que `encoder_lr` 1e-5 infraentrena, con una curva en U cuyo mínimo
está en 1e-4: con el checkpoint **MLM** ese cambio vale 0,071. Como todas las
ejecuciones de aquí usaban 1e-5, quedaba la duda de si la rejilla de
checkpoint × pooling estaba hecha sobre un valor malo.

**Se midió sobre MTR (10 ejecuciones, `batch_lr.sh`) y no se traslada.** Validación,
MTR + cls sin congelar:

| `encoder_lr` | Regresión (ST-RAE ↓) | Clasificación (MCC ↑) |
|---|---|---|
| 3e-5 *(1 semilla)* | 0,726 | 0,253 |
| **1e-5** *(de partida, 3 semillas)* | **0,732 ± 0,006** | **0,292 ± 0,021** |
| **1e-4** *(el de Daniel, 3 semillas)* | **0,728 ± 0,005** | **0,285 ± 0,012** |
| 3e-4 *(1 semilla)* | 0,731 | 0,263 |

- **Con MTR el learning rate no cambia nada** ($p = 0{,}39$ y $0{,}63$). Los cuatro
  valores caen entre 0,726 y 0,732 en regresión: el fondo es plano, no hay curva
  en U. En clasificación el mejor es directamente el de partida.
- **La sensibilidad depende del checkpoint**, así que un barrido hecho sobre MLM no
  vale para MTR. Encaja con lo del punto 1: la cabeza de preentrenamiento del MTR
  leía el token CLS, así que el cuerpo llega con un resumen que ya sirve y apenas
  necesita moverse; el de MLM nunca entrenó ese token y tiene que recorrer mucho
  más camino, y ahí sí importa a qué velocidad.
- **La configuración elegida no cambia**, y la objeción queda cerrada con medidas.

## Veredicto (test, evaluado una sola vez)

| Modelo | Regresión (ST-RAE ↓) | Clasificación (MCC ↑) |
|---|---|---|
| Morgan MLP | 0,864 | — |
| LightGBM | 0,783 | 0,141 |
| **ChemBERTa** | **0,778** | **0,183** |
| Atom-JEPA | **0,739** | **0,231** |

**ChemBERTa queda en medio, y de forma consistente en las dos tareas.** Gana al
baseline en ambas (en clasificación por un 30 % sobre LightGBM), y pierde contra
Atom-JEPA en ambas. En regresión la ventaja sobre LightGBM es de 0,005, dentro del
ruido, así que lo honesto es decir que **empata** con el baseline y pierde contra
Atom-JEPA.

Leer la molécula como texto basta para igualar a unas *features* hechas a mano y
para mejorarlas bastante en la tarea difícil, pero **no sustituye a ver la
geometría en 3D**.

De todo lo que se barrió —pooling, congelación, aleatorización, umbral, épocas,
learning rate— **la única decisión con un efecto mayor que el ruido fue MLM frente
a MTR**.

## Pendiente para el informe del grupo

**Hay que fijar un solo baseline de referencia.** El informe de Daniel compara con
Morgan MLP (0,843 en validación) y este con LightGBM (0,740). Son modelos distintos
y el enunciado pide elegir uno.

## Figuras

Las imágenes **no llevan pie dentro**: el pie va en `figuras/figuras.tex`, listo
para `\caption{}`, de modo que LaTeX las numere y el texto pueda citarlas con
`\ref{}`. Los títulos de los paneles dicen qué se representa, no qué concluimos.

Para regenerarlas hace falta `runs/` completo en la raíz del repo:

```bash
python miguel/figuras/hacer_figuras.py
```

## Resultados guardados

`resultados/` lleva, por ejecución, `config.json`, `history.jsonl` y las métricas de
validación y test. **No lleva los pesos** (`best.pt`, 1,2 GB entre todos) ni los
`split.csv`, que son idénticos en todas. Con lo que hay se regeneran las figuras y
se comprueba cualquier número del informe.

`resultados/resumen_runs.csv` tiene las 74 ejecuciones en una tabla, con su
configuración y sus notas de validación y test.
