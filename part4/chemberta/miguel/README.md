# Parte 4 — ChemBERTa-2: aportación de Miguel

Se construye sobre el `models/chemberta.py` que subió Daniel. Nada de esta carpeta
pisa archivos de la raíz: para usar el código ampliado hay que copiarlo encima.

```bash
cp part4/chemberta/miguel/models/chemberta.py models/
cp part4/chemberta/miguel/conf/model/chemberta.yaml conf/model/
```

## Qué hay

| Ruta | Contenido |
|---|---|
| `models/chemberta.py` | El modelo ampliado: cuatro arreglos y dos ajustes nuevos |
| `conf/model/chemberta.yaml` | Su configuración |
| `figures/` | Las **6 figuras del apéndice** (PDF + PNG), el script que las genera y `README.md` con el orden en que contarlas |
| `report/` | `parte4-chemberta-resultados.pdf`: el registro completo, con las figuras ya integradas |
| `results/` | Configuración, curvas y métricas de las **112 ejecuciones**, y `resumen_runs.csv` con todo en una tabla |
| `batch*.sh` | Las cinco tandas que se lanzaron, cada una con su porqué en la cabecera |

`results/` **no lleva los pesos** (`best.pt`, 1,2 GB entre todos) ni los
`split.csv`, idénticos en todas. Con lo que hay se regeneran las figuras y se
comprueba cualquier número del informe.

## Resultado final

**Test, evaluado una sola vez**, con la configuración cerrada por validación:

| Modelo | Regresión (ST-RAE ↓) | Clasificación (MCC ↑) |
|---|---|---|
| Morgan MLP | 0,864 | — |
| LightGBM | 0,783 | 0,141 |
| **ChemBERTa** (3 semillas) | **0,778 ± 0,010** | **0,183 ± 0,019** |
| Atom-JEPA | **0,739** | **0,231** |

**ChemBERTa queda en medio, consistentemente en las dos tareas.** Gana al
*baseline* en ambas —en clasificación por un 30 % sobre LightGBM— y pierde contra
Atom-JEPA en ambas. En regresión la ventaja sobre LightGBM (0,005) está dentro del
ruido, así que lo honesto es decir que **empata** con el *baseline*.

Leer la molécula como texto basta para igualar a unas *features* hechas a mano y
para mejorarlas bastante en la tarea difícil, pero **no sustituye a ver la
geometría en 3D**.

### Configuración elegida

```
checkpoint = DeepChem/ChemBERTa-77M-MTR   pooling = cls      freeze_encoder = false
augment = false                           epochs = 20        encoder_lr = 1e-5
```

Elegida **solo con validación**: 0,732 ± 0,006 en regresión y 0,292 ± 0,021 en
clasificación.

## Lo que se midió, y lo que resultó importar

Cada contraste con 3 semillas por lado, *t* de Welch sobre validación.

| Decisión | Efecto | ¿Importa? |
|---|---|---|
| **Checkpoint: MTR vs MLM** | +0,181 reg., +0,144 clas. | **Sí** (*p* < 0,001 y 0,001) |
| **Pooling: CLS vs mean** (con MTR) | +0,018 reg., +0,068 clas. | **Sí en clasificación** (*p* = 0,032) |
| Congelar el cuerpo | +0,007 reg., +0,015 clas. | No (*p* = 0,21 y 0,40) — y cuesta 5× menos |
| Aleatorizar SMILES | +0,008 reg., +0,017 clas. | No (*p* = 0,13 y 0,30) |
| *Learning rate* 1e-6 … 3e-4 | dentro de ±0,007 | No (*p* entre 0,40 y 1,00) |
| Umbral del MCC (max-MCC, Youden) | — | No supera al 0,5 fijo |

**De todo lo que se barrió, lo único que hay que acertar es el *checkpoint*.** No
porque lo demás no importe nunca, sino porque **aquí se midió y no importó**.

### Por qué MTR gana, y por qué eso predice lo del pooling

La cabeza de preentrenamiento del MTR es `dense` 384→384 + `out_proj` 384→199: el
patrón de una cabeza que lee **únicamente el token CLS**. Como ese era el único
vector que se leía para predecir los 199 descriptores de RDKit, el cuerpo quedó
obligado a escribir ahí un resumen de la molécula. En MLM la tarea leía el vector
*de cada posición tapada*, así que **el CLS no se usó nunca**.

Eso predice que con MTR el CLS debería ganar a promediar, y con MLM perder. Y es
lo que sale:

| | con MTR | con MLM |
|---|---|---|
| CLS vs mean (regresión) | +0,018 | **−0,096** (*p* = 0,008) |
| CLS vs mean (clasificación) | +0,068 (*p* = 0,032) | **−0,046** (*p* = 0,062) |

**El efecto del pooling cambia de signo según el checkpoint**, la misma inversión
en las dos tareas. Por eso los dos factores se cruzaron en vez de fijar uno: con
`pooling=cls` la diferencia entre checkpoints sale +0,181 y con la media +0,067;
la primera mezclaría el efecto del checkpoint con el perjuicio de usar un token
que MLM nunca entrenó.

### Dos sustos por medir con una sola semilla

1. **Los pooling parecían empatar.** Con la semilla 42, `mtr_mean` dio 0,292 en
   clasificación, el valor más alto de la primera tabla. Con 3 semillas su media
   real es 0,224 y aquel 0,292 era su mejor tirada de tres. Era ruido.
2. **Congelar parecía ganar** en clasificación con el *learning rate* de partida.
   Repetido con 3 semillas, no.

Por eso todo lo que se afirma aquí va con 3 semillas y su *p*.

## Cambios respecto a `models/chemberta.py` de la raíz

1. **`add_pooling_layer=False`** al cargar el encoder. Esa capa no viene en los
   checkpoints, así que nacía al azar, y `forward()` no la usa. El encoder pasa a
   contarse como 3,28 M pesos (los reales) en vez de 3,43 M, y desaparece el aviso
   `MISSING`.
2. **`pool()` lanza error** ante un `pooling` desconocido. Antes un error de
   escritura (`CLS` en mayúsculas) caía en silencio en la media.
3. **`pooling: concat`** (nuevo): concatena CLS y media; la cabeza recibe 768.
4. **`pretrained`** (nuevo, defecto `true`). Con `false`, misma arquitectura con
   pesos al azar: el *ablation* del preentrenamiento. Sin él el modelo no aprende
   nada (0,982 en regresión, que es predecir la media, y MCC 0,000).
   No es lo mismo que `freeze_encoder`: `pretrained` fija con qué valores
   **arranca** el encoder, `freeze_encoder` si esos valores **cambian**.
5. **`augment`** (nuevo, defecto `false`): aleatorización de SMILES, aplicada
   **solo en entrenamiento**, así que validación y test ven el SMILES canónico.

## Notas de método

- **La semilla de entrenamiento no es la del reparto.** El reparto sigue siendo el
  de la semilla 42; las 112 ejecuciones comparten el mismo `split_sha256`.
- **Con 3 semillas solo se detectan efectos grandes.** Que salga *p* = 0,21 no
  prueba que no haya diferencia: prueba que, si la hay, es menor que la variación
  entre semillas.
- **Comparaciones múltiples:** el barrido del *learning rate* son 5 contrastes
  contra la misma referencia, así que allí el umbral es Bonferroni
  0,05/5 = 0,01.
- **Dar a cada ejecución su `hydra.run.dir` con nombre.** Por defecto la carpeta
  es la fecha con segundos, y dos trabajos que arranquen en el mismo segundo
  comparten carpeta y se pisan.
- **La cola es el cuello de botella, no el cálculo.** De ahí los `batch*.sh`, que
  agrupan varias ejecuciones en un solo trabajo.

## Pendiente para el informe del grupo

**Hay que fijar un solo *baseline* de referencia.** El informe de Daniel compara
con Morgan MLP (0,843 en validación) y este con LightGBM (0,740). El enunciado
pide elegir **uno** de los cuatro.

Y recordar que la tabla de la parte 5 compara *«final validation-selected
configurations»*: LightGBM y Atom-JEPA están aquí con sus ajustes **por defecto**,
así que sus cifras subirán cuando cada uno optimice el suyo.
