# Parte 4 — ChemBERTa-2: aportación de Miguel

Todo lo que hay aquí se construye sobre el `models/chemberta.py` que subió Daniel.
Nada de esta carpeta pisa archivos de la raíz: para usar el código ampliado hay que
copiarlo encima (ver más abajo).

## Qué hay

| Carpeta | Contenido |
|---|---|
| `models/chemberta.py` | Versión ampliada del modelo: cuatro cambios y dos ajustes nuevos |
| `conf/model/chemberta.yaml` | Su configuración |
| `batch.sh` | Lanza varios entrenamientos dentro de un mismo trabajo de la cola |
| `probe_chemberta.py`, `probe_pooling.py` | Comprobaciones sueltas sobre el modelo y el pooling |
| `figuras/` | Seis figuras para el informe, en PDF y PNG, con el script que las genera |
| `informe/` | `parte4-chemberta-resultados.pdf`: el registro completo de experimentos |
| `resultados/` | Métricas e historiales de las 64 ejecuciones, y `resumen_runs.csv` |

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

## Un cabo suelto importante

**Estos experimentos usan `encoder_lr` = 1e-5 y los de Daniel demostraron que ese
valor infraentrena** (su barrido da una curva en U con mínimo en 1e-4). Las dos
mitades del trabajo no se han combinado todavía:

| | checkpoint | pooling | encoder_lr | Regresión |
|---|---|---|---|---|
| Daniel | MLM | mean | **1e-4** | 0,792 ± 0,008 |
| Miguel | **MTR** | cls | 1e-5 | 0,732 ± 0,006 |
| Falta | **MTR** | cls / mean | **1e-4** | ? |

Conviene lanzarlo antes de cerrar la configuración final: la rejilla de
checkpoint × pooling se hizo con un learning rate que ahora sabemos subóptimo, así
que las conclusiones podrían moverse.

Ojo también con el baseline de referencia: el informe de Daniel compara con
Morgan MLP (0,843) y este con LightGBM (0,740). Hay que fijar uno para el informe.

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

`resultados/resumen_runs.csv` tiene las 64 ejecuciones en una tabla, con su
configuración y sus notas de validación y test.
