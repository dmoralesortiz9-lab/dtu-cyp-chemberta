# Figuras para el informe (Parte 4 — ChemBERTa-2)

Dos carpetas, según dónde va cada figura. Todas en PDF (vectorial, para LaTeX) y
en PNG (para ver de un vistazo). Ninguna lleva pie dentro de la imagen: los pies
están abajo, listos para `\caption{}`.

```
cuerpo/     la figura que va en el informe de 4 páginas (solo cabe una)
apendice/   las seis que respaldan lo que dice el texto
```

## El orden en que contarlo

Lo que se hizo y el orden en que se cuenta **no son lo mismo**. Los experimentos
se hicieron en el orden en que fueron surgiendo las dudas; el informe se escribe
en el orden que mejor lo explica. En 4 páginas no hay sitio para suspense: el
hallazgo va primero y lo demás lo respalda.

### Cuerpo del informe (≈1 página, 1 figura)

| Párrafo | Qué dice | Apoyo |
|---|---|---|
| **1. Qué modelo y por qué** | ChemBERTa-2, checkpoint `DeepChem/ChemBERTa-77M-MTR`. Lee el SMILES como texto, sin geometría 3D: ahí está el contraste con Atom-JEPA. *El enunciado pide decir explícitamente qué checkpoint se usa.* | — |
| **2. El hallazgo: lo único que importa es el checkpoint** | MTR gana a MLM en las dos tareas y con los tres *pooling*. **Y por qué:** la cabeza de preentrenamiento del MTR (`dense` 384→384 + `out_proj` 384→199) leía únicamente el token CLS, así que el cuerpo quedó obligado a escribir ahí un resumen. En MLM la tarea leía el vector de cada posición tapada, y el CLS no se usó nunca. | Fig. A1 (apéndice) |
| **3. Todo lo demás se midió y no cambia nada** | *Pooling*, congelar el cuerpo, *learning rate* y aleatorización de SMILES: ninguna diferencia supera la variación entre semillas. | **Figura del cuerpo** |
| **4. Resultado y límite** | Tabla final. Gana al *baseline*, pierde contra Atom-JEPA. Leer la molécula como texto iguala a unas *features* hechas a mano, pero no sustituye a ver la geometría 3D. | Fig. A5 (apéndice) |

### Por qué este orden y no el cronológico

El orden en que se hizo fue: rejilla checkpoint × *pooling* → congelar → umbral
→ aleatorizar → 60 épocas → *learning rate* → semillas que faltaban. Contado así,
el lector llega al final sin saber qué importa.

Contado como arriba, la pregunta «¿y no será que no lo ajustasteis bien?» queda
respondida **antes** de que se la haga: el párrafo 3 enseña cuatro cosas medidas y
descartadas, no cuatro cosas sin explorar.

Una frase que conviene que aparezca tal cual, porque es la lectura honesta de la
figura del cuerpo: **las diferencias, de haberlas, son menores que la variación
entre semillas**. Con tres semillas solo se detectan efectos grandes; que salga
$p = 0{,}21$ no prueba que no haya diferencia, prueba que si la hay es más pequeña
que lo que este experimento puede ver.

## Las figuras

### `cuerpo/fig7_contrasts.pdf`

**Las cinco comparaciones con su intervalo de confianza.** Cada una cambia una sola
cosa respecto a la configuración elegida. El efecto está firmado de modo que
positivo siempre significa «la opción elegida es mejor», en las dos tareas. Punto
relleno = significativo al 5 %; hueco = no. La línea de puntos marca el cero.

Pie sugerido:

> **Figura N. Todas las decisiones de diseño, con su intervalo de confianza del
> 95 %.** Cada contraste cambia un solo ajuste respecto a la configuración elegida
> (MTR, CLS, cuerpo sin congelar, sin aleatorizar, `encoder_lr` 1e-5), con tres
> semillas de entrenamiento por lado. El efecto se expresa de modo que un valor
> positivo significa que la opción elegida es mejor, en las dos tareas. Marcador
> relleno: diferencia significativa ($p < 0{,}05$, $t$ de Welch); hueco: no
> significativa. Solo el *checkpoint* separa su intervalo del cero; las otras
> cuatro decisiones quedan dentro de la variación entre semillas.

### `apendice/`

| Archivo | Qué enseña | Se cita en |
|---|---|---|
| `figA1_pretraining_objective` | MLM vs MTR vs sin preentrenar, curvas por época | Párrafo 2 |
| `figA2_pooling` | CLS vs media vs concatenación, por *checkpoint* | Párrafo 3 |
| `figA3_frozen_vs_finetuned` | Cuerpo congelado vs afinado | Párrafo 3 |
| `figA4_augmentation` | Aleatorización de SMILES, 3 semillas con banda | Párrafo 3 |
| `figA5_validation_to_test` | Los tres modelos, de validación a test | Párrafo 4 |
| `figA6_per_endpoint` | Comportamiento por enzima | Párrafo 4 |

Los pies de todas están en `../figuras/figuras.tex`, listos para copiar.

## Cómo citarlas desde el texto

```latex
% en el preámbulo
\usepackage{graphicx}
\graphicspath{{figuras-informe/cuerpo/}{figuras-informe/apendice/}}

% en el texto
... ninguna supera la variación entre semillas (Fig.~\ref{fig:contrasts}),
y el detalle de cada una está en el apéndice (Figs.~\ref{fig:pooling}--\ref{fig:aug}).
```

## Regenerarlas

```bash
cd dtu-cyp-project
python ../figuras/hacer_figuras.py      # escribe en ../figuras/
```

Lee `runs/`, así que hace falta tenerlo. Luego se copian a estas dos carpetas.
