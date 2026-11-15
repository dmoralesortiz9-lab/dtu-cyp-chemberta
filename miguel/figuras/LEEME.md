# Figuras para el informe (Parte 4 — ChemBERTa-2)

Una figura por pregunta, en el orden en que se cuentan. Todas en PDF (vectorial,
para LaTeX) y PNG (para mirarlas en GitHub, que no previsualiza PDF). **Ninguna
lleva pie dentro de la imagen**: los pies están en
`../informe/parte4-chemberta-resultados.tex`, listos para `\caption{}`.

Las seis van al **apéndice** del informe; el cuerpo (máx. 4 páginas) no lleva
ninguna.

## El orden en que contarlo

Lo que se hizo y el orden en que se cuenta no son lo mismo. Los experimentos
salieron según iban apareciendo las dudas; el informe va en el orden que mejor lo
explica. Cada paso fija una cosa y pasa a la siguiente.

| Paso | Pregunta | Respuesta | Figura |
|---|---|---|---|
| 1 | ¿Qué *checkpoint* y qué *pooling*? | **MTR + CLS.** Único efecto grande, y los dos factores **interactúan** | `fig1` |
| 2 | ¿Importa el *learning rate*? | **No**, en tres órdenes de magnitud | `fig2` |
| 3 | ¿Congelar el cuerpo? | **Da igual**, y cuesta 5 veces menos | `fig3` |
| 4 | ¿Aleatorizar los SMILES? | **No ayuda** | `fig4` |
| 5 | ¿De dónde sale el resultado? | De CYP3A4 y CYP2C9, las que más datos tienen | `fig5` |
| 6 | Comparación final | Gana al *baseline*, pierde contra Atom-JEPA | `fig6` + tabla |

En los pasos 2–4 la respuesta es «da igual», y por eso **cada figura lleva el
$p$ dentro del panel de curvas**, no solo en el panel del contraste: quien mire
las curvas no tiene que irse al panel de al lado para saber si la separación que
ve es real.

## Por qué el paso 1 es una rejilla y los demás no

Cruzar todo sería inviable: 2 *checkpoints* × 3 *pooling* × 2 (congelar) ×
6 *learning rates* × 2 (aleatorizar) = 144 configuraciones, por 3 semillas y
2 tareas son 864 ejecuciones.

Se cruzan **solo los dos factores que pueden interactuar**, y de hecho interactúan:

| | con MTR | con MLM |
|---|---|---|
| CLS vs media (regresión) | **+0,018** (CLS mejor) | **−0,096** (CLS peor), $p=0{,}008$ |
| CLS vs media (clasificación) | **+0,068**, $p=0{,}032$ | **−0,046**, $p=0{,}062$ |

**El efecto del *pooling* cambia de signo según el *checkpoint*.** Si se hubiera
fijado `pooling=cls` y comparado *checkpoints*, la diferencia habría salido
$+0{,}181$ en vez de $+0{,}067$: mezclaría el efecto del *checkpoint* con el
perjuicio de usar un token que MLM nunca entrenó.

Los factores de entrenamiento (congelar, *learning rate*, aleatorizar) se
exploran **sobre la configuración ya elegida**, porque actúan sobre un modelo
fijado. Se comprobó además que el *learning rate* tampoco se traslada entre
*checkpoints*: con MLM vale 0,071 y con MTR no vale nada.

## Frases que conviene que estén tal cual

> «Las diferencias, de haberlas, son menores que la variación entre semillas.»

Con tres semillas solo se detectan efectos grandes. Que salga $p = 0{,}21$ **no
prueba que no haya diferencia**, prueba que si la hay es más pequeña de lo que
este experimento puede ver.

> «Los dos factores se cruzaron porque pueden interactuar, y de hecho interactúan:
> el efecto del *pooling* cambia de signo según el *checkpoint* ($p = 0{,}008$).»

> «Se barrieron tres órdenes de magnitud en el *learning rate*, con tres semillas
> por punto; el modelo es insensible entre 1e-6 y 3e-4 y se degrada fuera.»

## Cómo citarlas

```latex
\usepackage{graphicx}
\graphicspath{{figuras/}}
...
El efecto del pooling se invierte según el checkpoint (Fig.~\ref{fig:grid}).
Ninguna de las decisiones de entrenamiento supera la variación entre semillas
(Figs.~\ref{fig:lr}--\ref{fig:aug}).
```

## Regenerarlas

```bash
cd dtu-cyp-project
python ../figuras/hacer_figuras.py      # escribe en ../figuras/
```

Lee `runs/`, así que hace falta tenerlo en la raíz del repo.
