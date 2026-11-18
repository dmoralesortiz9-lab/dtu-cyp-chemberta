# Figures for the report (Part 4 — ChemBERTa-2)

One figure per question, in the order they are told in. Each is provided as PDF
(vector, for LaTeX) and PNG (to view on GitHub, which does not preview PDFs).
**No figure carries a caption inside the image**: the captions live in the report
itself, ready for `\caption{}`, so LaTeX numbers them and the text can cite them
with `\ref{}`.

All seven belong in the report's **appendix**; the body (max. 4 pages) carries
none. In the appendix `fig7` goes **first**, because it is the one that justifies
the three-seed protocol and the methodology section cites it.

`fig7_training_curves` is the equivalent of the root `plot_curves.py`, drawn from
our runs with three seeds: training loss against validation metric, for the
starting point, the best MLM model and the selected configuration. It is the only
figure that shows the training loss.

## The order to tell it in

What was done and the order it is told in are not the same thing. The experiments
came out as the questions appeared; the report goes in the order that explains it
best. Each step fixes one thing and moves on.

| Step | Question | Answer | Figure |
|---|---|---|---|
| 0 | Why three seeds? | The loss falls smoothly while validation rebounds | `fig7` |
| 1 | Which *checkpoint* and which *pooling*? | **MTR + CLS.** The only large effect, and the two factors **interact** | `fig1` |
| 2 | Does the *learning rate* matter? | **No**, across three orders of magnitude | `fig2` |
| 3 | Freeze the encoder? | **Makes no difference**, and costs 5× less | `fig3` |
| 4 | Randomise the SMILES? | **Does not help** | `fig4` |
| 5 | Where does the result come from? | CYP3A4 and CYP2C9, the ones with the most data | `fig5` |
| 6 | Final comparison | Beats the *baseline*, loses to Atom-JEPA | `fig6` + table |

In steps 2–4 the answer is "no difference", which is why **every figure carries
the $p$ value inside the curves panel**, not only in the contrast panel: a reader
looking at the curves should not have to move to the next panel to learn whether
the separation they see is real.

## Why step 1 is a grid and the others are not

Crossing everything was never an option: 2 *checkpoints* × 3 *poolings* ×
2 (freezing) × 6 *learning rates* × 2 (randomisation) = 144 configurations, which
over 3 seeds and 2 tasks is 864 runs.

Only **the two factors that can interact** are crossed, and they do interact:

| | with MTR | with MLM |
|---|---|---|
| CLS vs mean (regression) | **+0.018** (CLS better) | **−0.096** (CLS worse), $p=0.008$ |
| CLS vs mean (classification) | **+0.068**, $p=0.032$ | **−0.046**, $p=0.062$ |

**The pooling effect changes sign with the checkpoint.** Had the pooling been
fixed to `cls` and the checkpoints then compared, the difference would have read
$+0.181$ instead of $+0.067$: that would confound the checkpoint's effect with
the penalty of using a token MLM never trained.

The training factors (freezing, learning rate, randomisation) are explored **on
top of the already selected configuration**, because they act on a fixed model.
The learning rate was also checked not to transfer between checkpoints: on MLM it
is worth 0.071 and on MTR nothing at all.

## Sentences worth keeping verbatim

> "Any difference there may be is smaller than the variation between seeds."

Three seeds detect only large effects. A $p = 0.21$ **does not prove there is no
difference**; it proves that if there is one, it is smaller than this experiment
can see.

> "The two factors were crossed because they can interact, and they do: the
> pooling effect changes sign with the checkpoint ($p = 0.008$)."

> "The learning rate was swept over three orders of magnitude, three seeds per
> point; the model is insensitive between 1e-6 and 3e-4 and degrades outside it."

## How to cite them

```latex
\usepackage{graphicx}
\graphicspath{{figures/}}
...
The pooling effect inverts with the checkpoint (Fig.~\ref{fig:grid}).
None of the training decisions exceeds the variation between seeds
(Figs.~\ref{fig:lr}--\ref{fig:aug}).
```

## Regenerating them

```bash
# from the directory that holds runs/, i.e. the starter directory
python part4/chemberta/miguel/figures/make_figures.py
```

It reads `runs/` from the working directory and writes next to itself, so `runs/`
has to be there.
