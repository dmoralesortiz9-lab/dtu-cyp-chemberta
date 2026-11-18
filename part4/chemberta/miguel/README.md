# Part 4 — ChemBERTa-2: Miguel's work

Builds on the `models/chemberta.py` Daniel added. Nothing in this folder
overwrites files in the repository root, so to use the extended code, copy it
over:

```bash
cp part4/chemberta/miguel/models/chemberta.py models/
cp part4/chemberta/miguel/conf/model/chemberta.yaml conf/model/
```

## What is here

| Path | Contents |
|---|---|
| `models/chemberta.py` | The extended model: two fixes and three new settings |
| `conf/model/chemberta.yaml` | Its configuration |
| `figures/` | The **7 appendix figures** (PDF + PNG), the script that builds them, and a `README.md` giving the order to tell them in |
| `results/` | Config, curves and metrics for all **112 runs**, plus `runs_summary.csv` with everything in one table |
| `batch*.sh` | The five batches that were submitted, each with its reason in the header |

`results/` carries **no weights** (`best.pt`, 1.2 GB in total) and no `split.csv`,
which is identical across runs. What is here is enough to regenerate the figures
and to check any number in the report.

## Final result

**Test, evaluated once**, with the configuration fixed on validation:

| Model | Regression (ST-RAE ↓) | Classification (MCC ↑) |
|---|---|---|
| Morgan MLP | 0.864 | — |
| LightGBM | 0.783 | 0.141 |
| **ChemBERTa** (3 seeds) | **0.778 ± 0.010** | **0.183 ± 0.019** |
| Atom-JEPA | **0.739** | **0.231** |

**ChemBERTa lands in the middle, consistently on both tasks.** It beats the
baseline on both — by 30 % over LightGBM on classification — and loses to
Atom-JEPA on both. On regression the margin over LightGBM (0.005) is inside the
noise, so the honest reading is that it **ties** with the baseline.

Reading the molecule as text is enough to match hand-made features, and to
improve on them substantially on the harder task, but it is **no substitute for
seeing the 3D geometry**.

### Selected configuration

```
checkpoint = DeepChem/ChemBERTa-77M-MTR   pooling = cls      freeze_encoder = false
augment = false                           epochs = 20        encoder_lr = 1e-5
```

Chosen on **validation only**: 0.732 ± 0.006 on regression and 0.292 ± 0.021 on
classification.

## What was measured, and what turned out to matter

Every contrast uses 3 seeds per side, Welch's *t* on validation.

| Decision | Effect | Does it matter? |
|---|---|---|
| **Checkpoint: MTR vs MLM** | +0.181 reg., +0.144 clas. | **Yes** (*p* < 0.001 and 0.001) |
| **Pooling: CLS vs mean** (with MTR) | +0.018 reg., +0.068 clas. | **Yes on classification** (*p* = 0.032) |
| Freezing the encoder | +0.007 reg., +0.015 clas. | No (*p* = 0.21 and 0.40) — and it costs 5× less |
| SMILES randomisation | +0.008 reg., +0.017 clas. | No (*p* = 0.13 and 0.30) |
| Learning rate 1e-6 … 3e-4 | within ±0.007 | No (*p* between 0.40 and 1.00) |
| MCC threshold (max-MCC, Youden) | — | Does not beat the fixed 0.5 |

**Of everything swept, the only thing you have to get right is the
checkpoint.** Not because the rest never matters, but because **here it was
measured and it did not**.

### Why MTR wins, and why that predicts the pooling result

MTR's pretraining head is `dense` 384→384 + `out_proj` 384→199: the shape of a
head that reads **only the CLS token**. Since that was the single vector read to
predict the 199 RDKit descriptors, the encoder was forced to write a summary of
the molecule there. Under MLM the task read the vector *at each masked
position*, so **CLS was never used at all**.

That predicts CLS should beat averaging under MTR, and lose under MLM. Which is
what happens:

| | with MTR | with MLM |
|---|---|---|
| CLS vs mean (regression) | +0.018 | **−0.096** (*p* = 0.008) |
| CLS vs mean (classification) | +0.068 (*p* = 0.032) | **−0.046** (*p* = 0.062) |

**The pooling effect changes sign with the checkpoint**, the same inversion on
both tasks. That is why the two factors were crossed instead of fixing one:
under `pooling=cls` the gap between checkpoints reads +0.181 and under the mean
+0.067, and the first would confound the checkpoint's effect with the penalty of
using a token MLM never trained.

### Two scares from measuring with a single seed

1. **The two poolings looked tied.** On seed 42, `mtr_mean` scored 0.292 on
   classification, the highest value in the first table. Over 3 seeds its real
   mean is 0.224, and that 0.292 was its best of three. It was noise.
2. **Freezing looked like a win** on classification at the starting learning
   rate. Repeated over 3 seeds, it was not.

Hence everything claimed here comes with 3 seeds and a *p* value.

## Changes against the root `models/chemberta.py`

1. **`add_pooling_layer=False`** when loading the encoder. That layer is absent
   from these checkpoints, so it was being initialised at random, and
   `forward()` never uses it. The encoder now counts as 3.28 M parameters (the
   real ones) instead of 3.43 M, and the `MISSING` warning is gone.
2. **`pool()` raises** on an unknown `pooling`. Before, a typo (`CLS` in capitals)
   fell through to the mean silently.
3. **`pooling: concat`** (new): concatenates CLS and the mean, so the head gets 768.
4. **`pretrained`** (new, defaults to `true`). With `false`, the same architecture
   with random weights: the pretraining ablation. Without pretraining the model
   learns nothing (0.982 on regression, which is predicting the mean, and MCC
   0.000). It is not the same as `freeze_encoder`: `pretrained` sets the values
   the encoder **starts** from, `freeze_encoder` whether those values **change**.
5. **`augment`** (new, defaults to `false`): SMILES randomisation, applied **to
   training only**, so validation and test always see the canonical string.

## Method notes

- **The training seed is not the split seed.** The split is still the one from
  seed 42; all 112 runs share the same `split_sha256`.
- **Three seeds detect only large effects.** A *p* = 0.21 does not prove there is
  no difference: it proves that any difference is smaller than the spread
  between seeds.
- **Multiple comparisons:** the learning-rate sweep is 5 contrasts against the
  same reference, so the threshold there is Bonferroni 0.05/5 = 0.01.
- **Give every run a named `hydra.run.dir`.** The default folder is the timestamp
  to the second, so two jobs starting in the same second share a folder and
  overwrite each other.
- **The queue is the bottleneck, not the compute.** Hence the `batch*.sh`
  scripts, which group several runs into a single job.

## Open for the group report

**A single reference baseline has to be fixed.** Daniel's section compares
against Morgan MLP (0.843 on validation) and this one against LightGBM (0.740).
The project description asks for **one** of the four.

And note that the Part 5 table compares *"final validation-selected
configurations"*: LightGBM and Atom-JEPA appear here with their **default**
settings, so their figures will improve once each is tuned.
