#!/usr/bin/env python3
"""Report figures for Part 4 (ChemBERTa-2). Reads runs/, writes PDF + PNG.

One figure per question, in the order the report tells it:

    1. checkpoint x pooling   -> which model, and which summary vector
    2. learning rate          -> does it matter (on the winner)
    3. frozen vs fine-tuned   -> does it matter
    4. SMILES randomisation   -> does it matter
    5. per-endpoint           -> where the result comes from
    6. validation -> test     -> the final comparison

Every figure shows BOTH the curves and the contrast, because "it makes no
difference" is a claim that needs the measurement next to it. Curves are the mean
over three training seeds with a +-1 s.d. band; contrasts are Welch tests with 95 %
intervals, signed so positive always means "the chosen option is better".

Paper style: no caption is drawn inside the image. Captions live in the report .tex,
for \\caption{}, so LaTeX numbers them and the text can cite them with \\ref{}.
Panel titles say what is plotted, never what it means.

    cd ~/dtu-cyp-project && python ~/Proyecto-Deep-Learning/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RUNS, OUT = Path('runs'), Path(__file__).parent

# Categorical slots 1-4 of the validated reference palette (light surface), each
# paired with a line style so identity never rests on colour alone.
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
DASH = ['-', (0, (5, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2))]
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#dcdbd6'
REG, CLS = 'macro_ST-RAE', 'macro_MCC'
TASKS = [('regression', REG, 'MA-ST-RAE $\\downarrow$'),
         ('classification', CLS, 'mean MCC $\\uparrow$')]
ENZ = {'CYP1A2_pIC50_direct_inhibition': 'CYP1A2', 'CYP2C9_pIC50_direct_inhibition': 'CYP2C9',
       'CYP2D6_pIC50_direct_inhibition': 'CYP2D6', 'CYP3A4_pIC50_direct_inhibition': 'CYP3A4',
       'CYP2D6_is_TDI': 'CYP2D6', 'CYP3A4_is_TDI': 'CYP3A4'}
REFERENCE = 'mtr_cls_noaug'          # the selected configuration

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.7, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.7,
    'ytick.major.width': 0.7, 'legend.frameon': False, 'lines.linewidth': 1.6,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})


# ---------------------------------------------------------------- reading
def seed_curves(task, name, key, n=3):
    """Per-epoch validation score for every seed. Returns (epochs, matrix)."""
    rows = []
    for s in range(1, n + 1):
        p = RUNS / task / 'chemberta' / f'{name}_s{s}' / 'history.jsonl'
        if not p.exists():
            return None
        rows.append([json.loads(l) for l in p.open()])
    m = min(len(r) for r in rows)
    ep = np.array([r['epoch'] for r in rows[0][:m]])
    return ep, np.vstack([[r[key] for r in run[:m]] for run in rows])


def seed_finals(task, name, key, n=3, split='validation'):
    """Selected-epoch score for every seed (what finetune.py saved)."""
    out = []
    for s in range(1, n + 1):
        p = RUNS / task / 'chemberta' / f'{name}_s{s}' / f'{split}_metrics.json'
        if not p.exists():
            return None
        out.append(json.loads(p.read_text())[key])
    return np.array(out)


def other(task, model, key, split='validation'):
    d = next((RUNS / task / model).glob('*'))
    return json.loads((d / f'{split}_metrics.json').read_text())[key]


def contrast(task, key, chosen, alternative):
    """Welch contrast, signed so positive = the chosen option is better."""
    a, b = seed_finals(task, chosen, key), seed_finals(task, alternative, key)
    if a is None or b is None:
        return None
    if task == 'regression':          # smaller is better, so flip
        a, b = b, a
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (se**2)**2 / ((a.var(ddof=1)/len(a))**2/(len(a)-1)
                       + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    return d, stats.t.ppf(0.975, df) * se, 2 * (1 - stats.t.cdf(abs(d / se), df))


# ---------------------------------------------------------------- drawing
def frame(ax, xlab=None, ylab=None, title=None):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', alpha=1.0)
    ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    if title:
        ax.set_title(title, loc='left', color=INK)


def band(ax, ep, mat, colour, ls, label=None):
    """Mean over seeds with a +-1 s.d. band: the trend and its spread together."""
    m, sd = mat.mean(0), mat.std(0, ddof=1)
    ax.fill_between(ep, m - sd, m + sd, color=colour, alpha=0.15, lw=0, zorder=1)
    ax.plot(ep, m, color=colour, ls=ls, label=label, zorder=2)


def forest(ax, rows, xlabel):
    """Contrasts as point + 95 % interval. Filled marker = significant."""
    ys = [-i for i in range(len(rows))]
    for y, (_, d, half, p) in zip(ys, rows):
        sig = p < 0.05
        ax.errorbar(d, y, xerr=half, fmt='o', ms=6, color=C1,
                    mfc=C1 if sig else SURFACE, mec=C1, mew=1.6,
                    ecolor=C1, elinewidth=1.4, capsize=3, zorder=3)
    ax.axvline(0, color=INK2, lw=0.9, ls=':', zorder=1)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis='y', length=0)
    lo = min(d - h for _, d, h, _ in rows)
    hi = max(d + h for _, d, h, _ in rows)
    pad = 0.12 * (hi - lo) or 0.01
    ax.set_xlim(min(lo - pad, -pad), hi + 3.8 * pad)
    for y, (_, d, half, p) in zip(ys, rows):
        txt = p_text(p)
        ax.text(hi + 3.6 * pad, y, txt, ha='right', va='center', fontsize=7.5,
                color=INK if p < 0.05 else INK2,
                fontweight='bold' if p < 0.05 else 'normal')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', alpha=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)


def p_text(p, op='='):
    """Three decimals below 0.05: rounding 0.014 to "0.01" would make it look as
    if it met the Bonferroni threshold, which is exactly 0.01."""
    if p < 0.001:
        return '$p < 0.001$'
    return f'$p {op} {p:.3f}$' if p < 0.05 else f'$p {op} {p:.2f}$'



def annotate_p(ax, lines, loc='lower right'):
    """Put the verdict inside the curve panel, so it reads without the forest.

    Each line is (label, p). Significant ones are bold; the rest carry 'n.s.',
    which is what the reader is actually looking for in these figures.
    """
    if not lines:
        return
    txt = '\n'.join(f'{lab}: {p_text(p, op)}' + ('' if p < 0.05 else ' n.s.')
                    for lab, p, op in ((l + ('=',))[:3] for l in lines))
    va, y = ('bottom', 0.035) if 'lower' in loc else ('top', 0.965)
    ha, x = ('right', 0.975) if 'right' in loc else ('left', 0.025)
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, fontsize=7.2,
            color=INK2, linespacing=1.4, zorder=6,
            bbox=dict(boxstyle='round,pad=0.35', fc=SURFACE, ec=GRID, lw=0.6))


def legend_on_top(fig, handles, labels, ncol=3, legend_in=0.26):
    h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - (legend_in + 0.08) / h))
    fig.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.0))


def panel_tags(axes, tags=None):
    for ax, t in zip(np.ravel(axes), tags or 'abcdefgh'):
        ax.text(-0.17, 1.06, f'({t})', transform=ax.transAxes, fontsize=9.5,
                fontweight='bold', color=INK, va='bottom')


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=240)
    plt.close(fig)
    print(f'  {stem}')


def one_factor(stem, series, contrasts, figsize=(6.6, 4.6), ncol=2, loc='lower right'):
    """Curves on the left, the contrast that settles them on the right.

    The p-values are repeated inside the curve panel: a reader looking at the
    curves should not have to move to the next panel to learn whether the gap
    they are seeing is real.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for row, (task, key, lab) in enumerate(TASKS):
        for (name, nom), col, ls in zip(series, (C1, C2, C3, C4), DASH):
            c = seed_curves(task, name, key)
            if c is None:
                continue
            band(axes[row, 0], c[0], c[1], col, ls, nom)
        frame(axes[row, 0], 'Epoch', f'Validation {lab}',
              'Regression' if task == 'regression' else 'Classification')
        rows = [(nom, *contrast(task, key, a, b))
                for nom, a, b in contrasts if contrast(task, key, a, b)]
        if rows:
            forest(axes[row, 1], rows, f'Advantage ({lab.split(" $")[0]})')
            axes[row, 1].set_title('Regression' if task == 'regression' else 'Classification',
                                   loc='left', color=INK)
            annotate_p(axes[row, 0],
                       [(nom.replace('\n', ' '), p) for nom, _, _, p in rows], loc)
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=ncol)
    save(fig, stem)


# --- 1. checkpoint x pooling ------------------------------------------------
def fig_grid():
    """Four series, so the legend goes INSIDE the curve panel in two columns.

    A four-item legend on top of a 2x2 grid took as much height as the data.
    The p-values are not repeated here either: they sit in the panel next door.
    """
    series = [(REFERENCE, 'MTR + CLS'), ('mtr_mean', 'MTR + mean'),
              ('mlm_cls', 'MLM + CLS'), ('mlm_mean', 'MLM + mean')]
    contrasts = [('MTR vs MLM\n(with CLS)', REFERENCE, 'mlm_cls'),
                 ('MTR vs MLM\n(with mean)', 'mtr_mean', 'mlm_mean'),
                 ('CLS vs mean\n(with MTR)', REFERENCE, 'mtr_mean'),
                 ('CLS vs mean\n(with MLM)', 'mlm_cls', 'mlm_mean')]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    for row, (task, key, lab) in enumerate(TASKS):
        for (name, nom), col, ls in zip(series, (C1, C2, C3, C4), DASH):
            c = seed_curves(task, name, key)
            if c is not None:
                band(axes[row, 0], c[0], c[1], col, ls, nom)
        frame(axes[row, 0], 'Epoch', f'Validation {lab}',
              'Regression' if task == 'regression' else 'Classification')
        rows = [(nom, *contrast(task, key, a, b))
                for nom, a, b in contrasts if contrast(task, key, a, b)]
        if rows:
            forest(axes[row, 1], rows, f'Advantage ({lab.split(" $")[0]})')
            axes[row, 1].set_title('Regression' if task == 'regression'
                                   else 'Classification', loc='left', color=INK)
    # Four series over four panels: an inside legend covered the data, so it goes
    # on top in a single row and the figure is taller to pay for it.
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=4)
    save(fig, 'fig1_checkpoint_pooling')


# --- 2. learning rate -------------------------------------------------------
LR_POINTS = [('1e-6', 1e-6), ('1e-5', 1e-5), ('3e-5', 3e-5),
             ('1e-4', 1e-4), ('3e-4', 3e-4), ('1e-3', 1e-3)]
PLATEAU = (1e-6, 3e-4)
# The learning rate is ORDERED, so it gets an ordinal ramp (one hue, light to
# dark) rather than unrelated hues: the eye then reads the sequence, not six
# arbitrary categories. Steps 250-700 of the blue ramp; 250 is the lightest that
# still clears 2:1 against this surface.
LR_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#184f95', '#0d366b']


def fig_lr():
    """Left: the curve over epochs for each lr. Right: the lr curve itself."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    for row, (task, key, lab) in enumerate(TASKS):
        for (tag, _), col in zip(LR_POINTS, LR_RAMP):
            name = REFERENCE if tag == '1e-5' else f'mtr_cls_lr{tag}'
            c = seed_curves(task, name, key)
            if c is None:
                continue
            # Dashed only for the two ends, which are the ones that break: the
            # ramp carries the order, the dash marks "outside the plateau".
            ls = DASH[1] if tag in ('1e-6', '1e-3') else '-'
            axes[row, 0].plot(c[0], c[1].mean(0), color=col, ls=ls, lw=1.3, label=tag)
        frame(axes[row, 0], 'Epoch', f'Validation {lab}',
              'Regression' if task == 'regression' else 'Classification')
        # The verdict for the whole sweep, inside the curve panel.
        ps = [contrast(task, key, REFERENCE, f'mtr_cls_lr{t}')[2]
              for t, _ in LR_POINTS if t != '1e-5'
              and contrast(task, key, REFERENCE, f'mtr_cls_lr{t}')]
        plateau_p = [p for p, (t, _) in zip(ps, [x for x in LR_POINTS if x[0] != '1e-5'])
                     if t != '1e-3']
        if plateau_p:
            # min over the plateau, labelled as a bound so it does not read as
            # a single test.
            annotate_p(axes[row, 0],
                       [('plateau 1e-6 to 3e-4', min(plateau_p), '\\geq'),
                        ('1e-3', ps[-1], '=')], 'lower right')
        xs, ms, sds = [], [], []
        for tag, x in LR_POINTS:
            name = REFERENCE if tag == '1e-5' else f'mtr_cls_lr{tag}'
            v = seed_finals(task, name, key)
            if v is None:
                continue
            xs.append(x); ms.append(v.mean()); sds.append(v.std(ddof=1))
        xs, ms, sds = np.array(xs), np.array(ms), np.array(sds)
        ax = axes[row, 1]
        ax.axvspan(*PLATEAU, color=C1, alpha=0.07, lw=0, zorder=0)
        ax.fill_between(xs, ms - sds, ms + sds, color=C1, alpha=0.18, lw=0, zorder=1)
        ax.plot(xs, ms, color=C1, marker='o', ms=4.5, mec=SURFACE, mew=1.1, zorder=3)
        ax.plot([1e-5], [ms[list(xs).index(1e-5)]], 'o', ms=9, mfc='none',
                mec=INK, mew=1.2, zorder=4)
        ax.set_xscale('log')
        ax.set_xticks([p[1] for p in LR_POINTS], [p[0] for p in LR_POINTS], rotation=45)
        ax.minorticks_off()
        frame(ax, 'Encoder learning rate', f'Validation {lab}',
              'Regression' if task == 'regression' else 'Classification')
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=6)
    save(fig, 'fig2_learning_rate')


# --- 3. frozen vs fine-tuned ------------------------------------------------
def fig_frozen():
    one_factor('fig3_frozen',
               [(REFERENCE, 'encoder fine-tuned'), ('froz_mtr_cls_noaug', 'encoder frozen')],
               [('Fine-tuned\nvs frozen', REFERENCE, 'froz_mtr_cls_noaug')])


# --- 4. SMILES randomisation ------------------------------------------------
def fig_augmentation():
    one_factor('fig4_augmentation',
               [(REFERENCE, 'plain SMILES'), ('mtr_cls_aug', 'randomised SMILES')],
               [('Plain vs\nrandomised', REFERENCE, 'mtr_cls_aug')])


# --- 5. per-endpoint --------------------------------------------------------
def fig_per_enzyme():
    """One small panel per endpoint: four curves in one axes were unreadable."""
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(2, 4, left=0.085, right=0.985, top=0.90, bottom=0.11,
                          hspace=0.70, wspace=0.28)
    tags = iter('abcdef')
    for row, (task, key, _) in enumerate(TASKS):
        lower = task == 'regression'
        rows = [json.loads(l) for l in
                (RUNS / task / 'chemberta' / f'{REFERENCE}_s1' / 'history.jsonl').open()]
        ep = np.array([r['epoch'] for r in rows])
        sub = 'ST-RAE' if lower else 'MCC'
        keys = [k for k in rows[0] if k in ENZ]
        series = {ENZ[k]: np.array([r[k][sub] for r in rows], dtype=float) for k in keys}
        lo = min(v.min() for v in series.values())
        hi = max(v.max() for v in series.values())
        pad = 0.10 * (hi - lo)
        span = 4 // len(series)
        for i, (nom, v) in enumerate(series.items()):
            ax = fig.add_subplot(gs[row, i * span:(i + 1) * span])
            ax.plot(ep, v, color=C1, lw=1.5)
            j = int(np.argmin(v) if lower else np.argmax(v))
            ax.plot(ep[j], v[j], 'o', ms=5, color=C1, mec=SURFACE, mew=1.2, zorder=5)
            below = (v[j] - (lo - pad)) < 0.4 * (hi - lo + 2 * pad)
            ax.annotate(f'{v[j]:.2f}', (ep[j], v[j]), xytext=(5, 5 if below else -9),
                        textcoords='offset points', fontsize=7.5, color=INK)
            if lower:
                ax.axhline(1.0, color=INK2, lw=0.8, ls=':')
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([1, 10, 20])
            frame(ax, 'Epoch' if row else None,
                  ('ST-RAE $\\downarrow$' if lower else 'MCC $\\uparrow$') if i == 0 else None,
                  f'{nom} ({"pIC50" if lower else "TDI"})')
            if i:
                ax.set_yticklabels([])
            ax.text(0.0, 1.16, f'({next(tags)})', transform=ax.transAxes,
                    fontsize=9, fontweight='bold', color=INK, va='bottom')
    save(fig, 'fig5_per_endpoint')


# --- 6. validation -> test --------------------------------------------------
def fig_val_test():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, (task, key, lab) in zip(axes, TASKS):
        cb = {sp: float(seed_finals(task, REFERENCE, key, split=sp).mean())
              for sp in ('validation', 'test')}
        rows = [('LightGBM', [other(task, 'morgan_rdkit_lgbm', key, s) for s in ('validation', 'test')], C3),
                ('ChemBERTa', [cb['validation'], cb['test']], C2),
                ('Atom-JEPA', [other(task, 'atom_jepa', key, s) for s in ('validation', 'test')], C1)]
        for (nom, (v, t), col), ls in zip(rows, DASH):
            ax.plot([0, 1], [v, t], color=col, ls=ls, marker='o', ms=5,
                    mec=SURFACE, mew=1.2, label=nom)
            ax.annotate(f'{v:.3f}', (0, v), xytext=(-7, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=7.5, color=INK)
            ax.annotate(f'{t:.3f}', (1, t), xytext=(7, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=7.5, color=INK)
        ax.set_xticks([0, 1], ['validation', 'test'])
        ax.set_xlim(-0.42, 1.42)
        ax.grid(axis='y', alpha=1.0)
        ax.set_axisbelow(True)
        for s in ('top', 'right', 'bottom'):
            ax.spines[s].set_visible(False)
        ax.set_ylabel(lab)
        ax.set_title('Regression' if task == 'regression' else 'Classification',
                     loc='left', color=INK)
    panel_tags(axes)
    legend_on_top(fig, *axes[0].get_legend_handles_labels(), ncol=3)
    save(fig, 'fig6_validation_to_test')


# --- 7. training loss vs validation metric --------------------------------
# The equivalent of plot_curves.py in the root of the repository, drawn with our
# runs. It is the only figure showing the training loss, and it is what the
# methodology section points at: the loss falls smoothly while the validation
# metric rebounds, so the best epoch is the minimum of a noisy series.
OVERFIT = [('mlm_cls', 'MLM + CLS (starting point)'),
           ('mlm_mean', 'MLM + mean (best MLM)'),
           (REFERENCE, 'MTR + CLS (selected)')]


def fig_training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9))
    for row, (task, key, lab) in enumerate(TASKS):
        for (name, nom), col, ls in zip(OVERFIT, (C3, C4, C1), (DASH[2], DASH[1], DASH[0])):
            rows = []
            for sd in (1, 2, 3):
                p = RUNS / task / 'chemberta' / f'{name}_s{sd}' / 'history.jsonl'
                if p.exists():
                    rows.append([json.loads(l) for l in p.open()])
            if not rows:
                continue
            m = min(len(r) for r in rows)
            ep = np.array([r['epoch'] for r in rows[0][:m]])
            loss = np.vstack([[r['train_loss'] for r in run[:m]] for run in rows])
            val = np.vstack([[r[key] for r in run[:m]] for run in rows])
            band(axes[row, 0], ep, loss, col, ls, nom)
            band(axes[row, 1], ep, val, col, ls, nom)
        tl = 'squared error' if task == 'regression' else 'cross-entropy'
        nm = 'Regression' if task == 'regression' else 'Classification'
        frame(axes[row, 0], 'Epoch', f'Training loss ({tl})', nm)
        frame(axes[row, 1], 'Epoch', f'Validation {lab}', nm)
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=3)
    save(fig, 'fig7_training_curves')


if __name__ == '__main__':
    print('Figures written to', OUT)
    fig_grid(); fig_lr(); fig_frozen()
    fig_augmentation(); fig_per_enzyme(); fig_val_test()
    fig_training_curves()
