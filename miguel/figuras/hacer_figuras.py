#!/usr/bin/env python3
"""Report figures for Part 4 (ChemBERTa-2). Reads runs/, writes PDF + PNG.

Paper style: the images carry axes, data and legend only. Captions are NOT drawn
inside the image - they live in captions.tex, to be used with \\caption{} so LaTeX
numbers them and the text can cite them with \\ref{}. Panel titles state what is
plotted, never what it means; interpretation belongs in the caption or the body.

One figure per experiment. Curves wherever the quantity varies over epochs;
the only non-curve is the validation->test slope chart, where each run
contributes a single number per split.

    cd ~/dtu-cyp-project && python ~/Proyecto-Deep-Learning/figuras/hacer_figuras.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS, OUT = Path('runs'), Path(__file__).parent

# Categorical slots 1-3 of the validated reference palette (light surface),
# each paired with a line style so identity never rests on colour alone.
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
DASH = ['-', (0, (5, 2)), (0, (1, 1.6))]
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#dcdbd6'
REG, CLS = 'macro_ST-RAE', 'macro_MCC'
TASKS = [('regression', REG, 'MA-ST-RAE $\\downarrow$'),
         ('classification', CLS, 'mean MCC $\\uparrow$')]
ENZ = {'CYP1A2_pIC50_direct_inhibition': 'CYP1A2', 'CYP2C9_pIC50_direct_inhibition': 'CYP2C9',
       'CYP2D6_pIC50_direct_inhibition': 'CYP2D6', 'CYP3A4_pIC50_direct_inhibition': 'CYP3A4',
       'CYP2D6_is_TDI': 'CYP2D6', 'CYP3A4_is_TDI': 'CYP3A4'}

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


def curve(task, name, key):
    p = RUNS / task / 'chemberta' / name / 'history.jsonl'
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    return (np.array([r['epoch'] for r in rows]),
            np.array([r['train_loss'] for r in rows]),
            np.array([r[key] for r in rows]))


def final(task, name, key, split='validation'):
    p = RUNS / task / 'chemberta' / name / f'{split}_metrics.json'
    return json.loads(p.read_text())[key] if p.exists() else None


def other(task, model, key, split='validation'):
    d = next((RUNS / task / model).glob('*'))
    return json.loads((d / f'{split}_metrics.json').read_text())[key]


def smooth(y, w=3):
    """Centred rolling mean; the raw curve is still drawn underneath."""
    if len(y) < w:
        return y
    pad = np.r_[np.repeat(y[0], w // 2), y, np.repeat(y[-1], w // 2)]
    return np.convolve(pad, np.ones(w) / w, mode='valid')[:len(y)]


def trend(ax, x, y, colour, ls, label=None):
    """Raw curve translucent, smoothed curve on top: the trend without hiding the data."""
    ax.plot(x, y, color=colour, lw=0.9, alpha=0.22, zorder=1)
    ax.plot(x, smooth(y), color=colour, ls=ls, label=label, zorder=2)


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


def mark_best(ax, ep, val, lower, colour):
    i = int(np.argmin(val) if lower else np.argmax(val))
    ax.plot(ep[i], val[i], 'o', ms=5, color=colour, mec=SURFACE, mew=1.2, zorder=5)


def panel_tags(axes, tags=None):
    for ax, t in zip(np.ravel(axes), tags or 'abcdefgh'):
        ax.text(-0.17, 1.06, f'({t})', transform=ax.transAxes, fontsize=9.5,
                fontweight='bold', color=INK, va='bottom')


def legend_on_top(fig, handles, labels, ncol=3):
    """Legend above the axes, no caption block: the caption is LaTeX's job.

    Space is reserved in inches so the legend never overlaps the panel titles.
    """
    h = fig.get_size_inches()[1]
    legend_in = 0.26
    fig.tight_layout(rect=(0, 0, 1, 1 - (legend_in + 0.08) / h))
    fig.legend(handles, labels, loc='upper center', ncol=ncol,
               bbox_to_anchor=(0.5, 1.0))


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=240)
    plt.close(fig)
    print(f'  {stem}')


def grid_fig(series, stem, ncol=3):
    """2x2: rows = task, cols = (training loss, validation metric). One line per series."""
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    for row, (task, key, lab) in enumerate(TASKS):
        lower = task == 'regression'
        for (name, nom), col, ls in zip(series, (C1, C2, C3), DASH):
            c = curve(task, name.format(task=task), key)
            if c is None:
                continue
            ep, loss, val = c
            axes[row, 0].plot(ep, loss, color=col, ls=ls, label=nom)
            trend(axes[row, 1], ep, val, col, ls, nom)
            mark_best(axes[row, 1], ep, val, lower, col)
        tl = 'squared error' if task == 'regression' else 'cross-entropy'
        frame(axes[row, 0], 'Epoch', f'Training loss ({tl})',
              'Regression' if lower else 'Classification')
        frame(axes[row, 1], 'Epoch', f'Validation {lab}',
              'Regression' if lower else 'Classification')
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=ncol)
    save(fig, stem)


# --- 1. pretraining objective ----------------------------------------------
def fig_pretraining():
    grid_fig([('mlm_cls', 'MLM'), ('mtr_cls', 'MTR'), ('rand_cls', 'not pretrained')],
             'fig1_pretraining_objective')


# --- 2. pooling, per checkpoint --------------------------------------------
def fig_pooling():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    for row, (task, key, lab) in enumerate(TASKS):
        lower = task == 'regression'
        for ckpt_i, ckpt in enumerate(['mlm', 'mtr']):
            ax = axes[row, ckpt_i]
            for pool, col, ls in zip(['cls', 'mean', 'concat'], (C1, C2, C3), DASH):
                c = curve(task, f'{ckpt}_{pool}', key)
                if c is None:
                    continue
                ep, _, val = c
                trend(ax, ep, val, col, ls, pool)
                mark_best(ax, ep, val, lower, col)
            frame(ax, 'Epoch', f'Validation {lab}',
                  f'{ckpt.upper()} — {"Regression" if lower else "Classification"}')
        lo = min(axes[row, i].get_ylim()[0] for i in (0, 1))
        hi = max(axes[row, i].get_ylim()[1] for i in (0, 1))
        for i in (0, 1):
            axes[row, i].set_ylim(lo, hi)          # same scale across a row
    panel_tags(axes)
    legend_on_top(fig, *axes[0, 0].get_legend_handles_labels(), ncol=3)
    save(fig, 'fig2_pooling')


# --- 3. frozen vs fine-tuned ------------------------------------------------
def fig_frozen():
    grid_fig([('mtr_cls', 'encoder fine-tuned'), ('froz_mtr_cls', 'encoder frozen')],
             'fig3_frozen_vs_finetuned', ncol=2)


# --- 4. augmentation, mean +- s.d. over three seeds -------------------------
def fig_augmentation():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (task, key, lab) in zip(axes, TASKS):
        lower = task == 'regression'
        for (tag, nom), col, ls in zip([('noaug', 'no augmentation'),
                                        ('aug', 'randomised SMILES')], (C1, C2), DASH):
            runs = [curve(task, f'mtr_cls_{tag}_s{s}', key) for s in (1, 2, 3)]
            runs = [r for r in runs if r is not None]
            n = min(len(r[0]) for r in runs)
            ep = runs[0][0][:n]
            v = np.vstack([r[2][:n] for r in runs])
            m, sd = v.mean(0), v.std(0, ddof=1)
            ax.fill_between(ep, m - sd, m + sd, color=col, alpha=0.16, lw=0)
            ax.plot(ep, m, color=col, ls=ls, label=nom)
        frame(ax, 'Epoch', f'Validation {lab}',
              'Regression' if lower else 'Classification')
    panel_tags(axes)
    legend_on_top(fig, *axes[0].get_legend_handles_labels(), ncol=2)
    save(fig, 'fig4_augmentation')


# --- 5. validation -> test slope chart --------------------------------------
def fig_val_test():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, (task, key, lab) in zip(axes, TASKS):
        cb = {sp: float(np.mean([final(task, f'mtr_cls_noaug_s{s}', key, sp)
                                 for s in (1, 2, 3)])) for sp in ('validation', 'test')}
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
    save(fig, 'fig5_validation_to_test')


# --- 6. per-endpoint curves for the selected configuration ------------------
def fig_per_enzyme():
    """One small panel per endpoint: four curves in one axes were unreadable.

    Explicit gridspec geometry, so the panels keep equal widths across both rows.
    """
    fig = plt.figure(figsize=(6.6, 3.9))
    gs = fig.add_gridspec(2, 4, left=0.085, right=0.985, top=0.90, bottom=0.11,
                          hspace=0.70, wspace=0.28)
    tags = iter('abcdef')
    for row, (task, key, _) in enumerate(TASKS):
        lower = task == 'regression'
        rows = [json.loads(l) for l in
                (RUNS / task / 'chemberta' / 'mtr_cls_noaug_s1' / 'history.jsonl').open()]
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
            trend(ax, ep, v, C1, '-')
            j = int(np.argmin(v) if lower else np.argmax(v))
            ax.plot(ep[j], v[j], 'o', ms=5, color=C1, mec=SURFACE, mew=1.2, zorder=5)
            # Label away from the nearest edge, so it never lands on the axis line.
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
    save(fig, 'fig6_per_endpoint')


if __name__ == '__main__':
    print('Figures written to', OUT)
    fig_pretraining(); fig_pooling(); fig_frozen()
    fig_augmentation(); fig_val_test(); fig_per_enzyme()
