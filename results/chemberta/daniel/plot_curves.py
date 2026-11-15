"""Plot training loss and validation metric per epoch for one or more runs.

Usage:
    python plot_curves.py RUN_DIR [RUN_DIR ...] -o figure.pdf

Each RUN_DIR must contain history.jsonl (written by finetune.py).
The run label is taken from its .hydra/config.yaml.
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("macro_ST-RAE", "macro_MCC")
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e")


def read_history(run_dir):
    path = pathlib.Path(run_dir) / "history.jsonl"
    epochs, loss, score = [], [], []
    metric_name = None
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if metric_name is None:
                metric_name = next(m for m in METRICS if m in record)
            epochs.append(record["epoch"])
            loss.append(record["train_loss"])
            score.append(record[metric_name])
    return epochs, loss, score, metric_name


def read_label(run_dir):
    path = pathlib.Path(run_dir) / ".hydra" / "config.yaml"
    wanted = ("encoder_lr", "pooling", "freeze_encoder")
    parts = []
    with open(path) as handle:
        for line in handle:
            key = line.strip().split(":")[0]
            if key in wanted:
                parts.append(line.strip().split("#")[0].strip())
    return ", ".join(parts) if parts else pathlib.Path(run_dir).name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("-o", "--output", default="curves.pdf")
    args = parser.parse_args()

    fig, (ax_loss, ax_score) = plt.subplots(
        1, 2, figsize=(10, 3.8), constrained_layout=True)

    metric_name = None
    for i, run in enumerate(args.runs):
        epochs, loss, score, metric_name = read_history(run)
        color = COLORS[i % len(COLORS)]
        label = read_label(run)
        ax_loss.plot(epochs, loss, color=color, label=label)
        ax_score.plot(epochs, score, color=color, label=label)
        best = min(score) if metric_name == "macro_ST-RAE" else max(score)
        ax_score.scatter([epochs[score.index(best)]], [best],
                         color=color, zorder=3, s=40)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training")

    ax_score.set_xlabel("Epoch")
    ax_score.set_ylabel(metric_name)
    ax_score.set_title("Validation")
    if metric_name == "macro_ST-RAE":
        ax_score.axhline(1.0, color="grey", linestyle=":", linewidth=1)

    for ax in (ax_loss, ax_score):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    ax_score.legend(fontsize=8, frameon=False)

    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
