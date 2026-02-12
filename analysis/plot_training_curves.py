from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_one(metrics_path: str, out_dir: str):
    df = pd.read_csv(metrics_path)
    exp = os.path.basename(os.path.dirname(metrics_path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title(f"{exp} loss")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_title(f"{exp} accuracy")
    axes[1].legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp}_curves.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(results_root: str, out_dir: str):
    for metrics_path in glob.glob(os.path.join(results_root, "*", "metrics.csv")):
        plot_one(metrics_path, out_dir)
    print(f"saved curves to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs/results")
    parser.add_argument("--out-dir", default="outputs/figures/training_curves")
    args = parser.parse_args()
    main(args.results_root, args.out_dir)
