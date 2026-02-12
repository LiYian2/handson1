from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


def latest_metric_row(csv_path: str):
    df = pd.read_csv(csv_path)
    return df.iloc[-1]


def main(results_root: str = "outputs/results"):
    rows = []
    for metrics_path in glob.glob(os.path.join(results_root, "*", "metrics.csv")):
        exp = os.path.basename(os.path.dirname(metrics_path))
        row = latest_metric_row(metrics_path)
        rows.append(
            {
                "experiment": exp,
                "val_acc": float(row.get("val_acc", 0.0)),
                "val_loss": float(row.get("val_loss", 0.0)),
                "epoch": int(row.get("epoch", -1)),
            }
        )

    if not rows:
        print(f"no metrics.csv found under: {results_root}")
        return

    out = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    os.makedirs("outputs/figures/comparison_plots", exist_ok=True)
    out_path = "outputs/figures/comparison_plots/model_ranking.csv"
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs/results")
    args = parser.parse_args()
    main(args.results_root)
