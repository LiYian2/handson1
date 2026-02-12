from __future__ import annotations

import argparse
import os
import shutil


def copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def main(results_root: str, report_fig_dir: str):
    # Copy top-level generated figures into report folder.
    mapping = {
        "outputs/figures/comparison_plots/model_ranking.csv": os.path.join(report_fig_dir, "model_ranking.csv"),
    }
    for src, dst in mapping.items():
        copy_if_exists(src, dst)

    print(f"report figures prepared under: {report_fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="outputs/results")
    parser.add_argument("--report-fig-dir", default="report/figures")
    args = parser.parse_args()
    main(args.results_root, args.report_fig_dir)
