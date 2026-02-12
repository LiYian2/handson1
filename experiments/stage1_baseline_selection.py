from __future__ import annotations

import copy
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.run_experiment import run
from experiments.common import load_config


BASE = "configs/experiments/exp1_baseline.yaml"
MODELS = ["resnet9", "resnet18", "resnet50", "vgg11"]


if __name__ == "__main__":
    cfg = load_config(BASE)
    for name in MODELS:
        c = copy.deepcopy(cfg)
        c["model"]["name"] = name
        c["experiment"]["name"] = f"stage1_{name}"
        tmp_path = f"/tmp/stage1_{name}.yaml"
        import yaml

        with open(tmp_path, "w") as f:
            yaml.safe_dump(c, f)
        run(tmp_path, dry_run=True)
