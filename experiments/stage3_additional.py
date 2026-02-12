from __future__ import annotations

import copy
import os
import sys
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.common import load_config
from experiments.run_experiment import run


if __name__ == "__main__":
    base = load_config("configs/experiments/exp1_baseline.yaml")
    studies = [
        ("augmentation_on", {"data": {"augment": True}}),
        ("dropout_02", {"model": {"dropout": 0.2}}),
        ("dropout_05", {"model": {"dropout": 0.5}}),
    ]

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    for name, patch in studies:
        c = copy.deepcopy(base)
        merge(c, patch)
        c["experiment"]["name"] = f"stage3_{name}"
        p = f"/tmp/stage3_{name}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(c, f)
        run(p, dry_run=True)
