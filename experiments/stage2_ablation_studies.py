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
    base["model"]["name"] = "resnet9"

    ablations = [
        ("skip_off", {"model": {"name": "plainnet"}}),
        ("batchnorm_off", {"model": {"use_batchnorm": False}}),
        ("optimizer_adamw", {"training": {"optimizer": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4}}}),
        ("scheduler_step", {"training": {"lr_scheduler": {"type": "StepLR", "step_size": 10, "gamma": 0.1}}}),
        ("activation_gelu", {"model": {"activation": "gelu"}}),
    ]

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v

    for name, patch in ablations:
        c = copy.deepcopy(base)
        merge(c, patch)
        c["experiment"]["name"] = f"stage2_{name}"
        p = f"/tmp/stage2_{name}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(c, f)
        run(p, dry_run=True)
