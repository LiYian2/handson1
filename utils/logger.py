from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class CSVLogger:
    output_dir: str
    filename: str = "metrics.csv"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, self.filename)
        self._initialized = os.path.exists(self.path)

    def log(self, row: Dict[str, float]) -> None:
        write_header = not self._initialized
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
