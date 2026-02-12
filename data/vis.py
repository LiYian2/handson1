# Visualization utilities for data and results
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Image
from dataclasses import dataclass

@dataclass
class CFG:
    # Paths
    data_parquet: str = "FashionMnist-Resplit/data.parquet"
    train_meta_csv: str = "FashionMnist-Resplit/train.csv"
    test_meta_csv: str = "FashionMnist-Resplit/test.csv"

    # Column names (customizable)
    id_col: str = "id"
    target_col: str = "label"  # TODO match competition column
    image_col: str = "image"
    split_col: str = "split"

    # Image shape
    H: int = 28
    W: int = 28
    num_classes: int = 10



cfg = CFG()

ds_all = load_dataset("parquet", data_files=cfg.data_parquet)["train"]
ds_all = ds_all.cast_column(
    cfg.image_col, Image(decode=True)
)  # {"bytes": xxx, "path": None} -> PIL.Image.Image
print(ds_all)
print("Columns:", ds_all.column_names)

train_meta = pd.read_csv(cfg.train_meta_csv)
test_meta = pd.read_csv(cfg.test_meta_csv)

assert cfg.id_col in train_meta.columns, f"train.csv must contain {cfg.id_col}"
assert cfg.target_col in train_meta.columns, f"train.csv must contain {cfg.target_col}"
assert cfg.id_col in test_meta.columns, f"test.csv must contain {cfg.id_col}"

train_ids = set(train_meta[cfg.id_col].tolist())
test_ids = set(test_meta[cfg.id_col].tolist())

ds_train_full = ds_all.filter(lambda ex: ex[cfg.split_col] == "train")
ds_test = ds_all.filter(lambda ex: ex[cfg.split_col] == "test")

print("Train rows:", len(ds_train_full))
print("Test rows :", len(ds_test))

# Attach labels to train rows
id_to_label = dict(
    zip(train_meta[cfg.id_col].tolist(), train_meta[cfg.target_col].tolist())
)


def add_label(ex):
    ex[cfg.target_col] = int(id_to_label[ex[cfg.id_col]])
    return ex


ds_train_full = ds_train_full.map(add_label)
print(ds_train_full[0])
ds_train_full[0]["image"]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax = axs[i // 5, i % 5]
    img = ds_train_full[i]["image"]
    label = ds_train_full[i][cfg.target_col]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()