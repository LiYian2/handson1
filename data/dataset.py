from __future__ import annotations

import os
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.transforms import build_transforms


@dataclass
class DataConfig:
    data_dir: str = "./data/FashionMNIST-Resplit"
    batch_size: int = 128
    num_workers: int = 2
    pin_memory: bool = True
    image_size: int = 28
    augment: bool = False


class FashionMNISTResplitDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        split: str,
        transform=None,
        split_csv_path: Optional[str] = None,
    ) -> None:
        if split not in {"train", "test", "all"}:
            raise ValueError(f"Unsupported split: {split}")

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if split != "all":
            df = df[df["split"] == split].copy()

        if split_csv_path and os.path.exists(split_csv_path):
            meta = pd.read_csv(split_csv_path)
            if "id" in meta.columns and "label" in meta.columns:
                id_to_label = dict(zip(meta["id"], meta["label"]))
                df["label"] = df["id"].map(id_to_label).fillna(df["label"]).astype(int)

        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        image_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])
        sample_id = int(row["id"])
        return image, label, sample_id


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    data_dir = data_cfg.get("data_dir", "./data/FashionMNIST-Resplit")
    parquet_path = os.path.join(data_dir, "data.parquet")
    train_csv_path = os.path.join(data_dir, "train.csv")
    test_csv_path = os.path.join(data_dir, "test.csv")

    image_size = int(data_cfg.get("image_size", 28))
    augment = bool(data_cfg.get("augment", False))

    train_ds = FashionMNISTResplitDataset(
        parquet_path=parquet_path,
        split="train",
        transform=build_transforms(train=True, image_size=image_size, augment=augment),
        split_csv_path=train_csv_path,
    )
    test_ds = FashionMNISTResplitDataset(
        parquet_path=parquet_path,
        split="test",
        transform=build_transforms(train=False, image_size=image_size, augment=False),
        split_csv_path=test_csv_path,
    )

    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 128)),
        "num_workers": int(train_cfg.get("num_workers", 2)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
