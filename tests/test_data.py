import os

from data.dataset import FashionMNISTResplitDataset
from data.transforms import build_transforms


def test_dataset_can_read_one_sample():
    ds = FashionMNISTResplitDataset(
        parquet_path="data/FashionMNIST-Resplit/data.parquet",
        split="train",
        transform=build_transforms(train=False),
        split_csv_path="data/FashionMNIST-Resplit/train.csv",
    )
    x, y, sid = ds[0]
    assert x.shape[0] == 1
    assert x.shape[1] == 28
    assert x.shape[2] == 28
    assert 0 <= y <= 9
    assert sid >= 0
