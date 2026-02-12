# ResNet on FashionMNIST-Resplit

This project reproduces the core ResNet idea on the FashionMNIST-Resplit dataset.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 experiments/run_experiment.py --config configs/experiments/exp1_baseline.yaml --dry-run
```

## Dataset

Expected files:
- `data/FashionMNIST-Resplit/data.parquet`
- `data/FashionMNIST-Resplit/train.csv`
- `data/FashionMNIST-Resplit/test.csv`

## Notes

- Local machine is used for code validation and dry-runs only.
- Full training should be run on a GPU server.
