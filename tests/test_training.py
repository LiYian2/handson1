import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.trainer import Trainer


def test_trainer_single_epoch_runs():
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    ids = torch.arange(32)

    class Wrap(torch.utils.data.Dataset):
        def __init__(self, x, y, ids):
            self.x = x
            self.y = y
            self.ids = ids

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.x[i], self.y[i], self.ids[i]

    ds = Wrap(x, y, ids)
    loader = DataLoader(ds, batch_size=8)

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=torch.device("cpu"),
            output_dir=tmp,
            save_interval=1,
        )
        history = trainer.train(num_epochs=1)

    assert len(history.train_loss) == 1
    assert len(history.val_acc) == 1
