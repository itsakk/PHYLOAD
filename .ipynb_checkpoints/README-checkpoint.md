# phyload

`phyload` is a modular loader for spatio-temporal physics datasets. It targets
PDE-style data stored as dense tensors with shape `(B, C, X, Y, T)` and provides
utilities to instantiate lazy HDF5-backed datasets alongside ergonomic PyTorch
`DataLoader` factories.

## Installation

```bash
pip install -e .
```

## Usage

```python
from pathlib import Path
from phyload import init_datasets, init_dataloaders

root = Path("/Users/Armand/code/distant/scratch_jz/data/vorticity_1024")

datasets = init_datasets(
    root,
    data_key="states",
    group_pattern="{split}",  # each HDF5 file stores data under a split-named group
)
train_loader, val_loader, test_loader = init_dataloaders(
    train_dataset=datasets.get("train"),
    val_dataset=datasets.get("val"),
    test_dataset=datasets.get("test"),
    batch_size=4,
    num_workers=4,
)

batch = next(iter(train_loader))
print(batch[0].shape)
```

## Development

- Python 3.10+
- Format with `ruff` or `black` (not yet configured)
- Run tests with `pytest` (suite TBD)
