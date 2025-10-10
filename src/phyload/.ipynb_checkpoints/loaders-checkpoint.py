"""Helper utilities to create PyTorch data loaders."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Tuple, Union

from torch.utils.data import DataLoader, Dataset

__all__ = ["init_dataloaders"]


def init_dataloaders(
    config: Optional[Union[Mapping[str, object], None]] = None,
    **overrides,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create train/validation/test data loaders using shared configuration.

    ``config`` may be a mapping (e.g. parsed YAML). Keyword arguments override
    values supplied via ``config``.
    """

    if config is None:
        cfg = dict(overrides)
    elif isinstance(config, Mapping):
        cfg = dict(config)
        cfg.update(overrides)
    else:
        raise TypeError("config must be a mapping or None.")

    train_dataset = cfg.pop("train_dataset", None)
    val_dataset = cfg.pop("val_dataset", None)
    test_dataset = cfg.pop("test_dataset", None)

    batch_size = int(cfg.pop("batch_size", 1))
    val_batch_size = int(cfg.pop("val_batch_size", batch_size) or batch_size)
    test_batch_size = int(cfg.pop("test_batch_size", batch_size) or batch_size)

    shuffle_train = bool(cfg.pop("shuffle_train", True))
    shuffle_val = bool(cfg.pop("shuffle_val", False))
    shuffle_test = bool(cfg.pop("shuffle_test", False))

    num_workers = int(cfg.pop("num_workers", 0))
    pin_memory = bool(cfg.pop("pin_memory", False))
    drop_last = bool(cfg.pop("drop_last", False))
    persistent_workers = bool(cfg.pop("persistent_workers", False))

    loader_kwargs = {k: v for k, v in cfg.items() if v is not None}

    def _build(dataset: Optional[Dataset], *, batch: int, shuffle: bool) -> Optional[DataLoader]:
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            **loader_kwargs,
        )

    train_loader = _build(train_dataset, batch=batch_size, shuffle=shuffle_train)
    val_loader = _build(val_dataset, batch=val_batch_size, shuffle=shuffle_val)
    test_loader = _build(test_dataset, batch=test_batch_size, shuffle=shuffle_test)
    return train_loader, val_loader, test_loader
