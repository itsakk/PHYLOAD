"""Helper utilities to create PyTorch data loaders."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

try:  # optional import for distributed training
    from torch.utils.data.distributed import DistributedSampler
except Exception:  # pragma: no cover
    DistributedSampler = None  # type: ignore

from .multi import MultiDatasetCollection, CombinedLoader, COMBINED_LOADERS

__all__ = ["init_dataloaders"]


def init_dataloaders(
    datasets: Mapping[str, Dataset],
    config: Optional[Mapping[str, object]] = None,
    **overrides,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create train/validation/test data loaders using shared configuration.

    Parameters
    ----------
    datasets:
        Mapping that must contain at least ``train`` (optional ``val``/``test``)
        entries returned by :func:`init_datasets`.
    config:
        Optional mapping (e.g. parsed YAML) providing dataloader hyper-parameters.
        Keyword arguments supplied directly to :func:`init_dataloaders` override
        those found inside ``config``.
    """

    if not isinstance(datasets, Mapping):
        raise TypeError("'datasets' must be provided as a mapping of splits to Dataset instances.")

    if config is None:
        raw_cfg = dict(overrides)
    elif isinstance(config, Mapping):
        raw_cfg = dict(config)
        raw_cfg.update(overrides)
    else:
        raise TypeError("config must be a mapping or None.")

    if isinstance(datasets, MultiDatasetCollection):
        if datasets.mode in COMBINED_LOADERS.keys():
            return _build_homogeneous_loaders(datasets, raw_cfg)
        datasets = {split: datasets[split] for split in datasets}

    cfg = dict(raw_cfg)

    def _from_mapping(name: str) -> Optional[Dataset]:
        # support both "val" and "validation"
        if name in datasets:
            return datasets[name]
        if name == "val" and "validation" in datasets:
            return datasets["validation"]
        if name == "validation" and "val" in datasets:
            return datasets["val"]
        return None

    train_dataset = cfg.pop("train_dataset", _from_mapping("train"))
    val_dataset = cfg.pop("val_dataset", _from_mapping("val"))
    test_dataset = cfg.pop("test_dataset", _from_mapping("test"))

    batch_size = int(cfg.pop("batch_size", 1))
    val_batch_size = int(cfg.pop("val_batch_size", batch_size) or batch_size)
    test_batch_size = int(cfg.pop("test_batch_size", batch_size) or batch_size)

    shuffle_train = bool(cfg.pop("shuffle_train", True))
    shuffle_val = bool(cfg.pop("shuffle_val", False))
    shuffle_test = bool(cfg.pop("shuffle_test", False))

    num_workers = int(cfg.pop("num_workers", 0))
    pin_memory = bool(cfg.pop("pin_memory", torch.cuda.is_available()))
    drop_last = bool(cfg.pop("drop_last", False))
    persistent_workers = bool(cfg.pop("persistent_workers", num_workers > 0))
    prefetch_factor = cfg.pop("prefetch_factor", 4 if num_workers > 0 else None)
    worker_init_fn = cfg.pop("worker_init_fn", None)
    collate_fn = cfg.pop("collate_fn", None)

    multi_gpu = bool(cfg.pop("multi_gpu", False))
    loader_kwargs = {k: v for k, v in cfg.items() if v is not None}

    def _build(dataset: Optional[Dataset], *, batch: int, shuffle: bool) -> Optional[DataLoader]:
        if dataset is None:
            return None
        sampler = None
        shuffle_flag = shuffle
        if multi_gpu:
            if DistributedSampler is None:
                raise RuntimeError("torch.utils.data.distributed.DistributedSampler is unavailable.")
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle_flag = False
        loader_args = dict(
            batch_size=batch,
            shuffle=shuffle_flag,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_args["prefetch_factor"] = int(prefetch_factor)
        if collate_fn is not None:
            loader_args["collate_fn"] = collate_fn
        elif hasattr(dataset, "collate_fn"):
            candidate = getattr(dataset, "collate_fn")
            if callable(candidate):
                loader_args["collate_fn"] = candidate
        if worker_init_fn is not None:
            loader_args["worker_init_fn"] = worker_init_fn
        loader_args.update(loader_kwargs)
        return DataLoader(dataset, **loader_args)

    train_loader = _build(train_dataset, batch=batch_size, shuffle=shuffle_train)
    val_loader = _build(val_dataset, batch=val_batch_size, shuffle=shuffle_val)
    test_loader = _build(test_dataset, batch=test_batch_size, shuffle=shuffle_test)
    return train_loader, val_loader, test_loader

def _build_homogeneous_loaders(
    collection: MultiDatasetCollection,
    config: Mapping[str, object],
) -> Tuple[Optional[CombinedLoader], Optional[CombinedLoader], Optional[CombinedLoader]]:
    config_mapping = dict(config)
    # Extract options that belong to the CombinedLoader itself so they don't leak into torch DataLoader kwargs.
    combined_loader_kwargs = {}
    for cfg_key, target_key in (
        ("sync_dataset_per_step", "sync_dataset_per_step"),
        ("dataset_choice_seed", "seed"),
        ("combined_seed", "seed"),
    ):
        if cfg_key in config_mapping:
            combined_loader_kwargs[target_key] = config_mapping.pop(cfg_key)

    alias_to_loaders: Dict[str, Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]] = {}
    for alias, split_map in collection.per_alias_datasets().items():
        loaders = init_dataloaders(split_map, config_mapping)
        alias_to_loaders[alias] = loaders

    def _gather(index: int, shuffle_flag: bool) -> Optional[CombinedLoader]:
        loaders = {
            alias: loader_tuple[index]
            for alias, loader_tuple in alias_to_loaders.items()
            if loader_tuple[index] is not None
        }
        if not loaders:
            return None
        return COMBINED_LOADERS.get(collection.mode, "uniform")(
            loaders,
            shuffle=shuffle_flag,
            **combined_loader_kwargs,
            label=["train", "val", "test"][index] if index < 3 else None,
        )

    train_loader = _gather(0, bool(config_mapping.get("shuffle_train", True)))
    val_loader = _gather(1, bool(config_mapping.get("shuffle_val", False)))
    test_loader = _gather(2, bool(config_mapping.get("shuffle_test", False)))
    return train_loader, val_loader, test_loader
