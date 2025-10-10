"""Utilities for combining multiple trajectory datasets."""
from __future__ import annotations

import copy
import itertools
import random
from bisect import bisect_right
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = [
    "ConcatenatedTrajectoryDataset",
    "PaddedTrajectoryDataset",
    "MultiDatasetCollection",
    "HomogeneousCombinedLoader",
    "annotate_batch",
]


class ConcatenatedTrajectoryDataset(Dataset):
    """Concatenate multiple datasets while tracking their aliases."""

    def __init__(self, datasets: Mapping[str, Dataset]):
        if not datasets:
            raise ValueError("'datasets' must contain at least one entry.")
        self._aliases: Sequence[str] = list(datasets)
        self._datasets: Sequence[Dataset] = [datasets[alias] for alias in self._aliases]
        lengths = [len(dataset) for dataset in self._datasets]
        if any(length == 0 for length in lengths):
            raise ValueError("All component datasets must contain at least one sample.")
        self._cumulative_sizes = list(itertools.accumulate(lengths))

    @property
    def aliases(self) -> Sequence[str]:
        return self._aliases

    @property
    def cumulative_sizes(self) -> Sequence[int]:
        return self._cumulative_sizes

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def _locate(self, index: int) -> tuple[str, Dataset, int]:
        if index < 0:
            if -index > len(self):
                raise IndexError("index out of range")
            index = len(self) + index
        dataset_idx = bisect_right(self._cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self._cumulative_sizes[dataset_idx - 1]
        alias = self._aliases[dataset_idx]
        dataset = self._datasets[dataset_idx]
        return alias, dataset, sample_idx

    def __getitem__(self, index: int):
        alias, dataset, local_index = self._locate(index)
        sample = dataset[local_index]
        if isinstance(sample, Mapping):
            sample = dict(sample)
            sample.setdefault("dataset", alias)
            return sample
        return sample, alias

    def per_alias_datasets(self) -> Dict[str, Dataset]:
        return {alias: dataset for alias, dataset in zip(self._aliases, self._datasets)}


class PaddedTrajectoryDataset(Dataset):
    """Dataset that pads tensor fields to common shapes across datasets."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        pad_value: float = 0.0,
    ) -> None:
        self._base = ConcatenatedTrajectoryDataset(datasets)
        self.pad_value = float(pad_value)
        self.max_specs = self._compute_key_specs(datasets)
        self._templates = {key: torch.zeros(shape, dtype=dtype) for key, (shape, dtype) in self.max_specs.items()}
        self._mask_templates = {key: torch.zeros(shape, dtype=torch.bool) for key, (shape, _) in self.max_specs.items()}
        self._defaults = self._compute_default_values(datasets)

    @staticmethod
    def _compute_key_specs(
        datasets: Mapping[str, Dataset]
    ) -> Dict[str, tuple[Tuple[int, ...], torch.dtype]]:
        shapes: Dict[str, Tuple[int, ...]] = {}
        dims: Dict[str, int] = {}
        dtypes: Dict[str, torch.dtype] = {}
        for dataset in datasets.values():
            sample = dataset[0]
            for key, value in sample.items():
                if not isinstance(value, torch.Tensor):
                    continue
                shape = tuple(int(dim) for dim in value.shape)
                if key not in shapes:
                    shapes[key] = shape
                    dims[key] = value.dim()
                    dtypes[key] = value.dtype
                else:
                    if dims[key] != value.dim() or len(shapes[key]) != len(shape):
                        shapes.pop(key, None)
                        dtypes.pop(key, None)
                        dims[key] = -1
                        continue
                    shapes[key] = tuple(max(a, b) for a, b in zip(shapes[key], shape))
                    if value.dtype.is_floating_point and not dtypes[key].is_floating_point:
                        dtypes[key] = value.dtype
        return {
            key: (shape, dtypes[key])
            for key, shape in shapes.items()
            if dims.get(key, -1) >= 0
        }

    @staticmethod
    def _compute_default_values(datasets: Mapping[str, Dataset]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for dataset in datasets.values():
            sample = dataset[0]
            for key, value in sample.items():
                if key == "dataset":
                    continue
                if isinstance(value, torch.Tensor):
                    continue
                if isinstance(value, Mapping):
                    current = defaults.get(key)
                    if current is None:
                        defaults[key] = _merge_mappings({}, value)
                    else:
                        defaults[key] = _merge_mappings(current, value)
                else:
                    defaults.setdefault(key, copy.deepcopy(value))
        return defaults

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int):
        sample = self._base[index]
        for key, (target_shape, dtype) in self.max_specs.items():
            tensor = sample.get(key)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                tensor = self._templates[key].clone()
            padded, mask = _pad_tensor(tensor, target_shape, pad_value=self.pad_value)
            sample[key] = padded
            mask_key = f"{key}_padding_mask"
            sample[mask_key] = mask
            if key == "trajectory":
                sample["original_shape"] = torch.tensor(tensor.shape, dtype=torch.int64)
                sample["padding_mask"] = mask
        for key, default_value in self._defaults.items():
            if key not in sample:
                sample[key] = _clone_value(default_value)
            elif isinstance(default_value, Mapping) and isinstance(sample[key], Mapping):
                sample[key] = _merge_mappings(default_value, sample[key])
        for key in self._templates:
            sample.setdefault(key, self._templates[key].clone())
            sample.setdefault(f"{key}_padding_mask", self._mask_templates[key].clone())
        return sample

    @property
    def aliases(self) -> Sequence[str]:
        return self._base.aliases


def _pad_tensor(tensor: torch.Tensor, target_shape: Sequence[int], pad_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    target_shape = tuple(int(dim) for dim in target_shape)
    if tensor.shape == target_shape:
        mask = torch.ones_like(tensor, dtype=torch.bool)
        return tensor, mask
    if tensor.dim() != len(target_shape):
        raise ValueError("tensor dimension mismatch for padding")

    if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        fill_value = pad_value
    else:
        fill_value = 0
    padded_tensor = tensor.new_full(target_shape, fill_value)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded_tensor[slices] = tensor

    mask = torch.zeros(target_shape, dtype=torch.bool, device=tensor.device)
    mask[slices] = True
    return padded_tensor, mask


def _clone_value(value: Any):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def _merge_mappings(default_map: Mapping[str, Any], sample_map: Mapping[str, Any]):
    merged: Dict[str, Any] = {key: _clone_value(value) for key, value in default_map.items()}
    for key, value in sample_map.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = _clone_value(value)
    return merged


class MultiDatasetCollection(Mapping[str, Dataset]):
    """Container of combined datasets for each split."""

    def __init__(
        self,
        mode: str,
        per_split_alias_map: Mapping[str, Mapping[str, Dataset]],
        pad_value: float = 0.0,
    ) -> None:
        mode = mode.lower()
        if mode not in {"homogeneous", "mixed"}:
            raise ValueError("mode must be 'homogeneous' or 'mixed'")
        self.mode = mode
        self.pad_value = float(pad_value)
        self._alias_map = {
            split: dict(alias_map)
            for split, alias_map in per_split_alias_map.items()
        }
        self._datasets: Dict[str, Dataset] = {}
        for split, alias_map in self._alias_map.items():
            if mode == "mixed":
                self._datasets[split] = PaddedTrajectoryDataset(alias_map, pad_value=self.pad_value)
            else:
                self._datasets[split] = ConcatenatedTrajectoryDataset(alias_map)

    def __getitem__(self, key: str) -> Dataset:
        return self._datasets[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    @property
    def alias_map(self) -> Mapping[str, Mapping[str, Dataset]]:
        return self._alias_map

    def per_alias_datasets(self) -> Dict[str, Dict[str, Dataset]]:
        grouped: Dict[str, Dict[str, Dataset]] = defaultdict(dict)
        for split, alias_map in self._alias_map.items():
            for alias, dataset in alias_map.items():
                grouped[alias][split] = dataset
        return grouped


class HomogeneousCombinedLoader:
    """Round-robin loader combining batches from multiple loaders."""

    def __init__(self, loaders: Mapping[str, torch.utils.data.DataLoader], shuffle: bool = True) -> None:
        self.loaders = {alias: loader for alias, loader in loaders.items() if loader is not None}
        self.shuffle = bool(shuffle)
        if not self.loaders:
            raise ValueError("At least one loader is required for HomogeneousCombinedLoader.")

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders.values())

    def set_epoch(self, epoch: int) -> None:
        for loader in self.loaders.values():
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __iter__(self):
        entries = [(alias, iter(loader)) for alias, loader in self.loaders.items()]
        if self.shuffle:
            random.shuffle(entries)
        active = deque(entries)
        while active:
            alias, iterator = active.popleft()
            try:
                batch = next(iterator)
            except StopIteration:
                continue
            yield annotate_batch(batch, alias)
            active.append((alias, iterator))


def annotate_batch(batch, alias: str):
    if isinstance(batch, Mapping):
        annotated = dict(batch)
        annotated.setdefault("dataset", alias)
        return annotated
    return batch, alias
