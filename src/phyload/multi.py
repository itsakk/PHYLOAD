"""Utilities for combining multiple trajectory datasets."""
from __future__ import annotations

import copy
import itertools
from bisect import bisect_right
from collections import defaultdict
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple, Optional

from torch.utils.data.distributed import DistributedSampler

import torch
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
        if mode not in ["mixed"] + list(COMBINED_LOADERS.keys()):
            valid = "', '".join(["mixed"] + list(COMBINED_LOADERS.keys()))
            raise ValueError(f"mode must be one of '{valid}', not '{mode}'")
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


class CombinedLoader:
    def __init__(self, loaders: Mapping[str, torch.utils.data.DataLoader], shuffle: bool = True, seed: int = 42) -> None:
        self.loaders = {alias: loader for alias, loader in loaders.items() if loader is not None}
        self.lengths = {alias: len(loader) for alias, loader in self.loaders.items()}
        self.shuffle = bool(shuffle)
        self.seed = seed
        for loader in self.loaders.values():
            loader.shuffle = self.shuffle
        if not self.loaders:
            raise ValueError("At least one loader is required for CombinedLoader.")
        self.epoch = 0

    # def set_epoch(self, epoch: int) -> None:
    #     for loader in self.loaders.values():
    #         sampler = getattr(loader, "sampler", None)
    #         if sampler is not None and hasattr(sampler, "set_epoch"):
    #             sampler.set_epoch(epoch)
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        for loader in self.loaders.values():
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)
            # Also handle samplers wrapped in a BatchSampler
            elif hasattr(loader.sampler, "sampler") and isinstance(loader.sampler.sampler, DistributedSampler):
                loader.sampler.sampler.set_epoch(epoch)


class HomogeneousCombinedLoader(CombinedLoader):
    """
    Combines multiple dataloaders with homogeneous random sampling.

    Epoch definition (controlled by `mode`):
      - "fixed":   epoch = `steps_per_epoch` steps (you set this).
      - "largest": epoch = len(largest dataloader) steps; smaller ones cycle.

    DDP safety: the dataloader-selection RNG is seeded deterministically
    from (epoch, step), so all ranks make the same choice every step.
    Each sub-dataloader's DistributedSampler still shards data per rank.
    """

    def __init__(
        self,
        loaders: Mapping[str, torch.utils.data.DataLoader],
        steps_per_epoch: Optional[int] = None,  # required if mode="fixed"
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__(loaders, shuffle, seed)
        if not self.shuffle:
            self.steps_per_epoch = sum(self.lengths.values())
        else:
            self.steps_per_epoch = steps_per_epoch or max(self.lengths.values()) * len(self.lengths)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _cycling_iter(self, alias: str) -> Iterator:
        """Cycles over a dataloader by re-creating the iterator, avoiding memory accumulation."""
        while True:
            for batch in self.loaders[alias]:
                yield annotate_batch(batch, alias)

    def __iter__(self) -> Iterator:
        if not self.shuffle:
            for alias, loader in self.loaders.items():
                for batch in _iter_loader_batches(loader):
                    yield annotate_batch(batch, alias)
            return
        
        iterators = {alias: self._cycling_iter(alias) for alias in self.loaders}
        aliases = list(self.loaders.keys())
        n = len(aliases)

        for step in range(self.steps_per_epoch):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 100_000 + step)
            idx = torch.randint(n, (1,), generator=g).item()
            alias = aliases[idx]
            yield annotate_batch(next(iterators[alias]), alias)


class UniformCombinedLoader(CombinedLoader):
    """loads each chunk from each dataset uniformly through a weighted probability depending on the number of chunks in each dataset

    Args:
        loaders (Mapping[str, torch.utils.data.DataLoader]): list of the dataloaders to combine
        shuffle (bool, optional): if the samples should be shuffled (is irrelevant, doesn't have a deterministic behaviour implemented yet). Defaults to True.
    """

    def __init__(self, loaders: Mapping[str, torch.utils.data.DataLoader], shuffle: bool = True, seed: int = 42) -> None:
        super().__init__(loaders, shuffle, seed)
        self.lengths = {alias: len(loader) for alias, loader in self.loaders.items()}

    def __len__(self):
        return sum(self.lengths.values())

    def __iter__(self):
        if not self.shuffle:
            for alias, loader in self.loaders.items():
                for batch in loader:
                    yield annotate_batch(batch, alias)
            return
        
        entries = {alias: iter(loader) for alias, loader in self.loaders.items()}
        remaining = self.lengths.copy()

        step = 0
        while remaining:
            # Sample loader alias proportionally to remaining batch counts
            aliases = list(remaining.keys())
            weights = [remaining[a] for a in aliases]
            weights_tensor = torch.tensor(weights, dtype=torch.float)
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 100_000 + step)
            idx = torch.multinomial(weights_tensor, num_samples=1, generator=g).item()
            alias = aliases[idx]

            try:
                batch = next(entries[alias])
            except StopIteration:
                # shouldn't normally happen unless DataLoader behaves oddly
                del remaining[alias]
                continue

            # Decrease remaining count
            remaining[alias] -= 1
            if remaining[alias] == 0:
                del remaining[alias]
            step += 1

            yield annotate_batch(batch, alias)


def annotate_batch(batch, alias: str):
    if isinstance(batch, Mapping):
        annotated = dict(batch)
        annotated.setdefault("dataset", alias)
        return annotated
    return batch, alias


def _iter_loader_batches(loader: torch.utils.data.DataLoader) -> Iterator:
    """Iterate a loader and explicitly stop its workers when iteration ends."""
    loader_iterator = iter(loader)
    try:
        yield from loader_iterator
    finally:
        shutdown_workers = getattr(loader_iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        del loader_iterator


COMBINED_LOADERS = dict(
    homogeneous=HomogeneousCombinedLoader,
    uniform=UniformCombinedLoader,
)