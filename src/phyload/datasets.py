"""Dataset utilities for structured spatio-temporal trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List, MutableSequence, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .multi import MultiDatasetCollection

__all__ = [
    "FieldInfo",
    "TrajectoryDataset",
    "init_datasets",
    "MultiDatasetCollection",
]


@dataclass(frozen=True)
class FieldInfo:
    """Description of a tensor field stored in an HDF5 dataset."""

    path: str
    name: str
    components: int
    spatial_shape: Tuple[int, ...]
    has_time: bool
    dtype: np.dtype


class TrajectoryDataset(Dataset):
    """PyTorch dataset for spatio-temporal data exported in HDF5 collections."""

    SPLIT_DIR = {
        "train": "train",
        "val": "valid",
        "valid": "valid",
        "validation": "valid",
        "test": "test",
    }

    FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields", "fields")

    def __init__(
        self,
        root: Path | str,
        *,
        split: str = "train",
        time_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        time_stride: int = 1,
        dt: Optional[Union[int, float]] = None,
        dx: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
        num_steps: Optional[int] = None,
        window_stride: Optional[int] = None,
        max_trajectories: Optional[int] = None,
        use_normalization: bool = False,
        normalization_type: str = "zscore",
        normalize: Optional[bool] = None,
        return_ic: bool = False,
        return_bc: bool = False,
        return_params: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")

        split_key = split.lower()
        if split_key not in self.SPLIT_DIR:
            valid = ", ".join(sorted(self.SPLIT_DIR))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {valid}.")
        data_dir = self.root / "data" / self.SPLIT_DIR[split_key]
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory '{data_dir}' is missing.")

        self.files: List[Path] = sorted(data_dir.glob("*.hdf5"))
        if not self.files:
            raise FileNotFoundError(f"No HDF5 files found under '{data_dir}'.")

        self.dtype = dtype
        self.return_ic = bool(return_ic)
        self.return_bc = bool(return_bc)
        self.return_params = bool(return_params)
        if normalize is not None:
            self._use_normalization = bool(normalize)
        else:
            self._use_normalization = bool(use_normalization)
        self._normalization_type = normalization_type.lower()
        if self._normalization_type not in {"zscore", "rms"}:
            raise ValueError("normalization_type must be 'zscore' or 'rms'.")

        if max_trajectories is not None:
            max_trajectories = int(max_trajectories)
            if max_trajectories <= 0:
                raise ValueError("max_trajectories must be a positive integer or None.")
        self.max_trajectories = max_trajectories

        self.time_range = time_range
        self.temporal_stride = max(1, int(time_stride)) * self._factor_to_stride(dt)

        representative = self.files[0]
        (
            self.n_spatial_dims,
            self.field_infos,
            space_axes,
        ) = self._inspect_file(representative)

        self._channel_names: List[str] = []
        self._channel_lookup: Dict[str, int] = {}
        for info in self.field_infos:
            base = info.path.replace('/', '.')
            if info.components == 1:
                name = base
                self._channel_lookup[name] = len(self._channel_names)
                self._channel_names.append(name)
            else:
                for idx in range(info.components):
                    name = f"{base}[{idx}]"
                    self._channel_lookup[name] = len(self._channel_names)
                    self._channel_names.append(name)

        self._components_per_field: List[int] = [info.components for info in self.field_infos]
        self._total_channels = int(sum(self._components_per_field))

        self._spatial_strides = self._resolve_spatial_strides(dx, self.n_spatial_dims)
        self._space_axes = [axis[::stride] for axis, stride in zip(space_axes, self._spatial_strides)]
        self.space_grid = self._build_space_grid(self._space_axes).to(self.dtype)
        self._space_grid_template = self.space_grid.unsqueeze(-1)
        self._spatial_slices = tuple(slice(None, None, stride) for stride in self._spatial_strides)
        self._spatial_shape = tuple(len(axis) for axis in self._space_axes)
        self._normalization_eps = 1e-4
        self._stats: Dict[str, Dict[str, np.ndarray]] = {}
        self._channel_stats: Dict[str, List[Optional[float]]] = {
            "mean": [],
            "std": [],
            "rms": [],
        }
        self._channel_norm_tensors: Dict[str, torch.Tensor] = {}
        self._channel_norm_masks: Dict[str, torch.Tensor] = {}
        self._temporal_field: Optional[FieldInfo] = next(
            (info for info in self.field_infos if info.has_time), None
        )

        if self._use_normalization:
            stats_path = self.root / "stats.yaml"
            self._stats = self._load_stats_file(stats_path)
            required = {"mean", "std"} if self._normalization_type == "zscore" else {"rms"}
            missing = [key for key in required if key not in self._stats]
            if missing:
                raise KeyError(
                    f"Normalization type '{self._normalization_type}' requires stats keys: {missing}."
                )
            self._precompute_channel_stats()
            self._prepare_channel_norm_tensors()

        self.num_steps = num_steps if num_steps is None else int(num_steps)
        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer or None.")
        if window_stride is None:
            self.window_stride = self.num_steps if self.num_steps is not None else 1
        else:
            if window_stride <= 0:
                raise ValueError("window_stride must be a positive integer.")
            self.window_stride = int(window_stride)

        self.index: MutableSequence[Tuple[int, int, int]] = []
        self._handle_cache: Dict[int, h5py.File] = {}
        self._trajectory_counts: Dict[int, int] = {}
        self._time_axes: Dict[int, np.ndarray] = {}
        self._time_indices: Dict[int, np.ndarray] = {}

        for file_idx, file_path in enumerate(self.files):
            with h5py.File(file_path, "r") as handle:
                primary = self.field_infos[0]
                data = handle[primary.path]
                n_traj = data.shape[0]
                selected_traj = (
                    min(n_traj, self.max_trajectories)
                    if self.max_trajectories is not None
                    else n_traj
                )
                time_axis = self._load_time_axis(handle, self.field_infos)
                time_len = len(time_axis)
                if time_len <= 0:
                    raise ValueError(f"File '{file_path}' does not contain temporal samples.")

                time_indices = self._compute_time_indices(time_len)
                if time_indices.size == 0:
                    raise ValueError(
                        "The combination of time_range and temporal stride removed all frames."
                    )

                self._trajectory_counts[file_idx] = selected_traj
                self._time_axes[file_idx] = time_axis
                self._time_indices[file_idx] = time_indices

                effective_len = len(time_indices)
                if self.num_steps is not None and effective_len < self.num_steps:
                    raise ValueError(
                        f"Requested num_steps={self.num_steps} exceeds available steps ({effective_len})."
                    )

                if self.num_steps is None:
                    for traj_idx in range(selected_traj):
                        self.index.append((file_idx, traj_idx, 0))
                else:
                    stride = self.window_stride
                    limit = effective_len - self.num_steps
                    for traj_idx in range(selected_traj):
                        for start in range(0, limit + 1, stride):
                            self.index.append((file_idx, traj_idx, start))

        if not self.index:
            raise RuntimeError(f"Dataset '{self.root}' produced an empty index.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        file_id, traj_idx, window_start = self.index[idx]
        handle = self._get_handle(file_id)

        time_indices_full = self._time_indices[file_id]
        if self.num_steps is None:
            selected_indices = time_indices_full
        else:
            selected_indices = time_indices_full[window_start : window_start + self.num_steps]
        selected_time = self._time_axes[file_id][selected_indices]
        num_frames = len(selected_indices)

        time_selector = self._build_time_selector(selected_indices)
        trajectory_tensor = torch.empty(
            (num_frames, *self._spatial_shape, self._total_channels),
            dtype=self.dtype,
        )
        channel_offset = 0
        for info, components in zip(self.field_infos, self._components_per_field):
            dataset = handle[info.path]
            raw = self._read_field_block(dataset, info, traj_idx, time_selector, num_frames)
            field_values = raw.reshape(num_frames, *self._spatial_shape, components)
            field_values = np.ascontiguousarray(field_values, dtype=np.float32)
            src = torch.from_numpy(field_values)
            if src.dtype != self.dtype:
                src = src.to(self.dtype)
            trajectory_tensor[..., channel_offset : channel_offset + components].copy_(src)
            channel_offset += components

        sample: Dict[str, object] = {
            "index": {
                "file": self.files[file_id].name,
                "trajectory": traj_idx,
                "window_start": int(window_start),
            },
            "trajectory": trajectory_tensor,
        }

        if self.return_ic:
            sample["ic"] = trajectory_tensor[0].movedim(-1, 0).clone()

        if self.space_grid.numel() == 0:
            sample["space_grid"] = self.space_grid
        else:
            space_grid = self._space_grid_template.expand(*self.space_grid.shape, num_frames)
            space_grid = space_grid.movedim(-1, 0).to(self.dtype)
            sample["space_grid"] = space_grid

        time_grid = torch.from_numpy(selected_time.astype(np.float32)).to(self.dtype)
        sample["time_grid"] = time_grid

        if self.return_params:
            params = self._collect_parameters(handle, file_id, traj_idx)
            sample["params"] = params

        if self.return_bc:
            bc = self._collect_boundary_conditions(handle, file_id, traj_idx)
            sample["bc"] = bc

        return sample

    def collate_fn(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        collated = default_collate(batch)
        if not self._use_normalization:
            return collated

        trajectories = collated.get("trajectory")
        if isinstance(trajectories, torch.Tensor):
            ndim = trajectories.ndim
            view_shape = (1,) * (ndim - 1) + (trajectories.shape[-1],)
            if self._normalization_type == "zscore":
                mean = self._channel_norm_tensors["mean"].to(
                    device=trajectories.device,
                    dtype=trajectories.dtype,
                    copy=False,
                )
                std = self._channel_norm_tensors["std"].to(
                    device=trajectories.device,
                    dtype=trajectories.dtype,
                    copy=False,
                )
                trajectories.sub_(mean.view(view_shape))
                trajectories.div_(std.view(view_shape))
            else:
                rms = self._channel_norm_tensors["rms"].to(
                    device=trajectories.device,
                    dtype=trajectories.dtype,
                    copy=False,
                )
                trajectories.div_(rms.view(view_shape))
        return collated

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def _inspect_file(
        self,
        path: Path,
    ) -> Tuple[int, List[FieldInfo], List[np.ndarray]]:
        with h5py.File(path, "r") as handle:
            n_spatial_dims = int(handle.attrs.get("n_spatial_dims", 2))
            field_infos: List[FieldInfo] = []
            for group in self.FIELD_GROUPS:
                if group not in handle:
                    continue
                for name, dataset in handle[group].items():
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    info = self._build_field_info(group, name, dataset, n_spatial_dims)
                    field_infos.append(info)
            if not field_infos:
                raise ValueError(f"No tensor fields discovered in '{path}'.")

            space_axes = self._load_space_axes(handle, field_infos[0])
        return n_spatial_dims, field_infos, space_axes

    @staticmethod
    def _build_field_info(
        group: str,
        name: str,
        dataset: h5py.Dataset,
        n_spatial_dims: int,
    ) -> FieldInfo:
        shape = dataset.shape
        if len(shape) < 1 + n_spatial_dims:
            raise ValueError(f"Dataset '{group}/{name}' has incompatible shape {shape}.")
        has_time = len(shape) >= 2 + n_spatial_dims
        spatial_start = 1 if not has_time else 2
        spatial_shape = tuple(int(dim) for dim in shape[spatial_start : spatial_start + n_spatial_dims])
        remainder = shape[spatial_start + n_spatial_dims :]
        components = int(np.prod(remainder)) if remainder else 1
        return FieldInfo(
            path=f"{group}/{name}",
            name=name,
            components=components,
            spatial_shape=spatial_shape,
            has_time=has_time,
            dtype=dataset.dtype,
        )

    @staticmethod
    def _load_space_axes(handle: h5py.File, reference: FieldInfo) -> List[np.ndarray]:
        n_spatial_dims = len(reference.spatial_shape)
        axes: List[np.ndarray] = []
        if "dimensions" in handle:
            spatial_keys = [key for key in handle["dimensions"].keys() if key != "time"]
            for key in sorted(spatial_keys):
                axes.append(np.asarray(handle["dimensions"][key][()], dtype=np.float32))
            if len(axes) >= n_spatial_dims:
                return axes[:n_spatial_dims]
        return [np.arange(size, dtype=np.float32) for size in reference.spatial_shape]

    def _load_time_axis(
        self,
        handle: h5py.File,
        field_infos: Sequence[FieldInfo],
    ) -> np.ndarray:
        primary = next((info for info in field_infos if info.has_time), None)
        if "dimensions" in handle and "time" in handle["dimensions"]:
            return np.asarray(handle["dimensions"]["time"][()], dtype=np.float32)
        if primary is None:
            return np.zeros(1, dtype=np.float32)
        dataset = handle[primary.path]
        return np.arange(dataset.shape[1], dtype=np.float32)

    # ------------------------------------------------------------------
    # Data transforms
    # ------------------------------------------------------------------
    def _build_time_selector(self, indices: np.ndarray) -> Union[slice, List[int]]:
        if indices.size == 0:
            raise ValueError("Selected time indices are empty.")
        if indices.size == 1:
            start = int(indices[0])
            return slice(start, start + 1, 1)
        step = int(indices[1]) - int(indices[0])
        if step <= 0 or np.any(np.diff(indices) != step):
            return [int(idx) for idx in indices.tolist()]
        stop = int(indices[-1]) + step
        return slice(int(indices[0]), stop, step)

    def _read_field_block(
        self,
        dataset: h5py.Dataset,
        info: FieldInfo,
        traj_idx: int,
        time_selector: Union[slice, List[int]],
        num_frames: int,
    ) -> np.ndarray:
        selection: List[Union[int, slice, List[int]]] = [traj_idx]
        if info.has_time:
            selection.append(time_selector)
        spatial_slices = self._spatial_slices if self._spatial_slices else ()
        selection.extend(spatial_slices)
        remainder = dataset.ndim - len(selection)
        if remainder > 0:
            selection.extend([slice(None)] * remainder)
        array = dataset[tuple(selection)]
        arr_np = np.asarray(array)
        if arr_np.dtype != np.float32:
            arr_np = arr_np.astype(np.float32, copy=False)
        if not info.has_time:
            arr_np = np.repeat(arr_np[np.newaxis, ...], num_frames, axis=0)
        return np.ascontiguousarray(arr_np)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _collect_parameters(self, handle: h5py.File, file_id: int, traj_idx: int) -> Dict[str, object]:
        if "scalars" not in handle:
            return {}
        params: Dict[str, object] = {}
        n_traj = self._trajectory_counts.get(file_id)
        for name, dataset in handle["scalars"].items():
            if not isinstance(dataset, h5py.Dataset):
                continue
            if dataset.shape and n_traj is not None and dataset.shape[0] == n_traj:
                value = dataset[traj_idx]
            else:
                value = dataset[()]
            arr = np.asarray(value)
            if arr.dtype.kind in {"S", "U"}:
                params[name] = self._format_param_value(arr)
                continue
            if arr.dtype.kind == "O":
                formatted = self._format_param_value(arr)
                tentative = np.array(formatted)
                if tentative.dtype.kind in {"S", "U"}:
                    params[name] = self._format_param_value(tentative)
                    continue
                if tentative.dtype.kind == "b":
                    params[name] = torch.as_tensor(tentative.astype(np.bool_))
                    continue
                if np.issubdtype(tentative.dtype, np.number):
                    arr = tentative.astype(np.float32)
                else:
                    params[name] = formatted
                    continue
            if np.issubdtype(arr.dtype, np.bool_):
                params[name] = torch.as_tensor(arr.astype(np.bool_))
            elif np.issubdtype(arr.dtype, np.number):
                if arr.ndim == 0:
                    params[name] = torch.as_tensor(arr.item(), dtype=self.dtype)
                else:
                    params[name] = torch.from_numpy(arr.astype(np.float32)).to(self.dtype)
            else:
                params[name] = self._format_param_value(arr)
        return params

    def _format_param_value(self, value: object) -> object:
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, list):
            return [self._format_param_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._format_param_value(item) for item in value)
        return value

    def _collect_boundary_conditions(self, handle: h5py.File, file_id: int, traj_idx: int) -> Dict[str, object]:
        if "boundary_conditions" not in handle:
            return {}
        bc: Dict[str, object] = {}
        n_traj = self._trajectory_counts.get(file_id)
        group = handle["boundary_conditions"]
        for key, node in group.items():
            if isinstance(node, h5py.Dataset):
                bc[key] = self._process_bc_dataset(node, n_traj, traj_idx)
            else:
                nested: Dict[str, object] = {}
                for subkey, dataset in node.items():
                    nested[subkey] = self._process_bc_dataset(dataset, n_traj, traj_idx)
                bc[key] = nested
        return bc

    def _process_bc_dataset(
        self,
        dataset: h5py.Dataset,
        n_traj: Optional[int],
        traj_idx: int,
    ) -> torch.Tensor:
        if dataset.shape and n_traj is not None and dataset.shape[0] == n_traj:
            value = dataset[traj_idx]
        else:
            value = dataset[()]
        arr = np.asarray(value)
        if arr.dtype.kind == "b":
            tensor = torch.from_numpy(arr.astype(np.bool_))
        elif arr.dtype.kind in {"i", "u"}:
            tensor = torch.from_numpy(arr.astype(np.int64))
        else:
            tensor = torch.from_numpy(arr.astype(np.float32)).to(self.dtype)
        return tensor

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _normalize_field_array(self, array: np.ndarray, info: FieldInfo) -> np.ndarray:
        try:
            if self._normalization_type == "zscore":
                mean = self._field_stat_array("mean", info, array)
                std = np.clip(self._field_stat_array("std", info, array), self._normalization_eps, None)
                return (array - mean) / std
            if self._normalization_type == "rms":
                rms = np.clip(self._field_stat_array("rms", info, array), self._normalization_eps, None)
                return array / rms
        except KeyError:
            return array
        return array

    def _field_stat_array(self, key: str, info: FieldInfo, array: np.ndarray) -> np.ndarray:
        group = self._stats.get(key)
        if group is None or info.name not in group:
            raise KeyError
        value = group[info.name]
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 0 and value.ndim > array.ndim:
            raise ValueError(f"Statistic for field '{info.name}' has unexpected shape {value.shape}.")
        reshape = (1,) * (array.ndim - value.ndim) + value.shape
        return value.reshape(reshape)

    def _get_flat_stat(self, key: str, info: FieldInfo) -> Optional[np.ndarray]:
        group = self._stats.get(key)
        if group is None:
            return None
        value = group.get(info.name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).flatten()
        return arr

    def _precompute_channel_stats(self) -> None:
        for stats_key in self._channel_stats:
            self._channel_stats[stats_key] = []

        for info in self.field_infos:
            mean_flat = self._get_flat_stat("mean", info)
            std_flat = self._get_flat_stat("std", info)
            rms_flat = self._get_flat_stat("rms", info)
            for comp_idx in range(info.components):
                self._channel_stats["mean"].append(
                    float(mean_flat[comp_idx]) if mean_flat is not None and comp_idx < len(mean_flat) else None
                )
                if std_flat is not None and comp_idx < len(std_flat):
                    self._channel_stats["std"].append(float(max(std_flat[comp_idx], self._normalization_eps)))
                else:
                    self._channel_stats["std"].append(None)
                if rms_flat is not None and comp_idx < len(rms_flat):
                    self._channel_stats["rms"].append(float(max(rms_flat[comp_idx], self._normalization_eps)))
                else:
                    self._channel_stats["rms"].append(None)

    def _prepare_channel_norm_tensors(self) -> None:
        def _as_tensor(values: List[Optional[float]], *, fallback: float) -> Tuple[torch.Tensor, torch.Tensor]:
            data: List[float] = []
            mask: List[bool] = []
            for value in values:
                if value is None:
                    data.append(fallback)
                    mask.append(False)
                else:
                    data.append(float(value))
                    mask.append(True)
            return torch.tensor(data, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

        mean_tensor, mean_mask = _as_tensor(self._channel_stats["mean"], fallback=0.0)
        std_tensor, std_mask = _as_tensor(self._channel_stats["std"], fallback=1.0)
        rms_tensor, rms_mask = _as_tensor(self._channel_stats["rms"], fallback=1.0)

        self._channel_norm_tensors["mean"] = mean_tensor
        self._channel_norm_masks["mean"] = mean_mask
        self._channel_norm_tensors["std"] = std_tensor
        self._channel_norm_masks["std"] = std_mask
        self._channel_norm_tensors["rms"] = rms_tensor
        self._channel_norm_masks["rms"] = rms_mask

    def denormalize_channel(self, channel_tensor: torch.Tensor, channel_name: str) -> torch.Tensor:
        if not self._use_normalization:
            return channel_tensor
        if channel_name not in self._channel_lookup:
            raise KeyError(f"Unknown channel '{channel_name}'.")
        idx = self._channel_lookup[channel_name]
        if self._normalization_type == "zscore":
            mean = self._channel_stats["mean"][idx]
            std = self._channel_stats["std"][idx]
            if mean is None or std is None:
                return channel_tensor
            mean_t = torch.as_tensor(mean, dtype=channel_tensor.dtype, device=channel_tensor.device)
            std_t = torch.as_tensor(std, dtype=channel_tensor.dtype, device=channel_tensor.device)
            return channel_tensor * std_t + mean_t
        if self._normalization_type == "rms":
            rms = self._channel_stats["rms"][idx]
            if rms is None:
                return channel_tensor
            rms_t = torch.as_tensor(rms, dtype=channel_tensor.dtype, device=channel_tensor.device)
            return channel_tensor * rms_t
        return channel_tensor

    def denormalize_trajectory(self, trajectory_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalise an entire trajectory tensor stacked channel-first."""

        if not self._use_normalization:
            return trajectory_tensor
        if trajectory_tensor.shape[-1] != len(self._channel_names):
            raise ValueError(
                f"Expected last dimension {len(self._channel_names)}, got {trajectory_tensor.shape[-1]}."
            )

        result = trajectory_tensor.clone()
        for idx, name in enumerate(self._channel_names):
            channel = result[..., idx]
            channel = self.denormalize_channel(channel, name)
            result[..., idx] = channel
        return result

    def _compute_time_indices(self, total_length: int) -> np.ndarray:
        indices = np.arange(total_length)
        if self.time_range is not None:
            start, stop = self.time_range
            start_idx = self._resolve_bound(start, total_length)
            stop_idx = self._resolve_bound(stop, total_length)
            indices = indices[start_idx:stop_idx]
        return indices[:: self.temporal_stride]

    @staticmethod
    def _resolve_bound(bound: Optional[int], length: int) -> Optional[int]:
        if bound is None:
            return None
        value = int(bound)
        if value < 0:
            value = length + value
        return max(0, min(length, value))

    @staticmethod
    def _factor_to_stride(factor: Optional[Union[int, float]]) -> int:
        if factor is None:
            return 1
        if isinstance(factor, int):
            return max(1, int(factor))
        value = float(factor)
        if value <= 0:
            raise ValueError("Subsampling factor must be positive.")
        if value >= 1:
            return int(round(value))
        stride = int(round(1.0 / value))
        return max(1, stride)

    @staticmethod
    def _resolve_spatial_strides(
        factor: Optional[Union[int, float, Sequence[Union[int, float]]]],
        n_dims: int,
    ) -> Tuple[int, ...]:
        if factor is None:
            return tuple(1 for _ in range(n_dims))
        if isinstance(factor, Sequence) and not isinstance(factor, (str, bytes)):
            factors = list(factor)
            if len(factors) != n_dims:
                raise ValueError(
                    f"Expected {n_dims} spatial subsampling factors, received {len(factors)}."
                )
            return tuple(TrajectoryDataset._factor_to_stride(f) for f in factors)
        stride = TrajectoryDataset._factor_to_stride(factor)
        return tuple(stride for _ in range(n_dims))

    @staticmethod
    def _build_space_grid(axes: Sequence[np.ndarray]) -> torch.Tensor:
        if not axes:
            return torch.empty(0)
        mesh = np.meshgrid(*axes, indexing="ij")
        stacked = np.stack(mesh, axis=0).astype(np.float32)
        return torch.from_numpy(stacked)

    def _get_handle(self, file_id: int) -> h5py.File:
        handle = self._handle_cache.get(file_id)
        if handle is None or not handle.id:
            path = self.files[file_id]
            handle = h5py.File(path, "r", swmr=True)
            self._handle_cache[file_id] = handle
        return handle

    def close(self) -> None:
        for handle in self._handle_cache.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handle_cache.clear()

    def __getstate__(self):  # pragma: no cover
        state = dict(self.__dict__)
        state["_handle_cache"] = {}
        return state

    def __setstate__(self, state):  # pragma: no cover
        self.__dict__.update(state)
        self._handle_cache = {}

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    def channel_index(self, name: str) -> int:
        return self._channel_lookup[name]

    @staticmethod
    def _infer_time_length(dataset: h5py.Dataset, info: FieldInfo) -> int:
        return dataset.shape[1] if info.has_time else 1

    @staticmethod
    def _load_stats_file(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Normalization requested but '{path}' does not exist."
            )
        payload = yaml.safe_load(path.read_text()) or {}
        stats: Dict[str, Dict[str, np.ndarray]] = {}
        for key, mapping in payload.items():
            if not isinstance(mapping, dict):
                continue
            stats[key] = {
                field: np.asarray(value, dtype=np.float32)
                for field, value in mapping.items()
            }
        return stats


def _init_multi_dataset_collection(cfg: Dict[str, object]) -> MultiDatasetCollection:
    dataset_entries = cfg.get("datasets")
    if not isinstance(dataset_entries, Mapping) or not dataset_entries:
        raise ValueError("'datasets' must be a non-empty mapping when combining datasets.")

    fusion_mode = cfg.get("fusion_mode", "homogeneous")
    pad_value = cfg.get("pad_value", 0.0)

    global_defaults = {
        key: value
        for key, value in cfg.items()
        if key not in {"datasets", "fusion_mode", "pad_value"}
    }

    per_split_alias: Dict[str, Dict[str, TrajectoryDataset]] = defaultdict(dict)
    for alias, alias_cfg in dataset_entries.items():
        if not isinstance(alias_cfg, Mapping):
            raise TypeError(f"Configuration for dataset '{alias}' must be a mapping.")
        merged: Dict[str, object] = dict(global_defaults)
        merged.update(dict(alias_cfg))
        alias_datasets = init_datasets(merged)
        for split, dataset in alias_datasets.items():
            per_split_alias[split][alias] = dataset

    if not per_split_alias:
        raise FileNotFoundError("No datasets could be initialised from the multi-dataset configuration.")

    return MultiDatasetCollection(fusion_mode, per_split_alias, float(pad_value))


def init_datasets(
    config: Union[Path, str, Mapping],
    **overrides,
) -> Union[Dict[str, TrajectoryDataset], MultiDatasetCollection]:
    """Instantiate datasets for the requested splits.

    ``config`` may be a root path (with options passed via keyword arguments) or a
    mapping loaded from a configuration file. Keyword arguments always override
    values provided by ``config``.
    """

    if isinstance(config, (str, Path)):
        cfg: Dict[str, object] = {"root": config}
        cfg.update(overrides)
    elif isinstance(config, Mapping):
        cfg = dict(config)
        cfg.update(overrides)
    else:
        raise TypeError("config must be a path or a mapping.")

    if "datasets" in cfg:
        return _init_multi_dataset_collection(cfg)

    if "root" not in cfg:
        raise ValueError("'root' must be provided in the dataset configuration.")

    if "normalize" in cfg and "use_normalization" not in cfg:
        cfg["use_normalization"] = cfg.pop("normalize")
    cfg.setdefault("normalization_type", "zscore")
    if "use_normalization" in cfg:
        cfg["use_normalization"] = bool(cfg["use_normalization"])

    root = Path(cfg.pop("root"))
    dataset_name = cfg.pop("dataset_name", None)
    if dataset_name is not None:
        root = root / str(dataset_name)
    splits = tuple(cfg.pop("splits", ("train", "val", "test")))

    time_range = cfg.pop("time_range", None)
    if time_range is not None:
        time_range = tuple(time_range)

    ntrain = cfg.pop("ntrain", None)
    nval = cfg.pop("nval", None)
    ntest = cfg.pop("ntest", None)

    counts = {
        "train": ntrain,
        "val": nval,
        "valid": nval,
        "validation": nval,
        "test": ntest,
    }

    datasets: Dict[str, TrajectoryDataset] = {}
    for split in splits:
        split_lower = split.lower()
        max_traj = counts.get(split_lower)
        try:
            datasets[split] = TrajectoryDataset(
                root,
                split=split,
                time_range=time_range,
                max_trajectories=max_traj,
                **cfg,
            )
        except FileNotFoundError as error:
            message = str(error)
            if "stats.yaml" in message:
                raise
            continue
    if not datasets:
        raise FileNotFoundError(
            f"No datasets could be initialised from '{root}' with splits {splits}."
        )
    return datasets
