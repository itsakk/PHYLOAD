"""Dataset utilities for structured spatio-temporal trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableSequence, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

__all__ = [
    "FieldInfo",
    "TrajectoryDataset",
    "init_datasets",
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
        normalize: bool = False,
        time_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        time_stride: int = 1,
        dt: Optional[Union[int, float]] = None,
        dx: Optional[Union[int, float, Sequence[Union[int, float]]]] = None,
        num_steps: Optional[int] = None,
        window_stride: Optional[int] = None,
        max_trajectories: Optional[int] = None,
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
        self.normalize = bool(normalize)
        self.return_ic = bool(return_ic)
        self.return_bc = bool(return_bc)
        self.return_params = bool(return_params)

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

        self._spatial_strides = self._resolve_spatial_strides(dx, self.n_spatial_dims)
        self._space_axes = [axis[::stride] for axis, stride in zip(space_axes, self._spatial_strides)]
        self.space_grid = self._build_space_grid(self._space_axes).to(self.dtype)

        self.num_steps = num_steps if num_steps is None else int(num_steps)
        if self.num_steps is not None and self.num_steps <= 0:
            raise ValueError("num_steps must be a positive integer or None.")
        if window_stride is None:
            self.window_stride = self.num_steps if self.num_steps is not None else 1
        else:
            if window_stride <= 0:
                raise ValueError("window_stride must be a positive integer.")
            self.window_stride = int(window_stride)

        if self.normalize:
            self._norm_mean, self._norm_std = self._load_stats(self.root / "stats.yaml")
        else:
            self._norm_mean, self._norm_std = {}, {}

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
                time_len = self._infer_time_length(data, primary)
                if time_len <= 0:
                    raise ValueError(f"File '{file_path}' does not contain temporal samples.")

                time_axis = self._load_time_axis(handle, self.field_infos)
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

        channel_blocks: List[np.ndarray] = []
        for info in self.field_infos:
            dataset = handle[info.path]
            raw = dataset[traj_idx]
            time_aligned = self._extract_time(raw, info, selected_indices)
            spatial_aligned = self._apply_spatial_stride(time_aligned)
            channel = self._to_channel_first(spatial_aligned, info)
            if self.normalize:
                channel = self._apply_normalization(channel, info)
            channel_blocks.append(channel)

        trajectory = np.concatenate(channel_blocks, axis=0)
        trajectory_tensor = torch.from_numpy(trajectory.astype(np.float32)).to(self.dtype)

        sample: Dict[str, object] = {
            "index": {
                "file": self.files[file_id].name,
                "trajectory": traj_idx,
                "window_start": int(window_start),
            },
            "trajectory": trajectory_tensor,
        }

        if self.return_ic:
            sample["ic"] = trajectory_tensor[..., 0].clone()

        repeat_dims = [1] * self.space_grid.dim() + [num_frames]
        space_grid = self.space_grid.unsqueeze(-1).repeat(*repeat_dims).to(self.dtype)
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
        if primary is None:
            return np.zeros(1, dtype=np.float32)
        if "dimensions" in handle and "time" in handle["dimensions"]:
            return np.asarray(handle["dimensions"]["time"][()], dtype=np.float32)
        dataset = handle[primary.path]
        return np.arange(dataset.shape[1], dtype=np.float32)

    # ------------------------------------------------------------------
    # Data transforms
    # ------------------------------------------------------------------
    def _extract_time(
        self,
        array: np.ndarray,
        info: FieldInfo,
        selected_indices: np.ndarray,
    ) -> np.ndarray:
        if info.has_time:
            return array[selected_indices]
        # Time-invariant field: broadcast across selected frames
        expanded = np.repeat(array[np.newaxis, ...], len(selected_indices), axis=0)
        return expanded

    def _apply_spatial_stride(self, array: np.ndarray) -> np.ndarray:
        slices = [slice(None)]
        for stride in self._spatial_strides:
            slices.append(slice(None, None, stride))
        remainder = array.ndim - 1 - self.n_spatial_dims
        if remainder > 0:
            slices.extend([slice(None)] * remainder)
        return array[tuple(slices)]

    def _to_channel_first(self, array: np.ndarray, info: FieldInfo) -> np.ndarray:
        spatial_shape = array.shape[1 : 1 + self.n_spatial_dims]
        remainder = array.shape[1 + self.n_spatial_dims :]
        components = int(np.prod(remainder)) if remainder else 1
        reshaped = array.reshape((array.shape[0], *spatial_shape, components))
        moved = np.moveaxis(reshaped, -1, 0)  # (C, T, spatial...)
        axes = [0] + list(range(2, moved.ndim)) + [1]
        return moved.transpose(axes).astype(np.float32)

    def _apply_normalization(self, tensor: np.ndarray, info: FieldInfo) -> np.ndarray:
        name = info.name
        mean = self._norm_mean.get(name)
        std = self._norm_std.get(name)
        if mean is not None:
            mean_flat = np.asarray(mean, dtype=np.float32).reshape(-1)
            if mean_flat.size != info.components:
                repeats = int(np.ceil(info.components / mean_flat.size))
                mean_flat = np.tile(mean_flat, repeats)[: info.components]
            shape = (info.components,) + (1,) * self.n_spatial_dims + (1,)
            tensor = tensor - mean_flat.reshape(shape)
        if std is not None:
            std_flat = np.asarray(std, dtype=np.float32).reshape(-1)
            std_flat = np.where(std_flat == 0, 1.0, std_flat)
            if std_flat.size != info.components:
                repeats = int(np.ceil(info.components / std_flat.size))
                std_flat = np.tile(std_flat, repeats)[: info.components]
            shape = (info.components,) + (1,) * self.n_spatial_dims + (1,)
            tensor = tensor / std_flat.reshape(shape)
        return tensor

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
                params[name] = arr.tolist()
            elif arr.dtype.kind == "b":
                params[name] = torch.as_tensor(arr.astype(np.bool_))
            elif arr.ndim == 0:
                params[name] = torch.as_tensor(arr.tolist(), dtype=self.dtype)
            else:
                params[name] = torch.from_numpy(arr.astype(np.float32)).to(self.dtype)
        return params

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
    def _load_stats(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if not path.exists():
            raise FileNotFoundError(
                f"Normalization requested but '{path}' does not exist."
            )
        payload = yaml.safe_load(path.read_text()) or {}
        mean_map = payload.get("mean", {})
        std_map = payload.get("std", {})

        def convert(mapping: Dict) -> Dict[str, np.ndarray]:
            out: Dict[str, np.ndarray] = {}
            for key, value in mapping.items():
                out[key] = np.asarray(value, dtype=np.float32)
            return out

        return convert(mean_map), convert(std_map)


def init_datasets(
    config: Union[Path, str, Dict[str, object]],
    **overrides,
) -> Dict[str, TrajectoryDataset]:
    """Instantiate datasets for the requested splits.

    ``config`` may be a root path (with options passed via keyword arguments) or a
    mapping loaded from a configuration file. Keyword arguments always override
    values provided by ``config``.
    """

    if isinstance(config, (str, Path)):
        cfg: Dict[str, object] = {"root": config}
        cfg.update(overrides)
    elif isinstance(config, dict):
        cfg = dict(config)
        cfg.update(overrides)
    else:
        raise TypeError("config must be a path or a mapping.")

    if "root" not in cfg:
        raise ValueError("'root' must be provided in the dataset configuration.")

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
        except FileNotFoundError:
            continue
    if not datasets:
        raise FileNotFoundError(
            f"No datasets could be initialised from '{root}' with splits {splits}."
        )
    return datasets
