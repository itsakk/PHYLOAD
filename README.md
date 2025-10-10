# phyload

Utilities for loading multi-physics, spatio-temporal trajectory data exported by [PHYGEN](https://github.com/itsakk/PHYGEN). 
`phyload` discovers tensor fields, exposes them
through a flexible PyTorch `Dataset`, and provides ready-to-use `DataLoader`
factories that respect heterogeneous grids, windowing, and metadata.

## Highlights

- **Selective I/O** – read only the requested time window and spatial stride from
  disk, keeping data transfers minimal even for very long trajectories.
- **Flexible windowing** – generate fixed-length sub-trajectories with optional
  overlap, temporal stride, and spatial subsampling without touching the source
  files.
- **Rich metadata** – access collated boundary conditions, scalar parameters, time
  grids, and spatial meshes alongside the trajectory tensor.
- **Channel management** – channel-last layout `(T, *space_dims, C)` with stable
  channel names and helpers for z-score or RMS normalisation / de-normalisation.
- **Multi-dataset orchestration** – combine multiple datasets with aligned
  channels using `MultiDatasetCollection`, including padded variants for mixed
  shapes.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-org/PHYLOAD.git
cd PHYLOAD
pip install -e .
```

## Quick start

```python
from pathlib import Path
import yaml

from phyload import init_datasets, init_dataloaders

# Edit configs/main.yaml to point to your dataset root, target dataset,
# and desired loader options, then load it here.
cfg_path = Path("configs/main.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

datasets = init_datasets(cfg["data"])
train_loader, val_loader, test_loader = init_dataloaders(datasets, cfg["loaders"])

batch = next(iter(train_loader))
print(batch["trajectory"].shape)  # (B, T, X[, Y[, Z]], C)
print(batch["time_grid"].shape)   # (B, T)
print(batch["space_grid"].shape)  # (B, T, n_dims, ...)
print(batch["index"])             # provenance info

# Optional metadata
print(batch.get("ic"))            # (B, C, X[, Y[, Z]]) initial condition if available
print(batch.get("params"))        # dict of scalar parameters per sample
print(batch.get("bc"))            # boundary-condition tensors/masks

# De-normalise a channel (when normalisation is enabled)
chan_name = datasets["train"].channel_names[0]
restored = datasets["train"].denormalize_channel(batch["trajectory"][0, :, ..., 0], chan_name)
```

## Configuration essentials

All user-facing options are supplied through YAML configuration files:

- `configs/main.yaml`: single-dataset training/validation/test.
- `configs/main_multi.yaml`: multi-dataset (multi-physics) experiments.

### Data section (`data`)

Key hyper-parameters controlling how trajectories are processed:

- `root`, `dataset_name`: locate the source HDF5 files (`data/<dataset_name>/*.hdf5`).
- `time_range`: `(start, stop)` slice applied before any subsampling.
- `num_steps`: length of the temporal window returned by each sample.
- `window_stride`: offset between successive windows (defaults to `num_steps` for
  non-overlapping windows).
- `time_stride` / `dt`: temporal subsampling factors (integers or fractional).
- `dx`: spatial subsampling per dimension (scalar or list/tuple).
- `return_ic`, `return_bc`, `return_params`: request initial conditions, boundary
  conditions, and scalar parameters when available.
- `use_normalization`, `normalization_type`: apply per-channel statistics (`zscore`
  or `rms`) stored in `stats.yaml`.
- `ntrain`, `nval`, `ntest`: optional cap on the number of raw trajectories per split
  before windowing/subsampling.

### Loader section (`loaders`)

Define how batches are constructed:

- `batch_size`, `val_batch_size`, `test_batch_size`
- `shuffle_train`, `shuffle_val`, `shuffle_test`
- `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`
- `drop_last` (for incomplete batches)
- `multi_gpu`: toggle `DistributedSampler` for multi-GPU training.

Pass the full configuration to `init_datasets` / `init_dataloaders` as shown in the
quick-start snippet.

## Multi-physics workflows

`configs/main_multi.yaml` lets you load several datasets simultaneously. Each
entry under `datasets:` describes a dataset alias with its own data configuration.

Supported combination modes:

- **Homogeneous**: all datasets share the same channel layout and compatible spatial
  shapes. `init_dataloaders` returns `HomogeneousCombinedLoader`, delivering
  batched data per alias in sync.
- **Mixed (heterogeneous)**: datasets differ in shape or channel structure.
  `MultiDatasetCollection` keeps them separate; you can iterate per-alias batches
  or wrap them with `PaddedTrajectoryDataset` if you need aligned shapes.

Choose the mode via the `mode` key in `main_multi.yaml`, and use the same loader
configuration (`loaders` section) to build samplers for all participating datasets.

Channel metadata remains accessible through each sub-dataset, so downstream models
can map component names consistently across physics domains.