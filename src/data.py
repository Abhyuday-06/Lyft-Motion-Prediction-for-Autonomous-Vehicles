"""Worker-safe, memory-lean dataset plumbing for the Lyft motion prediction set.

Two problems this module works around, both of which bite on Windows where
DataLoader workers are spawned (not forked) and therefore have to pickle the
dataset object:

1. A built rasterizer holds the parsed protobuf semantic map, which is not
   picklable under either protobuf implementation. So we hold only the plain
   config/paths and build the rasterizer lazily, once per worker process.
2. ``AgentDataset.__init__`` materialises ``mask_indices`` as an int64 array the
   length of the full agent table (320M for train => ~2.5GB *per worker*). It is
   only needed by ``get_frame_indices``, which training never calls, so we skip
   it and keep just the ``agents_indices`` we actually index with.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MIN_FRAME_HISTORY = 10
MIN_FRAME_FUTURE = 1


def images_to_float(images):
    """Undo the uint8 packing done in LazyAgentDataset, on whatever device holds it."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images


def agents_indices_cache_path(cache_dir, zarr_key, threshold, min_history, min_future):
    stem = zarr_key.replace("/", "_").replace(".zarr", "")
    name = f"{stem}_idx_th{threshold}_h{min_history}_f{min_future}.npy"
    return os.path.join(cache_dir, name)


def build_agents_indices(data_root, zarr_key, threshold, min_history, min_future, cache_path):
    """Compute (and cache) the indices of agents that pass the availability filter.

    Reads the precomputed ``agents_mask/<threshold>`` array shipped in the zarr,
    which stores (frames_of_history, frames_of_future) per agent.
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    from zarr import convenience

    mask_path = Path(data_root) / zarr_key / f"agents_mask/{threshold}"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"no precomputed agents_mask at {mask_path}. "
            "Run src/build_agents_mask.py for this zarr first."
        )

    # Reduce in chunks so we never hold the full (N, 2) uint32 table plus copies.
    arr = convenience.load(str(mask_path))
    n = arr.shape[0]
    step = 20_000_000
    keep = []
    for start in range(0, n, step):
        block = arr[start : start + step]
        ok = (block[:, 0] >= min_history) & (block[:, 1] >= min_future)
        keep.append(np.nonzero(ok)[0].astype(np.int64) + start)
    indices = np.concatenate(keep)
    del arr, keep

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, indices)
        return np.load(cache_path, mmap_mode="r")
    return indices


class LazyAgentDataset(Dataset):
    """AgentDataset equivalent that builds its heavy state lazily, per worker.

    The object itself pickles as a handful of strings and an array of indices,
    so it survives the spawn-based DataLoader on Windows.
    """

    def __init__(
        self,
        cfg,
        data_root,
        zarr_key,
        agents_indices=None,
        agents_mask=None,
        cache_dir=None,
        min_history=MIN_FRAME_HISTORY,
        min_future=MIN_FRAME_FUTURE,
        slim=True,
        with_ids=False,
    ):
        self.cfg = cfg
        self.slim = slim
        self.with_ids = with_ids
        self.data_root = os.path.abspath(data_root)
        self.zarr_key = zarr_key
        self.threshold = cfg["raster_params"]["filter_agents_threshold"]

        self._indices_path = None
        if agents_indices is not None:
            self._indices = np.asarray(agents_indices)
        elif agents_mask is not None:
            # Explicit mask (the test set ships one in scenes/mask.npz).
            self._indices = np.nonzero(agents_mask)[0].astype(np.int64)
        else:
            cache_path = (
                agents_indices_cache_path(cache_dir, zarr_key, self.threshold, min_history, min_future)
                if cache_dir
                else None
            )
            built = build_agents_indices(
                self.data_root, zarr_key, self.threshold, min_history, min_future, cache_path
            )
            if cache_path and os.path.exists(cache_path):
                # Workers memmap the cache instead of receiving a pickled copy.
                self._indices_path = cache_path
                self._indices = None
                self._length = len(built)
            else:
                self._indices = built

        if self._indices is not None:
            self._length = len(self._indices)
        self._inner = None  # built per worker

    @property
    def agents_indices(self):
        if self._indices is None:
            self._indices = np.load(self._indices_path, mmap_mode="r")
        return self._indices

    def __len__(self):
        return self._length

    def _ensure_built(self):
        if self._inner is not None:
            return self._inner

        os.environ["L5KIT_DATA_FOLDER"] = self.data_root
        from l5kit.data import ChunkedDataset, LocalDataManager
        from l5kit.dataset import EgoDataset
        from l5kit.rasterization import build_rasterizer

        dm = LocalDataManager(None)
        rasterizer = build_rasterizer(self.cfg, dm)
        zarr = ChunkedDataset(dm.require(self.zarr_key)).open()
        # EgoDataset gives us get_frame() plus the scene/frame bookkeeping,
        # without AgentDataset's per-agent mask arrays.
        ego = EgoDataset(self.cfg, zarr, rasterizer)
        self._inner = ego
        self._agent_ends = zarr.frames["agent_index_interval"][:, 1]
        self._scene_ends = ego.cumulative_sizes
        self._zarr = zarr
        return ego

    def __getitem__(self, index):
        import bisect

        ego = self._ensure_built()
        agent_index = int(self.agents_indices[index])
        track_id = self._zarr.agents[agent_index]["track_id"]
        frame_index = bisect.bisect_right(self._agent_ends, agent_index)
        scene_index = bisect.bisect_right(self._scene_ends, frame_index)
        state_index = frame_index if scene_index == 0 else frame_index - self._scene_ends[scene_index - 1]
        data = ego.get_frame(scene_index, state_index, track_id=track_id)

        if not self.slim:
            return data

        # Ship only what the model needs, and ship the raster as uint8. The
        # rasterizer emits exact k/255 values, so the cast is lossless and it
        # quarters the bytes crossing the worker->main process boundary, which
        # is what actually limits throughput here. Undo with images_to_float().
        out = {
            "image": np.ascontiguousarray(np.round(data["image"] * 255.0).astype(np.uint8)),
            "target_positions": data["target_positions"],
            "target_availabilities": data["target_availabilities"],
        }
        if self.with_ids:
            out["timestamp"] = data["timestamp"]
            out["track_id"] = data["track_id"]
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop everything built inside a worker; it is rebuilt on first access.
        for key in ("_inner", "_agent_ends", "_scene_ends", "_zarr"):
            state.pop(key, None)
        state["_inner"] = None
        if state.get("_indices_path"):
            # Let the child memmap the cache file rather than copy 180MB of indices.
            state["_indices"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inner = None
