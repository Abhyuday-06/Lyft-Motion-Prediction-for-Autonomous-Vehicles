"""Diag v2: resolve real mount paths, install l5kit + deps, rasterize one sample.

Runs on CPU and touches a single frame, so it is cheap; the point is to prove
the whole chain works before spending a GPU session on 71k agents.
"""
import json
import os

code = []

code.append('''import os, sys, subprocess, glob
print("python", sys.version.split()[0])
# The input tree is nested (competitions/, datasets/), so discover rather than
# hardcode.
for p in sorted(glob.glob("/kaggle/input/*/*"))[:20]:
    print(p)
print("---")
for p in sorted(glob.glob("/kaggle/input/*/*/*"))[:30]:
    print(p)''')

code.append('''# l5kit 1.5.0 has no py3.12 wheel. --no-build-isolation makes pip use the
# environment's modern setuptools instead of fetching the stale pinned one
# (whose setuptools.extern.six is gone); --no-deps avoids its 2020-era pins.
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "--no-deps", "l5kit==1.5.0"],
    capture_output=True, text=True)
print("l5kit rc", r.returncode, r.stderr[-800:])

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "zarr==2.18.3", "numcodecs==0.13.1", "transforms3d", "pymap3d"],
    capture_output=True, text=True)
print("deps rc", r.returncode, r.stdout[-800:], r.stderr[-1500:])''')

code.append('''import importlib
for mod in ["zarr", "numcodecs", "transforms3d", "pymap3d", "numpy"]:
    try:
        m = importlib.import_module(mod)
        print("OK  ", mod, getattr(m, "__version__", ""))
    except Exception as e:
        print("MISS", mod, type(e).__name__, e)''')

code.append('''try:
    from l5kit.data import ChunkedDataset, LocalDataManager
    from l5kit.dataset import EgoDataset
    from l5kit.rasterization import build_rasterizer
    from l5kit.evaluation import write_pred_csv
    print("l5kit imports OK")
except Exception:
    import traceback; traceback.print_exc()''')

code.append('''# Find the competition data root (the dir holding scenes/) and the checkpoint.
cands = glob.glob("/kaggle/input/**/scenes", recursive=True)
print("scenes dirs:", cands)
DATA_ROOT = os.path.dirname(cands[0]) if cands else None
ck_hits = glob.glob("/kaggle/input/**/r18_128_best.pt", recursive=True)
print("ckpt hits:", ck_hits)
CKPT = ck_hits[0] if ck_hits else None
print("DATA_ROOT =", DATA_ROOT)
print("CKPT =", CKPT)
assert DATA_ROOT and CKPT
print(sorted(os.listdir(DATA_ROOT)))
print(sorted(os.listdir(os.path.join(DATA_ROOT, "scenes")))[:10])''')

code.append('''import numpy as np, torch, bisect
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
print("step", ck["step"], "val_loss", ck["val_loss"])
print("raster", cfg["raster_params"]["raster_size"])

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

dm = LocalDataManager(None)
rast = build_rasterizer(cfg, dm)
zarr_ds = ChunkedDataset(dm.require(cfg["test_data_loader"]["key"])).open()
ego = EgoDataset(cfg, zarr_ds, rast)
mask = np.load(dm.require("scenes/mask.npz"))["arr_0"]
idx = np.nonzero(mask)[0].astype(np.int64)
print("test agents:", len(idx))

agent_ends = zarr_ds.frames["agent_index_interval"][:, 1]
scene_ends = ego.cumulative_sizes
ai = int(idx[0])
tid = zarr_ds.agents[ai]["track_id"]
fi = bisect.bisect_right(agent_ends, ai)
si = bisect.bisect_right(scene_ends, fi)
sti = fi if si == 0 else fi - scene_ends[si - 1]
d = ego.get_frame(si, sti, track_id=tid)
print("image", d["image"].shape, d["image"].dtype, d["image"].min(), d["image"].max())
print("timestamp", d["timestamp"], "track_id", d["track_id"])
print("ALL GOOD")''')

nb = {
    "cells": [
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": c.split("\n")}
        for c in code
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
for cell in nb["cells"]:
    src = cell["source"]
    cell["source"] = [line + "\n" for line in src[:-1]] + [src[-1]]

os.makedirs("kaggle_diag2", exist_ok=True)
with open("kaggle_diag2/lyft-env-diag2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "abhyudayvaish/lyft-env-diag2",
    "title": "lyft-env-diag2",
    "code_file": "lyft-env-diag2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": ["abhyudayvaish/lyft-r18-128-weights"],
    "competition_sources": ["lyft-motion-prediction-autonomous-vehicles"],
    "kernel_sources": [],
}
with open("kaggle_diag2/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote kaggle_diag2/")
