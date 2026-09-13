"""Generate the Kaggle inference notebook for the Lyft competition.

The notebook has to be self-contained: Kaggle cannot import from src/, so the
dataset and model definitions are inlined.
"""
import json
import os

code = []

code.append('''# Lyft Motion Prediction - inference
# ResNet18, ImageNet-pretrained stem inflated 3->25ch, 128px raster.
# Trained locally for 60k steps; best val NLL 14.4998 @ step 57500.
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys, subprocess, glob, bisect

def pip_install(*packages):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])

pip_install("--no-build-isolation", "--no-deps", "l5kit==1.5.0")
pip_install("zarr==2.18.3", "numcodecs==0.13.1", "transforms3d", "pymap3d", "ptable")

import numpy as np
for name, value in {"float": float, "int": int, "bool": bool, "object": object, "str": str}.items():
    if not hasattr(np, name):
        setattr(np, name, value)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())''')

code.append('''competition_roots = [
    path for path in glob.glob("/kaggle/input/competitions/*")
    if os.path.isdir(os.path.join(path, "scenes"))
]
assert len(competition_roots) == 1, competition_roots
DATA_ROOT = competition_roots[0]

checkpoint_paths = glob.glob("/kaggle/input/datasets/**/r18_128_best.pt", recursive=True)
assert len(checkpoint_paths) == 1, checkpoint_paths
CKPT = checkpoint_paths[0]

os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT
print("DATA_ROOT:", DATA_ROOT)
print("CKPT:", CKPT)
print("data entries:", sorted(os.listdir(DATA_ROOT)))''')

code.append('''from torchvision.models.resnet import resnet18, resnet34, resnet50

ARCHS = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}


def build_backbone(architecture, in_channels, num_outputs):
    model = ARCHS[architecture](weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model


class LyftMultiModel(nn.Module):
    def __init__(self, cfg, num_modes=3):
        super().__init__()
        architecture = cfg["model_params"]["model_architecture"]
        history_num_frames = cfg["model_params"]["history_num_frames"]
        in_channels = 3 + (history_num_frames + 1) * 2
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.backbone = build_backbone(architecture, in_channels, self.num_preds + num_modes)

    def forward(self, x):
        out = self.backbone(x)
        bs = out.shape[0]
        pred, confidences = torch.split(out, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences''')

code.append('''def images_to_float(images):
    """Undo the uint8 packing done in LazyAgentDataset, on whatever device holds it."""
    if images.dtype == torch.uint8:
        return images.float().div_(255.0)
    return images


class LazyAgentDataset(Dataset):
    """AgentDataset equivalent that builds its heavy state lazily, per worker.

    A built rasterizer holds the parsed protobuf semantic map, which does not
    pickle; so only config/paths/indices are pickled and the rasterizer is built
    on first __getitem__ inside each worker.
    """

    def __init__(self, cfg, data_root, zarr_key, agents_mask, with_ids=True):
        self.cfg = cfg
        self.with_ids = with_ids
        self.data_root = os.path.abspath(data_root)
        self.zarr_key = zarr_key
        self._indices = np.nonzero(agents_mask)[0].astype(np.int64)
        self._length = len(self._indices)
        self._inner = None

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
        ego = EgoDataset(self.cfg, zarr, rasterizer)
        self._inner = ego
        self._agent_ends = zarr.frames["agent_index_interval"][:, 1]
        self._scene_ends = ego.cumulative_sizes
        self._zarr = zarr
        return ego

    def __getitem__(self, index):
        ego = self._ensure_built()
        agent_index = int(self._indices[index])
        track_id = self._zarr.agents[agent_index]["track_id"]
        frame_index = bisect.bisect_right(self._agent_ends, agent_index)
        scene_index = bisect.bisect_right(self._scene_ends, frame_index)
        state_index = (frame_index if scene_index == 0
                       else frame_index - self._scene_ends[scene_index - 1])
        data = ego.get_frame(scene_index, state_index, track_id=track_id)
        # Ship the raster as uint8: the rasterizer emits exact k/255 values, so
        # the cast is lossless and quarters the bytes crossing the worker pipe.
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
        for key in ("_inner", "_agent_ends", "_scene_ends", "_zarr"):
            state.pop(key, None)
        state["_inner"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inner = None''')

code.append('''from l5kit.data import LocalDataManager
from l5kit.evaluation import write_pred_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location=device, weights_only=False)
cfg = ck["cfg"]  # use the config the weights were trained with
print("checkpoint step", ck["step"], "val_loss", ck["val_loss"])
print("raster", cfg["raster_params"]["raster_size"],
      "history", cfg["model_params"]["history_num_frames"])

cfg["test_data_loader"]["batch_size"] = 64
cfg["test_data_loader"]["num_workers"] = 4

dm = LocalDataManager(None)
test_mask = np.load(dm.require("scenes/mask.npz"))["arr_0"]
test_dataset = LazyAgentDataset(cfg, DATA_ROOT, cfg["test_data_loader"]["key"],
                                agents_mask=test_mask, with_ids=True)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=cfg["test_data_loader"]["batch_size"],
    num_workers=cfg["test_data_loader"]["num_workers"],
    pin_memory=False,
)
print("test dataset len:", len(test_dataset))''')

code.append('''model = LyftMultiModel(cfg).to(device)
model.load_state_dict(ck["model"])
model = model.to(memory_format=torch.channels_last)
model.eval()

coords_all, confs_all, timestamps, agent_ids = [], [], [], []
with torch.no_grad():
    for data in tqdm(test_loader):
        inputs = images_to_float(data["image"].to(device, non_blocking=True))
        inputs = inputs.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred, confidences = model(inputs)
        coords_all.append(pred.float().cpu().numpy())
        confs_all.append(confidences.float().cpu().numpy())
        timestamps.append(data["timestamp"].numpy())
        agent_ids.append(data["track_id"].numpy())
print("batches:", len(coords_all))''')

code.append('''timestamps = np.concatenate(timestamps)
agent_ids = np.concatenate(agent_ids)
coords = np.concatenate(coords_all)
confs = np.concatenate(confs_all)

write_pred_csv(
    "submission.csv",
    timestamps=timestamps,
    track_ids=agent_ids,
    coords=coords,
    confs=confs,
)

import csv
with open("submission.csv", newline="") as f:
    rows = list(csv.reader(f))
header, values = rows[0], rows[1:]
confidence_sums = confs.sum(axis=1)
assert len(values) == len(test_dataset), (len(values), len(test_dataset))
assert len(header) == 305, len(header)
assert np.isfinite(coords).all()
assert np.isfinite(confs).all()
assert np.allclose(confidence_sums, 1.0, rtol=1e-5, atol=1e-6)
print("submission.csv validated")
print("rows:", len(values), "columns:", len(header))
print("conf sums: %.6f .. %.6f" % (confidence_sums.min(), confidence_sums.max()))''')

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": c.split("\n"),
        }
        for c in code
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
# nbformat keeps trailing newlines on every line but the last of each source list
for cell in nb["cells"]:
    src = cell["source"]
    cell["source"] = [line + "\n" for line in src[:-1]] + [src[-1]]

os.makedirs("kaggle_nb", exist_ok=True)
out = os.path.join("kaggle_nb", "lyft-r18-128-inference.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "cells:", len(nb["cells"]))
