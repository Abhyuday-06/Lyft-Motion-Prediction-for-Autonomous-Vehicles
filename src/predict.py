import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager
from l5kit.evaluation import write_pred_csv

from data import LazyAgentDataset, images_to_float
from model import LyftMultiModel

DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "agent_motion_config.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(DEFAULT_DATA_ROOT, "models", "resnet18_best.pt"))
    parser.add_argument("--out", default=os.path.join(DEFAULT_DATA_ROOT, "submissions", "submission.csv"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", choices=["fp16", "bf16", "off"], default="fp16")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    os.environ["L5KIT_DATA_FOLDER"] = data_root
    dm = LocalDataManager(None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "cfg" in ck:
        # Use the config the checkpoint was trained with, so raster size /
        # history depth always match the weights.
        cfg = ck["cfg"]
        print(f"checkpoint step={ck.get('step')} val_loss={ck.get('val_loss')}", flush=True)
    else:
        cfg = load_config_data(args.config)
    if args.batch_size is not None:
        cfg["test_data_loader"]["batch_size"] = args.batch_size

    test_cfg = cfg["test_data_loader"]
    test_mask = np.load(dm.require("scenes/mask.npz"))["arr_0"]
    test_dataset = LazyAgentDataset(
        cfg, data_root, test_cfg["key"], agents_mask=test_mask, with_ids=True
    )
    workers = test_cfg["num_workers"] if args.num_workers is None else args.num_workers
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=test_cfg["batch_size"],
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    print(f"test dataset len: {len(test_dataset)}", flush=True)

    model = LyftMultiModel(cfg, pretrained=False).to(device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model = model.to(memory_format=torch.channels_last)
    model.eval()

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[args.amp]
    if device.type != "cuda":
        amp_dtype = None

    coords_all, confs_all, timestamps, agent_ids = [], [], [], []

    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs = images_to_float(data["image"].to(device, non_blocking=True))
            inputs = inputs.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred, confidences = model(inputs)
            coords_all.append(pred.float().cpu().numpy())
            confs_all.append(confidences.float().cpu().numpy())
            timestamps.append(data["timestamp"].numpy())
            agent_ids.append(data["track_id"].numpy())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_pred_csv(
        args.out,
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),
        coords=np.concatenate(coords_all),
        confs=np.concatenate(confs_all),
    )
    print(f"wrote submission to {args.out}", flush=True)


if __name__ == "__main__":
    main()
