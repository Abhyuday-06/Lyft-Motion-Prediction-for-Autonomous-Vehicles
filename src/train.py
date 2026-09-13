import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from l5kit.configs import load_config_data

from data import LazyAgentDataset, images_to_float
from model import LyftMultiModel, pytorch_neg_multi_log_likelihood_batch

DEFAULT_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def build_loader(cfg, data_root, section, num_workers=None, subset=None, seed=42, prefetch=2):
    """Build a DataLoader for one of the config's *_data_loader sections.

    Worker count and prefetch depth are the main levers on host RAM here: each
    worker holds its own decompressed zarr chunks, and every prefetched batch is
    a few tens of MB. Oversubscribing starves the OS page cache that the zarr
    reads depend on, which shows up as throughput collapsing over time rather
    than as an out-of-memory error.
    """
    is_train = section == "train_data_loader"
    lcfg = cfg[section]
    dataset = LazyAgentDataset(
        cfg, data_root, lcfg["key"], cache_dir=os.path.join(data_root, "cache")
    )
    if subset is not None and subset < len(dataset):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(dataset), size=subset, replace=False)
        dataset = Subset(dataset, np.sort(idx))
    workers = lcfg["num_workers"] if num_workers is None else num_workers
    if not is_train:
        # Validation runs a few hundred batches every eval_every steps. Giving it
        # a full-size persistent pool keeps a second set of workers resident for
        # the whole run for no throughput gain.
        workers = max(1, workers // 3)
    loader = DataLoader(
        dataset,
        shuffle=lcfg["shuffle"],
        batch_size=lcfg["batch_size"],
        num_workers=workers,
        # pin_memory measured slower here: the uint8 batches are big and the
        # extra pinned-buffer copy costs more than the async transfer saves.
        pin_memory=False,
        drop_last=is_train,
        # Non-persistent val workers are torn down after each eval, returning
        # their RAM to the page cache.
        persistent_workers=is_train and workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
    )
    return dataset, loader


def forward_batch(model, data, device):
    # Images arrive as uint8 and are converted on-device; see data.LazyAgentDataset.
    inputs = data["image"].to(device, non_blocking=True)
    inputs = images_to_float(inputs).to(memory_format=torch.channels_last)
    targets = data["target_positions"].to(device, non_blocking=True)
    avails = data["target_availabilities"].to(device, non_blocking=True)
    pred, confidences = model(inputs)
    loss = pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, avails)
    return loss


@torch.no_grad()
def evaluate(model, loader, device, max_batches, amp_dtype):
    model.eval()
    losses = []
    for i, data in enumerate(tqdm(loader, total=max_batches, desc="val", leave=False)):
        if i >= max_batches:
            break
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = forward_batch(model, data, device)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "agent_motion_config.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--out-dir", default=os.path.join(DEFAULT_DATA_ROOT, "models"))
    parser.add_argument("--run-name", default="resnet18")
    parser.add_argument("--resume", default=None, help="path to a checkpoint .pt to resume from")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--val-subset", type=int, default=200000)
    parser.add_argument("--amp", choices=["fp16", "bf16", "off"], default="fp16")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument("--raster-size", type=int, default=None)
    parser.add_argument("--history-frames", type=int, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--prefetch", type=int, default=2)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    os.environ["L5KIT_DATA_FOLDER"] = data_root
    cfg = load_config_data(args.config)

    if args.max_steps is not None:
        cfg["train_params"]["max_num_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["train_data_loader"]["batch_size"] = args.batch_size
        cfg["val_data_loader"]["batch_size"] = args.batch_size
    if args.raster_size is not None:
        cfg["raster_params"]["raster_size"] = [args.raster_size, args.raster_size]
    if args.history_frames is not None:
        cfg["model_params"]["history_num_frames"] = args.history_frames
    if args.arch is not None:
        cfg["model_params"]["model_architecture"] = args.arch

    train_dataset, train_loader = build_loader(
        cfg, data_root, "train_data_loader", args.num_workers, prefetch=args.prefetch
    )
    print(f"train dataset len: {len(train_dataset)}", flush=True)

    val_loader = None
    if not args.no_val:
        val_dataset, val_loader = build_loader(
            cfg, data_root, "val_data_loader", args.num_workers,
            subset=args.val_subset, prefetch=args.prefetch,
        )
        print(f"val dataset len: {len(val_dataset)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LyftMultiModel(cfg, pretrained=not args.no_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True

    max_steps = cfg["train_params"]["max_num_steps"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max_steps, pct_start=0.1
    )
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[args.amp]
    if device.type != "cuda":
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_last = os.path.join(args.out_dir, f"{args.run_name}_last.pt")
    ckpt_best = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
    hist_path = os.path.join(args.out_dir, f"{args.run_name}_history.json")

    start_step = 0
    best_val = float("inf")
    history = []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_step = ck.get("step", 0)
        best_val = ck.get("best_val", float("inf"))
        history = ck.get("history", [])
        print(f"resumed from {args.resume} at step {start_step} (best_val={best_val:.4f})", flush=True)

    def save(path, step, val_loss=None):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "step": step,
                "best_val": best_val,
                "val_loss": val_loss,
                "history": history,
                "cfg": cfg,
            },
            path,
        )

    train_iter = iter(train_loader)
    losses = []
    model.train()
    start = time.time()
    progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps)
    for step in progress:
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = forward_batch(model, data, device)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % args.log_every == 0:
            avg = float(np.mean(losses[-args.log_every:]))
            elapsed = (time.time() - start) / 60
            progress.set_description(
                f"loss: {avg:.4f} lr: {scheduler.get_last_lr()[0]:.2e} elapsed: {elapsed:.1f}m"
            )
            history.append({"step": step + 1, "train_loss": avg, "lr": scheduler.get_last_lr()[0]})

        if val_loader is not None and (step + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, args.val_batches, amp_dtype)
            tqdm.write(f"[step {step+1}] val_loss: {val_loss:.4f}")
            history.append({"step": step + 1, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                save(ckpt_best, step + 1, val_loss)
                tqdm.write(f"  new best -> {ckpt_best}")
            save(ckpt_last, step + 1, val_loss)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=1)

        elif (step + 1) % args.save_every == 0:
            save(ckpt_last, step + 1)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=1)

    save(ckpt_last, max_steps)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=1)
    print(f"done. last={ckpt_last} best={ckpt_best} best_val={best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
