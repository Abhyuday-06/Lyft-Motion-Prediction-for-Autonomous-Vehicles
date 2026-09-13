import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "lyft-motion-prediction-autonomous-vehicles"


class RateLimiter:
    """Global pacing so concurrent workers don't burst past Kaggle's per-account rate limit.

    Kaggle's competition-file-download endpoint appears to enforce a fixed request quota per
    rolling window rather than a simple per-request interval: a burst of ~400 quick requests
    triggered a 429 lockout that persisted for 45+ minutes regardless of per-file backoff, and
    only cleared once no requests were sent for a while. So on any 429, ALL threads must stop
    issuing requests (not just the one that failed) until a shared cooldown elapses -- trickling
    in retries during the lockout risks extending it further.
    """

    def __init__(self, min_interval: float, cooldown_seconds: float = 300.0):
        self.min_interval = min_interval
        self.cooldown_seconds = cooldown_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0
        self.blocked_until = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            wait_until = max(self.last_call + self.min_interval, self.blocked_until)
            delay = wait_until - now
            if delay > 0:
                time.sleep(delay)
            self.last_call = time.monotonic()

    def report_429(self):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + self.cooldown_seconds)
            print(f"[rate-limiter] 429 seen, pausing ALL workers for {self.cooldown_seconds:.0f}s", flush=True)

# Known array shapes/chunk sizes, probed directly from each dataset's .zarray metadata.
DATASETS = {
    "sample": {"agents": (1893736, 20000), "frames": (24838, 10000), "scenes": (100, 10000)},
    "train": {"agents": (320124624, 20000), "frames": (4039527, 10000), "scenes": (16265, 10000)},
    "test": {"agents": (88594921, 20000), "frames": (1131400, 10000), "scenes": (11314, 10000)},
    "validate": {"agents": (312617887, 20000), "frames": (4030296, 10000), "scenes": (16220, 10000)},
}


def build_file_list(dataset_names):
    paths = ["scenes/mask.npz"]
    for ds in dataset_names:
        paths.append(f"scenes/{ds}.zarr/.zgroup")
        for arr, (shape, chunk) in DATASETS[ds].items():
            paths.append(f"scenes/{ds}.zarr/{arr}/.zarray")
            n_chunks = math.ceil(shape / chunk)
            for c in range(n_chunks):
                paths.append(f"scenes/{ds}.zarr/{arr}/{c}")
    return paths


def download_one(api: KaggleApi, name: str, dest_root: str, limiter: RateLimiter, retries: int = 8) -> str:
    local_path = os.path.join(dest_root, name.replace("/", os.sep))
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return name
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        limiter.wait()
        try:
            api.competition_download_file(COMPETITION, name, path=local_dir, force=True, quiet=True)
            return name
        except Exception as e:
            last_err = e
            if "429" in str(e):
                limiter.report_429()
            else:
                time.sleep(min(2 * (attempt + 1), 10))
    raise RuntimeError(f"failed to download {name}: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["sample", "train", "test"], choices=list(DATASETS.keys()))
    parser.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-interval", type=float, default=1.0, help="min seconds between API calls, shared across workers")
    parser.add_argument("--cooldown-seconds", type=float, default=300.0, help="on any 429, pause ALL workers this long before retrying anything")
    args = parser.parse_args()

    paths = build_file_list(args.datasets)
    print(f"total files to ensure present: {len(paths)}")

    api = KaggleApi()
    api.authenticate()
    limiter = RateLimiter(args.min_interval, args.cooldown_seconds)

    done = 0
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, api, p, args.dest, limiter): p for p in paths}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
                done += 1
                if done % 200 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (len(paths) - done) / rate if rate > 0 else float("inf")
                    print(f"progress {done}/{len(paths)}  ({elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining)", flush=True)
            except Exception as e:
                failed.append(name)
                print(f"FAILED: {name}: {e}", flush=True)

    print(f"DONE: {done}/{len(paths)}, failed: {len(failed)}")
    if failed:
        with open(os.path.join(args.dest, "failed_downloads.json"), "w") as f:
            json.dump(failed, f)


if __name__ == "__main__":
    main()
