import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "lyft-motion-prediction-autonomous-vehicles"


def download_one(api: KaggleApi, name: str, dest_root: str, retries: int = 5) -> str:
    local_path = os.path.join(dest_root, name)
    if os.path.exists(local_path):
        return name
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    for attempt in range(retries):
        try:
            api.competition_download_file(
                COMPETITION, name, path=os.path.dirname(local_path), force=True, quiet=True
            )
            return name
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {name} after {retries} retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-list", default=os.path.join(os.path.dirname(__file__), "..", "data", "file_list.json"))
    parser.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--prefixes", nargs="+", required=True, help="path prefixes to include, e.g. scenes/sample.zarr")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=None, help="stop after this many total bytes queued")
    args = parser.parse_args()

    with open(args.file_list) as f:
        all_files = json.load(f)

    selected = [f for f in all_files if any(f["name"].startswith(p) for p in args.prefixes)]
    total_bytes = sum(f["size"] for f in selected)
    print(f"selected {len(selected)} files, {total_bytes/1e9:.2f} GB")

    if args.max_bytes is not None and total_bytes > args.max_bytes:
        selected.sort(key=lambda f: f["name"])
        running = 0
        capped = []
        for f in selected:
            running += f["size"]
            if running > args.max_bytes:
                break
            capped.append(f)
        selected = capped
        print(f"capped to {len(selected)} files, {sum(f['size'] for f in selected)/1e9:.2f} GB")

    api = KaggleApi()
    api.authenticate()

    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, api, f["name"], args.dest): f["name"] for f in selected}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
                done += 1
                if done % 25 == 0:
                    print(f"progress: {done}/{len(selected)}", flush=True)
            except Exception as e:
                failed.append(name)
                print(f"FAILED: {name}: {e}")

    print(f"done: {done}/{len(selected)}, failed: {len(failed)}")
    if failed:
        with open(os.path.join(args.dest, "failed_downloads.json"), "w") as f:
            json.dump(failed, f)


if __name__ == "__main__":
    main()
