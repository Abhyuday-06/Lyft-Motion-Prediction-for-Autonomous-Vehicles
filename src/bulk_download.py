"""
Download the full Lyft competition dataset via the Kaggle CLI as a single zip,
then extract only the needed files.  This avoids the per-file API rate limit.

Usage:
    python src/bulk_download.py [--dest data]
"""
import argparse
import os
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--kaggle", default=None, help="path to kaggle executable")
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    # Find kaggle CLI
    kaggle_exe = args.kaggle
    if kaggle_exe is None:
        # Try conda env Scripts dir
        candidates = [
            os.path.join(sys.prefix, "Scripts", "kaggle.exe"),
            os.path.join(sys.prefix, "Scripts", "kaggle"),
            os.path.join(sys.prefix, "bin", "kaggle"),
        ]
        for c in candidates:
            if os.path.exists(c):
                kaggle_exe = c
                break
        if kaggle_exe is None:
            # Fall back to PATH
            kaggle_exe = "kaggle"

    comp = "lyft-motion-prediction-autonomous-vehicles"

    print(f"downloading competition data to {dest} ...")
    print(f"using kaggle CLI: {kaggle_exe}")
    sys.stdout.flush()

    # Download the full competition zip
    cmd = [kaggle_exe, "competitions", "download", "-c", comp, "-p", dest]
    print(f"running: {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"kaggle CLI exited with code {result.returncode}")
        sys.exit(1)

    # Find the downloaded zip(s) and extract
    for fname in os.listdir(dest):
        if fname.endswith(".zip"):
            zpath = os.path.join(dest, fname)
            print(f"extracting {zpath} ...")
            sys.stdout.flush()
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
            print(f"extracted {zpath}, removing zip to save disk")
            os.remove(zpath)

    # Check what we got
    for ds in ["sample", "train", "test"]:
        zarr_path = os.path.join(dest, "scenes", f"{ds}.zarr")
        if os.path.isdir(zarr_path):
            agents_dir = os.path.join(zarr_path, "agents")
            n_agents = len([f for f in os.listdir(agents_dir) if not f.startswith(".")]) if os.path.isdir(agents_dir) else 0
            print(f"  {ds}.zarr: agents dir has {n_agents} files")
        else:
            print(f"  {ds}.zarr: NOT FOUND")

    print("BULK DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
