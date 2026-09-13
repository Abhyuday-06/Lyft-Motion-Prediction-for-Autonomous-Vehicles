"""Generate a diagnostic Kaggle kernel: what mounts, and how l5kit installs."""
import json
import os

code = []

code.append('''import os, sys, subprocess
print("python", sys.version)
for root in ["/kaggle/input"]:
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        try:
            inner = sorted(os.listdir(p))[:12]
        except Exception as e:
            inner = repr(e)
        print(d, "->", inner)''')

code.append('''# Strategy A: no build isolation (avoids pip pulling an ancient setuptools
# whose vendored six is gone on py3.12), no deps (l5kit 1.5.0 pins are stale).
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
     "--no-deps", "l5kit==1.5.0"],
    capture_output=True, text=True)
print("returncode", r.returncode)
print(r.stdout[-2000:])
print(r.stderr[-3000:])''')

code.append('''try:
    import l5kit
    print("l5kit", l5kit.__version__)
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)''')

code.append('''# What does l5kit actually need at import time?
for mod in ["zarr", "numcodecs", "cv2", "transforms3d", "pymap3d", "shapely",
            "protobuf", "google.protobuf", "ptable", "prettytable", "numpy",
            "torch", "yaml", "tqdm"]:
    try:
        m = __import__(mod)
        print("OK  ", mod, getattr(m, "__version__", ""))
    except Exception as e:
        print("MISS", mod, type(e).__name__)''')

code.append('''try:
    from l5kit.data import ChunkedDataset, LocalDataManager
    from l5kit.rasterization import build_rasterizer
    from l5kit.evaluation import write_pred_csv
    print("l5kit submodules import OK")
except Exception as e:
    import traceback; traceback.print_exc()''')

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

os.makedirs("kaggle_diag", exist_ok=True)
with open("kaggle_diag/lyft-env-diag.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

meta = {
    "id": "abhyudayvaish/lyft-env-diag",
    "title": "lyft-env-diag",
    "code_file": "lyft-env-diag.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": ["abhyudayvaish/lyft-r18-128-weights"],
    "competition_sources": ["lyft-motion-prediction-autonomous-vehicles"],
    "kernel_sources": [],
}
with open("kaggle_diag/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("wrote kaggle_diag/")
