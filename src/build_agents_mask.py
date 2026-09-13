import argparse
import os

from l5kit.data import ChunkedDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--zarr", nargs="+", required=True, help="e.g. scenes/sample.zarr scenes/train.zarr")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    from l5kit.dataset.select_agents import select_agents, TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE

    for rel_path in args.zarr:
        path = os.path.join(os.path.abspath(args.data_root), rel_path)
        mask_path = os.path.join(path, "agents_mask", str(args.threshold))
        if os.path.exists(mask_path):
            print(f"mask already exists at {mask_path}, skipping")
            continue
        print(f"building agents_mask for {path} at threshold {args.threshold}")
        zarr_dataset = ChunkedDataset(path=path)
        zarr_dataset.open()
        select_agents(
            zarr_dataset,
            args.threshold,
            th_yaw_degree=TH_YAW_DEGREE,
            th_extent_ratio=TH_EXTENT_RATIO,
            th_distance_av=TH_DISTANCE_AV,
        )
        print(f"done with {path}")


if __name__ == "__main__":
    main()
