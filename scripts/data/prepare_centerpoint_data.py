import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.detection.config_check import check_centerpoint_configs
from inframot3d.detection.dataset_stats import summarize_dataset
from inframot3d.detection.v2x_seq_converter import convert_dataset
from inframot3d.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detectors/centerpoint/v2xseq.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    check_centerpoint_configs(root)
    split_file = Path(config["split_file"])
    if not split_file.is_absolute():
        split_file = root / split_file
    output_root = Path(config["project"]["centerpoint_root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    convert_dataset(
        config["project"]["data_root"],
        output_root,
        split_file,
        workers=args.workers,
        overwrite=args.force or args.overwrite,
    )
    report = summarize_dataset(
        output_root,
        config["point_cloud_range"],
        config["voxel_size"],
        config["max_points_per_voxel"],
        config["max_number_of_voxels"],
        workers=args.workers,
    )
    write_json(output_root / "metadata" / "range_report.json", report)


if __name__ == "__main__":
    main()
