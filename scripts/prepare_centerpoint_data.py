import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.detection.v2x_seq_converter import convert_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/centerpoint_v2xseq.yaml")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    split_file = Path(config["split_file"])
    if not split_file.is_absolute():
        split_file = root / split_file
    output_root = Path(config["project"]["centerpoint_root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    convert_dataset(config["project"]["data_root"], output_root, split_file, workers=args.workers)


if __name__ == "__main__":
    main()
