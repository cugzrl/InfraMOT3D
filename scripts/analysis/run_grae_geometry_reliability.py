import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.analysis.grae_geometry_reliability import run
from inframot3d.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/grae_geometry_reliability.yaml")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_root:
        root = config["_root"]
        path = Path(args.output_root)
        config["project"]["output_root"] = path if path.is_absolute() else root / path
    if args.skip_ablation:
        config["run_ablation"] = False
    if args.sequences:
        config["sequence_override"] = [str(item) for item in args.sequences]
    run(config)


if __name__ == "__main__":
    main()
