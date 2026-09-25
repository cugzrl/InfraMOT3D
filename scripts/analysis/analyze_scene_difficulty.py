import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.analysis.scene_difficulty import run
from inframot3d.config import load_config


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/v2xseq_scene_difficulty.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
